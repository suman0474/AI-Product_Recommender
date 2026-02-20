"""
Node: validate_product_node
============================

Step 1 of the Product Search Deep Agent.

Directly implements all logic that was previously inside ValidationTool.validate()
with NO class instantiation:

  1.  UI decision guard (rejects inputs like "User selected: continue")
  2.  Product type extraction  → extract_requirements_tool.invoke()
  3.  Schema load / PPI gen    → load_schema_tool.invoke()
  4.  Session enrichment cache deduplication  (saves 50-70 s on repeat calls)
  5.  Standards field population → populate_schema_fields_from_standards()
  6.  Comprehensive standards defaults [FIX #4] → extract_schema_field_values_from_standards()
  7.  Template specifications [FIX #5] → get_all_specs_for_product_type()
  8.  Standards section build → get_applicable_standards()
  9.  Standards RAG enrichment → enrich_identified_items_with_standards()           [P1-C: fixed signature]
  10. Deep-agent schema population → populate_schema_with_deep_agent()              [P1-D: NEW]
  11. Requirements validation → validate_requirements_tool.invoke()

P1 fixes applied
-----------------
  P1-A  product_type_refined: reads refined_product_type from validation result.
  P1-B  optional_fields saved to state.
  P1-C  Correct call signature for enrich_identified_items_with_standards().
  P1-D  populate_schema_with_deep_agent() call added (Step 1.2.1d).
  P1-E  Result-metadata dicts: rag_invocations, schema_population_info,
         deep_agent_info, strategy_info now populated in validation_result.
  P3-B  Enrichment cache moved to module-level singleton (created once).

Writes to state:
  product_type, original_product_type, product_type_refined,
  schema, schema_source, ppi_workflow_used, is_valid,
  missing_fields, optional_fields, provided_requirements, validation_result,
  standards_info, enrichment_result, current_step
"""

import logging
import time
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from common.agentic.models import ProductSearchDeepAgentState


# =============================================================================
# MODULE-LEVEL SINGLETON CACHE  [P3-B]
# =============================================================================
# Created once at import time so the TTL and contents survive across repeated
# calls to validate_product_node within the same process.
_enrichment_cache = None


def _get_enrichment_cache():
    """Return (or lazily create) the module-level enrichment cache."""
    global _enrichment_cache
    if _enrichment_cache is None:
        try:
            from common.infrastructure.caching.bounded_cache import get_or_create_cache
            _enrichment_cache = get_or_create_cache(
                name="session_enrichment",
                max_size=200,
                ttl_seconds=1800,
            )
        except Exception as e:
            logger.warning("[validate_product_node] Could not create cache: %s — using dict fallback", e)
            _enrichment_cache = {}  # plain dict as last resort
    return _enrichment_cache


def _cache_get(cache, key):
    """Unified get that works for both BoundedCache and plain dict."""
    try:
        return cache.get(key)
    except AttributeError:
        return cache.get(key)


def _cache_set(cache, key, value):
    """Unified set that works for both BoundedCache and plain dict."""
    try:
        cache.set(key, value)
    except AttributeError:
        cache[key] = value


def _cache_key(product_type: str, session_id: str) -> str:
    normalized = product_type.lower().strip()
    return f"{session_id}:{normalized}" if session_id else normalized


# =============================================================================
# NODE
# =============================================================================

def validate_product_node(state: "ProductSearchDeepAgentState") -> "ProductSearchDeepAgentState":
    """
    LangGraph node — Step 1: Validate product type, load schema, enrich with standards.
    """
    logger.info("[validate_product_node] ===== STEP 1: VALIDATE PRODUCT =====")
    user_input = state["user_input"]
    session_id = state.get("session_id")
    expected_product_type = state.get("expected_product_type")

    state["current_step"] = "validate_product"

    # =========================================================================
    # 1. UI DECISION GUARD
    # =========================================================================
    try:
        from debug_flags import is_ui_decision_input, get_ui_decision_error_message
        if is_ui_decision_input(user_input):
            logger.warning("[validate_product_node] UI decision pattern detected: '%s'", user_input)
            state["error"] = get_ui_decision_error_message(user_input)
            state["error_type"] = "UIDecisionPatternError"
            state["success"] = False
            return state
    except ImportError:
        ui_patterns = ["user selected:", "user clicked:", "decision:", "continue", "proceed"]
        normalized = user_input.lower().strip()
        for pattern in ui_patterns:
            if pattern in normalized or normalized == pattern:
                state["error"] = (
                    f"Input '{user_input}' appears to be a UI action. "
                    "Please provide product specifications."
                )
                state["error_type"] = "UIDecisionPatternError"
                state["success"] = False
                return state

    t_start = time.monotonic()

    try:
        # =====================================================================
        # 2. EXTRACT PRODUCT TYPE
        # =====================================================================
        logger.info("[validate_product_node] Step 1.1: Extracting product type from user input")
        from common.tools.intent_tools import extract_requirements_tool

        extract_result = extract_requirements_tool.invoke({"user_input": user_input})
        product_type = extract_result.get("product_type") or expected_product_type or ""
        original_product_type = product_type  # saved for P1-A

        if not product_type:
            _has_product_words = any(w in user_input.lower() for w in [
                "transmitter", "sensor", "meter", "gauge", "valve", "pump",
                "analyzer", "controller", "switch", "indicator", "recorder"
            ])
            state["error"] = (
                "Could not determine product type from your input. "
                "Please provide product specifications (e.g., 'pressure transmitter 4-20mA HART')."
            ) if _has_product_words else (
                "No product type detected. Please describe what product you need."
            )
            state["error_type"] = "ProductTypeExtractionError"
            state["success"] = False
            return state

        if expected_product_type and product_type.lower() != expected_product_type.lower():
            logger.warning(
                "[validate_product_node] Product type mismatch — Expected: %s, Detected: %s",
                expected_product_type, product_type
            )

        logger.info("[validate_product_node] ✓ Detected product type: %s", product_type)

        # =====================================================================
        # 3. LOAD SCHEMA (or generate via PPI)
        # =====================================================================
        logger.info("[validate_product_node] Step 1.2: Loading/generating schema for '%s'", product_type)
        from common.tools.schema_tools import load_schema_tool

        schema_result = load_schema_tool.invoke({
            "product_type": product_type,
            "enable_ppi": True,
        })
        schema = schema_result.get("schema", {})
        schema_source = schema_result.get("source", "unknown")
        ppi_used = schema_result.get("ppi_used", False)
        from_database = schema_result.get("from_database", False)

        if from_database:
            logger.info("[validate_product_node] ✓ Schema loaded from Azure Blob Storage")
        elif ppi_used:
            logger.info("[validate_product_node] ✓ Schema generated via PPI workflow")
        else:
            logger.warning("[validate_product_node] ⚠ Using default schema (fallback)")

        # =====================================================================
        # 4. SESSION ENRICHMENT CACHE CHECK  [FIX #A1 / P3-B]
        # =====================================================================
        cache = _get_enrichment_cache()
        key = _cache_key(product_type, session_id)
        cached = _cache_get(cache, key)

        standards_info = None
        enrichment_result = None

        # Track RAG invocation metadata for P1-E
        standards_rag_invoked = False
        standards_rag_success = False
        standards_rag_time = None

        if cached:
            logger.info("[validate_product_node] [FIX #A1] SESSION CACHE HIT — skipping re-enrichment")
            standards_info = cached.get("standards_info")
            enrichment_result = cached.get("enrichment_result")
            schema = cached.get("schema", schema)
            if cached.get("standards_section"):
                schema["standards"] = cached["standards_section"]

            # Restore RAG tracking from cache
            standards_rag_invoked = cached.get("standards_rag_invoked", False)
            standards_rag_success = cached.get("standards_rag_success", False)
            standards_rag_time = cached.get("standards_rag_time")
        else:
            logger.info("[validate_product_node] [FIX #A1] SESSION CACHE MISS — running enrichment")

            # -----------------------------------------------------------------
            # 5. POPULATE FIELD VALUES FROM STANDARDS  [Step 1.2.1a]
            # -----------------------------------------------------------------
            get_applicable_standards = None
            try:
                from common.tools.standards_enrichment_tool import (
                    get_applicable_standards,
                    populate_schema_fields_from_standards,
                )
                if not schema.get("_standards_population"):
                    schema = populate_schema_fields_from_standards(product_type, schema)
                    fields_populated = schema.get("_standards_population", {}).get("fields_populated", 0)
                    logger.info("[validate_product_node] ✓ Populated %d fields with standards values", fields_populated)
            except ImportError as e:
                logger.warning("[validate_product_node] standards_enrichment_tool unavailable: %s", e)

            # -----------------------------------------------------------------
            # 6. COMPREHENSIVE STANDARDS DEFAULTS  [FIX #4]
            # -----------------------------------------------------------------
            try:
                from common.agentic.deep_agent.schema.generation.field_extractor import extract_schema_field_values_from_standards
                fields_before = schema.get("_standards_population", {}).get("fields_populated", 0)
                schema = extract_schema_field_values_from_standards(product_type, schema)
                fields_after = schema.get("_schema_field_extraction", {}).get("fields_populated", 0)
                logger.info("[validate_product_node] [FIX #4] ✓ Standards defaults: %d → %d fields",
                            fields_before, fields_before + fields_after)
            except (ImportError, Exception) as e:
                logger.warning("[validate_product_node] [FIX #4] field_extractor not available: %s", e)

            # -----------------------------------------------------------------
            # 7. TEMPLATE SPECIFICATIONS  [FIX #5]
            # -----------------------------------------------------------------
            try:
                from common.agentic.deep_agent.specifications.templates.templates import get_all_specs_for_product_type
                template_specs = get_all_specs_for_product_type(product_type)
                if template_specs:
                    schema.setdefault("template_specifications", {})
                    specs_added = 0
                    for spec_key, spec_def in template_specs.items():
                        field_exists = any(
                            spec_key in schema.get(section, {})
                            for section in ["mandatory", "optional", "Performance", "Electrical", "Mechanical"]
                        )
                        if not field_exists and getattr(spec_def, "typical_value", None):
                            schema["template_specifications"][spec_key] = {
                                "value": str(spec_def.typical_value),
                                "unit": getattr(spec_def, "unit", "") or "",
                                "category": getattr(spec_def, "category", ""),
                                "description": getattr(spec_def, "description", ""),
                                "importance": (
                                    spec_def.importance.name
                                    if hasattr(spec_def, "importance") and hasattr(spec_def.importance, "name")
                                    else "OPTIONAL"
                                ),
                                "source": "template_specification",
                            }
                            specs_added += 1
                    schema["_template_specs_added"] = specs_added
                    logger.info("[validate_product_node] [FIX #5] ✓ Added %d template specifications", specs_added)
            except (ImportError, Exception) as e:
                logger.debug("[validate_product_node] [FIX #5] template specs unavailable: %s", e)

            # -----------------------------------------------------------------
            # 8. GET APPLICABLE STANDARDS → build schema['standards'] section
            # -----------------------------------------------------------------
            try:
                if get_applicable_standards is None:
                    from common.tools.standards_enrichment_tool import get_applicable_standards
                standards_info = get_applicable_standards(product_type, top_k=5)
                if standards_info.get("success") and "standards" not in schema:
                    schema["standards"] = {
                        "applicable_standards": standards_info.get("applicable_standards", []),
                        "certifications": standards_info.get("certifications", []),
                        "safety_requirements": standards_info.get("safety_requirements", {}),
                        "calibration_standards": standards_info.get("calibration_standards", {}),
                        "environmental_requirements": standards_info.get("environmental_requirements", {}),
                    }
            except (ImportError, Exception) as e:
                logger.warning("[validate_product_node] get_applicable_standards failed: %s", e)

            # -----------------------------------------------------------------
            # 9. STANDARDS RAG ENRICHMENT  [P1-C: corrected call signature]
            # -----------------------------------------------------------------
            standards_rag_invoked = True
            standards_rag_start = time.monotonic()
            standards_rag_time = None
            try:
                # FIXED-PATH: Use correct location in common.standards.rag
                from common.standards.rag.enrichment import (
                    enrich_identified_items_with_standards,
                )
                # P1-C FIX: wrap product as list-of-dicts the way the original did
                product_item = [{
                    "name": product_type,
                    "category": product_type,
                    "specifications": schema.get("mandatory", {}),
                }]
                enrichment_result = enrich_identified_items_with_standards(
                    items=product_item,
                    product_type=product_type,
                    top_k=3,
                )
                standards_rag_success = enrichment_result.get("success", False)
                standards_rag_time = f"{(time.monotonic() - standards_rag_start) * 1000:.0f}ms"
                logger.info(
                    "[validate_product_node] ✓ Standards RAG enrichment: success=%s (%s)",
                    standards_rag_success, standards_rag_time,
                )
            except TypeError:
                # Fallback: some implementations accept positional (product_type, schema, session_id)
                try:
                    from common.standards.rag.enrichment import (
                        enrich_identified_items_with_standards,
                    )
                    enrichment_result = enrich_identified_items_with_standards(
                        product_type=product_type,
                        schema=schema,
                        session_id=session_id,
                    )
                    standards_rag_success = enrichment_result.get("success", False)
                    standards_rag_time = f"{(time.monotonic() - standards_rag_start) * 1000:.0f}ms"
                except Exception as inner_e:
                    logger.warning("[validate_product_node] Standards RAG enrichment failed (fallback): %s", inner_e)
                    enrichment_result = {"success": False}
            except (ImportError, Exception) as e:
                logger.warning("[validate_product_node] Standards RAG enrichment failed: %s", e)
                enrichment_result = {"success": False}

            # -----------------------------------------------------------------
            # 10. DEEP-AGENT SCHEMA POPULATION  [P1-D: NEW — Step 1.2.1d]
            # -----------------------------------------------------------------
            try:
                from common.agentic.deep_agent.schema.populator_legacy import populate_schema_with_deep_agent
                logger.info("[validate_product_node] Step 1.2.1d: Running deep-agent schema population")
                # Fix: Capture both return values (schema, stats)
                schema, deep_agent_stats = populate_schema_with_deep_agent(
                    product_type=product_type,
                    schema=schema,
                    session_id=session_id,
                    use_memory=True,
                )
                
                # Injection: Store stats in schema for downstream reporting (metadata dicts)
                if "_deep_agent_population" not in schema:
                    schema["_deep_agent_population"] = deep_agent_stats
                
                deep_agent_fields = deep_agent_stats.get("fields_populated", 0)
                logger.info(
                    "[validate_product_node] ✓ Deep-agent schema population: %d fields populated",
                    deep_agent_fields,
                )
            except ImportError:
                logger.debug("[validate_product_node] populate_schema_with_deep_agent not available — skipping")
            except Exception as e:
                logger.warning("[validate_product_node] Deep-agent schema population failed: %s", e)

            # Cache the enrichment results for this session
            _cache_set(cache, key, {
                "standards_info": standards_info,
                "enrichment_result": enrichment_result,
                "schema": schema,
                "standards_section": schema.get("standards"),
                "standards_rag_invoked": standards_rag_invoked,
                "standards_rag_success": standards_rag_success,
                "standards_rag_time": standards_rag_time,
            })
            logger.info("[validate_product_node] [FIX #A1] Enrichment cached for session '%s'", session_id)

        # =====================================================================
        # 11. VALIDATE REQUIREMENTS AGAINST SCHEMA
        # =====================================================================
        logger.info("[validate_product_node] Step 1.3: Validating requirements against schema")
        try:
            from common.tools.schema_tools import validate_requirements_tool
            validation_result_raw = validate_requirements_tool.invoke({
                "user_input": user_input,
                "product_type": product_type,
                "schema": schema,
            })
            is_valid = validation_result_raw.get("is_valid", False)
            missing_fields = validation_result_raw.get("missing_fields", [])
            optional_fields = validation_result_raw.get("optional_fields", [])    # P1-B
            provided_requirements = validation_result_raw.get("provided_requirements", {})

            # P1-A: pick up refined product type if the validator detected a better match
            refined_product_type = validation_result_raw.get("refined_product_type")
            product_type_refined = False
            if refined_product_type and refined_product_type.lower() != product_type.lower():
                logger.info(
                    "[validate_product_node] [P1-A] Product type refined: '%s' → '%s'",
                    product_type, refined_product_type,
                )
                product_type = refined_product_type
                product_type_refined = True

        except (ImportError, Exception) as e:
            logger.warning("[validate_product_node] validate_requirements_tool failed: %s — using defaults", e)
            is_valid = True
            missing_fields = []
            optional_fields = []
            provided_requirements = {}
            validation_result_raw = {
                "is_valid": True,
                "missing_fields": [],
                "optional_fields": [],
                "provided_requirements": {},
            }
            refined_product_type = None
            product_type_refined = False

        processing_time_ms = int((time.monotonic() - t_start) * 1000)

        # =====================================================================
        # BUILD RESULT METADATA DICTS  [P1-E]
        # =====================================================================

        # --- rag_invocations ---
        rag_invocations = {
            "standards_rag": {
                "invoked": standards_rag_invoked,
                "invocation_time": standards_rag_time,
                "success": standards_rag_success,
                "product_type": product_type,
            },
            "strategy_rag": {
                "invoked": False,
                "note": "Strategy RAG is applied in analyze_vendors_node, not during validation",
            },
        }

        # --- schema_population_info ---
        pop_info = schema.get("_standards_population", {})
        extraction_info = schema.get("_schema_field_extraction", {})
        deep_agent_pop = schema.get("_deep_agent_population", {})
        template_added = schema.get("_template_specs_added", 0)
        total_fields = (
            pop_info.get("fields_populated", 0)
            + extraction_info.get("fields_populated", 0)
            + deep_agent_pop.get("fields_populated", 0)
            + template_added
        )
        schema_population_info = {
            "standards_population_fields": pop_info.get("fields_populated", 0),
            "field_extraction_fields": extraction_info.get("fields_populated", 0),
            "deep_agent_fields": deep_agent_pop.get("fields_populated", 0),
            "template_specification_fields": template_added,
            "total_fields_populated": total_fields,
            "target_achieved": total_fields >= 5,
        }

        # --- deep_agent_info ---
        deep_agent_info = {
            "fields_populated": deep_agent_pop.get("fields_populated", 0),
            "sections_populated": deep_agent_pop.get("sections", []),
            "processing_time_ms": processing_time_ms,
        }

        # --- strategy_info (placeholder — populated in vendor analysis step) ---
        strategy_info = {
            "applied": False,
            "rag_type": None,
            "preferred_vendors": [],
            "forbidden_vendors": [],
            "confidence": None,
            "strategy_notes": None,
        }

        # =====================================================================
        # WRITE TO STATE
        # =====================================================================
        state["product_type"] = product_type
        state["original_product_type"] = original_product_type    # P1-A
        state["product_type_refined"] = product_type_refined       # P1-A
        state["schema"] = schema
        state["schema_source"] = schema_source
        state["ppi_workflow_used"] = ppi_used
        state["is_valid"] = is_valid
        state["missing_fields"] = missing_fields
        state["optional_fields"] = optional_fields                 # P1-B
        state["provided_requirements"] = provided_requirements
        state["validation_result"] = {
            # Core fields
            "productType": product_type,
            "originalProductType": original_product_type,         # P1-A
            "productTypeRefined": product_type_refined,           # P1-A
            "detectedSchema": schema,
            "providedRequirements": provided_requirements,
            "ppiWorkflowUsed": ppi_used,
            "schemaSource": schema_source,
            "isValid": is_valid,
            "missingFields": missing_fields,
            "optionalFields": optional_fields,                    # P1-B
            # Metadata dicts [P1-E]
            "rag_invocations": rag_invocations,
            "schema_population_info": schema_population_info,
            "deep_agent_info": deep_agent_info,
            "strategy_info": strategy_info,
        }
        state["standards_info"] = standards_info
        state["enrichment_result"] = enrichment_result
        state["messages"] = state.get("messages", []) + [
            {
                "role": "system",
                "content": (
                    f"[Step 1] Validated product: {product_type} | "
                    f"schema_source={schema_source} | ppi={ppi_used} | "
                    f"refined={product_type_refined}"
                )
            }
        ]

        logger.info(
            "[validate_product_node] ✓ Completed — product_type=%s (refined=%s), is_valid=%s, missing=%d, optional=%d",
            product_type, product_type_refined, is_valid, len(missing_fields), len(optional_fields),
        )

    except Exception as exc:
        logger.error("[validate_product_node] ✗ Unexpected failure: %s", exc, exc_info=True)
        state["error"] = str(exc)
        state["error_type"] = type(exc).__name__
        state["success"] = False

    return state
