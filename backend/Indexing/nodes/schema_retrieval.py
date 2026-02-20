"""
Indexing Agent — Schema Retrieval Node
=====================================
Designs and retrieves product schemas from validated specifications.
Now includes built-in validation of synthesized/extracted specs.
Replaces SchemaArchitectAgent and ValidationAgent.
"""

import logging
from typing import Dict, Any, List

from ..utils.llm_helpers import get_llm, invoke_llm_with_prompt, parse_json_response
from ..utils.prompt_loader import load_prompt
from ..state import IndexingState
from .. import config

logger = logging.getLogger(__name__)


# ── Azure / Standards helpers (lazy-imported to avoid circular deps) ───────

def _try_load_schema_from_azure(product_type: str):
    """Attempt to load an existing schema from Azure Blob Storage."""
    try:
        from common.services.schema_service import schema_service
        schema = schema_service.get_schema(product_type)
        if schema and (schema.get("mandatory_requirements") or schema.get("mandatory")):
            return schema
    except Exception as e:
        logger.warning(f"Schema service lookup failed for '{product_type}': {e}")
    return None


def _enrich_with_standards(product_type: str, schema: dict) -> dict:
    """Enrich a schema with Standards RAG field values and standards section."""
    try:
        from common.tools.standards_enrichment_tool import (
            populate_schema_fields_from_standards,
            enrich_schema_with_standards,
        )
        if not schema.get("_standards_population"):
            schema = populate_schema_fields_from_standards(product_type, schema)
        if "standards" not in schema:
            schema = enrich_schema_with_standards(product_type, schema)
        fields_populated = schema.get("_standards_population", {}).get("fields_populated", 0)
        logger.info(f"Standards RAG enriched schema with {fields_populated} field values")
    except Exception as e:
        logger.warning(f"Standards RAG enrichment failed (using original): {e}")
    return schema


def _persist_schema_to_azure(product_type: str, schema: dict) -> bool:
    """Save a generated/enriched schema back to Azure for future reuse."""
    try:
        from common.services.schema_service import SchemaService
        svc = SchemaService()
        success = svc.save_schema(product_type, schema)
        if success:
            logger.info(f"Schema persisted to Azure for '{product_type}'")
        return success
    except Exception as e:
        logger.warning(f"Schema persistence failed for '{product_type}': {e}")
        return False

# ── Parameter categorisation keywords ──────────────────────────────────────
_CATEGORY_KEYWORDS = {
    "physical_characteristics": ["size", "weight", "dimension", "material", "construction"],
    "performance_specs": ["accuracy", "range", "resolution", "response", "sensitivity"],
    "operating_conditions": ["temperature", "pressure", "humidity", "environment"],
    "electrical_specs": ["voltage", "current", "power", "supply", "consumption"],
    "communication": ["protocol", "output", "signal", "interface", "communication"],
    "compliance": ["certification", "standard", "compliance", "approval", "rating"],
}


# ── Validation Helper functions ─────────────────────────────────────────────

def _check_consistency(
    specs: Dict[str, Any],
    all_sources: List[Dict[str, Any]],
) -> float:
    """Cross-source parameter consistency."""
    if len(all_sources) <= 1:
        return 0.8

    parameters = specs.get("parameters", {})
    if not parameters:
        return 0.3

    matches = total = 0
    for param_name in parameters:
        total += 1
        source_count = sum(
            1 for src in all_sources
            if param_name in src.get("parameters", {})
        )
        if source_count >= 2:
            matches += 1

    score = matches / total if total > 0 else 0.5
    logger.debug(f"Consistency: {matches}/{total} params matched across sources")
    return score


def _check_completeness(specs: Dict[str, Any]) -> float:
    """Essential-field and parameter-count checks."""
    score = checks = 0

    for field in ("parameters", "specifications", "model_families"):
        checks += 1
        if field in specs and specs[field]:
            score += 1.0

    param_count = len(specs.get("parameters", {}))
    checks += 1
    if param_count >= 15:
        score += 1.0
    elif param_count >= 10:
        score += 0.7
    elif param_count >= 5:
        score += 0.4

    model_families = specs.get("model_families", [])
    checks += 1
    if len(model_families) >= 3:
        score += 1.0
    elif len(model_families) >= 1:
        score += 0.6

    return score / checks if checks > 0 else 0.0


def _check_data_quality(specs: Dict[str, Any]) -> float:
    """Value / unit presence checks."""
    score = checks = 0

    for _, param_data in specs.get("parameters", {}).items():
        checks += 1
        if isinstance(param_data, dict):
            if "value" in param_data or "range" in param_data:
                score += 0.6
            if "unit" in param_data:
                score += 0.4
        elif param_data:
            score += 0.5

    return score / checks if checks > 0 else 0.5


def _check_coherence(
    specs: Dict[str, Any],
    product_type: str,
    llm,
    system_prompt: str,
    user_prompt_template: str,
) -> float:
    """LLM-based logical coherence check."""
    try:
        user_prompt = user_prompt_template.format(
            product_type=product_type,
            specifications=str(specs)[:3000],
        )
        response = invoke_llm_with_prompt(
            llm=llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        assessment = parse_json_response(response)
        if assessment and "coherence_score" in assessment:
            return float(assessment["coherence_score"])
        return 0.7
    except Exception as e:
        logger.warning(f"Coherence check failed: {e}")
        return 0.7


def _calculate_overall_confidence(validations: Dict[str, float]) -> float:
    """Weighted overall confidence (consistency 30%, completeness 25%, quality 25%, coherence 20%)."""
    weights = {
        "consistency": 0.30,
        "completeness": 0.25,
        "quality": 0.25,
        "coherence": 0.20,
    }
    total_score = total_weight = 0.0
    for check, score in validations.items():
        w = weights.get(check, 0.1)
        total_score += score * w
        total_weight += w
    return total_score / total_weight if total_weight > 0 else 0.5


def _identify_issues(validations: Dict[str, float]) -> List[str]:
    issues: List[str] = []
    if validations.get("consistency", 1.0) < 0.5:
        issues.append("Low cross-source consistency")
    if validations.get("completeness", 1.0) < 0.6:
        issues.append("Incomplete specification data")
    if validations.get("quality", 1.0) < 0.5:
        issues.append("Missing units or value ranges")
    if validations.get("coherence", 1.0) < 0.6:
        issues.append("Logical inconsistencies detected")
    return issues


def _identify_strengths(validations: Dict[str, float]) -> List[str]:
    strengths: List[str] = []
    if validations.get("consistency", 0.0) >= 0.8:
        strengths.append("High cross-source agreement")
    if validations.get("completeness", 0.0) >= 0.8:
        strengths.append("Comprehensive parameter coverage")
    if validations.get("quality", 0.0) >= 0.8:
        strengths.append("High data quality with units")
    if validations.get("coherence", 0.0) >= 0.8:
        strengths.append("Logically coherent specifications")
    return strengths


# ── Schema Helper functions ─────────────────────────────────────────────────

def _default_schema_structure() -> Dict[str, Any]:
    return {
        "categories": [
            "Physical Characteristics",
            "Performance Specifications",
            "Operating Conditions",
            "Electrical Specifications",
            "Communication & Interface",
            "Compliance & Standards",
        ],
        "required_fields": [
            "measurement_range",
            "accuracy",
            "operating_temperature",
        ],
    }


def _design_schema_structure(
    product_type: str,
    specifications: Dict[str, Any],
    llm,
    system_prompt: str,
    user_prompt_template: str,
) -> Dict[str, Any]:
    """LLM-designed schema structure."""
    try:
        user_prompt = user_prompt_template.format(
            product_type=product_type,
            specifications=str(specifications)[:4000],
        )
        response = invoke_llm_with_prompt(
            llm=llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        structure = parse_json_response(response)
        return structure if structure else _default_schema_structure()
    except Exception as e:
        logger.error(f"Schema structure design failed: {e}")
        return _default_schema_structure()


def _organize_parameters(
    parameters: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Categorise parameters by keyword matching."""
    categorized: Dict[str, List[Dict[str, Any]]] = {k: [] for k in _CATEGORY_KEYWORDS}
    categorized["other"] = []

    for param_name, param_data in parameters.items():
        param_lower = param_name.lower()
        placed = False
        for category, keywords in _CATEGORY_KEYWORDS.items():
            if any(kw in param_lower for kw in keywords):
                categorized[category].append({"name": param_name, "data": param_data})
                placed = True
                break
        if not placed:
            categorized["other"].append({"name": param_name, "data": param_data})

    return {k: v for k, v in categorized.items() if v}


def _extract_model_families(
    specifications: Dict[str, Any],
    vendor_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge model families from specs + vendor data."""
    families: List[Dict[str, Any]] = []
    seen_names: set = set()

    for model in specifications.get("model_families", []):
        if isinstance(model, str):
            families.append({"name": model, "variants": []})
            seen_names.add(model)
        elif isinstance(model, dict):
            families.append(model)
            seen_names.add(model.get("name", ""))

    for vendor in vendor_data:
        for model in vendor.get("model_families", []):
            if model not in seen_names:
                families.append({
                    "name": model,
                    "vendor": vendor.get("vendor", ""),
                    "variants": [],
                })
                seen_names.add(model)

    return families


def _build_schema(
    product_type: str,
    structure: Dict[str, Any],
    parameters: Dict[str, List[Dict[str, Any]]],
    model_families: List[Dict[str, Any]],
    specifications: Dict[str, Any],
    validation_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the final schema dict."""
    return {
        "product_type": product_type,
        "version": "1.0",
        "confidence_score": validation_results.get("overall_confidence", 0.5),
        "parameters": parameters,
        "model_families": model_families,
        "metadata": {
            "categories": structure.get("categories", []),
            "required_fields": structure.get("required_fields", []),
            "parameter_count": sum(len(p) for p in parameters.values()),
            "model_count": len(model_families),
            "validation_issues": validation_results.get("issues", []),
            "validation_strengths": validation_results.get("strengths", []),
        },
        "specifications": specifications.get("specifications", []),
        "features": specifications.get("features", []),
        "notes": specifications.get("notes", ""),
        "sources": {
            "pdf_count": len(
                specifications.get("synthesis_metadata", {}).get("sources", [])
            ),
            "synthesis_metadata": specifications.get("synthesis_metadata", {}),
        },
    }


def _validate_schema_completeness(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Quick completeness check on the generated schema."""
    report: Dict[str, Any] = {
        "is_complete": True,
        "missing_fields": [],
        "recommendations": [],
    }
    for field in ("product_type", "parameters", "model_families", "metadata"):
        if field not in schema or not schema[field]:
            report["is_complete"] = False
            report["missing_fields"].append(field)

    param_count = schema.get("metadata", {}).get("parameter_count", 0)
    if param_count < 5:
        report["recommendations"].append("Consider adding more parameters")

    if not schema.get("model_families"):
        report["recommendations"].append("Add model family information if available")

    return report


def _refine_schema(
    schema: Dict[str, Any],
    suggestions: List[str],
    llm,
) -> Dict[str, Any]:
    """LLM-based schema refinement."""
    try:
        prompt = (
            f"Refine the following product schema based on these suggestions:\n\n"
            f"Current Schema:\n{str(schema)[:3000]}\n\n"
            f"Refinement Suggestions:\n"
            + "\n".join(f"- {s}" for s in suggestions)
            + "\n\nReturn the refined schema as JSON."
        )
        response = llm.invoke(prompt)
        refined = parse_json_response(response.content)

        if refined:
            refined["metadata"] = schema.get("metadata", {})
            refined["metadata"]["refinement_applied"] = True
            return refined
        return schema

    except Exception as e:
        logger.error(f"Schema refinement failed: {e}")
        return schema


# ── Node function ───────────────────────────────────────────────────────────

def schema_retrieval_node(state: IndexingState) -> dict:
    """
    LangGraph node — validates specs AND retrieves product schema.
    Replaces both ValidationNode and SchemaRetrievalNode.

    Reads:
        ``product_type``, ``synthesized_specs``, ``extracted_specs``,
        ``vendors``, ``generated_schema``, ``qa_assessment``,
        ``refinement_count``

    Writes:
        ``validation_results``, ``generated_schema``, ``schema_completeness``,
        ``refinement_count``, ``current_stage``, ``agent_outputs``
    """
    product_type = state["product_type"]
    specifications = state.get("synthesized_specs", {})
    all_sources = state.get("extracted_specs", []) # Required for validation
    vendors = state.get("vendors", [])
    existing_schema = state.get("generated_schema", {})
    qa = state.get("qa_assessment", {})
    refinement_count = state.get("refinement_count", 0)

    llm = get_llm(model=config.DEFAULT_MODEL, temperature=0.2)

    # ── Refinement path (Skip validation for refinement) ────────────────
    if existing_schema and qa.get("improvement_recommendations"):
        logger.info(f"Refining schema (refinement #{refinement_count + 1})")
        schema = _refine_schema(
            existing_schema,
            qa["improvement_recommendations"],
            llm,
        )
        completeness = _validate_schema_completeness(schema)
        # Preserve previous validation results if they exist, or use defaults
        val_results = state.get("validation_results", {"overall_confidence": 0.5})

        return {
            "generated_schema": schema,
            "schema_completeness": completeness,
            "schema_source": "refined",
            "refinement_count": refinement_count + 1,
            "current_stage": "schema_retrieval",
            "agent_outputs": {
                "schema_retrieval": {
                    "parameter_count": schema.get("metadata", {}).get("parameter_count", 0),
                    "is_complete": completeness["is_complete"],
                    "refinement": refinement_count + 1,
                    "status": "completed",
                }
            },
        }

    # ── Fresh generation path (Includes Validation) ─────────────────────

    # A1: Check Azure Blob for an existing schema before full generation
    azure_schema = _try_load_schema_from_azure(product_type)
    if azure_schema:
        logger.info(f"Schema found in Azure for '{product_type}' — skipping generation")
        # Still enrich in case standards data has been updated since last save
        azure_schema = _enrich_with_standards(product_type, azure_schema)
        completeness = _validate_schema_completeness(azure_schema)

        return {
            "generated_schema": azure_schema,
            "schema_completeness": completeness,
            "schema_source": "azure",
            "refinement_count": 0,
            "current_stage": "schema_retrieval",
            "agent_outputs": {
                "schema_retrieval": {
                    "parameter_count": azure_schema.get("metadata", {}).get("parameter_count", 0),
                    "is_complete": completeness["is_complete"],
                    "source": "azure",
                    "status": "completed",
                }
            },
        }

    # 1. Validation Logic
    val_system_prompt = load_prompt("validation_agent_system_prompt")
    val_user_prompt = load_prompt("validation_agent_user_prompt")

    checks = {
        "consistency": _check_consistency(specifications, all_sources),
        "completeness": _check_completeness(specifications),
        "quality": _check_data_quality(specifications),
        "coherence": _check_coherence(
            specifications, product_type, llm,
            val_system_prompt, val_user_prompt,
        ),
    }

    overall_confidence = _calculate_overall_confidence(checks)
    issues = _identify_issues(checks)
    strengths = _identify_strengths(checks)

    validation_results = {
        "overall_confidence": overall_confidence,
        "validations": checks,
        "issues": issues,
        "strengths": strengths,
    }

    logger.info(f"Spec validation complete — confidence={overall_confidence:.2f}")

    # 2. Schema Generation Logic
    system_prompt = load_prompt("schema_architect_system_prompt")
    user_prompt_template = load_prompt("schema_architect_user_prompt")

    structure = _design_schema_structure(
        product_type, specifications, llm, system_prompt, user_prompt_template,
    )
    organised_params = _organize_parameters(specifications.get("parameters", {}))
    model_families = _extract_model_families(specifications, vendors)

    schema = _build_schema(
        product_type, structure, organised_params,
        model_families, specifications, validation_results,
    )

    # A2: Enrich generated schema with Standards RAG
    schema = _enrich_with_standards(product_type, schema)

    completeness = _validate_schema_completeness(schema)

    # A3: Persist the newly generated schema to Azure for future reuse
    _persist_schema_to_azure(product_type, schema)

    logger.info(
        f"Schema generated — "
        f"{schema.get('metadata', {}).get('parameter_count', 0)} parameters"
    )

    return {
        "validation_results": validation_results, # Write validation results to state
        "generated_schema": schema,
        "schema_completeness": completeness,
        "schema_source": "generated",
        "refinement_count": 0,
        "current_stage": "schema_retrieval",
        "agent_outputs": {
            "validation": { # Log validation output as sub-step
                "overall_confidence": overall_confidence,
                "issues_found": len(issues),
                "status": "completed",
            },
            "schema_retrieval": {
                "parameter_count": schema.get("metadata", {}).get("parameter_count", 0),
                "is_complete": completeness["is_complete"],
                "source": "generated",
                "status": "completed",
            }
        },
    }
