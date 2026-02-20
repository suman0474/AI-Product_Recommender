"""
Node: analyze_vendors_node
============================

Step 4 of the Product Search Deep Agent.

Directly implements all logic previously inside VendorAnalysisTool.analyze()
with NO class instantiation:

  1.  Load vendors from Azure Blob Storage  → get_vendors_for_product_type()
  2.  Strategy RAG filtering/prioritization → get_strategy_with_auto_fallback()
                                              + filter_vendors_by_strategy()
  3.  Load product data for filtered vendors → get_products_for_vendors()
  4.  Prepare requirements string
  5.  Prepare vendor payloads (JSON product catalog)
  6.  Parallel vendor analysis             → ThreadPoolExecutor
                                              + _analyze_single_vendor()
                                              → invoke_vendor_chain() directly
  7.  Enrich matches with strategy priority
  8.  Enrich with product images (optional)
  9.  Enrich with vendor logos   (optional)
  10. Enrich with pricing links  (optional)

Reads from state:
  structured_requirements, product_type, schema, session_id, max_vendor_workers

Writes to state:
  vendor_analysis_result, vendor_matches, strategy_context, current_step
"""

import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from common.agentic.models import ProductSearchDeepAgentState


# =============================================================================
# MODULE-LEVEL RESPONSE CACHE  [P3-A]
# =============================================================================
# Keyed by MD5(product_type + sorted requirements JSON).
# Saves 200-600 s on identical requests within the same process lifetime.
_response_cache: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_requirements(requirements: Dict[str, Any]) -> str:
    """Format requirements dict into a structured prompt string."""
    lines: List[str] = []

    mandatory = requirements.get("mandatoryRequirements") or requirements.get("mandatory") or {}
    optional = requirements.get("optionalRequirements") or requirements.get("optional") or {}
    advanced = requirements.get("selectedAdvancedParams") or requirements.get("advancedSpecs") or {}

    if mandatory:
        lines.append("## Mandatory Requirements")
        for key, value in mandatory.items():
            if value:
                lines.append(f"- {_title(key)}: {value}")

    if optional:
        lines.append("\n## Optional Requirements")
        for key, value in optional.items():
            if value:
                lines.append(f"- {_title(key)}: {value}")

    if advanced:
        lines.append("\n## Advanced Specifications")
        for key, value in advanced.items():
            if value:
                lines.append(f"- {key}: {value}")

    if not lines:
        return (
            "## Requirements Summary\n"
            "No specific mandatory or optional requirements have been provided for this product search.\n\n"
            "## Analysis Instruction\n"
            "Analyze available products and return JSON with general recommendations based on:\n"
            "- Standard industrial specifications and certifications\n"
            "- Product feature completeness and quality\n"
            "- Typical use case suitability for this product type\n"
            "- Provide match_score based on product quality (use 85-95 range for well-documented, certified products)"
        )

    return "\n".join(lines)


def _title(field: str) -> str:
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", field).replace("_", " ")
    return words.title()


def _analyze_single_vendor(
    components: Dict[str, Any],
    requirements_str: str,
    vendor: str,
    vendor_data: Dict[str, Any],
    applicable_standards: Optional[List[str]] = None,
    standards_specs: Optional[str] = None,
    max_retries: int = 3,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Analyze a single vendor using invoke_vendor_chain() directly.
    Mirrors VendorAnalysisTool._analyze_vendor() with retry on rate limits.
    """
    from core.chaining import invoke_vendor_chain, to_dict_if_pydantic, parse_vendor_analysis_response

    error: Optional[str] = None
    base_retry_delay = 15

    for attempt in range(max_retries):
        try:
            pdf_text = vendor_data.get("pdf_text", "")
            products = vendor_data.get("products", [])

            pdf_payload = json.dumps({vendor: pdf_text}, ensure_ascii=False) if pdf_text else "{}"
            products_payload = json.dumps(products, ensure_ascii=False)

            result = invoke_vendor_chain(
                components,
                vendor,
                requirements_str,
                products_payload,
                pdf_payload,
                components["vendor_format_instructions"],
                applicable_standards=applicable_standards or [],
                standards_specs=standards_specs or "No specific standards requirements provided.",
            )

            result = to_dict_if_pydantic(result)
            result = parse_vendor_analysis_response(result, vendor)

            logger.info("[analyze_vendors_node] ✓ Vendor '%s' analyzed successfully", vendor)
            return result, None

        except Exception as exc:
            error_msg = str(exc)
            is_rate_limit = any(
                x in error_msg.lower()
                for x in ["429", "resource has been exhausted", "quota", "503", "overloaded"]
            )
            if is_rate_limit and attempt < max_retries - 1:
                wait_time = base_retry_delay * (2 ** attempt)
                logger.warning("[analyze_vendors_node] Rate limit for %s, retry %d/%d after %ds",
                               vendor, attempt + 1, max_retries, wait_time)
                time.sleep(wait_time)
                continue
            error = error_msg
            logger.error("[analyze_vendors_node] ✗ Vendor '%s' failed: %s", vendor, error)
            break

    return None, error


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def analyze_vendors_node(state: "ProductSearchDeepAgentState") -> "ProductSearchDeepAgentState":
    """
    LangGraph node — Step 4: Load vendors, apply Strategy RAG, run parallel analysis.
    """
    logger.info("[analyze_vendors_node] ===== STEP 4: VENDOR ANALYSIS =====")
    state["current_step"] = "analyze_vendors"

    product_type = state.get("product_type", "")
    structured_requirements = state.get("structured_requirements") or {}
    session_id = state.get("session_id")
    max_workers = state.get("max_vendor_workers", 10)

    result: Dict[str, Any] = {
        "success": False,
        "product_type": product_type,
        "session_id": session_id,
    }

    # -------------------------------------------------------------------------
    # P3-A: Check module-level response cache (saves 200-600 s on repeat calls)
    # -------------------------------------------------------------------------
    try:
        _cache_key = hashlib.md5(
            f"{product_type}:{json.dumps(structured_requirements, sort_keys=True, default=str)}".encode()
        ).hexdigest()
        if _cache_key in _response_cache:
            logger.info(
                "[analyze_vendors_node] ✓ Response cache HIT (key=%s) — skipping analysis",
                _cache_key[:12],
            )
            cached = _response_cache[_cache_key]
            state["vendor_analysis_result"] = cached
            state["vendor_matches"] = cached.get("vendor_matches", [])
            state["strategy_context"] = cached.get("strategy_context")
            state["messages"] = state.get("messages", []) + [
                {
                    "role": "system",
                    "content": (
                        f"[Step 4] Vendor analysis (cached): {cached.get('total_matches', 0)} matches"
                    ),
                }
            ]
            return state
        logger.info("[analyze_vendors_node] Response cache MISS (key=%s)", _cache_key[:12])
    except Exception as cache_err:
        logger.warning("[analyze_vendors_node] Cache check failed: %s", cache_err)
        _cache_key = None  # type: ignore[assignment]

    try:
        from services.azure.blob_utils import get_vendors_for_product_type, get_products_for_vendors
        from core.chaining import setup_langchain_components, to_dict_if_pydantic

        # ---------------------------------------------------------------------
        # 1. LOAD VENDORS
        # ---------------------------------------------------------------------
        logger.info("[analyze_vendors_node] Step 4.1: Loading vendors for '%s'", product_type)
        vendors: List[str] = get_vendors_for_product_type(product_type) if product_type else []
        logger.info("[analyze_vendors_node] Found %d vendors", len(vendors))

        if not vendors:
            logger.warning("[analyze_vendors_node] No vendors found — returning empty result")
            result.update({
                "success": True,
                "vendor_matches": [],
                "vendor_run_details": [],
                "total_matches": 0,
                "vendors_analyzed": 0,
                "analysis_summary": "No vendors available for analysis",
            })
            state["vendor_analysis_result"] = result
            state["vendor_matches"] = []
            return state

        # ---------------------------------------------------------------------
        # 2. STRATEGY RAG FILTERING  [Step 2.5]
        # ---------------------------------------------------------------------
        logger.info("[analyze_vendors_node] Step 4.2: Applying Strategy RAG to filter/prioritize vendors")
        strategy_context: Optional[Dict[str, Any]] = None
        filtered_vendors = vendors.copy()
        excluded_vendors: List[Dict[str, Any]] = []
        vendor_priorities: Dict[str, int] = {}

        try:
            from common.strategy_rag.strategy_rag_enrichment import (
                get_strategy_with_auto_fallback,
                filter_vendors_by_strategy,
            )
            strategy_context = get_strategy_with_auto_fallback(
                product_type=product_type,
                requirements=structured_requirements,
                top_k=7,
            )
            if strategy_context and strategy_context.get("success"):
                rag_type = strategy_context.get("rag_type", "unknown")
                preferred = strategy_context.get("preferred_vendors", [])
                forbidden = strategy_context.get("forbidden_vendors", [])
                logger.info("[analyze_vendors_node] Strategy RAG (%s): preferred=%s, forbidden=%s",
                            rag_type, preferred, forbidden)

                filter_result = filter_vendors_by_strategy(vendors, strategy_context)
                accepted = filter_result.get("accepted_vendors", [])
                excluded_vendors = filter_result.get("excluded_vendors", [])

                if accepted:
                    filtered_vendors = [v["vendor"] for v in accepted]
                    vendor_priorities = {v["vendor"]: v.get("priority_score", 0) for v in accepted}
                    logger.info("[analyze_vendors_node] Strategy filtering: %d accepted, %d excluded",
                                len(filtered_vendors), len(excluded_vendors))
                else:
                    logger.warning("[analyze_vendors_node] No vendors passed strategy filter — using all")
            else:
                logger.warning("[analyze_vendors_node] Strategy RAG returned no results — using all vendors")

        except Exception as strategy_err:
            logger.warning("[analyze_vendors_node] Strategy RAG failed: %s — using all vendors", strategy_err)

        # ---------------------------------------------------------------------
        # 3. LOAD PRODUCT DATA
        # ---------------------------------------------------------------------
        logger.info("[analyze_vendors_node] Step 4.3: Loading product data for %d vendors", len(filtered_vendors))
        products_data = get_products_for_vendors(filtered_vendors, product_type)

        if not products_data:
            logger.warning("[analyze_vendors_node] No product data loaded")
            result.update({
                "success": True,
                "vendor_matches": [],
                "vendor_run_details": [],
                "total_matches": 0,
                "vendors_analyzed": 0,
                "analysis_summary": "No product data available for analysis",
            })
            state["vendor_analysis_result"] = result
            state["vendor_matches"] = []
            state["strategy_context"] = strategy_context
            return state

        # ---------------------------------------------------------------------
        # 4. PREPARE REQUIREMENTS STRING
        # ---------------------------------------------------------------------
        requirements_str = _format_requirements(structured_requirements)

        # ---------------------------------------------------------------------
        # 5. PREPARE VENDOR PAYLOADS
        # ---------------------------------------------------------------------
        vendor_payloads: Dict[str, Dict[str, Any]] = {}
        for vendor_name, products in products_data.items():
            if products:
                products_json = json.dumps(products, indent=2, ensure_ascii=False)
                vendor_payloads[vendor_name] = {
                    "products": products,
                    "pdf_text": products_json,
                }
        logger.info("[analyze_vendors_node] Prepared %d vendor payloads", len(vendor_payloads))

        if not vendor_payloads:
            result.update({
                "success": True,
                "vendor_matches": [],
                "vendor_run_details": [],
                "total_matches": 0,
                "vendors_analyzed": 0,
                "analysis_summary": "No vendor data available for analysis",
            })
            state["vendor_analysis_result"] = result
            state["vendor_matches"] = []
            state["strategy_context"] = strategy_context
            return state

        # Standards from requirements
        applicable_standards: List[str] = []
        standards_specs = "No specific standards requirements provided."
        if isinstance(structured_requirements, dict):
            applicable_standards = structured_requirements.get("applicable_standards", [])
            if "standards_specifications" in structured_requirements:
                standards_specs = structured_requirements["standards_specifications"]

        # ---------------------------------------------------------------------
        # 6. SETUP LANGCHAIN COMPONENTS
        # ---------------------------------------------------------------------
        logger.info("[analyze_vendors_node] Step 4.4: Setting up LangChain components")
        components = setup_langchain_components()

        # ---------------------------------------------------------------------
        # 7. PARALLEL VENDOR ANALYSIS
        # ---------------------------------------------------------------------
        logger.info("[analyze_vendors_node] Step 4.5: Running parallel analysis for %d vendors", len(vendor_payloads))
        vendor_matches: List[Dict[str, Any]] = []
        run_details: List[Dict[str, Any]] = []

        actual_workers = min(len(vendor_payloads), max_workers)
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {
                executor.submit(
                    _analyze_single_vendor,
                    components,
                    requirements_str,
                    vendor,
                    data,
                    applicable_standards,
                    standards_specs,
                ): vendor
                for vendor, data in vendor_payloads.items()
            }

            for future in as_completed(futures):
                vendor = futures[future]
                try:
                    vendor_result, error = future.result()
                    if vendor_result and isinstance(vendor_result.get("vendor_matches"), list):
                        for match in vendor_result["vendor_matches"]:
                            match_dict = to_dict_if_pydantic(match)
                            normalized = {
                                "productName": match_dict.get("product_name", match_dict.get("productName", "")),
                                "vendor": vendor,
                                "modelFamily": match_dict.get("model_family", match_dict.get("modelFamily", "")),
                                "productType": match_dict.get("product_type", match_dict.get("productType", "")),
                                "matchScore": match_dict.get("match_score", match_dict.get("matchScore", 0)),
                                "requirementsMatch": match_dict.get("requirements_match", match_dict.get("requirementsMatch", False)),
                                "reasoning": match_dict.get("reasoning", ""),
                                "limitations": match_dict.get("limitations", ""),
                                "productDescription": match_dict.get("product_description", match_dict.get("productDescription", "")),
                                "standardsCompliance": match_dict.get("standards_compliance", match_dict.get("standardsCompliance", {})),
                                "matchedRequirements": match_dict.get("matched_requirements", match_dict.get("matchedRequirements", {})),
                                "unmatchedRequirements": match_dict.get("unmatched_requirements", match_dict.get("unmatchedRequirements", [])),
                                "keyStrengths": match_dict.get("key_strengths", match_dict.get("keyStrengths", [])),
                                "recommendation": match_dict.get("recommendation", ""),
                            }
                            vendor_matches.append(normalized)
                        run_details.append({"vendor": vendor, "status": "success"})
                    else:
                        run_details.append({"vendor": vendor, "status": "failed" if error else "empty", "error": error})
                except Exception as exc:
                    run_details.append({"vendor": vendor, "status": "error", "error": str(exc)})
                    logger.error("[analyze_vendors_node] ✗ Exception for vendor '%s': %s", vendor, exc)

        # ---------------------------------------------------------------------
        # 8. ENRICH WITH STRATEGY PRIORITY
        # ---------------------------------------------------------------------
        preferred_lower = [
            p.lower() for p in (strategy_context.get("preferred_vendors", []) if strategy_context else [])
        ]
        for match in vendor_matches:
            vname = match.get("vendor", "")
            match["strategy_priority"] = vendor_priorities.get(vname, 0)
            match["is_preferred_vendor"] = vname.lower() in preferred_lower
            if "requirementsMatch" not in match:
                match["requirementsMatch"] = match.get("matchScore", 0) >= 80

        # ---------------------------------------------------------------------
        # 9. ENRICH WITH PRODUCT IMAGES (optional)
        # ---------------------------------------------------------------------
        try:
            from common.utils.vendor_images import fetch_images_for_vendor_matches
            vendor_groups: Dict[str, List] = {}
            for m in vendor_matches:
                vendor_groups.setdefault(m.get("vendor", ""), []).append(m)

            for vname, vmatches in vendor_groups.items():
                try:
                    enriched = fetch_images_for_vendor_matches(vendor_name=vname, matches=vmatches, max_workers=2)
                    for em in enriched:
                        for i, m in enumerate(vendor_matches):
                            if m.get("vendor") == em.get("vendor") and m.get("productName") == em.get("productName"):
                                vendor_matches[i] = em
                except Exception as img_err:
                    logger.warning("[analyze_vendors_node] Image enrichment failed for %s: %s", vname, img_err)
        except ImportError:
            logger.debug("[analyze_vendors_node] vendor_images not available — skipping")

        # ---------------------------------------------------------------------
        # 10. ENRICH WITH VENDOR LOGOS (optional)
        # ---------------------------------------------------------------------
        try:
            from common.utils.vendor_images import enrich_matches_with_logos
            vendor_matches = enrich_matches_with_logos(matches=vendor_matches, max_workers=2)
        except (ImportError, Exception) as logo_err:
            logger.debug("[analyze_vendors_node] Logo enrichment skipped: %s", logo_err)

        # ---------------------------------------------------------------------
        # 11. ENRICH WITH PRICING LINKS (optional)
        # ---------------------------------------------------------------------
        try:
            from common.utils.pricing_search import enrich_matches_with_pricing
            vendor_matches = enrich_matches_with_pricing(matches=vendor_matches, max_workers=5)
        except (ImportError, Exception) as price_err:
            logger.debug("[analyze_vendors_node] Pricing enrichment skipped: %s", price_err)

        # ---------------------------------------------------------------------
        # BUILD RESULT
        # ---------------------------------------------------------------------
        strategy_source = strategy_context.get("rag_type", "none") if strategy_context else "none"
        successful = sum(1 for d in run_details if d["status"] == "success")
        analysis_summary = (
            f"Strategy RAG ({strategy_source}): Filtered {len(vendors)} → {len(filtered_vendors)} vendors | "
            f"Analyzed {len(vendor_payloads)} vendors, {successful} successful | "
            f"Found {len(vendor_matches)} matching products"
        )

        result.update({
            "success": True,
            "vendor_matches": vendor_matches,
            "vendor_run_details": run_details,
            "total_matches": len(vendor_matches),
            "vendors_analyzed": len(vendor_payloads),
            "original_vendor_count": len(vendors),
            "filtered_vendor_count": len(filtered_vendors),
            "excluded_by_strategy": len(excluded_vendors),
            "strategy_context": {
                "applied": strategy_context is not None and strategy_context.get("success", False),
                "rag_type": strategy_context.get("rag_type") if strategy_context else None,
                "preferred_vendors": strategy_context.get("preferred_vendors", []) if strategy_context else [],
                "forbidden_vendors": strategy_context.get("forbidden_vendors", []) if strategy_context else [],
                "excluded_vendors": excluded_vendors,
                "vendor_priorities": vendor_priorities,
                "confidence": strategy_context.get("confidence", 0.0) if strategy_context else 0.0,
                "strategy_notes": strategy_context.get("strategy_notes", "") if strategy_context else "",
            },
            "analysis_summary": analysis_summary,
        })

        logger.info("[analyze_vendors_node] ✓ %s", analysis_summary)

    except Exception as exc:
        logger.error("[analyze_vendors_node] ✗ Unexpected failure: %s", exc, exc_info=True)
        result.update({
            "success": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
        })

    state["vendor_analysis_result"] = result
    state["vendor_matches"] = result.get("vendor_matches", [])
    state["strategy_context"] = result.get("strategy_context")
    state["messages"] = state.get("messages", []) + [
        {
            "role": "system",
            "content": (
                f"[Step 4] Vendor analysis: {result.get('total_matches', 0)} matches found"
                if result.get("success") else
                f"[Step 4] Vendor analysis failed: {result.get('error', 'unknown')}"
            ),
        }
    ]

    # P3-A: Write successful result to cache
    if result.get("success") and _cache_key:
        try:
            _response_cache[_cache_key] = result
            logger.info("[analyze_vendors_node] ✓ Cached result (key=%s)", _cache_key[:12])
        except Exception as cache_write_err:
            logger.warning("[analyze_vendors_node] Cache write failed: %s", cache_write_err)

    return state
