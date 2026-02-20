"""
Node: rank_products_node
==========================

Step 5 of the Product Search Deep Agent.

Directly implements all ranking logic previously inside RankingTool.rank()
with NO class instantiation:

  Primary path:
    - setup_langchain_components()
    - invoke_ranking_chain()        ← uses get_ranking_prompt internally
    - to_dict_if_pydantic()
    - _normalize_ranked_products()  ← camelCase + enrichment from vendor analysis

  Fallback path (if LLM ranking returns nothing):
    - get_final_ranking()           ← score-based sort without LLM

Reads from state:
  vendor_analysis_result, vendor_matches, session_id

Writes to state:
  ranking_result, overall_ranking, top_product, current_step
"""

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from common.agentic.models import ProductSearchDeepAgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_reasoning(field_data: Union[str, List[Any]]) -> str:
    """Normalize complex LLM output (str or list of dicts) into clean Markdown."""
    if isinstance(field_data, str):
        return field_data
    if isinstance(field_data, list):
        items: List[str] = []
        for item in field_data:
            if isinstance(item, dict):
                param = item.get("parameter", "")
                spec = item.get("input_value", item.get("product_specification", ""))
                text = item.get("holistic_explanation", item.get("explanation", item.get("limitation", "")))
                if not param and not spec and text:
                    items.append(f"- {text}")
                elif param:
                    items.append(f"**{param}**: {spec} - {text}")
                else:
                    items.append(str(item))
            else:
                items.append(str(item))
        return "\n\n".join(items)
    return str(field_data)


def _normalize_ranked_products(
    ranking_dict: Dict[str, Any],
    vendor_analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Normalize the raw LLM ranking output to camelCase and enrich with
    pricing/description/standards data from the vendor analysis matches.
    """
    ranked_products = ranking_dict.get("ranked_products", ranking_dict.get("rankedProducts", []))

    # Build lookup tables from vendor analysis
    pricing_lookup: Dict = {}
    description_lookup: Dict = {}
    standards_lookup: Dict = {}
    matched_reqs_lookup: Dict = {}

    for match in vendor_analysis.get("vendor_matches", []):
        key = (match.get("vendor", ""), match.get("productName", match.get("product_name", "")))
        pricing_lookup[key] = {
            "pricing_url": match.get("pricing_url", ""),
            "pricing_source": match.get("pricing_source", ""),
        }
        description_lookup[key] = match.get("productDescription", match.get("product_description", ""))
        standards_lookup[key] = match.get("standardsCompliance", match.get("standards_compliance", {}))
        matched_reqs_lookup[key] = {
            "matched_requirements": match.get("matchedRequirements", match.get("matched_requirements", {})),
            "unmatched_requirements": match.get("unmatchedRequirements", match.get("unmatched_requirements", [])),
        }

    normalized: List[Dict[str, Any]] = []
    for product in ranked_products:
        p_vendor = product.get("vendor", "")
        p_name = product.get("product_name", product.get("productName", ""))
        key = (p_vendor, p_name)

        pricing_info = pricing_lookup.get(key, {})
        product_desc = description_lookup.get(key, "")
        standards_info = standards_lookup.get(key, {})
        reqs_info = matched_reqs_lookup.get(key, {})

        normalized.append({
            "productName": p_name,
            "vendor": p_vendor,
            "modelFamily": product.get("model_family", product.get("modelFamily", "")),
            "overallScore": product.get("overall_score", product.get("overallScore",
                            product.get("match_score", product.get("matchScore", 0)))),
            "matchScore": product.get("match_score", product.get("matchScore",
                          product.get("overall_score", product.get("overallScore", 0)))),
            "requirementsMatch": product.get("requirements_match", product.get("requirementsMatch", False)),
            "keyStrengths": _normalize_reasoning(product.get("key_strengths", product.get("keyStrengths", []))),
            "concerns": _normalize_reasoning(product.get("concerns", product.get("limitations", []))),
            "productDescription": product_desc,
            "standardsCompliance": standards_info,
            "matchedRequirements": reqs_info.get("matched_requirements", {}),
            "unmatchedRequirements": reqs_info.get("unmatched_requirements", []),
            "pricing_url": pricing_info.get("pricing_url", ""),
            "pricing_source": pricing_info.get("pricing_source", ""),
        })

    return normalized


def _rank_by_score(vendor_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fallback: sort vendor_matches by matchScore without LLM.

    P1-H FIX: Enriches each product with pricing/desc/standards/matched-reqs
    from vendor_analysis matches — mirrors RankingTool._rank_by_score().
    """
    # Build enrichment look-up tables from vendor matches
    pricing_lookup: Dict = {}
    description_lookup: Dict = {}
    standards_lookup: Dict = {}
    matched_reqs_lookup: Dict = {}

    for match in vendor_analysis.get("vendor_matches", []):
        key = (match.get("vendor", ""), match.get("productName", match.get("product_name", "")))
        pricing_lookup[key] = {
            "pricing_url": match.get("pricing_url", ""),
            "pricing_source": match.get("pricing_source", ""),
        }
        description_lookup[key] = match.get("productDescription", match.get("product_description", ""))
        standards_lookup[key] = match.get("standardsCompliance", match.get("standards_compliance", {}))
        matched_reqs_lookup[key] = {
            "matched_requirements": match.get("matchedRequirements", match.get("matched_requirements", {})),
            "unmatched_requirements": match.get("unmatchedRequirements", match.get("unmatched_requirements", [])),
        }

    # Try get_final_ranking() from core.chaining first
    raw_ranked: Optional[List[Dict[str, Any]]] = None
    try:
        from core.chaining import get_final_ranking, to_dict_if_pydantic
        ranking_result = get_final_ranking(vendor_analysis)
        if ranking_result:
            result_dict = to_dict_if_pydantic(ranking_result)
            raw_ranked = result_dict.get("ranked_products", result_dict.get("rankedProducts", []))
    except Exception as exc:
        logger.warning("[rank_products_node] get_final_ranking failed: %s", exc)

    # Ultimate fallback: plain sort of vendor_matches
    if not raw_ranked:
        matches = vendor_analysis.get("vendor_matches", [])
        raw_ranked = sorted(
            matches,
            key=lambda x: x.get("matchScore", x.get("match_score", 0)),
            reverse=True,
        )

    # Enrich each product with pricing/desc/standards/reqs
    enriched: List[Dict[str, Any]] = []
    for product in raw_ranked:
        p_vendor = product.get("vendor", "")
        p_name = product.get("productName", product.get("product_name", ""))
        key = (p_vendor, p_name)

        pricing_info = pricing_lookup.get(key, {})
        product_desc = description_lookup.get(key, "")
        standards_info = standards_lookup.get(key, {})
        reqs_info = matched_reqs_lookup.get(key, {})

        enriched.append({
            **product,  # preserve all existing fields
            # Normalise key field names to camelCase
            "productName": p_name,
            "vendor": p_vendor,
            "modelFamily": product.get("modelFamily", product.get("model_family", "")),
            "overallScore": product.get("overallScore", product.get("overall_score",
                            product.get("matchScore", product.get("match_score", 0)))),
            "matchScore": product.get("matchScore", product.get("match_score", 0)),
            "requirementsMatch": product.get("requirementsMatch", product.get("requirements_match", False)),
            "keyStrengths": _normalize_reasoning(
                product.get("keyStrengths", product.get("key_strengths", []))
            ),
            "concerns": _normalize_reasoning(
                product.get("concerns", product.get("limitations", []))
            ),
            # Enrichment from vendor analysis
            "productDescription": product_desc or product.get("productDescription", ""),
            "standardsCompliance": standards_info or product.get("standardsCompliance", {}),
            "matchedRequirements": reqs_info.get("matched_requirements", {}),
            "unmatchedRequirements": reqs_info.get("unmatched_requirements", []),
            "pricing_url": pricing_info.get("pricing_url", ""),
            "pricing_source": pricing_info.get("pricing_source", ""),
        })

    return enriched


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def rank_products_node(state: "ProductSearchDeepAgentState") -> "ProductSearchDeepAgentState":
    """
    LangGraph node — Step 5: LLM-powered product ranking with score-based fallback.
    """
    logger.info("[rank_products_node] ===== STEP 5: RANK PRODUCTS =====")
    state["current_step"] = "rank_products"

    vendor_analysis = state.get("vendor_analysis_result") or {}
    vendor_matches = state.get("vendor_matches") or vendor_analysis.get("vendor_matches", [])
    session_id = state.get("session_id")

    result: Dict[str, Any] = {"success": False, "session_id": session_id}

    if not vendor_matches:
        logger.warning("[rank_products_node] No vendor matches to rank")
        result.update({
            "success": True,
            "overall_ranking": [],
            "top_product": None,
            "total_ranked": 0,
            "ranking_summary": "No products available to rank",
        })
        state["ranking_result"] = result
        state["overall_ranking"] = []
        state["top_product"] = None
        return state

    logger.info("[rank_products_node] Ranking %d vendor matches", len(vendor_matches))

    try:
        from core.chaining import setup_langchain_components, invoke_ranking_chain, to_dict_if_pydantic

        # -----------------------------------------------------------------
        # PRIMARY: LLM RANKING via invoke_ranking_chain (get_ranking_prompt)
        # -----------------------------------------------------------------
        components = setup_langchain_components()
        format_instructions = components.get("ranking_format_instructions", "")
        vendor_analysis_str = json.dumps(vendor_analysis, indent=2, default=str)

        logger.info("[rank_products_node] Invoking LLM ranking chain...")
        ranking_result_raw = invoke_ranking_chain(components, vendor_analysis_str, format_instructions)

        if ranking_result_raw:
            ranking_dict = to_dict_if_pydantic(ranking_result_raw)
            overall_ranking = _normalize_ranked_products(ranking_dict, vendor_analysis)
            logger.info("[rank_products_node] ✓ LLM ranking produced %d ranked products", len(overall_ranking))
        else:
            logger.warning("[rank_products_node] LLM ranking returned empty — using score fallback")
            overall_ranking = []

    except Exception as exc:
        logger.error("[rank_products_node] LLM ranking failed: %s — using score fallback", exc, exc_info=True)
        overall_ranking = []

    # -----------------------------------------------------------------
    # FALLBACK: Score-based ranking
    # -----------------------------------------------------------------
    if not overall_ranking:
        logger.info("[rank_products_node] Using score-based fallback ranking")
        overall_ranking = _rank_by_score(vendor_analysis)

    # Sort and number
    overall_ranking = sorted(
        overall_ranking,
        key=lambda x: x.get("overallScore", x.get("matchScore", x.get("overall_score", x.get("match_score", 0)))),
        reverse=True,
    )
    for i, product in enumerate(overall_ranking):
        product["rank"] = i + 1

    top_product = overall_ranking[0] if overall_ranking else None

    # Ranking summary
    if top_product:
        top_name = top_product.get("productName", top_product.get("product_name", "Unknown"))
        top_vendor = top_product.get("vendor", "Unknown")
        top_score = top_product.get("overallScore", top_product.get("matchScore", 0))
        ranking_summary = (
            f"Ranked {len(overall_ranking)} products. "
            f"Top recommendation: {top_name} by {top_vendor} with {top_score}% match score"
        )
    else:
        ranking_summary = "No products ranked"

    result.update({
        "success": True,
        "overall_ranking": overall_ranking,
        "top_product": top_product,
        "total_ranked": len(overall_ranking),
        "ranking_summary": ranking_summary,
    })

    logger.info("[rank_products_node] ✓ %s", ranking_summary)

    state["ranking_result"] = result
    state["overall_ranking"] = overall_ranking
    state["top_product"] = top_product
    state["messages"] = state.get("messages", []) + [
        {"role": "system", "content": f"[Step 5] {ranking_summary}"}
    ]

    return state
