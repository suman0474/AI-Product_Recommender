"""
Node: format_response_node
============================

Final node of the Product Search Deep Agent.

Responsibilities
----------------
- Assembles the `analysis_result` dict that the frontend reads
  (same structure as the current workflow.py run_analysis_only() output).
- Assembles `response_data` for API consumers.
- Marks the workflow instance as "completed" via WorkflowInstanceManager.
- Marks state["success"] = True.

P1-I fixes applied
-------------------
  - discoveredSpecifications  → from state["available_advanced_params"]
  - exactMatchCount           → count of products where requirementsMatch is True
  - approximateMatchCount     → count of products where requirementsMatch is False
  - steps_completed           → list built from which nodes recorded messages
  - ready_for_vendor_search   → True when there is at least one ranked product
  - available_advanced_params → included in analysis_result and response_data

Reads from state:
  product_type, schema, validation_result, vendor_matches, vendor_analysis_result,
  overall_ranking, top_product, ranking_result, structured_requirements,
  available_advanced_params, session_id, instance_id, workflow_thread_id,
  messages

Writes to state:
  analysis_result, response_data, success, current_step
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from common.agentic.models import ProductSearchDeepAgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STEP_TAG_MAP: Dict[str, str] = {
    "[Step 1]": "validation",
    "[Step 2]": "advanced_params",
    "[Step 3]": "collect_requirements",
    "[Step 4]": "vendor_analysis",
    "[Step 5]": "ranking",
    "[Final]": "format_response",
}


def _build_steps_completed(messages: List[Dict[str, Any]]) -> List[str]:
    """
    Reconstruct the steps_completed list from the system messages left by each node.
    Mirrors the original workflow.py which manually appended step names.
    """
    completed = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            for tag, step_name in _STEP_TAG_MAP.items():
                if content.startswith(tag) and step_name not in completed:
                    completed.append(step_name)
    return completed


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def format_response_node(state: "ProductSearchDeepAgentState") -> "ProductSearchDeepAgentState":
    """
    LangGraph node — Final: Assemble UI-ready analysis_result and mark instance completed.
    """
    logger.info("[format_response_node] ===== FINAL: FORMAT RESPONSE =====")
    state["current_step"] = "format_response"

    vendor_analysis = state.get("vendor_analysis_result") or {}
    ranking_result = state.get("ranking_result") or {}
    vendor_matches = state.get("vendor_matches") or vendor_analysis.get("vendor_matches", [])
    overall_ranking = state.get("overall_ranking") or ranking_result.get("overall_ranking", [])
    top_product = state.get("top_product") or ranking_result.get("top_product")
    product_type = state.get("product_type", "")
    schema = state.get("schema") or {}
    validation_result = state.get("validation_result") or {}
    structured_requirements = state.get("structured_requirements") or {}
    strategy_context = state.get("strategy_context") or {}
    available_advanced_params = state.get("available_advanced_params") or []   # P1-I
    discovered_specifications = state.get("discovered_specifications") or available_advanced_params  # P1-I
    session_id = state.get("session_id")
    instance_id = state.get("instance_id")
    workflow_thread_id = state.get("workflow_thread_id")
    messages = state.get("messages") or []

    # =========================================================================
    # COMPUTED COUNTS  [P1-I]
    # =========================================================================
    exact_match_count = sum(1 for p in overall_ranking if p.get("requirementsMatch"))
    approx_match_count = sum(1 for p in overall_ranking if not p.get("requirementsMatch"))
    ready_for_vendor_search = len(overall_ranking) > 0  # P1-I
    steps_completed = _build_steps_completed(messages)  # P1-I

    # =========================================================================
    # BUILD analysis_result (matches current run_analysis_only() output shape)
    # =========================================================================
    analysis_result: Dict[str, Any] = {
        "productType": product_type,
        "schema": schema,
        "validationResult": validation_result,
        "structuredRequirements": structured_requirements,
        "vendorAnalysis": {
            "vendorMatches": vendor_matches,
            "vendorRunDetails": vendor_analysis.get("vendor_run_details", []),
            "totalMatches": len(vendor_matches),
            "vendorsAnalyzed": vendor_analysis.get("vendors_analyzed", 0),
            "analysisSummary": vendor_analysis.get("analysis_summary", ""),
            "strategyContext": strategy_context,
            # P1-I: expose original / filtered counts
            "originalVendorCount": vendor_analysis.get("original_vendor_count", 0),
            "filteredVendorCount": vendor_analysis.get("filtered_vendor_count", 0),
            "excludedByStrategy": vendor_analysis.get("excluded_by_strategy", 0),
        },
        "overallRanking": {
            "rankedProducts": overall_ranking,
        },
        "topProduct": top_product,
        "topRecommendation": top_product,                          # backward compat alias
        "rankingSummary": ranking_result.get("ranking_summary", ""),
        "totalRanked": len(overall_ranking),
        # P1-I: missing fields that frontend / API callers expect
        "exactMatchCount": exact_match_count,
        "approximateMatchCount": approx_match_count,
        "ready_for_vendor_search": ready_for_vendor_search,
        "steps_completed": steps_completed,
        "available_advanced_params": available_advanced_params,
        "discoveredSpecifications": discovered_specifications,
        "advancedParameters": {
            "available_specs": available_advanced_params,
        },
        # IDs
        "sessionId": session_id,
        "workflowThreadId": workflow_thread_id,
        "instanceId": instance_id,
    }

    # =========================================================================
    # BUILD response_data (mimics Flask API response shape)
    # =========================================================================
    response_data: Dict[str, Any] = {
        "success": True,
        "productType": product_type,
        "schema": schema,
        "analysisResult": analysis_result,
        "vendorMatches": vendor_matches,
        "overallRanking": overall_ranking,
        "topProduct": top_product,
        # P1-I extras
        "exactMatchCount": exact_match_count,
        "approximateMatchCount": approx_match_count,
        "ready_for_vendor_search": ready_for_vendor_search,
        "steps_completed": steps_completed,
        "available_advanced_params": available_advanced_params,
        "discoveredSpecifications": discovered_specifications,
    }

    state["analysis_result"] = analysis_result
    state["response_data"] = response_data
    state["success"] = True

    # =========================================================================
    # UPDATE INSTANCE STATUS → "completed"
    # =========================================================================
    if instance_id:
        try:
            from common.infrastructure.state.execution.instance_manager import WorkflowInstanceManager
            WorkflowInstanceManager.update_status(instance_id, "completed")
            logger.info("[format_response_node] ✓ Instance '%s' marked as completed", instance_id)
        except Exception as exc:
            logger.warning("[format_response_node] Could not update instance status: %s", exc)

    final_msg = (
        f"[Final] Workflow complete — {len(overall_ranking)} products ranked "
        f"({exact_match_count} exact, {approx_match_count} approx). "
        f"Top: {top_product.get('productName', 'N/A') if top_product else 'N/A'}. "
        f"Steps: {steps_completed}"
    )
    state["messages"] = messages + [{"role": "system", "content": final_msg}]

    logger.info(
        "[format_response_node] ✓ Response assembled — %d ranked (%d exact, %d approx), top=%s",
        len(overall_ranking), exact_match_count, approx_match_count,
        top_product.get("productName", "N/A") if top_product else "N/A",
    )

    return state
