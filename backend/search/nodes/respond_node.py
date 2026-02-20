# search/nodes/respond_node.py
# =============================================================================
# RESPOND NODE (Node 7 - Final)
# =============================================================================
#
# Formats the final response for API/UI consumption.
#
# =============================================================================

import logging
from typing import TYPE_CHECKING, Dict, Any, List

from ..state import add_system_message, mark_step_complete

if TYPE_CHECKING:
    from ..state import SearchDeepAgentState

logger = logging.getLogger(__name__)


def respond_node(state: "SearchDeepAgentState") -> "SearchDeepAgentState":
    """
    LangGraph node - Phase 7: Format Response (Final).

    Formats the final response with all analysis results for API/UI consumption.

    Reads: All previous step outputs
    Writes: analysis_result, response_data, success
    """
    logger.info("[respond_node] ===== PHASE 7: FORMAT RESPONSE =====")
    state["current_step"] = "respond"

    try:
        # Build analysis_result (comprehensive output)
        analysis_result = _build_analysis_result(state)
        state["analysis_result"] = analysis_result

        # Build response_data (UI-ready, camelCase)
        response_data = _build_response_data(state)
        state["response_data"] = response_data

        # Set success flag
        state["success"] = not state.get("error")

        # Update workflow instance status (if available)
        _update_workflow_status(state, "completed")

        # Add final message
        total_ranked = len(state.get("overall_ranking", []))
        add_system_message(
            state,
            f"Response formatted: {total_ranked} ranked products",
            "respond",
        )

        mark_step_complete(state, "respond")

        logger.info(
            "[respond_node] Response formatted: success=%s, ranked=%d",
            state.get("success"),
            total_ranked,
        )

    except Exception as exc:
        logger.error("[respond_node] Response formatting failed: %s", exc, exc_info=True)

        state["analysis_result"] = {"error": str(exc)}
        state["response_data"] = {"error": str(exc)}
        state["success"] = False

        add_system_message(state, f"Response formatting failed: {str(exc)}", "respond")
        mark_step_complete(state, "respond")

    return state


def _build_analysis_result(state: "SearchDeepAgentState") -> Dict[str, Any]:
    """Build comprehensive analysis result."""
    overall_ranking = state.get("overall_ranking", [])
    top_product = state.get("top_product")

    # Compute match counts
    exact_matches = sum(
        1 for p in overall_ranking
        if (p.get("overallScore", 0) or p.get("matchScore", 0)) >= 90
    )
    approx_matches = sum(
        1 for p in overall_ranking
        if 70 <= (p.get("overallScore", 0) or p.get("matchScore", 0)) < 90
    )

    return {
        # Product & Schema
        "productType": state.get("product_type", ""),
        "schema": state.get("schema", {}),
        "schemaSource": state.get("schema_source", ""),

        # Validation
        "validationResult": state.get("validation_result", {}),
        "isValid": state.get("is_valid", False),
        "missingFields": state.get("missing_fields", []),
        "optionalFields": state.get("optional_fields", []),

        # Requirements
        "structuredRequirements": state.get("structured_requirements", {}),
        "providedRequirements": state.get("provided_requirements", {}),

        # Advanced Parameters
        "availableAdvancedParams": state.get("available_advanced_params", []),
        "discoveredSpecifications": state.get("discovered_specifications", []),

        # Vendor Analysis
        "vendorAnalysis": {
            "vendorMatches": state.get("vendor_matches", []),
            "totalMatches": len(state.get("vendor_matches", [])),
            "vendorsAnalyzed": state.get("vendors_analyzed", 0),
            "originalVendorCount": state.get("original_vendor_count", 0),
            "filteredVendorCount": state.get("filtered_vendor_count", 0),
            "strategyContext": state.get("strategy_context"),
            "analysisSummary": state.get("vendor_analysis_result", {}).get("analysis_summary", ""),
        },

        # Ranking
        "overallRanking": {"rankedProducts": overall_ranking},
        "topProduct": top_product,
        "topRecommendation": top_product.get("productName") if top_product else None,
        "rankingSummary": state.get("ranking_summary", ""),
        "totalRanked": len(overall_ranking),

        # Match Summary
        "exactMatchCount": exact_matches,
        "approximateMatchCount": approx_matches,
        "readyForVendorSearch": state.get("is_valid", False),

        # Steps & Tracking
        "stepsCompleted": _build_steps_completed(state),

        # Session Info
        "sessionId": state.get("session_id", ""),
        "workflowThreadId": state.get("workflow_thread_id", ""),
        "instanceId": state.get("instance_id", ""),

        # Strategy
        "strategy": state.get("strategy", "full"),
        "executionPlan": state.get("execution_plan", {}),

        # Errors
        "error": state.get("error"),
    }


def _build_response_data(state: "SearchDeepAgentState") -> Dict[str, Any]:
    """Build UI-ready response data (camelCase)."""
    overall_ranking = state.get("overall_ranking", [])
    top_product = state.get("top_product")

    return {
        "success": not state.get("error"),
        "productType": state.get("product_type", ""),
        "rankedProducts": overall_ranking,
        "topRecommendation": top_product,
        "totalRanked": len(overall_ranking),
        "vendorMatches": state.get("vendor_matches", []),
        "validationResult": {
            "isValid": state.get("is_valid", False),
            "missingFields": state.get("missing_fields", []),
            "providedRequirements": state.get("provided_requirements", {}),
        },
        "schema": state.get("schema", {}),
        "schemaSource": state.get("schema_source", ""),
        "availableAdvancedParams": state.get("available_advanced_params", []),
        "stepsCompleted": state.get("steps_completed", []),
        "sessionId": state.get("session_id", ""),
        "workflowThreadId": state.get("workflow_thread_id", ""),
        "error": state.get("error"),
    }


def _build_steps_completed(state: "SearchDeepAgentState") -> List[str]:
    """Build list of completed steps from state."""
    steps = state.get("steps_completed", [])

    if not steps:
        # Reconstruct from messages
        messages = state.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            if "[plan]" in content.lower():
                steps.append("planning")
            elif "[validate]" in content.lower():
                steps.append("validation")
            elif "[collect_requirements]" in content.lower():
                steps.append("requirements")
            elif "[discover_params]" in content.lower():
                steps.append("advanced_params")
            elif "[analyze_vendors]" in content.lower():
                steps.append("vendor_analysis")
            elif "[rank]" in content.lower():
                steps.append("ranking")
            elif "[respond]" in content.lower():
                steps.append("response")

    return list(dict.fromkeys(steps))  # Remove duplicates, preserve order


def _update_workflow_status(state: "SearchDeepAgentState", status: str) -> None:
    """Update workflow instance status if manager is available."""
    try:
        from common.infrastructure.state.workflow_instance_manager import (
            WorkflowInstanceManager,
        )

        instance_id = state.get("instance_id")
        if instance_id:
            WorkflowInstanceManager.update_status(instance_id, status)
            logger.debug("[respond_node] Updated workflow status to: %s", status)

    except ImportError:
        pass  # Manager not available
    except Exception as exc:
        logger.warning("[respond_node] Failed to update workflow status: %s", exc)
