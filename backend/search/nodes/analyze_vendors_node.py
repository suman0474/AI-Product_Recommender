# search/nodes/analyze_vendors_node.py
# =============================================================================
# ANALYZE VENDORS NODE (Node 5)
# =============================================================================
#
# Runs vendor analysis with Strategy RAG filtering and parallel processing.
#
# =============================================================================

import logging
from typing import TYPE_CHECKING

from ..agents import VendorAgent
from ..state import add_system_message, mark_step_complete, set_error

if TYPE_CHECKING:
    from ..state import SearchDeepAgentState

logger = logging.getLogger(__name__)


def analyze_vendors_node(state: "SearchDeepAgentState") -> "SearchDeepAgentState":
    """
    LangGraph node - Phase 5: Analyze Vendors.

    Runs vendor analysis with Strategy RAG filtering and parallel processing.

    Reads: structured_requirements, product_type, schema, max_vendor_workers
    Writes: vendor_analysis_result, vendor_matches, strategy_context
    """
    logger.info("[analyze_vendors_node] ===== PHASE 5: ANALYZE VENDORS =====")
    state["current_step"] = "analyze_vendors"

    try:
        # Get inputs
        requirements = state.get("structured_requirements", {})
        product_type = state.get("product_type", "")
        schema = state.get("schema", {})
        max_workers = state.get("max_vendor_workers", 10)
        session_id = state.get("session_id")

        if not product_type:
            set_error(state, "No product type for vendor analysis", "VendorAnalysisError")
            add_system_message(state, "No product type, cannot analyze vendors", "analyze_vendors")
            return state

        # Run vendor analysis agent
        agent = VendorAgent(max_workers=max_workers)
        result = agent.analyze(
            requirements=requirements,
            product_type=product_type,
            schema=schema,
            session_id=session_id,
            use_cache=True,
        )

        # Check for errors
        if not result.success:
            set_error(state, result.error or "Vendor analysis failed", "VendorAnalysisError")
            add_system_message(state, f"Vendor analysis failed: {result.error}", "analyze_vendors")
            # Still update state with partial results
            state["vendor_analysis_result"] = result.to_dict()
            state["vendor_matches"] = []
            return state

        # Update state with results
        state["vendor_analysis_result"] = result.to_dict()
        state["vendor_matches"] = result.vendor_matches
        state["strategy_context"] = result.strategy_context
        state["vendors_analyzed"] = result.vendors_analyzed
        state["original_vendor_count"] = result.original_vendor_count
        state["filtered_vendor_count"] = result.filtered_vendor_count
        state["excluded_by_strategy"] = result.excluded_by_strategy

        # Add system message
        cache_note = " (cached)" if result.cached else ""
        add_system_message(
            state,
            f"Analyzed {result.vendors_analyzed} vendors, found {result.total_matches} matches{cache_note}",
            "analyze_vendors",
        )

        mark_step_complete(state, "analyze_vendors")

        logger.info(
            "[analyze_vendors_node] Analysis complete: %d matches from %d vendors",
            result.total_matches,
            result.vendors_analyzed,
        )

    except Exception as exc:
        logger.error("[analyze_vendors_node] Analysis failed: %s", exc, exc_info=True)
        set_error(state, f"Vendor analysis failed: {str(exc)}", "VendorAnalysisError")
        add_system_message(state, f"Analysis error: {str(exc)}", "analyze_vendors")

        state["vendor_analysis_result"] = {"error": str(exc)}
        state["vendor_matches"] = []

    return state
