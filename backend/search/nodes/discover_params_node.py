# search/nodes/discover_params_node.py
# =============================================================================
# DISCOVER PARAMS NODE (Node 4)
# =============================================================================
#
# Discovers vendor-specific advanced parameters not in the base schema.
#
# =============================================================================

import logging
from typing import TYPE_CHECKING

from ..agents import ParamsAgent
from ..state import add_system_message, mark_step_complete

if TYPE_CHECKING:
    from ..state import SearchDeepAgentState

logger = logging.getLogger(__name__)


def discover_params_node(state: "SearchDeepAgentState") -> "SearchDeepAgentState":
    """
    LangGraph node - Phase 4: Discover Advanced Parameters.

    Queries vendor data to identify specifications beyond the base schema
    that could be relevant for matching.

    Reads: product_type, schema, session_id, skip_advanced_params
    Writes: available_advanced_params, discovered_specifications, advanced_params_result
    """
    logger.info("[discover_params_node] ===== PHASE 4: DISCOVER ADVANCED PARAMS =====")
    state["current_step"] = "discover_params"

    try:
        # Check if we should skip this phase
        if state.get("skip_advanced_params", False):
            logger.info("[discover_params_node] Skipping advanced params (skip_advanced_params=True)")

            state["advanced_params_result"] = {"skipped": True}
            state["available_advanced_params"] = []
            state["discovered_specifications"] = []

            add_system_message(state, "Advanced params discovery skipped", "discover_params")
            mark_step_complete(state, "discover_params")
            return state

        # Get inputs
        product_type = state.get("product_type", "")
        session_id = state.get("session_id")
        schema = state.get("schema", {})

        if not product_type:
            logger.warning("[discover_params_node] No product type, skipping discovery")

            state["advanced_params_result"] = {"error": "No product type"}
            state["available_advanced_params"] = []
            state["discovered_specifications"] = []

            add_system_message(state, "No product type, skipping discovery", "discover_params")
            mark_step_complete(state, "discover_params")
            return state

        # Run params discovery agent
        agent = ParamsAgent()
        result = agent.discover(
            product_type=product_type,
            session_id=session_id,
            existing_schema=schema,
        )

        # Update state
        state["advanced_params_result"] = result.to_dict()
        state["available_advanced_params"] = result.unique_specifications
        state["discovered_specifications"] = result.unique_specifications

        # Add system message
        add_system_message(
            state,
            f"Discovered {result.total_unique_specifications} unique specs from {len(result.vendors_searched)} vendors",
            "discover_params",
        )

        mark_step_complete(state, "discover_params")

        logger.info(
            "[discover_params_node] Discovery complete: %d unique specs",
            result.total_unique_specifications,
        )

    except Exception as exc:
        logger.error("[discover_params_node] Discovery failed: %s", exc, exc_info=True)

        state["advanced_params_result"] = {"error": str(exc)}
        state["available_advanced_params"] = []
        state["discovered_specifications"] = []

        add_system_message(state, f"Discovery failed: {str(exc)}", "discover_params")
        mark_step_complete(state, "discover_params")

    return state
