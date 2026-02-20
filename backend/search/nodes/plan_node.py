# search/nodes/plan_node.py
# =============================================================================
# PLAN NODE (Node 1)
# =============================================================================
#
# Creates an execution plan based on query analysis.
# Determines strategy (FAST/FULL/DEEP) and phases to run.
#
# =============================================================================

import logging
from typing import TYPE_CHECKING

from ..planner import SearchPlanner
from ..state import add_system_message, mark_step_complete

if TYPE_CHECKING:
    from ..state import SearchDeepAgentState

logger = logging.getLogger(__name__)


def plan_node(state: "SearchDeepAgentState") -> "SearchDeepAgentState":
    """
    LangGraph node - Phase 1: Planning.

    Analyzes the user input and creates an execution plan that
    determines which phases to run and with what configuration.

    Reads: user_input, expected_product_type, skip_advanced_params
    Writes: execution_plan, strategy, quality_thresholds, phases_to_run
    """
    logger.info("[plan_node] ===== PHASE 1: PLANNING =====")
    state["current_step"] = "plan"

    try:
        # Get inputs
        user_input = state.get("user_input", "")
        expected_product_type = state.get("expected_product_type")
        skip_advanced_params = state.get("skip_advanced_params")

        # Create planner and generate plan
        planner = SearchPlanner()
        plan = planner.plan(
            user_input=user_input,
            expected_product_type=expected_product_type,
            skip_advanced_params=skip_advanced_params,
        )

        # Update state with plan
        state["execution_plan"] = plan.to_dict()
        state["strategy"] = plan.strategy.value
        state["quality_thresholds"] = plan.quality_thresholds
        state["phases_to_run"] = plan.phases_to_run
        state["has_safety_requirements"] = plan.has_safety_requirements
        state["product_category"] = plan.product_category

        # Override skip_advanced_params from plan if not explicitly set
        if skip_advanced_params is None:
            state["skip_advanced_params"] = plan.skip_advanced_params

        # Add system message
        add_system_message(
            state,
            f"Strategy: {plan.strategy.value} | Phases: {len(plan.phases_to_run)} | "
            f"Safety: {plan.has_safety_requirements} | Confidence: {plan.confidence:.0%}",
            "plan",
        )

        mark_step_complete(state, "plan")

        logger.info(
            "[plan_node] Plan created: strategy=%s, phases=%d",
            plan.strategy.value,
            len(plan.phases_to_run),
        )

    except Exception as exc:
        logger.error("[plan_node] Planning failed: %s", exc, exc_info=True)
        state["error"] = f"Planning failed: {str(exc)}"
        state["error_type"] = "PlanningError"

        # Set default plan on error
        state["execution_plan"] = {}
        state["strategy"] = "full"
        state["quality_thresholds"] = {}
        state["phases_to_run"] = ["plan", "validate", "collect_requirements", "discover_params", "analyze_vendors", "rank", "respond"]

        add_system_message(state, f"Planning failed, using defaults: {str(exc)}", "plan")

    return state
