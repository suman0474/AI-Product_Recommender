# search/workflow.py
# =============================================================================
# SEARCH DEEP AGENT LANGGRAPH WORKFLOW
# =============================================================================
#
# Defines the LangGraph StateGraph for the Search Deep Agent workflow.
#
# Graph Topology:
#     START → plan → validate → collect_requirements (HITL)
#     → discover_params → analyze_vendors → rank → respond → END
#
# Conditional Paths:
#     - plan: error → END
#     - validate: error → END
#     - collect_requirements: HITL interrupted → END (pause for user)
#     - rank: retry_relaxed → analyze_vendors (loop)
#
# =============================================================================

import logging
from typing import Any, Dict, Literal

from langgraph.graph import StateGraph, END

from .state import SearchDeepAgentState
from .nodes import (
    plan_node,
    validate_node,
    collect_requirements_node,
    discover_params_node,
    analyze_vendors_node,
    rank_node,
    respond_node,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def _route_after_plan(state: SearchDeepAgentState) -> Literal["validate", "__end__"]:
    """
    Route after plan node.

    Returns:
        - "validate": Continue to validation
        - END: Stop on error
    """
    if state.get("error"):
        logger.info("[workflow] Plan error, ending workflow")
        return END

    return "validate"


def _route_after_validate(state: SearchDeepAgentState) -> Literal["collect_requirements", "__end__"]:
    """
    Route after validate node.

    Returns:
        - "collect_requirements": Continue to HITL requirements collection
        - END: Stop on error
    """
    if state.get("error"):
        logger.info("[workflow] Validation error, ending workflow")
        return END

    # Check if we have a product type
    if not state.get("product_type"):
        logger.info("[workflow] No product type detected, ending workflow")
        return END

    return "collect_requirements"


def _route_after_collect(state: SearchDeepAgentState) -> Literal["discover_params", "__end__"]:
    """
    Route after collect_requirements node.

    Returns:
        - "discover_params": Continue to advanced params discovery
        - END: Pause for HITL user input
    """
    if state.get("workflow_interrupted"):
        logger.info("[workflow] HITL interrupted, pausing workflow")
        return END

    if state.get("error"):
        logger.info("[workflow] Requirements error, ending workflow")
        return END

    return "discover_params"


def _route_after_analyze(state: SearchDeepAgentState) -> Literal["rank", "__end__"]:
    """
    Route after analyze_vendors node.

    Returns:
        - "rank": Continue to ranking
        - END: Stop on error or no matches
    """
    if state.get("error"):
        logger.info("[workflow] Vendor analysis error, ending workflow")
        return END

    return "rank"


def _route_after_rank(
    state: SearchDeepAgentState
) -> Literal["respond", "analyze_vendors", "__end__"]:
    """
    Route after rank node.

    Returns:
        - "respond": Continue to final response
        - "analyze_vendors": Retry with relaxed thresholds
        - END: Stop on error
    """
    if state.get("error"):
        logger.info("[workflow] Ranking error, ending workflow")
        return END

    # Check if we need to retry with relaxed thresholds
    overall_ranking = state.get("overall_ranking", [])
    quality_thresholds = state.get("quality_thresholds", {})
    retry_count = state.get("retry_count", 0)

    if not overall_ranking and retry_count < 1 and not state.get("relaxed_mode"):
        # No matches and haven't retried - trigger relaxed mode
        logger.info("[workflow] No matches, triggering relaxed mode retry")
        state["relaxed_mode"] = True
        state["retry_count"] = retry_count + 1
        state["retry_history"] = state.get("retry_history", []) + [{
            "reason": "no_matches",
            "retry_number": retry_count + 1,
        }]
        return "analyze_vendors"

    return "respond"


# =============================================================================
# GRAPH FACTORY
# =============================================================================

def create_search_workflow_graph(
    checkpointing_backend: str = "memory",
) -> Any:
    """
    Build and compile the Search Deep Agent StateGraph.

    Args:
        checkpointing_backend: Backend for LangGraph checkpointing
            ("memory", "sqlite", "mongodb", "azure")

    Returns:
        Compiled LangGraph app
    """
    logger.info("[workflow] Creating Search Deep Agent graph")

    graph = StateGraph(SearchDeepAgentState)

    # =========================================================================
    # ADD NODES (7 total)
    # =========================================================================
    graph.add_node("plan", plan_node)
    graph.add_node("validate", validate_node)
    graph.add_node("collect_requirements", collect_requirements_node)
    graph.add_node("discover_params", discover_params_node)
    graph.add_node("analyze_vendors", analyze_vendors_node)
    graph.add_node("rank", rank_node)
    graph.add_node("respond", respond_node)

    # =========================================================================
    # SET ENTRY POINT
    # =========================================================================
    graph.set_entry_point("plan")

    # =========================================================================
    # ADD EDGES WITH CONDITIONAL ROUTING
    # =========================================================================

    # Plan → Validate (or END on error)
    graph.add_conditional_edges(
        "plan",
        _route_after_plan,
        {
            "validate": "validate",
            END: END,
        },
    )

    # Validate → Collect Requirements (or END on error)
    graph.add_conditional_edges(
        "validate",
        _route_after_validate,
        {
            "collect_requirements": "collect_requirements",
            END: END,
        },
    )

    # Collect Requirements → Discover Params (or END for HITL)
    graph.add_conditional_edges(
        "collect_requirements",
        _route_after_collect,
        {
            "discover_params": "discover_params",
            END: END,
        },
    )

    # Discover Params → Analyze Vendors (always)
    graph.add_edge("discover_params", "analyze_vendors")

    # Analyze Vendors → Rank (or END on error)
    graph.add_conditional_edges(
        "analyze_vendors",
        _route_after_analyze,
        {
            "rank": "rank",
            END: END,
        },
    )

    # Rank → Respond (or retry to Analyze Vendors)
    graph.add_conditional_edges(
        "rank",
        _route_after_rank,
        {
            "respond": "respond",
            "analyze_vendors": "analyze_vendors",
            END: END,
        },
    )

    # Respond → END (always)
    graph.add_edge("respond", END)

    # =========================================================================
    # COMPILE WITH CHECKPOINTING
    # =========================================================================
    try:
        from common.infrastructure.state.checkpointing.local import compile_with_checkpointing

        compiled = compile_with_checkpointing(graph, checkpointing_backend)
        logger.info("[workflow] Graph compiled with checkpointing backend: %s", checkpointing_backend)
        return compiled

    except ImportError:
        logger.warning("[workflow] Checkpointing not available, compiling without")
        compiled = graph.compile()
        return compiled


# =============================================================================
# CONVENIENCE ALIAS
# =============================================================================

# Backward compatibility alias
create_product_search_graph = create_search_workflow_graph
