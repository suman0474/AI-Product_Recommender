# search/nodes/rank_node.py
# =============================================================================
# RANK NODE (Node 6)
# =============================================================================
#
# Ranks products using LLM or score-based fallback.
#
# =============================================================================

import logging
from typing import TYPE_CHECKING

from ..agents import RankingAgent
from ..state import add_system_message, mark_step_complete

if TYPE_CHECKING:
    from ..state import SearchDeepAgentState

logger = logging.getLogger(__name__)


def rank_node(state: "SearchDeepAgentState") -> "SearchDeepAgentState":
    """
    LangGraph node - Phase 6: Rank Products.

    Ranks products using LLM or score-based fallback.

    Reads: vendor_analysis_result, vendor_matches, structured_requirements
    Writes: ranking_result, overall_ranking, top_product, ranking_summary
    """
    logger.info("[rank_node] ===== PHASE 6: RANK PRODUCTS =====")
    state["current_step"] = "rank"

    try:
        # Get inputs
        vendor_analysis = state.get("vendor_analysis_result", {})
        vendor_matches = state.get("vendor_matches", [])
        requirements = state.get("structured_requirements", {})
        session_id = state.get("session_id")

        # Check if there are matches to rank
        if not vendor_matches:
            logger.info("[rank_node] No vendor matches to rank")

            state["ranking_result"] = {"success": True, "total_ranked": 0}
            state["overall_ranking"] = []
            state["top_product"] = None
            state["ranking_summary"] = "No products to rank"

            add_system_message(state, "No products to rank", "rank")
            mark_step_complete(state, "rank")
            return state

        # Run ranking agent
        agent = RankingAgent()
        result = agent.rank(
            vendor_analysis=vendor_analysis,
            requirements=requirements,
            use_llm=True,
            session_id=session_id,
        )

        # Update state with results
        state["ranking_result"] = result.to_dict()
        state["overall_ranking"] = result.overall_ranking
        state["top_product"] = result.top_product
        state["ranking_summary"] = result.ranking_summary

        # Add system message
        add_system_message(
            state,
            f"Ranked {result.total_ranked} products using {result.ranking_method}",
            "rank",
        )

        mark_step_complete(state, "rank")

        logger.info(
            "[rank_node] Ranking complete: %d products, method=%s",
            result.total_ranked,
            result.ranking_method,
        )

    except Exception as exc:
        logger.error("[rank_node] Ranking failed: %s", exc, exc_info=True)

        # Fallback: Use vendor matches as-is, sorted by score
        vendor_matches = state.get("vendor_matches", [])
        sorted_matches = sorted(
            vendor_matches,
            key=lambda x: x.get("overallScore") or x.get("matchScore", 0),
            reverse=True,
        )

        # Add rank numbers
        for i, match in enumerate(sorted_matches):
            match["rank"] = i + 1

        state["ranking_result"] = {"error": str(exc), "fallback": True}
        state["overall_ranking"] = sorted_matches
        state["top_product"] = sorted_matches[0] if sorted_matches else None
        state["ranking_summary"] = f"Ranked {len(sorted_matches)} products (fallback)"

        add_system_message(state, f"Ranking fallback: {str(exc)}", "rank")
        mark_step_complete(state, "rank")

    return state
