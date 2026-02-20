"""
Indexing — Graph Definition
============================
LangGraph StateGraph with conditional edges, checkpointing, and runner helpers.
"""

import logging
import time
import json
from typing import Dict, Any, Optional, Generator
from pathlib import Path

from langgraph.graph import StateGraph, END, START

from .state import IndexingState
from .nodes import (
    planning_node,
    discovery_node,
    search_node,
    extraction_node,
    schema_retrieval_node,
    quality_assurance_node,
)
from . import config
from .monitoring import get_monitor as get_workflow_monitor, get_collector as get_metrics_collector

logger = logging.getLogger(__name__)


# ── Conditional edge functions ──────────────────────────────────────────────

def _should_extract_or_fallback(state: IndexingState) -> str:
    """After search: extract if PDFs were downloaded, else skip to schema."""
    pdfs = state.get("pdf_results", [])
    downloaded = [p for p in pdfs if p.get("download_status") == "success"]
    if downloaded:
        return "extraction"
    logger.warning("No PDFs downloaded — skipping extraction")
    return "schema_retrieval"


def _should_refine_or_finish(state: IndexingState) -> str:
    """After QA: loop back if quality is low and we haven't refined too many times."""
    qa = state.get("qa_assessment", {})
    quality = qa.get("overall_quality_score", 0.0)
    count = state.get("refinement_count", 0)

    if quality < config.QUALITY_THRESHOLD_REFINEMENT and count < config.MAX_REFINEMENT_LOOPS:
        logger.info(f"Quality {quality:.2f} < {config.QUALITY_THRESHOLD_REFINEMENT} — refining (attempt {count + 1})")
        return "schema_retrieval"
    return "__end__"


# ── Graph builder ───────────────────────────────────────────────────────────

def build_indexing_graph() -> StateGraph:
    """
    Build the Indexing workflow graph.

    Nodes:
        planning → discovery → search →? extraction → schema_retrieval → quality_assurance →? END

    Conditional edges:
        1. search → extraction | schema_retrieval  (PDF availability)
        2. quality_assurance → schema_retrieval | END  (quality threshold)
    """
    graph = StateGraph(IndexingState)

    graph.add_node("planning", planning_node)
    graph.add_node("discovery", discovery_node)
    graph.add_node("search", search_node)
    graph.add_node("extraction", extraction_node)
    graph.add_node("schema_retrieval", schema_retrieval_node)
    graph.add_node("quality_assurance", quality_assurance_node)

    graph.add_edge(START, "planning")
    graph.add_edge("planning", "discovery")
    graph.add_edge("discovery", "search")

    graph.add_conditional_edges(
        "search",
        _should_extract_or_fallback,
        {"extraction": "extraction", "schema_retrieval": "schema_retrieval"},
    )

    graph.add_edge("extraction", "schema_retrieval")
    graph.add_edge("schema_retrieval", "quality_assurance")

    graph.add_conditional_edges(
        "quality_assurance",
        _should_refine_or_finish,
        {"schema_retrieval": "schema_retrieval", "__end__": END},
    )

    return graph


def create_indexing_workflow(checkpointer=None):
    """
    Create a compiled Indexing workflow graph.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. ``MemorySaver``).
                       When provided, enables state persistence and resumption.

    Returns:
        Compiled LangGraph ``CompiledGraph``.
    """
    graph = build_indexing_graph()

    compile_kwargs: Dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return graph.compile(**compile_kwargs)


# ── Runner helpers ──────────────────────────────────────────────────────────

def run_indexing_workflow(
    product_type: str,
    context: Optional[Dict[str, Any]] = None,
    checkpointer=None,
    session_id: str = "",
) -> Dict[str, Any]:
    """
    Execute the full Indexing workflow from start to finish.

    Args:
        product_type: Product type to index (e.g. "pressure transmitter").
        context: Optional context dict for the workflow.
        checkpointer: Optional checkpointer for state persistence.
        session_id: Optional session ID for tracking and caching.

    Returns:
        Final workflow state dict.
    """
    monitor = get_workflow_monitor()
    monitor.start_run(product_type)

    app = create_indexing_workflow(checkpointer)

    initial_state: Dict[str, Any] = {
        "product_type": product_type,
        "context": context or {},
        "session_id": session_id,
        "errors": [],
        "agent_outputs": {},
        "refinement_count": 0,
    }

    try:
        final_state = app.invoke(initial_state)
        report = monitor.end_run(final_state)
        get_metrics_collector().record_run(report)
        return final_state

    except Exception as e:
        logger.error(f"Indexing workflow failed: {e}")
        return {
            **initial_state,
            "errors": [str(e)],
            "current_stage": "failed",
        }


def run_indexing_workflow_streaming(
    product_type: str,
    context: Optional[Dict[str, Any]] = None,
    checkpointer=None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream Indexing workflow events as they happen.

    Yields dicts with at minimum ``{"node": ..., "state_update": ...}``.
    """
    app = create_indexing_workflow(checkpointer)

    initial_state: Dict[str, Any] = {
        "product_type": product_type,
        "context": context or {},
        "errors": [],
        "agent_outputs": {},
        "refinement_count": 0,
    }

    for event in app.stream(initial_state):
        for node_name, state_update in event.items():
            yield {
                "node": node_name,
                "state_update": state_update,
                "stage": state_update.get("current_stage", "unknown"),
            }


def run_indexing_workflow_legacy_compatible(product_type: str) -> Dict[str, Any]:
    """Backward-compatible wrapper matching the old API signature."""
    final_state = run_indexing_workflow(product_type)

    return {
        "success": "failed" not in (final_state.get("current_stage", "")),
        "product_type": product_type,
        "schema": final_state.get("generated_schema", {}),
        "quality_score": final_state.get("final_quality_score", 0.0),
        "deployment_ready": final_state.get("deployment_ready", False),
    }


def run_potential_product_indexing_workflow(
    product_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Public entry point used by the rest of the codebase.

    Delegates to ``run_indexing_workflow`` with added timing and error handling.
    """
    logger.info(f"Starting Indexing workflow for: {product_type}")
    start = time.time()

    try:
        final_state = run_indexing_workflow(product_type, context=context)
        duration = time.time() - start

        return {
            "success": final_state.get("current_stage", "") != "failed",
            "product_type": product_type,
            "schema": final_state.get("generated_schema", {}),
            "quality_score": final_state.get("final_quality_score", 0.0),
            "deployment_ready": final_state.get("deployment_ready", False),
            "qa_assessment": final_state.get("qa_assessment", {}),
            "agents_completed": list(final_state.get("agent_outputs", {}).keys()),
            "execution_time_seconds": duration,
            "errors": final_state.get("errors", []),
        }

    except Exception as e:
        duration = time.time() - start
        logger.error(f"Indexing workflow failed after {duration:.1f}s: {e}")
        return {
            "success": False,
            "product_type": product_type,
            "error": str(e),
            "execution_time_seconds": duration,
        }
