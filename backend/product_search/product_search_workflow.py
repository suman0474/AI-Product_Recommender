"""
Product Search Deep Agent — LangGraph StateGraph
==================================================

Defines the StateGraph and all public entry-point functions.

Graph topology
--------------
START
  ↓
validate_product               ← Step 1 (product type, schema, standards enrichment)
  ↓  (error → END)
discover_advanced_params       ← Step 2 (discover vendor specs)
  ↓
collect_requirements           ← Step 3 (HITL or auto)
  ↓  (interrupt if awaiting_user_input=True)
analyze_vendors                ← Step 4 (Strategy RAG + parallel LLM chains)
  ↓
rank_products                  ← Step 5 (LLM ranking + fallback)
  ↓
format_response                ← Final (assemble analysis_result)
  ↓
END

Public API
----------
run_product_search_workflow()        — full pipeline, main entry point
resume_product_search_workflow()     — resume after HITL interrupt
run_single_product_workflow()        — auto-mode alias (backward compat)
run_analysis_only()                  — skip to step 4+ with given requirements
run_validation_only()                — run only validate_product node
run_advanced_params_only()           — run only discover_advanced_params node  [P2-A]
process_from_instrument_identifier() — batch from instrument identifier
process_from_solution_workflow()     — batch from solution workflow
product_search_workflow()            — top-level convenience
create_product_search_graph()        — expose compiled graph for testing
generate_comparison_table()          — build UI comparison table from ranking  [P2-B]
get_schema_only()                    — load/generate schema without validation  [P2-C]
validate_with_schema()               — validate user input against provided schema  [P2-C]
validate_multiple_products_parallel() — parallel schema gen for N products  [P3-C]
enrich_schema_parallel()             — parallel standards RAG enrichment  [P3-C]
get_or_generate_schema_async()       — async complete schema lifecycle  [P3-D]
get_or_generate_schemas_batch_async() — async batch schema lifecycle  [P3-D]
"""

import asyncio
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt

from common.agentic.models import (
    ProductSearchDeepAgentState,
    create_product_search_deep_agent_state,
)
from common.infrastructure.state.checkpointing.local import compile_with_checkpointing

from product_search.nodes.validate_product_node import validate_product_node
from product_search.nodes.discover_advanced_params_node import discover_advanced_params_node
from product_search.nodes.collect_requirements_node import collect_requirements_node
from product_search.nodes.analyze_vendors_node import analyze_vendors_node
from product_search.nodes.rank_products_node import rank_products_node
from product_search.nodes.format_response_node import format_response_node

logger = logging.getLogger(__name__)


# =============================================================================
# INTERRUPT WRAPPER
# =============================================================================

def _collect_requirements_with_interrupt(state: ProductSearchDeepAgentState) -> ProductSearchDeepAgentState:
    """
    Thin wrapper around collect_requirements_node that raises a LangGraph
    interrupt when the node sets awaiting_user_input=True.

    On resume the graph will re-enter this wrapper; the inner node detects
    user_confirmed=True and skips the interrupt.
    """
    state = collect_requirements_node(state)

    if state.get("awaiting_user_input") and not state.get("user_confirmed"):
        # Expose state snapshot for the caller (Flask API / frontend)
        interrupt({
            "reason": "awaiting_user_confirmation",
            "sales_agent_response": state.get("sales_agent_response"),
            "product_type": state.get("product_type"),
            "schema": state.get("schema"),
            "missing_fields": state.get("missing_fields", []),
            "available_advanced_params": state.get("available_advanced_params", []),
        })
        # After interrupt is cleared the graph re-enters this node;
        # collect_requirements_node then takes the resume path.

    return state


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def _route_after_validate(state: ProductSearchDeepAgentState) -> str:
    """Route after validation: proceed to discover_advanced_params or END."""
    if state.get("error") or not state.get("product_type"):
        logger.warning("[route] validate_product → END (error or no product_type)")
        return END
    logger.info("[route] validate_product → discover_advanced_params")
    return "discover_advanced_params"


def _route_after_collect(state: ProductSearchDeepAgentState) -> str:
    """Route after requirements collection: proceed to analyze_vendors or END."""
    if state.get("awaiting_user_input") and not state.get("user_confirmed"):
        # Still waiting — stay in the interrupted state (graph checkpointed)
        logger.info("[route] collect_requirements → END (awaiting user input, interrupted)")
        return END
    logger.info("[route] collect_requirements → analyze_vendors")
    return "analyze_vendors"


def _route_after_vendor_analysis(state: ProductSearchDeepAgentState) -> str:
    """Route after vendor analysis: always proceed to rank_products."""
    return "rank_products"


# =============================================================================
# GRAPH FACTORY
# =============================================================================

def create_product_search_graph(checkpointing_backend: str = "memory") -> Any:
    """
    Build and compile the Product Search StateGraph.

    Args:
        checkpointing_backend: "memory" | "sqlite" | "mongodb" | "azure"

    Returns:
        Compiled LangGraph app (with checkpointing for HITL support)
    """
    graph = StateGraph(ProductSearchDeepAgentState)

    # Add nodes
    graph.add_node("validate_product", validate_product_node)
    graph.add_node("discover_advanced_params", discover_advanced_params_node)
    graph.add_node("collect_requirements", _collect_requirements_with_interrupt)
    graph.add_node("analyze_vendors", analyze_vendors_node)
    graph.add_node("rank_products", rank_products_node)
    graph.add_node("format_response", format_response_node)

    # Entry point
    graph.set_entry_point("validate_product")

    # Conditional edge: after validate_product
    graph.add_conditional_edges(
        "validate_product",
        _route_after_validate,
        {
            "discover_advanced_params": "discover_advanced_params",
            END: END,
        },
    )

    # Discover → collect (always)
    graph.add_edge("discover_advanced_params", "collect_requirements")

    # Conditional edge: after collect_requirements (HITL)
    graph.add_conditional_edges(
        "collect_requirements",
        _route_after_collect,
        {
            "analyze_vendors": "analyze_vendors",
            END: END,
        },
    )

    # Analyze → rank → format → END
    graph.add_edge("analyze_vendors", "rank_products")
    graph.add_edge("rank_products", "format_response")
    graph.add_edge("format_response", END)

    # Compile with checkpointing (required for HITL/interrupt support)
    return compile_with_checkpointing(graph, checkpointing_backend)


# =============================================================================
# SESSION / INSTANCE ID MANAGEMENT
# =============================================================================

def _setup_session_ids(
    session_id: Optional[str],
    main_thread_id: Optional[str],
    parent_workflow_id: Optional[str],
) -> tuple:
    """
    Resolve and generate the thread/instance ID hierarchy.

    Returns: (session_id, main_thread_id, workflow_thread_id, instance_id)
    """
    session_id = session_id or f"ps_{uuid.uuid4().hex[:8]}"
    main_thread_id = main_thread_id or session_id

    # Generate unique workflow thread id for this graph run
    workflow_thread_id = f"product_search_{uuid.uuid4().hex[:12]}"

    # Generate instance id
    instance_id = str(uuid.uuid4())

    # Register instance with WorkflowInstanceManager (best effort)
    try:
        from common.infrastructure.state.execution.instance_manager import WorkflowInstanceManager
        WorkflowInstanceManager.register(
            instance_id=instance_id,
            session_id=session_id,
            workflow_type="product_search",
            parent_workflow_id=parent_workflow_id,
            thread_id=workflow_thread_id,
        )
    except Exception as exc:
        logger.debug("WorkflowInstanceManager.register failed (non-critical): %s", exc)

    return session_id, main_thread_id, workflow_thread_id, instance_id


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_product_search_workflow(
    user_input: str,
    session_id: Optional[str] = None,
    main_thread_id: Optional[str] = None,
    parent_workflow_id: Optional[str] = None,
    expected_product_type: Optional[str] = None,
    auto_mode: bool = False,
    skip_advanced_params: bool = False,
    max_vendor_workers: int = 10,
    checkpointing_backend: str = "memory",
    zone: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full Product Search Deep Agent workflow.

    Thread/Instance ID hierarchy
    ----------------------------
    main_thread_id (= session root, from solution_workflow or frontend)
      └── workflow_thread_id  [product_search_<uuid>, used as LangGraph thread_id]
            └── instance_id  [uuid4, registered in WorkflowInstanceManager]
                  parent_workflow_id → item_thread_id from solution_workflow (or None)

    Args:
        user_input: Raw user requirement description
        session_id: Root session identifier (auto-generated if omitted)
        main_thread_id: Anchor thread ID (defaults to session_id)
        parent_workflow_id: Optional calling workflow thread ID
        expected_product_type: Hint for product type detection
        auto_mode: If True, skip HITL (requirements confirmed automatically)
        skip_advanced_params: If True, skip Step 2 discovery
        max_vendor_workers: Max parallel threads for vendor analysis (Step 4)
        checkpointing_backend: LangGraph checkpointing backend
        zone: Optional Azure zone hint

    Returns:
        Final ProductSearchDeepAgentState dict with analysis_result populated
    """
    logger.info("=" * 70)
    logger.info("[run_product_search_workflow] Starting Product Search Deep Agent")
    logger.info("[run_product_search_workflow] user_input: %s", user_input[:100])

    session_id, main_thread_id, workflow_thread_id, instance_id = _setup_session_ids(
        session_id, main_thread_id, parent_workflow_id
    )

    logger.info("[run_product_search_workflow] session_id=%s workflow_thread_id=%s instance_id=%s",
                session_id, workflow_thread_id, instance_id)

    initial_state = create_product_search_deep_agent_state(
        user_input=user_input,
        session_id=session_id,
        main_thread_id=main_thread_id,
        workflow_thread_id=workflow_thread_id,
        instance_id=instance_id,
        parent_workflow_id=parent_workflow_id,
        expected_product_type=expected_product_type,
        sales_agent_mode="auto" if auto_mode else "interactive",
        skip_advanced_params=skip_advanced_params,
        max_vendor_workers=max_vendor_workers,
        zone=zone,
    )

    app = create_product_search_graph(checkpointing_backend)
    config = {"configurable": {"thread_id": workflow_thread_id}}

    try:
        final_state = app.invoke(initial_state, config=config)
        logger.info(
            "[run_product_search_workflow] ✓ Completed — success=%s, ranked=%d",
            final_state.get("success"),
            len(final_state.get("overall_ranking", [])),
        )
        return final_state
    except Exception as exc:
        logger.error("[run_product_search_workflow] ✗ Failed: %s", exc, exc_info=True)
        initial_state["success"] = False
        initial_state["error"] = str(exc)
        initial_state["error_type"] = type(exc).__name__
        return initial_state


# =============================================================================
# HITL RESUME ENTRY POINT
# =============================================================================

def resume_product_search_workflow(
    workflow_thread_id: str,
    user_response: str = "YES",
    selected_advanced_params: Optional[Dict[str, Any]] = None,
    checkpointing_backend: str = "memory",
) -> Dict[str, Any]:
    """
    Resume a workflow that was interrupted waiting for user confirmation.

    Args:
        workflow_thread_id: The thread ID from the interrupted run
        user_response: Text response from user ("YES" or additional details)
        selected_advanced_params: Optional dict of selected advanced specs
        checkpointing_backend: Must match the backend used in the initial run

    Returns:
        Final state after workflow completes
    """
    logger.info("[resume_product_search_workflow] Resuming thread '%s'", workflow_thread_id)

    app = create_product_search_graph(checkpointing_backend)
    config = {"configurable": {"thread_id": workflow_thread_id}}

    # Fetch current state to merge requirements safely
    try:
        current_snapshot = app.get_state(config)
        current_values = current_snapshot.values
        provided_reqs = current_values.get("provided_requirements", {}).copy()
    except Exception as exc:
        logger.warning("[resume_product_search_workflow] Could not fetch state: %s", exc)
        provided_reqs = {}

    # Merge advanced params if provided
    if selected_advanced_params:
        # We store selected params in provided_requirements.selectedAdvancedParams
        # so collect_requirements_node can pick them up
        provided_reqs["selectedAdvancedParams"] = {
            **provided_reqs.get("selectedAdvancedParams", {}),
            **selected_advanced_params
        }

    # Prepare user updates
    user_updates = {
        "user_confirmed": True,
        "awaiting_user_input": False,
        "user_input": user_response,  # Update input for context
        "provided_requirements": provided_reqs,
    }

    try:
        final_state = app.invoke(user_updates, config=config)
        logger.info("[resume_product_search_workflow] ✓ Resumed and completed")
        return final_state
    except Exception as exc:
        logger.error("[resume_product_search_workflow] ✗ Failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc), "workflow_thread_id": workflow_thread_id}


# =============================================================================
# BACKWARD-COMPATIBLE CONVENIENCE WRAPPERS
# =============================================================================

def run_single_product_workflow(
    user_input: str,
    expected_product_type: Optional[str] = None,
    session_id: Optional[str] = None,
    skip_advanced_params: bool = False,
) -> Dict[str, Any]:
    """
    Backward-compatible alias for run_product_search_workflow in auto mode.

    Matches the original ProductSearchWorkflow.run_single_product_workflow() signature.
    """
    return run_product_search_workflow(
        user_input=user_input,
        session_id=session_id,
        expected_product_type=expected_product_type,
        auto_mode=True,
        skip_advanced_params=skip_advanced_params,
    )


def run_analysis_only(
    structured_requirements: Dict[str, Any],
    product_type: str,
    schema: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    skip_advanced_params: bool = False,
    max_vendor_workers: int = 10,
    discovered_specs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run only the analysis phase (Steps 4-5): vendor analysis + ranking.

    This is called after the user has confirmed requirements through the
    interactive sales-agent and clicks Proceed.

    P1-G FIX: If ``skip_advanced_params`` is False *and* ``discovered_specs``
    is provided (or can be fetched), merges the discovered specs into
    ``structured_requirements['selectedAdvancedParams']`` before analysis —
    matching the original ProductSearchWorkflow.run_analysis_only() behaviour.

    Matches the original ProductSearchWorkflow.run_analysis_only() signature.
    """
    session_id, main_thread_id, workflow_thread_id, instance_id = _setup_session_ids(
        session_id, None, None
    )

    # -------------------------------------------------------------------------
    # P1-G: merge discovered specs into structured_requirements
    # -------------------------------------------------------------------------
    merged_requirements = dict(structured_requirements)  # shallow copy
    if not skip_advanced_params and discovered_specs:
        existing_advanced = merged_requirements.get("selectedAdvancedParams", {})
        if isinstance(existing_advanced, dict) and isinstance(discovered_specs, dict):
            merged_requirements["selectedAdvancedParams"] = {**discovered_specs, **existing_advanced}
        elif not existing_advanced:
            merged_requirements["selectedAdvancedParams"] = discovered_specs
        logger.info(
            "[run_analysis_only] Merged %d discovered spec(s) into selectedAdvancedParams",
            len(discovered_specs) if isinstance(discovered_specs, dict) else 0,
        )

    # Build a minimal state that starts at analyze_vendors
    initial_state = create_product_search_deep_agent_state(
        user_input=f"[analysis_only] {product_type}",
        session_id=session_id,
        main_thread_id=main_thread_id,
        workflow_thread_id=workflow_thread_id,
        instance_id=instance_id,
        expected_product_type=product_type,
        sales_agent_mode="auto",
        skip_advanced_params=skip_advanced_params,
        max_vendor_workers=max_vendor_workers,
    )
    # Inject pre-built data so graph nodes can use it without re-running earlier steps
    initial_state["product_type"] = product_type
    initial_state["schema"] = schema or {}
    initial_state["structured_requirements"] = merged_requirements   # use merged version
    initial_state["user_confirmed"] = True
    initial_state["provided_requirements"] = merged_requirements

    # Build a sub-graph that only runs steps 4-5-format
    sub_graph = StateGraph(ProductSearchDeepAgentState)
    sub_graph.add_node("analyze_vendors", analyze_vendors_node)
    sub_graph.add_node("rank_products", rank_products_node)
    sub_graph.add_node("format_response", format_response_node)
    sub_graph.set_entry_point("analyze_vendors")
    sub_graph.add_edge("analyze_vendors", "rank_products")
    sub_graph.add_edge("rank_products", "format_response")
    sub_graph.add_edge("format_response", END)
    app = compile_with_checkpointing(sub_graph, "memory")

    config = {"configurable": {"thread_id": workflow_thread_id}}
    try:
        return app.invoke(initial_state, config=config)
    except Exception as exc:
        logger.error("[run_analysis_only] Failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc), "product_type": product_type}


def run_validation_only(
    user_input: str,
    expected_product_type: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run only Step 1 (validate_product) and return its result dict.

    Matches the original ProductSearchWorkflow.run_validation_only() signature.
    """
    session_id, main_thread_id, workflow_thread_id, instance_id = _setup_session_ids(
        session_id, None, None
    )

    initial_state = create_product_search_deep_agent_state(
        user_input=user_input,
        session_id=session_id,
        main_thread_id=main_thread_id,
        workflow_thread_id=workflow_thread_id,
        instance_id=instance_id,
        expected_product_type=expected_product_type,
        sales_agent_mode="auto",
    )

    # Run only the validate_product node (single node sub-graph)
    sub_graph = StateGraph(ProductSearchDeepAgentState)
    sub_graph.add_node("validate_product", validate_product_node)
    sub_graph.set_entry_point("validate_product")
    sub_graph.add_edge("validate_product", END)
    app = compile_with_checkpointing(sub_graph, "memory")

    config = {"configurable": {"thread_id": workflow_thread_id}}
    try:
        final = app.invoke(initial_state, config=config)
        return final.get("validation_result", final)
    except Exception as exc:
        logger.error("[run_validation_only] Failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


def run_advanced_params_only(
    product_type: str,
    session_id: Optional[str] = None,
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run only Step 2 (discover_advanced_params) and return the result.

    [P2-A] Matches the original ProductSearchWorkflow.run_advanced_params_only() signature.

    Args:
        product_type: Product type to discover advanced parameters for.
        session_id: Session identifier (auto-generated if omitted).
        schema: Optional pre-loaded schema (passed into the state for context).

    Returns:
        Dict with keys: success, unique_specifications, total_unique_specifications,
        vendors_searched, discovery_successful, advanced_params_result.
    """
    session_id, main_thread_id, workflow_thread_id, instance_id = _setup_session_ids(
        session_id, None, None
    )

    initial_state = create_product_search_deep_agent_state(
        user_input=f"[advanced_params_only] {product_type}",
        session_id=session_id,
        main_thread_id=main_thread_id,
        workflow_thread_id=workflow_thread_id,
        instance_id=instance_id,
        expected_product_type=product_type,
        sales_agent_mode="auto",
    )
    initial_state["product_type"] = product_type
    initial_state["schema"] = schema or {}

    sub_graph = StateGraph(ProductSearchDeepAgentState)
    sub_graph.add_node("discover_advanced_params", discover_advanced_params_node)
    sub_graph.set_entry_point("discover_advanced_params")
    sub_graph.add_edge("discover_advanced_params", END)
    app = compile_with_checkpointing(sub_graph, "memory")

    config = {"configurable": {"thread_id": workflow_thread_id}}
    try:
        final = app.invoke(initial_state, config=config)
        return final.get("advanced_params_result", final)
    except Exception as exc:
        logger.error("[run_advanced_params_only] Failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc), "product_type": product_type}


def process_from_instrument_identifier(
    identifier_output: Dict[str, Any],
    session_id: Optional[str] = None,
    parent_workflow_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process input from the Instruments Identifier Workflow.

    Matches the original ProductSearchWorkflow.process_from_instrument_identifier() signature.

    Args:
        identifier_output: {
            "identified_instruments": [
                {
                    "product_type": "Pressure Transmitter",
                    "quantity": 2,
                    "basic_requirements": {"measurementRange": "0-100 bar", ...}
                }
            ]
        }
    """
    logger.info("[process_from_instrument_identifier] Processing output from Instruments Identifier")

    identified = identifier_output.get("identified_instruments", [])
    results: List[Dict[str, Any]] = []

    for i, instrument in enumerate(identified, 1):
        product_type = instrument.get("product_type", "")
        basic_requirements = instrument.get("basic_requirements", {})
        quantity = instrument.get("quantity", 1)

        logger.info("[process_from_instrument_identifier] [%d/%d] %s (Qty: %d)",
                    i, len(identified), product_type, quantity)

        user_input = f"I need {quantity} {product_type}"
        if basic_requirements:
            specs = ", ".join(f"{k}: {v}" for k, v in basic_requirements.items())
            user_input += f" with {specs}"

        result = run_product_search_workflow(
            user_input=user_input,
            session_id=session_id,
            parent_workflow_id=parent_workflow_id,
            expected_product_type=product_type,
            auto_mode=True,
        )
        results.append({
            "product_type": product_type,
            "quantity": quantity,
            "workflow_result": result,
        })

    return {
        "source": "instruments_identifier",
        "results": results,
        "total_products": len(results),
        "successful": sum(1 for r in results if r["workflow_result"].get("success")),
    }


def process_from_solution_workflow(
    solution_output: Dict[str, Any],
    session_id: Optional[str] = None,
    parent_workflow_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process input from the Solution Workflow.

    Matches the original ProductSearchWorkflow.process_from_solution_workflow() signature.

    Args:
        solution_output: {
            "solution_name": "Water Treatment Plant",
            "required_products": [
                {
                    "product_type": "pH Meter",
                    "application": "Wastewater monitoring",
                    "requirements": {...}
                }
            ]
        }
    """
    logger.info("[process_from_solution_workflow] Processing output from Solution Workflow")

    solution_name = solution_output.get("solution_name", "Unknown Solution")
    
    # -------------------------------------------------------------------------
    # ADAPTIVE HANDLING: Solution Agent Output Formats
    # -------------------------------------------------------------------------
    # New format has "items" (enriched/normalized list)
    # Legacy format has "required_products"
    
    required_products = solution_output.get("required_products", [])
    
    if not required_products and "items" in solution_output:
        logger.info("[process_from_solution_workflow] Detected new Solution format ('items'). Transforming payload.")
        items_list = solution_output.get("items", [])
        
        for item in items_list:
            # 1. Use Canonical Name (from Taxonomy RAG) -> Product Name -> Name
            p_type = item.get("canonical_name") or item.get("product_name") or item.get("name") or "Unknown Product"
            
            # 2. Extract specs and sample input
            specs = item.get("specifications", {})
            sample_in = item.get("sample_input", "")
            
            product_entry = {
                "product_type": p_type,
                "quantity": item.get("quantity", 1),
                "application": f"{solution_name} - {item.get('category', 'General')}",
                "requirements": specs,
                "sample_input": sample_in,
                "original_name": item.get("name", ""),
                "taxonomy_matched": item.get("taxonomy_matched", False)
            }
            required_products.append(product_entry)

    logger.info("[process_from_solution_workflow] Solution: %s | Products: %d",
                solution_name, len(required_products))

    results: List[Dict[str, Any]] = []

    for i, product in enumerate(required_products, 1):
        product_type = product.get("product_type", "")
        application = product.get("application", "")
        requirements = product.get("requirements", {})
        sample_input = product.get("sample_input", "")

        logger.info("[process_from_solution_workflow] [%d/%d] %s — %s",
                    i, len(required_products), product_type, application)

        # Use explicitly provided sample input if available (highest fidelity)
        if sample_input:
            user_input = sample_input
            logger.info("  Using provided sample input: %s", user_input[:60])
        else:
            # Fallback construction
            user_input = f"I need a {product_type}"
            if application:
                user_input += f" for {application}"
            if requirements:
                # Limit specs to avoid overflowing prompt limits if list is huge
                specs_list = [f"{k}: {v}" for k, v in requirements.items()]
                specs = ", ".join(specs_list[:15]) 
                user_input += f" with {specs}"
            logger.info("  Constructed user input: %s...", user_input[:60])

        result = run_product_search_workflow(
            user_input=user_input,
            session_id=session_id,
            parent_workflow_id=parent_workflow_id,
            expected_product_type=product_type,
            auto_mode=True,
        )
        results.append({
            "product_type": product_type,
            "application": application,
            "workflow_result": result,
        })

    return {
        "source": "solution_workflow",
        "solution_name": solution_name,
        "results": results,
        "total_products": len(results),
        "successful": sum(1 for r in results if r["workflow_result"].get("success")),
    }


def product_search_workflow(
    user_input: str,
    expected_product_type: Optional[str] = None,
    enable_ppi: bool = True,
    skip_advanced_params: bool = False,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Top-level convenience wrapper — matches the original module-level
    product_search_workflow() function in product_search_workflow/workflow.py.

    Args:
        user_input: User requirement description
        expected_product_type: Hint for product type detection
        enable_ppi: Ignored (PPI is always enabled via load_schema_tool)
        skip_advanced_params: Skip advanced parameters discovery
        session_id: Session identifier

    Returns:
        Complete workflow result with analysis_result
    """
    return run_product_search_workflow(
        user_input=user_input,
        session_id=session_id,
        expected_product_type=expected_product_type,
        auto_mode=True,
        skip_advanced_params=skip_advanced_params,
    )


# =============================================================================
# P2-B: generate_comparison_table
# =============================================================================

def _format_field_name(field: str) -> str:
    """Convert camelCase or snake_case to Title Case."""
    words = re.sub(r'([a-z])([A-Z])', r'\1 \2', field)
    words = words.replace('_', ' ')
    return words.title()


def generate_comparison_table(
    overall_ranking: List[Dict[str, Any]],
    requirements: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a UI-ready comparison table from a ranked products list.

    [P2-B] Mirrors RankingTool.generate_comparison_table() exactly.

    Args:
        overall_ranking: List of ranked product dicts (from rank_products_node /
                         run_analysis_only result['overallRanking']['rankedProducts']).
        requirements: The structured requirements dict with keys
                      'mandatoryRequirements' / 'mandatory' and
                      'optionalRequirements' / 'optional'.

    Returns:
        {
            "columns": [{"key": ..., "label": ...}, ...],
            "rows":    [{"rank": 1, "vendor": ..., ...}, ...]
        }
    """
    if not overall_ranking:
        return {"columns": [], "rows": []}

    # Collect all unique parameter names from mandatory + optional requirements
    all_params: set = set()
    mandatory = requirements.get('mandatoryRequirements', requirements.get('mandatory', {}))
    optional  = requirements.get('optionalRequirements',  requirements.get('optional',  {}))
    all_params.update(mandatory.keys())
    all_params.update(optional.keys())

    # Fixed leading columns
    columns: List[Dict[str, str]] = [
        {"key": "rank",         "label": "Rank"},
        {"key": "vendor",       "label": "Vendor"},
        {"key": "product_name", "label": "Product"},
        {"key": "match_score",  "label": "Match %"},
    ]

    # One column per requirement parameter (sorted for stable order)
    for param in sorted(all_params):
        columns.append({"key": param, "label": _format_field_name(param)})

    # Build rows
    rows: List[Dict[str, Any]] = []
    for product in overall_ranking:
        row: Dict[str, Any] = {
            "rank":         product.get('rank', 0),
            "vendor":       product.get('vendor', 'Unknown'),
            "product_name": product.get('productName', product.get('product_name', 'Unknown')),
            "match_score":  product.get('matchScore',  product.get('match_score', 0)),
        }

        # Populate parameter columns from keyStrengths
        key_strengths = product.get('keyStrengths', product.get('key_strengths', []))
        for strength in key_strengths:
            if isinstance(strength, dict):
                param_name  = strength.get('parameter', strength.get('name', ''))
                param_value = strength.get('specification', strength.get('value', ''))
                for param in all_params:
                    if param.lower() in param_name.lower():
                        row[param] = param_value
                        break

        rows.append(row)

    return {"columns": columns, "rows": rows}


# =============================================================================
# P2-C: get_schema_only / validate_with_schema
# =============================================================================

def get_schema_only(
    product_type: str,
    enable_ppi: bool = True,
) -> Dict[str, Any]:
    """
    Load or generate a schema for *product_type* without running the full
    validation pipeline.

    [P2-C] Mirrors ValidationTool.get_schema_only().

    Args:
        product_type: The product type string (e.g. "pressure transmitter").
        enable_ppi:   Whether to allow PPI workflow to generate the schema if
                      it is not found in Azure Blob Storage.  Default True.

    Returns:
        Dict with at least:
            success    (bool)
            schema     (dict)
            source     (str)  — "azure_blob" | "ppi" | "default" | ...
            ppi_used   (bool)
    """
    logger.info("[get_schema_only] Loading schema for '%s'", product_type)
    try:
        from common.tools.schema_tools import load_schema_tool
        result = load_schema_tool.invoke({
            "product_type": product_type,
            "enable_ppi": enable_ppi,
        })
        logger.info("[get_schema_only] ✓ source=%s", result.get("source", "unknown"))
        return result
    except Exception as exc:
        logger.error("[get_schema_only] ✗ Failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc), "schema": {}}


def validate_with_schema(
    user_input: str,
    product_type: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate user input against a *pre-loaded* schema without running the
    full validate_product_node pipeline.

    [P2-C] Mirrors ValidationTool.validate_with_schema().

    Args:
        user_input:   Raw user requirement description.
        product_type: Product type string.
        schema:       Pre-loaded schema dict (from get_schema_only or state).

    Returns:
        Dict with at least:
            success               (bool)
            product_type          (str)
            provided_requirements (dict)
            missing_fields        (list)
            optional_fields       (list)   — P1-B field included
            is_valid              (bool)
    """
    logger.info("[validate_with_schema] Validating '%s' against provided schema", product_type)
    try:
        from common.tools.schema_tools import validate_requirements_tool
        result = validate_requirements_tool.invoke({
            "user_input":   user_input,
            "product_type": product_type,
            "schema":       schema,
        })
        return {
            "success":               True,
            "product_type":          product_type,
            "provided_requirements": result.get("provided_requirements", {}),
            "missing_fields":        result.get("missing_fields", []),
            "optional_fields":       result.get("optional_fields", []),   # P1-B
            "is_valid":              result.get("is_valid", False),
        }
    except Exception as exc:
        logger.error("[validate_with_schema] ✗ Failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


# =============================================================================
# P3-C: validate_multiple_products_parallel / enrich_schema_parallel
# =============================================================================

def validate_multiple_products_parallel(
    product_types: List[str],
    session_id: Optional[str] = None,
    max_workers: int = 3,
    force_regenerate: bool = False,
) -> Dict[str, Any]:
    """
    Generate schemas for multiple products in parallel (Phase 2 optimisation).

    [P3-C] Mirrors ValidationTool.validate_multiple_products_parallel().

    Uses ``ParallelSchemaGenerator`` (ThreadPoolExecutor backed, bounded to the
    global executor) to overlap N PPI/Azure schema lookups concurrently.
    Falls back to sequential ``get_schema_only()`` when the generator is not available.

    Args:
        product_types:    List of product type strings.
        session_id:       Optional session identifier (for logging only).
        max_workers:      Thread pool size passed to ParallelSchemaGenerator (default 3).
        force_regenerate: When True, bypass Azure/session caches.

    Returns:
        Dict[product_type -> result_dict]  each has: success, schema, schema_source,
        product_type, optimization ("phase2_parallel" | "sequential_fallback").
    """
    if not product_types:
        return {}

    logger.info(
        "[validate_multiple_products_parallel] Parallel schema gen for %d products "
        "(session=%s, workers=%d)",
        len(product_types), session_id, max_workers,
    )

    # --- Try ParallelSchemaGenerator (Phase 2) --------------------------------
    try:
        from common.agentic.deep_agent.schema.generation.parallel_generator import ParallelSchemaGenerator

        actual_workers = min(max_workers, len(product_types))
        generator = ParallelSchemaGenerator(max_workers=actual_workers)
        raw = generator.generate_schemas_in_parallel(
            product_types, force_regenerate=force_regenerate
        )

        results: Dict[str, Any] = {}
        for pt, schema_result in raw.items():
            if schema_result.get("success"):
                results[pt] = {
                    "success": True,
                    "product_type": pt,
                    "schema": schema_result.get("schema", {}),
                    "schema_source": schema_result.get("source", "parallel_generator"),
                    "optimization": "phase2_parallel",
                }
            else:
                results[pt] = {
                    "success": False,
                    "product_type": pt,
                    "error": schema_result.get("error", "Unknown error"),
                    "optimization": "phase2_parallel",
                }

        logger.info(
            "[validate_multiple_products_parallel] ✓ %d/%d succeeded",
            sum(1 for r in results.values() if r.get("success")), len(product_types),
        )
        return results

    except ImportError:
        logger.warning(
            "[validate_multiple_products_parallel] ParallelSchemaGenerator not available "
            "— falling back to sequential"
        )
    except Exception as exc:
        logger.error(
            "[validate_multiple_products_parallel] Parallel failed: %s "
            "— falling back to sequential", exc,
        )

    # --- Sequential fallback --------------------------------------------------
    results = {}
    for pt in product_types:
        try:
            r = get_schema_only(pt)
            results[pt] = {
                "success": r.get("success", False),
                "product_type": pt,
                "schema": r.get("schema", {}),
                "schema_source": r.get("source", "sequential_fallback"),
                "optimization": "sequential_fallback",
                **({"error": r["error"]} if not r.get("success") else {}),
            }
        except Exception as exc:
            results[pt] = {
                "success": False,
                "product_type": pt,
                "error": str(exc),
                "optimization": "sequential_fallback",
            }
    return results


def enrich_schema_parallel(
    product_type: str,
    schema: Dict[str, Any],
    max_workers: int = 5,
) -> Dict[str, Any]:
    """
    Enrich a schema using parallel field-group standards RAG queries (Phase 2).

    [P3-C] Mirrors ValidationTool.enrich_schema_parallel().

    Queries all field groups simultaneously via ``ParallelStandardsEnrichment``
    (approx. 5× faster than sequential queries).  Returns schema unchanged when
    the module is not available.

    Args:
        product_type: Product type string.
        schema:       Schema dict to enrich.
        max_workers:  Thread pool size (default 5).

    Returns:
        Enriched schema dict.
    """
    logger.info(
        "[enrich_schema_parallel] Parallel standards enrichment for '%s'", product_type
    )
    try:
        from common.standards.shared.parallel_standards_enrichment import (
            ParallelStandardsEnrichment,
        )
        enricher = ParallelStandardsEnrichment(max_workers=max_workers)
        enriched = enricher.enrich_schema_in_parallel(product_type, schema)
        logger.info("[enrich_schema_parallel] ✓ Enrichment complete for '%s'", product_type)
        return enriched
    except ImportError:
        logger.warning(
            "[enrich_schema_parallel] ParallelStandardsEnrichment not available — schema unchanged"
        )
        return schema
    except Exception as exc:
        logger.error(
            "[enrich_schema_parallel] Failed for '%s': %s — schema unchanged", product_type, exc
        )
        return schema


# =============================================================================
# P3-D: get_or_generate_schema_async / get_or_generate_schemas_batch_async
# =============================================================================

async def get_or_generate_schema_async(
    product_type: str,
    session_id: Optional[str] = None,
    force_regenerate: bool = False,
) -> Dict[str, Any]:
    """
    Get or generate a schema using the complete async lifecycle (Phase 3).

    [P3-D] Mirrors ValidationTool.get_or_generate_schema_async().

    Full lifecycle (priority order):
        1. Session cache check
        2. Azure Blob Storage lookup
        3. PPI workflow generation
        4. Async-parallel standards enrichment
        5. Store enriched schema back to Azure

    Fallback chain: Phase 3 SchemaWorkflow → Phase 2 parallel → sync get_schema_only.

    Args:
        product_type:     Product type string.
        session_id:       Session identifier (used for session-cache dedup).
        force_regenerate: Skip all caches when True.

    Returns:
        Dict with at least: success, schema, source, product_type.
    """
    logger.info(
        "[get_or_generate_schema_async] Async lifecycle for '%s' (force=%s)",
        product_type, force_regenerate,
    )

    # --- Try SchemaWorkflow (Phase 3) ----------------------------------------
    try:
        from common.agentic.workflows.schema.schema_workflow import SchemaWorkflow
        workflow = SchemaWorkflow(use_phase3_async=True)
        result = await workflow.get_or_generate_schema(
            product_type,
            session_id=session_id,
            force_regenerate=force_regenerate,
        )
        logger.info("[get_or_generate_schema_async] ✓ Phase 3 complete for '%s'", product_type)
        return result
    except ImportError:
        logger.warning(
            "[get_or_generate_schema_async] SchemaWorkflow not available "
            "— falling back to Phase 2 parallel"
        )
    except Exception as exc:
        logger.error(
            "[get_or_generate_schema_async] Phase 3 failed for '%s': %s "
            "— falling back to Phase 2 parallel", product_type, exc,
        )

    # --- Fall back to Phase 2 parallel ---------------------------------------
    try:
        phase2_results = validate_multiple_products_parallel(
            [product_type], session_id=session_id, force_regenerate=force_regenerate
        )
        result = phase2_results.get(product_type, {})
        if result.get("success"):
            logger.info(
                "[get_or_generate_schema_async] ✓ Phase 2 fallback succeeded for '%s'",
                product_type,
            )
            return result
    except Exception as exc2:
        logger.error("[get_or_generate_schema_async] Phase 2 fallback failed: %s", exc2)

    # --- Last resort: sync ---------------------------------------------------
    logger.info("[get_or_generate_schema_async] Falling back to sync get_schema_only")
    return get_schema_only(product_type)


async def get_or_generate_schemas_batch_async(
    product_types: List[str],
    session_id: Optional[str] = None,
    force_regenerate: bool = False,
    max_concurrent: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    Get or generate schemas for multiple products concurrently (Phase 3 batch).

    [P3-D] Mirrors ValidationTool.get_or_generate_schemas_batch_async().

    Uses SchemaWorkflow.get_or_generate_schemas_batch() with asyncio.gather()
    for true concurrent execution.

    Performance estimate (3 products):
        Sequential:   (437 + 210) × 3 ≈ 1941 s
        Phase 3 batch:  100 – 120 s       (≈ 19× faster)

    Fallback chain:
        Phase 3 SchemaWorkflow batch
        → Phase 2 ThreadPoolExecutor (validate_multiple_products_parallel)
        → asyncio.gather() of get_or_generate_schema_async() per product

    Args:
        product_types:    List of product type strings.
        session_id:       Session identifier for cache dedup.
        force_regenerate: Skip all caches when True.
        max_concurrent:   asyncio.Semaphore limit for last-resort path (default 3).

    Returns:
        Dict[product_type -> result_dict]
    """
    if not product_types:
        return {}

    logger.info(
        "[get_or_generate_schemas_batch_async] Async batch for %d products "
        "(session=%s, max_concurrent=%d)",
        len(product_types), session_id, max_concurrent,
    )

    # --- Try SchemaWorkflow batch (Phase 3) -----------------------------------
    try:
        from common.agentic.workflows.schema.schema_workflow import SchemaWorkflow
        workflow = SchemaWorkflow(use_phase3_async=True)
        results = await workflow.get_or_generate_schemas_batch(
            product_types, session_id=session_id
        )
        logger.info(
            "[get_or_generate_schemas_batch_async] ✓ Phase 3 batch complete: %d/%d succeeded",
            sum(1 for r in results.values() if r.get("success")), len(product_types),
        )
        return results
    except ImportError:
        logger.warning(
            "[get_or_generate_schemas_batch_async] SchemaWorkflow not available "
            "— falling back to Phase 2 parallel"
        )
    except Exception as exc:
        logger.error(
            "[get_or_generate_schemas_batch_async] Phase 3 batch failed: %s "
            "— falling back to Phase 2 parallel", exc,
        )

    # --- Fall back to Phase 2 parallel (ThreadPoolExecutor) ------------------
    try:
        results = validate_multiple_products_parallel(
            product_types, session_id=session_id, force_regenerate=force_regenerate
        )
        logger.info(
            "[get_or_generate_schemas_batch_async] ✓ Phase 2 fallback: %d/%d succeeded",
            sum(1 for r in results.values() if r.get("success")), len(product_types),
        )
        return results
    except Exception as exc2:
        logger.error(
            "[get_or_generate_schemas_batch_async] Phase 2 fallback failed: %s "
            "— running per-product async tasks", exc2,
        )

    # --- Last resort: concurrent asyncio tasks --------------------------------
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _bounded(pt: str):
        async with semaphore:
            return pt, await get_or_generate_schema_async(
                pt, session_id=session_id, force_regenerate=force_regenerate
            )

    pairs = await asyncio.gather(
        *[_bounded(pt) for pt in product_types], return_exceptions=True
    )

    results = {}
    for item in pairs:
        if isinstance(item, Exception):
            logger.error("[get_or_generate_schemas_batch_async] Task exception: %s", item)
            continue
        pt, result = item
        results[pt] = result

    return results
