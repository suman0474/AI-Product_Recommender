# search/entry_points.py
# =============================================================================
# SEARCH DEEP AGENT ENTRY POINTS
# =============================================================================
#
# All public entry functions for the Search Deep Agent workflow.
# Follows the Solution Deep Agent pattern with:
# - Proper time tracking
# - Error handling with try/except/finally
# - Thread-safe execution with @with_workflow_lock
# - Consistent return structure (success/response/response_data)
# - Streaming support
#
# =============================================================================

import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Callable

from .state import SearchDeepAgentState, create_search_deep_agent_state
from .workflow import create_search_workflow_graph
from common.infrastructure.state.context.lock_monitor import with_workflow_lock

logger = logging.getLogger(__name__)


# =============================================================================
# MAIN ENTRY POINTS
# =============================================================================

@with_workflow_lock(session_id_param="session_id", timeout=120.0)
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
    Run the full Search Deep Agent workflow.

    This is the primary entry point for running a complete product search.
    Uses thread-safe locking to prevent concurrent searches on the same session.

    Args:
        user_input: Raw user requirement description
        session_id: Root session identifier
        main_thread_id: Parent thread ID (for nested workflows)
        parent_workflow_id: Parent workflow ID (for nested workflows)
        expected_product_type: Hint for product type detection
        auto_mode: If True, skip HITL interactions
        skip_advanced_params: If True, skip advanced params discovery
        max_vendor_workers: Max parallel workers for vendor analysis
        checkpointing_backend: LangGraph checkpointing backend
        zone: Geographic zone for vendor filtering

    Returns:
        Dict with success, response, response_data, and error fields
    """
    start_time = time.time()
    generated_session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"

    try:
        logger.info(f"[SearchDeepAgent] Starting for session: {generated_session_id}")
        logger.info(f"[SearchDeepAgent] Input: {user_input[:100]}...")

        # Create initial state
        state = create_search_deep_agent_state(
            user_input=user_input,
            session_id=generated_session_id,
            expected_product_type=expected_product_type,
            execution_mode="auto" if auto_mode else "interactive",
            auto_mode=auto_mode,
            skip_advanced_params=skip_advanced_params,
            max_vendor_workers=max_vendor_workers,
            main_thread_id=main_thread_id,
            parent_workflow_id=parent_workflow_id,
            zone=zone,
        )

        # Create and run graph
        graph = create_search_workflow_graph(checkpointing_backend)

        config = {
            "configurable": {
                "thread_id": state["workflow_thread_id"],
            }
        }

        # Invoke the graph
        result = graph.invoke(state, config)

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[SearchDeepAgent] Complete: {len(result.get('overall_ranking', []))} "
            f"products ranked in {elapsed_ms}ms"
        )

        return {
            "success": True,
            "response": result.get("response", ""),
            "response_data": result.get("response_data", {}),
            "error": result.get("error"),
        }

    except TimeoutError as e:
        logger.error(f"[SearchDeepAgent] Lock timeout: {e}")
        return {
            "success": False,
            "error": "Another search is running on this session. Please try again.",
        }
    except Exception as e:
        logger.error(f"[SearchDeepAgent] Failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "response": f"Error searching for products: {str(e)}",
            "response_data": {
                "workflow": "search",
                "error": str(e),
            },
        }


@with_workflow_lock(session_id_param="session_id", timeout=120.0)
def run_product_search_workflow_stream(
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
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Run the Search Deep Agent workflow with streaming progress updates.

    Similar to run_product_search_workflow but provides progress callbacks
    as each phase completes.

    Args:
        user_input: Raw user requirement description
        session_id: Root session identifier
        main_thread_id: Parent thread ID (for nested workflows)
        parent_workflow_id: Parent workflow ID (for nested workflows)
        expected_product_type: Hint for product type detection
        auto_mode: If True, skip HITL interactions
        skip_advanced_params: If True, skip advanced params discovery
        max_vendor_workers: Max parallel workers for vendor analysis
        checkpointing_backend: LangGraph checkpointing backend
        zone: Geographic zone for vendor filtering
        progress_callback: Optional callback(phase_data) called as phases complete

    Returns:
        Dict with success, response, response_data, and error fields
    """
    start_time = time.time()
    generated_session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"

    try:
        logger.info(f"[SearchDeepAgent] Starting stream for session: {generated_session_id}")
        logger.info(f"[SearchDeepAgent] Input: {user_input[:100]}...")

        # Create initial state
        state = create_search_deep_agent_state(
            user_input=user_input,
            session_id=generated_session_id,
            expected_product_type=expected_product_type,
            execution_mode="auto" if auto_mode else "interactive",
            auto_mode=auto_mode,
            skip_advanced_params=skip_advanced_params,
            max_vendor_workers=max_vendor_workers,
            main_thread_id=main_thread_id,
            parent_workflow_id=parent_workflow_id,
            zone=zone,
        )

        # Create graph
        graph = create_search_workflow_graph(checkpointing_backend)

        config = {
            "configurable": {
                "thread_id": state["workflow_thread_id"],
            }
        }

        # Stream events and call progress callback
        if progress_callback:
            for event in graph.stream(state, config):
                progress_callback(event)
                logger.debug(f"[SearchDeepAgent] Stream event: {event}")

        # Get final state via invoke
        result = graph.invoke(state, config)

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[SearchDeepAgent] Stream complete: {len(result.get('overall_ranking', []))} "
            f"products ranked in {elapsed_ms}ms"
        )

        return {
            "success": True,
            "response": result.get("response", ""),
            "response_data": result.get("response_data", {}),
            "error": result.get("error"),
        }

    except TimeoutError as e:
        logger.error(f"[SearchDeepAgent] Lock timeout: {e}")
        return {
            "success": False,
            "error": "Another search is running on this session. Please try again.",
        }
    except Exception as e:
        logger.error(f"[SearchDeepAgent] Stream failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "response": f"Error searching for products: {str(e)}",
            "response_data": {
                "workflow": "search",
                "error": str(e),
            },
        }


def resume_product_search_workflow(
    workflow_thread_id: str,
    user_response: Optional[str] = None,
    selected_advanced_params: Optional[Dict[str, Any]] = None,
    resume_state: Optional[Dict[str, Any]] = None,
    checkpointing_backend: str = "memory",
) -> Dict[str, Any]:
    """
    Resume a HITL-interrupted workflow.

    Args:
        workflow_thread_id: The thread ID of the interrupted workflow
        user_response: User's text response
        selected_advanced_params: Selected advanced parameters
        resume_state: State updates to apply before resuming
        checkpointing_backend: LangGraph checkpointing backend

    Returns:
        Final workflow state as dictionary
    """
    logger.info("[entry_points] Resuming workflow: %s", workflow_thread_id)

    # Get the graph
    graph = create_search_workflow_graph(checkpointing_backend)

    config = {
        "configurable": {
            "thread_id": workflow_thread_id,
        }
    }

    # Build state updates
    updates = resume_state or {}

    if user_response:
        updates["additional_specs_raw"] = user_response
        updates["additional_specs_collected"] = True

    if selected_advanced_params:
        updates["user_provided_values"] = selected_advanced_params

    updates["user_confirmed"] = True
    updates["workflow_interrupted"] = False

    # Resume with updates
    result = graph.invoke(updates, config)

    logger.info(
        "[entry_points] Resume complete: success=%s",
        result.get("success"),
    )

    return dict(result)


def run_single_product_workflow(
    user_input: str,
    expected_product_type: Optional[str] = None,
    session_id: Optional[str] = None,
    skip_advanced_params: bool = False,
) -> Dict[str, Any]:
    """
    Auto-mode convenience wrapper for single product search.

    Args:
        user_input: Raw user requirement description
        expected_product_type: Hint for product type detection
        session_id: Session identifier
        skip_advanced_params: Skip advanced params discovery

    Returns:
        Final workflow state as dictionary
    """
    return run_product_search_workflow(
        user_input=user_input,
        session_id=session_id,
        expected_product_type=expected_product_type,
        auto_mode=True,
        skip_advanced_params=skip_advanced_params,
    )


# =============================================================================
# STEP-SPECIFIC ENTRY POINTS
# =============================================================================

def run_validation_only(
    user_input: str,
    expected_product_type: Optional[str] = None,
    session_id: Optional[str] = None,
    enable_ppi: bool = True,
) -> Dict[str, Any]:
    """
    Run only the validation step.

    Args:
        user_input: Raw user requirement description
        expected_product_type: Hint for product type detection
        session_id: Session identifier
        enable_ppi: Allow PPI schema generation

    Returns:
        Validation result dictionary
    """
    logger.info("[entry_points] Running validation only")

    from .agents import ValidationAgent

    agent = ValidationAgent()
    result = agent.validate(
        user_input=user_input,
        expected_product_type=expected_product_type,
        enable_ppi=enable_ppi,
    )

    return result.to_dict()


def run_advanced_params_only(
    product_type: str,
    session_id: Optional[str] = None,
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run only the advanced parameters discovery step.

    Args:
        product_type: The product type to discover params for
        session_id: Session identifier
        schema: Existing schema to filter against

    Returns:
        Advanced params result dictionary
    """
    logger.info("[entry_points] Running advanced params only")

    from .agents import ParamsAgent

    agent = ParamsAgent()
    result = agent.discover(
        product_type=product_type,
        session_id=session_id,
        existing_schema=schema,
    )

    return result.to_dict()


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
    Run only vendor analysis and ranking (Steps 4-5).

    This is useful for chaining from Solution workflow or other sources.

    Args:
        structured_requirements: Pre-built requirements dictionary
        product_type: The product type
        schema: Optional schema for context
        session_id: Session identifier
        skip_advanced_params: Skip advanced params discovery
        max_vendor_workers: Max parallel workers
        discovered_specs: Pre-discovered specifications

    Returns:
        Analysis result dictionary
    """
    logger.info("[entry_points] Running analysis only for: %s", product_type)

    from .agents import VendorAgent, RankingAgent

    # Run vendor analysis
    vendor_agent = VendorAgent(max_workers=max_vendor_workers)
    vendor_result = vendor_agent.analyze(
        requirements=structured_requirements,
        product_type=product_type,
        schema=schema,
        session_id=session_id,
    )

    if not vendor_result.success:
        return {
            "success": False,
            "error": vendor_result.error,
            "vendor_matches": [],
            "overall_ranking": [],
        }

    # Run ranking
    ranking_agent = RankingAgent()
    ranking_result = ranking_agent.rank(
        vendor_analysis=vendor_result.to_dict(),
        requirements=structured_requirements,
    )

    return {
        "success": True,
        "vendor_analysis": vendor_result.to_dict(),
        "vendor_matches": vendor_result.vendor_matches,
        "overall_ranking": ranking_result.overall_ranking,
        "top_product": ranking_result.top_product,
        "ranking_summary": ranking_result.ranking_summary,
    }


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_from_instrument_identifier(
    identifier_output: Dict[str, Any],
    session_id: Optional[str] = None,
    parent_workflow_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Batch processing from Instrument Identifier output.

    Args:
        identifier_output: Output from instrument identifier
        session_id: Session identifier
        parent_workflow_id: Parent workflow ID

    Returns:
        Batch processing result
    """
    logger.info("[entry_points] Processing from instrument identifier")

    instruments = identifier_output.get("instruments", [])
    results = []

    for instrument in instruments:
        product_type = instrument.get("type") or instrument.get("name", "")
        requirements = instrument.get("specifications", {})

        result = run_analysis_only(
            structured_requirements=requirements,
            product_type=product_type,
            session_id=session_id,
        )

        results.append({
            "instrument": instrument,
            "search_result": result,
        })

    return {
        "success": True,
        "total_instruments": len(instruments),
        "results": results,
    }


def process_from_solution_workflow(
    solution_output: Dict[str, Any],
    session_id: Optional[str] = None,
    parent_workflow_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Batch processing from Solution Workflow output.

    Args:
        solution_output: Output from solution workflow
        session_id: Session identifier
        parent_workflow_id: Parent workflow ID

    Returns:
        Batch processing result
    """
    logger.info("[entry_points] Processing from solution workflow")

    instruments = solution_output.get("identified_instruments", [])
    accessories = solution_output.get("identified_accessories", [])

    all_items = instruments + accessories
    results = []

    for item in all_items:
        product_type = item.get("type") or item.get("name", "")
        requirements = item.get("specifications", {})

        result = run_analysis_only(
            structured_requirements=requirements,
            product_type=product_type,
            session_id=session_id,
        )

        results.append({
            "item": item,
            "search_result": result,
        })

    return {
        "success": True,
        "total_items": len(all_items),
        "results": results,
    }


# =============================================================================
# SCHEMA UTILITIES
# =============================================================================

def get_schema_only(
    product_type: str,
    enable_ppi: bool = True,
) -> Dict[str, Any]:
    """
    Load or generate schema without validation.

    Args:
        product_type: The product type
        enable_ppi: Allow PPI schema generation

    Returns:
        Schema dictionary
    """
    logger.info("[entry_points] Getting schema only for: %s", product_type)

    try:
        from common.services.schema_service import schema_service

        schema = schema_service.get_schema(product_type)

        if schema:
            return {
                "success": True,
                "schema": schema,
                "source": "database",
            }

        if enable_ppi:
            # Trigger PPI generation
            return {
                "success": False,
                "schema": {},
                "source": "ppi_pending",
                "message": "Schema not found, PPI generation required",
            }

        return {
            "success": False,
            "schema": {},
            "source": "not_found",
        }

    except Exception as exc:
        logger.error("[entry_points] Schema lookup failed: %s", exc)
        return {
            "success": False,
            "schema": {},
            "source": "error",
            "error": str(exc),
        }


def validate_with_schema(
    user_input: str,
    product_type: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate user input against pre-loaded schema.

    Args:
        user_input: Raw user input
        product_type: The product type
        schema: Pre-loaded schema

    Returns:
        Validation result
    """
    logger.info("[entry_points] Validating with schema")

    from .agents import ValidationAgent

    agent = ValidationAgent()

    # Use internal validation method
    is_valid, missing, optional, provided = agent._validate_requirements(
        user_input, product_type, schema
    )

    return {
        "success": True,
        "is_valid": is_valid,
        "missing_fields": missing,
        "optional_fields": optional,
        "provided_requirements": provided,
    }


def validate_multiple_products_parallel(
    product_types: List[str],
    session_id: Optional[str] = None,
    max_workers: int = 3,
    force_regenerate: bool = False,
) -> Dict[str, Any]:
    """
    Validate multiple products in parallel.

    Args:
        product_types: List of product types
        session_id: Session identifier
        max_workers: Maximum parallel workers
        force_regenerate: Force schema regeneration

    Returns:
        Batch validation results
    """
    logger.info("[entry_points] Validating %d products in parallel", len(product_types))

    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_schema_only, pt): pt
            for pt in product_types
        }

        for future in as_completed(futures):
            product_type = futures[future]
            try:
                result = future.result()
                results[product_type] = result
            except Exception as exc:
                results[product_type] = {
                    "success": False,
                    "error": str(exc),
                }

    return {
        "success": True,
        "total": len(product_types),
        "results": results,
    }


def enrich_schema_parallel(
    schemas: Dict[str, Dict[str, Any]],
    session_id: Optional[str] = None,
    max_workers: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    Enrich multiple schemas with standards in parallel.

    Args:
        schemas: Dict mapping product_type to schema
        session_id: Session identifier
        max_workers: Maximum parallel workers

    Returns:
        Dict mapping product_type to enriched schema
    """
    logger.info("[entry_points] Enriching %d schemas in parallel", len(schemas))

    results = {}

    def enrich_one(product_type: str, schema: Dict[str, Any]) -> tuple:
        try:
            from common.tools.standards_enrichment_tool import (
                get_applicable_standards,
                populate_schema_fields_from_standards,
            )

            standards = get_applicable_standards(product_type)
            enriched = populate_schema_fields_from_standards(product_type, schema)

            return (product_type, enriched.get("schema", schema))
        except Exception as exc:
            return (product_type, schema)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(enrich_one, pt, schema)
            for pt, schema in schemas.items()
        ]

        for future in as_completed(futures):
            product_type, enriched_schema = future.result()
            results[product_type] = enriched_schema

    return results


# =============================================================================
# ASYNC UTILITIES
# =============================================================================

async def get_or_generate_schema_async(
    product_type: str,
    session_id: Optional[str] = None,
    force_regenerate: bool = False,
) -> Dict[str, Any]:
    """
    Async schema lifecycle.

    Args:
        product_type: The product type
        session_id: Session identifier
        force_regenerate: Force regeneration

    Returns:
        Schema result
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        get_schema_only,
        product_type,
        True,
    )


async def get_or_generate_schemas_batch_async(
    product_types: List[str],
    session_id: Optional[str] = None,
    force_regenerate: bool = False,
    max_concurrent: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    Async batch schema lifecycle.

    Args:
        product_types: List of product types
        session_id: Session identifier
        force_regenerate: Force regeneration
        max_concurrent: Max concurrent operations

    Returns:
        Dict mapping product_type to result
    """
    results = {}

    async def get_one(pt: str) -> tuple:
        result = await get_or_generate_schema_async(pt, session_id, force_regenerate)
        return (pt, result)

    # Process in batches
    for i in range(0, len(product_types), max_concurrent):
        batch = product_types[i:i + max_concurrent]
        tasks = [get_one(pt) for pt in batch]
        batch_results = await asyncio.gather(*tasks)

        for pt, result in batch_results:
            results[pt] = result

    return results


# =============================================================================
# UI UTILITIES
# =============================================================================

def generate_comparison_table(
    overall_ranking: List[Dict[str, Any]],
    requirements: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build UI comparison table from ranked products.

    Args:
        overall_ranking: List of ranked products
        requirements: Original requirements for comparison

    Returns:
        Comparison table data
    """
    if not overall_ranking:
        return {
            "headers": [],
            "rows": [],
            "total_products": 0,
        }

    # Build headers from requirements
    req_keys = [k for k in requirements.keys() if not k.startswith("_")]
    headers = ["Rank", "Vendor", "Model", "Score"] + req_keys

    # Build rows
    rows = []
    for product in overall_ranking:
        row = [
            product.get("rank", 0),
            product.get("vendor", ""),
            product.get("productName") or product.get("model", ""),
            product.get("overallScore", 0),
        ]

        # Add requirement values
        specs = product.get("specifications", {})
        for key in req_keys:
            row.append(specs.get(key, "N/A"))

        rows.append(row)

    return {
        "headers": headers,
        "rows": rows,
        "total_products": len(rows),
    }


# =============================================================================
# TOP-LEVEL CONVENIENCE WRAPPER
# =============================================================================

def product_search_workflow(
    user_input: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    Top-level convenience wrapper for product search.

    This is the simplest entry point - just pass user input.

    Args:
        user_input: Raw user requirement description
        **kwargs: Additional arguments passed to run_product_search_workflow

    Returns:
        Final workflow state as dictionary
    """
    return run_product_search_workflow(user_input=user_input, **kwargs)
