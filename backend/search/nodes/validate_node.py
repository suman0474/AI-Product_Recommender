# search/nodes/validate_node.py
# =============================================================================
# VALIDATE NODE (Node 2)
# =============================================================================
#
# Extracts product type, loads schema, applies standards enrichment,
# and validates requirements.
#
# =============================================================================

import logging
from typing import TYPE_CHECKING

from ..agents import ValidationAgent
from ..caching import get_session_enrichment_cache, compute_session_cache_key
from ..state import add_system_message, mark_step_complete, set_error

if TYPE_CHECKING:
    from ..state import SearchDeepAgentState

logger = logging.getLogger(__name__)


def validate_node(state: "SearchDeepAgentState") -> "SearchDeepAgentState":
    """
    LangGraph node - Phase 2: Validation.

    Extracts product type from user input, loads the schema,
    applies standards enrichment, and validates requirements.

    Reads: user_input, expected_product_type, execution_plan, session_id
    Writes: product_type, schema, validation_result, is_valid, missing_fields, etc.
    """
    logger.info("[validate_node] ===== PHASE 2: VALIDATION =====")
    state["current_step"] = "validate"

    try:
        # Get inputs
        user_input = state.get("user_input", "")
        expected_product_type = state.get("expected_product_type")
        session_id = state.get("session_id", "default")
        plan = state.get("execution_plan", {})

        # Get settings from plan
        enable_ppi = plan.get("tool_hints", {}).get("enable_ppi", True)
        standards_depth = plan.get("tool_hints", {}).get("standards_depth", "shallow")

        # Check session cache for previous validation
        cache = get_session_enrichment_cache()
        cache_key = compute_session_cache_key(session_id, expected_product_type or "unknown")

        cached = cache.get(cache_key)
        if cached:
            logger.info("[validate_node] Session cache HIT - using cached validation")
            _apply_cached_result(state, cached)
            add_system_message(state, "Validation loaded from cache", "validate")
            mark_step_complete(state, "validate")
            return state

        # Run validation agent
        agent = ValidationAgent()
        result = agent.validate(
            user_input=user_input,
            expected_product_type=expected_product_type,
            enable_ppi=enable_ppi,
            standards_depth=standards_depth,
        )

        # Check for errors
        if result.error:
            set_error(state, result.error, "ValidationError")
            add_system_message(state, f"Validation failed: {result.error}", "validate")
            return state

        # Update state with validation results
        state["product_type"] = result.product_type
        state["original_product_type"] = result.original_product_type
        state["product_type_refined"] = result.product_type_refined
        state["schema"] = result.schema
        state["schema_source"] = result.schema_source
        state["validation_result"] = result.to_dict()
        state["is_valid"] = result.is_valid
        state["missing_fields"] = result.missing_fields
        state["optional_fields"] = result.optional_fields
        state["provided_requirements"] = result.provided_requirements
        state["standards_enrichment_applied"] = result.standards_applied
        state["standards_info"] = result.standards_info
        state["enrichment_result"] = result.enrichment_result
        state["rag_invocations"] = result.rag_invocations

        # Cache the result
        cache.set(cache_key, result.to_cache_dict())

        # Add system message
        add_system_message(
            state,
            f"Product: {result.product_type} | Schema: {result.schema_source} | "
            f"Valid: {result.is_valid} | Missing: {len(result.missing_fields)}",
            "validate",
        )

        mark_step_complete(state, "validate")

        logger.info(
            "[validate_node] Validation complete: product_type=%s, is_valid=%s, missing=%d",
            result.product_type,
            result.is_valid,
            len(result.missing_fields),
        )

    except Exception as exc:
        logger.error("[validate_node] Validation failed: %s", exc, exc_info=True)
        set_error(state, f"Validation failed: {str(exc)}", "ValidationError")
        add_system_message(state, f"Validation error: {str(exc)}", "validate")

    return state


def _apply_cached_result(state: "SearchDeepAgentState", cached: dict) -> None:
    """Apply cached validation result to state."""
    state["product_type"] = cached.get("product_type", "")
    state["schema"] = cached.get("schema", {})
    state["schema_source"] = cached.get("schema_source", "cache")
    state["standards_info"] = cached.get("standards_info")
    state["standards_enrichment_applied"] = cached.get("standards_applied", False)

    # These need to be computed fresh
    state["is_valid"] = True
    state["missing_fields"] = []
    state["provided_requirements"] = {}
