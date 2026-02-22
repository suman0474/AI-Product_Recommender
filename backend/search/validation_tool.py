"""
Validation Tool for Product Search Workflow
============================================

Step 1 of Product Search Workflow:
- Detects product type from user input
- Loads or generates schema (with PPI workflow if needed)
- Validates requirements against schema
- Returns structured validation result

This tool integrates:
- Intent extraction (product type detection)
- Schema loading/generation (with PPI workflow)
- Requirements validation
"""

import logging
from typing import Dict, Any, Optional, List
import threading

# Configure logging
logger = logging.getLogger(__name__)

# Debug flags for validation tool debugging
try:
    from debug_flags import debug_log, timed_execution, is_debug_enabled
    DEBUG_AVAILABLE = True
except ImportError:
    DEBUG_AVAILABLE = False
    def debug_log(module, **kwargs):
        def decorator(func):
            return func
        return decorator
    def timed_execution(module, **kwargs):
        def decorator(func):
            return func
        return decorator
    def is_debug_enabled(module):
        return False

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  [FIX #A1] SESSION-LEVEL ENRICHMENT DEDUPLICATION                       ║
# ║  Prevents redundant Standards RAG calls for same product in one session ║
# ║  Cache: product_type -> enrichment_result (thread-safe)                 ║
# ║  [PHASE 1] Using BoundedCache with TTL/LRU to prevent memory leaks      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
from common.infrastructure.caching.bounded_cache import get_or_create_cache, BoundedCache

_session_enrichment_cache: BoundedCache = get_or_create_cache(
    name="session_enrichment",
    max_size=200,           # Max 200 concurrent sessions
    ttl_seconds=1800        # 30 minute session TTL
)


def _get_session_enrichment(product_type: str, session_id: str) -> Optional[Dict[str, Any]]:
    """
    Get cached enrichment result for this session.

    Args:
        product_type: Product type to look up
        session_id: Session identifier (REQUIRED for isolation between concurrent requests)

    Returns:
        Cached enrichment result or None if not found/session_id missing

    Note:
        [FIX Feb 2026] Made session_id required to prevent cross-session contamination.
        Previously, missing session_id caused global cache scope which led to
        concurrent requests interfering with each other's state.
    """
    if not session_id:
        logger.warning("[FIX #A1] _get_session_enrichment called without session_id - skipping cache lookup to prevent cross-session contamination")
        return None  # Don't use global cache - return cache miss

    normalized_type = product_type.lower().strip()
    # Always include session_id prefix for strict isolation
    key = f"enrichment:{session_id}:{normalized_type}"

    # BoundedCache is thread-safe internally, no lock needed
    return _session_enrichment_cache.get(key)


def _cache_session_enrichment(product_type: str, enrichment_result: Dict[str, Any], session_id: str):
    """
    Cache enrichment result for this session.

    Args:
        product_type: Product type to cache
        enrichment_result: Data to cache
        session_id: Session identifier (REQUIRED for isolation between concurrent requests)

    Note:
        [FIX Feb 2026] Made session_id required to prevent cross-session contamination.
        Previously, missing session_id caused global cache scope which led to
        concurrent requests interfering with each other's state.
    """
    if not session_id:
        logger.warning("[FIX #A1] _cache_session_enrichment called without session_id - skipping cache write to prevent cross-session contamination")
        return  # Don't pollute global cache

    normalized_type = product_type.lower().strip()
    # Always include session_id prefix for strict isolation
    key = f"enrichment:{session_id}:{normalized_type}"

    # BoundedCache is thread-safe internally, no lock needed
    _session_enrichment_cache.set(key, enrichment_result)
    logger.info(f"[FIX #A1] Cached enrichment for {key} (size: {len(_session_enrichment_cache)})")


def clear_session_enrichment_cache():
    """Clear session enrichment cache (call at start of new session)."""
    # BoundedCache.clear() is thread-safe internally
    count = _session_enrichment_cache.clear()
    logger.info(f"[FIX #A1] Session enrichment cache cleared ({count} entries)")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION CONTEXT CACHE - Stores product_type for HITL YES/NO responses    ║
# ║  When user says "YES" without product_type, retrieve from this cache      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
_session_context_cache: BoundedCache = get_or_create_cache(
    name="session_context",
    max_size=500,           # Max 500 concurrent sessions
    ttl_seconds=3600        # 1 hour session context TTL
)


def _get_session_context(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Get cached session context (product_type, schema, etc.) for HITL responses.

    Args:
        session_id: Session identifier (REQUIRED for isolation)

    Returns:
        Cached context dict or None if not found

    Note:
        [FIX Feb 2026] Added 'context:' prefix to key for namespace isolation
    """
    if not session_id:
        logger.warning("[SessionContext] _get_session_context called without session_id")
        return None
    # Prefix key for namespace isolation
    key = f"context:{session_id}"
    return _session_context_cache.get(key)


def _cache_session_context(session_id: str, context: Dict[str, Any]):
    """
    Cache session context after validation for HITL response handling.

    Args:
        session_id: Session identifier (REQUIRED for isolation)
        context: Dict containing product_type, schema, provided_requirements, etc.

    Note:
        [FIX Feb 2026] Added 'context:' prefix to key for namespace isolation
    """
    if not session_id:
        logger.warning("[SessionContext] _cache_session_context called without session_id - skipping")
        return
    # Prefix key for namespace isolation
    key = f"context:{session_id}"
    _session_context_cache.set(key, context)
    logger.info(f"[SessionContext] Cached context for {key}: product_type={context.get('product_type')}")


def clear_session_cache(session_id: str):
    """
    Clear all cached data for a specific session.
    Call this when a session ends or user starts a new search.

    Args:
        session_id: Session identifier to clear

    Note:
        [FIX Feb 2026] Added to support proper session cleanup and prevent
        stale data from interfering with new searches.
    """
    if not session_id:
        logger.warning("[SessionCleanup] clear_session_cache called without session_id")
        return

    # Clear context cache for this session
    context_key = f"context:{session_id}"
    _session_context_cache.delete(context_key)

    # Note: For enrichment cache, we use prefixed keys per product type
    # We can't easily clear all keys for a session without iterating
    # The TTL (30 min) will handle cleanup, but we log a warning
    logger.info(f"[SessionCleanup] Cleared session context cache for: {session_id}")
    logger.info(f"[SessionCleanup] Note: Enrichment cache entries will expire via TTL (30 min)")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  THREAD-LOCAL CONTEXT FOR REQUEST ISOLATION                               ║
# ║  [FIX Feb 2026] Ensures each concurrent request has isolated state        ║
# ║  [UPGRADE Feb 2026] Now integrates with ExecutionContext system           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
import contextvars

# Import ExecutionContext for proper context integration
from common.infrastructure.context import (
    ExecutionContext,
    get_context as get_execution_context,
    get_session_id as get_ctx_session_id,
    get_cache_key as get_ctx_cache_key
)

# Thread-local context for current request - prevents cross-request contamination
# NOTE: These are maintained for backward compatibility. New code should use ExecutionContext.
_current_session_id: contextvars.ContextVar[str] = contextvars.ContextVar('current_session_id', default='')
_current_workflow_thread_id: contextvars.ContextVar[str] = contextvars.ContextVar('current_workflow_thread_id', default='')


def set_request_context(session_id: str, workflow_thread_id: str = ''):
    """
    Set thread-local context for current request.
    Call at the start of request processing.

    Args:
        session_id: Session identifier for this request
        workflow_thread_id: Optional workflow thread ID

    Note:
        This maintains backward compatibility. The ExecutionContext is set
        separately by the API layer using execution_context() context manager.
    """
    _current_session_id.set(session_id)
    _current_workflow_thread_id.set(workflow_thread_id)
    logger.debug(f"[RequestContext] Set context: session={session_id}, thread={workflow_thread_id}")


def get_request_session_id() -> str:
    """
    Get session_id for current request from thread-local context.

    Checks in order:
    1. ExecutionContext (preferred, set by API layer)
    2. Legacy _current_session_id contextvar

    Returns:
        Session ID or empty string if not set
    """
    # First try ExecutionContext (preferred)
    ctx = get_execution_context()
    if ctx and ctx.session_id:
        return ctx.session_id

    # Fallback to legacy contextvar
    return _current_session_id.get()


def get_request_workflow_thread_id() -> str:
    """
    Get workflow_thread_id for current request from thread-local context.

    Checks in order:
    1. ExecutionContext (preferred, set by API layer)
    2. Legacy _current_workflow_thread_id contextvar

    Returns:
        Workflow thread ID or empty string if not set
    """
    # First try ExecutionContext (preferred)
    ctx = get_execution_context()
    if ctx and ctx.workflow_id:
        return ctx.workflow_id

    # Fallback to legacy contextvar
    return _current_workflow_thread_id.get()


def get_isolated_cache_key(suffix: str) -> str:
    """
    Get a cache key scoped to the current execution context.

    Uses ExecutionContext for proper session/workflow isolation.

    Args:
        suffix: Cache key suffix (e.g., "enrichment:pressure_transmitter")

    Returns:
        Isolated cache key in format: ctx:{session_id}:{workflow_id}:{suffix}
    """
    ctx = get_execution_context()
    if ctx:
        return ctx.to_cache_key(suffix)

    # Fallback to session_id-based key
    session_id = get_request_session_id()
    if session_id:
        return f"session:{session_id}:{suffix}"

    logger.warning(f"[CacheKey] No context for cache key '{suffix}' - using global scope")
    return f"global:{suffix}"


def clear_request_context():
    """Clear thread-local context after request completes."""
    _current_session_id.set('')
    _current_workflow_thread_id.set('')


class ValidationTool:
    """
    Validation Tool - Step 1 of Product Search Workflow

    Responsibilities:
    1. Extract product type from user input
    2. Load or generate product schema (PPI workflow if needed)
    3. Validate user requirements against schema
    4. Return structured validation result
    """

    def __init__(
        self,
        enable_ppi: bool = True,
        enable_phase2: bool = True,
        enable_phase3: bool = True,
        use_async_workflow: bool = True,
        enable_standards_enrichment: bool = True
    ):
        """
        Initialize the validation tool.

        Args:
            enable_ppi: Enable PPI workflow for schema generation
            enable_phase2: Enable Phase 2 parallel optimization
            enable_phase3: Enable Phase 3 async optimization (highest priority)
            use_async_workflow: Use new async SchemaWorkflow (recommended)
            enable_standards_enrichment: Enable Standards RAG enrichment (Step 1.2.1)
        """
        self.enable_ppi = enable_ppi
        self.enable_phase2 = enable_phase2
        self.enable_phase3 = enable_phase3
        self.use_async_workflow = use_async_workflow
        self.enable_standards_enrichment = enable_standards_enrichment

        logger.info("[ValidationTool] Initialized with PPI workflow: %s",
                   "enabled" if enable_ppi else "disabled")
        logger.info("[ValidationTool] Phase 2 optimization: %s",
                   "enabled" if enable_phase2 else "disabled")
        logger.info("[ValidationTool] Phase 3 async optimization: %s",
                   "enabled" if enable_phase3 else "disabled")
        logger.info("[ValidationTool] Standards enrichment: %s",
                   "enabled" if enable_standards_enrichment else "disabled")

    def update_specifications(self, current_specs: Dict[str, Any], new_input: str) -> Dict[str, Any]:
        """Update existing specifications with newly provided values from the user."""
        from common.tools.intent_tools import extract_requirements_tool
        logger.info(f"[ValidationTool] Updating specifications with new input: '{new_input}'")
        try:
            extract_result = extract_requirements_tool.invoke({
                "user_input": new_input
            })
            new_specs = extract_result.get("specifications", {})
            updated_specs = dict(current_specs)
            updated_specs.update(new_specs)
            return updated_specs
        except Exception as e:
            logger.error(f"[ValidationTool] Failed to update specifications: {e}")
            return current_specs

    @timed_execution("VALIDATION_TOOL", threshold_ms=20000)
    @debug_log("VALIDATION_TOOL", log_args=True, log_result=False)
    def validate(
        self,
        user_input: str,
        expected_product_type: Optional[str] = None,
        session_id: Optional[str] = None,
        enable_standards_enrichment: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Validate user input and requirements.

        Args:
            user_input: User's requirement description
            expected_product_type: Expected product type (optional, for validation)
            session_id: Session identifier (for logging/tracking)

        Returns:
            Validation result with:
            {
                "success": bool,
                "product_type": str,
                "schema": dict,
                "provided_requirements": dict,
                "missing_fields": list,
                "is_valid": bool,
                "ppi_workflow_used": bool,
                "schema_source": str,  # "azure", "ppi_workflow", or "default_fallback"
                "session_id": str
            }
        """
        logger.info("[ValidationTool] Starting validation")
        logger.info("[ValidationTool] Session: %s", session_id or "N/A")
        logger.info("[ValidationTool] Input: %s", user_input[:100] + "..." if len(user_input) > 100 else user_input)

        result = {
            "success": False,
            "session_id": session_id
        }

        # =====================================================================
        # FIX: Detect UI decision patterns BEFORE calling LLM
        # UI decisions like "User selected: continue" should not be processed
        # as product requirements. This prevents unnecessary LLM calls and
        # provides clearer error messages.
        # =====================================================================
        try:
            from debug_flags import is_ui_decision_input, get_ui_decision_error_message

            if is_ui_decision_input(user_input):
                logger.warning(f"[ValidationTool] UI decision pattern detected: '{user_input}'")
                logger.warning("[ValidationTool] This input should be routed to a different API endpoint")
                result.update({
                    "success": False,
                    "error": get_ui_decision_error_message(user_input),
                    "error_type": "UIDecisionPatternError",
                    "is_ui_decision": True,
                    "hint": "Use /api/agentic/run-analysis for 'continue' actions after requirements are collected"
                })
                return result
        except ImportError:
            # Fallback if debug_flags module not available
            ui_patterns = ["user selected:", "user clicked:", "decision:", "continue", "proceed"]
            normalized_input = user_input.lower().strip()
            for pattern in ui_patterns:
                if pattern in normalized_input or normalized_input == pattern:
                    logger.warning(f"[ValidationTool] UI decision pattern detected: '{user_input}'")
                    result.update({
                        "success": False,
                        "error": f"Input '{user_input}' is a UI action, not a product requirement. Please provide product specifications.",
                        "error_type": "UIDecisionPatternError",
                        "is_ui_decision": True
                    })
                    return result

        try:
            # Import required tools
            from common.tools.schema_tools import load_schema_tool, validate_requirements_tool
            from common.tools.intent_tools import extract_requirements_tool

            # ╔══════════════════════════════════════════════════════════════════════════╗
            # ║  EARLY DETECTION: YES HITL RESPONSE → BYPASS VALIDATION, RUN ADV SPECS    ║
            # ║  When user responds "yes" to HITL prompt, STOP validation_tool and        ║
            # ║  IMMEDIATELY trigger AdvancedSpecificationAgent only                       ║
            # ╚══════════════════════════════════════════════════════════════════════════╝
            normalized_input_lower = user_input.lower().strip()
            is_hitl_yes_response = normalized_input_lower in ["yes", "y"]
            is_hitl_no_response = normalized_input_lower in ["no", "n"] or normalized_input_lower.startswith("no ")

            # ═══════════════════════════════════════════════════════════════════════════
            # "YES" RESPONSE: STOP VALIDATION → RUN ADVANCED SPECIFICATION AGENT ONLY
            # ═══════════════════════════════════════════════════════════════════════════
            if is_hitl_yes_response and expected_product_type:
                logger.info("=" * 70)
                logger.info("[ValidationTool] ✓ USER SAID 'YES' - BYPASSING VALIDATION")
                logger.info("[ValidationTool] → Triggering AdvancedSpecificationAgent ONLY")
                logger.info(f"   Product Type: {expected_product_type}")
                logger.info(f"   Session: {session_id}")
                logger.info("=" * 70)

                # IMMEDIATELY trigger AdvancedSpecificationAgent and return
                try:
                    from search.advanced_specification_agent import AdvancedSpecificationAgent

                    adv_agent = AdvancedSpecificationAgent()
                    advanced_specs_result = adv_agent.discover(
                        product_type=expected_product_type,
                        session_id=session_id
                    )

                    if advanced_specs_result.get("success"):
                        num_specs = advanced_specs_result.get("total_unique_specifications", 0)
                        logger.info(f"[ValidationTool] ✓ Advanced specs discovered: {num_specs} parameters")

                        # Return ONLY advanced specs result - validation is bypassed
                        # BUG FIX: also return schema from session context so Turn 1 schema is preserved
                        _ctx = _get_session_context(session_id) or {}
                        return {
                            "success": True,
                            "product_type": expected_product_type,
                            "schema": _ctx.get("schema", {}),
                            "provided_requirements": _ctx.get("provided_requirements", {}),
                            "is_valid": True,  # User confirmed to proceed
                            "hitl_response": "yes",
                            "validation_bypassed": True,
                            "advanced_specs_info": {
                                "triggered": True,
                                "success": True,
                                "unique_specifications": advanced_specs_result.get("unique_specifications", []),
                                "total_unique_specifications": num_specs,
                                "existing_specifications_filtered": advanced_specs_result.get("existing_specifications_filtered", 0),
                                "discovery_successful": advanced_specs_result.get("discovery_successful", False),
                                "fallback_used": advanced_specs_result.get("fallback_used", False)
                            },
                            "session_id": session_id,
                            "message": f"Discovered {num_specs} advanced specifications for {expected_product_type}"
                        }
                    else:
                        logger.warning(f"[ValidationTool] ⚠ Advanced specs discovery failed: {advanced_specs_result.get('error')}")
                        return {
                            "success": False,
                            "product_type": expected_product_type,
                            "hitl_response": "yes",
                            "validation_bypassed": True,
                            "error": advanced_specs_result.get("error", "Advanced specification discovery failed"),
                            "error_type": "AdvancedSpecsError",
                            "session_id": session_id
                        }

                except ImportError as e:
                    logger.error(f"[ValidationTool] AdvancedSpecificationAgent not available: {e}")
                    return {
                        "success": False,
                        "product_type": expected_product_type,
                        "hitl_response": "yes",
                        "validation_bypassed": True,
                        "error": f"AdvancedSpecificationAgent module not available: {e}",
                        "error_type": "ImportError",
                        "session_id": session_id
                    }
                except Exception as e:
                    logger.error(f"[ValidationTool] Advanced specs discovery error: {e}")
                    return {
                        "success": False,
                        "product_type": expected_product_type,
                        "hitl_response": "yes",
                        "validation_bypassed": True,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "session_id": session_id
                    }

            # ═══════════════════════════════════════════════════════════════════════════
            # "YES" WITHOUT CONTEXT: Try to retrieve from session cache
            # ═══════════════════════════════════════════════════════════════════════════
            elif is_hitl_yes_response and not expected_product_type:
                logger.info("[ValidationTool] 'YES' response without expected_product_type - checking session cache")

                # Try to retrieve product_type from session context cache
                cached_context = _get_session_context(session_id)
                if cached_context and cached_context.get("product_type"):
                    cached_product_type = cached_context["product_type"]
                    logger.info(f"[ValidationTool] ✓ Retrieved product_type from session cache: {cached_product_type}")

                    # Now proceed with AdvancedSpecificationAgent using cached product_type
                    logger.info("=" * 70)
                    logger.info("[ValidationTool] ✓ USER SAID 'YES' - BYPASSING VALIDATION (using cached context)")
                    logger.info("[ValidationTool] → Triggering AdvancedSpecificationAgent ONLY")
                    logger.info(f"   Product Type: {cached_product_type}")
                    logger.info(f"   Session: {session_id}")
                    logger.info("=" * 70)

                    try:
                        from search.advanced_specification_agent import AdvancedSpecificationAgent

                        adv_agent = AdvancedSpecificationAgent()
                        advanced_specs_result = adv_agent.discover(
                            product_type=cached_product_type,
                            session_id=session_id
                        )

                        if advanced_specs_result.get("success"):
                            num_specs = advanced_specs_result.get("total_unique_specifications", 0)
                            logger.info(f"[ValidationTool] ✓ Advanced specs discovered: {num_specs} parameters")

                            return {
                                "success": True,
                                "product_type": cached_product_type,
                                "schema": cached_context.get("schema", {}),  # BUG FIX: preserve schema from Turn 1
                                "provided_requirements": cached_context.get("provided_requirements", {}),  # BUG FIX: preserve requirements
                                "is_valid": True,
                                "hitl_response": "yes",
                                "validation_bypassed": True,
                                "advanced_specs_info": {
                                    "triggered": True,
                                    "success": True,
                                    "unique_specifications": advanced_specs_result.get("unique_specifications", []),
                                    "total_unique_specifications": num_specs,
                                    "existing_specifications_filtered": advanced_specs_result.get("existing_specifications_filtered", 0),
                                    "discovery_successful": advanced_specs_result.get("discovery_successful", False),
                                    "fallback_used": advanced_specs_result.get("fallback_used", False)
                                },
                                "session_id": session_id,
                                "message": f"Discovered {num_specs} advanced specifications for {cached_product_type}"
                            }
                        else:
                            logger.warning(f"[ValidationTool] ⚠ Advanced specs discovery failed: {advanced_specs_result.get('error')}")
                            return {
                                "success": False,
                                "product_type": cached_product_type,
                                "hitl_response": "yes",
                                "validation_bypassed": True,
                                "error": advanced_specs_result.get("error", "Advanced specification discovery failed"),
                                "error_type": "AdvancedSpecsError",
                                "session_id": session_id
                            }
                    except Exception as e:
                        logger.error(f"[ValidationTool] Advanced specs discovery error: {e}")
                        return {
                            "success": False,
                            "product_type": cached_product_type,
                            "hitl_response": "yes",
                            "validation_bypassed": True,
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "session_id": session_id
                        }
                else:
                    # No cached context - return error
                    logger.warning("[ValidationTool] 'YES' response without context (no cached session context)")
                    result.update({
                        "success": False,
                        "error": "Please provide the product type context. Your 'YES' response needs to be part of an ongoing conversation with product context.",
                        "error_type": "MissingContextError",
                        "hint": "The calling API should pass 'expected_product_type' from the previous validation result"
                    })
                    return result

            # ═══════════════════════════════════════════════════════════════════════════
            # "NO" RESPONSE: Continue with validation to collect more fields
            # ═══════════════════════════════════════════════════════════════════════════
            elif is_hitl_no_response and expected_product_type:
                logger.info("=" * 70)
                logger.info("[ValidationTool] USER SAID 'NO' - User wants to add missing specifications")
                logger.info(f"   Product Type: {expected_product_type}")
                logger.info("=" * 70)

                # Use expected product type but continue with validation flow
                product_type = expected_product_type
                extract_result = {"product_type": expected_product_type, "specifications": {}}

            elif is_hitl_no_response and not expected_product_type:
                logger.info("[ValidationTool] 'NO' response without expected_product_type - checking session cache")

                # Try to retrieve product_type from session context cache
                cached_context = _get_session_context(session_id)
                if cached_context and cached_context.get("product_type"):
                    cached_product_type = cached_context["product_type"]
                    logger.info(f"[ValidationTool] ✓ Retrieved product_type from session cache: {cached_product_type}")
                    logger.info("=" * 70)
                    logger.info("[ValidationTool] USER SAID 'NO' - User wants to add missing specifications (using cached context)")
                    logger.info(f"   Product Type: {cached_product_type}")
                    logger.info("=" * 70)

                    # Use cached product type and continue with validation flow
                    product_type = cached_product_type
                    extract_result = {"product_type": cached_product_type, "specifications": {}}
                else:
                    logger.warning("[ValidationTool] 'NO' response without context (no cached session context)")
                    result.update({
                        "success": False,
                        "error": "Please provide the product type context. Your 'NO' response needs to be part of an ongoing conversation with product context.",
                        "error_type": "MissingContextError",
                        "hint": "The calling API should pass 'expected_product_type' from the previous validation result"
                    })
                    return result

            else:
                # =================================================================
                # STEP 1.1: EXTRACT PRODUCT TYPE (normal flow)
                # =================================================================
                logger.info("[ValidationTool] Step 1.1: Extracting product type")

                extract_result = extract_requirements_tool.invoke({
                    "user_input": user_input
                })

                product_type = extract_result.get("product_type") or expected_product_type or ""

            # =================================================================
            # PRODUCT TYPE NORMALIZATION: Expand bare measurement words
            # LLMs sometimes return "Level", "Pressure", "Flow", "Temperature"
            # instead of "Level Transmitter", "Pressure Transmitter", etc.
            # We check for this and expand to the proper full product type.
            # =================================================================
            _BARE_MEASUREMENT_FIX = {
                "level": "Level Transmitter",
                "pressure": "Pressure Transmitter",
                "flow": "Flow Meter",  # Default to Meter, but will adjust below
                "temperature": "Temperature Transmitter",
                "differential pressure": "Differential Pressure Transmitter",
                "dp": "Differential Pressure Transmitter",
                "relative pressure": "Pressure Transmitter",
                "gauge pressure": "Pressure Transmitter",
                "absolute pressure": "Pressure Transmitter",
                "density": "Density Meter",
                "viscosity": "Viscosity Meter",
            }
            if product_type and product_type.strip().lower() in _BARE_MEASUREMENT_FIX:
                original_pt = product_type
                product_type = _BARE_MEASUREMENT_FIX[product_type.strip().lower()]
                
                # Contextual adjustment for Flow
                if original_pt.lower() == "flow":
                    input_lower = user_input.lower()
                    if "transmitter" in input_lower and "meter" not in input_lower:
                        product_type = "Flow Transmitter"
                    elif "switch" in input_lower:
                        product_type = "Flow Switch"
                
                # Contextual adjustment for Level
                elif original_pt.lower() == "level":
                    if "switch" in user_input.lower():
                        product_type = "Level Switch"
                    elif "gauge" in user_input.lower():
                        product_type = "Level Gauge"

                logger.warning(
                    "[ValidationTool] ⚠ Bare measurement word detected — normalized '%s' → '%s' (based on context)",
                    original_pt, product_type
                )
            
            # Handle case when LLM extraction failed (quota exceeded, API error, etc.)
            if not product_type:
                logger.error("[ValidationTool] ✗ Product type extraction failed - no product type detected")
                logger.error("[ValidationTool] ✗ User input was: %s", user_input[:200])

                # Check if input looks like requirements or just noise
                has_product_words = any(word in user_input.lower() for word in [
                    "transmitter", "sensor", "meter", "gauge", "valve", "pump",
                    "analyzer", "controller", "switch", "indicator", "recorder"
                ])

                if has_product_words:
                    error_msg = (
                        "Could not determine product type from your input. "
                        "The LLM service may be temporarily unavailable. Please try again."
                    )
                else:
                    error_msg = (
                        "No product type detected in your input. "
                        "Please describe what product you need (e.g., 'I need a pressure transmitter with 4-20mA output')."
                    )

                result.update({
                    "success": False,
                    "error": error_msg,
                    "error_type": "ProductTypeExtractionError",
                    "user_input_preview": user_input[:100] + "..." if len(user_input) > 100 else user_input
                })
                return result
            
            logger.info("[ValidationTool] ✓ Detected product type: %s", product_type)

            # Validate against expected type if provided
            if expected_product_type and product_type and product_type.lower() != expected_product_type.lower():
                logger.warning(
                    "[ValidationTool] ⚠ Product type mismatch - Expected: %s, Detected: %s",
                    expected_product_type,
                    product_type
                )

            # =================================================================
            # STEP 1.2: LOAD OR GENERATE SCHEMA
            # =================================================================
            logger.info("[ValidationTool] Step 1.2: Loading/generating schema")

            schema_result = load_schema_tool.invoke({
                "product_type": product_type,
                "enable_ppi": self.enable_ppi
            })

            schema = schema_result.get("schema", {})
            schema_source = schema_result.get("source", "unknown")
            ppi_used = schema_result.get("ppi_used", False)
            from_database = schema_result.get("from_database", False)

            # Log schema source
            if from_database:
                logger.info("[ValidationTool] ✓ Schema loaded from Azure Blob Storage")
            elif ppi_used:
                logger.info("[ValidationTool] ✓ Schema generated via PPI workflow")
            else:
                logger.warning("[ValidationTool] ⚠ Using default schema (fallback)")

            # =================================================================
            # STEP 1.2.1: ENRICH SCHEMA WITH STANDARDS RAG (FIELD VALUES + STANDARDS SECTION)
            # Uses both tools.standards_enrichment_tool and agentic.standards_rag_enrichment
            # =================================================================
            if enable_standards_enrichment is None:
                enable_standards_enrichment = self.enable_standards_enrichment

            standards_info = None
            enrichment_result = None
            standards_rag_invoked = False
            standards_rag_invocation_time = None

            if enable_standards_enrichment:
                try:
                    # ╔══════════════════════════════════════════════════════════════╗
                    # ║           🔵 STANDARDS RAG INVOCATION STARTING 🔵            ║
                    # ╚══════════════════════════════════════════════════════════════╝
                    import datetime
                    standards_rag_invocation_time = datetime.datetime.now().isoformat()
                    logger.info("="*70)
                    logger.info("STANDARDS RAG INVOKED")
                    logger.info(f"   Timestamp: {standards_rag_invocation_time}")
                    logger.info(f"   Product Type: {product_type}")
                    logger.info(f"   Session: {session_id}")
                    logger.info("="*70)
                
                    standards_rag_invoked = True
                    logger.info("[ValidationTool] Step 1.2.1: Enriching schema with Standards RAG")

                    # FIX #5: Import with fallback for missing modules
                    try:
                        from common.tools.standards_enrichment_tool import (
                            get_applicable_standards,
                            populate_schema_fields_from_standards
                        )
                    except ImportError as e:
                        logger.warning(f"[FIX5] Failed to import from common.tools.standards_enrichment_tool: {e}")
                        from common.tools.standards_enrichment_tool import get_applicable_standards
                        def populate_schema_fields_from_standards(product_type, schema):
                            """Fallback for missing function"""
                            logger.info("[FIX5] Using fallback for populate_schema_fields_from_standards")
                            return schema

                    try:
                        from common.rag.standards import (
                            enrich_identified_items_with_standards,
                            is_standards_related_question
                        )
                    except ImportError as e:
                        logger.warning(f"[FIX5] Failed to import from standards_rag_enrichment: {e}")
                        def enrich_identified_items_with_standards(*args, **kwargs):
                            """Fallback for missing function"""
                            logger.info("[FIX5] Using fallback for enrich_identified_items_with_standards")
                            return {"success": False}

                        def is_standards_related_question(*args, **kwargs):
                            """Fallback for missing function"""
                            return False

                    # NOTE: Deep Agent module removed (always returned 0/0 fields)
                    # Use Standards RAG enrichment instead, which is working and faster

                    # ╔══════════════════════════════════════════════════════════════════════════╗
                    # ║  [FIX #A1] SESSION-LEVEL DEDUPLICATION                               ║
                    # ║  Check if product was already enriched in this session               ║
                    # ║  If yes: reuse cached result (saves 50-70 seconds!)                  ║
                    # ║  If no: do enrichment once and cache for reuse                       ║
                    # ╚══════════════════════════════════════════════════════════════════════╝

                    # Check session cache first
                    cached_enrichment = _get_session_enrichment(product_type, session_id)

                    if cached_enrichment:
                        # ✅ SESSION CACHE HIT - Reuse previous enrichment result
                        logger.info(f"[FIX #A1] 🎯 SESSION CACHE HIT for {product_type} - Reusing enrichment (saves 50-70s!)")
                        standards_info = cached_enrichment.get('standards_info')
                        enrichment_result = cached_enrichment.get('enrichment_result')
                        schema = cached_enrichment.get('schema', schema)

                        # Apply cached schema updates
                        if cached_enrichment.get('standards_section'):
                            schema['standards'] = cached_enrichment['standards_section']

                        print(f"\n{'='*70}")
                        print(f"🎯 [FIX #A1] SESSION CACHE HIT - Skipping redundant Standards RAG call")
                        print(f"   Product: {product_type}")
                        print(f"   Saves: ~50-70 seconds!")
                        print(f"{'='*70}\n")
                    else:
                        # ❌ SESSION CACHE MISS - Need to do enrichment
                        logger.info(f"[FIX #A1] 🔴 SESSION CACHE MISS for {product_type} - Running enrichment")

                        # Step 1.2.1a: POPULATE field values from Standards RAG (using tools module)
                        if not schema.get("_standards_population"):
                            logger.info("[ValidationTool] Step 1.2.1a: Populating schema field values from standards")
                            schema = populate_schema_fields_from_standards(product_type, schema)
                            fields_populated = schema.get("_standards_population", {}).get("fields_populated", 0)
                            logger.info(f"[ValidationTool] ✓ Populated {fields_populated} fields with standards values")
                        else:
                            logger.info("[ValidationTool] ✓ Schema already has standards-populated field values")

                        # ╔══════════════════════════════════════════════════════════════════════════╗
                        # ║  [FIX #4] APPLY COMPREHENSIVE STANDARDS DEFAULTS (60+ FIELDS)           ║
                        # ║  Uses schema_field_extractor.py with 50-80 defaults per product type    ║
                        # ║  Fills gaps with IEC, ISO, ASME, NAMUR standards-referenced values      ║
                        # ╚══════════════════════════════════════════════════════════════════════════╝
                        try:
                            from agentic.deep_agent.schema.generation.field_extractor import extract_schema_field_values_from_standards

                            logger.info("[FIX #4] Applying comprehensive standards defaults (target: 60+ fields)")
                            fields_before = schema.get("_standards_population", {}).get("fields_populated", 0)

                            # Apply comprehensive defaults (50-80 fields per product type)
                            schema = extract_schema_field_values_from_standards(product_type, schema)

                            fields_after = schema.get("_schema_field_extraction", {}).get("fields_populated", 0)
                            total_fields = schema.get("_schema_field_extraction", {}).get("fields_total", 0)

                            logger.info(f"[FIX #4] ✓ Standards defaults applied: {fields_before} → {fields_before + fields_after} fields")
                            logger.info(f"[FIX #4] ✓ Total schema fields: {total_fields}, Populated: {fields_before + fields_after}")

                            logger.info("="*70)
                            logger.info("[FIX #4] COMPREHENSIVE STANDARDS DEFAULTS APPLIED")
                            logger.info(f"   Before: {fields_before} fields")
                            logger.info(f"   After: {fields_before + fields_after} fields (target: 60+)")
                            logger.info(f"   Source: schema_field_extractor.py (IEC, ISO, ASME, NAMUR standards)")
                            logger.info("="*70)

                        except ImportError as e:
                            logger.warning(f"[FIX #4] schema_field_extractor not available: {e}")
                        except Exception as e:
                            logger.warning(f"[FIX #4] Standards defaults application failed: {e}")

                        # ╔══════════════════════════════════════════════════════════════════════════╗
                        # ║  [FIX #5] APPLY TEMPLATE SPECIFICATIONS (62+ FIELDS PER PRODUCT)        ║
                        # ║  Uses specification_templates.py with structured spec definitions       ║
                        # ║  Provides comprehensive coverage with importance levels & typical values║
                        # ╚══════════════════════════════════════════════════════════════════════════╝
                        try:
                            from agentic.deep_agent.specifications.templates.templates import (
                                get_all_specs_for_product_type,
                                export_template_as_dict
                            )

                            # Get template specifications for this product type
                            template_specs = get_all_specs_for_product_type(product_type)

                            if template_specs:
                                logger.info(f"[FIX #5] Applying template specifications ({len(template_specs)} specs)")

                                # Create or update template_specifications section in schema
                                if "template_specifications" not in schema:
                                    schema["template_specifications"] = {}

                                # Add template spec values as defaults for empty fields
                                specs_added = 0
                                for spec_key, spec_def in template_specs.items():
                                    # Check if this field is already populated
                                    field_exists = False
                                    for section in ["mandatory", "optional", "Performance", "Electrical", "Mechanical"]:
                                        if section in schema and spec_key in schema[section]:
                                            field_exists = True
                                            break

                                    if not field_exists and spec_def.typical_value:
                                        # Add to template_specifications section
                                        schema["template_specifications"][spec_key] = {
                                            "value": str(spec_def.typical_value),
                                            "unit": spec_def.unit or "",
                                            "category": spec_def.category,
                                            "description": spec_def.description,
                                            "importance": spec_def.importance.name if hasattr(spec_def.importance, 'name') else "OPTIONAL",
                                            "source": "template_specification"
                                        }
                                        specs_added += 1

                                logger.info(f"[FIX #5] ✓ Added {specs_added} template specifications")
                                schema["_template_specs_added"] = specs_added

                                logger.info("="*70)
                                logger.info("[FIX #5] TEMPLATE SPECIFICATIONS APPLIED")
                                logger.info(f"   Template specs available: {len(template_specs)}")
                                logger.info(f"   New specs added: {specs_added}")
                                logger.info(f"   Source: specification_templates.py")
                                logger.info("="*70)
                            else:
                                logger.debug(f"[FIX #5] No template found for product type: {product_type}")

                        except ImportError as e:
                            logger.debug(f"[FIX #5] specification_templates not available: {e}")
                        except Exception as e:
                            logger.warning(f"[FIX #5] Template specifications application failed: {e}")

                        # Step 1.2.1b: GET applicable standards for standards section (using tools module)
                        standards_info = get_applicable_standards(product_type, top_k=5)

                        enrichment_result = None
                        standards_section = None

                        if standards_info.get('success'):
                            # Add standards to schema if not already present
                            if 'standards' not in schema:
                                standards_section = {
                                    'applicable_standards': standards_info.get('applicable_standards', []),
                                    'certifications': standards_info.get('certifications', []),
                                    'safety_requirements': standards_info.get('safety_requirements', {}),
                                    'calibration_standards': standards_info.get('calibration_standards', {}),
                                    'environmental_requirements': standards_info.get('environmental_requirements', {}),
                                    'communication_protocols': standards_info.get('communication_protocols', []),
                                    'sources': standards_info.get('sources', []),
                                    'confidence': standards_info.get('confidence', 0.0)
                                }
                                schema['standards'] = standards_section

                            num_standards = len(standards_info.get('applicable_standards', []))
                            num_certs = len(standards_info.get('certifications', []))
                            logger.info(f"[ValidationTool] ✓ Standards enriched: {num_standards} standards, {num_certs} certifications")

                            # Log success indicator
                            logger.info("="*70)
                            logger.info("STANDARDS RAG COMPLETED SUCCESSFULLY")
                            logger.info(f"   Standards Found: {num_standards}")
                            logger.info(f"   Certifications Found: {num_certs}")
                            logger.info("="*70)
                        else:
                            logger.warning(f"[ValidationTool] Standards RAG returned no results: {standards_info.get('error', 'Unknown')}")

                        # Step 1.2.1c: ENRICH with normalized category using agentic module
                        # This provides structured standards_info with normalized_category for the product type
                        try:
                            # Create a mock item representing this product type for enrichment
                            product_item = [{
                                "name": product_type,
                                "category": product_type,
                                "specifications": schema.get("mandatory", {})
                            }]

                            enriched_items = enrich_identified_items_with_standards(
                                items=product_item,
                                product_type=product_type,
                                top_k=3
                            )

                            if enriched_items and len(enriched_items) > 0:
                                enrichment_result = enriched_items[0].get("standards_info", {})

                                # Add normalized category to schema if available
                                if enriched_items[0].get("normalized_category"):
                                    schema["normalized_category"] = enriched_items[0]["normalized_category"]
                                    logger.info(f"[ValidationTool] ✓ Normalized category: {schema['normalized_category']}")

                                # Merge additional enrichment info into standards section
                                if enrichment_result.get("enrichment_status") == "success":
                                    if "standards" in schema:
                                        # Merge communication protocols if new ones found
                                        existing_protocols = set(schema["standards"].get("communication_protocols", []))
                                        new_protocols = set(enrichment_result.get("communication_protocols", []))
                                        schema["standards"]["communication_protocols"] = list(existing_protocols | new_protocols)

                                        # Merge certifications if new ones found
                                        existing_certs = set(schema["standards"].get("certifications", []))
                                        new_certs = set(enrichment_result.get("certifications", []))
                                        schema["standards"]["certifications"] = list(existing_certs | new_certs)

                                    logger.info("[ValidationTool] ✓ Additional enrichment merged from standards_rag_enrichment")

                        except Exception as enrich_err:
                            logger.debug(f"[ValidationTool] Additional enrichment skipped: {enrich_err}")
                            # Non-critical - continue without additional enrichment

                        # ╔══════════════════════════════════════════════════════════════╗
                        # ║  [FIX #A1] CACHE ENRICHMENT RESULT FOR SESSION               ║
                        # ║  Store for reuse in subsequent validation/enrichment calls   ║
                        # ╚══════════════════════════════════════════════════════════════╝
                        enrichment_cache_data = {
                            'standards_info': standards_info,
                            'enrichment_result': enrichment_result,
                            'schema': schema,
                            'standards_section': schema.get('standards')
                        }
                        _cache_session_enrichment(product_type, enrichment_cache_data, session_id)
                        logger.info(f"[FIX #A1] 💾 Cached enrichment result for {product_type} (Session: {session_id})")

                    # ╔══════════════════════════════════════════════════════════════════════════╗
                    # ║  [NEW] DEEP AGENT SCHEMA POPULATION WITH FAILURE MEMORY                   ║
                    # ║  Uses SchemaGenerationDeepAgent with learning from past failures          ║
                    # ║  - Parallel multi-source generation                                       ║
                    # ║  - Adaptive prompts based on history                                      ║
                    # ║  - Recovery from failures using learned patterns                          ║
                    # ╚══════════════════════════════════════════════════════════════════════════╝
                    logger.info("[ValidationTool] Step 1.2.1d: Deep Agent schema population starting")
                    logger.info("="*70)
                    logger.info("[DEEP AGENT] SCHEMA POPULATION STARTING")
                    logger.info(f"   Product: {product_type}")
                    logger.info("="*70)

                    try:
                        from agentic.deep_agent.schema.populator_legacy import (
                            populate_schema_with_deep_agent,
                            predict_population_success
                        )

                        # Predict success rate before attempting
                        prediction = predict_population_success(product_type)
                        logger.info(f"[ValidationTool] Deep Agent risk level: {prediction.get('risk_level', 'unknown')}")

                        # Run deep agent population
                        schema, population_stats = populate_schema_with_deep_agent(
                            product_type=product_type,
                            schema=schema,
                            session_id=session_id,
                            use_memory=True
                        )

                        fields_populated = population_stats.get("fields_populated", 0)
                        total_fields = population_stats.get("total_fields", 0)
                        sources = population_stats.get("sources_used", [])

                        logger.info(f"[ValidationTool] ✓ Deep Agent populated {fields_populated}/{total_fields} fields")
                        logger.info(f"[ValidationTool] ✓ Sources: {', '.join(sources) if sources else 'none'}")

                        print("\n" + "="*70)
                        print("[DEEP_AGENT] SCHEMA POPULATION COMPLETED")
                        print(f"   Fields Populated: {fields_populated}/{total_fields}")
                        print(f"   Sources Used: {len(sources)}")
                        print("="*70 + "\n")

                    except ImportError as e:
                        logger.warning(f"[FIX5] Module not found - deep_agent_schema_populator: {e}")
                        logger.info("[FIX5] Using fallback for populate_schema_with_deep_agent")
                        print("\n" + "="*70)
                        print("[DEEP_AGENT] SCHEMA POPULATION COMPLETED (fallback)")
                        print("   Fields Populated: 0/0")
                        print("   Sources Used: 0")
                        print("="*70 + "\n")
                    except Exception as e:
                        logger.warning(f"[ValidationTool] Deep Agent population failed (non-critical): {e}")
                        print("\n" + "="*70)
                        print(f"[DEEP_AGENT] POPULATION FAILED: {str(e)[:50]}")
                        print("="*70 + "\n")

                except Exception as standards_error:
                    logger.error(f"[ValidationTool] Standards enrichment failed (non-critical): {standards_error}", exc_info=True)
                    # Continue without standards - this is non-critical

            # NOTE: Strategy RAG is NOT applied during initial validation.
            # Strategy-based vendor filtering is applied during Final Vendor Analysis
            # in vendor_analysis_tool.py to filter/prioritize vendors before analysis.
            strategy_info = None  # Placeholder for result compatibility


            # =================================================================
            # STEP 1.3: VALIDATE REQUIREMENTS
            # =================================================================
            logger.info("[ValidationTool] Step 1.3: Validating requirements")

            validation_result = validate_requirements_tool.invoke({
                "user_input": user_input,
                "product_type": product_type,
                "product_schema": schema
            })

            provided_requirements = validation_result.get("provided_requirements", {})
            
            # Helper to flatten nested JSON fields to simple strings for UI rendering
            def flatten_field_value(field_data):
                if isinstance(field_data, dict):
                    if "value" in field_data:
                        val = field_data.get("value", "")
                        unit = field_data.get("unit", "")
                        return f"{val} {unit}".strip()
                    elif "min" in field_data and "max" in field_data:
                        unit = field_data.get("unit", "")
                        return f"{field_data['min']} - {field_data['max']} {unit}".strip()
                    else:
                        # Fallback for unexpected dictionary structures
                        return ", ".join(f"{k}: {v}" for k, v in field_data.items() if str(v).strip())
                elif isinstance(field_data, list):
                    return ", ".join(str(item) for item in field_data)
                return field_data
                
            for k, v in provided_requirements.items():
                provided_requirements[k] = flatten_field_value(v)

            # ══════════════════════════════════════════════════════════════════
            # FIX #5 (CRITICAL): Write user-provided values back into the schema
            # field objects so the frontend sidebar can display them.
            #
            # Background: validate_requirements_tool extracts values from the
            # user's prose input (e.g. "accuracy": "±0.1%") into provided_requirements.
            # The LeftSidebar renders schema.mandatoryRequirements / optionalRequirements
            # directly — it does NOT read provided_requirements. So without this
            # merge the user's values are silently discarded and all fields show
            # "Not specified".
            # ══════════════════════════════════════════════════════════════════
            def _normalize_for_match(s: str) -> str:
                """Lowercase snake_case / camelCase key for fuzzy matching."""
                import re
                s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
                return s.lower().replace(' ', '_').replace('-', '_')

            def _merge_provided_into_schema(schema_dict: dict, provided: dict) -> None:
                """
                Walk schema sections and inject user-provided values into
                field objects that still show 'Not specified' or an empty value.
                User-provided values have the highest priority.
                """
                if not provided or not schema_dict:
                    return
                # Build a normalized lookup: norm_key -> flat_value
                norm_provided = {_normalize_for_match(k): str(v) for k, v in provided.items() if v}

                def _apply_to_section(section: dict) -> None:
                    for field_key, field_val in section.items():
                        if field_key.startswith('_'):
                            continue
                        norm_field = _normalize_for_match(field_key)
                        if isinstance(field_val, dict) and 'value' in field_val:
                            # Only overwrite if user provided a value
                            if norm_field in norm_provided:
                                field_val['value'] = norm_provided[norm_field]
                                field_val['source'] = 'user_provided'
                        elif isinstance(field_val, dict):
                            # Nested section one level deep
                            _apply_to_section(field_val)

                for section_name, section_val in schema_dict.items():
                    if section_name.startswith('_'):
                        continue
                    if isinstance(section_val, dict):
                        _apply_to_section(section_val)

            _merge_provided_into_schema(schema, provided_requirements)
            logger.info(
                "[ValidationTool] [FIX #5] Merged %d provided_requirements into schema field objects",
                len(provided_requirements)
            )

            missing_fields = validation_result.get("missing_fields", [])
            is_valid = validation_result.get("is_valid", False)

            # =================================================================
            # FIX: HANDLE YES/NO HITL CONTINUE DECISION
            # =================================================================
            normalized_input_for_decision = user_input.lower().strip()
            advanced_specs_result = None  # Will hold AdvancedSpecificationAgent result

            if normalized_input_for_decision in ["yes", "y"]:
                logger.info("[ValidationTool] User responded YES. Continuing without missing specs.")
                is_valid = True
                missing_fields = []

                # ╔══════════════════════════════════════════════════════════════════════════╗
                # ║  TRIGGER ADVANCED SPECIFICATION AGENT ON "YES" RESPONSE                   ║
                # ║  When user confirms to proceed, discover additional advanced parameters   ║
                # ║  that can enhance the product search beyond basic schema fields           ║
                # ╚══════════════════════════════════════════════════════════════════════════╝
                try:
                    from search.advanced_specification_agent import AdvancedSpecificationAgent

                    logger.info("=" * 70)
                    logger.info("[ValidationTool] TRIGGERING ADVANCED SPECIFICATION AGENT")
                    logger.info(f"   Product Type: {product_type}")
                    logger.info(f"   Session: {session_id}")
                    logger.info("=" * 70)

                    adv_agent = AdvancedSpecificationAgent()
                    advanced_specs_result = adv_agent.discover_with_filtering(
                        product_type=product_type,
                        existing_schema=schema,
                        session_id=session_id
                    )

                    if advanced_specs_result.get("success"):
                        num_specs = advanced_specs_result.get("total_unique_specifications", 0)
                        logger.info(f"[ValidationTool] ✓ Advanced specs discovered: {num_specs} parameters")
                    else:
                        logger.warning(f"[ValidationTool] ⚠ Advanced specs discovery failed: {advanced_specs_result.get('error')}")

                except ImportError as e:
                    logger.warning(f"[ValidationTool] AdvancedSpecificationAgent not available: {e}")
                except Exception as e:
                    logger.error(f"[ValidationTool] Advanced specs discovery error (non-critical): {e}")
            elif normalized_input_for_decision.startswith("no ") or normalized_input_for_decision in ["no", "n"]:
                logger.info("[ValidationTool] User responded NO or provided updates.")
                is_valid = False
                if normalized_input_for_decision not in ["no", "n"]:
                    provided_requirements = self.update_specifications(provided_requirements, user_input)
            else:
                user_specs = extract_result.get("specifications", {})
                if not user_specs and not provided_requirements:
                    logger.warning("[ValidationTool] No specific parameters provided in user input. Forcing HITL prompt.")
                    is_valid = False
                    
                    if not missing_fields:
                        schema_mandatory = schema.get("mandatory", [])
                        if schema_mandatory:
                            missing_fields = schema_mandatory[:5]
                        else:
                            mandatory_reqs = schema.get("mandatoryRequirements", {})
                            if isinstance(mandatory_reqs, dict):
                                missing_fields = list(mandatory_reqs.keys())[:5]
                            elif isinstance(mandatory_reqs, list):
                                missing_fields = mandatory_reqs[:5]
                            else:
                                properties = schema.get("properties", {})
                                if properties:
                                    mandatory = [
                                        k for k, v in properties.items() 
                                        if isinstance(v, dict) and (v.get("required") == True or str(v.get("importance")).lower() == "high")
                                    ]
                                    if not mandatory:
                                        mandatory = list(properties.keys())
                                    missing_fields = mandatory[:5]

            # Re-check missing_fields to generate the exact required HITL message
            hitl_message = None
            if not is_valid:
                found_str = ", ".join(f"{k}: {v}" for k, v in provided_requirements.items()) if provided_requirements else "None"
                missing_str = ", ".join(missing_fields) if missing_fields else "required specifications"
                hitl_message = f'These are the specifications i found ({found_str}) and these are the missing specifications and values ({missing_str}), so do you want to add those missing values and specifications or shall i continue without adding the missing specifications and values tell me "YES" or "NO"'

            # =================================================================
            # FIX: PROPAGATE REFINED PRODUCT TYPE
            # If validate_requirements_tool refined the product type (e.g.,
            # "Industrial Instrument" -> "flow meter"), use the refined type
            # for all subsequent operations including vendor analysis.
            # =================================================================
            refined_product_type = validation_result.get("product_type")
            product_type_was_refined = validation_result.get("product_type_refined", False)

            if refined_product_type and refined_product_type != product_type:
                logger.info(
                    "[ValidationTool] FIX: Product type REFINED by validation: '%s' -> '%s'",
                    product_type, refined_product_type
                )
                product_type = refined_product_type
                product_type_was_refined = True
            elif product_type_was_refined:
                logger.info(
                    "[ValidationTool] Product type was refined (already using: %s)",
                    product_type
                )

            # Log validation results
            if is_valid:
                logger.info("[ValidationTool] ✓ All mandatory fields provided")
            else:
                logger.info("[ValidationTool] ⚠ Missing mandatory fields: %s", missing_fields)

            # =================================================================
            # FIX: SANITIZE SCHEMA FOR UI RENDERING
            # Flatten dicts with 'value' to strings so the UI doesn't render [object Object]
            # =================================================================
            for section_key, section_data in schema.items():
                if isinstance(section_data, dict) and not section_key.startswith("_") and section_key not in ["properties", "standards"]:
                    for field_key, field_val in section_data.items():
                        if isinstance(field_val, dict) and "value" in field_val:
                            val_raw = field_val.get("value", "")
                            unit_raw = field_val.get("unit", "")
                            val_str = str(val_raw) if val_raw is not None else ""
                            unit_str = str(unit_raw) if unit_raw is not None else ""
                            section_data[field_key] = f"{val_str} {unit_str}".strip()

            # =================================================================
            # BUILD RESULT
            # =================================================================
            result.update({
                "success": True,
                "product_type": product_type,  # This is now the refined product type if refinement occurred
                "product_type_refined": product_type_was_refined,  # FIX: Flag indicating refinement
                "original_product_type": extract_result.get("product_type"),  # FIX: Original detected type
                "normalized_category": schema.get("normalized_category"),  # From standards_rag_enrichment
                "schema": schema,
                "provided_requirements": provided_requirements,
                "missing_fields": missing_fields,
                "optional_fields": validation_result.get("optional_fields", []),
                "is_valid": is_valid,
                "hitl_message": hitl_message,
                "ppi_workflow_used": ppi_used,
                "schema_source": schema_source,
                "from_database": from_database,
                # ══════════════════════════════════════════════════════════════
                # RAG INVOCATION TRACKING - Visible in browser Network tab
                # ══════════════════════════════════════════════════════════════
                "rag_invocations": {
                    "standards_rag": {
                        "invoked": standards_rag_invoked,
                        "invocation_time": standards_rag_invocation_time,
                        "success": standards_info.get('success', False) if standards_info else False,
                        "product_type": product_type,
                        "results_count": len(standards_info.get('applicable_standards', [])) if standards_info else 0
                    },
                    "strategy_rag": {
                        "invoked": False,
                        "note": "Strategy RAG is applied in vendor_analysis_tool.py, not during validation"
                    }
                },
                # Standards RAG enrichment results (combined from tools and agentic modules)
                "standards_info": {
                    "applicable_standards": standards_info.get('applicable_standards', []) if standards_info else [],
                    "certifications": standards_info.get('certifications', []) if standards_info else [],
                    "communication_protocols": standards_info.get('communication_protocols', []) if standards_info else [],
                    "safety_requirements": standards_info.get('safety_requirements', {}) if standards_info else {},
                    "environmental_requirements": standards_info.get('environmental_requirements', {}) if standards_info else {},
                    "confidence": standards_info.get('confidence', 0.0) if standards_info else 0.0,
                    "sources": standards_info.get('sources', []) if standards_info else [],
                    "enrichment_success": standards_info.get('success', False) if standards_info else False,
                    # Additional data from standards_rag_enrichment module
                    "additional_enrichment": {
                        "status": enrichment_result.get("enrichment_status") if enrichment_result else "not_performed",
                        "merged_protocols": enrichment_result.get("communication_protocols", []) if enrichment_result else [],
                        "merged_certifications": enrichment_result.get("certifications", []) if enrichment_result else []
                    } if enrichment_result else None
                },
                # Strategy RAG enrichment results (TRUE RAG with vector store)
                "strategy_info": {
                    "preferred_vendors": strategy_info.get('preferred_vendors', []) if strategy_info else [],
                    "forbidden_vendors": strategy_info.get('forbidden_vendors', []) if strategy_info else [],
                    "neutral_vendors": strategy_info.get('neutral_vendors', []) if strategy_info else [],
                    "procurement_priorities": strategy_info.get('procurement_priorities', {}) if strategy_info else {},
                    "strategy_notes": strategy_info.get('strategy_notes', '') if strategy_info else '',
                    "confidence": strategy_info.get('confidence', 0.0) if strategy_info else 0.0,
                    "rag_type": strategy_info.get('rag_type', 'unknown') if strategy_info else 'not_performed',  # 'true_rag' or 'llm_inference'
                    "sources_used": strategy_info.get('sources_used', []) if strategy_info else [],
                    "enrichment_success": strategy_info.get('success', False) if strategy_info else False
                },
                # ══════════════════════════════════════════════════════════════
                # DEEP AGENT SCHEMA POPULATION - Section-based specifications
                # ══════════════════════════════════════════════════════════════
                "deep_agent_info": {
                    "population_performed": schema.get("_deep_agent_population") is not None,
                    "fields_populated": schema.get("_deep_agent_population", {}).get("fields_populated", 0),
                    "total_fields": schema.get("_deep_agent_population", {}).get("total_fields", 0),
                    "sections_processed": schema.get("_deep_agent_population", {}).get("sections_processed", 0),
                    "sources_used": schema.get("_deep_agent_population", {}).get("sources_used", []),
                    "processing_time_ms": schema.get("_deep_agent_population", {}).get("processing_time_ms", 0),
                    # Section-based field values for UI display
                    "sections": schema.get("_deep_agent_sections", {})
                },
                # ══════════════════════════════════════════════════════════════
                # [FIX #4 + #5] COMPREHENSIVE SCHEMA POPULATION INFO (60+ FIELDS)
                # Tracks fields from all sources for visibility
                # ══════════════════════════════════════════════════════════════
                "schema_population_info": {
                    # FIX #3: Standards RAG extraction (regex-based)
                    "standards_rag_fields": schema.get("_standards_population", {}).get("fields_populated", 0),
                    # FIX #4: Comprehensive standards defaults (IEC, ISO, ASME, NAMUR)
                    "standards_defaults_fields": schema.get("_schema_field_extraction", {}).get("fields_populated", 0),
                    # FIX #5: Template specifications (62+ per product type)
                    "template_specs_fields": schema.get("_template_specs_added", 0),
                    # Total fields populated
                    "total_fields_populated": (
                        schema.get("_standards_population", {}).get("fields_populated", 0) +
                        schema.get("_schema_field_extraction", {}).get("fields_populated", 0) +
                        schema.get("_template_specs_added", 0)
                    ),
                    # Target achievement
                    "target_fields": 60,
                    "target_achieved": (
                        schema.get("_standards_population", {}).get("fields_populated", 0) +
                        schema.get("_schema_field_extraction", {}).get("fields_populated", 0) +
                        schema.get("_template_specs_added", 0)
                    ) >= 60,
                    # Sources used
                    "sources": [
                        "standards_rag_extraction" if schema.get("_standards_population") else None,
                        "schema_field_extractor_defaults" if schema.get("_schema_field_extraction") else None,
                        "specification_templates" if schema.get("_template_specs_added") else None
                    ]
                },
                # ══════════════════════════════════════════════════════════════
                # ADVANCED SPECIFICATIONS - Triggered when user says "YES"
                # Discovers additional advanced parameters beyond basic schema
                # ══════════════════════════════════════════════════════════════
                "advanced_specs_info": {
                    "triggered": advanced_specs_result is not None,
                    "success": advanced_specs_result.get("success", False) if advanced_specs_result else False,
                    "unique_specifications": advanced_specs_result.get("unique_specifications", []) if advanced_specs_result else [],
                    "total_unique_specifications": advanced_specs_result.get("total_unique_specifications", 0) if advanced_specs_result else 0,
                    "existing_specifications_filtered": advanced_specs_result.get("existing_specifications_filtered", 0) if advanced_specs_result else 0,
                    "discovery_successful": advanced_specs_result.get("discovery_successful", False) if advanced_specs_result else False,
                    "fallback_used": advanced_specs_result.get("fallback_used", False) if advanced_specs_result else False,
                    "error": advanced_specs_result.get("error") if advanced_specs_result else None
                }
            })

            logger.info("[ValidationTool] ✓ Validation completed successfully")
            if advanced_specs_result and advanced_specs_result.get("success"):
                logger.info("[ValidationTool] ✓ Advanced specifications included: %d parameters",
                           advanced_specs_result.get("total_unique_specifications", 0))

            # ═══════════════════════════════════════════════════════════════════════════
            # CACHE SESSION CONTEXT for HITL YES/NO responses
            # This allows retrieving product_type when user responds without context
            # ═══════════════════════════════════════════════════════════════════════════
            if session_id and product_type:
                _cache_session_context(session_id, {
                    "product_type": product_type,
                    "schema": schema,
                    "provided_requirements": provided_requirements,
                    "missing_fields": missing_fields,
                    "is_valid": is_valid
                })

            return result

        except Exception as e:
            logger.error("[ValidationTool] ✗ Validation failed: %s", e, exc_info=True)
            result.update({
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            })
            return result

    def get_schema_only(
        self,
        product_type: str,
        enable_ppi: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Load or generate schema without validation.

        Args:
            product_type: Product type to get schema for
            enable_ppi: Override PPI setting (uses instance setting if None)

        Returns:
            Schema result with source information
        """
        logger.info("[ValidationTool] Loading schema for: %s", product_type)

        try:
            from common.tools.schema_tools import load_schema_tool

            schema_result = load_schema_tool.invoke({
                "product_type": product_type,
                "enable_ppi": enable_ppi if enable_ppi is not None else self.enable_ppi
            })

            logger.info("[ValidationTool] ✓ Schema loaded from: %s",
                       schema_result.get("source", "unknown"))

            return schema_result

        except Exception as e:
            logger.error("[ValidationTool] ✗ Schema loading failed: %s", e, exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "schema": {}
            }

    def validate_with_schema(
        self,
        user_input: str,
        product_type: str,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate user input against a provided schema.

        Args:
            user_input: User's requirement description
            product_type: Product type
            schema: Pre-loaded schema

        Returns:
            Validation result
        """
        logger.info("[ValidationTool] Validating with provided schema")

        try:
            from common.tools.schema_tools import validate_requirements_tool

            validation_result = validate_requirements_tool.invoke({
                "user_input": user_input,
                "product_type": product_type,
                "schema": schema
            })

            return {
                "success": True,
                "product_type": product_type,
                "provided_requirements": validation_result.get("provided_requirements", {}),
                "missing_fields": validation_result.get("missing_fields", []),
                "is_valid": validation_result.get("is_valid", False)
            }

        except Exception as e:
            logger.error("[ValidationTool] ✗ Validation failed: %s", e, exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    # ════════════════════════════════════════════════════════════════════════════
    # PHASE 2: PARALLEL OPTIMIZATION METHODS
    # ════════════════════════════════════════════════════════════════════════════

    def validate_multiple_products_parallel(
        self,
        product_types: List[str],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate schemas for multiple products in parallel (Phase 2 optimization).

        Use this when user needs schemas for multiple products at once.
        Example: "I need schemas for Temperature Transmitter, Pressure Gauge, Level Switch"

        Args:
            product_types: List of product types
            session_id: Session identifier

        Returns:
            Dictionary mapping product_type -> validation result
        """
        if not self.enable_phase2:
            logger.warning("[Phase 2] Parallel optimization disabled, falling back to sequential")
            return self._validate_sequentially(product_types, session_id)

        try:
            from agentic.deep_agent.schema.generation.parallel_generator import ParallelSchemaGenerator

            logger.info(f"[Phase 2] Starting parallel schema generation for {len(product_types)} products")

            generator = ParallelSchemaGenerator(max_workers=min(3, len(product_types)))
            schemas = generator.generate_schemas_in_parallel(product_types, force_regenerate=False)

            # Validate each schema
            results = {}
            for product_type, schema_result in schemas.items():
                if schema_result.get('success'):
                    results[product_type] = {
                        "success": True,
                        "product_type": product_type,
                        "schema": schema_result.get('schema'),
                        "schema_source": schema_result.get('source'),
                        "optimization": "phase2_parallel"
                    }
                else:
                    results[product_type] = {
                        "success": False,
                        "product_type": product_type,
                        "error": schema_result.get('error')
                    }

            logger.info(f"[Phase 2] Parallel generation completed for {len(results)} products")
            return results

        except ImportError:
            logger.warning("[Phase 2] Parallel Schema Generator not available, using sequential")
            return self._validate_sequentially(product_types, session_id)
        except Exception as e:
            logger.error(f"[Phase 2] Error in parallel validation: {e}")
            return self._validate_sequentially(product_types, session_id)

    def _validate_sequentially(
        self,
        product_types: List[str],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fallback to sequential validation when parallel not available."""
        results = {}
        for product_type in product_types:
            result = self.validate(product_type, session_id=session_id)
            results[product_type] = result
        return results

    def enrich_schema_parallel(
        self,
        product_type: str,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich schema using parallel field group queries (Phase 2 optimization).

        Instead of querying Standards RAG sequentially for each field group,
        query all field groups in parallel (5x faster!).

        Args:
            product_type: Product type
            schema: Schema to enrich

        Returns:
            Enriched schema with populated fields
        """
        if not self.enable_phase2:
            logger.warning("[Phase 2] Parallel enrichment disabled")
            return schema

        try:
            from agentic.workflows.standards_rag.parallel_standards_enrichment import ParallelStandardsEnrichment

            logger.info(f"[Phase 2] Starting parallel enrichment for {product_type}")

            enricher = ParallelStandardsEnrichment(max_workers=5)
            enriched = enricher.enrich_schema_in_parallel(product_type, schema)

            logger.info(f"[Phase 2] Parallel enrichment completed for {product_type}")
            return enriched

        except ImportError:
            logger.warning("[Phase 2] Parallel Standards Enrichment not available")
            return schema
        except Exception as e:
            logger.error(f"[Phase 2] Error in parallel enrichment: {e}")
            return schema

    # ════════════════════════════════════════════════════════════════════════════
    # PHASE 3: ASYNC WORKFLOW (Complete Schema Lifecycle)
    # ════════════════════════════════════════════════════════════════════════════

    async def get_or_generate_schema_async(
        self,
        product_type: str,
        session_id: Optional[str] = None,
        force_regenerate: bool = False
    ) -> Dict[str, Any]:
        """
        Get or generate schema using complete async workflow (Phase 3).

        Complete lifecycle:
        1. Check session cache (FIX #A1)
        2. Check database
        3. Generate via PPI (with Phase 1+2+3 optimizations)
        4. Enrich with standards (async parallel)
        5. Store to database
        6. Return to user

        This is the RECOMMENDED method for single or multiple products.

        Args:
            product_type: Product type to get schema for
            session_id: Session identifier
            force_regenerate: Force regeneration (skip caches)

        Returns:
            Dictionary with schema and metadata
        """

        if not self.enable_phase3:
            logger.warning("[Phase 3] Async workflow disabled")
            # Fall back to sync validation
            return self.validate(product_type, session_id=session_id)

        try:
            from agentic.workflows.schema.schema_workflow import SchemaWorkflow

            logger.info(f"[Phase 3] Starting async workflow for: {product_type}")

            workflow = SchemaWorkflow(use_phase3_async=True)
            result = await workflow.get_or_generate_schema(
                product_type,
                session_id=session_id,
                force_regenerate=force_regenerate
            )

            return result

        except ImportError:
            logger.warning("[Phase 3] SchemaWorkflow not available, falling back to Phase 2")
            return self.validate_multiple_products_parallel([product_type], session_id)
        except Exception as e:
            logger.error(f"[Phase 3] Error in async workflow: {e}")
            return self.validate(product_type, session_id=session_id)

    async def get_or_generate_schemas_batch_async(
        self,
        product_types: List[str],
        session_id: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get or generate schemas for multiple products concurrently (Phase 3).

        Uses async concurrent execution for multiple products with benefits:
        - All products generated concurrently
        - Shared database lookups
        - Combined enrichment queries
        - Non-blocking I/O throughout

        Expected Performance (3 products):
        - Before: (437 + 210) × 3 = 1941 seconds
        - After Phase 3: 100-120 seconds (19x faster!)

        Args:
            product_types: List of product types
            session_id: Session identifier

        Returns:
            Dictionary mapping product_type -> schema result
        """

        if not self.enable_phase3:
            logger.warning("[Phase 3] Async batch disabled, using Phase 2")
            return self.validate_multiple_products_parallel(product_types, session_id)

        try:
            from agentic.workflows.schema.schema_workflow import SchemaWorkflow

            logger.info(f"[Phase 3] Starting async batch workflow for {len(product_types)} products")

            workflow = SchemaWorkflow(use_phase3_async=True)
            results = await workflow.get_or_generate_schemas_batch(
                product_types,
                session_id=session_id
            )

            logger.info(f"[Phase 3] Batch workflow completed")

            return results

        except ImportError:
            logger.warning("[Phase 3] SchemaWorkflow not available, falling back to Phase 2")

            # Fallback: Use Phase 2 in parallel
            results = {}
            for product_type in product_types:
                result = self.validate_multiple_products_parallel(
                    [product_type],
                    session_id
                )
                results.update(result)
            return results

        except Exception as e:
            logger.error(f"[Phase 3] Error in async batch workflow: {e}")

            # Fallback: Sequential validation
            results = {}
            for product_type in product_types:
                result = self.validate(product_type, session_id=session_id)
                results[product_type] = result
            return results


# ============================================================================
# STANDALONE USAGE EXAMPLE
# ============================================================================

def example_usage():
    """Example usage of ValidationTool"""
    print("\n" + "="*70)
    print("VALIDATION TOOL - STANDALONE EXAMPLE")
    print("="*70)

    # Initialize tool
    tool = ValidationTool(enable_ppi=True)

    # Example 1: Validate user input
    print("\n[Example 1] Validate user input:")
    result = tool.validate(
        user_input="I need a pressure transmitter with 4-20mA output, 0-100 PSI range",
        session_id="test_session_001"
    )

    print(f"✓ Success: {result['success']}")
    print(f"✓ Product Type: {result.get('product_type')}")
    print(f"✓ Valid: {result.get('is_valid')}")
    print(f"✓ Schema Source: {result.get('schema_source')}")
    print(f"✓ PPI Used: {result.get('ppi_workflow_used')}")
    print(f"✓ Missing Fields: {result.get('missing_fields', [])}")

    # Example 2: Get schema only
    print("\n[Example 2] Get schema only:")
    schema_result = tool.get_schema_only("flow meter")
    print(f"✓ Schema Source: {schema_result.get('source')}")
    print(f"✓ Has Mandatory Fields: {bool(schema_result.get('schema', {}).get('mandatory'))}")


if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    example_usage()
