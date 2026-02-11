"""
Agentic Debug Utilities
=======================

Extends debug_flags.py with agentic-specific debugging helpers.
Combines debug_flags + rag_logger for comprehensive workflow debugging.

Usage:
    from agentic.utils.debug import (
        debug_rag_query, log_workflow_transition,
        track_node_execution, is_debug_enabled,
        debug_log, timed_execution, issue_debug
    )

    @debug_rag_query("INDEX")
    def query_index(user_query: str):
        ...

    def my_workflow_node(state):
        log_workflow_transition("AGENTIC_WORKFLOW", "node_a", "node_b")
        ...
"""

import functools
import time
import logging
from typing import Callable, Optional, Dict, Any, List

# Import from debug_flags
from debug_flags import (
    is_debug_enabled,
    debug_log,
    timed_execution,
    debug_state,
    debug_langgraph_node,
    debug_log_async,
    timed_execution_async,
    issue_debug,
    enable_preset,
    disable_preset,
    get_enabled_flags,
    get_issue_counters,
    reset_issue_counters,
    set_debug_flag,
    DEBUG_FLAGS,
    DEBUG_PRESETS,
)

# Import from rag_logger
from agentic.rag.logger import (
    RAGLogger,
    set_trace_id,
    get_trace_id,
    clear_trace_id,
    log_node_timing,
)

logger = logging.getLogger(__name__)

# Re-export for convenience
__all__ = [
    # From debug_flags
    'is_debug_enabled',
    'debug_log',
    'timed_execution',
    'debug_state',
    'debug_langgraph_node',
    'debug_log_async',
    'timed_execution_async',
    'issue_debug',
    'enable_preset',
    'disable_preset',
    'get_enabled_flags',
    'get_issue_counters',
    'reset_issue_counters',
    'set_debug_flag',
    'DEBUG_FLAGS',
    'DEBUG_PRESETS',
    # From rag_logger
    'RAGLogger',
    'set_trace_id',
    'get_trace_id',
    'clear_trace_id',
    'log_node_timing',
    # New utilities
    'debug_rag_query',
    'debug_api_endpoint',
    'track_node_execution',
    'log_workflow_transition',
    'log_state_mutation',
    'track_llm_query',
    'track_llm_fallback',
    'track_cache_operation',
    'DebugContext',
]


# ============================================================================
# RAG-SPECIFIC DEBUGGING
# ============================================================================

def debug_rag_query(rag_name: str, threshold_ms: float = 3000):
    """
    Combined decorator for RAG queries with timing and trace logging.

    Automatically:
    - Ensures trace ID exists for request correlation
    - Logs query entry with trace ID
    - Measures execution time
    - Warns if threshold exceeded

    Args:
        rag_name: RAG system name (INDEX, STRATEGY, STANDARDS)
        threshold_ms: Warning threshold in milliseconds

    Usage:
        @debug_rag_query("INDEX", threshold_ms=2000)
        def query(user_query: str, top_k: int = 5):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            module = f"{rag_name}_RAG"

            if not is_debug_enabled(module):
                return func(*args, **kwargs)

            # Ensure trace ID exists for request tracking
            trace_id = get_trace_id()
            if not trace_id:
                trace_id = set_trace_id()

            func_name = func.__name__
            start_time = time.time()

            # Log entry
            logger.debug(f"[{module}] >> ENTER {func_name} [trace_id={trace_id}]")

            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000

                if elapsed_ms > threshold_ms:
                    logger.warning(
                        f"[{module}] SLOW: {func_name} took {elapsed_ms:.2f}ms "
                        f"(threshold: {threshold_ms}ms)"
                    )
                else:
                    logger.debug(f"[{module}] << EXIT {func_name} | {elapsed_ms:.2f}ms")

                return result

            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"[{module}] ERROR: {func_name} failed after {elapsed_ms:.2f}ms | "
                    f"{type(e).__name__}: {str(e)[:100]}"
                )
                raise

        return wrapper
    return decorator


def debug_api_endpoint(module: str = "AGENTIC_API", log_payload: bool = False):
    """
    Decorator for API endpoint debugging with trace ID management.

    Automatically:
    - Sets up trace ID from header or generates new one
    - Logs request entry/exit
    - Tracks timing

    Args:
        module: Module name for debug flag check
        log_payload: Whether to log request payload (default: False for security)

    Usage:
        @app.route('/api/workflow', methods=['POST'])
        @debug_api_endpoint("AGENTIC_API")
        def start_workflow():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_debug_enabled(module):
                return func(*args, **kwargs)

            func_name = func.__name__
            start_time = time.time()

            # Try to get trace ID from request headers
            try:
                from flask import request
                trace_id = request.headers.get('X-Trace-ID')
                if not trace_id:
                    trace_id = set_trace_id()
                else:
                    set_trace_id(trace_id)

                method = request.method
                path = request.path
            except Exception:
                trace_id = set_trace_id()
                method = "?"
                path = func_name

            logger.debug(f"[{module}] >> {method} {path} [trace_id={trace_id}]")

            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000

                # Try to get response status
                try:
                    if hasattr(result, '__iter__') and len(result) > 1:
                        status = result[1] if isinstance(result[1], int) else 200
                    else:
                        status = 200
                except Exception:
                    status = 200

                logger.debug(f"[{module}] << {status} {path} | {elapsed_ms:.2f}ms")

                return result

            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"[{module}] << ERROR {path} | {elapsed_ms:.2f}ms | "
                    f"{type(e).__name__}: {str(e)[:100]}"
                )
                raise
            finally:
                clear_trace_id()

        return wrapper
    return decorator


# ============================================================================
# WORKFLOW DEBUGGING
# ============================================================================

def log_workflow_transition(
    module: str,
    from_node: str,
    to_node: str,
    condition: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Log workflow graph transitions for debugging routing logic.

    Useful for tracking which workflow path is taken and why.

    Args:
        module: Module name for debug check (e.g., "AGENTIC_WORKFLOW")
        from_node: Source node name
        to_node: Target node name
        condition: Optional condition that triggered transition
        metadata: Optional metadata about the transition

    Usage:
        if user_intent == "question":
            log_workflow_transition(
                "AGENTIC_WORKFLOW",
                "classify_intent",
                "retrieve_context",
                condition="intent==question"
            )
    """
    if not is_debug_enabled(module):
        return

    cond_str = f" [condition: {condition}]" if condition else ""
    meta_str = f" {metadata}" if metadata else ""

    logger.debug(f"[{module}] TRANSITION: {from_node} -> {to_node}{cond_str}{meta_str}")


def track_node_execution(
    module: str,
    node_name: str,
    duration_ms: float,
    success: bool = True,
    error: Optional[Exception] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Track execution of a workflow node.

    Args:
        module: Module name
        node_name: Node name
        duration_ms: Execution time in milliseconds
        success: Whether execution succeeded
        error: Exception if failed
        metadata: Additional context

    Usage:
        start = time.time()
        try:
            result = execute_node(state)
            track_node_execution("WORKFLOW", "my_node", (time.time()-start)*1000)
        except Exception as e:
            track_node_execution("WORKFLOW", "my_node", (time.time()-start)*1000, False, e)
    """
    if not is_debug_enabled(module):
        return

    status = "SUCCESS" if success else "FAILED"
    msg = f"[{module}] NODE: {node_name} | {duration_ms:.2f}ms | {status}"

    if metadata:
        msg += f" | {metadata}"

    if error:
        msg += f" | {type(error).__name__}: {str(error)[:100]}"
        logger.error(msg)
    else:
        logger.debug(msg)


def log_state_mutation(
    module: str,
    node_name: str,
    old_state: Dict[str, Any],
    new_state: Dict[str, Any]
):
    """
    Log state mutations between workflow nodes.

    Tracks added/removed/modified keys.

    Args:
        module: Module name
        node_name: Node that mutated state
        old_state: State before node execution
        new_state: State after node execution

    Usage:
        old_state_copy = dict(state)
        # ... modify state ...
        log_state_mutation("AGENTIC_WORKFLOW", "my_node", old_state_copy, state)
    """
    if not is_debug_enabled(module):
        return

    old_keys = set(old_state.keys()) if isinstance(old_state, dict) else set()
    new_keys = set(new_state.keys()) if isinstance(new_state, dict) else set()

    added = new_keys - old_keys
    removed = old_keys - new_keys

    msg = f"[{module}] STATE_MUTATION: {node_name}"
    if added:
        msg += f" | +{list(added)}"
    if removed:
        msg += f" | -{list(removed)}"

    if added or removed:
        logger.debug(msg)


# ============================================================================
# LLM DEBUGGING INTEGRATION
# ============================================================================

def track_llm_query(
    module: str,
    model: str,
    purpose: str,
    tokens_estimated: int = 0
):
    """
    Track an LLM API call.

    Integrates with issue_debug and metrics tracking.

    Args:
        module: Module making the call
        model: LLM model name
        purpose: Purpose of the call
        tokens_estimated: Estimated token count

    Usage:
        track_llm_query("DEEP_AGENT", "gpt-4", "spec_extraction", tokens_estimated=5000)
    """
    issue_debug.llm_call(model, purpose, tokens_estimated)

    if is_debug_enabled(module):
        logger.debug(
            f"[{module}] LLM_CALL: model={model}, purpose={purpose}, "
            f"tokens_est={tokens_estimated}"
        )


def track_llm_fallback(
    module: str,
    from_model: str,
    to_model: str,
    reason: str
):
    """
    Track fallback to alternative LLM model.

    Args:
        module: Module experiencing fallback
        from_model: Original model that failed
        to_model: Fallback model being used
        reason: Reason for fallback

    Usage:
        track_llm_fallback("DEEP_AGENT", "gpt-4", "gpt-3.5-turbo", "rate_limit")
    """
    issue_debug.llm_fallback_triggered(from_model, to_model, reason)

    if is_debug_enabled(module):
        logger.warning(f"[{module}] LLM_FALLBACK: {from_model} -> {to_model} ({reason})")


# ============================================================================
# CACHING DEBUGGING
# ============================================================================

def track_cache_operation(
    cache_type: str,
    operation: str,
    key: str,
    success: bool = True
):
    """
    Track cache operations for performance analysis.

    Args:
        cache_type: Type of cache (RAG, LLM, EMBEDDING, etc.)
        operation: Operation type ("hit", "miss", "write")
        key: Cache key (truncated in logs)
        success: Whether operation succeeded (for writes)

    Usage:
        if result := cache.get(key):
            track_cache_operation("RAG", "hit", key)
        else:
            track_cache_operation("RAG", "miss", key)
    """
    if operation == "hit":
        issue_debug.cache_hit(cache_type, key)
    elif operation == "miss":
        issue_debug.cache_miss(cache_type, key)
    elif operation == "write":
        issue_debug.cache_write(cache_type, key, success)


# ============================================================================
# DEBUG CONTEXT MANAGER
# ============================================================================

class DebugContext:
    """
    Context manager for debug session management.

    Automatically sets up trace ID and logs entry/exit.

    Usage:
        with DebugContext("AGENTIC_WORKFLOW", "process_request", session_id="123"):
            # Your code here
            pass

        # Or as decorator
        @DebugContext.wrap("AGENTIC_WORKFLOW", "process_request")
        def my_function():
            pass
    """

    def __init__(
        self,
        module: str,
        operation: str,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ):
        self.module = module
        self.operation = operation
        self.session_id = session_id
        self.trace_id = trace_id
        self.start_time = None
        self._enabled = is_debug_enabled(module)

    def __enter__(self):
        if not self._enabled:
            return self

        self.start_time = time.time()

        # Set up trace ID
        if self.trace_id:
            set_trace_id(self.trace_id)
        elif not get_trace_id():
            self.trace_id = set_trace_id()
        else:
            self.trace_id = get_trace_id()

        # Set session for issue debugging
        if self.session_id:
            issue_debug.set_session(self.session_id)

        logger.debug(
            f"[{self.module}] >> BEGIN {self.operation} "
            f"[trace_id={self.trace_id}]"
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._enabled:
            return False

        elapsed_ms = (time.time() - self.start_time) * 1000 if self.start_time else 0

        if exc_type is not None:
            logger.error(
                f"[{self.module}] << END {self.operation} | {elapsed_ms:.2f}ms | "
                f"ERROR: {exc_type.__name__}: {str(exc_val)[:100]}"
            )
        else:
            logger.debug(
                f"[{self.module}] << END {self.operation} | {elapsed_ms:.2f}ms | SUCCESS"
            )

        return False  # Don't suppress exceptions

    @classmethod
    def wrap(cls, module: str, operation: str):
        """Use DebugContext as a decorator."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with cls(module, operation):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


# ============================================================================
# DEBUG DASHBOARD HELPERS
# ============================================================================

def get_debug_summary() -> Dict[str, Any]:
    """
    Get comprehensive debug summary for monitoring/dashboard.

    Returns:
        Dict containing enabled flags, issue counters, and trace info
    """
    return {
        "enabled_flags": get_enabled_flags(),
        "issue_counters": get_issue_counters(),
        "trace_id": get_trace_id(),
        "available_presets": list(DEBUG_PRESETS.keys()),
    }


def print_debug_summary():
    """Print formatted debug summary to console."""
    from datetime import datetime

    summary = get_debug_summary()

    print("\n" + "=" * 60)
    print("DEBUG SESSION SUMMARY")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Trace ID: {summary['trace_id'] or 'None'}")
    print()

    print("Enabled Flags:")
    enabled = summary['enabled_flags']
    if enabled:
        for flag in sorted(enabled.keys()):
            print(f"  + {flag}")
    else:
        print("  (none)")
    print()

    print("Issue Counters:")
    counters = summary['issue_counters']
    for name, count in sorted(counters.items()):
        if count > 0:
            print(f"  {name}: {count}")
    print("=" * 60 + "\n")
