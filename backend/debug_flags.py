"""
Debug Flags Module for Backend Functions
========================================

Centralized debug flag management for all backend functions and agentic workflows.
Provides decorators for function entry/exit logging, performance timing,
LangGraph node tracing, and async support.

Environment Variables (Module-level):
- DEBUG_ALL=1                     Enable all debug flags
- DEBUG_VALIDATION_TOOL=1         Enable validation tool debugging
- DEBUG_ADVANCED_PARAMS=1         Enable advanced parameters debugging
- DEBUG_VENDOR_ANALYSIS=1         Enable vendor analysis debugging
- DEBUG_RANKING_TOOL=1            Enable ranking tool debugging
- DEBUG_WORKFLOW=1                Enable workflow orchestration debugging
- DEBUG_INTENT_TOOLS=1            Enable intent tools debugging
- DEBUG_LLM_FALLBACK=1            Enable LLM fallback debugging
- DEBUG_API_ENDPOINTS=1           Enable API endpoint debugging

Agentic System Flags:
- DEBUG_AGENTIC_API=1             Enable agentic REST API debugging
- DEBUG_AGENTIC_WORKFLOW=1        Enable agentic workflow node tracing
- DEBUG_INTENT_ROUTER=1           Enable intent classification debugging
- DEBUG_INDEX_RAG=1               Enable Index RAG debugging
- DEBUG_STRATEGY_RAG=1            Enable Strategy RAG debugging
- DEBUG_STANDARDS_RAG=1           Enable Standards RAG debugging
- DEBUG_DEEP_AGENT=1              Enable Deep Agent debugging
- DEBUG_ENGENIE_CHAT=1            Enable EnGenie Chat debugging
- DEBUG_SESSION_ORCHESTRATOR=1    Enable session orchestrator debugging
- DEBUG_CHECKPOINTING=1           Enable checkpointing debugging
- DEBUG_CIRCUIT_BREAKER=1         Enable circuit breaker debugging
- DEBUG_RATE_LIMITER=1            Enable rate limiter debugging
- DEBUG_WORKFLOW_STATE=1          Enable workflow state debugging

Issue-specific Flags (enabled by default):
- DEBUG_API_KEY=1                 Track API key rotation/leaks
- DEBUG_EMBEDDING=1               Track embedding API calls
- DEBUG_CACHE=1                   Track cache hits/misses
- DEBUG_LLM_CALLS=1               Track LLM call counting

Usage:
    from debug_flags import (
        debug_log, timed_execution, debug_state,
        debug_langgraph_node, debug_log_async, timed_execution_async,
        is_debug_enabled, enable_preset, issue_debug
    )

    # Standard function debugging
    @debug_log("VALIDATION_TOOL")
    def my_function(arg1, arg2):
        pass

    # Performance timing
    @timed_execution("WORKFLOW", threshold_ms=5000)
    def my_workflow():
        pass

    # LangGraph node tracing
    @debug_langgraph_node("AGENTIC_WORKFLOW", "classify_intent")
    def classify_intent_node(state):
        return state

    # Async function debugging
    @debug_log_async("DEEP_AGENT")
    async def extract_async(data):
        pass

    # Enable preset
    enable_preset("workflow")  # Enable workflow-related flags
    enable_preset("rag")       # Enable RAG-related flags
"""

import os
import time
import logging
import functools
from typing import Any, Callable, Optional, Dict, List

logger = logging.getLogger(__name__)


# ============================================================================
# DEBUG FLAGS CONFIGURATION
# ============================================================================

DEBUG_FLAGS: Dict[str, bool] = {
    # Module-level flags (set via environment variables)
    "VALIDATION_TOOL": os.getenv("DEBUG_VALIDATION_TOOL", "0") == "1",
    "ADVANCED_PARAMS": os.getenv("DEBUG_ADVANCED_PARAMS", "0") == "1",
    "VENDOR_ANALYSIS": os.getenv("DEBUG_VENDOR_ANALYSIS", "0") == "1",
    "RANKING_TOOL": os.getenv("DEBUG_RANKING_TOOL", "0") == "1",
    "WORKFLOW": os.getenv("DEBUG_WORKFLOW", "0") == "1",
    "INTENT_TOOLS": os.getenv("DEBUG_INTENT_TOOLS", "0") == "1",
    "LLM_FALLBACK": os.getenv("DEBUG_LLM_FALLBACK", "0") == "1",
    "API_ENDPOINTS": os.getenv("DEBUG_API_ENDPOINTS", "0") == "1",
    "SALES_AGENT": os.getenv("DEBUG_SALES_AGENT", "0") == "1",
    "PPI_WORKFLOW": os.getenv("DEBUG_PPI_WORKFLOW", "0") == "1",
    "STANDARDS_RAG": os.getenv("DEBUG_STANDARDS_RAG", "0") == "1",
    "STRATEGY_RAG": os.getenv("DEBUG_STRATEGY_RAG", "0") == "1",

    # Agentic System Flags (new)
    "AGENTIC_API": os.getenv("DEBUG_AGENTIC_API", "0") == "1",
    "AGENTIC_WORKFLOW": os.getenv("DEBUG_AGENTIC_WORKFLOW", "0") == "1",
    "INTENT_ROUTER": os.getenv("DEBUG_INTENT_ROUTER", "0") == "1",
    "INDEX_RAG": os.getenv("DEBUG_INDEX_RAG", "0") == "1",
    "DEEP_AGENT": os.getenv("DEBUG_DEEP_AGENT", "0") == "1",
    "ENGENIE_CHAT": os.getenv("DEBUG_ENGENIE_CHAT", "0") == "1",
    "SESSION_ORCHESTRATOR": os.getenv("DEBUG_SESSION_ORCHESTRATOR", "0") == "1",
    "CHECKPOINTING": os.getenv("DEBUG_CHECKPOINTING", "0") == "1",
    "CIRCUIT_BREAKER": os.getenv("DEBUG_CIRCUIT_BREAKER", "0") == "1",
    "RATE_LIMITER": os.getenv("DEBUG_RATE_LIMITER", "0") == "1",
    "WORKFLOW_STATE": os.getenv("DEBUG_WORKFLOW_STATE", "0") == "1",

    # Issue-specific debug flags (enabled by default for problem detection)
    # Search terminal logs with: grep "[DEBUG:FLAG_NAME]" log.txt
    "API_KEY": os.getenv("DEBUG_API_KEY", "1") == "1",        # API key rotation/leaks
    "EMBEDDING": os.getenv("DEBUG_EMBEDDING", "1") == "1",    # Embedding API calls
    "IMAGE": os.getenv("DEBUG_IMAGE", "1") == "1",            # Generic image fallbacks
    "CACHE": os.getenv("DEBUG_CACHE", "1") == "1",            # Cache hits/misses
    "LLM_CALLS": os.getenv("DEBUG_LLM_CALLS", "1") == "1",    # LLM call counting
    "UNICODE": os.getenv("DEBUG_UNICODE", "1") == "1",        # Unicode encoding issues
    "JSON": os.getenv("DEBUG_JSON", "1") == "1",              # JSON parsing errors
    "TIKTOKEN": os.getenv("DEBUG_TIKTOKEN", "1") == "1",      # tiktoken import issues

    # Global flag to enable all debugging
    "ALL": os.getenv("DEBUG_ALL", "0") == "1",
}


# ============================================================================
# FLAG MANAGEMENT FUNCTIONS
# ============================================================================

def is_debug_enabled(module: str) -> bool:
    """
    Check if debugging is enabled for a specific module.

    Args:
        module: Module name (e.g., "VALIDATION_TOOL", "WORKFLOW")

    Returns:
        True if debugging is enabled for the module or globally
    """
    return DEBUG_FLAGS.get("ALL", False) or DEBUG_FLAGS.get(module, False)


def get_debug_flag(module: str) -> bool:
    """
    Get the current debug flag value for a module.

    Args:
        module: Module name

    Returns:
        Current flag value (True/False)
    """
    return DEBUG_FLAGS.get(module, False)


def set_debug_flag(module: str, enabled: bool) -> None:
    """
    Set a debug flag programmatically (useful for testing).

    Args:
        module: Module name
        enabled: Whether to enable (True) or disable (False) debugging
    """
    DEBUG_FLAGS[module] = enabled
    logger.info(f"[DEBUG_FLAGS] Set {module} = {enabled}")


def enable_all_debug() -> None:
    """Enable debugging for all modules."""
    DEBUG_FLAGS["ALL"] = True
    logger.info("[DEBUG_FLAGS] All debugging enabled")


def disable_all_debug() -> None:
    """Disable global debugging (individual flags still apply)."""
    DEBUG_FLAGS["ALL"] = False
    logger.info("[DEBUG_FLAGS] Global debugging disabled")


def get_enabled_flags() -> Dict[str, bool]:
    """
    Get a dictionary of all currently enabled debug flags.

    Returns:
        Dictionary of flag_name -> enabled status
    """
    return {k: v for k, v in DEBUG_FLAGS.items() if v}


# ============================================================================
# DECORATORS
# ============================================================================

def debug_log(module: str, log_args: bool = True, log_result: bool = False):
    """
    Decorator for function entry/exit logging.

    Logs function entry with arguments and exit with result (if enabled).
    Only logs when debugging is enabled for the module.

    Args:
        module: Module name for debug flag check
        log_args: Whether to log function arguments (default: True)
        log_result: Whether to log return value (default: False, can be verbose)

    Usage:
        @debug_log("VALIDATION_TOOL")
        def validate(user_input, session_id=None):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_debug_enabled(module):
                return func(*args, **kwargs)

            func_name = func.__name__

            # Log entry
            if log_args:
                # Truncate long arguments for readability
                args_str = ", ".join(repr(a)[:100] for a in args)
                kwargs_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in kwargs.items())
                all_args = f"({args_str}{', ' if args_str and kwargs_str else ''}{kwargs_str})"
                logger.debug(f"[{module}] >> ENTER {func_name}{all_args}")
            else:
                logger.debug(f"[{module}] >> ENTER {func_name}()")

            try:
                result = func(*args, **kwargs)

                # Log exit
                if log_result and result is not None:
                    result_str = repr(result)[:200]
                    logger.debug(f"[{module}] << EXIT {func_name} => {result_str}")
                else:
                    logger.debug(f"[{module}] << EXIT {func_name} => OK")

                return result

            except Exception as e:
                logger.debug(f"[{module}] << EXIT {func_name} => EXCEPTION: {type(e).__name__}: {e}")
                raise

        return wrapper
    return decorator


def timed_execution(module: str, threshold_ms: Optional[float] = None):
    """
    Decorator for performance timing.

    Measures and logs function execution time.
    Optionally warns if execution exceeds threshold.

    Args:
        module: Module name for debug flag check
        threshold_ms: Optional threshold in milliseconds. If exceeded, logs a warning.

    Usage:
        @timed_execution("WORKFLOW", threshold_ms=5000)
        def run_workflow():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_debug_enabled(module):
                return func(*args, **kwargs)

            func_name = func.__name__
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000

                # Log timing
                if threshold_ms and elapsed_ms > threshold_ms:
                    logger.warning(
                        f"[{module}] SLOW: {func_name} took {elapsed_ms:.2f}ms "
                        f"(threshold: {threshold_ms}ms)"
                    )
                else:
                    logger.debug(f"[{module}] TIMING: {func_name} took {elapsed_ms:.2f}ms")

                return result

            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"[{module}] TIMING: {func_name} failed after {elapsed_ms:.2f}ms "
                    f"({type(e).__name__})"
                )
                raise

        return wrapper
    return decorator


def debug_state(module: str, state_name: str = "state"):
    """
    Decorator for logging workflow state at entry and exit.

    Useful for debugging LangGraph-style workflows where state is passed through.

    Args:
        module: Module name for debug flag check
        state_name: Name of the state parameter (default: "state")

    Usage:
        @debug_state("WORKFLOW")
        def process_step(state: Dict):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_debug_enabled(module):
                return func(*args, **kwargs)

            func_name = func.__name__

            # Try to find state in args or kwargs
            state = kwargs.get(state_name) or (args[0] if args else None)

            if state and isinstance(state, dict):
                state_keys = list(state.keys())
                logger.debug(f"[{module}] STATE_IN {func_name}: keys={state_keys}")
            else:
                logger.debug(f"[{module}] STATE_IN {func_name}: (no dict state)")

            result = func(*args, **kwargs)

            if result and isinstance(result, dict):
                result_keys = list(result.keys())
                logger.debug(f"[{module}] STATE_OUT {func_name}: keys={result_keys}")

            return result

        return wrapper
    return decorator


def debug_langgraph_node(module: str, node_name: str = None):
    """
    Decorator for LangGraph workflow nodes with state tracking.

    Automatically logs:
    - Node entry with state keys summary
    - Message count in state
    - Node exit with timing
    - State mutations (new/removed keys)
    - Exceptions with traceback

    Args:
        module: Module name for debug flag check
        node_name: Optional node name (defaults to function name)

    Usage:
        @debug_langgraph_node("AGENTIC_WORKFLOW", "classify_intent")
        def classify_intent_node(state: WorkflowState) -> WorkflowState:
            ...

    Output:
        [AGENTIC_WORKFLOW] NODE_ENTER: classify_intent | state_keys=['input', 'messages'] | messages=5
        [AGENTIC_WORKFLOW] NODE_EXIT: classify_intent | 234.56ms | +['next_step']
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(state: Any, *args, **kwargs):
            if not is_debug_enabled(module):
                return func(state, *args, **kwargs)

            node_name_str = node_name or func.__name__
            start_time = time.time()

            # Log entry with state summary
            if isinstance(state, dict):
                state_keys = list(state.keys())
                messages_count = len(state.get("messages", []))
                logger.debug(
                    f"[{module}] NODE_ENTER: {node_name_str} | "
                    f"state_keys={state_keys} | messages={messages_count}"
                )
            else:
                logger.debug(f"[{module}] NODE_ENTER: {node_name_str} | state=<{type(state).__name__}>")

            try:
                result = func(state, *args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000

                # Log exit with changes
                if isinstance(result, dict):
                    result_keys = list(result.keys())
                    # Detect mutations
                    state_keys_set = set(state.keys()) if isinstance(state, dict) else set()
                    result_keys_set = set(result.keys())
                    added = result_keys_set - state_keys_set
                    removed = state_keys_set - result_keys_set

                    mutation_str = ""
                    if added:
                        mutation_str += f" | +{list(added)}"
                    if removed:
                        mutation_str += f" | -{list(removed)}"

                    logger.debug(
                        f"[{module}] NODE_EXIT: {node_name_str} | {elapsed_ms:.2f}ms{mutation_str}"
                    )
                else:
                    logger.debug(f"[{module}] NODE_EXIT: {node_name_str} | {elapsed_ms:.2f}ms")

                return result

            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"[{module}] NODE_ERROR: {node_name_str} | {elapsed_ms:.2f}ms | "
                    f"{type(e).__name__}: {str(e)[:100]}"
                )
                raise

        return wrapper
    return decorator


def debug_log_async(module: str, log_args: bool = True, log_result: bool = False):
    """
    Async version of debug_log decorator for coroutines.

    Args:
        module: Module name for debug flag check
        log_args: Whether to log function arguments (default: True)
        log_result: Whether to log return value (default: False)

    Usage:
        @debug_log_async("DEEP_AGENT")
        async def extract_specs_async(product_type: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not is_debug_enabled(module):
                return await func(*args, **kwargs)

            func_name = func.__name__

            # Log entry
            if log_args:
                args_str = ", ".join(repr(a)[:100] for a in args)
                kwargs_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in kwargs.items())
                all_args = f"({args_str}{', ' if args_str and kwargs_str else ''}{kwargs_str})"
                logger.debug(f"[{module}] >> ENTER {func_name}{all_args} [ASYNC]")
            else:
                logger.debug(f"[{module}] >> ENTER {func_name}() [ASYNC]")

            try:
                result = await func(*args, **kwargs)

                # Log exit
                if log_result and result is not None:
                    result_str = repr(result)[:200]
                    logger.debug(f"[{module}] << EXIT {func_name} => {result_str}")
                else:
                    logger.debug(f"[{module}] << EXIT {func_name} => OK")

                return result

            except Exception as e:
                logger.debug(f"[{module}] << EXIT {func_name} => EXCEPTION: {type(e).__name__}: {e}")
                raise

        return wrapper
    return decorator


def timed_execution_async(module: str, threshold_ms: Optional[float] = None):
    """
    Async version of timed_execution decorator.

    Args:
        module: Module name for debug flag check
        threshold_ms: Optional threshold in milliseconds. If exceeded, logs a warning.

    Usage:
        @timed_execution_async("DEEP_AGENT", threshold_ms=30000)
        async def extract_specs_parallel(products: list):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not is_debug_enabled(module):
                return await func(*args, **kwargs)

            func_name = func.__name__
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000

                if threshold_ms and elapsed_ms > threshold_ms:
                    logger.warning(
                        f"[{module}] SLOW: {func_name} took {elapsed_ms:.2f}ms "
                        f"(threshold: {threshold_ms}ms) [ASYNC]"
                    )
                else:
                    logger.debug(f"[{module}] TIMING: {func_name} took {elapsed_ms:.2f}ms [ASYNC]")

                return result

            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"[{module}] TIMING: {func_name} failed after {elapsed_ms:.2f}ms "
                    f"({type(e).__name__}) [ASYNC]"
                )
                raise

        return wrapper
    return decorator


# ============================================================================
# DEBUG PRESETS (Quick enable/disable combinations)
# ============================================================================

DEBUG_PRESETS: Dict[str, list] = {
    "minimal": [
        "API_ENDPOINTS",
        "LLM_CALLS",
    ],
    "workflow": [
        "AGENTIC_API",
        "AGENTIC_WORKFLOW",
        "INTENT_ROUTER",
        "SESSION_ORCHESTRATOR",
    ],
    "rag": [
        "INDEX_RAG",
        "STRATEGY_RAG",
        "STANDARDS_RAG",
        "EMBEDDING",
        "CACHE",
    ],
    "deep": [
        "DEEP_AGENT",
        "STANDARDS_RAG",
        "LLM_CALLS",
    ],
    "chat": [
        "ENGENIE_CHAT",
        "INTENT_ROUTER",
        "LLM_CALLS",
        "SESSION_ORCHESTRATOR",
    ],
    "resilience": [
        "CIRCUIT_BREAKER",
        "RATE_LIMITER",
        "LLM_FALLBACK",
    ],
    "full": [],  # Dynamically set to all flags
}


def enable_preset(preset_name: str) -> None:
    """
    Enable a debug preset by name.

    Available presets:
        - minimal: Core API and LLM tracking
        - workflow: Agentic workflow orchestration
        - rag: RAG system performance
        - deep: Deep agent and standards analysis
        - chat: Chat interface
        - resilience: Circuit breaker and rate limiting
        - full: All debugging

    Usage:
        from debug_flags import enable_preset
        enable_preset("rag")  # Enable all RAG debugging
    """
    flags = DEBUG_PRESETS.get(preset_name)

    if flags is None:
        logger.warning(f"[DEBUG_FLAGS] Unknown preset: {preset_name}")
        return

    if preset_name == "full" or not flags:
        # Enable all non-issue flags for "full" preset
        flags = [k for k in DEBUG_FLAGS.keys() if k != "ALL"]

    for flag in flags:
        if flag in DEBUG_FLAGS:
            set_debug_flag(flag, True)
        else:
            logger.warning(f"[DEBUG_FLAGS] Preset '{preset_name}': Unknown flag '{flag}'")

    logger.info(f"[DEBUG_FLAGS] Enabled preset: {preset_name} ({len(flags)} flags)")


def disable_preset(preset_name: str) -> None:
    """Disable a debug preset by name."""
    flags = DEBUG_PRESETS.get(preset_name)

    if flags is None:
        logger.warning(f"[DEBUG_FLAGS] Unknown preset: {preset_name}")
        return

    if preset_name == "full" or not flags:
        flags = [k for k in DEBUG_FLAGS.keys() if k != "ALL"]

    for flag in flags:
        if flag in DEBUG_FLAGS:
            set_debug_flag(flag, False)

    logger.info(f"[DEBUG_FLAGS] Disabled preset: {preset_name}")


def get_available_presets() -> Dict[str, list]:
    """Get all available debug presets."""
    return dict(DEBUG_PRESETS)


# ============================================================================
# UI DECISION PATTERN DETECTION (For validation_tool.py Fix)
# ============================================================================

# Patterns that indicate UI decision states, NOT product requirements
UI_DECISION_PATTERNS = [
    "user selected:",
    "user clicked:",
    "user chose:",
    "decision:",
    "action:",
    "button clicked:",
    "continue",
    "proceed",
    "go back",
    "cancel",
    "skip",
    "confirm",
]


def is_ui_decision_input(user_input: str) -> bool:
    """
    Check if the input is a UI decision pattern rather than product requirements.

    UI decision patterns are inputs like "User selected: continue" that come from
    the frontend when a user clicks a button. These should NOT be processed as
    product requirements.

    Args:
        user_input: The user input string to check

    Returns:
        True if the input matches a UI decision pattern
    """
    if not user_input:
        return False

    normalized = user_input.lower().strip()

    # Check for exact matches (single-word decisions)
    single_word_decisions = {"continue", "proceed", "skip", "cancel", "confirm", "back"}
    if normalized in single_word_decisions:
        return True

    # Check for pattern matches
    for pattern in UI_DECISION_PATTERNS:
        if pattern in normalized:
            return True

    return False


def get_ui_decision_error_message(user_input: str) -> str:
    """
    Get an appropriate error message for UI decision inputs.

    Args:
        user_input: The UI decision input

    Returns:
        Descriptive error message
    """
    return (
        f"Input '{user_input}' appears to be a UI navigation action, not a product requirement. "
        "Please provide actual product requirements (e.g., 'I need a pressure transmitter with 4-20mA output') "
        "or use the appropriate API endpoint for the current workflow state."
    )


# ============================================================================
# INITIALIZATION LOGGING
# ============================================================================

def _log_debug_status():
    """Log current debug flag status on module load."""
    enabled = get_enabled_flags()
    if enabled:
        logger.info(f"[DEBUG_FLAGS] Enabled flags: {list(enabled.keys())}")
    else:
        logger.debug("[DEBUG_FLAGS] No debug flags enabled")


# Log status on import
_log_debug_status()


# ============================================================================
# ISSUE-SPECIFIC DEBUG LOGGER (For Terminal Log Analysis)
# ============================================================================
# All logs use format: [DEBUG:CATEGORY] message
# This makes them easy to grep/search in terminal logs.
#
# Usage:
#     from debug_flags import issue_debug
#     issue_debug.api_key_rotated(0, 1, "rate_limit")
#     issue_debug.embedding_call("embedding-001", 5, "semantic_classifier")
#     issue_debug.image_fallback("Pump Valve", ["Pump", "Valve"], "Valve")
# ============================================================================

import threading

# Thread-safe counters for issue tracking
_issue_counters = {
    "embedding_calls": 0,
    "llm_calls": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "image_fallbacks": 0,
    "api_key_rotations": 0,
    "json_errors": 0,
    "unicode_errors": 0,
}
_counter_lock = threading.Lock()


def _increment_counter(name: str, amount: int = 1) -> int:
    """Thread-safe counter increment. Returns new value."""
    with _counter_lock:
        _issue_counters[name] = _issue_counters.get(name, 0) + amount
        return _issue_counters[name]


def get_issue_counters() -> Dict[str, int]:
    """Get snapshot of all issue debug counters."""
    with _counter_lock:
        return dict(_issue_counters)


def reset_issue_counters():
    """Reset all counters (useful between sessions)."""
    with _counter_lock:
        for key in _issue_counters:
            _issue_counters[key] = 0


class IssueDebugger:
    """
    Issue-specific debug logger for terminal log analysis.
    
    All methods log with format: [DEBUG:CATEGORY] message
    This makes it easy to grep/search logs for specific issues.
    
    Grep Examples:
        grep "[DEBUG:API_KEY]" terminal.log
        grep "[DEBUG:EMBEDDING]" terminal.log | wc -l
        grep "[DEBUG:CACHE] MISS" terminal.log
        grep "[DEBUG:LLM_CALLS]" terminal.log | wc -l
    """
    
    def __init__(self):
        self._session_id: Optional[str] = None
    
    def set_session(self, session_id: str):
        """Set current session for context in debug messages."""
        self._session_id = session_id
    
    def _log(self, category: str, message: str, level: str = "info"):
        """Internal logging with consistent [DEBUG:CATEGORY] format."""
        if not is_debug_enabled(category):
            return
        
        session_str = f" [sess={self._session_id[:8]}]" if self._session_id else ""
        full_message = f"[DEBUG:{category}]{session_str} {message}"
        
        if level == "warning":
            logger.warning(full_message)
        elif level == "error":
            logger.error(full_message)
        else:
            logger.info(full_message)
    
    # =========================================================================
    # API KEY DEBUGGING
    # =========================================================================
    
    def api_key_rotated(self, from_idx: int, to_idx: int, reason: str):
        """Log API key rotation event."""
        count = _increment_counter("api_key_rotations")
        self._log("API_KEY", f"ROTATED #{count}: index {from_idx} → {to_idx} (reason: {reason})")
    
    def api_key_leaked(self, key_preview: str, error_msg: str):
        """Log leaked API key detection (critical!)."""
        _increment_counter("api_key_rotations")
        self._log("API_KEY", f"⚠️ LEAKED KEY DETECTED! Preview: {key_preview}... Error: {error_msg}", "error")
    
    def api_key_exhausted(self, key_idx: int, retry_after: int):
        """Log quota exhausted for key."""
        self._log("API_KEY", f"QUOTA_EXHAUSTED: key #{key_idx}, retry after {retry_after}s", "warning")
    
    # =========================================================================
    # EMBEDDING DEBUGGING
    # =========================================================================
    
    def embedding_call(self, model: str, text_count: int, source: str):
        """Log embedding API call."""
        count = _increment_counter("embedding_calls")
        self._log("EMBEDDING", f"CALL #{count}: model={model}, texts={text_count}, source={source}")
    
    def embedding_cache_hit(self, key: str):
        """Log embedding cache hit."""
        _increment_counter("cache_hits")
        self._log("EMBEDDING", f"CACHE_HIT: {key[:50]}...")
    
    def embedding_cache_miss(self, key: str):
        """Log embedding cache miss."""
        _increment_counter("cache_misses")
        self._log("EMBEDDING", f"CACHE_MISS: {key[:50]}...")
    
    # =========================================================================
    # IMAGE DEBUGGING
    # =========================================================================
    
    def image_lookup(self, product_type: str, source: str, cached: bool):
        """Log image lookup result."""
        status = "CACHED" if cached else "GENERATED"
        self._log("IMAGE", f"LOOKUP: '{product_type}' → {source} ({status})")
    
    def image_fallback(self, original: str, fallbacks_tried: list, final_match: str):
        """Log image fallback chain."""
        count = _increment_counter("image_fallbacks", len(fallbacks_tried))
        chain = " → ".join([original] + fallbacks_tried)
        self._log("IMAGE", f"FALLBACK #{count}: {chain} → FOUND: '{final_match}'")
    
    def image_no_match(self, product_type: str, fallbacks_tried: list):
        """Log when no image found after all fallbacks."""
        self._log("IMAGE", f"NO_MATCH: '{product_type}' after {len(fallbacks_tried)} fallbacks", "warning")
    
    def image_llm_generation(self, product_type: str, success: bool, time_ms: int = 0):
        """Log LLM image generation attempt."""
        status = "SUCCESS" if success else "FAILED"
        time_str = f" ({time_ms}ms)" if time_ms else ""
        self._log("IMAGE", f"LLM_GEN: '{product_type}' → {status}{time_str}")
    
    # =========================================================================
    # CACHE DEBUGGING
    # =========================================================================
    
    def cache_hit(self, cache_type: str, key: str):
        """Log general cache hit."""
        _increment_counter("cache_hits")
        self._log("CACHE", f"HIT [{cache_type}]: {key[:60]}...")
    
    def cache_miss(self, cache_type: str, key: str):
        """Log general cache miss."""
        _increment_counter("cache_misses")
        self._log("CACHE", f"MISS [{cache_type}]: {key[:60]}...")
    
    def cache_write(self, cache_type: str, key: str, success: bool):
        """Log cache write attempt."""
        status = "OK" if success else "FAILED"
        level = "info" if success else "warning"
        self._log("CACHE", f"WRITE [{cache_type}]: {key[:60]}... → {status}", level)
    
    # =========================================================================
    # LLM CALL DEBUGGING
    # =========================================================================
    
    def llm_call(self, model: str, purpose: str, tokens: int = 0):
        """Log LLM API call."""
        count = _increment_counter("llm_calls")
        token_str = f", ~{tokens} tokens" if tokens else ""
        self._log("LLM_CALLS", f"CALL #{count}: model={model}, purpose={purpose}{token_str}")
    
    def llm_response(self, model: str, success: bool, time_ms: int):
        """Log LLM response received."""
        status = "OK" if success else "FAILED"
        self._log("LLM_CALLS", f"RESPONSE: model={model} → {status} ({time_ms}ms)")
    
    def llm_fallback_triggered(self, from_model: str, to_model: str, reason: str):
        """Log when fallback to different model is triggered."""
        self._log("LLM_CALLS", f"FALLBACK: {from_model} → {to_model} (reason: {reason})", "warning")
    
    # =========================================================================
    # UNICODE DEBUGGING
    # =========================================================================
    
    def unicode_error(self, char: str, context: str):
        """Log Unicode encoding error."""
        count = _increment_counter("unicode_errors")
        try:
            code_point = f"U+{ord(char):04X}"
        except:
            code_point = "unknown"
        self._log("UNICODE", f"ERROR #{count}: char='{char}' ({code_point}) in {context}", "error")
    
    def unicode_sanitized(self, original: str, sanitized: str, context: str):
        """Log when string was sanitized for Unicode issues."""
        self._log("UNICODE", f"SANITIZED in {context}: '{original[:30]}...' → '{sanitized[:30]}...'")
    
    # =========================================================================
    # JSON DEBUGGING
    # =========================================================================
    
    def json_parse_error(self, source: str, error: str, raw_preview: str = ""):
        """Log JSON parsing error."""
        count = _increment_counter("json_errors")
        preview = raw_preview[:80].replace("\n", "\\n") if raw_preview else ""
        self._log("JSON", f"PARSE_ERROR #{count} in {source}: {error}. Preview: {preview}...", "error")
    
    def json_validation_error(self, source: str, missing_fields: list):
        """Log JSON schema validation error."""
        _increment_counter("json_errors")
        self._log("JSON", f"VALIDATION_ERROR in {source}: missing fields {missing_fields}", "warning")
    
    # =========================================================================
    # TIKTOKEN DEBUGGING
    # =========================================================================
    
    def tiktoken_init_error(self, error: str):
        """Log tiktoken initialization failure."""
        self._log("TIKTOKEN", f"INIT_FAILED: {error}", "error")
    
    def tiktoken_circular_import(self, module: str):
        """Log tiktoken circular import detection."""
        self._log("TIKTOKEN", f"CIRCULAR_IMPORT detected in {module}", "error")
    
    def tiktoken_fallback(self, alternative: str):
        """Log tiktoken fallback to alternative."""
        self._log("TIKTOKEN", f"FALLBACK: using {alternative} instead", "warning")
    
    # =========================================================================
    # SESSION SUMMARY
    # =========================================================================
    
    def print_summary(self):
        """Print summary of all issue counters for the session."""
        counters = get_issue_counters()
        lines = [
            "=" * 60,
            "[DEBUG:SUMMARY] Session Issue Statistics",
            "=" * 60,
            f"  LLM Calls:         {counters.get('llm_calls', 0)}",
            f"  Embedding Calls:   {counters.get('embedding_calls', 0)}",
            f"  Cache Hits:        {counters.get('cache_hits', 0)}",
            f"  Cache Misses:      {counters.get('cache_misses', 0)}",
            f"  Image Fallbacks:   {counters.get('image_fallbacks', 0)}",
            f"  API Key Rotations: {counters.get('api_key_rotations', 0)}",
            f"  JSON Errors:       {counters.get('json_errors', 0)}",
            f"  Unicode Errors:    {counters.get('unicode_errors', 0)}",
            "=" * 60,
        ]
        for line in lines:
            logger.info(line)


# Global singleton instance for issue debugging
issue_debug = IssueDebugger()


# ============================================================================
# SAFE PRINT UTILITY (For Windows Unicode handling)
# ============================================================================

def safe_print(message: str, fallback: str = None):
    """
    Print message safely, handling Unicode encoding issues on Windows.
    
    Args:
        message: The message to print (may contain emojis/unicode)
        fallback: Optional ASCII fallback message if encoding fails
    """
    try:
        print(message)
    except UnicodeEncodeError as e:
        issue_debug.unicode_error(
            char=message[e.start] if e.start < len(message) else "?",
            context="safe_print"
        )
        if fallback:
            print(fallback)
        else:
            # Strip problematic characters
            safe_msg = message.encode('ascii', 'replace').decode('ascii')
            print(safe_msg)

