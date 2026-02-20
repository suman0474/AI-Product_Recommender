"""
Debug Flags Module for Backend Functions
========================================

Centralized debug flag management for all backend functions and agentic workflows.
Provides decorators for function entry/exit logging, performance timing,
LangGraph node tracing, and async support.

Consolidated Debug Flags:
- API                 : API Endpoints & Agentic API
- WORKFLOW            : Orchestration, State, PPI, Checkpointing
- INTENT              : Router, Intent Tools
- RAG                 : Standards, Strategy, Index, Embedding
- LLM                 : Calls, Fallback, Token counting
- TOOLS               : Validation, Advanced Params, Vendor Analysis, Ranking
- SALES               : Sales Agent
- CHAT                : EnGenie Chat
- DEEP_AGENT          : Deep Agent
- SYSTEM              : Circuit Breaker, Rate Limiter, Cache
- DATA                : Unicode, JSON
- SECURITY            : API Keys
- IMAGE               : Image Generation

Environment Variables:
- DEBUG_ALL=1         Enable all debug flags
- DEBUG_<FLAG>=1      Enable specific flag (e.g., DEBUG_WORKFLOW=1)
                      Legacy env vars (e.g. DEBUG_SESSION_ORCHESTRATOR) are supported for backward compatibility.

Usage:
    from common.debug_flags import (
        debug_log, timed_execution, debug_state,
        debug_langgraph_node, debug_log_async, timed_execution_async,
        is_debug_enabled, enable_preset, issue_debug
    )

    # Standard function debugging
    @debug_log("TOOLS")
    def my_function(arg1, arg2):
        pass

    # Performance timing
    @timed_execution("WORKFLOW", threshold_ms=5000)
    def my_workflow():
        pass
"""

import os
import time
import logging
import functools
import threading
from typing import Any, Callable, Optional, Dict, List

logger = logging.getLogger(__name__)


# ============================================================================
# DEBUG FLAGS CONFIGURATION
# ============================================================================

DEBUG_FLAGS: Dict[str, bool] = {
    # Core Functional Areas
    "API": os.getenv("DEBUG_API", "0") == "1" or os.getenv("DEBUG_API_ENDPOINTS", "0") == "1",
    "WORKFLOW": os.getenv("DEBUG_WORKFLOW", "0") == "1" or os.getenv("DEBUG_PPI_WORKFLOW", "0") == "1" or os.getenv("DEBUG_AGENTIC_WORKFLOW", "0") == "1",
    "INTENT": os.getenv("DEBUG_INTENT", "0") == "1" or os.getenv("DEBUG_INTENT_ROUTER", "0") == "1",
    "RAG": os.getenv("DEBUG_RAG", "0") == "1" or os.getenv("DEBUG_STANDARDS_RAG", "0") == "1",
    "LLM": os.getenv("DEBUG_LLM", "0") == "1" or os.getenv("DEBUG_LLM_CALLS", "1") == "1",  # Default on
    "TOOLS": os.getenv("DEBUG_TOOLS", "0") == "1" or os.getenv("DEBUG_VALIDATION_TOOL", "0") == "1",
    "SALES": os.getenv("DEBUG_SALES", "0") == "1" or os.getenv("DEBUG_SALES_AGENT", "0") == "1",
    "DEEP_AGENT": os.getenv("DEBUG_DEEP_AGENT", "0") == "1",
    "CHAT": os.getenv("DEBUG_CHAT", "0") == "1" or os.getenv("DEBUG_ENGENIE_CHAT", "0") == "1",
    
    # Infrastructure & System
    "SYSTEM": os.getenv("DEBUG_SYSTEM", "0") == "1" or os.getenv("DEBUG_CACHE", "1") == "1", # Default on (cache)
    "DATA": os.getenv("DEBUG_DATA", "0") == "1" or os.getenv("DEBUG_JSON", "1") == "1",      # Default on (json/unicode)
    "SECURITY": os.getenv("DEBUG_SECURITY", "0") == "1" or os.getenv("DEBUG_API_KEY", "1") == "1", # Default on
    "IMAGE": os.getenv("DEBUG_IMAGE", "1") == "1",                                           # Default on

    # Global flag
    "ALL": os.getenv("DEBUG_ALL", "0") == "1",
}


# ============================================================================
# FLAG MANAGEMENT FUNCTIONS
# ============================================================================

def is_debug_enabled(module: str) -> bool:
    """
    Check if debugging is enabled for a specific module.
    """
    return DEBUG_FLAGS.get("ALL", False) or DEBUG_FLAGS.get(module, False)


def get_debug_flag(module: str) -> bool:
    return is_debug_enabled(module)


def set_debug_flag(module: str, enabled: bool) -> None:
    DEBUG_FLAGS[module] = enabled
    logger.info(f"[DEBUG_FLAGS] Set {module} = {enabled}")


def enable_all_debug() -> None:
    DEBUG_FLAGS["ALL"] = True
    logger.info("[DEBUG_FLAGS] All debugging enabled")


def disable_all_debug() -> None:
    DEBUG_FLAGS["ALL"] = False
    logger.info("[DEBUG_FLAGS] Global debugging disabled")


def get_enabled_flags() -> Dict[str, bool]:
    return {k: v for k, v in DEBUG_FLAGS.items() if v}


# ============================================================================
# DECORATORS
# ============================================================================

def debug_log(module: str, log_args: bool = True, log_result: bool = False):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_debug_enabled(module):
                return func(*args, **kwargs)

            func_name = func.__name__
            if log_args:
                args_str = ", ".join(repr(a)[:100] for a in args)
                kwargs_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in kwargs.items())
                all_args = f"({args_str}{', ' if args_str and kwargs_str else ''}{kwargs_str})"
                logger.debug(f"[{module}] >> ENTER {func_name}{all_args}")
            else:
                logger.debug(f"[{module}] >> ENTER {func_name}()")

            try:
                result = func(*args, **kwargs)
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
                if threshold_ms and elapsed_ms > threshold_ms:
                    logger.warning(f"[{module}] SLOW: {func_name} took {elapsed_ms:.2f}ms (threshold: {threshold_ms}ms)")
                else:
                    logger.debug(f"[{module}] TIMING: {func_name} took {elapsed_ms:.2f}ms")
                return result
            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"[{module}] TIMING: {func_name} failed after {elapsed_ms:.2f}ms ({type(e).__name__})")
                raise
        return wrapper
    return decorator


def debug_state(module: str, state_name: str = "state"):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_debug_enabled(module):
                return func(*args, **kwargs)

            func_name = func.__name__
            state = kwargs.get(state_name) or (args[0] if args else None)
            if state and isinstance(state, dict):
                logger.debug(f"[{module}] STATE_IN {func_name}: keys={list(state.keys())}")
            else:
                logger.debug(f"[{module}] STATE_IN {func_name}: (no dict state)")

            result = func(*args, **kwargs)

            if result and isinstance(result, dict):
                logger.debug(f"[{module}] STATE_OUT {func_name}: keys={list(result.keys())}")
            return result
        return wrapper
    return decorator


def debug_langgraph_node(module: str, node_name: str = None):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(state: Any, *args, **kwargs):
            if not is_debug_enabled(module):
                return func(state, *args, **kwargs)

            node_name_str = node_name or func.__name__
            start_time = time.time()

            if isinstance(state, dict):
                messages_count = len(state.get("messages", []))
                logger.debug(f"[{module}] NODE_ENTER: {node_name_str} | state_keys={list(state.keys())} | messages={messages_count}")
            else:
                logger.debug(f"[{module}] NODE_ENTER: {node_name_str} | state=<{type(state).__name__}>")

            try:
                result = func(state, *args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000

                if isinstance(result, dict):
                    # Simple mutation check
                    state_keys = set(state.keys()) if isinstance(state, dict) else set()
                    result_keys = set(result.keys())
                    added = result_keys - state_keys
                    removed = state_keys - result_keys
                    mutation_str = ""
                    if added: mutation_str += f" | +{list(added)}"
                    if removed: mutation_str += f" | -{list(removed)}"
                    logger.debug(f"[{module}] NODE_EXIT: {node_name_str} | {elapsed_ms:.2f}ms{mutation_str}")
                else:
                    logger.debug(f"[{module}] NODE_EXIT: {node_name_str} | {elapsed_ms:.2f}ms")

                return result
            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.error(f"[{module}] NODE_ERROR: {node_name_str} | {elapsed_ms:.2f}ms | {type(e).__name__}: {str(e)[:100]}")
                raise
        return wrapper
    return decorator


def debug_log_async(module: str, log_args: bool = True, log_result: bool = False):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not is_debug_enabled(module):
                return await func(*args, **kwargs)

            func_name = func.__name__
            if log_args:
                args_str = ", ".join(repr(a)[:100] for a in args)
                kwargs_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in kwargs.items())
                all_args = f"({args_str}{', ' if args_str and kwargs_str else ''}{kwargs_str})"
                logger.debug(f"[{module}] >> ENTER {func_name}{all_args} [ASYNC]")
            else:
                logger.debug(f"[{module}] >> ENTER {func_name}() [ASYNC]")

            try:
                result = await func(*args, **kwargs)
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
                    logger.warning(f"[{module}] SLOW: {func_name} took {elapsed_ms:.2f}ms (threshold: {threshold_ms}ms) [ASYNC]")
                else:
                    logger.debug(f"[{module}] TIMING: {func_name} took {elapsed_ms:.2f}ms [ASYNC]")
                return result
            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"[{module}] TIMING: {func_name} failed after {elapsed_ms:.2f}ms ({type(e).__name__}) [ASYNC]")
                raise
        return wrapper
    return decorator


# ============================================================================
# DEBUG PRESETS
# ============================================================================

DEBUG_PRESETS: Dict[str, list] = {
    "minimal": ["API", "LLM"],
    "workflow": ["WORKFLOW", "INTENT", "TOOLS"],
    "rag": ["RAG", "SYSTEM"],
    "deep": ["DEEP_AGENT", "RAG", "LLM"],
    "chat": ["CHAT", "INTENT", "WORKFLOW"],
    "full": [],
}


def enable_preset(preset_name: str) -> None:
    flags = DEBUG_PRESETS.get(preset_name)
    if flags is None:
        logger.warning(f"[DEBUG_FLAGS] Unknown preset: {preset_name}")
        return

    if preset_name == "full" or not flags:
        flags = [k for k in DEBUG_FLAGS.keys() if k != "ALL"]

    for flag in flags:
        if flag in DEBUG_FLAGS:
            DEBUG_FLAGS[flag] = True
        else:
            logger.warning(f"[DEBUG_FLAGS] Preset '{preset_name}': Unknown flag '{flag}'")

    logger.info(f"[DEBUG_FLAGS] Enabled preset: {preset_name} ({len(flags)} flags)")


def disable_preset(preset_name: str) -> None:
    # Similar logic, just disable
    flags = DEBUG_PRESETS.get(preset_name)
    if flags is None: return
    if preset_name == "full" or not flags:
        flags = [k for k in DEBUG_FLAGS.keys() if k != "ALL"]
    for flag in flags:
        if flag in DEBUG_FLAGS:
            DEBUG_FLAGS[flag] = False
    logger.info(f"[DEBUG_FLAGS] Disabled preset: {preset_name}")


def get_available_presets() -> Dict[str, list]:
    return dict(DEBUG_PRESETS)


# ============================================================================
# UI DECISION PATTERN DETECTION
# ============================================================================

UI_DECISION_PATTERNS = [
    "user selected:", "user clicked:", "user chose:", "decision:",
    "action:", "button clicked:", "continue", "proceed",
    "go back", "cancel", "skip", "confirm",
]

def is_ui_decision_input(user_input: str) -> bool:
    if not user_input: return False
    normalized = user_input.lower().strip()
    single_word_decisions = {"continue", "proceed", "skip", "cancel", "confirm", "back"}
    if normalized in single_word_decisions: return True
    for pattern in UI_DECISION_PATTERNS:
        if pattern in normalized: return True
    return False

def get_ui_decision_error_message(user_input: str) -> str:
    return (
        f"Input '{user_input}' appears to be a UI navigation action, not a product requirement. "
        "Please provide actual product requirements."
    )


# ============================================================================
# ISSUE-SPECIFIC DEBUG LOGGER
# ============================================================================

_issue_counters = {
    "embedding_calls": 0, "llm_calls": 0, "cache_hits": 0, "cache_misses": 0,
    "image_fallbacks": 0, "api_key_rotations": 0, "json_errors": 0, "unicode_errors": 0,
}
_counter_lock = threading.Lock()

def _increment_counter(name: str, amount: int = 1) -> int:
    with _counter_lock:
        _issue_counters[name] = _issue_counters.get(name, 0) + amount
        return _issue_counters[name]

def get_issue_counters() -> Dict[str, int]:
    with _counter_lock: return dict(_issue_counters)

def reset_issue_counters() -> None:
    """Reset all issue counters to zero."""
    with _counter_lock:
        _issue_counters.clear()

class IssueDebugger:
    def __init__(self): self._session_id = None
    def set_session(self, session_id: str): self._session_id = session_id
    
    def _log(self, category: str, message: str, level: str = "info"):
        if not is_debug_enabled(category): return
        session_str = f" [sess={self._session_id[:8]}]" if self._session_id else ""
        full_message = f"[DEBUG:{category}]{session_str} {message}"
        if level == "warning": logger.warning(full_message)
        elif level == "error": logger.error(full_message)
        else: logger.info(full_message)

    # API KEY -> SECURITY
    def api_key_rotated(self, from_idx: int, to_idx: int, reason: str):
        count = _increment_counter("api_key_rotations")
        self._log("SECURITY", f"ROTATED #{count}: index {from_idx} -> {to_idx} (reason: {reason})")

    def api_key_leaked(self, key_preview: str, error_msg: str):
        _increment_counter("api_key_rotations")
        self._log("SECURITY", f"⚠️ LEAKED KEY DETECTED! Preview: {key_preview}... Error: {error_msg}", "error")

    def api_key_exhausted(self, key_idx: int, retry_after: int):
        self._log("SECURITY", f"QUOTA_EXHAUSTED: key #{key_idx}, retry after {retry_after}s", "warning")

    # EMBEDDING -> RAG
    def embedding_call(self, model: str, text_count: int, source: str):
        count = _increment_counter("embedding_calls")
        self._log("RAG", f"EMBEDDING CALL #{count}: model={model}, texts={text_count}, source={source}")

    def embedding_cache_hit(self, key: str):
        _increment_counter("cache_hits")
        self._log("RAG", f"EMBEDDING CACHE_HIT: {key[:50]}...")

    def embedding_cache_miss(self, key: str):
        _increment_counter("cache_misses")
        self._log("RAG", f"EMBEDDING CACHE_MISS: {key[:50]}...")

    # IMAGE -> IMAGE (Keep)
    def image_lookup(self, product_type: str, source: str, cached: bool):
        status = "CACHED" if cached else "GENERATED"
        self._log("IMAGE", f"LOOKUP: '{product_type}' -> {source} ({status})")

    def image_fallback(self, original: str, fallbacks_tried: list, final_match: str):
        count = _increment_counter("image_fallbacks", len(fallbacks_tried))
        chain = " -> ".join([original] + fallbacks_tried)
        self._log("IMAGE", f"FALLBACK #{count}: {chain} -> FOUND: '{final_match}'")

    def image_no_match(self, product_type: str, fallbacks_tried: list):
        self._log("IMAGE", f"NO_MATCH: '{product_type}' after {len(fallbacks_tried)} fallbacks", "warning")

    def image_llm_generation(self, product_type: str, success: bool, time_ms: int = 0):
        status = "SUCCESS" if success else "FAILED"
        self._log("IMAGE", f"LLM_GEN: '{product_type}' -> {status} ({time_ms}ms)")

    # CACHE -> SYSTEM
    def cache_hit(self, cache_type: str, key: str):
        _increment_counter("cache_hits")
        self._log("SYSTEM", f"CACHE HIT [{cache_type}]: {key[:60]}...")

    def cache_miss(self, cache_type: str, key: str):
        _increment_counter("cache_misses")
        self._log("SYSTEM", f"CACHE MISS [{cache_type}]: {key[:60]}...")

    def cache_write(self, cache_type: str, key: str, success: bool):
        status = "OK" if success else "FAILED"
        level = "info" if success else "warning"
        self._log("SYSTEM", f"CACHE WRITE [{cache_type}]: {key[:60]}... -> {status}", level)

    # LLM_CALLS -> LLM
    def llm_call(self, model: str, purpose: str, tokens: int = 0):
        count = _increment_counter("llm_calls")
        token_str = f", ~{tokens} tokens" if tokens else ""
        self._log("LLM", f"CALL #{count}: model={model}, purpose={purpose}{token_str}")

    def llm_response(self, model: str, success: bool, time_ms: int):
        status = "OK" if success else "FAILED"
        self._log("LLM", f"RESPONSE: model={model} -> {status} ({time_ms}ms)")

    def llm_fallback_triggered(self, from_model: str, to_model: str, reason: str):
        self._log("LLM", f"FALLBACK: {from_model} -> {to_model} (reason: {reason})", "warning")

    # UNICODE -> DATA
    def unicode_error(self, char: str, context: str):
        count = _increment_counter("unicode_errors")
        try: code_point = f"U+{ord(char):04X}"
        except: code_point = "unknown"
        self._log("DATA", f"UNICODE ERROR #{count}: char='{char}' ({code_point}) in {context}", "error")

    def unicode_sanitized(self, original: str, sanitized: str, context: str):
        self._log("DATA", f"SANITIZED in {context}: '{original[:30]}...' -> '{sanitized[:30]}...'")

    # JSON -> DATA
    def json_parse_error(self, source: str, error: str, raw_preview: str = ""):
        count = _increment_counter("json_errors")
        preview = raw_preview[:80].replace("\n", "\\n") if raw_preview else ""
        self._log("DATA", f"JSON PARSE_ERROR #{count} in {source}: {error}. Preview: {preview}...", "error")

    def json_validation_error(self, source: str, missing_fields: list):
        _increment_counter("json_errors")
        self._log("DATA", f"JSON VALIDATION_ERROR in {source}: missing fields {missing_fields}", "warning")

    # TIKTOKEN -> LLM
    def tiktoken_init_error(self, error: str):
        self._log("LLM", f"TIKTOKEN INIT_FAILED: {error}", "error")

    def tiktoken_circular_import(self, module: str):
        self._log("LLM", f"TIKTOKEN CIRCULAR_IMPORT detected in {module}", "error")

    def tiktoken_fallback(self, alternative: str):
        self._log("LLM", f"TIKTOKEN FALLBACK: using {alternative} instead", "warning")

    def print_summary(self):
        counters = get_issue_counters()
        logger.info("=" * 60)
        logger.info("[DEBUG:SUMMARY] Session Issue Statistics")
        logger.info("=" * 60)
        for k, v in counters.items():
            logger.info(f"  {k.replace('_', ' ').title()}: {v}")
        logger.info("=" * 60)

issue_debug = IssueDebugger()

def safe_print(message: str, fallback: str = None):
    try:
        print(message)
    except UnicodeEncodeError as e:
        issue_debug.unicode_error(char=message[e.start] if e.start < len(message) else "?", context="safe_print")
        if fallback: print(fallback)
        else: print(message.encode('ascii', 'replace').decode('ascii'))

_log_debug_status = lambda: logger.info(f"[DEBUG_FLAGS] Enabled flags: {list(get_enabled_flags().keys())}") if get_enabled_flags() else logger.debug("[DEBUG_FLAGS] No debug flags enabled")
_log_debug_status()
