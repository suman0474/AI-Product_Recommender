"""
Execution Context Package - Unified context for session/workflow/task isolation.

This package provides:
- ExecutionContext: Dataclass representing the execution context hierarchy
- Context Manager: Thread-local storage for context propagation
- Decorators: For automatic context handling

Usage:
    from common.infrastructure.context import (
        ExecutionContext,
        execution_context,
        get_context,
        set_context,
        get_cache_key
    )

    # Create context at API entry point
    ctx = ExecutionContext(
        session_id="main_user123_US_WEST_abc123_20260222",
        workflow_type="product_search",
        user_id="user123",
        zone="US_WEST"
    )

    # Use context manager for automatic cleanup
    with execution_context(ctx):
        # All nested calls can access ctx
        result = run_workflow(...)

    # Access context from anywhere in the call stack
    current_ctx = get_context()
    cache_key = get_cache_key("my_data")
"""

# ExecutionContext dataclass
from .execution_context import ExecutionContext

# Core context management
from .context_manager import (
    # Core functions
    set_context,
    get_context,
    get_context_or_raise,
    clear_context,
    reset_context,

    # Context manager
    execution_context,

    # Decorators
    with_execution_context,
    requires_context,

    # Convenience functions
    get_session_id,
    get_workflow_id,
    get_correlation_id,
    get_cache_key,
    get_log_context,

    # Propagation helpers
    propagate_context_to_thread,
    copy_context_for_async,

    # Diagnostic functions
    has_context,
    describe_context
)

# Context-aware logging
from .context_logger import (
    ContextLoggerAdapter,
    get_context_logger,
    log_with_context,
    ContextFilter,
    get_context_formatter
)

__all__ = [
    # ExecutionContext
    "ExecutionContext",

    # Core functions
    "set_context",
    "get_context",
    "get_context_or_raise",
    "clear_context",
    "reset_context",

    # Context manager
    "execution_context",

    # Decorators
    "with_execution_context",
    "requires_context",

    # Convenience functions
    "get_session_id",
    "get_workflow_id",
    "get_correlation_id",
    "get_cache_key",
    "get_log_context",

    # Propagation helpers
    "propagate_context_to_thread",
    "copy_context_for_async",

    # Diagnostic functions
    "has_context",
    "describe_context",

    # Context-aware logging
    "ContextLoggerAdapter",
    "get_context_logger",
    "log_with_context",
    "ContextFilter",
    "get_context_formatter"
]
