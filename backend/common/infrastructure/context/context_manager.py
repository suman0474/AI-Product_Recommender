"""
Context Manager - Thread-local storage for ExecutionContext.

This module provides thread-local (and async-safe) storage for the
ExecutionContext, allowing any function in the call stack to access
the current context without explicit parameter passing.

Usage:
    from common.infrastructure.context import (
        ExecutionContext,
        set_context,
        get_context,
        execution_context
    )

    # Set context at API entry point
    ctx = ExecutionContext(session_id="...", workflow_type="product_search")

    # Method 1: Context manager (recommended)
    with execution_context(ctx):
        # All nested calls can access ctx via get_context()
        result = run_workflow(...)

    # Method 2: Manual set/clear
    set_context(ctx)
    try:
        result = run_workflow(...)
    finally:
        clear_context()
"""

import contextvars
import functools
import logging
from typing import Optional, Generator, TypeVar, Callable, Any
from contextlib import contextmanager

from .execution_context import ExecutionContext

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# THREAD-LOCAL STORAGE using contextvars (async-safe)
# ═══════════════════════════════════════════════════════════════════════════

# Primary context variable - stores the current ExecutionContext
_current_context: contextvars.ContextVar[Optional[ExecutionContext]] = (
    contextvars.ContextVar('execution_context', default=None)
)

# Token storage for proper context reset
_context_token: contextvars.ContextVar[Optional[contextvars.Token]] = (
    contextvars.ContextVar('context_token', default=None)
)


# ═══════════════════════════════════════════════════════════════════════════
# CORE CONTEXT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def set_context(ctx: ExecutionContext) -> contextvars.Token:
    """
    Set the execution context for the current thread/async task.

    Args:
        ctx: ExecutionContext to set

    Returns:
        Token that can be used to reset to previous state
    """
    token = _current_context.set(ctx)
    _context_token.set(token)
    logger.debug(f"[Context] Set: {ctx.to_log_context()}")
    return token


def get_context() -> Optional[ExecutionContext]:
    """
    Get the current execution context.

    Returns:
        Current ExecutionContext or None if not set
    """
    return _current_context.get()


def get_context_or_raise() -> ExecutionContext:
    """
    Get the current execution context or raise if not set.

    Returns:
        Current ExecutionContext

    Raises:
        RuntimeError: If no context is set
    """
    ctx = _current_context.get()
    if ctx is None:
        raise RuntimeError(
            "No ExecutionContext set. Ensure you're within an execution_context() "
            "block or have called set_context() at the API entry point."
        )
    return ctx


def clear_context() -> None:
    """
    Clear the current execution context.

    Resets to the previous context if one was set, otherwise clears completely.
    """
    ctx = _current_context.get()
    if ctx:
        logger.debug(f"[Context] Cleared: {ctx.correlation_id}")

    token = _context_token.get()
    if token:
        _current_context.reset(token)
        _context_token.set(None)
    else:
        _current_context.set(None)


def reset_context(token: contextvars.Token) -> None:
    """
    Reset context to a specific previous state.

    Args:
        token: Token returned from set_context()
    """
    _current_context.reset(token)
    _context_token.set(None)


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT MANAGER
# ═══════════════════════════════════════════════════════════════════════════

@contextmanager
def execution_context(ctx: ExecutionContext) -> Generator[ExecutionContext, None, None]:
    """
    Context manager for setting execution context.

    Ensures proper cleanup even if exceptions occur.

    Usage:
        with execution_context(ctx) as ctx:
            # ctx is available here and in all nested calls via get_context()
            result = some_function()

    Args:
        ctx: ExecutionContext to use

    Yields:
        The ExecutionContext
    """
    token = _current_context.set(ctx)
    logger.info(f"[Context] Enter: {ctx.to_log_context()}")

    try:
        yield ctx
    finally:
        _current_context.reset(token)
        logger.info(f"[Context] Exit: {ctx.correlation_id}")


# ═══════════════════════════════════════════════════════════════════════════
# DECORATORS
# ═══════════════════════════════════════════════════════════════════════════

F = TypeVar('F', bound=Callable[..., Any])


def with_execution_context(func: F) -> F:
    """
    Decorator that ensures function has access to execution context.

    If context is not set, attempts to create one from function arguments.
    Looks for 'ctx', 'session_id', or 'context' in kwargs.

    Usage:
        @with_execution_context
        def my_function(session_id: str, ...):
            ctx = get_context()
            # ctx is guaranteed to be set

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ctx = get_context()

        if ctx is None:
            # Try to extract context from kwargs
            ctx = kwargs.get('ctx')

            if ctx is None:
                # Try to create from session_id
                session_id = (
                    kwargs.get('session_id') or
                    kwargs.get('context', {}).get('session_id') if isinstance(kwargs.get('context'), dict) else None
                )

                if session_id:
                    ctx = ExecutionContext(
                        session_id=session_id,
                        workflow_type=kwargs.get('workflow_type', func.__name__)
                    )
                    logger.debug(
                        f"[Context] Auto-created context from session_id for {func.__name__}"
                    )

            if ctx:
                token = set_context(ctx)
                try:
                    return func(*args, **kwargs)
                finally:
                    reset_context(token)
            else:
                # No context available - proceed without (will fail if context is required)
                logger.warning(
                    f"[Context] No context available for {func.__name__}. "
                    f"Pass 'ctx' or 'session_id' parameter."
                )
                return func(*args, **kwargs)
        else:
            # Context already set
            return func(*args, **kwargs)

    return wrapper  # type: ignore


def requires_context(func: F) -> F:
    """
    Decorator that enforces execution context is set.

    Raises RuntimeError if no context is available.

    Usage:
        @requires_context
        def my_function(...):
            ctx = get_context_or_raise()
            # Function will only execute if context is set

    Args:
        func: Function to decorate

    Returns:
        Decorated function

    Raises:
        RuntimeError: If no context is set when function is called
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ctx = get_context()
        if ctx is None:
            raise RuntimeError(
                f"Function {func.__name__} requires ExecutionContext but none is set. "
                f"Ensure the caller has set context via execution_context() or set_context()."
            )
        return func(*args, **kwargs)

    return wrapper  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_session_id() -> str:
    """
    Get session_id from current context.

    Returns:
        Session ID or empty string if no context set
    """
    ctx = get_context()
    return ctx.session_id if ctx else ""


def get_workflow_id() -> str:
    """
    Get workflow_id from current context.

    Returns:
        Workflow ID or empty string if no context set
    """
    ctx = get_context()
    return ctx.workflow_id if ctx else ""


def get_correlation_id() -> str:
    """
    Get correlation_id from current context.

    If no context is set, generates a new correlation ID.

    Returns:
        Correlation ID (existing or newly generated)
    """
    ctx = get_context()
    if ctx:
        return ctx.correlation_id
    import uuid
    return uuid.uuid4().hex[:12]


def get_cache_key(suffix: str) -> str:
    """
    Get cache key scoped to current context.

    Falls back to global scope with warning if no context set.

    Args:
        suffix: Cache key suffix

    Returns:
        Scoped cache key
    """
    ctx = get_context()
    if ctx:
        return ctx.to_cache_key(suffix)
    logger.warning(f"[Context] No context for cache key '{suffix}' - using global scope")
    return f"global:{suffix}"


def get_log_context() -> dict:
    """
    Get log context fields from current context.

    Returns:
        Dict with context fields for structured logging, or empty dict if no context
    """
    ctx = get_context()
    return ctx.to_log_context() if ctx else {}


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT PROPAGATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def propagate_context_to_thread(target_func: Callable, *args, **kwargs) -> Callable:
    """
    Wrap a function to propagate current context to a new thread.

    contextvars doesn't automatically propagate to new threads, so this
    helper captures the current context and sets it in the new thread.

    Usage:
        import threading
        ctx = get_context()

        # Context will be available in the new thread
        thread = threading.Thread(
            target=propagate_context_to_thread(worker_function, arg1, arg2)
        )
        thread.start()

    Args:
        target_func: Function to run in new thread
        *args: Args for target_func
        **kwargs: Kwargs for target_func

    Returns:
        Wrapped function that sets context before execution
    """
    ctx = get_context()

    def wrapper():
        if ctx:
            with execution_context(ctx):
                return target_func(*args, **kwargs)
        else:
            return target_func(*args, **kwargs)

    return wrapper


def copy_context_for_async() -> Optional[ExecutionContext]:
    """
    Get a copy of the current context for async operations.

    Use this when spawning async tasks that need the same context.

    Returns:
        Copy of current context, or None if no context set
    """
    ctx = get_context()
    if ctx:
        # Return a new instance with same values
        return ExecutionContext.from_dict(ctx.to_dict())
    return None


# ═══════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def has_context() -> bool:
    """
    Check if an execution context is currently set.

    Returns:
        True if context is set, False otherwise
    """
    return get_context() is not None


def describe_context() -> str:
    """
    Get a human-readable description of the current context.

    Useful for debugging and logging.

    Returns:
        Description string
    """
    ctx = get_context()
    if ctx is None:
        return "No ExecutionContext set"

    return (
        f"ExecutionContext:\n"
        f"  Session: {ctx.session_id}\n"
        f"  Workflow: {ctx.workflow_id} ({ctx.workflow_type})\n"
        f"  Task: {ctx.task_id or 'None'} ({ctx.task_type or 'None'})\n"
        f"  Parent: {ctx.parent_workflow_id or 'None'}\n"
        f"  Correlation: {ctx.correlation_id}\n"
        f"  Zone: {ctx.zone}\n"
        f"  User: {ctx.user_id}"
    )
