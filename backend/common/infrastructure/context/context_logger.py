"""
Context Logger - Structured logging with ExecutionContext.

This module provides a logger adapter that automatically includes
context fields (session_id, workflow_id, correlation_id) in log messages.

Usage:
    from common.infrastructure.context import get_context_logger

    # Get logger with context
    logger = get_context_logger(__name__)

    # Log messages automatically include context
    logger.info("Processing request")
    # Output: [ctx:session123:workflow456:abc12] Processing request

    # Or use with extra context fields
    logger.info("Found products", extra={"product_count": 5})
"""

import logging
from typing import Optional, Dict, Any, MutableMapping
from functools import lru_cache

from .context_manager import get_context


class ContextLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically includes ExecutionContext in log messages.

    Adds context fields as a prefix to log messages and includes them in
    the 'extra' dict for structured logging handlers.
    """

    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        """
        Initialize the context logger adapter.

        Args:
            logger: Underlying logger instance
            extra: Additional static extra fields
        """
        super().__init__(logger, extra or {})

    def process(
        self,
        msg: str,
        kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        """
        Process log message to include context information.

        Args:
            msg: Original log message
            kwargs: Logging kwargs

        Returns:
            Tuple of (formatted message, updated kwargs)
        """
        ctx = get_context()

        if ctx:
            # Build context prefix for message
            ctx_prefix = f"[{ctx.correlation_id}]"

            # Add truncated IDs for readability
            session_short = ctx.session_id[:12] if ctx.session_id else "-"
            workflow_short = ctx.workflow_id[:12] if ctx.workflow_id else "-"

            # Include workflow type if available
            if ctx.workflow_type:
                ctx_prefix = f"[{ctx.correlation_id}:{ctx.workflow_type}]"

            # Add context fields to extra for structured logging
            extra = kwargs.get('extra', {})
            extra.update({
                'ctx_session_id': ctx.session_id,
                'ctx_workflow_id': ctx.workflow_id,
                'ctx_workflow_type': ctx.workflow_type,
                'ctx_task_id': ctx.task_id,
                'ctx_correlation_id': ctx.correlation_id,
                'ctx_zone': ctx.zone,
                'ctx_user_id': ctx.user_id,
            })
            kwargs['extra'] = extra

            # Prepend context to message
            msg = f"{ctx_prefix} {msg}"
        else:
            # No context - add a marker
            msg = f"[no-ctx] {msg}"

        # Merge with static extra fields
        if self.extra:
            kwargs.setdefault('extra', {}).update(self.extra)

        return msg, kwargs


def get_context_logger(name: str, extra: Optional[Dict[str, Any]] = None) -> ContextLoggerAdapter:
    """
    Get a logger adapter that includes ExecutionContext in log messages.

    Args:
        name: Logger name (typically __name__)
        extra: Optional static extra fields to include in all log messages

    Returns:
        ContextLoggerAdapter instance

    Example:
        logger = get_context_logger(__name__)
        logger.info("Processing user request")
        # With context: [abc123:product_search] Processing user request
        # Without context: [no-ctx] Processing user request
    """
    base_logger = logging.getLogger(name)
    return ContextLoggerAdapter(base_logger, extra)


@lru_cache(maxsize=128)
def _get_cached_context_logger(name: str) -> ContextLoggerAdapter:
    """
    Get a cached context logger (for frequently accessed loggers).

    Note: This caches the adapter instance but the context is still
    dynamically resolved on each log call.

    Args:
        name: Logger name

    Returns:
        Cached ContextLoggerAdapter instance
    """
    return get_context_logger(name)


def log_with_context(
    logger: logging.Logger,
    level: int,
    msg: str,
    *args,
    **kwargs
) -> None:
    """
    Log a message with automatic context inclusion.

    Utility function for when you have an existing logger but want
    to include context in a specific log call.

    Args:
        logger: Existing logger instance
        level: Log level (logging.INFO, logging.DEBUG, etc.)
        msg: Log message
        *args: Message format args
        **kwargs: Logging kwargs

    Example:
        log_with_context(logger, logging.INFO, "Found %d products", 5)
    """
    ctx = get_context()

    if ctx:
        ctx_prefix = f"[{ctx.correlation_id}]"
        msg = f"{ctx_prefix} {msg}"

        extra = kwargs.get('extra', {})
        extra.update(ctx.to_log_context())
        kwargs['extra'] = extra

    logger.log(level, msg, *args, **kwargs)


class ContextFilter(logging.Filter):
    """
    Logging filter that adds ExecutionContext fields to log records.

    Use this filter when you want context fields available in log record
    attributes but don't want to modify message formatting.

    Usage:
        handler = logging.StreamHandler()
        handler.addFilter(ContextFilter())
        logger.addHandler(handler)

        # In formatter, access fields like %(ctx_correlation_id)s
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add context fields to log record.

        Always returns True (doesn't filter out records).

        Args:
            record: Log record to augment

        Returns:
            True (always passes the record)
        """
        ctx = get_context()

        if ctx:
            record.ctx_session_id = ctx.session_id
            record.ctx_workflow_id = ctx.workflow_id
            record.ctx_workflow_type = ctx.workflow_type or ""
            record.ctx_task_id = ctx.task_id or ""
            record.ctx_correlation_id = ctx.correlation_id
            record.ctx_zone = ctx.zone
            record.ctx_user_id = ctx.user_id
        else:
            record.ctx_session_id = ""
            record.ctx_workflow_id = ""
            record.ctx_workflow_type = ""
            record.ctx_task_id = ""
            record.ctx_correlation_id = ""
            record.ctx_zone = ""
            record.ctx_user_id = ""

        return True


# Convenience function to get pre-configured formatter
def get_context_formatter(
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None
) -> logging.Formatter:
    """
    Get a log formatter that includes context fields.

    Args:
        fmt: Custom format string (default includes context)
        datefmt: Date format string

    Returns:
        Configured logging.Formatter

    Example:
        handler = logging.StreamHandler()
        handler.setFormatter(get_context_formatter())
        handler.addFilter(ContextFilter())
        logger.addHandler(handler)
    """
    if fmt is None:
        fmt = (
            "%(asctime)s [%(levelname)s] "
            "[%(ctx_correlation_id)s:%(ctx_workflow_type)s] "
            "%(name)s: %(message)s"
        )

    return logging.Formatter(fmt, datefmt=datefmt)
