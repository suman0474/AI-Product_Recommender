"""
Structured Logger - Production-grade JSON logging with correlation tracking

Provides:
- JSON-formatted logs for Azure Monitor/Log Analytics
- Correlation IDs to trace requests across components
- Session ID + Instance ID in every log entry
- Context variables to automatically propagate IDs

Usage:
    from observability.structured_logger import (
        StructuredLogger,
        set_correlation_context,
        get_correlation_id
    )
    
    # At request entry point
    set_correlation_context(
        correlation_id="req-abc123",
        session_id="main_user1_uuid_ts",
        instance_id="inst-xyz789"
    )
    
    # In any component
    logger = StructuredLogger("SessionManager")
    logger.info("Session created", user_id="abc", duration_ms=45)
"""

import json
import logging
import uuid
import traceback
from contextvars import ContextVar
from datetime import datetime
from typing import Optional, Dict, Any
from functools import wraps
import time

# =============================================================================
# CONTEXT VARIABLES (Thread-safe correlation tracking)
# =============================================================================

correlation_id_var: ContextVar[str] = ContextVar('correlation_id', default='')
session_id_var: ContextVar[str] = ContextVar('session_id', default='')
instance_id_var: ContextVar[str] = ContextVar('instance_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')
workflow_type_var: ContextVar[str] = ContextVar('workflow_type', default='')


def set_correlation_context(
    correlation_id: str,
    session_id: str = "",
    instance_id: str = "",
    user_id: str = "",
    workflow_type: str = ""
) -> None:
    """
    Set correlation context for the current request thread.
    
    Called at request entry point (Flask/FastAPI middleware).
    All subsequent logs in this request will include these IDs.
    
    Args:
        correlation_id: Unique ID for this request (trace across all components)
        session_id: Current user's session ID
        instance_id: Current workflow instance ID (if applicable)
        user_id: User identifier
        workflow_type: Type of workflow being executed
    """
    correlation_id_var.set(correlation_id)
    if session_id:
        session_id_var.set(session_id)
    if instance_id:
        instance_id_var.set(instance_id)
    if user_id:
        user_id_var.set(user_id)
    if workflow_type:
        workflow_type_var.set(workflow_type)


def get_correlation_id() -> str:
    """Get current correlation ID."""
    return correlation_id_var.get()


def get_session_id() -> str:
    """Get current session ID from context."""
    return session_id_var.get()


def get_instance_id() -> str:
    """Get current instance ID from context."""
    return instance_id_var.get()


def generate_correlation_id() -> str:
    """Generate a new correlation ID for a request."""
    return f"req-{uuid.uuid4().hex[:12]}-{int(time.time())}"


def clear_correlation_context() -> None:
    """Clear all correlation context variables."""
    correlation_id_var.set('')
    session_id_var.set('')
    instance_id_var.set('')
    user_id_var.set('')
    workflow_type_var.set('')


# =============================================================================
# STRUCTURED LOGGER
# =============================================================================

class StructuredLogger:
    """
    Production-grade structured JSON logger with correlation tracking.
    
    Features:
    - JSON-formatted output for log aggregation (Azure Monitor, ELK, etc.)
    - Automatic correlation ID injection
    - Session/Instance ID tracking
    - Error stack trace capture
    - Performance timing helpers
    
    Usage:
        logger = StructuredLogger("SessionManager", service="orchestrator")
        logger.info("Session created", user_id="abc", duration_ms=45)
        logger.error("Failed to create session", error=exception)
    """
    
    def __init__(self, name: str, service: str = "orchestrator"):
        """
        Initialize structured logger.
        
        Args:
            name: Component name (e.g., "SessionManager", "RouterAgent")
            service: Service name for log filtering
        """
        self.logger = logging.getLogger(name)
        self.name = name
        self.service = service
        
        # Ensure logger has handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
    
    def _build_log_entry(
        self,
        level: str,
        message: str,
        error: Optional[Exception] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Build structured JSON log entry with all context.
        
        Args:
            level: Log level (INFO, ERROR, DEBUG, WARNING)
            message: Log message
            error: Optional exception for error logs
            **kwargs: Additional fields to include
            
        Returns:
            Dictionary ready for JSON serialization
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "service": self.service,
            "component": self.name,
            "correlation_id": correlation_id_var.get() or None,
            "session_id": session_id_var.get() or None,
            "instance_id": instance_id_var.get() or None,
            "user_id": user_id_var.get() or None,
            "workflow_type": workflow_type_var.get() or None,
            "message": message,
        }
        
        # Add error details if present
        if error:
            entry["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "stacktrace": traceback.format_exc()
            }
        
        # Add any additional fields
        entry.update(kwargs)
        
        # Remove None values for cleaner logs
        return {k: v for k, v in entry.items() if v is not None}
    
    def _log(self, level: str, message: str, error: Optional[Exception] = None, **kwargs):
        """Internal logging method."""
        entry = self._build_log_entry(level, message, error, **kwargs)
        log_line = json.dumps(entry, default=str)
        
        if level == "DEBUG":
            self.logger.debug(log_line)
        elif level == "INFO":
            self.logger.info(log_line)
        elif level == "WARNING":
            self.logger.warning(log_line)
        elif level == "ERROR":
            self.logger.error(log_line)
        elif level == "CRITICAL":
            self.logger.critical(log_line)
    
    def debug(self, message: str, **kwargs):
        """Log DEBUG level message."""
        self._log("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log INFO level message."""
        self._log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log WARNING level message."""
        self._log("WARNING", message, **kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log ERROR level message with optional exception details."""
        self._log("ERROR", message, error=error, **kwargs)
    
    def critical(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log CRITICAL level message with optional exception details."""
        self._log("CRITICAL", message, error=error, **kwargs)


# =============================================================================
# TIMING DECORATOR
# =============================================================================

def timed_operation(logger: StructuredLogger, operation_name: str):
    """
    Decorator to log operation duration.
    
    Usage:
        logger = StructuredLogger("SessionManager")
        
        @timed_operation(logger, "create_session")
        def create_session(user_id):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"{operation_name} completed",
                    operation=operation_name,
                    duration_ms=round(duration_ms, 2),
                    status="success"
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"{operation_name} failed",
                    error=e,
                    operation=operation_name,
                    duration_ms=round(duration_ms, 2),
                    status="error"
                )
                raise
        return wrapper
    return decorator


# =============================================================================
# ASYNC TIMING DECORATOR
# =============================================================================

def async_timed_operation(logger: StructuredLogger, operation_name: str):
    """
    Async decorator to log operation duration.
    
    Usage:
        logger = StructuredLogger("SessionManager")
        
        @async_timed_operation(logger, "create_session")
        async def create_session(user_id):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"{operation_name} completed",
                    operation=operation_name,
                    duration_ms=round(duration_ms, 2),
                    status="success"
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"{operation_name} failed",
                    error=e,
                    operation=operation_name,
                    duration_ms=round(duration_ms, 2),
                    status="error"
                )
                raise
        return wrapper
    return decorator


# =============================================================================
# GLOBAL LOGGER INSTANCE
# =============================================================================

# Default logger for quick access
_default_logger = StructuredLogger("orchestrator")


def get_logger(name: str, service: str = "orchestrator") -> StructuredLogger:
    """
    Get a structured logger for a component.
    
    Args:
        name: Component name
        service: Service name
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, service)
