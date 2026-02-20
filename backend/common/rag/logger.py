"""
Standardized Logging for RAG Systems

Provides consistent log formatting with trace IDs and node context
across all RAG workflows.
"""

import logging
import uuid
from typing import Optional
from contextvars import ContextVar
from functools import wraps
import time

# Request trace ID context (thread-local)
_trace_id: ContextVar[str] = ContextVar('trace_id', default='')


def set_trace_id(trace_id: Optional[str] = None) -> str:
    """
    Set trace ID for current request context.
    
    Args:
        trace_id: Optional trace ID to use (generates new one if None)
        
    Returns:
        The trace ID that was set
    """
    tid = trace_id or str(uuid.uuid4())[:8]
    _trace_id.set(tid)
    return tid


def get_trace_id() -> str:
    """Get current trace ID."""
    return _trace_id.get() or "no-trace"


def clear_trace_id() -> None:
    """Clear the trace ID (call at end of request)."""
    _trace_id.set('')


class RAGLogger:
    """
    Standardized logger for RAG systems.
    
    Provides consistent log formatting:
    [trace_id][RAG_NAME][NODE] message
    
    Example:
        >>> logger = RAGLogger("IndexRAG")
        >>> logger.info("Node1", "Processing query")
        # Output: [abc12345][IndexRAG][Node1] Processing query
    """
    
    def __init__(self, rag_name: str):
        """
        Initialize RAG logger.
        
        Args:
            rag_name: Name of the RAG system (e.g., "IndexRAG", "StandardsRAG")
        """
        self.rag_name = rag_name
        self.logger = logging.getLogger(f"agentic.{rag_name.lower()}")
    
    def _format(self, node: str, message: str) -> str:
        """Format log message with trace ID and context."""
        trace = get_trace_id()
        return f"[{trace}][{self.rag_name}][{node}] {message}"
    
    def info(self, node: str, message: str) -> None:
        """Log info message."""
        self.logger.info(self._format(node, message))
    
    def debug(self, node: str, message: str) -> None:
        """Log debug message."""
        self.logger.debug(self._format(node, message))
    
    def warning(self, node: str, message: str) -> None:
        """Log warning message."""
        self.logger.warning(self._format(node, message))
    
    def error(self, node: str, message: str, exc_info: bool = False) -> None:
        """Log error message."""
        self.logger.error(self._format(node, message), exc_info=exc_info)
    
    def node_start(self, node: str, details: str = "") -> None:
        """Log node start (for timing/tracing)."""
        msg = f"Starting{': ' + details if details else ''}"
        self.info(node, msg)
    
    def node_end(self, node: str, duration_ms: float, success: bool = True) -> None:
        """Log node completion with timing."""
        status = "completed" if success else "failed"
        self.info(node, f"{status.capitalize()} in {duration_ms:.0f}ms")
    
    def node_error(self, node: str, error: Exception) -> None:
        """Log node error with exception details."""
        self.error(node, f"Error: {type(error).__name__}: {str(error)}", exc_info=True)


def log_node_timing(rag_name: str, node_name: str):
    """
    Decorator to automatically log node start/end with timing.
    
    Args:
        rag_name: RAG system name
        node_name: Node name
        
    Example:
        >>> @log_node_timing("IndexRAG", "Node1")
        ... def classify_intent_node(state):
        ...     # ... node logic
        ...     return state
    """
    logger = RAGLogger(rag_name)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.node_start(node_name)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.node_end(node_name, duration_ms, success=True)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.node_end(node_name, duration_ms, success=False)
                logger.node_error(node_name, e)
                raise
        
        return wrapper
    return decorator


# Pre-configured loggers for each RAG system
class IndexRAGLogger(RAGLogger):
    """Logger for Index RAG workflow."""
    def __init__(self):
        super().__init__("IndexRAG")


class StandardsRAGLogger(RAGLogger):
    """Logger for Standards RAG workflow."""
    def __init__(self):
        super().__init__("StandardsRAG")


class StrategyRAGLogger(RAGLogger):
    """Logger for Strategy RAG workflow."""
    def __init__(self):
        super().__init__("StrategyRAG")


class OrchestratorLogger(RAGLogger):
    """Logger for EnGenie Chat Orchestrator."""
    def __init__(self):
        super().__init__("Orchestrator")


# Convenience function to get logger by name
def get_rag_logger(rag_name: str) -> RAGLogger:
    """Get a RAG logger by name."""
    return RAGLogger(rag_name)
