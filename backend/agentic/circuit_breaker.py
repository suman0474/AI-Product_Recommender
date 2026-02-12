"""
Circuit Breaker Pattern for RAG Services

⚠️ DEPRECATED: This module is deprecated and will be removed in a future version.
Please use `infrastructure.circuit_breaker` instead:

    from infrastructure.circuit_breaker import CircuitBreaker, get_circuit_breaker, circuit_protected

The new implementation includes:
- Async support (call_async)
- Graceful degradation (fallback callbacks)
- Observability hooks (on_open, on_close)

Prevents cascading failures when external services are unavailable.
Implements the standard circuit breaker pattern with three states:
- CLOSED: Normal operation, requests pass through
- OPEN: Service failing, requests rejected immediately
- HALF_OPEN: Testing recovery, limited requests allowed
"""

import time
import logging
import warnings
from typing import Dict, Any, Callable, Optional
from enum import Enum
from threading import Lock
from functools import wraps

# Issue deprecation warning
warnings.warn(
    "agentic.circuit_breaker is deprecated. Use infrastructure.circuit_breaker instead. "
    "The new implementation supports async, fallbacks, and observability.",
    DeprecationWarning,
    stacklevel=2
)

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation - requests pass through
    OPEN = "open"           # Service failing - reject requests immediately
    HALF_OPEN = "half_open" # Testing if service recovered


class CircuitBreaker:
    """
    Thread-safe circuit breaker for external service calls.
    
    The circuit breaker monitors for failures and opens the circuit
    when a threshold is reached, preventing further calls to a
    failing service. After a timeout period, the circuit enters
    a half-open state to test if the service has recovered.
    
    Example:
        >>> breaker = CircuitBreaker("pinecone", failure_threshold=5)
        >>> if breaker.can_execute():
        ...     try:
        ...         result = pinecone.query(...)
        ...         breaker.record_success()
        ...     except Exception as e:
        ...         breaker.record_failure()
        ...         raise
        ... else:
        ...     raise CircuitOpenError("Pinecone circuit is open")
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Identifier for this circuit (for logging)
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Seconds to wait before testing recovery (entering HALF_OPEN)
            success_threshold: Successes in HALF_OPEN state needed to close circuit
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.success_threshold = success_threshold
        
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._last_failure_time = 0.0
        self._lock = Lock()
        
        logger.info(
            f"[CIRCUIT] {name} initialized: "
            f"threshold={failure_threshold}, timeout={reset_timeout}s"
        )
    
    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        with self._lock:
            return self._state
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self.state == CircuitState.OPEN
    
    def can_execute(self) -> bool:
        """
        Check if a request should be allowed.
        
        Returns:
            True if request can proceed, False if circuit is open
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            
            if self._state == CircuitState.OPEN:
                # Check if we should transition to HALF_OPEN
                if time.time() - self._last_failure_time >= self.reset_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._successes = 0
                    logger.info(f"[CIRCUIT] {self.name}: OPEN -> HALF_OPEN (testing recovery)")
                    return True
                return False
            
            # HALF_OPEN - allow requests to test recovery
            return True
    
    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._successes += 1
                logger.debug(
                    f"[CIRCUIT] {self.name}: HALF_OPEN success "
                    f"({self._successes}/{self.success_threshold})"
                )
                if self._successes >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failures = 0
                    logger.info(f"[CIRCUIT] {self.name}: HALF_OPEN -> CLOSED (recovered)")
            else:
                # Reset failure count on success
                self._failures = 0
    
    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN reopens the circuit
                self._state = CircuitState.OPEN
                logger.warning(f"[CIRCUIT] {self.name}: HALF_OPEN -> OPEN (recovery failed)")
            elif self._failures >= self.failure_threshold:
                # Threshold reached, open the circuit
                self._state = CircuitState.OPEN
                logger.warning(
                    f"[CIRCUIT] {self.name}: CLOSED -> OPEN "
                    f"(failure threshold {self.failure_threshold} reached)"
                )
    
    def reset(self) -> None:
        """Force reset the circuit to CLOSED state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._successes = 0
            logger.info(f"[CIRCUIT] {self.name}: Force reset to CLOSED")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failures": self._failures,
                "successes": self._successes,
                "failure_threshold": self.failure_threshold,
                "time_since_last_failure": (
                    time.time() - self._last_failure_time 
                    if self._last_failure_time > 0 
                    else None
                )
            }


class CircuitOpenError(Exception):
    """Raised when circuit is open and request is rejected."""
    
    def __init__(self, circuit_name: str, message: str = None):
        self.circuit_name = circuit_name
        self.message = message or f"Circuit '{circuit_name}' is open - service unavailable"
        super().__init__(self.message)


# Global circuit breakers registry
_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = Lock()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    reset_timeout: float = 60.0
) -> CircuitBreaker:
    """
    Get or create a circuit breaker for a service.
    
    Args:
        name: Service identifier (e.g., "pinecone", "gemini", "blob_storage")
        failure_threshold: Failures before opening circuit
        reset_timeout: Seconds before testing recovery
        
    Returns:
        CircuitBreaker instance for the service
    """
    with _registry_lock:
        if name not in _breakers:
            _breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                reset_timeout=reset_timeout
            )
        return _breakers[name]


def get_all_circuit_states() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all registered circuit breakers."""
    with _registry_lock:
        return {name: breaker.get_stats() for name, breaker in _breakers.items()}


def reset_all_circuits() -> None:
    """Force reset all circuit breakers to CLOSED state."""
    with _registry_lock:
        for breaker in _breakers.values():
            breaker.reset()
        logger.info(f"[CIRCUIT] Reset all {len(_breakers)} circuits")


def circuit_protected(circuit_name: str, failure_threshold: int = 5):
    """
    Decorator to protect a function with a circuit breaker.
    
    Args:
        circuit_name: Name for the circuit breaker
        failure_threshold: Number of failures before opening
        
    Example:
        >>> @circuit_protected("pinecone", failure_threshold=5)
        ... def query_pinecone(query: str):
        ...     return pinecone.query(query)
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            breaker = get_circuit_breaker(circuit_name, failure_threshold)
            
            if not breaker.can_execute():
                raise CircuitOpenError(circuit_name)
            
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        
        return wrapper
    return decorator
