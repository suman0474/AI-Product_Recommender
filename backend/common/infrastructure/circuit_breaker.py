"""
Production-Grade Circuit Breaker Pattern

Consolidated implementation with graceful degradation, async support, and observability.
Prevents cascading failures when external services are unavailable.

Usage:
    # Manual usage
    breaker = get_circuit_breaker("gemini")
    try:
        result = breaker.call(api_function, arg1, arg2, fallback=cached_function)
    except CircuitBreakerError:
        # Handle circuit open
        pass
    
    # Decorator usage
    @circuit_protected("my_service", failure_threshold=5)
    def my_function():
        # Your code
        pass
"""

import time
import logging
import asyncio
import threading
from typing import Dict, Any, Callable, Optional, List
from enum import Enum
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation - requests pass through
    OPEN = "open"           # Service failing - reject requests immediately
    HALF_OPEN = "half_open" # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit is open and request is rejected."""
    
    def __init__(self, circuit_name: str, message: str = None, time_until_reset: float = 0):
        self.circuit_name = circuit_name
        self.time_until_reset = time_until_reset
        self.message = message or (
            f"Circuit '{circuit_name}' is open - service unavailable. "
            f"Retry in {time_until_reset:.0f}s"
        )
        super().__init__(self.message)


class CircuitBreaker:
    """
    Thread-safe circuit breaker with graceful degradation.
    
    Features:
    - Three-state pattern (CLOSED/OPEN/HALF_OPEN)
    - Fallback callback support for graceful degradation
    - Async/await support
    - Thread-safe with RLock
    - Observability hooks
    - Detailed statistics tracking
    
    Example:
        >>> breaker = CircuitBreaker("pinecone", failure_threshold=5)
        >>> def fallback():
        ...     return "cached_response"
        >>> result = breaker.call(risky_function, fallback=fallback)
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        success_threshold: int = 2,
        half_open_max_calls: int = 3,
        on_open: Optional[Callable] = None,
        on_close: Optional[Callable] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Identifier for this circuit (for logging)
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Seconds to wait before testing recovery (entering HALF_OPEN)
            success_threshold: Successes in HALF_OPEN state needed to close circuit
            half_open_max_calls: Maximum concurrent calls allowed in HALF_OPEN state
            on_open: Callback when circuit opens (for metrics/alerts)
            on_close: Callback when circuit closes (for metrics)
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.success_threshold = success_threshold
        self.half_open_max_calls = half_open_max_calls
        self.on_open = on_open
        self.on_close = on_close
        
        # State tracking
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        
        # Statistics
        self._total_calls = 0
        self._total_successes = 0
        self._total_failures = 0
        self._circuit_opened_count = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(
            f"[CIRCUIT:{name}] Initialized - "
            f"threshold={failure_threshold}, timeout={reset_timeout}s"
        )
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state (with automatic HALF_OPEN transition)."""
        with self._lock:
            # Auto-transition from OPEN to HALF_OPEN if timeout expired
            if self._state == CircuitState.OPEN and self._should_attempt_reset():
                self._transition_to_half_open()
            return self._state
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self.state == CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if not self._last_failure_time:
            return False
        return time.time() - self._last_failure_time >= self.reset_timeout
    
    def _transition_to_half_open(self):
        """Transition from OPEN to HALF_OPEN state."""
        logger.info(f"[CIRCUIT:{self.name}] OPEN → HALF_OPEN (testing recovery)")
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        self._successes = 0
    
    def _transition_to_open(self):
        """Transition to OPEN state (circuit tripped)."""
        if self._state != CircuitState.OPEN:
            self._state = CircuitState.OPEN
            self._circuit_opened_count += 1
            self._last_failure_time = time.time()
            
            logger.error(
                f"[CIRCUIT:{self.name}] ⚠️  Circuit OPEN - "
                f"Failures: {self._failures}/{self.failure_threshold}, "
                f"Will retry in {self.reset_timeout}s"
            )
            
            # Call observability hook
            if self.on_open:
                try:
                    self.on_open(self.name, self.get_stats())
                except Exception as e:
                    logger.error(f"[CIRCUIT:{self.name}] on_open callback failed: {e}")
    
    def _transition_to_closed(self):
        """Transition to CLOSED state (recovered)."""
        if self._state != CircuitState.CLOSED:
            logger.info(
                f"[CIRCUIT:{self.name}] ✓ Circuit CLOSED (recovered) - "
                f"Success count: {self._successes}"
            )
            
            # Call observability hook
            if self.on_close:
                try:
                    self.on_close(self.name, self.get_stats())
                except Exception as e:
                    logger.error(f"[CIRCUIT:{self.name}] on_close callback failed: {e}")
        
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._half_open_calls = 0
    
    def can_execute(self) -> bool:
        """
        Check if a request should be allowed.
        
        Returns:
            True if request can proceed, False if circuit is open
        """
        with self._lock:
            current_state = self.state  # This auto-transitions to HALF_OPEN if timeout expired
            
            if current_state == CircuitState.CLOSED:
                return True
            
            if current_state == CircuitState.OPEN:
                return False
            
            # HALF_OPEN - limit concurrent calls
            if self._half_open_calls >= self.half_open_max_calls:
                return False
            
            return True
    
    def call(
        self,
        func: Callable,
        *args,
        fallback: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        """
        Execute function through circuit breaker with graceful degradation.
        
        Args:
            func: Function to call
            *args: Positional arguments for func
            fallback: Optional fallback function to call if circuit is open
            **kwargs: Keyword arguments for func
        
        Returns:
            Result from func or fallback
        
        Raises:
            CircuitBreakerError: If circuit is open and no fallback provided
            Exception: Original exception from function
        """
        with self._lock:
            self._total_calls += 1
            current_state = self.state
            
            # Circuit is OPEN - try fallback or fail fast
            if current_state == CircuitState.OPEN:
                time_until_reset = self._time_until_reset()
                
                if fallback:
                    logger.warning(
                        f"[CIRCUIT:{self.name}] Circuit OPEN, using fallback "
                        f"(retry in {time_until_reset:.0f}s)"
                    )
                    return fallback(*args, **kwargs)
                
                logger.warning(
                    f"[CIRCUIT:{self.name}] Request rejected - Circuit is OPEN "
                    f"(retry in {time_until_reset:.0f}s)"
                )
                raise CircuitBreakerError(
                    self.name,
                    time_until_reset=time_until_reset
                )
            
            # HALF_OPEN - limit concurrent calls
            if current_state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    if fallback:
                        logger.warning(
                            f"[CIRCUIT:{self.name}] HALF_OPEN limit reached, using fallback"
                        )
                        return fallback(*args, **kwargs)
                    
                    raise CircuitBreakerError(
                        self.name,
                        message=f"Circuit '{self.name}' is testing recovery. Please retry shortly."
                    )
                self._half_open_calls += 1
        
        # Execute the function (outside lock to allow concurrency)
        try:
            result = func(*args, **kwargs)
            
            # Success!
            with self._lock:
                self._on_success()
            
            return result
        
        except Exception as e:
            # Failure
            with self._lock:
                self._on_failure()
            raise
    
    async def call_async(
        self,
        func: Callable,
        *args,
        fallback: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        """
        Async version of call() for async functions.
        
        Args:
            func: Async function to call
            *args: Positional arguments
            fallback: Optional async fallback function
            **kwargs: Keyword arguments
        
        Returns:
            Result from func or fallback
        
        Raises:
            CircuitBreakerError: If circuit is open and no fallback
        """
        with self._lock:
            self._total_calls += 1
            current_state = self.state
            
            if current_state == CircuitState.OPEN:
                time_until_reset = self._time_until_reset()
                
                if fallback:
                    logger.warning(
                        f"[CIRCUIT:{self.name}] Circuit OPEN, using async fallback"
                    )
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback(*args, **kwargs)
                    return fallback(*args, **kwargs)
                
                raise CircuitBreakerError(self.name, time_until_reset=time_until_reset)
            
            if current_state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    if fallback:
                        if asyncio.iscoroutinefunction(fallback):
                            return await fallback(*args, **kwargs)
                        return fallback(*args, **kwargs)
                    raise CircuitBreakerError(self.name)
                self._half_open_calls += 1
        
        # Execute async function
        try:
            result = await func(*args, **kwargs)
            
            with self._lock:
                self._on_success()
            
            return result
        
        except Exception as e:
            with self._lock:
                self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        self._total_successes += 1
        self._successes += 1
        
        if self._state == CircuitState.HALF_OPEN:
            # Recovering - check if we can close the circuit
            logger.debug(
                f"[CIRCUIT:{self.name}] HALF_OPEN success "
                f"({self._successes}/{self.success_threshold})"
            )
            if self._successes >= self.success_threshold:
                self._transition_to_closed()
        
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            if self._failures > 0:
                logger.debug(
                    f"[CIRCUIT:{self.name}] Success - resetting failure count "
                    f"(was {self._failures})"
                )
                self._failures = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self._total_failures += 1
        self._failures += 1
        self._last_failure_time = time.time()
        
        logger.warning(
            f"[CIRCUIT:{self.name}] Failure {self._failures}/{self.failure_threshold}"
        )
        
        if self._state == CircuitState.HALF_OPEN:
            # Failed during recovery - reopen circuit
            logger.error(
                f"[CIRCUIT:{self.name}] Recovery failed - reopening circuit"
            )
            self._transition_to_open()
        
        elif self._failures >= self.failure_threshold:
            # Threshold reached - open circuit
            self._transition_to_open()
    
    def _time_until_reset(self) -> float:
        """Get seconds until circuit can attempt reset."""
        if not self._last_failure_time:
            return 0
        
        elapsed = time.time() - self._last_failure_time
        remaining = self.reset_timeout - elapsed
        return max(0, remaining)
    
    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        with self._lock:
            logger.info(f"[CIRCUIT:{self.name}] Manual reset")
            self._transition_to_closed()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "total_calls": self._total_calls,
                "total_successes": self._total_successes,
                "total_failures": self._total_failures,
                "current_failures": self._failures,
                "current_successes": self._successes,
                "circuit_opened_count": self._circuit_opened_count,
                "time_until_reset": self._time_until_reset() if self._state == CircuitState.OPEN else 0,
                "success_rate": (
                    (self._total_successes / self._total_calls * 100)
                    if self._total_calls > 0 else 0
                )
            }


# ============================================================================
# GLOBAL CIRCUIT BREAKER REGISTRY
# ============================================================================

_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.RLock()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    reset_timeout: float = 60.0,
    **kwargs
) -> CircuitBreaker:
    """
    Get or create a circuit breaker for a service.
    
    Args:
        name: Service identifier (e.g., "gemini", "pinecone", "blob_storage")
        failure_threshold: Failures before opening circuit
        reset_timeout: Seconds before testing recovery
        **kwargs: Additional arguments for CircuitBreaker
    
    Returns:
        CircuitBreaker instance for the service
    """
    with _registry_lock:
        if name not in _breakers:
            _breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                reset_timeout=reset_timeout,
                **kwargs
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


def circuit_protected(
    circuit_name: str,
    failure_threshold: int = 5,
    reset_timeout: float = 60.0
):
    """
    Decorator to protect a function with a circuit breaker.
    
    Args:
        circuit_name: Name for the circuit breaker
        failure_threshold: Number of failures before opening
        reset_timeout: Seconds before testing recovery
    
    Example:
        >>> @circuit_protected("pinecone", failure_threshold=5)
        ... def query_pinecone(query: str):
        ...     return pinecone.query(query)
    """
    def decorator(func: Callable):
        breaker = get_circuit_breaker(
            circuit_name,
            failure_threshold=failure_threshold,
            reset_timeout=reset_timeout
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker.call_async(func, *args, **kwargs)
        
        # Return async wrapper if function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'CircuitState',
    'CircuitBreakerError',
    'CircuitBreaker',
    'get_circuit_breaker',
    'get_all_circuit_states',
    'reset_all_circuits',
    'circuit_protected',
]
