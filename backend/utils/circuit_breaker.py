"""
Circuit Breaker Pattern Implementation

⚠️ DEPRECATED: This module is deprecated and will be removed in a future version.
Please use `infrastructure.circuit_breaker` instead:

    from infrastructure.circuit_breaker import CircuitBreaker, get_circuit_breaker

Migration guide:
- Old: get_gemini_circuit_breaker()
- New: get_circuit_breaker("gemini")

Prevents cascading failures when external services (like Google Gemini API) are down
or experiencing issues. The circuit breaker has three states:

1. CLOSED (normal): All requests pass through
2. OPEN (failing): All requests fail fast without calling the service
3. HALF_OPEN (testing): Allow limited requests to test if service recovered

This saves time, API quota, and provides better error messages to users.
"""

import logging
import time
import threading
import warnings
from typing import Callable, Any, Optional
from enum import Enum
from datetime import datetime, timedelta

# Issue deprecation warning
warnings.warn(
    "utils.circuit_breaker is deprecated. Use infrastructure.circuit_breaker instead. "
    "See module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject all requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is OPEN"""
    pass


class CircuitBreaker:
    """
    Circuit Breaker pattern implementation for protecting against cascading failures.

    Usage:
        breaker = CircuitBreaker(
            failure_threshold=5,
            timeout_seconds=60,
            expected_exception=QuotaExceededError
        )

        try:
            result = breaker.call(expensive_api_call, arg1, arg2)
        except CircuitBreakerError:
            # Circuit is open, service is down
            return fallback_response()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        expected_exception: type = Exception,
        half_open_max_calls: int = 3,
        name: str = "default"
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening circuit
            timeout_seconds: Seconds to wait before attempting recovery (OPEN -> HALF_OPEN)
            expected_exception: The exception type that triggers circuit breaker
            half_open_max_calls: Max calls to allow in HALF_OPEN state
            name: Name of this circuit breaker (for logging)
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.expected_exception = expected_exception
        self.half_open_max_calls = half_open_max_calls
        self.name = name

        # State tracking
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._circuit_opened_count = 0

        logger.info(
            f"[CIRCUIT_BREAKER:{self.name}] Initialized - "
            f"failure_threshold={failure_threshold}, timeout={timeout_seconds}s"
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN and self._should_attempt_reset():
                self._transition_to_half_open()
            return self._state

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if not self._last_failure_time:
            return False

        elapsed = datetime.now() - self._last_failure_time
        return elapsed.total_seconds() >= self.timeout_seconds

    def _transition_to_half_open(self):
        """Transition from OPEN to HALF_OPEN state"""
        with self._lock:
            logger.info(f"[CIRCUIT_BREAKER:{self.name}] OPEN -> HALF_OPEN (testing recovery)")
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0

    def _transition_to_open(self):
        """Transition to OPEN state (circuit tripped)"""
        with self._lock:
            if self._state != CircuitState.OPEN:
                self._state = CircuitState.OPEN
                self._circuit_opened_count += 1
                self._last_failure_time = datetime.now()
                logger.error(
                    f"[CIRCUIT_BREAKER:{self.name}] ⚠️ Circuit OPEN - "
                    f"Failures: {self._failure_count}/{self.failure_threshold}, "
                    f"Will retry in {self.timeout_seconds}s"
                )

    def _transition_to_closed(self):
        """Transition to CLOSED state (recovered)"""
        with self._lock:
            if self._state != CircuitState.CLOSED:
                logger.info(
                    f"[CIRCUIT_BREAKER:{self.name}] ✓ Circuit CLOSED (recovered) - "
                    f"Success count: {self._success_count}"
                )
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from function

        Raises:
            CircuitBreakerError: If circuit is OPEN
            Exception: Original exception from function
        """
        with self._lock:
            self._total_calls += 1

            # Check circuit state
            current_state = self.state

            if current_state == CircuitState.OPEN:
                # Fail fast - don't even try
                logger.warning(
                    f"[CIRCUIT_BREAKER:{self.name}] Request rejected - Circuit is OPEN "
                    f"(will retry in {self._time_until_reset():.0f}s)"
                )
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Service is unavailable. Retry in {self._time_until_reset():.0f}s"
                )

            if current_state == CircuitState.HALF_OPEN:
                # Limit calls in HALF_OPEN state
                if self._half_open_calls >= self.half_open_max_calls:
                    logger.warning(
                        f"[CIRCUIT_BREAKER:{self.name}] Request rejected - "
                        f"HALF_OPEN limit reached ({self._half_open_calls}/{self.half_open_max_calls})"
                    )
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is in HALF_OPEN state. "
                        f"Testing recovery, please retry in a few seconds."
                    )
                self._half_open_calls += 1

        # Execute the function
        try:
            result = func(*args, **kwargs)

            # Success!
            with self._lock:
                self._on_success()

            return result

        except self.expected_exception as e:
            # Expected failure - count towards circuit breaker
            with self._lock:
                self._on_failure()
            raise

        except Exception as e:
            # Unexpected exception - don't count towards circuit breaker
            logger.warning(
                f"[CIRCUIT_BREAKER:{self.name}] Unexpected exception (not counted): {type(e).__name__}"
            )
            raise

    def _on_success(self):
        """Handle successful call"""
        self._total_successes += 1
        self._success_count += 1

        if self._state == CircuitState.HALF_OPEN:
            # Recovering - check if we can close the circuit
            logger.info(
                f"[CIRCUIT_BREAKER:{self.name}] HALF_OPEN success "
                f"({self._success_count}/{self.half_open_max_calls})"
            )
            if self._success_count >= self.half_open_max_calls:
                self._transition_to_closed()
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            if self._failure_count > 0:
                logger.debug(
                    f"[CIRCUIT_BREAKER:{self.name}] Success - resetting failure count "
                    f"(was {self._failure_count})"
                )
                self._failure_count = 0

    def _on_failure(self):
        """Handle failed call"""
        self._total_failures += 1
        self._failure_count += 1
        self._last_failure_time = datetime.now()

        logger.warning(
            f"[CIRCUIT_BREAKER:{self.name}] Failure {self._failure_count}/{self.failure_threshold}"
        )

        if self._state == CircuitState.HALF_OPEN:
            # Failed during recovery - reopen circuit
            logger.error(
                f"[CIRCUIT_BREAKER:{self.name}] Recovery failed - reopening circuit"
            )
            self._transition_to_open()

        elif self._failure_count >= self.failure_threshold:
            # Threshold reached - open circuit
            self._transition_to_open()

    def _time_until_reset(self) -> float:
        """Get seconds until circuit can attempt reset"""
        if not self._last_failure_time:
            return 0

        elapsed = datetime.now() - self._last_failure_time
        remaining = self.timeout_seconds - elapsed.total_seconds()
        return max(0, remaining)

    def reset(self):
        """Manually reset circuit breaker to CLOSED state"""
        with self._lock:
            logger.info(f"[CIRCUIT_BREAKER:{self.name}] Manual reset")
            self._transition_to_closed()

    def get_stats(self) -> dict:
        """Get circuit breaker statistics"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "total_calls": self._total_calls,
                "total_successes": self._total_successes,
                "total_failures": self._total_failures,
                "current_failure_count": self._failure_count,
                "circuit_opened_count": self._circuit_opened_count,
                "time_until_reset": self._time_until_reset() if self._state == CircuitState.OPEN else 0,
                "success_rate": (
                    (self._total_successes / self._total_calls * 100)
                    if self._total_calls > 0 else 0
                )
            }


# ============================================================================
# GLOBAL CIRCUIT BREAKERS FOR COMMON SERVICES
# ============================================================================

# Google Gemini API circuit breaker
_gemini_breaker: Optional[CircuitBreaker] = None

def get_gemini_circuit_breaker() -> CircuitBreaker:
    """
    Get or create the global Gemini API circuit breaker.

    Returns:
        CircuitBreaker instance for Gemini API
    """
    global _gemini_breaker
    if _gemini_breaker is None:
        # Import here to avoid circular dependency
        try:
            from langchain_core.exceptions import LLMError
            exception_type = Exception  # Catch all API errors
        except ImportError:
            exception_type = Exception

        _gemini_breaker = CircuitBreaker(
            failure_threshold=5,  # Open after 5 consecutive failures
            timeout_seconds=60,  # Wait 60s before testing recovery
            expected_exception=exception_type,
            half_open_max_calls=2,  # Allow 2 test calls in HALF_OPEN
            name="gemini_api"
        )

    return _gemini_breaker


def reset_all_circuit_breakers():
    """Reset all global circuit breakers (for testing/admin use)"""
    global _gemini_breaker

    if _gemini_breaker:
        _gemini_breaker.reset()

    logger.info("[CIRCUIT_BREAKER] All circuit breakers reset")


def get_all_circuit_breaker_stats() -> dict:
    """Get statistics from all circuit breakers"""
    stats = {}

    if _gemini_breaker:
        stats["gemini_api"] = _gemini_breaker.get_stats()

    return stats
