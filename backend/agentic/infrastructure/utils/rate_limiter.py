"""
Rate Limiter for LLM API Calls

Prevents quota exhaustion and provides backoff.
Uses token bucket algorithm for smooth rate limiting.
"""

import time
import logging
from typing import Dict, Optional
from threading import Lock

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    
    Implements a token bucket algorithm that allows bursting
    up to a maximum capacity while maintaining an average rate.
    
    Features:
    - Smooth rate limiting with token bucket
    - Configurable burst capacity
    - Blocking and non-blocking acquire modes
    - Thread-safe
    
    Example:
        >>> limiter = RateLimiter(calls_per_minute=60, burst=10)
        >>> if limiter.acquire():
        ...     make_api_call()
        ... else:
        ...     logger.warning("Rate limit exceeded")
    """
    
    def __init__(
        self, 
        calls_per_minute: int = 60, 
        burst: int = 10,
        name: str = "default"
    ):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_minute: Target rate (tokens added per minute)
            burst: Maximum token capacity (allows bursting)
            name: Identifier for logging
        """
        self.name = name
        self.rate = calls_per_minute / 60.0  # tokens per second
        self.burst = burst
        self.tokens = float(burst)  # Start at full capacity
        self.last_update = time.time()
        self._lock = Lock()
        
        logger.info(
            f"[RATE_LIMITER] {name} initialized: "
            f"{calls_per_minute} calls/min, burst={burst}"
        )
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time (called with lock held)."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_update = now
    
    def acquire(self, timeout: float = 30.0, blocking: bool = True) -> bool:
        """
        Acquire a token for making an API call.
        
        Args:
            timeout: Maximum seconds to wait if blocking
            blocking: If True, wait for a token; if False, return immediately
            
        Returns:
            True if token acquired, False if timeout or non-blocking fail
        """
        deadline = time.time() + timeout
        
        while True:
            with self._lock:
                self._refill()
                
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    logger.debug(
                        f"[RATE_LIMITER] {self.name} acquired token "
                        f"(remaining: {self.tokens:.1f})"
                    )
                    return True
            
            if not blocking:
                logger.warning(f"[RATE_LIMITER] {self.name} token unavailable (non-blocking)")
                return False
            
            if time.time() >= deadline:
                logger.warning(f"[RATE_LIMITER] {self.name} timeout waiting for token")
                return False
            
            # Wait a bit before retrying
            time.sleep(0.1)
    
    def try_acquire(self) -> bool:
        """Non-blocking token acquisition."""
        return self.acquire(blocking=False)
    
    def get_wait_time(self) -> float:
        """
        Get estimated wait time for a token.
        
        Returns:
            Seconds until a token will be available (0 if available now)
        """
        with self._lock:
            self._refill()
            
            if self.tokens >= 1.0:
                return 0.0
            
            # Calculate time to get 1 token
            tokens_needed = 1.0 - self.tokens
            return tokens_needed / self.rate
    
    def get_stats(self) -> Dict[str, any]:
        """Get rate limiter statistics."""
        with self._lock:
            self._refill()
            return {
                "name": self.name,
                "available_tokens": round(self.tokens, 2),
                "burst_capacity": self.burst,
                "rate_per_minute": round(self.rate * 60, 2),
                "wait_time_seconds": round(self.get_wait_time(), 2),
            }


# Global rate limiters registry
_limiters: Dict[str, RateLimiter] = {}
_registry_lock = Lock()


def get_rate_limiter(
    service: str, 
    calls_per_minute: int = 60,
    burst: int = 10
) -> RateLimiter:
    """
    Get or create a rate limiter for a service.
    
    Args:
        service: Service identifier (e.g., "gemini", "pinecone")
        calls_per_minute: Target rate
        burst: Burst capacity
        
    Returns:
        RateLimiter instance for the service
    """
    with _registry_lock:
        if service not in _limiters:
            _limiters[service] = RateLimiter(
                calls_per_minute=calls_per_minute,
                burst=burst,
                name=service
            )
        return _limiters[service]


def acquire_for_service(service: str, timeout: float = 30.0) -> bool:
    """
    Acquire a rate limit token for a service.
    
    Args:
        service: Service identifier
        timeout: Maximum wait time
        
    Returns:
        True if acquired, False if timeout
    """
    limiter = get_rate_limiter(service)
    return limiter.acquire(timeout=timeout)


def get_all_limiter_stats() -> Dict[str, Dict]:
    """Get statistics for all registered rate limiters."""
    with _registry_lock:
        return {name: limiter.get_stats() for name, limiter in _limiters.items()}


# Default rate limits for common services
DEFAULT_LIMITS = {
    "gemini": {"calls_per_minute": 60, "burst": 15},
    "gemini_pro": {"calls_per_minute": 30, "burst": 10},
    "pinecone": {"calls_per_minute": 100, "burst": 20},
    "blob_storage": {"calls_per_minute": 120, "burst": 30},
}


def init_default_limiters() -> None:
    """Initialize rate limiters with default configurations."""
    for service, config in DEFAULT_LIMITS.items():
        get_rate_limiter(service, **config)
    logger.info(f"[RATE_LIMITER] Initialized {len(DEFAULT_LIMITS)} default limiters")


# Rate-limited decorator
def rate_limited(service: str, timeout: float = 30.0):
    """
    Decorator to rate-limit a function.
    
    Args:
        service: Service name for rate limiting
        timeout: Maximum wait time for rate limit token
        
    Example:
        >>> @rate_limited("gemini")
        ... def call_gemini(prompt):
        ...     return llm.invoke(prompt)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not acquire_for_service(service, timeout):
                raise RateLimitExceededError(f"Rate limit exceeded for {service}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded and timeout occurs."""
    pass
