"""
Retry Utilities with Exponential Backoff

Provides decorators and utilities for retrying failed operations with
exponential backoff and jitter to prevent thundering herd problems.
"""
import time
import random
import logging
import asyncio
from functools import wraps
from typing import Tuple, Type, Callable, Any, Optional

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    Decorator for retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delay
        retryable_exceptions: Tuple of exception types that trigger retry
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def call_external_api():
            return requests.get("https://api.example.com")
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"[RETRY] {func.__name__} failed after {max_retries + 1} attempts: {e}")
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay *= (0.5 + random.random())
                    
                    logger.warning(
                        f"[RETRY] {func.__name__} attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


def async_retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    Async decorator for retry with exponential backoff.
    
    Same as retry_with_backoff but for async functions.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"[RETRY] {func.__name__} failed after {max_retries + 1} attempts: {e}")
                        raise
                    
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    if jitter:
                        delay *= (0.5 + random.random())
                    
                    logger.warning(
                        f"[RETRY] {func.__name__} attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
            
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


class RetryConfig:
    """Configuration class for retry behavior."""
    
    # Preset configurations for common use cases
    QUICK = {"max_retries": 2, "base_delay": 0.5, "max_delay": 5.0}
    STANDARD = {"max_retries": 3, "base_delay": 1.0, "max_delay": 30.0}
    PATIENT = {"max_retries": 5, "base_delay": 2.0, "max_delay": 60.0}
    
    # Service-specific presets
    LLM_RETRY = {"max_retries": 3, "base_delay": 2.0, "max_delay": 30.0}
    VECTOR_STORE_RETRY = {"max_retries": 3, "base_delay": 1.0, "max_delay": 20.0}
    EXTERNAL_API_RETRY = {"max_retries": 4, "base_delay": 1.5, "max_delay": 45.0}
