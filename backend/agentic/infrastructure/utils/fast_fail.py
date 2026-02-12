"""
Shared Fast-Fail Utility for RAG Workflows

Detects unrecoverable errors and prevents wasteful retries.
Implements pattern detection for network failures, quota exhaustion, and service unavailability.
"""

import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

# =============================================================================
# FAST-FAIL PATTERNS: Non-LLM errors that should trigger immediate failure
# =============================================================================
# IMPORTANT: LLM-related errors (timeout, rate limits) are NOT included here
# because they have their own 10-minute timeout and retry logic in llm_fallback.py
# Fast-fail is for NON-LLM infrastructure errors only
# =============================================================================

UNRECOVERABLE_ERRORS: List[str] = [
    # Network connectivity issues (NON-LLM)
    "connection refused",
    "connection reset",
    "winerror 10061",  # Windows connection refused
    "econnrefused",
    "econnreset",
    "ehostunreach",
    "enetunreach",
    "no such host",
    "name or service not known",
    "getaddrinfo failed",
    "connection attempt failed",
    "network is unreachable",

    # Authentication/Authorization (immediate failure)
    "unauthorized",
    "forbidden",
    "invalid api key",
    "authentication failed",
    "access denied",

    # Data issues (retrying won't fix)
    "index not found",
    "namespace not found",
    "collection not found",
    "file not found",
    "no such file",
    "does not exist",

    # Configuration errors
    "missing setting",
    "invalid configuration",
    "missing required",
]

# =============================================================================
# PATTERNS THAT SHOULD NOT TRIGGER FAST-FAIL (LLM-related - have their own retry)
# =============================================================================
# These are handled by llm_fallback.py with 10-minute timeout and proper retries:
# - "timed out", "timeout expired", "connection timeout" (LLM can be slow)
# - "resource_exhausted", "quota exceeded", "rate limit exceeded" (LLM quotas)
# - "too many requests", "429" (LLM rate limits)
# - "service unavailable", "503", "502", "504" (LLM service issues)
# - "gateway timeout" (LLM gateway issues)
# =============================================================================


def should_fail_fast(error: Exception) -> Tuple[bool, str]:
    """
    Check if an error is unrecoverable and should trigger fast-fail.
    
    This prevents wasting time on retries when the underlying issue
    cannot be resolved by retrying (e.g., network down, quota exceeded).
    
    Args:
        error: The exception to analyze
        
    Returns:
        Tuple of (should_fail_fast: bool, reason: str)
        
    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     fail_fast, reason = should_fail_fast(e)
        ...     if fail_fast:
        ...         return early_exit_result
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    
    # Check error message against patterns
    for pattern in UNRECOVERABLE_ERRORS:
        if pattern in error_str or pattern in error_type:
            reason = f"Unrecoverable error detected: '{pattern}' in error"
            logger.warning(f"[FAST-FAIL] {reason}")
            return True, reason
    
    # Check for specific exception types
    unrecoverable_types = [
        "connectionerror",
        "connectionrefusederror",
        "connectionreseterror",
        "timeouterror",
        "sockettimeout",
        "socketerror",
        "sslerror",
        "urlerror",
        "authenticationerror",
        "permissionerror",
    ]
    
    if error_type in unrecoverable_types:
        reason = f"Unrecoverable exception type: {type(error).__name__}"
        logger.warning(f"[FAST-FAIL] {reason}")
        return True, reason
    
    return False, ""


def log_fast_fail(
    error: Exception, 
    rag_name: str, 
    max_retries: int,
    current_retry: int = 0
) -> None:
    """
    Log fast-fail event with time savings estimate.
    
    Args:
        error: The exception that triggered fast-fail
        rag_name: Name of the RAG system (for logging)
        max_retries: Maximum retries configured
        current_retry: Current retry count (how many already attempted)
    """
    skipped_retries = max_retries - current_retry
    # Estimate ~20 seconds average per retry attempt
    estimated_savings = skipped_retries * 20
    
    logger.warning(
        f"[FAST-FAIL] [{rag_name}] Skipping {skipped_retries} retries "
        f"(~{estimated_savings}s saved). Error: {type(error).__name__}: {str(error)[:100]}"
    )


def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is potentially retryable.
    
    This is the inverse of should_fail_fast - returns True for transient
    errors that might succeed on retry.
    
    Args:
        error: The exception to analyze
        
    Returns:
        True if the error might succeed on retry, False otherwise
    """
    fail_fast, _ = should_fail_fast(error)
    return not fail_fast


# Convenience function for use in state dictionaries
def check_and_set_fast_fail(state: dict, error: Exception, rag_name: str = "RAG") -> dict:
    """
    Check if fast-fail should be triggered and update state accordingly.
    
    Args:
        state: The workflow state dictionary (must have 'should_fail_fast' key)
        error: The exception to check
        rag_name: Name of the RAG system for logging
        
    Returns:
        The updated state dictionary
    """
    fail_fast, reason = should_fail_fast(error)
    
    if fail_fast:
        state['should_fail_fast'] = True
        state['fast_fail_reason'] = reason
        log_fast_fail(error, rag_name, state.get('max_retries', 2), state.get('retry_count', 0))
    
    return state
