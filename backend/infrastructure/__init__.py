"""
Infrastructure Package

Core infrastructure components for the backend application.
Includes circuit breakers, LLM fallback management, and observability.
"""

from infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    get_circuit_breaker,
    get_all_circuit_states,
    reset_all_circuits,
    circuit_protected
)

from infrastructure.llm_fallback_manager import (
    LLMFallbackManager,
    LLMProvider,
    get_llm_manager
)

__all__ = [
    # Circuit Breaker
    'CircuitBreaker',
    'CircuitBreakerError',
    'CircuitState',
    'get_circuit_breaker',
    'get_all_circuit_states',
    'reset_all_circuits',
    'circuit_protected',
    # LLM Fallback
    'LLMFallbackManager',
    'LLMProvider',
    'get_llm_manager',
]
