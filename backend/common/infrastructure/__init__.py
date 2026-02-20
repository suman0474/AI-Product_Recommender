"""
Infrastructure Package

Core infrastructure components for the backend application.
Includes circuit breakers, LLM fallback management, observability, and normalization.
"""

from common.infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    get_circuit_breaker,
    get_all_circuit_states,
    reset_all_circuits,
    circuit_protected
)

from common.infrastructure.llm_fallback_manager import (
    LLMFallbackManager,
    LLMProvider,
    get_llm_fallback_manager,
    get_llm_manager,  # backward-compat alias for get_llm_fallback_manager
)

# Normalization module - centralized spec validation and normalization
# Merged from common.agentic.infrastructure.normalization
try:
    from common.infrastructure.normalization import (
        # Validators
        is_valid_spec_value,
        is_valid_spec_key,
        get_value_confidence_score,
        is_descriptive_text,
        # Key normalization
        normalize_key,
        normalize_spec_key,
        # Value normalization
        ValueNormalizer,
        get_value_normalizer,
        normalize_spec_value,
        extract_and_validate_spec,
        clean_value,
        # Deduplication
        deduplicate_specs,
        deduplicate_and_merge_list,
        clean_and_flatten_specs,
        count_valid_specs,
    )
    HAS_NORMALIZATION = True
except ImportError:
    HAS_NORMALIZATION = False

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
    'get_llm_fallback_manager',
    'get_llm_manager',  # backward-compat alias
]

if HAS_NORMALIZATION:
    __all__.extend([
        'is_valid_spec_value', 'is_valid_spec_key', 'get_value_confidence_score', 'is_descriptive_text',
        'normalize_key', 'normalize_spec_key',
        'ValueNormalizer', 'get_value_normalizer', 'normalize_spec_value', 'extract_and_validate_spec', 'clean_value',
        'deduplicate_specs', 'deduplicate_and_merge_list', 'clean_and_flatten_specs', 'count_valid_specs',
    ])
