# agentic/infrastructure/normalization/__init__.py
# =============================================================================
# CENTRALIZED NORMALIZATION MODULE
# =============================================================================
#
# Single source of truth for all specification normalization functions.
#
# This module consolidates previously duplicated code from:
# - deep_agent/processing/value_normalizer.py
# - standards/generation/normalizer.py
# - standards/generation/verifier.py
# - standards/shared/enrichment.py
# - deep_agent/specifications/aggregator.py
#
# Usage:
#     from common.infrastructure.normalization import (
#         is_valid_spec_value,
#         is_valid_spec_key,
#         normalize_key,
#         normalize_spec_value,
#         deduplicate_specs,
#     )
#
# =============================================================================

# Pattern constants
from .patterns import (
    TECHNICAL_PATTERNS,
    INVALID_PATTERNS,
    LEADING_PHRASES,
    TRAILING_PHRASES,
    STANDARDS_PATTERN,
    INVALID_EXACT_VALUES,
    HALLUCINATED_KEY_PATTERNS,
    INVALID_KEY_TERMS,
    NA_PATTERNS,
    get_compiled_technical_patterns,
    get_compiled_invalid_patterns,
    get_compiled_leading_patterns,
    get_compiled_trailing_patterns,
    get_compiled_standards_pattern,
    get_compiled_na_patterns,
)

# Validators
from .validators import (
    is_valid_spec_value,
    is_valid_spec_key,
    get_value_confidence_score,
    is_descriptive_text,
)

# Key normalization
from .key_normalizer import (
    STANDARD_KEY_MAPPINGS,
    normalize_key,
    normalize_spec_key,
    camel_to_snake,
    snake_to_camel,
    get_canonical_key,
)

# Value normalization
from .value_normalizer import (
    ValueNormalizer,
    get_value_normalizer,
    normalize_spec_value,
    extract_and_validate_spec,
    clean_value,
    extract_technical_values,
    extract_value_from_nested,
)

# Deduplication
from .deduplication import (
    deduplicate_specs,
    deduplicate_and_merge_list,
    deduplicate_by_normalized_key,
    clean_and_flatten_specs,
    merge_spec_sources,
    count_valid_specs,
    get_spec_count_summary,
)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Pattern constants
    "TECHNICAL_PATTERNS",
    "INVALID_PATTERNS",
    "LEADING_PHRASES",
    "TRAILING_PHRASES",
    "STANDARDS_PATTERN",
    "INVALID_EXACT_VALUES",
    "HALLUCINATED_KEY_PATTERNS",
    "INVALID_KEY_TERMS",
    "NA_PATTERNS",
    "get_compiled_technical_patterns",
    "get_compiled_invalid_patterns",
    "get_compiled_leading_patterns",
    "get_compiled_trailing_patterns",
    "get_compiled_standards_pattern",
    "get_compiled_na_patterns",
    # Validators
    "is_valid_spec_value",
    "is_valid_spec_key",
    "get_value_confidence_score",
    "is_descriptive_text",
    # Key normalization
    "STANDARD_KEY_MAPPINGS",
    "normalize_key",
    "normalize_spec_key",
    "camel_to_snake",
    "snake_to_camel",
    "get_canonical_key",
    # Value normalization
    "ValueNormalizer",
    "get_value_normalizer",
    "normalize_spec_value",
    "extract_and_validate_spec",
    "clean_value",
    "extract_technical_values",
    "extract_value_from_nested",
    # Deduplication
    "deduplicate_specs",
    "deduplicate_and_merge_list",
    "deduplicate_by_normalized_key",
    "clean_and_flatten_specs",
    "merge_spec_sources",
    "count_valid_specs",
    "get_spec_count_summary",
]
