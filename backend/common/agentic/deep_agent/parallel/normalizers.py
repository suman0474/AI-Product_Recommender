# agentic/deep_agent/normalizers.py
# =============================================================================
# UNIFIED SPECIFICATION NORMALIZER MODULE
# =============================================================================
#
# Consolidated from:
# - value_normalizer.py: Core value cleaning and validation
# - spec_output_normalizer.py: Output structure normalization
# - spec_verifier.py: Verification and re-extraction agents
#
# =============================================================================

# Re-export everything from the original modules for backward compatibility
# This allows a gradual migration - imports from the old files still work,
# but new code can import from this unified module.

from common.infrastructure.normalization import (
    ValueNormalizer,
    get_value_normalizer,
    normalize_spec_value,
    is_valid_spec_value,
    extract_and_validate_spec,
    is_valid_spec_key,
)

from common.standards.generation.normalizer import (
    normalize_key,
    extract_value_from_nested,
    clean_value,
    extract_technical_values,
    is_descriptive_text,
    normalize_specification_output,
    normalize_section_specs,
    normalize_full_item_specs,
    deduplicate_specs,
    STANDARD_KEY_MAPPINGS,
)

from common.standards.generation.verifier import (
    SpecVerifierAgent,
    DescriptionExtractorAgent,
    verify_and_reextract_specs,
)


# =============================================================================
# EXPORTS - All normalization functionality in one place
# =============================================================================

__all__ = [
    # Value Normalizer (core cleaning)
    "ValueNormalizer",
    "get_value_normalizer",
    "normalize_spec_value",
    "is_valid_spec_value",
    "extract_and_validate_spec",
    
    # Output Normalizer (structure formatting)
    "normalize_key",
    "extract_value_from_nested",
    "clean_value",
    "extract_technical_values",
    "is_descriptive_text",
    "normalize_specification_output",
    "normalize_section_specs",
    "normalize_full_item_specs",
    "deduplicate_specs",
    "STANDARD_KEY_MAPPINGS",
    
    # Spec Verifier (verification agents)
    "SpecVerifierAgent",
    "DescriptionExtractorAgent",
    "verify_and_reextract_specs",
]
