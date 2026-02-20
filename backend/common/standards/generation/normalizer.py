# agentic/standards/generation/normalizer.py
# =============================================================================
# BACKWARD COMPATIBILITY WRAPPER
# =============================================================================
#
# This module now re-exports from the consolidated location:
# agentic.infrastructure.normalization
#
# All functionality has been moved to:
# - agentic/infrastructure/normalization/patterns.py
# - agentic/infrastructure/normalization/key_normalizer.py
# - agentic/infrastructure/normalization/value_normalizer.py
# - agentic/infrastructure/normalization/deduplication.py
#
# This file is kept for backward compatibility. New code should import from:
#     from common.infrastructure.normalization import (
#         normalize_key,
#         clean_value,
#         deduplicate_specs,
#         normalize_specification_output,
#     )
#
# =============================================================================

import logging
from typing import Dict, Any

# Re-export everything from the consolidated module
from common.infrastructure.normalization import (
    # Key normalization
    STANDARD_KEY_MAPPINGS,
    normalize_key,
    normalize_spec_key,
    # Value normalization
    clean_value,
    extract_technical_values,
    extract_value_from_nested,
    # Validators
    is_valid_spec_value,
    is_valid_spec_key,
    is_descriptive_text,
    # Deduplication
    deduplicate_specs,
    # Patterns
    NA_PATTERNS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SPECIFICATION NORMALIZATION (kept here for complex nested logic)
# =============================================================================

def normalize_specification_output(
    raw_specs: Dict[str, Any],
    preserve_ghost_values: bool = True
) -> Dict[str, Any]:
    """
    Normalize a raw specification dictionary to clean, flat key-value pairs.

    Transforms nested structures like:
    {
        "accuracy": {"value": "±1%", "confidence": 0.9, "note": "..."}
    }

    Into:
    {
        "accuracy": "±1%",
        "_ghost": {"accuracy": {"_confidence": 0.9, "_note": "..."}}
    }

    Args:
        raw_specs: The raw specification dictionary from LLM
        preserve_ghost_values: If True, preserve confidence/notes as ghost values

    Returns:
        Normalized specification dictionary with ghost values if requested
    """
    normalized = {}
    ghost_values = {}

    for key, value in raw_specs.items():
        # Skip internal/meta fields
        if key.startswith('_'):
            continue

        # Normalize the key
        norm_key = normalize_key(key)

        # Extract value from nested structure
        extracted_value, ghost = extract_value_from_nested(value, preserve_ghost_values)

        # Handle dict values (nested specs)
        if isinstance(extracted_value, dict):
            # Recursively normalize nested dicts
            nested_result = normalize_specification_output(
                extracted_value,
                preserve_ghost_values
            )
            # Merge nested results
            if "_ghost" in nested_result:
                for nk, nv in nested_result.pop("_ghost").items():
                    ghost_values[f"{norm_key}.{nk}"] = nv
            normalized[norm_key] = {
                normalize_key(k): v
                for k, v in nested_result.items()
            }
        else:
            # Clean the value
            cleaned = clean_value(str(extracted_value))
            normalized[norm_key] = cleaned

            # Store ghost metadata if present
            if ghost and preserve_ghost_values:
                ghost_values[norm_key] = ghost

    # Add ghost values if any
    if ghost_values and preserve_ghost_values:
        normalized["_ghost"] = ghost_values

    return normalized


def normalize_section_specs(
    section_specs: Dict[str, Any],
    section_name: str = "",
    preserve_ghost_values: bool = True
) -> Dict[str, Any]:
    """
    Normalize specifications within a specific section (e.g., mandatory, safety).

    Args:
        section_specs: Specs within a section
        section_name: Name of the section for logging
        preserve_ghost_values: If True, preserve ghost values

    Returns:
        Normalized section specifications
    """
    result = normalize_specification_output(section_specs, preserve_ghost_values)

    # Log normalization summary
    original_count = len(section_specs)
    final_count = len([k for k in result.keys() if not k.startswith('_')])

    if original_count != final_count:
        logger.debug(f"[Normalizer] Section '{section_name}': {original_count} -> {final_count} specs")

    return result


def normalize_full_item_specs(
    item: Dict[str, Any],
    preserve_ghost_values: bool = True
) -> Dict[str, Any]:
    """
    Normalize all specifications within an instrument/accessory item.

    Args:
        item: Full item dictionary with specifications
        preserve_ghost_values: If True, preserve ghost values

    Returns:
        Item with normalized specifications
    """
    result = item.copy()

    if "specifications" in result:
        specs = result["specifications"]
        normalized_specs = {}

        for section_name, section_data in specs.items():
            if isinstance(section_data, dict):
                normalized_specs[section_name] = normalize_section_specs(
                    section_data,
                    section_name,
                    preserve_ghost_values
                )
            else:
                normalized_specs[section_name] = section_data

        result["specifications"] = normalized_specs

    return result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Re-exported from infrastructure.normalization
    "normalize_key",
    "extract_value_from_nested",
    "clean_value",
    "extract_technical_values",
    "is_descriptive_text",
    "deduplicate_specs",
    "STANDARD_KEY_MAPPINGS",
    # Local functions
    "normalize_specification_output",
    "normalize_section_specs",
    "normalize_full_item_specs",
]
