# agentic/infrastructure/normalization/deduplication.py
# =============================================================================
# CENTRALIZED DEDUPLICATION FUNCTIONS
# =============================================================================
#
# Single source of truth for specification deduplication.
# Consolidated from:
# - standards/generation/normalizer.py
# - deep_agent/specifications/aggregator.py
# - standards/shared/enrichment.py
#
# =============================================================================

import logging
from typing import Dict, Any, List, Optional

from .key_normalizer import normalize_key, normalize_spec_key
from .validators import is_valid_spec_value, is_valid_spec_key

logger = logging.getLogger(__name__)


# =============================================================================
# SPEC DEDUPLICATION
# =============================================================================

def deduplicate_specs(specs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove duplicate specifications across sections.

    If the same key appears in multiple sections with identical values,
    keep only the first occurrence.

    Args:
        specs: Specifications dictionary with multiple sections

    Returns:
        Deduplicated specifications
    """
    if not isinstance(specs, dict):
        return specs

    seen_values = {}  # key -> (section, value)
    result = {}

    for section, section_data in specs.items():
        if not isinstance(section_data, dict):
            result[section] = section_data
            continue

        deduped_section = {}
        for key, value in section_data.items():
            if key.startswith('_'):
                deduped_section[key] = value
                continue

            # Check for duplicates
            if key in seen_values:
                prev_section, prev_value = seen_values[key]
                if prev_value == value:
                    logger.debug(f"[Deduplication] Removing duplicate: {key} (same value in {prev_section})")
                    continue

            seen_values[key] = (section, value)
            deduped_section[key] = value

        result[section] = deduped_section

    return result


def deduplicate_and_merge_list(specs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge list of spec dictionaries with priority strategies.
    Earlier dictionaries in the list have higher priority.

    Args:
        specs_list: List of spec dictionaries to merge

    Returns:
        Merged and deduplicated specs
    """
    merged = {}
    filtered_count = 0

    # Iterate in reverse order so higher priority (earlier items) overwrite later ones
    for specs in reversed(specs_list):
        if not specs:
            continue

        for key, val in specs.items():
            # Skip error keys
            if key.lower() in ["error", "error_message", "exception"]:
                filtered_count += 1
                continue

            # Validate value before merging
            if not is_valid_spec_value(val):
                filtered_count += 1
                continue

            merged[key] = val

    if filtered_count > 0:
        logger.debug(f"[Deduplication] Filtered {filtered_count} invalid/error values")

    return merged


def deduplicate_by_normalized_key(specs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deduplicate specs by normalizing keys.

    Keeps the first occurrence when multiple keys normalize to the same value.

    Args:
        specs: Specifications dictionary

    Returns:
        Deduplicated specifications with normalized keys
    """
    result = {}
    seen_normalized = {}  # normalized_key -> original_key

    for key, value in specs.items():
        if key.startswith('_'):
            result[key] = value
            continue

        normalized = normalize_spec_key(key)

        if normalized in seen_normalized:
            original = seen_normalized[normalized]
            logger.debug(f"[Deduplication] Duplicate key: '{key}' -> '{normalized}' (keeping '{original}')")
            continue

        seen_normalized[normalized] = key
        result[key] = value

    return result


# =============================================================================
# SPEC CLEANING AND FLATTENING
# =============================================================================

def clean_and_flatten_specs(raw_specs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and flatten specifications from LLM response.

    Handles nested structures and filters out N/A values.

    Args:
        raw_specs: Raw specs from LLM

    Returns:
        Cleaned and flattened specs
    """
    clean_specs = {}

    for key, value in raw_specs.items():
        # Skip internal/metadata keys
        if key.startswith("_"):
            continue

        if isinstance(value, dict):
            # Check if it's a spec with value/confidence
            if "value" in value:
                clean_specs[key] = {
                    "value": value.get("value", str(value)),
                    "confidence": value.get("confidence", 0.7)
                }
            else:
                # Flatten nested dict
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, dict) and "value" in nested_value:
                        clean_specs[nested_key] = nested_value
                    elif nested_value and str(nested_value).lower() not in ["null", "none", "n/a"]:
                        clean_specs[nested_key] = {"value": str(nested_value), "confidence": 0.7}
        else:
            clean_specs[key] = {"value": str(value), "confidence": 0.7}

    # Filter out N/A values and invalid keys
    final_specs = {}
    for k, v in clean_specs.items():
        if not is_valid_spec_key(k):
            continue

        val_str = str(v.get("value", "")).lower()
        if val_str and val_str not in ["null", "none", "n/a", "", "not specified", "unknown"]:
            final_specs[k] = v

    return final_specs


def merge_spec_sources(
    user_specs: Dict[str, Any],
    standards_specs: Dict[str, Any],
    llm_specs: Dict[str, Any],
    priority: str = "user_first"
) -> Dict[str, Any]:
    """
    Merge specifications from multiple sources with priority.

    Args:
        user_specs: Specs from user input
        standards_specs: Specs from standards documents
        llm_specs: Specs generated by LLM
        priority: Priority order ("user_first", "standards_first", "llm_first")

    Returns:
        Merged specifications
    """
    if priority == "user_first":
        sources = [user_specs, standards_specs, llm_specs]
    elif priority == "standards_first":
        sources = [standards_specs, user_specs, llm_specs]
    else:  # llm_first
        sources = [llm_specs, standards_specs, user_specs]

    return deduplicate_and_merge_list(sources)


# =============================================================================
# SPEC COUNTING
# =============================================================================

def count_valid_specs(specs: Dict[str, Any]) -> int:
    """
    Count valid specifications in a dictionary.

    Filters out invalid values like N/A, None, etc.

    Args:
        specs: Specifications dictionary

    Returns:
        Count of valid specs
    """
    if not specs or not isinstance(specs, dict):
        return 0

    count = 0
    for key, value in specs.items():
        if key.startswith('_'):
            continue

        if isinstance(value, dict):
            # Nested section - count recursively
            if "value" in value:
                if is_valid_spec_value(value.get("value")):
                    count += 1
            else:
                count += count_valid_specs(value)
        elif is_valid_spec_value(value):
            count += 1

    return count


def get_spec_count_summary(specs: Dict[str, Any]) -> Dict[str, int]:
    """
    Get spec count summary by section.

    Args:
        specs: Specifications dictionary with sections

    Returns:
        Dict mapping section names to spec counts
    """
    summary = {}
    total = 0

    for section, section_data in specs.items():
        if section.startswith('_'):
            continue

        if isinstance(section_data, dict):
            count = count_valid_specs(section_data)
            summary[section] = count
            total += count

    summary["_total"] = total
    return summary


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "deduplicate_specs",
    "deduplicate_and_merge_list",
    "deduplicate_by_normalized_key",
    "clean_and_flatten_specs",
    "merge_spec_sources",
    "count_valid_specs",
    "get_spec_count_summary",
]
