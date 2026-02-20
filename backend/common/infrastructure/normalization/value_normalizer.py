# agentic/infrastructure/normalization/value_normalizer.py
# =============================================================================
# CENTRALIZED VALUE NORMALIZATION
# =============================================================================
#
# Single source of truth for specification value normalization.
# Consolidated from:
# - deep_agent/processing/value_normalizer.py
# - standards/generation/normalizer.py
#
# =============================================================================

import re
import logging
from typing import Optional, Tuple, List, Any, Dict

from .patterns import (
    get_compiled_leading_patterns,
    get_compiled_trailing_patterns,
    get_compiled_standards_pattern,
    get_compiled_na_patterns,
    get_compiled_technical_patterns,
)
from .validators import is_valid_spec_value, get_value_confidence_score

logger = logging.getLogger(__name__)


# =============================================================================
# VALUE NORMALIZER CLASS
# =============================================================================

class ValueNormalizer:
    """
    Post-processor that normalizes extracted specification values.

    Strips leading phrases, removes LLM artifacts, and ensures
    values are in proper technical specification format.
    """

    def __init__(self):
        """Initialize the ValueNormalizer with compiled patterns."""
        self.leading_patterns = get_compiled_leading_patterns()
        self.trailing_patterns = get_compiled_trailing_patterns()
        self.standards_pattern = get_compiled_standards_pattern()
        logger.debug("[ValueNormalizer] Initialized with pattern matching")

    def normalize(self, value: str, field_name: str = "") -> Tuple[str, List[str]]:
        """
        Normalize a specification value.

        Args:
            value: The raw extracted value
            field_name: Optional field name for context-aware processing

        Returns:
            Tuple of (normalized_value, extracted_standards_references)
        """
        if not value:
            return "", []

        original = str(value).strip()
        normalized = original
        standards_refs = []

        # Step 1: Extract standards references before stripping
        standards_matches = self.standards_pattern.findall(normalized)
        if standards_matches:
            standards_refs = [s.upper().replace("  ", " ").strip() for s in standards_matches]
            standards_refs = list(set(standards_refs))

        # Step 2: Strip leading phrases
        for pattern in self.leading_patterns:
            normalized = pattern.sub('', normalized, count=1)

        # Step 3: Strip trailing phrases
        for pattern in self.trailing_patterns:
            normalized = pattern.sub('', normalized)

        # Step 4: Clean up whitespace and punctuation
        normalized = normalized.strip()
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
        normalized = re.sub(r'^[,;:\s\-•*]+', '', normalized)  # Strip leading punctuation
        normalized = re.sub(r'[,;:\s]+$', '', normalized)  # Strip trailing punctuation

        # Step 5: Remove markdown/formatting artifacts
        normalized = re.sub(r'\*+', '', normalized)
        normalized = re.sub(r'`+', '', normalized)
        normalized = re.sub(r'#+\s*', '', normalized)

        # Step 6: Clean up orphaned parentheses/brackets
        normalized = re.sub(r'\(\s*\)', '', normalized)
        normalized = re.sub(r'\[\s*\]', '', normalized)

        # Step 7: Final trim
        normalized = normalized.strip()

        if normalized != original:
            logger.debug(f"[ValueNormalizer] Normalized '{field_name}': '{original[:50]}...' -> '{normalized[:50]}...'")

        return normalized, standards_refs

    def extract_and_validate(self, value: str, field_name: str = "") -> Tuple[Optional[str], List[str]]:
        """
        Normalize a value and validate it in one step.

        Args:
            value: The raw extracted value
            field_name: Optional field name for context

        Returns:
            Tuple of (normalized_value or None if invalid, standards_references)
        """
        normalized, refs = self.normalize(value, field_name)

        if normalized and is_valid_spec_value(normalized):
            return normalized, refs

        return None, refs

    def extract_and_validate_with_confidence(
        self, value: str, field_name: str = "", min_confidence: float = 0.3
    ) -> Tuple[Optional[str], List[str], float]:
        """
        Normalize a value and return with confidence score instead of binary rejection.

        This variant allows acceptance of borderline values with lower confidence scores,
        useful for maximizing specification extraction from standards documents.

        Args:
            value: The raw extracted value
            field_name: Optional field name for context
            min_confidence: Minimum confidence threshold to accept (default 0.3 = 30%)

        Returns:
            Tuple of (normalized_value or None, standards_references, confidence_score 0-1)
        """
        normalized, refs = self.normalize(value, field_name)

        if not normalized:
            return None, refs, 0.0

        confidence = get_value_confidence_score(normalized)

        # Accept if confidence meets minimum threshold
        if confidence >= min_confidence:
            return normalized, refs, confidence

        return None, refs, confidence


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_normalizer: Optional[ValueNormalizer] = None


def get_value_normalizer() -> ValueNormalizer:
    """Get the singleton ValueNormalizer instance."""
    global _normalizer
    if _normalizer is None:
        _normalizer = ValueNormalizer()
    return _normalizer


# =============================================================================
# VALUE CLEANING FUNCTIONS
# =============================================================================

def clean_value(value: str) -> str:
    """
    Clean a specification value by removing descriptive prefixes and normalizing.

    Args:
        value: The raw value string

    Returns:
        Cleaned value or "N/A"
    """
    if not isinstance(value, str):
        return str(value) if value is not None else "N/A"

    value = value.strip()

    # Check if it's explicitly N/A
    for pattern in get_compiled_na_patterns():
        if pattern.match(value):
            return "N/A"

    # Handle N/A with parenthetical explanation
    na_with_info = re.match(r'^N/A\s*\((.+)\)$', value, re.IGNORECASE | re.DOTALL)
    if na_with_info:
        inner = na_with_info.group(1).strip()

        # Check if the inner content contains actual values we can extract
        tech_values = extract_technical_values(inner)
        if tech_values:
            return tech_values

        return "N/A"

    # Remove document references like [51948133245220†L3910-L3925]
    value = re.sub(r'\[\d+[†\u2020]L\d+-L\d+\]', '', value)
    value = re.sub(r'【\d+[†\u2020]L\d+-L\d+】', '', value)

    # Clean up trailing punctuation
    value = value.rstrip('.').strip()

    # Normalize whitespace
    value = ' '.join(value.split())

    return value if value else "N/A"


def extract_technical_values(text: str) -> Optional[str]:
    """
    Extract technical specification values from descriptive text.

    Args:
        text: Text that may contain embedded technical values

    Returns:
        Extracted values or None if no values found
    """
    # Common patterns for technical values
    patterns = [
        # Protocol lists: "HART, Foundation Fieldbus, Profibus PA"
        r'((?:HART|Modbus|Profibus|Foundation Fieldbus|EtherNet/IP|DeviceNet|BACnet)(?:\s*,\s*(?:HART|Modbus|Profibus|Foundation Fieldbus|EtherNet/IP|DeviceNet|BACnet))*)',

        # Thermocouple types: "Type K, J, T, N, E, S, R, B"
        r'(Type\s+[A-Z](?:\s*,\s*[A-Z])+)',  # Multiple types like "Type K, J, T"
        r'(Type\s+[KJTNERSB])',  # Single type like "Type K"

        # Signal types: "4-20mA", "0-10V"
        r'(\d+-\d+\s*mA(?:\s*DC)?)',
        r'(\d+-\d+\s*V(?:\s*DC)?)',

        # Temperature ranges: "-40 to 85°C"
        r'(-?\d+\s*(?:to|-)\s*-?\d+\s*°?[CF])',

        # Pressure: "0-350 bar", "1500 psi"
        r'(\d+(?:\.\d+)?\s*(?:bar|psi|kPa|MPa))',

        # IP ratings: "IP66", "IP67"
        r'(IP\d{2}(?:/IP\d{2})?)',

        # NEMA ratings: "NEMA 4X"
        r'(NEMA\s*\d+[A-Z]?)',

        # SIL ratings: "SIL 2", "SIL 3"
        r'(SIL\s*\d)',

        # Accuracy: "±0.1%", "±2°C"
        r'([±]\s*\d+(?:\.\d+)?\s*%)',
        r'([±]\s*\d+(?:\.\d+)?\s*°?[CF])',

        # Materials: "316L Stainless Steel", "Hastelloy C-276"
        r'(316L?\s*(?:Stainless\s*Steel|SS))',
        r'(Hastelloy\s*[A-Z]-?\d+)',
        r'(Inconel\s*\d+)',
    ]

    found_values = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found_values.extend(matches)

    if found_values:
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for v in found_values:
            v_clean = v.strip()
            if v_clean.lower() not in seen:
                seen.add(v_clean.lower())
                unique.append(v_clean)
        return ", ".join(unique)

    return None


def extract_value_from_nested(data: Any, preserve_ghost: bool = True) -> Tuple[Any, Dict[str, Any]]:
    """
    Extract the actual value from a nested structure.

    Args:
        data: The data which may be nested or flat
        preserve_ghost: If True, return confidence as ghost value

    Returns:
        Tuple of (extracted_value, ghost_metadata)
    """
    ghost_metadata = {}

    if data is None:
        return "N/A", ghost_metadata

    if isinstance(data, str):
        return data, ghost_metadata

    if isinstance(data, (int, float, bool)):
        return str(data), ghost_metadata

    if isinstance(data, dict):
        # Extract confidence and notes as ghost values
        if preserve_ghost:
            if "confidence" in data:
                ghost_metadata["_confidence"] = data["confidence"]
            if "note" in data:
                ghost_metadata["_note"] = data["note"]
            if "source" in data:
                ghost_metadata["_source"] = data["source"]

        # Look for value field
        if "value" in data:
            value = data["value"]
            if isinstance(value, str):
                return value, ghost_metadata
            elif isinstance(value, dict):
                # Recursively extract
                nested_val, nested_ghost = extract_value_from_nested(value, preserve_ghost)
                ghost_metadata.update(nested_ghost)
                return nested_val, ghost_metadata
            else:
                return str(value) if value is not None else "N/A", ghost_metadata

        # No value field - might be the spec itself as a dict
        return data, ghost_metadata

    if isinstance(data, list):
        # Join list values
        if all(isinstance(item, str) for item in data):
            return ", ".join(data), ghost_metadata
        return str(data), ghost_metadata

    return str(data), ghost_metadata


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def normalize_spec_value(value: str, field_name: str = "") -> Tuple[str, List[str]]:
    """
    Convenience function to normalize a specification value.

    Args:
        value: The raw extracted value
        field_name: Optional field name for context

    Returns:
        Tuple of (normalized_value, standards_references)
    """
    return get_value_normalizer().normalize(value, field_name)


def extract_and_validate_spec(value: str, field_name: str = "") -> Tuple[Optional[str], List[str]]:
    """
    Convenience function to normalize and validate in one step.

    Args:
        value: The raw extracted value
        field_name: Optional field name for context

    Returns:
        Tuple of (normalized_value or None if invalid, standards_references)
    """
    return get_value_normalizer().extract_and_validate(value, field_name)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ValueNormalizer",
    "get_value_normalizer",
    "normalize_spec_value",
    "extract_and_validate_spec",
    "clean_value",
    "extract_technical_values",
    "extract_value_from_nested",
]
