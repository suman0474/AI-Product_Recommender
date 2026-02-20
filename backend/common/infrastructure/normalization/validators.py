# agentic/infrastructure/normalization/validators.py
# =============================================================================
# CENTRALIZED SPECIFICATION VALIDATORS
# =============================================================================
#
# Single source of truth for all specification validation functions.
# Consolidated from:
# - deep_agent/processing/value_normalizer.py
# - standards/shared/enrichment.py
# - deep_agent/processing/parallel/optimized_agent.py
#
# =============================================================================

import re
import logging
from typing import Any, List

from .patterns import (
    INVALID_EXACT_VALUES,
    HALLUCINATED_KEY_PATTERNS,
    INVALID_KEY_TERMS,
    get_compiled_invalid_patterns,
    get_compiled_technical_patterns,
)

logger = logging.getLogger(__name__)


# =============================================================================
# VALUE VALIDATION
# =============================================================================

def is_valid_spec_value(value: Any) -> bool:
    """
    Check if a specification value is valid.

    A value is considered invalid if it:
    - Is None or empty
    - Matches known invalid patterns (N/A, TBD, etc.)
    - Contains error messages or references
    - Is purely descriptive text without technical content
    - Is excessively long (>150 chars, likely an error message)

    Args:
        value: The value to validate

    Returns:
        True if valid, False otherwise
    """
    if value is None:
        return False

    # Handle dict values (spec with value/confidence structure)
    if isinstance(value, dict):
        if "error" in value and "value" not in value:
            return False
        if not value:
            return False
        if "value" in value:
            inner_val = str(value["value"]).lower().strip()
            if not inner_val or inner_val in INVALID_EXACT_VALUES:
                return False
        return True

    value_str = str(value).strip()
    if len(value_str) < 2:
        return False

    value_lower = value_str.lower()

    # Check exact match invalid values
    if value_lower in INVALID_EXACT_VALUES:
        return False

    # Substring checks for common invalid patterns
    invalid_substrings = [
        "not specified", "not applicable", "no value", "unknown"
    ]
    for substring in invalid_substrings:
        if substring in value_lower:
            if substring == "unknown" and len(value_lower) >= 10:
                continue  # Allow "unknown" in longer text
            return False

    # Error patterns to filter
    error_patterns = [
        "i found relevant", "temporarily unavailable", "api quota",
        "please try again", "service is", "error:", "failed to",
        ".docx)", ".pdf)", "standards documents", "ai service"
    ]
    if any(pattern in value_lower for pattern in error_patterns):
        return False

    # Too long (likely description or error message)
    if len(value_str) > 150:
        return False

    # Check for invalid/description patterns using compiled regex
    for pattern in get_compiled_invalid_patterns():
        if pattern.search(value_lower):
            logger.debug(f"[Validators] Invalid pattern matched: {value_str[:50]}")
            return False

    # Check for sentence structure (subject + verb = likely description)
    sentence_indicators = [
        r'^(the|a|an|this|that|it|there)\s+\w+\s+(is|are|has|have|was|were|can|will|should)',
        r'\b(is|are|has|have|was|were)\s+(a|an|the)\s+',
        r'^(i|we|you|they)\s+',
    ]
    for pattern in sentence_indicators:
        if re.search(pattern, value_lower, re.IGNORECASE):
            logger.debug(f"[Validators] Sentence structure detected: {value_str[:50]}")
            return False

    # Check for question structure
    if re.search(r'^(what|how|when|where|why|which|who)\b', value_lower, re.IGNORECASE):
        return False

    # Must contain at least one technical indicator for longer values
    has_technical = any(
        pattern.search(value_lower)
        for pattern in get_compiled_technical_patterns()
    )

    if not has_technical:
        # Short alphanumeric codes may be valid (e.g., "2-wire", "Type K")
        if len(value_str) < 30 and re.search(r'\d', value_str):
            word_count = len(value_str.split())
            if word_count <= 4:
                return True

        logger.debug(f"[Validators] No technical indicator: {value_str[:50]}")
        return False

    return True


def is_valid_spec_key(key: str) -> bool:
    """
    Check if a specification key is valid.

    Filters out:
    1. Hallucinated keys (e.g. 'loss_of_ambition')
    2. Generic keys (e.g. 'new_spec', 'item_1')
    3. Excessively long keys (>80 chars)
    4. Recursive patterns (e.g. 'self-self-self-...')

    Args:
        key: The key to validate

    Returns:
        True if valid, False otherwise
    """
    if not key or len(key) > 80:
        return False

    key_lower = key.lower()

    # Recursive/repetitive patterns
    if 'self-self-' in key_lower or key_lower.count('self-') > 3:
        return False

    # Hallucinated protection keys
    if any(pattern in key_lower for pattern in HALLUCINATED_KEY_PATTERNS):
        logger.debug(f"[Validators] Hallucinated key detected: {key}")
        return False

    # Generic/Invalid terms
    if any(term in key_lower for term in INVALID_KEY_TERMS):
        return False

    return True


# =============================================================================
# CONFIDENCE SCORING
# =============================================================================

def get_value_confidence_score(value: str) -> float:
    """
    Calculate a confidence score (0-1) for a value instead of binary pass/fail.

    Weighted scoring allows acceptance of borderline specifications with lower
    confidence scores, useful for maximizing specification extraction.

    Args:
        value: The value to score

    Returns:
        float: Confidence score 0-1 where:
        - 0.9-1.0: High confidence (clean technical spec)
        - 0.7-0.9: Good confidence (minor issues but acceptable)
        - 0.5-0.7: Moderate confidence (some uncertainties)
        - 0.3-0.5: Low confidence (multiple issues but possibly valid)
        - <0.3: Very low confidence (reject)
    """
    if not value or len(str(value).strip()) < 2:
        return 0.0

    value_str = str(value).strip()
    score = 0.5  # Base score: neutral

    # SOFT indicators (deduct but don't reject entirely)
    soft_indicators = [
        (r'\b(typically|usually|generally|normally|often|sometimes)\b', -0.15),
        (r'\b(approximately|about|around)\s+(?!\d)', -0.10),
        (r'\b(may|might|could|can)\s+(?:be|vary|differ)', -0.10),
        (r'\b(should\s+be|must\s+be)\b', -0.10),
        (r'\b(depends?\s+(?:on|upon))\b', -0.15),
        (r'\b(varies?\s+(?:by|with|depending|based))\b', -0.15),
    ]

    for pattern, penalty in soft_indicators:
        if re.search(pattern, value_str, re.IGNORECASE):
            score += penalty

    # HARD rejection patterns (immediate low score)
    hard_patterns = [
        (r'\b(TBD|TBC|T\.B\.D\.|T\.B\.C\.)\b', -0.5),
        (r'\b(not\s+(?:applicable|available|specified|defined|provided))\b', -0.5),
        (r'\b(N/?A|n/?a)\b', -0.5),
    ]

    for pattern, penalty in hard_patterns:
        if re.search(pattern, value_str, re.IGNORECASE):
            score += penalty
            if score < 0.2:  # Hard rejections
                return 0.0

    # Check for sentence structure (deduct for likely description)
    sentence_indicators = [
        r'^(the|a|an|this|that|it|there)\s+\w+\s+(is|are|has|have|was|were|can|will|should)',
        r'\b(is|are|has|have|was|were)\s+(a|an|the)\s+',
    ]
    for pattern in sentence_indicators:
        if re.search(pattern, value_str, re.IGNORECASE):
            score -= 0.20

    # BONUS for technical indicators (increase confidence)
    technical_bonus = [
        (r'[±\+\-]\s*\d+\.?\d*\s*%', 0.25),              # Accuracy like ±0.1%
        (r'[±\+\-]\s*\d+\.?\d*\s*(?:°[CF]|K)', 0.25),    # Temperature ±1°C
        (r'\d+\.?\d*\s*(?:to|[-–])\s*\d+\.?\d*', 0.20),  # Range like 0-400
        (r'\d+\s*(?:VDC|VAC|mA|V|Hz|bar|psi)', 0.20),   # Units
        (r'(?:IP|NEMA|SIL|ATEX)\s*\d+', 0.25),          # Safety ratings
        (r'(?:SS|316|304|Hastelloy|Inconel)\s*\d*', 0.20),  # Materials
        (r'(?:NPT|BSP|BSPT|DN|Tri-Clamp)', 0.15),       # Connections
        (r'(?:HART|Modbus|Profibus|4-20mA)', 0.20),     # Protocols
    ]

    for pattern, bonus in technical_bonus:
        if re.search(pattern, value_str):
            score += bonus

    # Ensure score stays in valid range
    return max(0.0, min(score, 1.0))


# =============================================================================
# DESCRIPTIVE TEXT DETECTION
# =============================================================================

def is_descriptive_text(text: str) -> bool:
    """
    Check if the text is purely descriptive (not a technical value).

    Args:
        text: Text to check

    Returns:
        True if text is descriptive, False if it contains technical values
    """
    if not text:
        return True

    descriptive_patterns = [
        r'typically\s+a\s+',
        r'will\s+be\s+used',
        r'is\s+provided\s+via',
        r'needs\s+to\s+be',
        r'should\s+be',
        r'may\s+be',
        r'can\s+be',
        r'refer\s+to',
        r'consult\s+',
        r'depending\s+on',
        r'varies\s+with',
        r'application[\s-]specific',
        r'application[\s-]dependent',
        r'options\s+for',
    ]

    for pattern in descriptive_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "is_valid_spec_value",
    "is_valid_spec_key",
    "get_value_confidence_score",
    "is_descriptive_text",
]
