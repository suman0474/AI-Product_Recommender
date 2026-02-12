# agentic/standards_detector.py
# =============================================================================
# STANDARDS DETECTION UTILITY
# =============================================================================
#
# PURPOSE: Detect whether user input contains standards/safety indicators
# that require deep standards enrichment.
#
# This module provides centralized detection logic to determine if a request
# needs expensive standards extraction (Phase 3 in optimized_parallel_agent.py).
#
# Detection Sources:
# 1. Text keywords: SIL, ATEX, IEC, ISO, API, hazardous, etc.
# 2. Domain keywords: Oil & Gas, Pharma, explosion-proof, etc.
# 3. Provided requirements: sil_level, hazardous_area, domain fields
# 4. Parameters: Specific ranges/values indicating safety context
#
# =============================================================================

import logging
from typing import Dict, Any, List, Optional, TypedDict

logger = logging.getLogger(__name__)


class StandardsDetectionResult(TypedDict, total=False):
    """Result of standards detection analysis"""
    detected: bool  # Whether standards indicators were found
    confidence: float  # Confidence level (0.0-1.0)
    indicators: List[str]  # List of matched indicators
    detection_source: str  # Where detection came from (text, requirements, etc)
    matched_keywords: List[str]  # Actual keywords that matched


# =============================================================================
# STANDARD KEYWORDS AND PATTERNS
# =============================================================================

# Safety Standards - Functional Safety & Certification
SAFETY_STANDARDS = {
    "sil": ["sil", "sil1", "sil2", "sil3", "sil4", "iec 61508", "iec 61511", "isa 84"],
    "atex": ["atex", "iecex", "hazloc", "iec 60079"],
    "certification": ["ul listed", "fm approved", "csa", "ce marking", "rohs", "reach"],
    "other_standards": ["iso 9001", "iso 14001", "iso 17025", "nace", "nema", "ansi", "iec"],
}

# Pressure/Process Standards
PROCESS_STANDARDS = {
    "api": ["api", "api 520", "api 521", "api 526", "api 2000"],
    "asme": ["asme", "asme b31", "asme viii"],
    "pressure": ["psi", "bar", "mpa", "pressure rating"],
}

# Domain & Safety Keywords
DOMAIN_KEYWORDS = {
    "hazardous": [
        "hazardous", "hazardous area", "hazardous zone", "explosive atmosphere",
        "explosion-proof", "explosionproof", "atex zone 0", "atex zone 1", "atex zone 2",
        "combustible dust", "flammable gas", "flammable liquid",
    ],
    "oil_gas": [
        "oil", "gas", "crude oil", "petroleum", "refinery", "distillation",
        "pipeline", "upstream", "downstream", "midstream", "wellhead",
    ],
    "pharma": [
        "pharma", "pharmaceutical", "gmp", "cleanroom", "sterile", "biopharm",
        "vaccine", "drug manufacturing",
    ],
    "food": [
        "food", "beverage", "dairy", "meat", "sanitary", "washdown",
    ],
    "chemical": [
        "chemical", "petrochemical", "corrosive", "caustic", "acid",
    ],
}

# Safety & Operational Requirements
REQUIREMENT_KEYWORDS = {
    "sil_level": ["sil level", "sil rating", "sil certification"],
    "hazard": ["hazardous", "explosive", "flammable"],
    "pressure": ["pressure", "psi", "bar", "mpa"],
    "temperature": ["temperature", "cryogenic", "high temp"],
}

# IP Rating Standards
IP_RATINGS = [
    "ip rating", "ip65", "ip66", "ip67", "ip68", "ip69",
]

# Temperature/Pressure keywords that indicate critical specs
CRITICAL_SPEC_KEYWORDS = [
    "cryogenic", "deep cryogenic",
    "high temperature", "extreme temperature",
    "high pressure", "extreme pressure",
    "vacuum", "low pressure",
]


def _flatten_keywords(keyword_dict: Dict[str, List[str]]) -> List[str]:
    """Flatten nested keyword dictionary into single list."""
    flattened = []
    for category_list in keyword_dict.values():
        flattened.extend(category_list)
    return flattened


def _normalize_text(text: str) -> str:
    """Normalize text for keyword matching."""
    return text.lower().strip()


def _find_keywords_in_text(
    text: str,
    keywords: List[str],
    normalize: bool = True
) -> List[str]:
    """Find which keywords appear in text."""
    if not text:
        return []

    search_text = _normalize_text(text) if normalize else text.lower()
    matched = []

    for keyword in keywords:
        normalized_keyword = _normalize_text(keyword) if normalize else keyword.lower()
        # Check for word-boundary match to avoid partial matches
        # E.g., "SIL" shouldn't match "DETAIL"
        if f" {normalized_keyword} " in f" {search_text} " or \
           search_text.startswith(normalized_keyword + " ") or \
           search_text.endswith(f" {normalized_keyword}") or \
           search_text == normalized_keyword:
            matched.append(keyword)

    return matched


def detect_standards_indicators(
    user_input: str,
    provided_requirements: Optional[Dict[str, Any]] = None,
    confidence_threshold: float = 0.5
) -> StandardsDetectionResult:
    """
    Detects standards/strategy indicators from user input and requirements.

    Scans for:
    1. Text: Safety standards (SIL, ATEX, IEC, ISO, UL, FM, CSA, CE)
    2. Text: Domain keywords (Oil & Gas, Pharma, hazardous, explosion-proof)
    3. Text: Parameters (temperature range, pressure range, flow rate)
    4. Text: Strategy mentions (strategy, procurement rule, constraint)
    5. Requirements: sil_level, hazardous_area, domain, industry fields

    Detection Logic:
    - 3+ indicators → confidence=0.9 → detected=True
    - 2 indicators → confidence=0.7 → detected=True
    - 1 indicator → confidence=0.5 → detected=True
    - 0 indicators → confidence=0.0 → detected=False

    Args:
        user_input: User's text input
        provided_requirements: Requirements dict with sil_level, domain, etc.
        confidence_threshold: Minimum confidence to return detected=True

    Returns:
        StandardsDetectionResult with detection status and details
    """
    indicators: List[str] = []
    matched_keywords: List[str] = []
    detection_source: str = ""

    if not user_input:
        user_input = ""

    if provided_requirements is None:
        provided_requirements = {}

    # =========================================================================
    # 1. Check user_input for standards keywords
    # =========================================================================

    # Flatten all keyword categories
    all_safety_standards = _flatten_keywords(SAFETY_STANDARDS)
    all_domain_keywords = _flatten_keywords(DOMAIN_KEYWORDS)
    all_process_standards = _flatten_keywords(PROCESS_STANDARDS)

    matched_safety = _find_keywords_in_text(user_input, all_safety_standards)
    if matched_safety:
        indicators.append("safety_standards")
        matched_keywords.extend(matched_safety)
        detection_source = "text_standards"

    matched_process = _find_keywords_in_text(user_input, all_process_standards)
    if matched_process:
        indicators.append("process_standards")
        matched_keywords.extend(matched_process)
        if not detection_source:
            detection_source = "text_standards"

    matched_domain = _find_keywords_in_text(user_input, all_domain_keywords)
    if matched_domain:
        indicators.append("domain_keywords")
        matched_keywords.extend(matched_domain)
        if not detection_source:
            detection_source = "text_domain"

    # Check for IP ratings
    matched_ip = _find_keywords_in_text(user_input, IP_RATINGS)
    if matched_ip:
        indicators.append("ip_ratings")
        matched_keywords.extend(matched_ip)

    # Check for critical specs (cryogenic, high temp, high pressure)
    matched_critical = _find_keywords_in_text(user_input, CRITICAL_SPEC_KEYWORDS)
    if matched_critical:
        indicators.append("critical_specs")
        matched_keywords.extend(matched_critical)

    # =========================================================================
    # 2. Check provided_requirements for safety fields
    # =========================================================================

    if provided_requirements:
        # Check for explicit safety/domain fields
        if provided_requirements.get("sil_level"):
            indicators.append("sil_level_requirement")
            matched_keywords.append(f"sil_level={provided_requirements['sil_level']}")
            detection_source = "requirements_sil"

        if provided_requirements.get("hazardous_area"):
            indicators.append("hazardous_area_requirement")
            matched_keywords.append("hazardous_area=true")
            detection_source = "requirements_hazard"

        domain = provided_requirements.get("domain") or provided_requirements.get("industry")
        if domain:
            domain_lower = domain.lower()
            # Check if domain is one of the critical domains
            if any(keyword in domain_lower for keyword in
                   ["oil", "gas", "pharma", "chemical", "hazard", "explosive"]):
                indicators.append("domain_requirement")
                matched_keywords.append(f"domain={domain}")
                if not detection_source:
                    detection_source = "requirements_domain"

        if provided_requirements.get("atex_zone"):
            indicators.append("atex_zone_requirement")
            matched_keywords.append(f"atex_zone={provided_requirements['atex_zone']}")
            detection_source = "requirements_atex"

    # =========================================================================
    # 3. Calculate confidence
    # =========================================================================

    indicator_count = len(indicators)

    if indicator_count == 0:
        confidence = 0.0
        detected = False
    elif indicator_count == 1:
        confidence = 0.5
        detected = confidence >= confidence_threshold
    elif indicator_count == 2:
        confidence = 0.7
        detected = True
    else:  # 3+
        confidence = 0.9
        detected = True

    logger.info(
        f"[STANDARDS_DETECTION] Detected={detected}, Confidence={confidence:.2f}, "
        f"Indicators={indicator_count}, Source={detection_source or 'none'}"
    )

    if matched_keywords:
        logger.debug(f"[STANDARDS_DETECTION] Matched keywords: {matched_keywords[:5]}")

    return StandardsDetectionResult(
        detected=detected,
        confidence=confidence,
        indicators=indicators,
        detection_source=detection_source,
        matched_keywords=matched_keywords
    )


def should_run_standards_enrichment(
    user_input: str,
    provided_requirements: Optional[Dict[str, Any]] = None,
    confidence_threshold: float = 0.5
) -> bool:
    """
    Convenience function: Returns True if standards enrichment should run.

    Args:
        user_input: User's text input
        provided_requirements: Requirements dict
        confidence_threshold: Minimum confidence threshold

    Returns:
        bool: True if enrichment should run, False if should skip
    """
    result = detect_standards_indicators(
        user_input=user_input,
        provided_requirements=provided_requirements,
        confidence_threshold=confidence_threshold
    )
    return result["detected"]
