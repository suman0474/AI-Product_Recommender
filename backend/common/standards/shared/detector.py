# agentic/standards/detector.py
# =============================================================================
# STANDARDS MODULE - DETECTION & CLASSIFICATION
# =============================================================================
#
# This file consolidates detection and classification functions from:
# - standards_detector.py (detect_standards_indicators, should_run_standards_enrichment)
# - domain_classifier.py (classify_domain, get_routed_documents, get_domain_for_product_type)
#
# =============================================================================

import logging
from typing import Dict, Any, List, Optional, TypedDict, Set

from .keywords import (
    StandardsDomain,
    DOMAIN_KEYWORDS,
    DOMAIN_TO_DOCUMENTS,
    PRODUCT_TYPE_TO_DOMAINS,
    SAFETY_STANDARD_KEYWORDS,
    PROCESS_STANDARD_KEYWORDS,
    DOMAIN_INDICATOR_KEYWORDS,
    IP_RATING_PATTERNS,
    CRITICAL_SPEC_KEYWORDS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class StandardsDetectionResult(TypedDict, total=False):
    """Result of standards detection analysis."""
    detected: bool  # Whether standards indicators were found
    confidence: float  # Confidence level (0.0-1.0)
    indicators: List[str]  # List of matched indicators
    detection_source: str  # Where detection came from (text, requirements, etc)
    matched_keywords: List[str]  # Actual keywords that matched


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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
        if (f" {normalized_keyword} " in f" {search_text} " or
            search_text.startswith(normalized_keyword + " ") or
            search_text.endswith(f" {normalized_keyword}") or
            search_text == normalized_keyword):
            matched.append(keyword)

    return matched


def _add_safety_if_needed(
    text: str,
    domains: List[StandardsDomain]
) -> List[StandardsDomain]:
    """Add SAFETY domain if safety keywords detected but not already included."""
    safety_keywords = ["sil", "atex", "iecex", "hazardous", "explosion", "zone 1", "zone 2", "zone 0"]

    if any(kw in text for kw in safety_keywords):
        if StandardsDomain.SAFETY not in domains:
            # Insert at beginning (highest priority)
            domains = [StandardsDomain.SAFETY] + list(domains)
            logger.info("[DomainClassifier] Added SAFETY domain (safety keywords detected)")

    return domains


# =============================================================================
# DETECTION FUNCTIONS (from standards_detector.py)
# =============================================================================

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
    - 3+ indicators -> confidence=0.9 -> detected=True
    - 2 indicators -> confidence=0.7 -> detected=True
    - 1 indicator -> confidence=0.5 -> detected=True
    - 0 indicators -> confidence=0.0 -> detected=False

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
    all_safety_standards = _flatten_keywords(SAFETY_STANDARD_KEYWORDS)
    all_domain_keywords = _flatten_keywords(DOMAIN_INDICATOR_KEYWORDS)
    all_process_standards = _flatten_keywords(PROCESS_STANDARD_KEYWORDS)

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
    matched_ip = _find_keywords_in_text(user_input, IP_RATING_PATTERNS)
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


def is_standards_related_question(question: str) -> bool:
    """
    Check if a question is related to standards (merged logic from standards_rag_enrichment.py).

    Args:
        question: The question text to check

    Returns:
        True if the question appears to be standards-related
    """
    result = detect_standards_indicators(question, confidence_threshold=0.3)
    return result["detected"]


# =============================================================================
# CLASSIFICATION FUNCTIONS (from domain_classifier.py)
# =============================================================================

def classify_domain(
    user_input: str,
    product_type: Optional[str] = None,
    max_domains: int = 3
) -> List[StandardsDomain]:
    """
    Classify input into relevant standards domain(s).

    STRATEGY: Tokenize -> Match Keywords -> Score Domains -> Return Top N

    Args:
        user_input: User's query or requirements text
        product_type: Optional detected product type (for direct mapping)
        max_domains: Maximum domains to return (default 3)

    Returns:
        List of StandardsDomain enums, sorted by relevance score
    """
    # Combine inputs for matching
    text = f"{user_input} {product_type or ''}".lower().strip()

    if not text:
        logger.warning("[DomainClassifier] Empty input, returning empty domain list")
        return []

    # Step 1: Try direct product type mapping first (fastest path)
    if product_type:
        product_lower = product_type.lower().strip()
        if product_lower in PRODUCT_TYPE_TO_DOMAINS:
            direct_domains = PRODUCT_TYPE_TO_DOMAINS[product_lower]
            logger.info(f"[DomainClassifier] Direct match for '{product_type}': {[d.value for d in direct_domains]}")

            # Always check for safety keywords even with direct match
            domains = _add_safety_if_needed(text, direct_domains)
            return domains[:max_domains]

    # Step 2: Keyword-based scoring
    domain_scores: Dict[StandardsDomain, int] = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            # Count occurrences (weighted by keyword length for specificity)
            if keyword in text:
                # Longer keywords = more specific = higher score
                weight = len(keyword.split())
                score += weight

        if score > 0:
            domain_scores[domain] = score

    if not domain_scores:
        logger.warning(f"[DomainClassifier] No domain match for: '{text[:100]}...'")
        return []

    # Step 3: Sort by score descending
    sorted_domains = sorted(
        domain_scores.keys(),
        key=lambda d: domain_scores[d],
        reverse=True
    )

    # Step 4: Always include SAFETY if SIL/ATEX/hazardous mentioned
    sorted_domains = _add_safety_if_needed(text, sorted_domains)

    result = sorted_domains[:max_domains]

    logger.info(
        f"[DomainClassifier] Classified '{text[:50]}...' -> "
        f"{[d.value for d in result]} (scores: {dict((d.value, domain_scores.get(d, 0)) for d in result)})"
    )

    return result


def get_routed_documents(domains: List[StandardsDomain]) -> List[str]:
    """
    Get list of document filenames to query for given domains.

    Args:
        domains: List of StandardsDomain values

    Returns:
        Deduplicated list of document filenames to query
    """
    if not domains:
        logger.warning("[DomainClassifier] No domains provided, returning empty document list")
        return []

    documents: Set[str] = set()

    for domain in domains:
        domain_docs = DOMAIN_TO_DOCUMENTS.get(domain, [])
        documents.update(domain_docs)

    result = list(documents)

    logger.info(
        f"[DomainClassifier] Routed {len(domains)} domains to {len(result)} documents: {result}"
    )

    return result


def get_domain_for_product_type(product_type: str) -> List[StandardsDomain]:
    """
    Quick lookup for product type to domain mapping.

    Args:
        product_type: Product type string

    Returns:
        List of relevant domains for this product type
    """
    product_lower = product_type.lower().strip()

    # Try direct mapping
    if product_lower in PRODUCT_TYPE_TO_DOMAINS:
        return PRODUCT_TYPE_TO_DOMAINS[product_lower]

    # Fallback to keyword classification
    return classify_domain("", product_type)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "StandardsDetectionResult",
    # Detection functions
    "detect_standards_indicators",
    "should_run_standards_enrichment",
    "is_standards_related_question",
    # Classification functions
    "classify_domain",
    "get_routed_documents",
    "get_domain_for_product_type",
    # Helpers (public for advanced use)
    "_flatten_keywords",
    "_normalize_text",
    "_find_keywords_in_text",
    "_add_safety_if_needed",
]
