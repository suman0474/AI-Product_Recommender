"""
Input Validator Module

Validates user queries to reject invalid/non-industrial queries before LLM processing.
Uses a 3-layer validation approach:
1. Rule-based validation (fast, no LLM)
2. Industrial keyword whitelist (prevents false positives)
3. LLM validation fallback (for ambiguous cases)
"""

import logging
import re
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# LAYER 1: INVALID QUERY PATTERNS (Rule-Based)
# =============================================================================

# Categories of invalid queries with keyword patterns
INVALID_PATTERNS = {
    "weather": [
        r"\bweather\b", r"\btemperature\b.*\b(today|tomorrow|forecast)\b",
        r"\brain\b.*\b(today|tomorrow)\b", r"\bforecast\b", r"\bclimate\b.*\b(today|change)\b",
        r"\bhot\b.*\btoday\b", r"\bcold\b.*\btoday\b", r"\bhumidity\b.*\b(today|now)\b",
        r"\bwind\b.*\b(speed|today)\b", r"\bstorm\b", r"\bhurricane\b", r"\bcyclone\b"
    ],
    "sports": [
        r"\bcricket\b.*\b(match|score|player|team)\b", r"\bfootball\b.*\b(match|score|player|team)\b",
        r"\bsoccer\b", r"\b(ipl|worldcup|world cup)\b", r"\btennis\b.*\bmatch\b",
        r"\bbasketball\b", r"\bhockey\b.*\bmatch\b", r"\b(wicket|goal|run)s?\b.*\bscore\b",
        r"\bplayer\b.*\b(score|stats|performance)\b", r"\bteam\b.*\b(won|lost|ranking)\b"
    ],
    "general_knowledge": [
        r"\bcapital\s+of\b", r"\bpresident\s+of\b", r"\bprime minister\b",
        r"\bpm\s+of\b", r"\bpopulation\s+of\b", r"\bwho\s+is\s+(president|pm|prime minister)\b",
        r"\bwhen\s+was\b.*\bborn\b", r"\bhow\s+old\s+is\b", r"\bhistory\s+of\b.*\b(country|city)\b",
        r"\bfamous\s+for\b", r"\btourist\b.*\bplace\b"
    ],
    "entertainment": [
        r"\bmovie\b", r"\bfilm\b.*\b(actor|actress|director)\b", r"\bactor\b", r"\bactress\b",
        r"\bsong\b", r"\bmusic\b.*\b(artist|singer)\b", r"\bcelebrity\b", r"\bsinger\b",
        r"\bnetflix\b", r"\byoutube\b.*\b(video|channel)\b", r"\btv\s+show\b"
    ],
    "food": [
        r"\brecipe\b", r"\bcooking\b.*\b(method|time)\b", r"\brestaurant\b",
        r"\bfood\b.*\b(near me|delivery)\b", r"\bdish\b.*\b(ingredients|recipe)\b"
    ]
}

# =============================================================================
# LAYER 2: INDUSTRIAL KEYWORDS (Whitelist)
# =============================================================================

# If query contains these, it's likely industrial - skip validation
INDUSTRIAL_KEYWORDS = [
    # Equipment types
    "transmitter", "sensor", "transducer", "meter", "gauge", "indicator",
    "valve", "actuator", "controller", "plc", "scada", "dcs", "hmi",
    "analyzer", "detector", "monitor", "recorder", "regulator",
    
    # Process parameters
    "pressure", "flow", "level", "temperature measurement", "ph", "conductivity",
    "turbidity", "viscosity", "density", "humidity measurement",
    
    # Standards and certifications
    "iec", "iso", "api", "asme", "atex", "sil", "certification",
    "standard", "compliance", "approval", "rating",
    
    # Industrial terms
    "process", "automation", "instrumentation", "control system",
    "fieldbus", "profibus", "modbus", "hart", "foundation fieldbus",
    "4-20ma", "output signal", "input signal", "calibration",
    
    # Vendors and brands  
    "rosemount", "emerson", "endress", "hauser", "siemens", "abb",
    "yokogawa", "honeywell", "schneider", "rockwell", "allen bradley",
    
    # Product specifications
    "accuracy", "range", "span", "output", "input", "connection",
    "material", "housing", "explosion proof", "intrinsically safe",
    "ip rating", "nema", "mounting", "installation"
]

# Compile regex patterns for efficiency
_invalid_pattern_cache = {}
for category, patterns in INVALID_PATTERNS.items():
    _invalid_pattern_cache[category] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

_industrial_keyword_pattern = re.compile(
    r'\b(' + '|'.join(re.escape(kw) for kw in INDUSTRIAL_KEYWORDS) + r')\b',
    re.IGNORECASE
)

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def is_obviously_invalid(query: str) -> Tuple[bool, Optional[str]]:
    """
    Layer 1: Rule-based validation using keyword patterns.
    
    Returns:
        Tuple of (is_invalid, rejection_reason)
        - (True, reason) if query matches invalid patterns
        - (False, None) if query passes rule-based check
    """
    query_lower = query.lower().strip()
    
    # Check each invalid category
    for category, patterns in _invalid_pattern_cache.items():
        for pattern in patterns:
            if pattern.search(query_lower):
                logger.info(f"[INPUT_VALIDATOR] Rule-based rejection: '{query[:60]}...' - matches {category} pattern")
                return True, f"This appears to be a {category}-related query. I can only assist with industrial automation topics."
    
    return False, None


def contains_industrial_keywords(query: str) -> bool:
    """
    Layer 2: Check if query contains industrial keywords (whitelist).
    
    Returns:
        True if query contains industrial terms (likely valid)
        False otherwise
    """
    matches = _industrial_keyword_pattern.findall(query)
    if matches:
        logger.info(f"[INPUT_VALIDATOR] Industrial keywords found: {matches[:3]}")
        return True
    return False


def validate_input_with_llm(query: str) -> Tuple[bool, Optional[str]]:
    """
    Layer 3: LLM-based validation for ambiguous queries.
    
    Uses existing classify_intent_tool with temperature=0.0 for deterministic
    INVALID_INPUT detection.
    
    Returns:
        Tuple of (is_valid, rejection_reason)
        - (False, reason) if LLM classifies as INVALID_INPUT
        - (True, None) if LLM allows the query
    """
    try:
        from .intent_tools import classify_intent_tool
        
        result = classify_intent_tool(
            user_input=query,
            current_step="validation",
            context="Input validation check"
        )
        
        intent = result.get("intent", "").upper()
        
        if intent == "INVALID_INPUT":
            reasoning = result.get("reasoning", "This query is outside my expertise area.")
            logger.info(f"[INPUT_VALIDATOR] LLM validation rejected: '{query[:60]}...' - {reasoning}")
            return False, reasoning
        
        logger.info(f"[INPUT_VALIDATOR] LLM validation passed: '{query[:60]}...' - intent: {intent}")
        return True, None
        
    except Exception as e:
        logger.error(f"[INPUT_VALIDATOR] LLM validation error: {e}")
        # On error, allow query to proceed (fail open)
        return True, None


def validate_query(query: str, skip_llm: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Main validation function with 3-layer approach.
    
    Args:
        query: User query to validate
        skip_llm: If True, skip LLM validation (faster but less accurate)
        
    Returns:
        Tuple of (is_valid, rejection_message)
        - (True, None) if query is valid
        - (False, message) if query should be rejected
    """
    query = query.strip()
    
    if not query:
        return False, "Please provide a query."
    
    # Layer 1: Rule-based validation
    is_invalid, reason = is_obviously_invalid(query)
    if is_invalid:
        return False, reason
    
    # Layer 2: Industrial keyword whitelist
    if contains_industrial_keywords(query):
        # Query contains industrial terms - likely valid, skip LLM
        logger.info(f"[INPUT_VALIDATOR] Query passed: contains industrial keywords")
        return True, None
    
    # Layer 3: LLM validation for ambiguous queries
    if not skip_llm:
        is_valid, reason = validate_input_with_llm(query)
        return is_valid, reason
    
    # If skip_llm=True and no obvious indicators, allow query
    logger.info(f"[INPUT_VALIDATOR] Query passed: no obvious invalid patterns (skip_llm=True)")
    return True, None


# =============================================================================
# REJECTION MESSAGE
# =============================================================================

OUT_OF_DOMAIN_MESSAGE = """I'm EnGenie, your industrial automation assistant. I can help with:

✓ Industrial automation products (transmitters, sensors, valves, PLCs, controllers)
✓ Technical specifications and certifications  
✓ Standards and compliance (IEC, ISO, ATEX, SIL, API, ASME)
✓ Vendor information and product comparisons
✓ Process instrumentation and control systems

Your query appears to be outside my expertise area. Please ask about industrial automation equipment, technical standards, or related topics."""
