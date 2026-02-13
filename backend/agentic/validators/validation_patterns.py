"""
Validation Patterns and Keywords

Shared patterns and keywords used across validation modules.
Extracted from legacy input_validator.py and consolidated here.
"""

import re
from typing import List, Pattern

# =============================================================================
# INDUSTRIAL KEYWORDS (Whitelist)
# =============================================================================

# If query contains these, it's likely industrial - useful for fast path validation
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

# =============================================================================
# INVALID QUERY PATTERNS (from legacy validator - for reference/fast path)
# =============================================================================

# These patterns can be used for a fast-path rule-based check before semantic classification
INVALID_QUERY_PATTERNS = {
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
# COMPILED PATTERNS (for performance)
# =============================================================================

# Compile industrial keyword pattern
_industrial_keyword_pattern = re.compile(
    r'\b(' + '|'.join(re.escape(kw) for kw in INDUSTRIAL_KEYWORDS) + r')\b',
    re.IGNORECASE
)

# Compile invalid patterns
_invalid_pattern_cache = {}
for category, patterns in INVALID_QUERY_PATTERNS.items():
    _invalid_pattern_cache[category] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def contains_industrial_keywords(query: str) -> bool:
    """
    Check if query contains industrial keywords (whitelist).
    
    This is a fast-path check - queries with industrial keywords
    are very likely to be valid and can skip deeper validation.
    
    Args:
        query: User query string
        
    Returns:
        True if query contains industrial terms (likely valid), False otherwise
    """
    matches = _industrial_keyword_pattern.findall(query)
    return bool(matches)


def matches_invalid_pattern(query: str) -> tuple[bool, str | None]:
    """
    Fast-path rule-based check for obviously invalid queries.
    
    This can be used as a pre-filter before semantic classification
    to save LLM calls for obvious cases.
    
    Args:
        query: User query string
        
    Returns:
        Tuple of (is_invalid, category)
        - (True, category) if query matches invalid patterns
        - (False, None) if query passes rule-based check
    """
    query_lower = query.lower().strip()
    
    # Check each invalid category
    for category, patterns in _invalid_pattern_cache.items():
        for pattern in patterns:
            if pattern.search(query_lower):
                return True, category
    
    return False, None


def get_industrial_keyword_matches(query: str) -> List[str]:
    """
    Get list of industrial keywords found in query.
    
    Useful for logging and debugging.
    
    Args:
        query: User query string
        
    Returns:
        List of matched industrial keywords
    """
    return _industrial_keyword_pattern.findall(query)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'INDUSTRIAL_KEYWORDS',
    'INVALID_QUERY_PATTERNS',
    'contains_industrial_keywords',
    'matches_invalid_pattern',
    'get_industrial_keyword_matches',
]
