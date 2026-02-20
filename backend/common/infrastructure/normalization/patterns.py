# agentic/infrastructure/normalization/patterns.py
# =============================================================================
# CENTRALIZED PATTERN CONSTANTS
# =============================================================================
#
# Single source of truth for all specification validation patterns.
# Consolidated from:
# - deep_agent/processing/value_normalizer.py
# - standards/generation/normalizer.py
# - standards/generation/verifier.py
# - standards/shared/enrichment.py
#
# =============================================================================

import re
from typing import List, Pattern

# =============================================================================
# TECHNICAL VALUE PATTERNS
# =============================================================================

TECHNICAL_PATTERNS: List[str] = [
    # Accuracy/tolerance
    r'[±\+\-]\s*\d+\.?\d*\s*%',              # ±0.1%, +0.5%
    r'[±\+\-]\s*\d+\.?\d*\s*(?:°[CF]|K)',    # ±1°C, ±0.5K
    r'[±\+\-]\s*\d+\.?\d*\s*(?:bar|psi|Pa|kPa|MPa)',  # ±0.5bar

    # Ranges
    r'-?\d+\.?\d*\s*(?:to|[-–])\s*[+\-]?\d+\.?\d*\s*(?:°[CF]|K)',  # -40 to 85°C
    r'\d+\.?\d*\s*(?:to|[-–])\s*\d+\.?\d*\s*(?:bar|psi|Pa|kPa|MPa)',  # 0-400 bar
    r'\d+\.?\d*\s*(?:to|[-–])\s*\d+\.?\d*\s*(?:mA|V|Hz)',  # 4-20mA, 0-10V
    r'\d+\.?\d*\s*(?:to|[-–])\s*\d+\.?\d*\s*(?:mm|m|cm|inch|in)',  # Length ranges

    # Electrical
    r'\d+\s*(?:VDC|VAC|V\s*DC|V\s*AC)',      # 24 VDC
    r'\d+[-–]\d+\s*mA',                       # 4-20mA
    r'\d+\s*mA(?:\s+HART)?',                  # 20mA, 4-20mA HART
    r'\d+[-–]\d+\s*V(?:DC|AC)?',              # 10-30VDC

    # Protection/safety ratings
    r'IP\s*\d{2}[A-Z]?',                      # IP67, IP65K
    r'NEMA\s*\d+[A-Z]*',                      # NEMA 4X
    r'SIL\s*[1-4]',                           # SIL 2
    r'ATEX(?:\s+Zone\s*[0-2])?',              # ATEX, ATEX Zone 1
    r'IECEx',                                  # IECEx
    r'Class\s*[I]+\s*(?:,\s*)?Div(?:ision)?\s*[1-2]',  # Class I Div 1
    r'Zone\s*[0-2](?:/[0-2]+)?',              # Zone 1, Zone 0/1

    # Materials
    r'SS\s*\d+L?',                            # SS316L
    r'316L?(?:\s*SS)?',                       # 316L, 316 SS
    r'304(?:\s*SS)?',                         # 304, 304 SS
    r'Hastelloy\s*[A-Z]?(?:-?\d+)?',          # Hastelloy C-276
    r'Inconel\s*\d*',                         # Inconel 625
    r'Monel\s*\d*',                           # Monel 400
    r'Titanium(?:\s+Grade\s*\d+)?',           # Titanium Grade 2
    r'PTFE|PFA|FEP|PEEK',                     # Fluoropolymers
    r'Viton|Kalrez|EPDM|NBR|FKM',             # Elastomers

    # Connections
    r'\d+/\d+\s*(?:NPT|BSP|BSPT|BSPP)',       # 1/2 NPT
    r'DN\s*\d+',                              # DN50
    r'G\s*\d+/\d+',                           # G1/2
    r'Tri-?[Cc]lamp',                         # Tri-Clamp
    r'(?:RF|RTJ|FF)\s*[Ff]lange',             # RF Flange
    r'\d+"\s*(?:ANSI|PN)\s*\d+',              # 2" ANSI 150

    # Communication protocols
    r'HART(?:\s*\d+)?',                       # HART, HART 7
    r'Modbus\s*(?:RTU|TCP)?',                 # Modbus RTU
    r'Profibus(?:\s*(?:PA|DP))?',             # Profibus PA
    r'Foundation\s*Fieldbus',                 # Foundation Fieldbus
    r'EtherNet/IP',                           # EtherNet/IP
    r'PROFINET',                              # PROFINET

    # Time/response
    r'[<>≤≥]?\s*\d+\.?\d*\s*(?:ms|s|sec|min)',  # <250ms, 1.5s
    r'T\d+\s*[<>≤≥]?\s*\d+\s*(?:s|ms)',      # T90 < 5s

    # Ratios
    r'\d+\s*:\s*\d+',                         # 100:1 turndown

    # Percentages with context
    r'\d+\.?\d*\s*%\s*(?:FS|span|URL|range)?',  # 0.1% FS

    # Specific units
    r'\d+\.?\d*\s*(?:Ω|ohm|kΩ|MΩ)',          # Resistance
    r'\d+\.?\d*\s*(?:pF|nF|µF|mF)',          # Capacitance
    r'\d+\.?\d*\s*(?:g|kg|lb)',              # Weight
]


# =============================================================================
# INVALID/DESCRIPTION PATTERNS
# =============================================================================

INVALID_PATTERNS: List[str] = [
    # Uncertainty phrases
    r'\b(depends?\s+(?:on|upon))',
    r'\b(varies?\s+(?:by|with|depending|based))',
    r'\b(according\s+to)',
    r'\b(subject\s+to)',
    r'\b(based\s+on)',

    # Reference/instruction phrases
    r'\b(refer\s+to)',
    r'\b(see\s+(?:the|page|section|datasheet|manual))',
    r'\b(check\s+(?:with|the))',
    r'\b(contact\s+(?:the\s+)?(?:manufacturer|supplier|vendor))',
    r'\b(consult\s+(?:the|your))',
    r'\b(please\s+(?:refer|see|check|contact))',
    r'\b(for\s+more\s+(?:information|details))',
    r'\b(visit\s+(?:the|our))',

    # Uncertainty modifiers
    r'\b(typically|usually|generally|normally|often|sometimes)\b',
    r'\b(approximately|about|around|roughly)\s+(?!\d)',  # Allow "around 5%" but not "around the"
    r'\b(may|might|could|can)\s+(?:be|vary|differ)',
    r'\b(should\s+be)',
    r'\b(must\s+be)',

    # Placeholder/unknown
    r'\b(TBD|TBC|T\.B\.D\.|T\.B\.C\.)\b',
    r'\b(N/?A|n/?a)\b',
    r'\b(not\s+(?:applicable|available|specified|defined|provided))',
    r'\b(to\s+be\s+(?:determined|confirmed|specified))',
    r'\b(upon\s+request)',
    r'\b(available\s+(?:on|upon)\s+request)',
    r'\b(as\s+per\s+(?:customer|project|order))',
    r'\b(customer\s+specific)',
    r'\b(project\s+specific)',
    r'\b(application\s+specific)',
    r'\b(model\s+dependent)',

    # Descriptive/instructional
    r'\b(available\s+options?\s+(?:are|include))',
    r'\b(can\s+be\s+(?:configured|selected|specified))',
    r'\b(options?\s+(?:are|include))',
    r'\b(standard\s+options?\s+(?:are|include))',
    r'\b(is\s+(?:determined|selected|specified)\s+by)',

    # LLM template artifacts
    r'extracted\s+value',
    r'consolidated\s+value',
    r'value\s+if\s+applicable',
    r'from\s+the\s+document',
    r'based\s+on\s+(?:the\s+)?(?:document|standard)',
    r'no\s+(?:specific\s+)?(?:information|data|value)',
    r"(?:i\s+)?(?:don't|do\s+not)\s+have",
    r'not\s+(?:explicitly\s+)?(?:mentioned|specified|found|stated)',
    r'information\s+(?:is\s+)?not\s+(?:available|provided)',

    # Sentence indicators
    r'\.\s*$',  # Ends with period (likely a sentence)
]


# =============================================================================
# LEADING/TRAILING PHRASE PATTERNS
# =============================================================================

LEADING_PHRASES: List[str] = [
    # Common sentence starters
    r'^the\s+\w+\s+(is|are|has|was|were)\s+',
    r'^it\s+(is|has|can\s+be|should\s+be|was)\s+',
    r'^this\s+(is|has|should|was)\s+',
    r'^that\s+(is|has|should|was)\s+',
    r'^there\s+(is|are)\s+',

    # Field name prefixes
    r'^output\s+(signal\s+)?(is|:)\s*',
    r'^accuracy\s+(is|:)\s*',
    r'^range\s+(is|:)\s*',
    r'^temperature\s+(range\s+)?(is|:)\s*',
    r'^pressure\s+(range\s+)?(is|:)\s*',
    r'^measurement\s+(range\s+)?(is|:)\s*',
    r'^operating\s+(range\s+)?(is|:)\s*',
    r'^value\s+(is|:)\s*',
    r'^specification\s+(is|:)\s*',
    r'^signal\s+(is|:)\s*',
    r'^power\s+(supply\s+)?(is|:)\s*',
    r'^voltage\s+(is|:)\s*',
    r'^current\s+(is|:)\s*',
    r'^rating\s+(is|:)\s*',
    r'^protection\s+(is|:)\s*',
    r'^material\s+(is|:)\s*',

    # Modifiers
    r'^rated\s+(at|for)\s+',
    r'^typically\s+',
    r'^usually\s+',
    r'^generally\s+',
    r'^normally\s+',
    r'^approximately\s+',
    r'^about\s+',
    r'^around\s+',
    r'^up\s+to\s+',
    r'^at\s+least\s+',
    r'^minimum\s+(of\s+)?',
    r'^maximum\s+(of\s+)?',
    r'^standard\s+(is\s+)?',
    r'^default\s+(is\s+)?',
    r'^nominal\s+',

    # LLM artifacts
    r'^\*+\s*',
    r'^-\s+',
    r'^•\s+',
]

TRAILING_PHRASES: List[str] = [
    # Standards references (will be extracted first)
    r'\s*(?:per|as\s+per|according\s+to|ref\.?|reference)\s+(?:IEC|ISO|API|ANSI|ISA|EN|NFPA|ASME|IEEE|ASTM|BS|DIN)[\s\d\-\.]+.*$',
    r'\s*\((?:per|ref\.?|see)\s+.*\)$',

    # Contextual suffixes
    r'\s*,?\s*(?:calibrated|measured|tested)\s+(?:at|in).*$',
    r'\s*,?\s*at\s+\d+\s*°[CF].*$',
    r'\s*,?\s*at\s+ambient.*$',
    r'\s*,?\s*under\s+(?:standard|normal|reference).*$',
    r'\s*,?\s*for\s+(?:outdoor|indoor|hazardous).*$',
    r'\s*,?\s*when\s+.*$',
    r'\s*,?\s*if\s+.*$',

    # Parenthetical notes
    r'\s*\((?:see|refer|check|note|optional|standard|typical).*\)$',
    r'\s*\[source:.*\]$',
    r'\s*\[ref:.*\]$',

    # LLM artifacts
    r'\s*\*+$',
    r'\s*\.+$',
    r'\s*;+$',
]


# =============================================================================
# STANDARDS REFERENCE PATTERN
# =============================================================================

STANDARDS_PATTERN: str = r'\b((?:IEC|ISO|API|ANSI|ISA|EN|NFPA|ASME|IEEE|ASTM|BS|DIN|NIST|OIML|MIL)\s*[\d\-\.]+(?:[A-Z])?)\b'


# =============================================================================
# INVALID EXACT VALUES
# =============================================================================

INVALID_EXACT_VALUES: set = {
    "null", "none", "", "extracted value or null",
    "not specified", "not_specified", "n/a", "unknown",
    "fail", "error", "no value found", "not found",
    "not applicable", "value not found", "not defined",
    "none provided", "-", "see datasheet", "not mentioned"
}


# =============================================================================
# HALLUCINATED KEY PATTERNS
# =============================================================================

HALLUCINATED_KEY_PATTERNS: List[str] = [
    "loss_of_ambition", "loss_of_appetite", "loss_of_calm",
    "loss_of_creativity", "loss_of_desire", "loss_of_emotion",
    "loss_of_feeling", "loss_of_hope", "loss_of_imagination",
    "loss_of_passion", "loss_of_peace", "loss_of_purpose",
    "protection_against_loss_of_", "cable_tray_cable_protection_against_loss"
]


# =============================================================================
# INVALID KEY TERMS
# =============================================================================

INVALID_KEY_TERMS: List[str] = [
    "new spec", "new_spec", "specification", "generated",
    "unknown", "item_", "spec_", "not specified"
]


# =============================================================================
# N/A PATTERNS
# =============================================================================

NA_PATTERNS: List[str] = [
    r"^N/A$",
    r"^n/a$",
    r"^NA$",
    r"^not\s+applicable$",
    r"^none$",
    r"^null$",
    r"^-$",
    r"^\s*$",
]


# =============================================================================
# COMPILED PATTERN CACHES
# =============================================================================

_compiled_technical: List[Pattern] = None
_compiled_invalid: List[Pattern] = None
_compiled_leading: List[Pattern] = None
_compiled_trailing: List[Pattern] = None
_compiled_standards: Pattern = None
_compiled_na: List[Pattern] = None


def get_compiled_technical_patterns() -> List[Pattern]:
    """Get compiled technical patterns (cached)."""
    global _compiled_technical
    if _compiled_technical is None:
        _compiled_technical = [re.compile(p, re.IGNORECASE) for p in TECHNICAL_PATTERNS]
    return _compiled_technical


def get_compiled_invalid_patterns() -> List[Pattern]:
    """Get compiled invalid patterns (cached)."""
    global _compiled_invalid
    if _compiled_invalid is None:
        _compiled_invalid = [re.compile(p, re.IGNORECASE) for p in INVALID_PATTERNS]
    return _compiled_invalid


def get_compiled_leading_patterns() -> List[Pattern]:
    """Get compiled leading phrase patterns (cached)."""
    global _compiled_leading
    if _compiled_leading is None:
        _compiled_leading = [re.compile(p, re.IGNORECASE) for p in LEADING_PHRASES]
    return _compiled_leading


def get_compiled_trailing_patterns() -> List[Pattern]:
    """Get compiled trailing phrase patterns (cached)."""
    global _compiled_trailing
    if _compiled_trailing is None:
        _compiled_trailing = [re.compile(p, re.IGNORECASE) for p in TRAILING_PHRASES]
    return _compiled_trailing


def get_compiled_standards_pattern() -> Pattern:
    """Get compiled standards reference pattern (cached)."""
    global _compiled_standards
    if _compiled_standards is None:
        _compiled_standards = re.compile(STANDARDS_PATTERN, re.IGNORECASE)
    return _compiled_standards


def get_compiled_na_patterns() -> List[Pattern]:
    """Get compiled N/A patterns (cached)."""
    global _compiled_na
    if _compiled_na is None:
        _compiled_na = [re.compile(p, re.IGNORECASE) for p in NA_PATTERNS]
    return _compiled_na


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Pattern lists
    "TECHNICAL_PATTERNS",
    "INVALID_PATTERNS",
    "LEADING_PHRASES",
    "TRAILING_PHRASES",
    "STANDARDS_PATTERN",
    "INVALID_EXACT_VALUES",
    "HALLUCINATED_KEY_PATTERNS",
    "INVALID_KEY_TERMS",
    "NA_PATTERNS",
    # Compiled pattern getters
    "get_compiled_technical_patterns",
    "get_compiled_invalid_patterns",
    "get_compiled_leading_patterns",
    "get_compiled_trailing_patterns",
    "get_compiled_standards_pattern",
    "get_compiled_na_patterns",
]
