"""
Indexing Agent — Centralized Configuration
=======================================
All tunable parameters for the Indexing Agent workflow.
"""

# ── LLM Models ──────────────────────────────────────────────────────────────
DEFAULT_MODEL = "gemini-2.0-flash-exp"
FALLBACK_MODEL = "gemini-1.5-flash"
REASONING_MODEL = "gemini-2.0-flash-thinking-exp"

# ── Parallelisation ─────────────────────────────────────────────────────────
MAX_SEARCH_WORKERS = 3
MAX_DOWNLOAD_WORKERS = 3
MAX_EXTRACTION_WORKERS = 3

# ── Timeouts (seconds) ─────────────────────────────────────────────────────
PDF_DOWNLOAD_TIMEOUT = 60
DNS_VALIDATION_TIMEOUT = 3
LLM_REQUEST_TIMEOUT = 60

# ── Search ──────────────────────────────────────────────────────────────────
DEFAULT_VENDOR_COUNT = 5
MAX_PDFS_PER_VENDOR = 3
MAX_SEARCH_QUERIES = 3

# ── Quality thresholds ─────────────────────────────────────────────────────
QUALITY_THRESHOLD_PRODUCTION = 0.85
QUALITY_THRESHOLD_STAGING = 0.70
QUALITY_THRESHOLD_REFINEMENT = 0.70
MAX_REFINEMENT_LOOPS = 2

# ── PDF processing ─────────────────────────────────────────────────────────
MAX_PDF_PAGES = 50
MAX_PDF_SIZE_MB = 50
MAX_EXTRACTION_TOKENS = 25_000

# ── Vendor domain map ──────────────────────────────────────────────────────
VENDOR_DOMAIN_MAP = {
    "rosemount": "emerson.com",
    "emerson": "emerson.com",
    "endress+hauser": "endress.com",
    "endress": "endress.com",
    "yokogawa": "yokogawa.com",
    "siemens": "siemens.com",
    "abb": "abb.com",
    "honeywell": "honeywell.com",
    "schneider": "se.com",
}

# ── Common fallback vendors ────────────────────────────────────────────────
FALLBACK_VENDORS = [
    "Rosemount (Emerson)",
    "Endress+Hauser",
    "Yokogawa",
    "Siemens",
    "ABB",
    "Honeywell",
    "Schneider Electric",
]

# ── Complexity → strategy mapping ──────────────────────────────────────────
COMPLEXITY_STRATEGIES = {
    "simple": "fast_path",
    "moderate": "standard_discovery",
    "complex": "comprehensive_search",
    "very_complex": "multi_source_synthesis",
}

# ── Duration estimates (seconds) ──────────────────────────────────────────
DURATION_ESTIMATES = {
    "simple": {"min": 20, "max": 40, "typical": 30},
    "moderate": {"min": 30, "max": 60, "typical": 45},
    "complex": {"min": 45, "max": 90, "typical": 60},
    "very_complex": {"min": 60, "max": 120, "typical": 90},
}
