import os
import json

# Minimum specs from standards documents per item (other 30 come from LLM → total 60)
MIN_STANDARDS_SPECS_COUNT = int(os.getenv("MIN_STANDARDS_SPECS_COUNT", 30))

# Overall minimum total specs per item (standards + LLM combined)
MIN_TOTAL_SPECS_COUNT = int(os.getenv("MIN_TOTAL_SPECS_COUNT", 60))

# Maximum specs per item (effectively unlimited - include ALL specs)
MAX_SPECS_PER_ITEM = int(os.getenv("MAX_SPECS_PER_ITEM", 9999))

# Maximum specs per domain for comprehensive coverage
MAX_SPECS_PER_DOMAIN = int(os.getenv("MAX_SPECS_PER_DOMAIN", 1000))


# Maximum iterations to prevent infinite loops
MAX_STANDARDS_ITERATIONS = int(os.getenv("MAX_STANDARDS_ITERATIONS", 5))

# Number of domains to analyze in parallel
MAX_STANDARDS_PARALLEL_WORKERS = int(os.getenv("MAX_STANDARDS_PARALLEL_WORKERS", 3))

# Chunk size for batch processing (prevents LLM timeouts)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 12))

# LLM Model for Deep Agent nodes
DEEP_AGENT_LLM_MODEL = os.getenv("DEEP_AGENT_LLM_MODEL", "gemini-2.5-flash-lite")

# Cache configuration for standards results
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", 100))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 600))  # 10 minutes


# Default number of documents to retrieve
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))

# Maximum retries for RAG operations
DEFAULT_MAX_RETRIES = int(os.getenv("DEFAULT_MAX_RETRIES", 2))

# Recursion limit for workflow
DEFAULT_RECURSION_LIMIT = int(os.getenv("DEFAULT_RECURSION_LIMIT", 15))

# Maximum concurrent API requests
MAX_CONCURRENT_API_REQUESTS = int(os.getenv("MAX_CONCURRENT_API_REQUESTS", 10))

# Compute standards document directory path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(_current_dir)))
_standards_candidate_1 = os.path.join(_base_dir, "chroma_data", "Standards")
_standards_candidate_2 = os.path.join(_base_dir, "backend", "chroma_data", "Standards")

# Use whichever path exists, fallback to candidate 1 (Production)
# Check for environment variable override first
_env_standards_path = os.getenv("STANDARDS_DOCX_DIR_PATH")

if _env_standards_path:
    STANDARDS_DOCX_DIR = _env_standards_path
elif os.path.isdir(_standards_candidate_2) and os.path.exists(
    os.path.join(_standards_candidate_2, "instrumentation_safety_standards.docx")
):
    STANDARDS_DOCX_DIR = _standards_candidate_2
elif os.path.isdir(_standards_candidate_1) and os.path.exists(
    os.path.join(_standards_candidate_1, "instrumentation_safety_standards.docx")
):
    STANDARDS_DOCX_DIR = _standards_candidate_1
else:
    STANDARDS_DOCX_DIR = _standards_candidate_1  # Default

# Order of domains to try if minimum specs not reached
FALLBACK_DOMAINS_ORDER = [
    "safety", "calibration", "communication", "accessories",
    "pressure", "temperature", "flow", "level", "control",
    "valves", "condition_monitoring", "analytical"
]   

_env_fallback_domains = os.getenv("FALLBACK_DOMAINS_ORDER")
if _env_fallback_domains:
    try:
        FALLBACK_DOMAINS_ORDER = json.loads(_env_fallback_domains)
    except Exception:
        pass # Keep default if parsing fails

__all__ = [
    # Spec counts
    "MIN_STANDARDS_SPECS_COUNT",
    "MIN_TOTAL_SPECS_COUNT",
    "MAX_SPECS_PER_ITEM",
    "MAX_SPECS_PER_DOMAIN",
    # Iteration/parallel
    "MAX_STANDARDS_ITERATIONS",
    "MAX_STANDARDS_PARALLEL_WORKERS",
    "CHUNK_SIZE",
    # LLM
    "DEEP_AGENT_LLM_MODEL",
    # Cache
    "CACHE_MAX_SIZE",
    "CACHE_TTL_SECONDS",
    # RAG
    "DEFAULT_TOP_K",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RECURSION_LIMIT",
    # API
    "MAX_CONCURRENT_API_REQUESTS",
    # Paths
    "STANDARDS_DOCX_DIR",
    # Fallback
    "FALLBACK_DOMAINS_ORDER",
]
