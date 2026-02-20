# Standards Shared Utilities Module

from .cache import (
    get_cached_standards,
    cache_standards,
    clear_standards_cache,
    get_cache_key,
)

from .constants import (
    MIN_STANDARDS_SPECS_COUNT,
    MAX_SPECS_PER_ITEM,
    MAX_SPECS_PER_DOMAIN,
    MAX_STANDARDS_ITERATIONS,
    MAX_STANDARDS_PARALLEL_WORKERS,
    CHUNK_SIZE,
    DEEP_AGENT_LLM_MODEL,
    CACHE_MAX_SIZE,
    CACHE_TTL_SECONDS,
    DEFAULT_TOP_K,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RECURSION_LIMIT,
    MAX_CONCURRENT_API_REQUESTS,
    STANDARDS_DOCX_DIR,
    FALLBACK_DOMAINS_ORDER,
)

from .keywords import (
    StandardsDomain,
    DOMAIN_KEYWORDS,
    DOMAIN_TO_DOCUMENTS,
    FIELD_GROUPS,
)

from .detector import (
    classify_domain,
    get_routed_documents,
    is_standards_related_question,
    detect_standards_indicators,
)

from .enrichment import (
    ParallelSchemaEnricher,
    enrich_schema_parallel,
    enrich_schema_async,
    enrich_items_parallel,
    run_3_source_enrichment,
    is_valid_spec_value,
    normalize_category,
)

__all__ = [
    # Cache
    'get_cached_standards',
    'cache_standards',
    'clear_standards_cache',
    'get_cache_key',
    # Constants
    'MIN_STANDARDS_SPECS_COUNT',
    'MAX_SPECS_PER_ITEM',
    'MAX_SPECS_PER_DOMAIN',
    'MAX_STANDARDS_ITERATIONS',
    'MAX_STANDARDS_PARALLEL_WORKERS',
    'CHUNK_SIZE',
    'DEEP_AGENT_LLM_MODEL',
    'CACHE_MAX_SIZE',
    'CACHE_TTL_SECONDS',
    'DEFAULT_TOP_K',
    'DEFAULT_MAX_RETRIES',
    'DEFAULT_RECURSION_LIMIT',
    'MAX_CONCURRENT_API_REQUESTS',
    'STANDARDS_DOCX_DIR',
    'FALLBACK_DOMAINS_ORDER',
    # Keywords
    'StandardsDomain',
    'DOMAIN_KEYWORDS',
    'DOMAIN_TO_DOCUMENTS',
    'FIELD_GROUPS',
    # Detector
    'classify_domain',
    'get_routed_documents',
    'is_standards_related_question',
    'detect_standards_indicators',
    # Enrichment
    'ParallelSchemaEnricher',
    'enrich_schema_parallel',
    'enrich_schema_async',
    'enrich_items_parallel',
    'run_3_source_enrichment',
    'is_valid_spec_value',
    'normalize_category',
]
