# search/caching/__init__.py
"""
Search Deep Agent Caching Module.

Provides centralized cache management for the search workflow.
"""

from .search_caches import (
    get_session_enrichment_cache,
    get_vendor_response_cache,
    get_schema_cache,
    compute_vendor_cache_key,
    compute_session_cache_key,
    normalize_product_type,
    clear_session_caches,
    clear_all_caches,
    get_cache_stats,
    get_or_set_session_enrichment,
    get_or_set_vendor_response,
)

__all__ = [
    "get_session_enrichment_cache",
    "get_vendor_response_cache",
    "get_schema_cache",
    "compute_vendor_cache_key",
    "compute_session_cache_key",
    "normalize_product_type",
    "clear_session_caches",
    "clear_all_caches",
    "get_cache_stats",
    "get_or_set_session_enrichment",
    "get_or_set_vendor_response",
]
