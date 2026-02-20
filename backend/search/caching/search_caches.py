# search/caching/search_caches.py
# =============================================================================
# SEARCH DEEP AGENT CACHING UTILITIES
# =============================================================================
#
# Provides centralized cache management for the Search Deep Agent workflow.
#
# Cache Types:
# 1. Session Enrichment Cache - Prevents duplicate Standards RAG calls
# 2. Vendor Response Cache - Caches vendor analysis results (expensive)
# 3. Schema Cache - Caches schema lookups
#
# =============================================================================

import hashlib
import json
import logging
from typing import Dict, Any, Optional

from common.infrastructure.caching.bounded_cache import BoundedCache

logger = logging.getLogger(__name__)


# =============================================================================
# MODULE-LEVEL CACHE INSTANCES (Lazy-initialized singletons)
# =============================================================================

_session_enrichment_cache: Optional[BoundedCache] = None
_vendor_response_cache: Optional[BoundedCache] = None
_schema_cache: Optional[BoundedCache] = None


# =============================================================================
# CACHE FACTORY FUNCTIONS
# =============================================================================

def get_session_enrichment_cache() -> BoundedCache:
    """
    Get or create the session enrichment cache.

    This cache prevents duplicate Standards RAG calls within the same session.
    Uses infinite TTL (ttl_seconds=0) since session-scoped.

    Cache key format: "{session_id}:{normalized_product_type}"

    Returns:
        BoundedCache instance for session enrichment
    """
    global _session_enrichment_cache
    if _session_enrichment_cache is None:
        _session_enrichment_cache = BoundedCache(
            name="search_session_enrichment",
            max_size=500,
            ttl_seconds=0,  # No TTL - session-scoped
            on_evict=lambda k, v: logger.debug(
                "[search_session_enrichment] Evicted key: %s", k
            ),
        )
        logger.info("[search_caches] Created session_enrichment_cache (max=500, ttl=infinite)")
    return _session_enrichment_cache


def get_vendor_response_cache() -> BoundedCache:
    """
    Get or create the vendor analysis response cache.

    This cache stores complete vendor analysis results to avoid expensive
    re-computation for identical requests. Saves 200-600s on repeat calls.

    Cache key: MD5 hash of (product_type + sorted requirements JSON)

    Returns:
        BoundedCache instance for vendor responses
    """
    global _vendor_response_cache
    if _vendor_response_cache is None:
        _vendor_response_cache = BoundedCache(
            name="search_vendor_response",
            max_size=100,
            ttl_seconds=3600,  # 1 hour TTL
            on_evict=lambda k, v: logger.debug(
                "[search_vendor_response] Evicted key: %s", k
            ),
        )
        logger.info("[search_caches] Created vendor_response_cache (max=100, ttl=3600s)")
    return _vendor_response_cache


def get_schema_cache() -> BoundedCache:
    """
    Get or create the schema lookup cache.

    This cache stores schema lookups to avoid repeated database queries.

    Cache key: normalized product type

    Returns:
        BoundedCache instance for schema lookups
    """
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = BoundedCache(
            name="search_schema",
            max_size=200,
            ttl_seconds=14400,  # 4 hour TTL
            on_evict=lambda k, v: logger.debug(
                "[search_schema] Evicted key: %s", k
            ),
        )
        logger.info("[search_caches] Created schema_cache (max=200, ttl=14400s)")
    return _schema_cache


# =============================================================================
# CACHE KEY UTILITIES
# =============================================================================

def compute_vendor_cache_key(
    product_type: str,
    requirements: Dict[str, Any],
) -> str:
    """
    Compute a cache key for vendor analysis results.

    Uses MD5 hash of product_type + sorted requirements JSON to create
    a deterministic key for identical requests.

    Args:
        product_type: The product type being searched
        requirements: The structured requirements dict

    Returns:
        MD5 hash string suitable for cache key
    """
    # Normalize product type
    normalized_type = product_type.lower().strip()

    # Sort requirements for deterministic JSON
    sorted_reqs = json.dumps(requirements, sort_keys=True, default=str)

    # Combine and hash
    combined = f"{normalized_type}:{sorted_reqs}"
    return hashlib.md5(combined.encode()).hexdigest()


def compute_session_cache_key(
    session_id: str,
    product_type: str,
) -> str:
    """
    Compute a cache key for session-scoped enrichment.

    Args:
        session_id: The session identifier
        product_type: The product type

    Returns:
        Cache key string
    """
    normalized_type = product_type.lower().strip().replace(" ", "_")
    return f"{session_id}:{normalized_type}"


def normalize_product_type(product_type: str) -> str:
    """
    Normalize product type for cache key consistency.

    Args:
        product_type: Raw product type string

    Returns:
        Normalized product type
    """
    return product_type.lower().strip().replace(" ", "_")


# =============================================================================
# CACHE MANAGEMENT UTILITIES
# =============================================================================

def clear_session_caches(session_id: str) -> int:
    """
    Clear all caches for a specific session.

    This should be called when a session ends to free memory.

    Args:
        session_id: The session to clear

    Returns:
        Number of entries cleared
    """
    cleared = 0

    # Clear session enrichment cache entries for this session
    cache = get_session_enrichment_cache()
    keys_to_remove = []

    # Find all keys for this session
    # Note: Accessing internal _cache is not ideal but necessary for filtering
    with cache._lock:
        for key in cache._cache.keys():
            if key.startswith(f"{session_id}:"):
                keys_to_remove.append(key)

    # Remove found keys
    for key in keys_to_remove:
        # Use internal eviction to trigger callbacks
        with cache._lock:
            if key in cache._cache:
                cache._cache.pop(key)
                cleared += 1

    if cleared > 0:
        logger.info("[search_caches] Cleared %d entries for session %s", cleared, session_id)

    return cleared


def clear_all_caches() -> Dict[str, int]:
    """
    Clear all search-related caches.

    This should be called during application shutdown.

    Returns:
        Dict mapping cache name to number of entries cleared
    """
    results = {}

    if _session_enrichment_cache is not None:
        count = len(_session_enrichment_cache._cache)
        _session_enrichment_cache.clear()
        results["session_enrichment"] = count

    if _vendor_response_cache is not None:
        count = len(_vendor_response_cache._cache)
        _vendor_response_cache.clear()
        results["vendor_response"] = count

    if _schema_cache is not None:
        count = len(_schema_cache._cache)
        _schema_cache.clear()
        results["schema"] = count

    logger.info("[search_caches] Cleared all caches: %s", results)
    return results


def get_cache_stats() -> Dict[str, Dict[str, int]]:
    """
    Get statistics for all search caches.

    Returns:
        Dict mapping cache name to stats dict (hits, misses, evictions, size)
    """
    stats = {}

    if _session_enrichment_cache is not None:
        cache = _session_enrichment_cache
        stats["session_enrichment"] = {
            **cache._stats,
            "size": len(cache._cache),
            "max_size": cache.max_size,
        }

    if _vendor_response_cache is not None:
        cache = _vendor_response_cache
        stats["vendor_response"] = {
            **cache._stats,
            "size": len(cache._cache),
            "max_size": cache.max_size,
        }

    if _schema_cache is not None:
        cache = _schema_cache
        stats["schema"] = {
            **cache._stats,
            "size": len(cache._cache),
            "max_size": cache.max_size,
        }

    return stats


# =============================================================================
# CACHE-AWARE HELPERS
# =============================================================================

def get_or_set_session_enrichment(
    session_id: str,
    product_type: str,
    compute_fn: callable,
) -> Any:
    """
    Get cached enrichment result or compute and cache it.

    Args:
        session_id: Session identifier
        product_type: Product type
        compute_fn: Function to compute value if not cached (no args)

    Returns:
        Cached or freshly computed value
    """
    cache = get_session_enrichment_cache()
    key = compute_session_cache_key(session_id, product_type)

    # Try cache first
    cached = cache.get(key)
    if cached is not None:
        logger.debug("[search_caches] Session enrichment cache HIT: %s", key)
        return cached

    # Compute and cache
    logger.debug("[search_caches] Session enrichment cache MISS: %s", key)
    result = compute_fn()
    cache.set(key, result)
    return result


def get_or_set_vendor_response(
    product_type: str,
    requirements: Dict[str, Any],
    compute_fn: callable,
) -> Any:
    """
    Get cached vendor analysis or compute and cache it.

    Args:
        product_type: Product type
        requirements: Structured requirements
        compute_fn: Function to compute value if not cached (no args)

    Returns:
        Cached or freshly computed value
    """
    cache = get_vendor_response_cache()
    key = compute_vendor_cache_key(product_type, requirements)

    # Try cache first
    cached = cache.get(key)
    if cached is not None:
        logger.info("[search_caches] Vendor response cache HIT: %s", key[:16])
        return cached

    # Compute and cache
    logger.info("[search_caches] Vendor response cache MISS: %s", key[:16])
    result = compute_fn()
    cache.set(key, result)
    return result
