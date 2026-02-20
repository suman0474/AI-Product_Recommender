# agentic/standards/cache.py
# =============================================================================
# STANDARDS MODULE - UNIFIED CACHING LAYER
# =============================================================================
#
# This file consolidates all caching functions from:
# - standards_rag_enrichment.py (_standards_cache, _get_cached_standards, etc.)
# - standards_enrichment_tool.py (_standards_results_cache, cache functions)
#
# Uses the existing BoundedCache infrastructure for TTL/LRU management.
#
# =============================================================================

import logging
import threading
from typing import Dict, Any, Optional

from .constants import CACHE_MAX_SIZE, CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)

# =============================================================================
# CACHE INITIALIZATION
# =============================================================================

# Import BoundedCache from infrastructure
try:
    from common.infrastructure.caching.bounded_cache import get_or_create_cache, BoundedCache
    _USE_BOUNDED_CACHE = True
except ImportError:
    logger.warning("[StandardsCache] BoundedCache not available, using simple dict cache")
    _USE_BOUNDED_CACHE = False


# Create the unified standards cache
if _USE_BOUNDED_CACHE:
    _standards_cache: BoundedCache = get_or_create_cache(
        name="standards_unified",
        max_size=CACHE_MAX_SIZE,
        ttl_seconds=CACHE_TTL_SECONDS
    )
else:
    # Fallback: simple dict cache (no TTL/LRU)
    _standards_cache: Dict[str, Any] = {}
    _standards_cache_lock = threading.Lock()


# =============================================================================
# CACHE KEY GENERATION
# =============================================================================

def get_cache_key(product_type: str, source_filter: Optional[str] = None) -> str:
    """
    Generate cache key from product type and optional source filter.

    Args:
        product_type: Product type string
        source_filter: Optional comma-separated list of source documents

    Returns:
        Normalized cache key string
    """
    base_key = product_type.lower().strip()
    if source_filter:
        return f"{base_key}|{source_filter}"
    return base_key


# =============================================================================
# CACHE OPERATIONS
# =============================================================================

def get_cached_standards(
    product_type: str,
    source_filter: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get cached standards result for a product type.

    Args:
        product_type: Product type to look up
        source_filter: Optional source filter used in the original query

    Returns:
        Cached result dict if found and not expired, None otherwise
    """
    key = get_cache_key(product_type, source_filter)

    if _USE_BOUNDED_CACHE:
        # BoundedCache handles TTL expiration internally
        result = _standards_cache.get(key)
    else:
        # Simple dict fallback
        with _standards_cache_lock:
            result = _standards_cache.get(key)

    if result is not None:
        logger.debug(f"[StandardsCache] Cache HIT for: {key}")
    else:
        logger.debug(f"[StandardsCache] Cache MISS for: {key}")

    return result


def cache_standards(
    product_type: str,
    result: Dict[str, Any],
    source_filter: Optional[str] = None
) -> None:
    """
    Cache standards result for a product type.

    Args:
        product_type: Product type being cached
        result: Result dict to cache
        source_filter: Optional source filter used in the query
    """
    key = get_cache_key(product_type, source_filter)

    if _USE_BOUNDED_CACHE:
        # BoundedCache handles max_size and LRU eviction internally
        _standards_cache.set(key, result)
    else:
        # Simple dict fallback
        with _standards_cache_lock:
            _standards_cache[key] = result

    logger.debug(f"[StandardsCache] Cached result for: {key}")


def clear_standards_cache() -> int:
    """
    Clear the standards cache.

    Returns:
        Number of entries cleared
    """
    if _USE_BOUNDED_CACHE:
        count = _standards_cache.clear()
    else:
        with _standards_cache_lock:
            count = len(_standards_cache)
            _standards_cache.clear()

    logger.info(f"[StandardsCache] Cache cleared ({count} entries)")
    return count


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.

    Returns:
        Dict with cache statistics (hits, misses, size, etc.)
    """
    if _USE_BOUNDED_CACHE:
        return _standards_cache.get_stats()
    else:
        with _standards_cache_lock:
            return {
                "size": len(_standards_cache),
                "max_size": CACHE_MAX_SIZE,
                "hits": 0,  # Not tracked in simple cache
                "misses": 0,
                "type": "simple_dict"
            }


def invalidate_cache_entry(
    product_type: str,
    source_filter: Optional[str] = None
) -> bool:
    """
    Invalidate a specific cache entry.

    Args:
        product_type: Product type to invalidate
        source_filter: Optional source filter

    Returns:
        True if entry was found and removed, False otherwise
    """
    key = get_cache_key(product_type, source_filter)

    if _USE_BOUNDED_CACHE:
        # BoundedCache should have a delete method
        if hasattr(_standards_cache, 'delete'):
            return _standards_cache.delete(key)
        else:
            # Fallback: get and check
            if _standards_cache.get(key) is not None:
                _standards_cache.set(key, None)
                return True
            return False
    else:
        with _standards_cache_lock:
            if key in _standards_cache:
                del _standards_cache[key]
                return True
            return False


# =============================================================================
# CACHE WARMING
# =============================================================================

def warm_cache_for_product_types(
    product_types: list,
    fetcher_func: callable
) -> Dict[str, bool]:
    """
    Pre-warm the cache for a list of product types.

    Args:
        product_types: List of product types to warm
        fetcher_func: Function to call to fetch standards for a product type

    Returns:
        Dict mapping product_type -> success status
    """
    results = {}

    for product_type in product_types:
        # Check if already cached
        if get_cached_standards(product_type) is not None:
            results[product_type] = True
            continue

        try:
            # Fetch and cache
            result = fetcher_func(product_type)
            if result:
                cache_standards(product_type, result)
                results[product_type] = True
            else:
                results[product_type] = False
        except Exception as e:
            logger.warning(f"[StandardsCache] Failed to warm cache for {product_type}: {e}")
            results[product_type] = False

    logger.info(
        f"[StandardsCache] Cache warming complete: "
        f"{sum(results.values())}/{len(results)} successful"
    )
    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Key generation
    "get_cache_key",
    # Core operations
    "get_cached_standards",
    "cache_standards",
    "clear_standards_cache",
    "get_cache_stats",
    "invalidate_cache_entry",
    # Warming
    "warm_cache_for_product_types",
]
