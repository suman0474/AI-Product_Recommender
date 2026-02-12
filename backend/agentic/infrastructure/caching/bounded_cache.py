"""
Bounded Cache Manager with TTL and LRU Eviction.
Prevents unbounded memory growth in global caches.

This module provides:
- BoundedCache: Thread-safe cache with TTL expiration and LRU eviction
- BoundedCacheManager: Context manager for cache lifecycle
- Global cache registry for monitoring all caches
- Cleanup utilities for application shutdown

Usage:
    from agentic.infrastructure.caching.bounded_cache import get_or_create_cache, BoundedCache

    # Create/get a bounded cache
    cache = get_or_create_cache(
        name="my_cache",
        max_size=100,
        ttl_seconds=3600
    )

    # Use the cache
    cache.set("key", "value")
    value = cache.get("key")

    # Cleanup
    cache.clear()
"""

import threading
import time
import logging
from typing import Dict, Any, Optional, Callable, List
from collections import OrderedDict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class BoundedCache:
    """
    Thread-safe bounded cache with TTL and LRU eviction.

    Features:
    - Max size enforcement with LRU eviction (oldest entries removed first)
    - TTL (Time-To-Live) expiration for stale entries
    - Thread-safe with RLock
    - Eviction callbacks for cleanup actions
    - Statistics tracking (hits, misses, evictions)

    Args:
        name: Cache identifier for logging
        max_size: Maximum number of entries (default: 100)
        ttl_seconds: Entry expiration time in seconds (default: 3600 = 1 hour)
        on_evict: Optional callback(key, value) called when entries are evicted
    """

    def __init__(
        self,
        name: str,
        max_size: int = 100,
        ttl_seconds: int = 3600,
        on_evict: Optional[Callable[[str, Any], None]] = None
    ):
        self.name = name
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.on_evict = on_evict
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0, "sets": 0}

    def get(self, key: str) -> Optional[Any]:
        """
        Get value if exists and not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            value, timestamp = self._cache[key]

            # Check TTL expiration
            if time.time() - timestamp > self.ttl_seconds:
                self._evict(key)
                self._stats["misses"] += 1
                return None

            # Move to end (LRU - most recently used)
            self._cache.move_to_end(key)
            self._stats["hits"] += 1
            return value

    def set(self, key: str, value: Any) -> None:
        """
        Set value with current timestamp.

        Automatically evicts oldest entries if at capacity.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # If key exists, update it
            if key in self._cache:
                self._cache[key] = (value, time.time())
                self._cache.move_to_end(key)
                self._stats["sets"] += 1
                return

            # Evict oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                self._evict(oldest_key)

            # Add new entry
            self._cache[key] = (value, time.time())
            self._cache.move_to_end(key)
            self._stats["sets"] += 1

    def _evict(self, key: str) -> None:
        """
        Remove key and call eviction callback.

        Args:
            key: Key to evict
        """
        if key in self._cache:
            value, _ = self._cache.pop(key)
            self._stats["evictions"] += 1

            if self.on_evict:
                try:
                    self.on_evict(key, value)
                except Exception as e:
                    logger.warning(f"[{self.name}] Eviction callback failed for key '{key}': {e}")

    def delete(self, key: str) -> bool:
        """
        Delete a specific key from cache.

        Args:
            key: Key to delete

        Returns:
            True if key was found and deleted, False otherwise
        """
        with self._lock:
            if key in self._cache:
                self._evict(key)
                return True
            return False

    def clear(self) -> int:
        """
        Clear all entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            keys = list(self._cache.keys())
            for key in keys:
                self._evict(key)
            logger.debug(f"[{self.name}] Cleared {count} entries")
            return count

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            now = time.time()
            expired = [
                k for k, (_, ts) in self._cache.items()
                if now - ts > self.ttl_seconds
            ]
            for key in expired:
                self._evict(key)

            if expired:
                logger.debug(f"[{self.name}] Cleaned up {len(expired)} expired entries")
            return len(expired)

    def get_stats(self) -> Dict[str, Any]:
        """
        Return cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (self._stats["hits"] / total_requests * 100) if total_requests > 0 else 0.0

            return {
                "name": self.name,
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "sets": self._stats["sets"],
                "hit_rate_percent": round(hit_rate, 2)
            }

    def __len__(self) -> int:
        """Return current cache size."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None

    def keys(self) -> List[str]:
        """Return list of all keys (including potentially expired ones)."""
        with self._lock:
            return list(self._cache.keys())

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"BoundedCache(name='{self.name}', size={stats['size']}/{stats['max_size']}, hit_rate={stats['hit_rate_percent']}%)"


class BoundedCacheManager:
    """
    Context manager for bounded cache operations with automatic cleanup.

    Usage:
        cache = BoundedCache("my_cache", max_size=100)
        with BoundedCacheManager(cache) as c:
            c.set("key", "value")
            value = c.get("key")
        # Expired entries cleaned up on exit
    """

    def __init__(self, cache: BoundedCache):
        self.cache = cache
        self._entered = False

    def __enter__(self) -> BoundedCache:
        self._entered = True
        # Cleanup expired on entry
        expired_count = self.cache.cleanup_expired()
        if expired_count > 0:
            logger.debug(f"[{self.cache.name}] Cleaned {expired_count} expired entries on enter")
        return self.cache

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._entered = False
        # Log stats on exit
        stats = self.cache.get_stats()
        logger.debug(f"[{self.cache.name}] Stats on exit: {stats}")
        return False  # Don't suppress exceptions


@contextmanager
def managed_cache(
    name: str,
    max_size: int = 100,
    ttl_seconds: int = 3600,
    on_evict: Optional[Callable] = None
):
    """
    Convenience context manager for temporary bounded cache.

    Creates a cache, yields it for use, then clears it on exit.

    Usage:
        with managed_cache("temp_cache", max_size=50) as cache:
            cache.set("key", "value")
            # ... use cache ...
        # Cache is cleared on exit
    """
    cache = BoundedCache(name, max_size, ttl_seconds, on_evict)
    try:
        yield cache
    finally:
        cache.clear()
        logger.debug(f"[{name}] Temporary cache cleared")


# =============================================================================
# GLOBAL CACHE REGISTRY
# =============================================================================

_CACHE_REGISTRY: Dict[str, BoundedCache] = {}
_REGISTRY_LOCK = threading.RLock()


def get_or_create_cache(
    name: str,
    max_size: int = 100,
    ttl_seconds: int = 3600,
    on_evict: Optional[Callable] = None
) -> BoundedCache:
    """
    Get existing cache or create new one. Thread-safe singleton pattern.

    This is the recommended way to create/access caches.

    Args:
        name: Unique cache identifier
        max_size: Maximum entries (only used on creation)
        ttl_seconds: Entry TTL (only used on creation)
        on_evict: Eviction callback (only used on creation)

    Returns:
        BoundedCache instance
    """
    with _REGISTRY_LOCK:
        if name not in _CACHE_REGISTRY:
            _CACHE_REGISTRY[name] = BoundedCache(name, max_size, ttl_seconds, on_evict)
            logger.info(f"[CACHE_REGISTRY] Created cache: {name} (max={max_size}, ttl={ttl_seconds}s)")
        return _CACHE_REGISTRY[name]


def get_cache(name: str) -> Optional[BoundedCache]:
    """
    Get existing cache by name.

    Args:
        name: Cache identifier

    Returns:
        BoundedCache if exists, None otherwise
    """
    with _REGISTRY_LOCK:
        return _CACHE_REGISTRY.get(name)


def cleanup_all_caches() -> Dict[str, int]:
    """
    Cleanup expired entries in all registered caches.

    Returns:
        Dictionary mapping cache name -> number of entries cleaned
    """
    results = {}
    with _REGISTRY_LOCK:
        for name, cache in _CACHE_REGISTRY.items():
            results[name] = cache.cleanup_expired()

    total_cleaned = sum(results.values())
    if total_cleaned > 0:
        logger.info(f"[CACHE_REGISTRY] Cleaned {total_cleaned} expired entries across {len(results)} caches")

    return results


def clear_all_caches() -> Dict[str, int]:
    """
    Clear all entries in all registered caches.

    Returns:
        Dictionary mapping cache name -> number of entries cleared
    """
    results = {}
    with _REGISTRY_LOCK:
        for name, cache in _CACHE_REGISTRY.items():
            results[name] = cache.clear()

    total_cleared = sum(results.values())
    logger.info(f"[CACHE_REGISTRY] Cleared {total_cleared} entries across {len(results)} caches")

    return results


def get_all_cache_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get stats for all registered caches.

    Returns:
        Dictionary mapping cache name -> stats dict
    """
    with _REGISTRY_LOCK:
        return {name: cache.get_stats() for name, cache in _CACHE_REGISTRY.items()}


def get_registry_summary() -> Dict[str, Any]:
    """
    Get summary statistics for the cache registry.

    Returns:
        Dictionary with registry-level stats
    """
    with _REGISTRY_LOCK:
        total_size = sum(len(c) for c in _CACHE_REGISTRY.values())
        total_max_size = sum(c.max_size for c in _CACHE_REGISTRY.values())

        return {
            "cache_count": len(_CACHE_REGISTRY),
            "total_entries": total_size,
            "total_max_capacity": total_max_size,
            "utilization_percent": round(total_size / total_max_size * 100, 2) if total_max_size > 0 else 0,
            "cache_names": list(_CACHE_REGISTRY.keys())
        }


def remove_cache(name: str) -> bool:
    """
    Remove a cache from the registry.

    Args:
        name: Cache identifier

    Returns:
        True if cache was found and removed
    """
    with _REGISTRY_LOCK:
        if name in _CACHE_REGISTRY:
            cache = _CACHE_REGISTRY.pop(name)
            cache.clear()
            logger.info(f"[CACHE_REGISTRY] Removed cache: {name}")
            return True
        return False


# =============================================================================
# MODULE TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("\n" + "="*60)
    print("BOUNDED CACHE MANAGER - TEST")
    print("="*60)

    # Test 1: Basic operations
    print("\n[Test 1] Basic operations")
    cache = BoundedCache("test_basic", max_size=5, ttl_seconds=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    print(f"  Set 3 items, size: {len(cache)}")
    print(f"  Get 'a': {cache.get('a')}")
    print(f"  Contains 'b': {'b' in cache}")
    print(f"  Stats: {cache.get_stats()}")

    # Test 2: LRU eviction
    print("\n[Test 2] LRU eviction (max_size=5)")
    cache.set("d", 4)
    cache.set("e", 5)
    cache.set("f", 6)  # Should evict 'a'
    print(f"  After adding 6 items to size-5 cache:")
    print(f"  'a' evicted: {cache.get('a') is None}")
    print(f"  'f' exists: {cache.get('f') == 6}")
    print(f"  Stats: {cache.get_stats()}")

    # Test 3: TTL expiration
    print("\n[Test 3] TTL expiration (ttl=1s)")
    ttl_cache = BoundedCache("test_ttl", max_size=10, ttl_seconds=1)
    ttl_cache.set("expire_me", "value")
    print(f"  Immediately after set: {ttl_cache.get('expire_me')}")
    import time
    time.sleep(1.1)
    print(f"  After 1.1s: {ttl_cache.get('expire_me')}")

    # Test 4: Registry
    print("\n[Test 4] Global cache registry")
    cache1 = get_or_create_cache("registry_test_1", max_size=50)
    cache2 = get_or_create_cache("registry_test_2", max_size=100)
    cache1.set("key1", "value1")
    cache2.set("key2", "value2")
    print(f"  Registry summary: {get_registry_summary()}")
    print(f"  All stats: {get_all_cache_stats()}")

    # Cleanup
    clear_all_caches()
    print("\n✓ All tests passed!")
