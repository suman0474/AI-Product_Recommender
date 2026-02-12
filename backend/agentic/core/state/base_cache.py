# agentic/base_cache.py
# =============================================================================
# BASE LRU CACHE - Generic Thread-Safe Cache with TTL and LRU Eviction
# =============================================================================
#
# This module provides a generic base class for LRU caches that is extended
# by specific cache implementations:
#   - EmbeddingCacheManager
#   - LLMResponseCache
#   - ValidationCache
#
# Common features:
#   - LRU (Least Recently Used) eviction
#   - Optional TTL (Time To Live) expiration
#   - Thread-safe operations
#   - Statistics tracking (hits, misses, evictions)
#   - Generic typing support
#
# Usage:
#   # Simple cache without TTL
#   cache = BaseLRUCache[str, List[float]](max_size=1000)
#   cache.put("key", [0.1, 0.2, 0.3])
#   value = cache.get("key")
#
#   # Cache with TTL
#   cache = BaseLRUCache[str, dict](max_size=500, ttl_seconds=3600)
#
# =============================================================================

import hashlib
import logging
import threading
import time
from abc import ABC
from collections import OrderedDict
from typing import (
    Dict, Any, Optional, TypeVar, Generic, Callable, Union, List
)

logger = logging.getLogger(__name__)

# Type variables for generic cache
K = TypeVar('K')  # Key type
V = TypeVar('V')  # Value type


class BaseLRUCache(Generic[K, V]):
    """
    Generic LRU cache with optional TTL support.

    Features:
    - LRU eviction when capacity exceeded
    - Optional TTL-based expiration
    - Thread-safe operations
    - Statistics tracking
    - Customizable hash function for keys

    Type Parameters:
        K: Key type
        V: Value type

    Example:
        # Cache for embeddings (no TTL)
        cache = BaseLRUCache[str, List[float]](max_size=1000)

        # Cache for LLM responses (with 24hr TTL)
        cache = BaseLRUCache[str, str](max_size=500, ttl_seconds=86400)
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: Optional[int] = None,
        name: str = "LRUCache",
        require_admin: bool = False
    ):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of entries (default: 1000)
            ttl_seconds: Time-to-live in seconds, None for no expiration
            name: Name for logging purposes
            require_admin: If True, require admin token to clear cache (PHASE 2 FIX)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.name = name

        # OrderedDict for LRU ordering
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
            "puts": 0
        }

        # PHASE 2 FIX: Protection against accidental cache clears
        self._require_admin = require_admin
        self._admin_token = None

        logger.info(
            f"[{self.name}] Initialized: max_size={max_size}, "
            f"ttl={ttl_seconds}s" if ttl_seconds else f"[{self.name}] Initialized: max_size={max_size}, no TTL"
        )

    # =========================================================================
    # KEY HASHING
    # =========================================================================

    def _hash_key(self, key: K) -> str:
        """
        Generate a hash for the cache key.

        Override in subclasses for custom key hashing.

        Args:
            key: The key to hash

        Returns:
            String hash of the key
        """
        if isinstance(key, str):
            return hashlib.sha256(key.encode()).hexdigest()
        else:
            return hashlib.sha256(str(key).encode()).hexdigest()

    def _create_composite_key(self, *args, **kwargs) -> str:
        """
        Create a composite key from multiple values.

        Useful for caching based on multiple parameters.

        Args:
            *args: Positional values to include in key
            **kwargs: Keyword values to include in key

        Returns:
            Hashed composite key

        Example:
            key = cache._create_composite_key("model", "prompt", temperature=0.7)
        """
        parts = [str(arg) for arg in args]
        parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        composite = "|".join(parts)
        return hashlib.sha256(composite.encode()).hexdigest()

    # =========================================================================
    # CORE OPERATIONS
    # =========================================================================

    def get(self, key: K) -> Optional[V]:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value, or None if not found/expired
        """
        cache_key = self._hash_key(key)

        with self._lock:
            if cache_key not in self._cache:
                self._stats["misses"] += 1
                return None

            entry = self._cache[cache_key]

            # Check TTL if enabled
            if self.ttl_seconds is not None:
                age = time.time() - entry.get("_created_at", 0)
                if age > self.ttl_seconds:
                    # Expired, remove and return None
                    del self._cache[cache_key]
                    self._stats["misses"] += 1
                    self._stats["expirations"] += 1
                    logger.debug(f"[{self.name}] Expired: {cache_key[:8]}...")
                    return None

            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            self._stats["hits"] += 1
            logger.debug(f"[{self.name}] Hit: {cache_key[:8]}...")

            return entry.get("value")

    def put(self, key: K, value: V) -> None:
        """
        Put a value into the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        cache_key = self._hash_key(key)

        with self._lock:
            # If exists, update and move to end
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                self._cache[cache_key] = {
                    "value": value,
                    "_created_at": time.time()
                }
                logger.debug(f"[{self.name}] Updated: {cache_key[:8]}...")
            else:
                # New entry
                self._cache[cache_key] = {
                    "value": value,
                    "_created_at": time.time()
                }
                self._stats["puts"] += 1
                logger.debug(f"[{self.name}] Added: {cache_key[:8]}...")

            # Enforce size limit
            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if over capacity."""
        while len(self._cache) > self.max_size:
            oldest_key, _ = self._cache.popitem(last=False)
            self._stats["evictions"] += 1
            logger.debug(f"[{self.name}] Evicted: {oldest_key[:8]}...")

    def delete(self, key: K) -> bool:
        """
        Delete a specific key from the cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key existed and was deleted
        """
        cache_key = self._hash_key(key)

        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                logger.debug(f"[{self.name}] Deleted: {cache_key[:8]}...")
                return True
            return False

    def contains(self, key: K) -> bool:
        """
        Check if a key exists in cache (without updating LRU order).

        Args:
            key: Cache key

        Returns:
            True if key exists and is not expired
        """
        cache_key = self._hash_key(key)

        with self._lock:
            if cache_key not in self._cache:
                return False

            # Check TTL if enabled
            if self.ttl_seconds is not None:
                entry = self._cache[cache_key]
                age = time.time() - entry.get("_created_at", 0)
                if age > self.ttl_seconds:
                    return False

            return True

    def get_or_compute(
        self,
        key: K,
        compute_fn: Callable[[], V]
    ) -> V:
        """
        Get from cache or compute and cache the value.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached

        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value

        # Compute and cache
        value = compute_fn()
        self.put(key, value)
        return value

    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================

    def get_many(self, keys: List[K]) -> Dict[K, V]:
        """
        Get multiple values from cache.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary of key -> value for found entries
        """
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result

    def put_many(self, items: Dict[K, V]) -> None:
        """
        Put multiple values into cache.

        Args:
            items: Dictionary of key -> value to cache
        """
        for key, value in items.items():
            self.put(key, value)

    # =========================================================================
    # MAINTENANCE
    # =========================================================================

    def clear(self, admin_token: Optional[str] = None) -> int:
        """
        Clear all entries from the cache.

        PHASE 2 FIX: If cache is protected, requires admin token.

        Args:
            admin_token: Admin token (required if cache is protected)

        Returns:
            Number of entries cleared

        Raises:
            PermissionError: If cache is protected and token is invalid
        """
        # PHASE 2 FIX: Check admin token if cache is protected
        if self._require_admin:
            if admin_token != self._admin_token:
                raise PermissionError(
                    f"Cannot clear protected cache '{self.name}' without valid admin token"
                )

        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.warning(f"[{self.name}] Cleared {count} entries (admin: {bool(admin_token)})")
            return count

    def set_admin_token(self, token: str) -> None:
        """
        Set admin token required for protected operations.

        PHASE 2 FIX: Call this during initialization to enable cache protection.

        Args:
            token: Admin token (should come from environment variable)

        Example:
            from advanced_parameters import SCHEMA_PARAM_CACHE
            import os

            cache_admin_token = os.getenv("CACHE_ADMIN_TOKEN")
            if cache_admin_token:
                SCHEMA_PARAM_CACHE.set_admin_token(cache_admin_token)
                logger.info("Cache protected with admin token")
        """
        with self._lock:
            self._admin_token = token
            logger.info(f"[{self.name}] Admin token set (protection enabled)")

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self._stats = {
                "hits": 0,
                "misses": 0,
                "expirations": 0,
                "evictions": 0,
                "puts": 0
            }

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Only has effect if TTL is enabled.

        Returns:
            Number of entries removed
        """
        if self.ttl_seconds is None:
            return 0

        current_time = time.time()
        expired_keys = []

        with self._lock:
            for cache_key, entry in self._cache.items():
                age = current_time - entry.get("_created_at", 0)
                if age > self.ttl_seconds:
                    expired_keys.append(cache_key)

            for key in expired_keys:
                del self._cache[key]
                self._stats["expirations"] += 1

        if expired_keys:
            logger.info(f"[{self.name}] Cleaned up {len(expired_keys)} expired entries")

        return len(expired_keys)

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        with self._lock:
            total_accesses = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                (self._stats["hits"] / total_accesses * 100)
                if total_accesses > 0 else 0
            )

            return {
                "name": self.name,
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate_percent": f"{hit_rate:.1f}",
                "evictions": self._stats["evictions"],
                "expirations": self._stats["expirations"],
                "total_puts": self._stats["puts"]
            }

    def get_hit_rate(self) -> float:
        """
        Get cache hit rate as percentage.

        Returns:
            Hit rate (0-100)
        """
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            if total == 0:
                return 0.0
            return (self._stats["hits"] / total) * 100

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self._stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "expirations": 0,
                "puts": 0
            }
            logger.info(f"[{self.name}] Stats reset")

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def size(self) -> int:
        """Current number of entries in cache."""
        with self._lock:
            return len(self._cache)

    @property
    def is_full(self) -> bool:
        """Check if cache is at capacity."""
        return self.size >= self.max_size


# =============================================================================
# SPECIALIZED CACHE CLASSES
# =============================================================================

class HashKeyCache(BaseLRUCache[str, V], Generic[V]):
    """
    Cache that uses string keys directly (pre-hashed or simple strings).

    Useful when keys are already unique identifiers.
    """

    def _hash_key(self, key: str) -> str:
        """Use key directly if it looks like a hash, otherwise hash it."""
        if len(key) == 64 and all(c in '0123456789abcdef' for c in key.lower()):
            return key  # Already a SHA256 hash
        return super()._hash_key(key)


class CompositeKeyCache(BaseLRUCache[tuple, V], Generic[V]):
    """
    Cache that uses tuple keys for multi-parameter caching.

    Example:
        cache = CompositeKeyCache[str](max_size=500)
        cache.put(("model", "prompt", 0.7), "response")
    """

    def _hash_key(self, key: tuple) -> str:
        """Hash a tuple key."""
        parts = [str(part) for part in key]
        composite = "|".join(parts)
        return hashlib.sha256(composite.encode()).hexdigest()


# =============================================================================
# SINGLETON FACTORY
# =============================================================================

def create_cache_singleton(
    cache_class: type,
    instance_holder: Dict[str, Any],
    instance_key: str = "_instance",
    lock_key: str = "_lock",
    **kwargs
) -> BaseLRUCache:
    """
    Factory for creating singleton cache instances.

    Args:
        cache_class: Cache class to instantiate
        instance_holder: Dict to store singleton
        instance_key: Key for instance in holder
        lock_key: Key for lock in holder
        **kwargs: Arguments to pass to cache constructor

    Returns:
        Singleton cache instance
    """
    if lock_key not in instance_holder:
        instance_holder[lock_key] = threading.Lock()

    with instance_holder[lock_key]:
        if instance_key not in instance_holder:
            instance_holder[instance_key] = cache_class(**kwargs)
            logger.info(f"[{cache_class.__name__}] Singleton instance created")

    return instance_holder[instance_key]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BaseLRUCache',
    'HashKeyCache',
    'CompositeKeyCache',
    'create_cache_singleton'
]
