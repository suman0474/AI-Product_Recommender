"""
Query Result Caching for RAG Systems

Reduces redundant vector store queries by caching recent results.
Uses LRU eviction with TTL-based expiry.
"""

import hashlib
import time
import logging
from typing import Dict, Any, Optional, Tuple
from threading import Lock
from collections import OrderedDict

logger = logging.getLogger(__name__)


class RAGCache:
    """
    LRU cache for RAG query results with TTL.
    
    Caches query results to avoid redundant calls to vector stores
    (Pinecone, ChromaDB) and LLM re-processing.
    
    Features:
    - TTL-based expiry (default 5 minutes)
    - LRU eviction when capacity reached
    - Thread-safe operations
    - Cache hit/miss statistics
    
    Example:
        >>> cache = RAGCache(ttl=300, max_size=500)
        >>> result = cache.get("What is SIL 3?", "standards", 5)
        >>> if result is None:
        ...     result = expensive_rag_query(...)
        ...     cache.set("What is SIL 3?", "standards", 5, result)
    """
    
    DEFAULT_TTL = 300  # 5 minutes
    MAX_SIZE = 500
    
    def __init__(self, ttl: int = DEFAULT_TTL, max_size: int = MAX_SIZE):
        """
        Initialize the cache.
        
        Args:
            ttl: Time-to-live in seconds for cache entries
            max_size: Maximum number of entries before LRU eviction
        """
        self.ttl = ttl
        self.max_size = max_size
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }
        logger.info(f"[RAG_CACHE] Initialized: ttl={ttl}s, max_size={max_size}")
    
    def _make_key(self, query: str, source: str, top_k: int) -> str:
        """
        Generate cache key from query parameters.
        
        Uses MD5 hash of normalized query + source + top_k.
        """
        normalized_query = query.lower().strip()
        # Remove extra whitespace
        normalized_query = " ".join(normalized_query.split())
        content = f"{source}:{top_k}:{normalized_query}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(
        self, 
        query: str, 
        source: str, 
        top_k: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result if valid.
        
        Args:
            query: The search query
            source: RAG source (index, standards, strategy)
            top_k: Number of results requested
            
        Returns:
            Cached result dictionary if found and not expired, None otherwise
        """
        key = self._make_key(query, source, top_k)
        
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                age = time.time() - entry["timestamp"]
                
                if age < self.ttl:
                    # Valid cache hit
                    self._stats["hits"] += 1
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    logger.debug(
                        f"[RAG_CACHE] HIT: source={source}, age={age:.1f}s, "
                        f"hits={self._stats['hits']}"
                    )
                    return entry["result"]
                else:
                    # Expired entry
                    del self._cache[key]
                    self._stats["expirations"] += 1
                    logger.debug(f"[RAG_CACHE] EXPIRED: source={source}, age={age:.1f}s")
        
        self._stats["misses"] += 1
        logger.debug(
            f"[RAG_CACHE] MISS: source={source}, misses={self._stats['misses']}"
        )
        return None
    
    def set(
        self, 
        query: str, 
        source: str, 
        top_k: int, 
        result: Dict[str, Any]
    ) -> None:
        """
        Cache a query result.
        
        Args:
            query: The search query
            source: RAG source (index, standards, strategy)
            top_k: Number of results requested
            result: The result to cache
        """
        key = self._make_key(query, source, top_k)
        
        with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                self._stats["evictions"] += 1
                logger.debug(f"[RAG_CACHE] EVICTED: LRU entry removed")
            
            self._cache[key] = {
                "timestamp": time.time(),
                "result": result,
                "source": source,
                "query_prefix": query[:50],  # For debugging
            }
            self._cache.move_to_end(key)
            
            logger.debug(
                f"[RAG_CACHE] SET: source={source}, "
                f"cache_size={len(self._cache)}/{self.max_size}"
            )
    
    def invalidate(self, query: str, source: str, top_k: int) -> bool:
        """
        Invalidate a specific cache entry.
        
        Args:
            query: The search query
            source: RAG source
            top_k: Number of results
            
        Returns:
            True if entry was found and removed, False otherwise
        """
        key = self._make_key(query, source, top_k)
        
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"[RAG_CACHE] INVALIDATED: source={source}")
                return True
        return False
    
    def invalidate_source(self, source: str) -> int:
        """
        Invalidate all cache entries for a source.
        
        Args:
            source: RAG source to invalidate (index, standards, strategy)
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            keys_to_remove = [
                key for key, entry in self._cache.items()
                if entry.get("source") == source
            ]
            for key in keys_to_remove:
                del self._cache[key]
            
            if keys_to_remove:
                logger.info(
                    f"[RAG_CACHE] INVALIDATED {len(keys_to_remove)} entries "
                    f"for source={source}"
                )
            return len(keys_to_remove)
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"[RAG_CACHE] CLEARED: {count} entries removed")
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with hits, misses, evictions, hit rate, etc.
        """
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0.0
            
            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "expirations": self._stats["expirations"],
                "hit_rate_percent": round(hit_rate, 2),
                "current_size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl,
            }
    
    def stats(self) -> Dict[str, Any]:
        """Alias for get_stats() for convenience."""
        return self.get_stats()
    
    def cleanup_expired(self) -> int:
        """
        Manually clean up expired entries.
        
        Called automatically on get(), but can be called manually
        for maintenance.
        
        Returns:
            Number of expired entries removed
        """
        now = time.time()
        
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if now - entry["timestamp"] >= self.ttl
            ]
            
            for key in expired_keys:
                del self._cache[key]
                self._stats["expirations"] += 1
            
            if expired_keys:
                logger.info(
                    f"[RAG_CACHE] CLEANUP: {len(expired_keys)} expired entries removed"
                )
            
            return len(expired_keys)


# Global cache instance
_rag_cache: Optional[RAGCache] = None


def get_rag_cache(ttl: int = 300, max_size: int = 500) -> RAGCache:
    """
    Get or create the global RAG cache instance.
    
    Args:
        ttl: Cache TTL in seconds
        max_size: Maximum cache entries
        
    Returns:
        Global RAGCache instance
    """
    global _rag_cache
    if _rag_cache is None:
        _rag_cache = RAGCache(ttl=ttl, max_size=max_size)
    return _rag_cache


# Convenience functions
def cache_get(query: str, source: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
    """Get from cache using global instance."""
    return get_rag_cache().get(query, source, top_k)


def cache_set(query: str, source: str, top_k: int, result: Dict[str, Any]) -> None:
    """Set in cache using global instance."""
    get_rag_cache().set(query, source, top_k, result)


def cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return get_rag_cache().get_stats()
