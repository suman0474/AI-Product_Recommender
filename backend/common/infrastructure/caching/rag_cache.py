"""
Query Result Caching for RAG Systems

Reduces redundant vector store queries by caching recent results.
Uses LRU eviction with TTL-based expiry.
"""

import hashlib
import time
import logging
from typing import Dict, Any, Optional, Tuple
from .base_cache import BaseLRUCache

logger = logging.getLogger(__name__)


class RAGCache(BaseLRUCache[Tuple[str, str, int], Dict[str, Any]]):
    """
    LRU cache for RAG query results with TTL.
    
    Caches query results to avoid redundant calls to vector stores
    (Pinecone, ChromaDB) and LLM re-processing.
    
    Now inherits from BaseLRUCache for centralized infrastructure.
    """
    
    def __init__(self, ttl: int = 300, max_size: int = 500):
        """Initialize the RAG cache using BaseLRUCache."""
        super().__init__(
            max_size=max_size,
            ttl_seconds=ttl,
            name="RAGCache"
        )
    
    def _hash_key(self, key: Tuple[str, str, int]) -> str:
        """
        Generate cache key from query parameters.
        Normalized query + source + top_k.
        """
        query, source, top_k = key
        normalized_query = query.lower().strip()
        # Remove extra whitespace
        normalized_query = " ".join(normalized_query.split())
        content = f"{source}:{top_k}:{normalized_query}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(
        self, 
        query: str, 
        source: str, 
        top_k: int = 5
    ) -> Optional[Dict[str, Any]]:
        """Get cached result if valid."""
        return super().get((query, source, top_k))
    
    def set(
        self, 
        query: str, 
        source: str, 
        top_k: int, 
        result: Dict[str, Any]
    ) -> None:
        """Cache a query result."""
        # Store metadata in the value for invalidate_source
        tagged_result = {
            "result": result,
            "source": source
        }
        super().put((query, source, top_k), tagged_result)
    
    def put(self, key: Tuple[str, str, int], value: Dict[str, Any]) -> None:
        """Override put to handle tagged results if needed, but normally set() is used."""
        super().put(key, value)
    
    def get_val(self, key: Tuple[str, str, int]) -> Optional[Dict[str, Any]]:
        """Internal helper to get the raw tagged entry."""
        cache_key = self._hash_key(key)
        with self._lock:
            entry = self._cache.get(cache_key)
            if entry:
                # Basic TTL check (duplicating BaseLRUCache logic slightly for internal use)
                if self.ttl_seconds is not None:
                    age = time.time() - entry.get("_created_at", 0)
                    if age > self.ttl_seconds:
                        return None
                return entry.get("value")
            return None

    def invalidate(self, query: str, source: str, top_k: int) -> bool:
        """Invalidate a specific cache entry."""
        return super().delete((query, source, top_k))
    
    def invalidate_source(self, source: str) -> int:
        """Invalidate all cache entries for a source."""
        with self._lock:
            keys_to_remove = []
            for cache_key, entry in self._cache.items():
                val = entry.get("value")
                if isinstance(val, dict) and val.get("source") == source:
                    keys_to_remove.append(cache_key)
            
            for key in keys_to_remove:
                del self._cache[key]
            
            if keys_to_remove:
                logger.info(
                    f"[RAG_CACHE] INVALIDATED {len(keys_to_remove)} entries "
                    f"for source={source}"
                )
            return len(keys_to_remove)

    def stats(self) -> Dict[str, Any]:
        """Alias for convenience."""
        return self.get_stats()

    # Re-map get to return the actual result part, not the tagged part
    def get(self, query: str, source: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
        val = super().get((query, source, top_k))
        if val and isinstance(val, dict) and "result" in val:
            return val["result"]
        return val


def get_rag_cache(ttl: int = 300, max_size: int = 500) -> RAGCache:
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
