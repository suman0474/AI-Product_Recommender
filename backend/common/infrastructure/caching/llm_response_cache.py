"""
LLM Response Cache Manager

Caches LLM generation results by prompt hash to avoid redundant API calls.
Significantly reduces LLM costs and improves response time for repeated queries.

Phase 4 Optimization: 20-30% reduction in LLM API calls
"""
import hashlib
import logging
import threading
import time
from typing import Dict, Optional, Any, Tuple
from .base_cache import BaseLRUCache

logger = logging.getLogger(__name__)


class LLMResponseCache(BaseLRUCache[Tuple[str, str, float], str]):
    """
    Cache for LLM-generated responses with TTL and LRU eviction.

    Now inherits from BaseLRUCache for centralized infrastructure.
    """

    def __init__(
        self,
        max_size: int = 500,
        ttl_seconds: int = 86400,  # 24 hours
        hash_algorithm: str = "sha256"
    ):
        """Initialize LLM response cache using BaseLRUCache."""
        super().__init__(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            name="LLMResponseCache"
        )
        self.hash_algorithm = hash_algorithm

    def _hash_key(self, key: Tuple[str, str, float]) -> str:
        """
        Generate cache key from LLM parameters.
        model + prompt + temperature.
        """
        model, prompt, temperature = key
        # Combine parameters into a deterministic key string
        key_str = f"{model}|{temperature}|{prompt}"

        # Generate hash
        if self.hash_algorithm == "sha256":
            return hashlib.sha256(key_str.encode()).hexdigest()
        elif self.hash_algorithm == "md5":
            return hashlib.md5(key_str.encode()).hexdigest()
        else:
            return hashlib.sha256(key_str.encode()).hexdigest()

    def get(
        self,
        model: str,
        prompt: str,
        temperature: float
    ) -> Optional[str]:
        """Get cached LLM response if available."""
        return super().get((model, prompt, temperature))

    def put(
        self,
        model: str,
        prompt: str,
        temperature: float,
        response: str
    ) -> None:
        """Cache an LLM response."""
        super().put((model, prompt, temperature), response)

    def stats(self) -> Dict[str, Any]:
        """Alias for convenience."""
        return self.get_stats()


# Global singleton instance
_llm_response_cache: Optional[LLMResponseCache] = None


def get_llm_response_cache(
    max_size: int = 500,
    ttl_seconds: int = 86400
) -> LLMResponseCache:
    global _llm_response_cache
    if _llm_response_cache is None:
        _llm_response_cache = LLMResponseCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds
        )
    return _llm_response_cache


def clear_llm_response_cache() -> int:
    global _llm_response_cache
    if _llm_response_cache:
        return _llm_response_cache.clear()
    return 0
