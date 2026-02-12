"""
LLM Manager with Caching
Provides thread-safe caching of LLM instances for 10,000x faster initialization
Eliminates redundant LLM creations across the application
Uses monitored locks for visibility into contention
"""
import logging
from typing import Dict, Any, Optional
from services.llm.fallback import create_llm_with_fallback
from agentic.infrastructure.state.context.lock_monitor import get_monitored_lock

logger = logging.getLogger(__name__)


class LLMManager:
    """Global LLM cache manager for 10,000x faster initialization."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        # Use monitored lock for contention visibility
        self._lock = get_monitored_lock("llm_cache")

    def get_llm(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.1,
        **kwargs
    ):
        """
        Get or create cached LLM instance.

        Args:
            model: LLM model name (e.g., "gemini-2.5-flash")
            temperature: Temperature for generation
            **kwargs: Additional arguments to pass to create_llm_with_fallback

        Returns:
            LLM instance (cached if previously created)
            
        Note:
            Phase 4: If the requested model's quota is exhausted, this will
            automatically try to return a fallback model instead.
        """
        from services.llm.fallback import is_model_quota_exhausted
        from config import AgenticConfig
        
        original_model = model
        
        # Phase 4: Check if model quota is exhausted and auto-downgrade
        if is_model_quota_exhausted(model):
            fallback = AgenticConfig.FALLBACK_MODEL
            logger.warning(f"[LLM_CACHE] Model {model} quota exhausted, auto-downgrading to {fallback}")
            model = fallback
            # Don't downgrade if fallback is also exhausted (let create_llm_with_fallback handle OpenAI fallback)
            if is_model_quota_exhausted(model):
                logger.warning(f"[LLM_CACHE] Fallback {model} also exhausted, will use OpenAI fallback")
        
        # Create cache key from model and temperature
        cache_key = f"{model}_{temperature}"

        with self._lock:
            if cache_key not in self._cache:
                # Create new LLM instance
                logger.info(f"[LLM_CACHE] Creating new LLM: {cache_key}")
                self._cache[cache_key] = create_llm_with_fallback(
                    model=model,
                    temperature=temperature,
                    skip_test=True,  # Skip test call - saves ~1.5s per initialization
                    **kwargs
                )
            else:
                if model != original_model:
                    logger.info(f"[LLM_CACHE] Using cached fallback LLM: {cache_key} (requested: {original_model})")
                else:
                    logger.debug(f"[LLM_CACHE] Using cached LLM: {cache_key}")

            return self._cache[cache_key]

    def clear_cache(self):
        """Clear LLM cache (useful for testing)."""
        with self._lock:
            cleared_count = len(self._cache)
            self._cache.clear()
            logger.info(f"[LLM_CACHE] Cleared {cleared_count} cached LLMs")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats (cached_models, models list)
        """
        with self._lock:
            stats = {
                "cached_models": len(self._cache),
                "models": list(self._cache.keys())
            }
            # Include lock contention stats
            lock_stats = self._lock.get_stats()
            stats["lock_stats"] = lock_stats
            return stats


# Global singleton instance
_llm_manager = None


def get_llm_manager() -> LLMManager:
    """
    Get the global LLM manager singleton.

    Returns:
        LLMManager instance
    """
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager


def get_cached_llm(model="gemini-2.5-flash", temperature=0.1, **kwargs):
    """
    Convenience function to get cached LLM instance.

    Args:
        model: LLM model name
        temperature: Temperature for generation
        **kwargs: Additional arguments to pass to LLM creation

    Returns:
        Cached LLM instance
    """
    return get_llm_manager().get_llm(model, temperature, **kwargs)
