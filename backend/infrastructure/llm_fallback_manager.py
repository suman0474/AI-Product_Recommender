"""
Multi-Tier LLM Fallback Manager

Manages automatic fallback across multiple LLM providers with circuit breakers.
Provides graceful degradation when primary LLM services fail.

Architecture:
    User Request → Gemini (primary) → GPT-4 (fallback) → GPT-3.5 (last resort) → Cache

Usage:
    manager = get_llm_manager()
    response = manager.invoke("Analyze this text...")
"""

import os
import logging
import hashlib
import asyncio
import threading
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass

from infrastructure.circuit_breaker import get_circuit_breaker, CircuitBreakerError

logger = logging.getLogger(__name__)


@dataclass
class LLMProvider:
    """Configuration for a single LLM provider."""
    name: str
    client: Any
    cost_per_1k: float
    priority: int
    model: str
    
    def __post_init__(self):
        """Initialize circuit breaker for this provider."""
        self.breaker = get_circuit_breaker(
            name=f"llm:{self.name}",
            failure_threshold=5,
            reset_timeout=60.0
        )


class LLMFallbackManager:
    """
    Manages multi-tier LLM fallback with circuit breakers.
    
    Features:
    - Automatic fallback across multiple providers
    - Circuit breaker integration
    - Cost-optimized provider selection
    - Optional caching (Redis)
    - Observability and metrics
    
    Usage:
        manager = LLMFallbackManager()
        response = manager.invoke("Analyze this text...")
    """
    
    def __init__(self, enable_cache: bool = False):
        """
        Initialize LLM fallback manager.
        
        Args:
            enable_cache: Enable Redis caching for responses
        """
        self.providers: List[LLMProvider] = []
        self.enable_cache = enable_cache
        self._cache: Optional[Any] = None
        
        self._initialize_providers()
        
        if enable_cache:
            self._initialize_cache()
    
    def _initialize_providers(self):
        """Initialize LLM providers in priority order (cheapest first)."""
        
        # Priority 1: Google Gemini (cheapest, primary)
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY1"):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY1")
                self.providers.append(LLMProvider(
                    name="gemini-pro",
                    client=ChatGoogleGenerativeAI(
                        model="gemini-pro",
                        google_api_key=api_key
                    ),
                    cost_per_1k=0.0005,
                    priority=1,
                    model="gemini-pro"
                ))
                logger.info("✓ Gemini provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
        
        # Priority 2: OpenAI GPT-4 (expensive, high quality)
        if os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import ChatOpenAI
                
                self.providers.append(LLMProvider(
                    name="gpt-4",
                    client=ChatOpenAI(
                        model="gpt-4",
                        openai_api_key=os.getenv("OPENAI_API_KEY")
                    ),
                    cost_per_1k=0.03,
                    priority=2,
                    model="gpt-4"
                ))
                logger.info("✓ GPT-4 provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize GPT-4: {e}")
        
        # Priority 3: OpenAI GPT-3.5 (cheaper, fast, last resort)
        if os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import ChatOpenAI
                
                self.providers.append(LLMProvider(
                    name="gpt-3.5-turbo",
                    client=ChatOpenAI(
                        model="gpt-3.5-turbo",
                        openai_api_key=os.getenv("OPENAI_API_KEY")
                    ),
                    cost_per_1k=0.002,
                    priority=3,
                    model="gpt-3.5-turbo"
                ))
                logger.info("✓ GPT-3.5-turbo provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize GPT-3.5: {e}")
        
        # Sort by priority
        self.providers.sort(key=lambda p: p.priority)
        
        if not self.providers:
            logger.error("⚠️  No LLM providers available! Check API keys.")
        else:
            logger.info(
                f"LLMFallbackManager initialized with {len(self.providers)} providers: "
                f"{[p.name for p in self.providers]}"
            )
    
    def _initialize_cache(self):
        """Initialize Redis cache (optional)."""
        try:
            import redis
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", 6379))
            
            self._cache = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True
            )
            # Test connection
            self._cache.ping()
            logger.info(f"✓ Redis cache connected ({redis_host}:{redis_port})")
        except Exception as e:
            logger.warning(f"Redis cache not available: {e}")
            self._cache = None
            self.enable_cache = False
    
    def _generate_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key from prompt and parameters."""
        # Create deterministic hash
        key_data = f"{prompt}:{sorted(kwargs.items())}"
        hash_obj = hashlib.md5(key_data.encode())
        return f"llm:response:{hash_obj.hexdigest()}"
    
    def invoke(
        self,
        prompt: str,
        cache_key: Optional[str] = None,
        fallback_response: Optional[str] = None,
        **llm_kwargs
    ) -> str:
        """
        Invoke LLM with automatic fallback.
        
        Args:
            prompt: The prompt to send to LLM
            cache_key: Optional cache key for storing/retrieving responses
            fallback_response: Optional hardcoded fallback if all providers fail
            **llm_kwargs: Additional arguments to pass to LLM (temperature, max_tokens, etc.)
        
        Returns:
            LLM response text
        
        Raises:
            Exception: If all providers fail and no cache/fallback available
        """
        if not self.providers:
            raise Exception("No LLM providers configured. Check API keys.")
        
        # Generate cache key if not provided
        if cache_key is None and self.enable_cache:
            cache_key = self._generate_cache_key(prompt, **llm_kwargs)
        
        # Try cached response first
        if cache_key and self._cache:
            try:
                cached = self._cache.get(cache_key)
                if cached:
                    logger.info(f"✓ Cache HIT for key: {cache_key[:16]}...")
                    return cached
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}")
        
        last_error = None
        
        # Try each provider in priority order
        for provider in self.providers:
            try:
                logger.debug(f"Trying LLM provider: {provider.name}")
                
                # Execute through circuit breaker
                def _invoke():
                    return provider.client.invoke(prompt, **llm_kwargs)
                
                response = provider.breaker.call(func=_invoke)
                
                # Success! Extract content and cache
                if hasattr(response, 'content'):
                    result = response.content
                else:
                    result = str(response)
                
                # Cache successful response
                if cache_key and self._cache:
                    try:
                        self._cache.setex(cache_key, 3600, result)  # 1 hour TTL
                    except Exception as e:
                        logger.warning(f"Cache set failed: {e}")
                
                logger.info(f"✓ LLM success with provider: {provider.name}")
                return result
            
            except CircuitBreakerError as e:
                logger.warning(
                    f"Circuit breaker OPEN for {provider.name}, trying next provider..."
                )
                last_error = e
                continue
            
            except Exception as e:
                logger.warning(
                    f"Provider {provider.name} failed: {str(e)[:100]}, trying next provider..."
                )
                last_error = e
                continue
        
        # All providers failed
        logger.error(f"❌ All {len(self.providers)} LLM providers failed")
        
        # Try stale cache
        if cache_key and self._cache:
            try:
                cached = self._cache.get(cache_key)
                if cached:
                    logger.warning(f"⚠️  All providers failed, returning stale cache")
                    return cached
            except:
                pass
        
        # Use hardcoded fallback if provided
        if fallback_response:
            logger.warning(f"⚠️  All providers failed, using fallback response")
            return fallback_response
        
        # No cache or fallback, raise error
        raise Exception(
            f"All LLM providers unavailable. Providers tried: "
            f"{[p.name for p in self.providers]}. Last error: {str(last_error)}"
        )
    
    async def invoke_async(
        self,
        prompt: str,
        cache_key: Optional[str] = None,
        fallback_response: Optional[str] = None,
        **llm_kwargs
    ) -> str:
        """
        Async version of invoke().
        
        Args:
            prompt: The prompt to send to LLM
            cache_key: Optional cache key
            fallback_response: Optional hardcoded fallback
            **llm_kwargs: Additional LLM arguments
        
        Returns:
            LLM response text
        """
        if not self.providers:
            raise Exception("No LLM providers configured")
        
        # Generate cache key
        if cache_key is None and self.enable_cache:
            cache_key = self._generate_cache_key(prompt, **llm_kwargs)
        
        # Try cache (async)
        if cache_key and self._cache:
            try:
                # Redis is sync, so we need to run in executor
                loop = asyncio.get_event_loop()
                cached = await loop.run_in_executor(None, self._cache.get, cache_key)
                if cached:
                    logger.info(f"✓ Cache HIT (async)")
                    return cached
            except Exception as e:
                logger.warning(f"Async cache lookup failed: {e}")
        
        last_error = None
        
        # Try each provider
        for provider in self.providers:
            try:
                logger.debug(f"Trying async LLM provider: {provider.name}")
                
                # Check if client has async invoke
                if hasattr(provider.client, 'ainvoke'):
                    async def _ainvoke():
                        return await provider.client.ainvoke(prompt, **llm_kwargs)
                    
                    response = await provider.breaker.call_async(func=_ainvoke)
                else:
                    # Fall back to sync version
                    def _invoke():
                        return provider.client.invoke(prompt, **llm_kwargs)
                    
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: provider.breaker.call(func=_invoke)
                    )
                
                # Extract content
                if hasattr(response, 'content'):
                    result = response.content
                else:
                    result = str(response)
                
                # Cache result
                if cache_key and self._cache:
                    try:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None,
                            lambda: self._cache.setex(cache_key, 3600, result)
                        )
                    except Exception as e:
                        logger.warning(f"Async cache set failed: {e}")
                
                logger.info(f"✓ Async LLM success with: {provider.name}")
                return result
            
            except CircuitBreakerError as e:
                logger.warning(f"Circuit OPEN for {provider.name} (async)")
                last_error = e
                continue
            
            except Exception as e:
                logger.warning(f"Provider {provider.name} failed (async): {str(e)[:100]}")
                last_error = e
                continue
        
        # All failed - try stale cache or fallback
        if cache_key and self._cache:
            try:
                loop = asyncio.get_event_loop()
                cached = await loop.run_in_executor(None, self._cache.get, cache_key)
                if cached:
                    logger.warning("⚠️  Returning stale cache (async)")
                    return cached
            except:
                pass
        
        if fallback_response:
            logger.warning("⚠️  Using fallback response (async)")
            return fallback_response
        
        raise Exception(
            f"All async LLM providers unavailable. Last error: {str(last_error)}"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all providers."""
        return {
            "providers": [
                {
                    "name": provider.name,
                    "model": provider.model,
                    "priority": provider.priority,
                    "cost_per_1k": provider.cost_per_1k,
                    "circuit": provider.breaker.get_stats()
                }
                for provider in self.providers
            ],
            "cache_enabled": self.enable_cache,
            "cache_connected": self._cache is not None
        }
    
    def reset_all_circuits(self):
        """Reset all provider circuit breakers."""
        for provider in self.providers:
            provider.breaker.reset()
        logger.info(f"Reset all {len(self.providers)} provider circuits")


# ============================================================================
# GLOBAL SINGLETON
# ============================================================================

_llm_manager: Optional[LLMFallbackManager] = None
_manager_lock = threading.RLock()


def get_llm_manager(enable_cache: bool = False) -> LLMFallbackManager:
    """
    Get or create global LLM fallback manager.
    
    Args:
        enable_cache: Enable Redis caching (default: False)
    
    Returns:
        LLMFallbackManager singleton instance
    """
    global _llm_manager
    
    # Import here to avoid circular dependency issues
    import threading
    
    with _manager_lock:
        if _llm_manager is None:
            _llm_manager = LLMFallbackManager(enable_cache=enable_cache)
        return _llm_manager


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'LLMProvider',
    'LLMFallbackManager',
    'get_llm_manager',
]
