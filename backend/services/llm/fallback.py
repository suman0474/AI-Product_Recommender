"""
LLM Fallback Utility
Provides automatic fallback from Google Gemini to OpenAI when Gemini fails
Includes timeout support for LLM calls

PHASE 1 FIX: Uses centralized API Key Manager instead of global state
"""
import os
import logging
import threading
from typing import Optional, Any
from functools import wraps

logger = logging.getLogger(__name__)

# Issue-specific debug logging for terminal log analysis
try:
    from debug_flags import issue_debug, debug_log, timed_execution, is_debug_enabled
except ImportError:
    issue_debug = None  # Fallback if debug_flags not available
    debug_log = lambda *a, **kw: lambda f: f  # No-op decorator
    timed_execution = lambda *a, **kw: lambda f: f  # No-op decorator
    is_debug_enabled = lambda m: False

# PHASE 1 FIX: Use centralized API Key Manager
# Replaces global state with thread-safe singleton
try:
    from config.api_key_manager import api_key_manager
    GOOGLE_API_KEY = api_key_manager.get_current_google_key()
    OPENAI_API_KEY = api_key_manager.get_openai_key()
    GOOGLE_API_KEYS = [api_key_manager.get_google_key_by_index(i)
                       for i in range(api_key_manager.get_google_key_count())]
except ImportError:
    # Fallback if api_key_manager not available (initialization phase)
    logger.warning("[LLM_FALLBACK] API Key Manager not available, using environment variables directly")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEYS = [GOOGLE_API_KEY] if GOOGLE_API_KEY else []

def get_current_google_api_key() -> str:
    """
    Get the current Google API key.
    PHASE 1 FIX: Uses API Key Manager instead of global state.
    """
    try:
        from config.api_key_manager import api_key_manager
        return api_key_manager.get_current_google_key() or ""
    except ImportError:
        return GOOGLE_API_KEY or ""

def rotate_google_api_key() -> bool:
    """
    Rotate to the next available Google API key.
    PHASE 1 FIX: Uses API Key Manager (thread-safe) instead of global state.

    Returns:
        True if rotation succeeded, False if only one key available
    """
    try:
        from config.api_key_manager import api_key_manager
        return api_key_manager.rotate_google_key()
    except ImportError:
        logger.warning("[LLM_FALLBACK] API Key Manager not available, cannot rotate")
        return False

# Log available keys at startup
if len(GOOGLE_API_KEYS) > 1:
    logger.info(f"[LLM_FALLBACK] 🔑 Loaded {len(GOOGLE_API_KEYS)} Google API keys for rotation")
    if issue_debug:
        issue_debug._log("API_KEY", f"INIT: Loaded {len(GOOGLE_API_KEYS)} keys for rotation")

# Model mappings: Gemini -> OpenAI equivalent
MODEL_MAPPINGS = {
    "gemini-2.5-flash": "gpt-4o-mini",
    "gemini-2.5-pro": "gpt-4o",
    "gemini-1.5-pro": "gpt-4o",
}


# ============================================================================
# RATE LIMIT OPTIMIZATION: Proactive Request Tracking
# ============================================================================
# Tracks requests per minute to fail-fast before hitting API quota
# This avoids 30-60s retry waits when quota is exhausted

import time as _time
from threading import Lock as _Lock

_rate_tracker = {
    "count": 0,
    "reset_time": _time.time() + 60,
    "max_per_minute": 15,  # Leave headroom for 20 RPM free tier limit
    "quota_exhausted_until": 0,  # Timestamp when quota will reset (from 429 errors)
}
_rate_tracker_lock = _Lock()


def check_rate_limit() -> bool:
    """
    Check if we should proceed with an API request.
    
    Returns:
        True if we should proceed, False if rate limited
        
    RATE LIMIT OPTIMIZATION: Fail-fast before hitting API quota.
    """
    with _rate_tracker_lock:
        now = _time.time()
        
        # If we know quota is exhausted, check if reset time has passed
        if now < _rate_tracker["quota_exhausted_until"]:
            remaining = int(_rate_tracker["quota_exhausted_until"] - now)
            logger.warning(f"[RATE_LIMIT] Quota exhausted, {remaining}s until reset")
            return False
        
        # Reset counter if minute has passed
        if now > _rate_tracker["reset_time"]:
            _rate_tracker["count"] = 0
            _rate_tracker["reset_time"] = now + 60
        
        # Check if we're at the limit
        if _rate_tracker["count"] >= _rate_tracker["max_per_minute"]:
            logger.warning(f"[RATE_LIMIT] Rate limit reached ({_rate_tracker['max_per_minute']}/min)")
            return False
        
        _rate_tracker["count"] += 1
        return True


def mark_quota_exhausted(retry_after_seconds: int = 60):
    """
    Mark quota as exhausted after receiving a 429 error.
    
    Args:
        retry_after_seconds: Seconds from the RetryInfo in the 429 response
    """
    with _rate_tracker_lock:
        _rate_tracker["quota_exhausted_until"] = _time.time() + retry_after_seconds
        logger.warning(f"[RATE_LIMIT] Marked quota exhausted for {retry_after_seconds}s")


def get_rate_limit_status() -> dict:
    """Get current rate limit status for monitoring."""
    with _rate_tracker_lock:
        now = _time.time()
        return {
            "requests_this_minute": _rate_tracker["count"],
            "max_per_minute": _rate_tracker["max_per_minute"],
            "seconds_until_reset": max(0, int(_rate_tracker["reset_time"] - now)),
            "quota_exhausted": now < _rate_tracker["quota_exhausted_until"],
            "quota_reset_in": max(0, int(_rate_tracker["quota_exhausted_until"] - now))
        }


# ============================================================================
# DIRECT OPENAI WRAPPER (FALLBACK FOR TIKTOKEN ISSUES)
# ============================================================================
# When LangChain's ChatOpenAI fails due to tiktoken circular imports,
# this wrapper provides a compatible interface using the direct OpenAI client.

class DirectOpenAIWrapper:
    """
    LangChain-compatible wrapper for direct OpenAI API calls.
    
    Used as a fallback when LangChain's ChatOpenAI fails to initialize
    (e.g., due to tiktoken circular import issues on Windows).
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", 
                 temperature: float = 0.1, max_tokens: int = 4096):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
    
    def _get_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def invoke(self, messages, **kwargs):
        """
        LangChain-compatible invoke method.
        
        Accepts either:
        - A string prompt
        - A list of LangChain message objects
        - A list of OpenAI message dicts
        """
        client = self._get_client()
        
        # Convert LangChain messages to OpenAI format
        openai_messages = []
        if isinstance(messages, str):
            openai_messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            for msg in messages:
                if hasattr(msg, 'type') and hasattr(msg, 'content'):
                    # LangChain HumanMessage/AIMessage/SystemMessage
                    role = "assistant" if msg.type == "ai" else ("system" if msg.type == "system" else "user")
                    openai_messages.append({"role": role, "content": msg.content})
                elif isinstance(msg, dict):
                    openai_messages.append(msg)
        
        # Make API call
        response = client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Return content in a LangChain-compatible format
        content = response.choices[0].message.content
        
        # Create a simple object that has .content attribute
        class AIMessageLike:
            def __init__(self, content):
                self.content = content
                self.type = "ai"
        
        return AIMessageLike(content)
    
    def __call__(self, messages, **kwargs):
        """Make wrapper callable like LangChain LLM."""
        return self.invoke(messages, **kwargs)


# ============================================================================
# PHASE 3: PER-MODEL QUOTA TRACKING WITH FAIL-FAST
# ============================================================================
# Tracks which specific models have exhausted quota (e.g., gemini-2.5-pro has 0 daily limit)
# This allows fast fallback to models that still have available quota

_model_quota_status = {
    # Example: "gemini-2.5-pro": {"exhausted_until": timestamp, "daily_limit": 0, "reason": "RESOURCE_EXHAUSTED"}
}
_model_quota_lock = _Lock()


def is_model_quota_exhausted(model: str) -> bool:
    """
    Check if a specific model's quota is exhausted.
    
    This allows fail-fast when a model like gemini-2.5-pro has 0 daily limit,
    rather than waiting for the API to return 429 errors.
    
    Args:
        model: The model name to check (e.g., "gemini-2.5-pro")
        
    Returns:
        True if the model's quota is exhausted, False otherwise
    """
    with _model_quota_lock:
        if model not in _model_quota_status:
            return False
            
        status = _model_quota_status[model]
        now = _time.time()
        
        # Check if exhausted_until has passed
        if status.get("exhausted_until", 0) > 0:
            if now < status["exhausted_until"]:
                remaining = int(status["exhausted_until"] - now)
                logger.debug(f"[MODEL_QUOTA] {model} exhausted for {remaining}s more")
                return True
            else:
                # Reset the status if time has passed
                del _model_quota_status[model]
                logger.info(f"[MODEL_QUOTA] {model} quota reset, now available")
                return False
        
        # Check if model has 0 daily limit (permanent exhaustion for the day)
        if status.get("daily_limit") == 0:
            logger.warning(f"[MODEL_QUOTA] {model} has 0 daily limit - permanently exhausted for today")
            return True
            
        return False


def mark_model_quota_exhausted(model: str, retry_after_seconds: int = 60, daily_limit: int = None):
    """
    Mark a specific model's quota as exhausted after receiving a 429 error.
    
    Args:
        model: The model name (e.g., "gemini-2.5-pro")
        retry_after_seconds: Seconds until quota resets (from RetryInfo in 429 response)
        daily_limit: If known to be 0, marks model as permanently exhausted for the day
    """
    with _model_quota_lock:
        _model_quota_status[model] = {
            "exhausted_until": _time.time() + retry_after_seconds,
            "daily_limit": daily_limit,
            "reason": "RESOURCE_EXHAUSTED",
            "marked_at": _time.time()
        }
        
        if daily_limit == 0:
            logger.warning(f"[MODEL_QUOTA] ❌ {model} marked as PERMANENTLY exhausted (0 daily limit)")
        else:
            logger.warning(f"[MODEL_QUOTA] ⚠️ {model} marked exhausted for {retry_after_seconds}s")


def get_model_quota_status() -> dict:
    """Get per-model quota status for monitoring."""
    with _model_quota_lock:
        now = _time.time()
        result = {}
        for model, status in _model_quota_status.items():
            exhausted_until = status.get("exhausted_until", 0)
            result[model] = {
                "is_exhausted": now < exhausted_until or status.get("daily_limit") == 0,
                "exhausted_until": exhausted_until,
                "seconds_remaining": max(0, int(exhausted_until - now)),
                "daily_limit": status.get("daily_limit"),
                "reason": status.get("reason", "unknown")
            }
        return result


def _parse_retry_delay(error_str: str) -> int:
    """
    Parse the retry delay from a 429 error message.
    
    Gemini API returns errors like:
    - "retryDelay: 60s"
    - "Retry-After: 120"
    - "retry_delay_seconds: 30"
    
    Args:
        error_str: The error message string
        
    Returns:
        Retry delay in seconds (default 60 if not parseable)
    """
    import re
    
    # Try to find retryDelay pattern (e.g., "retryDelay: 60s")
    match = re.search(r'retryDelay:\s*(\d+)', error_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Try Retry-After header pattern
    match = re.search(r'retry[-_]?after:\s*(\d+)', error_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Try retry_delay_seconds pattern
    match = re.search(r'retry_delay_seconds:\s*(\d+)', error_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Try generic seconds pattern
    match = re.search(r'(\d+)\s*(?:s|seconds?)', error_str)
    if match:
        delay = int(match.group(1))
        if 1 <= delay <= 3600:  # Sanity check: 1 second to 1 hour
            return delay
    
    # Default to 60 seconds if no delay found
    return 60


class LLMTimeoutError(Exception):
    """Exception raised when LLM call exceeds timeout"""
    pass


from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output
from pydantic import Field, PrivateAttr
from typing import Iterator, List, Union

class LLMWithTimeout(Runnable):
    """
    Wrapper for LLM that adds timeout functionality.
    Uses threading for cross-platform compatibility (Windows-safe).

    Properly implements LangChain Runnable interface for chain compatibility.

    [FIX #3] DEFAULT TIMEOUT: 60 seconds (1 minute) to prevent worst-case hangs.
    Most LLM calls complete in 2-30 seconds. For known long operations (ranking,
    complex synthesis), pass timeout=120 or timeout=180 explicitly.
    """

    # Use PrivateAttr for non-serializable fields (Pydantic v2 compatible)
    _base_llm: Any = PrivateAttr()
    _timeout_seconds: int = PrivateAttr(default=60)  # [FIX #3] 1 minute default (was 10 minutes)
    _model_name: str = PrivateAttr(default="unknown")

    def __init__(self, base_llm: Any, timeout_seconds: int = 60, **kwargs):
        """
        Initialize LLM with timeout wrapper

        Args:
            base_llm: The underlying LLM instance
            timeout_seconds: Maximum seconds to wait for LLM response (default: 60 = 1 minute)
                            [FIX #3] Reduced from 600s to prevent worst-case 10-minute hangs.
        """
        super().__init__(**kwargs)
        self._base_llm = base_llm
        self._timeout_seconds = timeout_seconds
        self._model_name = getattr(base_llm, 'model_name', getattr(base_llm, 'model', 'unknown'))

    @property
    def base_llm(self) -> Any:
        """Get the underlying LLM instance."""
        return self._base_llm

    @property
    def timeout_seconds(self) -> int:
        """Get the timeout in seconds."""
        return self._timeout_seconds

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    class Config:
        """Pydantic config for arbitrary types."""
        arbitrary_types_allowed = True

    def _invoke_with_timeout(self, method_name: str, *args, **kwargs):
        """
        Execute LLM method with timeout using threading

        Args:
            method_name: Name of the method to call ('invoke', 'ainvoke', etc.)
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            Result from the LLM method

        Raises:
            LLMTimeoutError: If the call exceeds timeout_seconds
        """
        result = [None]
        error = [None]

        def target():
            try:
                method = getattr(self._base_llm, method_name)
                result[0] = method(*args, **kwargs)
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self._timeout_seconds)

        if thread.is_alive():
            logger.error(f"[LLM_TIMEOUT] {method_name} call exceeded {self._timeout_seconds}s timeout (model: {self._model_name})")
            raise LLMTimeoutError(
                f"LLM call exceeded {self._timeout_seconds}s timeout (model: {self._model_name})"
            )

        if error[0]:
            raise error[0]

        return result[0]

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs) -> Any:
        """
        Invoke the LLM with timeout protection.

        This is the main method required by the Runnable interface.

        Args:
            input: Input to pass to the LLM (can be str, list of messages, etc.)
            config: Optional runnable config
            **kwargs: Additional keyword arguments

        Returns:
            LLM response

        Raises:
            LLMTimeoutError: If the call exceeds timeout
        """
        # Don't pass config to underlying LLM - it handles its own config
        # This avoids parameter conflicts with different LLM implementations
        return self._invoke_with_timeout('invoke', input, **kwargs)

    def batch(
        self,
        inputs: List[Any],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        **kwargs
    ) -> List[Any]:
        """Batch invoke with timeout"""
        return self._invoke_with_timeout('batch', inputs, **kwargs)

    def stream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> Iterator[Any]:
        """Stream invoke (no timeout applied for streaming)"""
        return self._base_llm.stream(input, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the base LLM"""
        # Avoid recursion for private attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self._base_llm, name)

    def __or__(self, other: Any) -> Any:
        """Implement pipe operator support for LangChain"""
        from langchain_core.runnables import RunnableSequence
        return RunnableSequence(first=self, last=other)

    def __ror__(self, other: Any) -> Any:
        """Implement reverse pipe operator support for LangChain"""
        from langchain_core.runnables import RunnableSequence
        return RunnableSequence(first=other, last=self)

    @property
    def InputType(self) -> type:
        """Return the input type for this runnable."""
        return Any

    @property
    def OutputType(self) -> type:
        """Return the output type for this runnable."""
        return Any


def get_openai_equivalent(gemini_model: str) -> str:
    """
    Get the OpenAI model equivalent for a Gemini model

    Args:
        gemini_model: Gemini model name

    Returns:
        OpenAI model name
    """
    return MODEL_MAPPINGS.get(gemini_model, "gpt-4o-mini")


def _create_openai_fallback(
    gemini_model: str,
    temperature: float,
    openai_api_key: str,
    max_tokens: Optional[int],
    timeout: Optional[int],
    extra_kwargs: dict
) -> Any:
    """
    Helper to create OpenAI fallback LLM.
    
    Used by the fail-fast logic when a Gemini model's quota is known to be exhausted.
    
    Args:
        gemini_model: Original Gemini model name (for mapping to OpenAI equivalent)
        temperature: Temperature for generation
        openai_api_key: OpenAI API key
        max_tokens: Maximum tokens for generation
        timeout: Timeout in seconds (None to disable)
        extra_kwargs: Additional kwargs to pass to ChatOpenAI
        
    Returns:
        LLM instance (ChatOpenAI)
    """
    from langchain_openai import ChatOpenAI
    
    openai_model = get_openai_equivalent(gemini_model)
    
    logger.info(f"[LLM_FALLBACK] 🔀 Using OpenAI fallback: {openai_model} (equivalent to {gemini_model})")
    
    llm = ChatOpenAI(
        model=openai_model,
        temperature=temperature,
        openai_api_key=openai_api_key,
        max_tokens=max_tokens,
        **extra_kwargs
    )
    
    logger.info(f"[LLM_FALLBACK] ✅ Successfully initialized OpenAI: {openai_model}")
    
    if timeout is not None:
        logger.info(f"[LLM_FALLBACK] Wrapping LLM with {timeout}s timeout")
        return LLMWithTimeout(llm, timeout_seconds=timeout)
    return llm


# =============================================================================
# ROBUST LANGCHAIN FALLBACK WRAPPER
# =============================================================================

class LLMWithFallback(Runnable):
    """
    A LangChain Runnable that wraps a primary LLM and a fallback LLM.
    If the primary LLM fails (e.g. rate limit, timeout), it tries the fallback.
    """
    def __init__(self, primary_llm, fallback_llm=None, model_name="unknown"):
        self.primary_llm = primary_llm
        self.fallback_llm = fallback_llm
        self.model_name = model_name

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs) -> Any:
        try:
            return self.primary_llm.invoke(input, config=config, **kwargs)
        except Exception as e:
            # Check if we should fallback
            # FIX #4: Expanded trigger list to include API key errors (prevents hangs on expired keys)
            error_str = str(e)
            fallback_triggers = [
                '429', 'Resource exhausted', 'RESOURCE_EXHAUSTED', 'quota', 'timed out',
                # FIX #4: API key/auth errors - fallback to OpenAI when Google key expires
                'API_KEY_INVALID', 'api key expired', 'key expired', 'InvalidArgument',
                'API key not valid', 'invalid api key', 'authentication failed'
            ]
            should_fallback = any(trigger.lower() in error_str.lower() for trigger in fallback_triggers)

            if self.fallback_llm and should_fallback:
                logger.warning(f"[LLM_FALLBACK] Primary model {self.model_name} failed ({type(e).__name__}). Switching to FALLBACK.")
                try:
                    return self.fallback_llm.invoke(input, config=config, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"[LLM_FALLBACK] Fallback model also failed: {fallback_error}")
                    raise e # Raise original error if fallback fails
            
            # If not rate limit or no fallback, re-raise
            raise e

    def batch(self, inputs: list, config: Optional[RunnableConfig] = None, **kwargs) -> list:
        return [self.invoke(i, config=config, **kwargs) for i in inputs]

    def stream(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs):
        return self.primary_llm.stream(input, config=config, **kwargs)
    
    def __getattr__(self, name):
        """Delegate other attributes to primary LLM"""
        return getattr(self.primary_llm, name)


@debug_log("LLM_FALLBACK", log_args=False)
@timed_execution("LLM_FALLBACK", threshold_ms=5000)
def create_llm_with_fallback(
    model: str = "gemini-2.5-flash",
    temperature: float = 0.1,
    google_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    skip_test: bool = True,
    max_afc_calls: Optional[int] = None,
    **kwargs
) -> Any:
    """
    Create an LLM instance with AUTOMATIC RUNTIME FALLBACK.
    Returns a custom Runnable that tries Gemini first, then OpenAI if Gemini fails.
    """
    google_key = google_api_key or GOOGLE_API_KEY
    openai_key = openai_api_key or OPENAI_API_KEY
    
    primary_llm = None
    fallback_llm = None
    
    # 1. Initialize Primary (Gemini)
    if google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            # Pass retry only to Gemini; avoid polluting kwargs for OpenAI (prevents "multiple values for max_retries")
            gemini_kwargs = {k: v for k, v in kwargs.items() if k != "max_retries"}
            gemini_kwargs["max_retries"] = 5

            gemini_llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=google_key,
                max_output_tokens=max_tokens,
                **gemini_kwargs
            )
            
            # Always wrap with timeout for production stability
            # [FIX #3] Default: 60 seconds (1 minute) - reduced from 600s to prevent hangs
            effective_timeout = timeout if timeout else 60
            primary_llm = LLMWithTimeout(gemini_llm, timeout_seconds=effective_timeout)
            logger.info(f"[LLM_FALLBACK] Primary LLM wrapped with {effective_timeout}s timeout")
                
        except Exception as e:
            logger.warning(f"[LLM_FALLBACK] Failed to init Primary ({model}): {e}")
    
    # 2. Initialize Fallback (OpenAI)
    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
            openai_model = get_openai_equivalent(model)

            # Do not pass kwargs['max_retries'] to avoid "multiple values for keyword argument 'max_retries'"
            openai_kwargs = {k: v for k, v in kwargs.items() if k != "max_retries"}

            openai_llm = ChatOpenAI(
                model=openai_model,
                temperature=temperature,
                openai_api_key=openai_key,
                max_tokens=max_tokens,
                max_retries=3,
                **openai_kwargs
            )
            
            # Always wrap with timeout for production stability
            # [FIX #3] Default: 60 seconds (1 minute) - reduced from 600s to prevent hangs
            effective_timeout = timeout if timeout else 60
            fallback_llm = LLMWithTimeout(openai_llm, timeout_seconds=effective_timeout)
            logger.info(f"[LLM_FALLBACK] Fallback LLM wrapped with {effective_timeout}s timeout")
                
        except Exception as e:
            error_str = str(e).lower()
            logger.warning(f"[LLM_FALLBACK] Failed to init Fallback ({model}): {e}")
            
            # Log tiktoken circular import issues for debugging
            if issue_debug and 'tiktoken' in error_str:
                issue_debug.tiktoken_init_error(str(e)[:100])
            if issue_debug and 'circular' in error_str:
                issue_debug.tiktoken_circular_import('llm_fallback')
            
            # FIX: Fallback to direct OpenAI client when LangChain fails (tiktoken issue)
            if 'tiktoken' in error_str or 'circular' in error_str or 'import' in error_str:
                logger.info("[LLM_FALLBACK] Attempting direct OpenAI client (bypassing LangChain)...")
                try:
                    from openai import OpenAI
                    # Create a wrapper that mimics LangChain interface
                    fallback_llm = DirectOpenAIWrapper(
                        api_key=openai_key,
                        model=get_openai_equivalent(model),
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    logger.info("[LLM_FALLBACK] ✓ Direct OpenAI client initialized successfully")
                except Exception as direct_e:
                    logger.error(f"[LLM_FALLBACK] Direct OpenAI also failed: {direct_e}")

    # 3. Construct Final Runnable
    if primary_llm:
        if fallback_llm:
            return LLMWithFallback(primary_llm, fallback_llm, model_name=model)
        else:
            return primary_llm
    elif fallback_llm:
        return fallback_llm
    else:
        raise RuntimeError("Could not initialize ANY LLM (Primary or Fallback)")


def create_llm_langchain(
    model: str = "gemini-2.5-flash",
    temperature: float = 0.1,
    google_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Alias for create_llm_with_fallback for LangChain compatibility

    This is the recommended function to use across the codebase.
    """
    return create_llm_with_fallback(
        model=model,
        temperature=temperature,
        google_api_key=google_api_key,
        openai_api_key=openai_api_key,
        **kwargs
    )


class FallbackLLMClient:
    """
    Wrapper class for non-LangChain LLM usage with fallback support.
    Supports multi-key rotation for Google API keys.
    Similar to GeminiClient but with OpenAI fallback.
    
    Note: Imports from agentic.infrastructure.state.context.managers are done lazily in methods
    to avoid circular import issues.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        self.client = None
        self.client_type = None  # 'gemini' or 'openai'
        self._current_key_idx = 0
        
        # Determine which Google API keys to use
        if api_key:
            self._google_api_keys = [api_key]
        elif GOOGLE_API_KEYS:
            self._google_api_keys = GOOGLE_API_KEYS.copy()
        elif GOOGLE_API_KEY:
            self._google_api_keys = [GOOGLE_API_KEY]
        else:
            self._google_api_keys = []
        
        # Context manager support - lazy initialization
        self._resource_manager = None
        self._registry = None

        self._initialize_client()
        
    def __enter__(self):
        """Enable context manager usage"""
        # Lazy import to avoid circular dependency
        from agentic.infrastructure.state.context.managers import LLMResourceManager
        self._resource_manager = LLMResourceManager(
            "llm_client",
            f"fallback_{self.model_name}_{id(self)}",
            timeout_seconds=150
        )
        self._resource_manager.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup LLM resources"""
        if self._resource_manager:
            self._resource_manager.__exit__(exc_type, exc_val, exc_tb)
        return False

    def _initialize_client(self):
        """Initialize the LLM client with multi-key rotation and fallback logic."""
        # Try each Google API key
        for key_idx, google_key in enumerate(self._google_api_keys):
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import google.generativeai as genai
                genai.configure(api_key=google_key)
                self.client = genai.GenerativeModel(self.model_name)
                self.client_type = 'gemini'
                self._current_key_idx = key_idx
                logger.info(f"[FallbackLLMClient] Using Gemini: {self.model_name} (key #{key_idx + 1}/{len(self._google_api_keys)})")
                return
            except Exception as e:
                logger.warning(f"[FallbackLLMClient] Gemini key #{key_idx + 1} failed: {e}")
                continue

        # Fallback to OpenAI
        if self.openai_api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.openai_api_key)
                self.client_type = 'openai'
                self.openai_model = get_openai_equivalent(self.model_name)
                logger.info(f"[FallbackLLMClient] 🔀 Using OpenAI fallback: {self.openai_model}")
                return
            except Exception as e:
                logger.error(f"[FallbackLLMClient] OpenAI initialization failed: {e}")
                raise RuntimeError(f"Both Gemini and OpenAI initialization failed")

        raise RuntimeError("No valid API keys available for LLM initialization")
    
    def _rotate_to_next_key(self) -> bool:
        """Try rotating to the next available Google API key. Returns True if successful."""
        if len(self._google_api_keys) <= 1:
            return False
        
        start_idx = self._current_key_idx
        for _ in range(len(self._google_api_keys) - 1):
            next_idx = (self._current_key_idx + 1) % len(self._google_api_keys)
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import google.generativeai as genai
                genai.configure(api_key=self._google_api_keys[next_idx])
                self.client = genai.GenerativeModel(self.model_name)
                self.client_type = 'gemini'
                self._current_key_idx = next_idx
                logger.info(f"[FallbackLLMClient] 🔄 Rotated to key #{next_idx + 1}/{len(self._google_api_keys)}")
                if issue_debug:
                    issue_debug.api_key_rotated(start_idx, next_idx, "quota_error")
                return True
            except Exception as e:
                self._current_key_idx = next_idx
                logger.warning(f"[FallbackLLMClient] Key #{next_idx + 1} rotation failed: {e}")
                continue
        
        return False

    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Invoke the LLM with a prompt. Uses key rotation on quota errors.

        Args:
            prompt: The input prompt
            **kwargs: Additional arguments

        Returns:
            Generated text response
        """
        import time
        
        max_retries = 5
        base_retry_delay = 5  # Base delay in seconds (exponential backoff)
        last_exception = None
        keys_tried = 0  # Track how many keys we've rotated through
        
        for attempt in range(max_retries):
            try:
                if self.client_type == 'gemini':
                    response = self.client.generate_content(
                        prompt,
                        generation_config={
                            'temperature': self.temperature,
                            **kwargs
                        }
                    )
                    return response.text

                elif self.client_type == 'openai':
                    response = self.client.chat.completions.create(
                        model=self.openai_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        **kwargs
                    )
                    return response.choices[0].message.content

            except Exception as e:
                error_msg = str(e)
                is_rate_limit = any(x in error_msg for x in ['429', 'Resource exhausted', 'RESOURCE_EXHAUSTED', 'quota', '503', 'overloaded'])
                
                if is_rate_limit and self.client_type == 'gemini':
                    # First, try rotating to another Google API key
                    if keys_tried < len(self._google_api_keys) - 1:
                        logger.warning(f"[FallbackLLMClient] ⚠️ Quota exhausted on key #{self._current_key_idx + 1}, trying next key...")
                        if self._rotate_to_next_key():
                            keys_tried += 1
                            continue  # Retry immediately with new key
                    
                    # If no more keys or rotation failed, wait and retry
                    if attempt < max_retries - 1:
                        wait_time = base_retry_delay * (2 ** attempt)
                        logger.warning(f"[FallbackLLMClient] Rate limit hit, retry {attempt + 1}/{max_retries} after {wait_time}s")
                        time.sleep(wait_time)
                        last_exception = e
                        continue
                elif is_rate_limit and attempt < max_retries - 1:
                    # OpenAI rate limits - just do exponential backoff
                    wait_time = base_retry_delay * (2 ** attempt)
                    logger.warning(f"[FallbackLLMClient] Rate limit hit, retry {attempt + 1}/{max_retries} after {wait_time}s")
                    time.sleep(wait_time)
                    last_exception = e
                    continue
                
                logger.error(f"[FallbackLLMClient] Error invoking LLM: {e}")
                last_exception = e

                # Try to failover to OpenAI if using Gemini
                if self.client_type == 'gemini' and self.openai_api_key:
                    logger.info("[FallbackLLMClient] 🔀 Attempting runtime failover to OpenAI...")
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=self.openai_api_key)
                        openai_model = get_openai_equivalent(self.model_name)

                        response = client.chat.completions.create(
                            model=openai_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=self.temperature,
                            **kwargs
                        )

                        # Update client for future calls
                        self.client = client
                        self.client_type = 'openai'
                        self.openai_model = openai_model
                        logger.info(f"[FallbackLLMClient] ✅ Runtime failover successful to {openai_model}")

                        return response.choices[0].message.content
                    except Exception as fallback_error:
                        logger.error(f"[FallbackLLMClient] ❌ Runtime failover failed: {fallback_error}")

                raise e
        
        # If we exhausted all retries without success
        if last_exception:
            raise last_exception
        raise RuntimeError("LLM invocation failed after retries")


@debug_log("LLM_FALLBACK", log_args=False)
@timed_execution("LLM_FALLBACK", threshold_ms=30000)
def invoke_with_retry_fallback(
    chain_or_llm: Any,
    input_data: dict,
    max_retries: int = 3,
    fallback_to_openai: bool = True,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.1
) -> Any:
    """
    Invoke LLM chain with automatic retry, key rotation, and OpenAI fallback.

    This function provides runtime error handling for LangChain chains/LLMs,
    handling RESOURCE_EXHAUSTED errors by:
    1. Checking proactive rate limit before invoking (FAST-FAIL)
    2. Retrying with exponential backoff
    3. Rotating to next available Google API key
    4. Falling back to OpenAI if all Google keys are exhausted

    Args:
        chain_or_llm: LangChain chain or LLM to invoke
        input_data: Input dictionary to pass to invoke()
        max_retries: Maximum retry attempts per key (default: 3)
        fallback_to_openai: Whether to fallback to OpenAI after Google keys exhausted
        model: Model name for OpenAI fallback mapping (default: gemini-2.5-flash)
        temperature: Temperature for OpenAI fallback (default: 0.1)

    Returns:
        Result from the chain/LLM invoke

    Raises:
        RuntimeError: If all retry and fallback options are exhausted
    """
    import time
    import re

    # RATE LIMIT OPTIMIZATION: Check proactive rate limit before any API call
    if not check_rate_limit():
        status = get_rate_limit_status()
        if status["quota_exhausted"]:
            error_msg = f"Rate limited: quota exhausted, retry in {status['quota_reset_in']}s"
        else:
            error_msg = f"Rate limited: {status['requests_this_minute']}/{status['max_per_minute']} requests this minute"
        logger.warning(f"[LLM_FALLBACK] {error_msg}")
        raise RuntimeError(error_msg)

    # Track which keys we've tried
    keys_tried = set()
    total_attempts = 0
    max_total_attempts = max_retries * max(len(GOOGLE_API_KEYS), 1) + (max_retries if fallback_to_openai else 0)
    last_error = None

    while total_attempts < max_total_attempts:
        try:
            result = chain_or_llm.invoke(input_data)
            return result

        except Exception as e:
            error_str = str(e)
            total_attempts += 1
            last_error = e

            # Check if it's a rate limit / quota error
            is_quota_error = any(x in error_str for x in [
                '429', 'RESOURCE_EXHAUSTED', 'quota',
                'Rate limit', 'overloaded', '503'
            ])

            if is_quota_error:
                logger.warning(f"[LLM_FALLBACK] Quota error on attempt {total_attempts}: {error_str[:100]}")
                
                # Phase 3/6: Extract retry-after from error and mark quota exhausted
                retry_seconds = _parse_retry_delay(error_str)
                mark_quota_exhausted(retry_seconds)
                
                # Phase 3: Also mark the specific model as exhausted for fail-fast
                mark_model_quota_exhausted(model, retry_after_seconds=retry_seconds)

                # Try rotating to next Google API key
                current_key = get_current_google_api_key()
                if current_key and current_key not in keys_tried:
                    keys_tried.add(current_key)

                if rotate_google_api_key():
                    new_key = get_current_google_api_key()
                    if new_key and new_key not in keys_tried:
                        logger.info(f"[LLM_FALLBACK] Rotated to new key, retrying...")
                        # Recreate the LLM with the new key
                        try:
                            new_llm = create_llm_with_fallback(
                                model=model,
                                temperature=temperature,
                                skip_test=True
                            )
                            # If chain_or_llm is a chain (has prompt | llm), try to rebuild
                            if hasattr(chain_or_llm, 'first') and hasattr(chain_or_llm, 'last'):
                                # It's a RunnableSequence - rebuild with new LLM
                                # Try to get the prompt and parser
                                chain_or_llm = chain_or_llm.first | new_llm | chain_or_llm.last
                            else:
                                chain_or_llm = new_llm
                            continue
                        except Exception as rebuild_error:
                            logger.warning(f"[LLM_FALLBACK] Failed to rebuild chain: {rebuild_error}")

                # If we've tried all Google keys, try OpenAI fallback
                if fallback_to_openai and OPENAI_API_KEY:
                    logger.info(f"[LLM_FALLBACK] 🔀 Attempting OpenAI fallback...")
                    try:
                        from langchain_openai import ChatOpenAI
                        openai_model = get_openai_equivalent(model)

                        openai_llm = ChatOpenAI(
                            model=openai_model,
                            temperature=temperature,
                            openai_api_key=OPENAI_API_KEY
                        )

                        # Rebuild chain with OpenAI LLM if possible
                        if hasattr(chain_or_llm, 'first') and hasattr(chain_or_llm, 'last'):
                            chain_or_llm = chain_or_llm.first | openai_llm | chain_or_llm.last
                        else:
                            chain_or_llm = openai_llm

                        logger.info(f"[LLM_FALLBACK] ✅ Switched to OpenAI {openai_model}, retrying...")
                        fallback_to_openai = False  # Don't try OpenAI again
                        continue

                    except Exception as openai_error:
                        logger.error(f"[LLM_FALLBACK] ❌ OpenAI fallback failed: {openai_error}")

                # Exponential backoff before retry (reduced from 30s max to 10s for faster response)
                if total_attempts < max_total_attempts:
                    wait_time = min(2 ** (total_attempts - 1) * 2, 10)  # Cap at 10 seconds
                    logger.info(f"[LLM_FALLBACK] Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
            else:
                # Non-quota error - don't retry
                logger.error(f"[LLM_FALLBACK] Non-retryable error: {error_str}")
                raise e

    # All attempts exhausted
    error_msg = f"All {total_attempts} LLM invocation attempts failed. Last error: {last_error}"
    logger.error(f"[LLM_FALLBACK] ❌ {error_msg}")
    raise RuntimeError(error_msg)


# Convenience functions
@debug_log("LLM_FALLBACK")
def get_default_llm(temperature: float = 0.1, model: str = "gemini-2.5-flash") -> Any:
    """Get a default LLM instance with fallback"""
    return create_llm_with_fallback(model=model, temperature=temperature)


@debug_log("LLM_FALLBACK")
def get_llm_for_task(task_type: str = "general", temperature: float = 0.1) -> Any:
    """
    Get an LLM optimized for a specific task type

    Args:
        task_type: Type of task ('general', 'fast', 'precise', 'creative')
        temperature: Temperature setting

    Returns:
        LLM instance with fallback
    """
    task_configs = {
        "general": {"model": "gemini-2.5-flash", "temp": 0.1},
        "fast": {"model": "gemini-2.5-flash", "temp": 0.0},
        "precise": {"model": "gemini-2.5-flash", "temp": 0.0},
        "creative": {"model": "gemini-2.5-flash", "temp": 0.7},
    }

    config = task_configs.get(task_type, task_configs["general"])
    return create_llm_with_fallback(
        model=config["model"],
        temperature=temperature if temperature != 0.1 else config["temp"]
    )