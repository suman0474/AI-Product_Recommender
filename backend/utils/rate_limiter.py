"""
Rate Limiter and API Quota Management

This module provides rate limiting and quota tracking for API calls to prevent
exceeding API limits and provide better error handling when limits are reached.

Key Features:
- Token bucket rate limiting algorithm
- Per-API quota tracking
- Automatic backoff and retry
- Quota exhaustion detection
- Statistics and monitoring
"""

import time
import threading
import logging
from typing import Dict, Optional, Callable, Any
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class APIQuotaInfo:
    """Information about API quota limits and usage"""
    api_name: str
    requests_per_minute: int = 60
    requests_per_day: Optional[int] = None
    tokens_per_minute: Optional[int] = None
    current_minute_count: int = 0
    current_day_count: int = 0
    current_tokens: int = 0
    minute_reset_time: datetime = field(default_factory=datetime.now)
    day_reset_time: datetime = field(default_factory=datetime.now)
    total_requests: int = 0
    total_quota_errors: int = 0
    last_quota_error: Optional[datetime] = None


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Implements sliding window rate limiting with configurable limits.
    Thread-safe for concurrent access.

    Usage:
        limiter = RateLimiter(requests_per_minute=60)

        # Block until rate limit allows request
        limiter.acquire()
        make_api_call()

        # Or check without blocking
        if limiter.try_acquire():
            make_api_call()
        else:
            print("Rate limit exceeded, try later")
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: Optional[int] = None,
        requests_per_day: Optional[int] = None,
        name: str = "default"
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Max requests per minute
            requests_per_hour: Max requests per hour (optional)
            requests_per_day: Max requests per day (optional)
            name: Name for logging
        """
        self.name = name
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day

        # Sliding window tracking
        self.minute_window = deque(maxlen=requests_per_minute)
        self.hour_window = deque() if requests_per_hour else None
        self.day_window = deque() if requests_per_day else None

        # Statistics
        self.total_requests = 0
        self.total_blocked = 0

        # Thread safety
        self._lock = threading.RLock()

        logger.info(
            f"[RATE_LIMITER:{name}] Initialized - "
            f"{requests_per_minute} req/min, "
            f"{requests_per_hour or 'unlimited'} req/hour, "
            f"{requests_per_day or 'unlimited'} req/day"
        )

    def _clean_old_requests(self):
        """Remove requests outside the time windows"""
        now = time.time()

        # Clean minute window (keep last 60 seconds)
        while self.minute_window and now - self.minute_window[0] > 60:
            self.minute_window.popleft()

        # Clean hour window (keep last 3600 seconds)
        if self.hour_window:
            while self.hour_window and now - self.hour_window[0] > 3600:
                self.hour_window.popleft()

        # Clean day window (keep last 86400 seconds)
        if self.day_window:
            while self.day_window and now - self.day_window[0] > 86400:
                self.day_window.popleft()

    def try_acquire(self) -> bool:
        """
        Try to acquire a slot without blocking.

        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        with self._lock:
            self._clean_old_requests()

            # Check all limits
            if len(self.minute_window) >= self.requests_per_minute:
                self.total_blocked += 1
                return False

            if self.hour_window and self.requests_per_hour:
                if len(self.hour_window) >= self.requests_per_hour:
                    self.total_blocked += 1
                    return False

            if self.day_window and self.requests_per_day:
                if len(self.day_window) >= self.requests_per_day:
                    self.total_blocked += 1
                    return False

            # Record request
            now = time.time()
            self.minute_window.append(now)
            if self.hour_window is not None:
                self.hour_window.append(now)
            if self.day_window is not None:
                self.day_window.append(now)

            self.total_requests += 1
            return True

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a slot, blocking until available or timeout.

        Args:
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            True if acquired, False if timeout reached
        """
        start_time = time.time()
        wait_logged = False

        while True:
            if self.try_acquire():
                return True

            # Check timeout
            if timeout is not None and (time.time() - start_time) >= timeout:
                logger.warning(f"[RATE_LIMITER:{self.name}] Timeout after {timeout}s")
                return False

            # Log only once that we're waiting
            if not wait_logged:
                wait_time = self._time_until_available()
                logger.debug(
                    f"[RATE_LIMITER:{self.name}] Rate limit reached, "
                    f"waiting ~{wait_time:.1f}s"
                )
                wait_logged = True

            # Wait a short time before retrying
            time.sleep(0.1)

    def _time_until_available(self) -> float:
        """Calculate seconds until a slot becomes available"""
        with self._lock:
            self._clean_old_requests()

            if len(self.minute_window) < self.requests_per_minute:
                return 0.0

            # Time until oldest request in minute window expires
            oldest = self.minute_window[0]
            wait_time = 60 - (time.time() - oldest)
            return max(0.0, wait_time)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        with self._lock:
            self._clean_old_requests()

            return {
                "name": self.name,
                "requests_per_minute": self.requests_per_minute,
                "current_minute_count": len(self.minute_window),
                "current_hour_count": len(self.hour_window) if self.hour_window else 0,
                "current_day_count": len(self.day_window) if self.day_window else 0,
                "total_requests": self.total_requests,
                "total_blocked": self.total_blocked,
                "time_until_available": self._time_until_available()
            }

    def reset(self):
        """Reset rate limiter (for testing)"""
        with self._lock:
            self.minute_window.clear()
            if self.hour_window:
                self.hour_window.clear()
            if self.day_window:
                self.day_window.clear()
            self.total_requests = 0
            self.total_blocked = 0
            logger.info(f"[RATE_LIMITER:{self.name}] Reset")


class APIQuotaTracker:
    """
    Track API quota usage and detect quota exhaustion.

    Monitors quota errors and provides recommendations for quota increases.

    Usage:
        tracker = APIQuotaTracker("gemini", requests_per_day=1500)

        try:
            response = api_call()
            tracker.record_success()
        except QuotaError as e:
            tracker.record_quota_error(str(e))
            if tracker.is_quota_exhausted():
                print("Quota exhausted! See recommendations.")
                print(tracker.get_quota_recommendations())
    """

    def __init__(
        self,
        api_name: str,
        requests_per_minute: int = 60,
        requests_per_day: Optional[int] = None,
        tokens_per_minute: Optional[int] = None
    ):
        """
        Initialize quota tracker.

        Args:
            api_name: Name of the API being tracked
            requests_per_minute: Max requests per minute
            requests_per_day: Max requests per day (if known)
            tokens_per_minute: Max tokens per minute (if applicable)
        """
        self.quota_info = APIQuotaInfo(
            api_name=api_name,
            requests_per_minute=requests_per_minute,
            requests_per_day=requests_per_day,
            tokens_per_minute=tokens_per_minute
        )

        # Error tracking
        self.recent_errors = deque(maxlen=100)

        # Thread safety
        self._lock = threading.RLock()

        logger.info(
            f"[QUOTA_TRACKER:{api_name}] Initialized - "
            f"{requests_per_minute} RPM, {requests_per_day or 'unlimited'} RPD"
        )

    def record_success(self):
        """Record a successful API call"""
        with self._lock:
            self.quota_info.total_requests += 1
            self._update_counters()

    def record_quota_error(self, error_message: str):
        """Record a quota/rate limit error"""
        with self._lock:
            self.quota_info.total_quota_errors += 1
            self.quota_info.last_quota_error = datetime.now()
            self.recent_errors.append({
                "timestamp": datetime.now(),
                "message": error_message
            })

            logger.warning(
                f"[QUOTA_TRACKER:{self.quota_info.api_name}] "
                f"Quota error #{self.quota_info.total_quota_errors}: {error_message}"
            )

    def _update_counters(self):
        """Update time-based counters"""
        now = datetime.now()

        # Reset minute counter if needed
        if now - self.quota_info.minute_reset_time > timedelta(minutes=1):
            self.quota_info.current_minute_count = 0
            self.quota_info.minute_reset_time = now

        # Reset day counter if needed
        if now - self.quota_info.day_reset_time > timedelta(days=1):
            self.quota_info.current_day_count = 0
            self.quota_info.day_reset_time = now

        self.quota_info.current_minute_count += 1
        self.quota_info.current_day_count += 1

    def is_quota_exhausted(self, threshold: float = 0.9) -> bool:
        """
        Check if quota is exhausted or near exhaustion.

        Args:
            threshold: Consider exhausted if usage > threshold (default: 0.9 = 90%)

        Returns:
            True if quota is exhausted or near limit
        """
        with self._lock:
            if self.quota_info.requests_per_day:
                usage_ratio = (
                    self.quota_info.current_day_count /
                    self.quota_info.requests_per_day
                )
                if usage_ratio >= threshold:
                    return True

            # Check for frequent recent errors (5+ errors in last minute)
            recent_errors_count = sum(
                1 for err in self.recent_errors
                if (datetime.now() - err["timestamp"]).total_seconds() < 60
            )

            return recent_errors_count >= 5

    def get_quota_recommendations(self) -> str:
        """
        Get recommendations for increasing quota.

        Returns:
            Markdown-formatted recommendations
        """
        recommendations = f"""
## API Quota Recommendations for {self.quota_info.api_name}

### Current Status
- Total Requests: {self.quota_info.total_requests}
- Total Quota Errors: {self.quota_info.total_quota_errors}
- Daily Limit: {self.quota_info.requests_per_day or 'Unknown'}
- Daily Usage: {self.quota_info.current_day_count}

### Error Rate
- Recent Errors (last 100): {len(self.recent_errors)}
- Last Error: {self.quota_info.last_quota_error or 'None'}

### Recommendations

#### 1. Increase API Quota (Google Cloud Console)

**For Google Gemini API:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project
3. Navigate to: **APIs & Services** → **Gemini API**
4. Click **Quotas** tab
5. Request quota increase:
   - Current: {self.quota_info.requests_per_day or 'Free tier (1,500 RPD)'}
   - Recommended: 10,000+ requests per day

**Pricing Tiers:**
- **Free Tier**: 1,500 requests/day, 60 requests/minute
- **Paid Tier**: Up to 1,000,000+ requests/day
- **Enterprise**: Custom limits available

#### 2. Enable Billing

If not already enabled:
1. Go to **Billing** → **Link a billing account**
2. Add payment method
3. Quota limits increase automatically

#### 3. Optimization Strategies

While waiting for quota increase:
- **Use Circuit Breaker**: Already enabled (fails fast after 5 errors)
- **Batch Requests**: Combine multiple operations when possible
- **Cache Results**: Cache frequently used responses
- **Retry Strategy**: Use exponential backoff (already implemented)
- **Off-Peak Usage**: Schedule heavy operations during low-traffic hours

#### 4. Alternative Solutions

If quota cannot be increased:
- **OpenAI Fallback**: Already configured (automatic fallback)
- **Load Distribution**: Spread load across multiple API keys
- **Queue System**: Implement request queuing during peak times

### Contact Information

**Google Cloud Support:**
- Enterprise customers: [Contact Support](https://cloud.google.com/support)
- Community: [Google Cloud Community](https://www.googlecloudcommunity.com/)

**Estimated Time to Resolution:**
- Billing enablement: Immediate
- Quota increase request: 1-2 business days
- Enterprise custom quotas: 3-5 business days
"""
        return recommendations

    def get_stats(self) -> Dict[str, Any]:
        """Get quota tracker statistics"""
        with self._lock:
            return {
                "api_name": self.quota_info.api_name,
                "total_requests": self.quota_info.total_requests,
                "total_quota_errors": self.quota_info.total_quota_errors,
                "current_day_count": self.quota_info.current_day_count,
                "requests_per_day_limit": self.quota_info.requests_per_day,
                "is_exhausted": self.is_quota_exhausted(),
                "last_error": (
                    self.quota_info.last_quota_error.isoformat()
                    if self.quota_info.last_quota_error else None
                ),
                "recent_error_count": len(self.recent_errors)
            }


# =============================================================================
# Global Rate Limiters and Quota Trackers
# =============================================================================

_gemini_rate_limiter: Optional[RateLimiter] = None
_gemini_quota_tracker: Optional[APIQuotaTracker] = None

def get_gemini_rate_limiter() -> RateLimiter:
    """
    Get or create the global Gemini API rate limiter.

    Default limits (Free tier):
    - 60 requests per minute
    - 1,500 requests per day

    Returns:
        RateLimiter instance for Gemini API
    """
    global _gemini_rate_limiter
    if _gemini_rate_limiter is None:
        _gemini_rate_limiter = RateLimiter(
            requests_per_minute=60,
            requests_per_day=1500,
            name="gemini_api"
        )
    return _gemini_rate_limiter


def get_gemini_quota_tracker() -> APIQuotaTracker:
    """
    Get or create the global Gemini API quota tracker.

    Returns:
        APIQuotaTracker instance for Gemini API
    """
    global _gemini_quota_tracker
    if _gemini_quota_tracker is None:
        _gemini_quota_tracker = APIQuotaTracker(
            api_name="Google Gemini",
            requests_per_minute=60,
            requests_per_day=1500
        )
    return _gemini_quota_tracker


def get_all_rate_limiter_stats() -> Dict[str, Any]:
    """Get statistics from all rate limiters"""
    stats = {}

    if _gemini_rate_limiter:
        stats["gemini_rate_limiter"] = _gemini_rate_limiter.get_stats()

    if _gemini_quota_tracker:
        stats["gemini_quota_tracker"] = _gemini_quota_tracker.get_stats()

    return stats
