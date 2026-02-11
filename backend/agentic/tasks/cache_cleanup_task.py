"""
Background task for periodic cache cleanup.
Prevents memory accumulation over long-running sessions.

This daemon thread periodically:
1. Cleans up expired entries in all bounded caches
2. Reports cache statistics
3. Prevents unbounded memory growth

Usage:
    from agentic.tasks.cache_cleanup_task import start_cache_cleanup, stop_cache_cleanup

    # Start at application startup
    start_cache_cleanup(interval_seconds=300)  # Run every 5 minutes

    # Stop at application shutdown (called automatically by atexit)
    stop_cache_cleanup()
"""

import threading
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CacheCleanupTask:
    """
    Background thread for periodic cache maintenance.

    Features:
    - Runs as a daemon thread (doesn't block shutdown)
    - Proactively removes expired entries
    - Reports cache statistics
    - Thread-safe start/stop
    """

    def __init__(self, interval_seconds: int = 300):
        """
        Initialize the cleanup task.

        Args:
            interval_seconds: Time between cleanup runs (default: 5 minutes)
        """
        self.interval = interval_seconds
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False

    def start(self):
        """Start the cleanup task in a background thread."""
        if self._is_running:
            logger.warning("[CACHE_CLEANUP] Task already running, ignoring start request")
            return

        if self._thread and self._thread.is_alive():
            logger.warning("[CACHE_CLEANUP] Thread already alive, ignoring start request")
            return

        self._stop_event.clear()
        self._is_running = True
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="CacheCleanupTask"
        )
        self._thread.start()
        logger.info(f"[CACHE_CLEANUP] ✓ Started (interval: {self.interval}s)")

    def stop(self):
        """Stop the cleanup task gracefully."""
        if not self._is_running:
            logger.debug("[CACHE_CLEANUP] Task not running")
            return

        logger.info("[CACHE_CLEANUP] Stopping...")
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            # Wait with timeout
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("[CACHE_CLEANUP] Thread did not stop within timeout")

        self._is_running = False
        logger.info("[CACHE_CLEANUP] ✓ Stopped")

    def _run(self):
        """Main cleanup loop."""
        from agentic.infrastructure.caching.bounded_cache import (
            cleanup_all_caches,
            get_all_cache_stats,
            get_registry_summary
        )

        logger.info(f"[CACHE_CLEANUP] Thread started, running cleanup every {self.interval}s")

        while not self._stop_event.is_set():
            try:
                # Run cleanup
                results = cleanup_all_caches()
                total_cleaned = sum(results.values())

                # Log results if there was cleanup work
                if total_cleaned > 0:
                    logger.info(f"[CACHE_CLEANUP] Cleaned {total_cleaned} expired entries")

                    # Log cache statistics
                    summary = get_registry_summary()
                    stats = get_all_cache_stats()

                    logger.debug(f"[CACHE_CLEANUP] Registry: {summary}")
                    for cache_name, cache_stats in stats.items():
                        hit_rate = cache_stats.get('hit_rate_percent', 0)
                        logger.debug(
                            f"[CACHE_CLEANUP] {cache_name}: "
                            f"size={cache_stats['size']}/{cache_stats['max_size']}, "
                            f"hits={cache_stats['hits']}, "
                            f"hit_rate={hit_rate}%"
                        )

            except ImportError:
                logger.debug("[CACHE_CLEANUP] BoundedCache module not available")
                break
            except Exception as e:
                logger.error(f"[CACHE_CLEANUP] Error during cleanup: {e}", exc_info=True)

            # Wait for next interval or stop signal
            if not self._stop_event.wait(self.interval):
                # Timeout - continue loop
                pass

        logger.info("[CACHE_CLEANUP] Thread exiting")


# Global singleton instance
_cleanup_task: Optional[CacheCleanupTask] = None
_task_lock = threading.Lock()


def start_cache_cleanup(interval_seconds: int = 300):
    """
    Start global cache cleanup task.

    Args:
        interval_seconds: Cleanup interval in seconds (default: 5 minutes)
    """
    global _cleanup_task

    with _task_lock:
        if _cleanup_task is None:
            _cleanup_task = CacheCleanupTask(interval_seconds)

        _cleanup_task.start()


def stop_cache_cleanup():
    """Stop global cache cleanup task."""
    global _cleanup_task

    with _task_lock:
        if _cleanup_task:
            _cleanup_task.stop()


def is_cleanup_running() -> bool:
    """Check if cleanup task is running."""
    with _task_lock:
        return _cleanup_task is not None and _cleanup_task._is_running


def get_cleanup_status() -> dict:
    """Get cleanup task status."""
    with _task_lock:
        if _cleanup_task is None:
            return {"status": "not_initialized"}

        return {
            "status": "running" if _cleanup_task._is_running else "stopped",
            "interval_seconds": _cleanup_task.interval,
            "thread_alive": _cleanup_task._thread.is_alive() if _cleanup_task._thread else False
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*60)
    print("CACHE CLEANUP TASK - TEST")
    print("="*60 + "\n")

    # Start cleanup
    print("[Test] Starting cleanup task with 2s interval...")
    start_cache_cleanup(interval_seconds=2)

    # Check status
    status = get_cleanup_status()
    print(f"[Test] Status: {status}")

    # Run for a few seconds
    print("[Test] Running for 6 seconds...")
    time.sleep(6)

    # Stop cleanup
    print("[Test] Stopping cleanup task...")
    stop_cache_cleanup()

    # Check final status
    status = get_cleanup_status()
    print(f"[Test] Final status: {status}")

    print("\n✓ Cleanup task test complete")
