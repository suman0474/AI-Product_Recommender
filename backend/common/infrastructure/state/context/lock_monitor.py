"""
Lock Monitor
Provides visibility into thread lock contention for debugging and performance monitoring
"""
import threading
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MonitoredLock:
    """
    Thread lock with monitoring and contention tracking.

    wraps threading.Lock to provide visibility into lock wait times,
    acquisition counts, and contention patterns.

    Also includes Session-Level Locking (WorkflowLockManager) for coordinating
    access to shared resources.
    """

    def __init__(self, name: str, reentrant: bool = False):
        """
        Initialize monitored lock.

        Args:
            name: Lock name for identification in logs
            reentrant: If True, use threading.RLock instead of threading.Lock
        """
        self.name = name
        self._lock = threading.RLock() if reentrant else threading.Lock()
        self._stats = {
            "acquisitions": 0,
            "total_wait_time": 0.0,
            "max_wait_time": 0.0,
            "contentions": 0  # waits > 1ms
        }
        lock_type = "RLock" if reentrant else "Lock"
        logger.debug(f"[LOCK_MONITOR] Created {lock_type}: {name}")

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        """
        Acquire lock with monitoring.

        Args:
            blocking: Block until acquired
            timeout: Max time to wait (-1 for infinite)

        Returns:
            True if acquired, False otherwise
        """
        start_time = time.time()
        acquired = self._lock.acquire(blocking, timeout)

        if acquired:
            wait_time = time.time() - start_time

            # Track contention (waits > 1ms)
            if wait_time > 0.001:
                self._stats["contentions"] += 1
                logger.debug(
                    f"[LOCK_MONITOR] {self.name} waited {wait_time*1000:.1f}ms (contention)"
                )

            self._stats["acquisitions"] += 1
            self._stats["total_wait_time"] += wait_time
            self._stats["max_wait_time"] = max(self._stats["max_wait_time"], wait_time)

        return acquired

    def release(self):
        """Release lock."""
        self._lock.release()

    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get lock contention statistics.

        Returns:
            Dictionary with lock statistics
        """
        avg_wait_ms = 0.0
        contention_rate = 0.0

        if self._stats["acquisitions"] > 0:
            avg_wait_ms = (
                self._stats["total_wait_time"] / self._stats["acquisitions"] * 1000
            )
            contention_rate = (
                self._stats["contentions"] / self._stats["acquisitions"] * 100
            )

        return {
            "name": self.name,
            "acquisitions": self._stats["acquisitions"],
            "avg_wait_ms": f"{avg_wait_ms:.2f}",
            "max_wait_ms": f"{self._stats['max_wait_time']*1000:.2f}",
            "contentions": self._stats["contentions"],
            "contention_rate_percent": f"{contention_rate:.1f}"
        }

    def reset_stats(self) -> None:
        """Reset statistics (for testing)."""
        self._stats = {
            "acquisitions": 0,
            "total_wait_time": 0.0,
            "max_wait_time": 0.0,
            "contentions": 0
        }
        logger.debug(f"[LOCK_MONITOR] Reset stats for {self.name}")


class LockMonitoringManager:
    """Global lock monitoring for system visibility."""

    def __init__(self):
        """Initialize lock monitoring manager."""
        self._locks: Dict[str, MonitoredLock] = {}
        self._manager_lock = threading.Lock()

    def get_or_create_lock(self, name: str, reentrant: bool = False) -> MonitoredLock:
        """
        Get or create a monitored lock by name.

        Args:
            name: Lock name
            reentrant: If True, use RLock (only checks on creation)

        Returns:
            MonitoredLock instance
        """
        with self._manager_lock:
            if name not in self._locks:
                self._locks[name] = MonitoredLock(name, reentrant=reentrant)
                # logger.info(f"[LOCK_MONITORING] Created lock: {name}")
            return self._locks[name]

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all monitored locks.

        Returns:
            Dictionary of lock name -> stats
        """
        with self._manager_lock:
            return {
                name: lock.get_stats()
                for name, lock in self._locks.items()
            }

    def get_lock_stats(self, name: str) -> Dict[str, Any]:
        """
        Get statistics for specific lock.

        Args:
            name: Lock name

        Returns:
            Lock statistics or empty dict if not found
        """
        with self._manager_lock:
            if name in self._locks:
                return self._locks[name].get_stats()
            return {}


# Global monitoring manager
_monitoring_manager = None


def get_lock_monitoring_manager() -> LockMonitoringManager:
    """Get global lock monitoring manager."""
    global _monitoring_manager
    if _monitoring_manager is None:
        _monitoring_manager = LockMonitoringManager()
        logger.info("[LOCK_MONITORING] Global manager created")
    return _monitoring_manager


def get_monitored_lock(name: str, reentrant: bool = False) -> MonitoredLock:
    """
    Get or create a monitored lock.

    Args:
        name: Lock name
        reentrant: If True, use RLock

    Returns:
        MonitoredLock instance
    """
    manager = get_lock_monitoring_manager()
    return manager.get_or_create_lock(name, reentrant=reentrant)


# ============================================================================
# SESSION-LEVEL LOCKING
# ============================================================================

class WorkflowLockManager:
    """
    Manages locks per session_id to prevent concurrent workflow executions
    from interfering with each other.
    """

    def __init__(self):
        self._locks: Dict[str, threading.RLock] = {}
        self._lock_times: Dict[str, float] = {}
        self._manager_lock = threading.Lock()
        self._cleanup_interval = 3600  # 1 hour
        self._last_cleanup = time.time()

    def get_lock(self, session_id: str) -> threading.RLock:
        """
        Get or create a lock for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            RLock for the session
        """
        with self._manager_lock:
            # Periodic cleanup of old locks
            if time.time() - self._last_cleanup > self._cleanup_interval:
                self._cleanup_old_locks()

            if session_id not in self._locks:
                # Use monitored lock for visibility
                lock_name = f"session_workflow:{session_id}"
                self._locks[session_id] = get_monitored_lock(lock_name, reentrant=True)
                self._lock_times[session_id] = time.time()
                logger.debug(f"Created new monitored workflow lock: {lock_name}")

            return self._locks[session_id]

    def _cleanup_old_locks(self) -> None:
        """Remove locks that haven't been used in 24 hours"""
        cutoff = time.time() - (24 * 3600)
        expired = [
            sid for sid, lock_time in self._lock_times.items()
            if lock_time < cutoff
        ]

        for sid in expired:
            del self._locks[sid]
            del self._lock_times[sid]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired session locks")

        self._last_cleanup = time.time()

    def release_lock(self, session_id: str) -> None:
        """
        Explicitly release a session lock (optional, locks auto-release).

        Args:
            session_id: Session identifier
        """
        with self._manager_lock:
            if session_id in self._locks:
                del self._locks[session_id]
                del self._lock_times[session_id]
                logger.debug(f"Released lock for session {session_id}")


# Global lock manager instance
_workflow_lock_manager = WorkflowLockManager()


from contextlib import contextmanager

@contextmanager
def workflow_lock(session_id: str, timeout: float = 30.0):
    """
    Context manager for session-level workflow locking.

    Args:
        session_id: Session identifier
        timeout: Maximum time to wait for lock (seconds)

    Yields:
        None

    Raises:
        TimeoutError: If lock cannot be acquired within timeout

    Usage:
        with workflow_lock(session_id):
            # Execute workflow nodes
            ...
    """
    lock = _workflow_lock_manager.get_lock(session_id)

    acquired = lock.acquire(timeout=timeout)
    if not acquired:
        raise TimeoutError(
            f"Could not acquire workflow lock for session {session_id} "
            f"within {timeout} seconds. Another workflow may be running."
        )

    try:
        logger.debug(f"Acquired workflow lock for session {session_id}")
        yield
    finally:
        lock.release()
        logger.debug(f"Released workflow lock for session {session_id}")

import functools
from typing import Callable

def with_workflow_lock(session_id_param: str = "session_id", timeout: float = 30.0):
    """
    Decorator to add workflow-level locking to functions.

    Args:
        session_id_param: Name of parameter containing session_id
        timeout: Lock acquisition timeout

    Usage:
        @with_workflow_lock(session_id_param="session_id")
        def my_workflow(state: dict):
            session_id = state["session_id"]
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract session_id from args or kwargs
            session_id = None

            # Try to get from state dict (first arg for workflow nodes)
            if args and isinstance(args[0], dict):
                session_id = args[0].get(session_id_param)

            # Try kwargs
            if not session_id and session_id_param in kwargs:
                session_id = kwargs[session_id_param]

            if not session_id:
                logger.warning(
                    f"{func.__name__} called without session_id, "
                    "skipping workflow lock"
                )
                return func(*args, **kwargs)

            # Acquire lock and execute
            with workflow_lock(session_id, timeout=timeout):
                return func(*args, **kwargs)

        return wrapper
    return decorator


__all__ = [
    'MonitoredLock',
    'LockMonitoringManager',
    'get_lock_monitoring_manager',
    'get_monitored_lock',
    'WorkflowLockManager',
    'workflow_lock',
    'with_workflow_lock'
]
