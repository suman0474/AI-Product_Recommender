"""
Thread-safe result collection and batch processing for parallel execution.
"""

from typing import Any, Dict, List, Optional
import threading
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ThreadSafeResultCollector:
    """
    Thread-safe collector for parallel execution results.
    """

    def __init__(self):
        self.results: list = []
        self.errors: list = []
        self._lock = threading.Lock()

    def add_result(self, result: Any) -> None:
        """Add a successful result"""
        with self._lock:
            self.results.append(result)

    def add_error(self, error: Exception, context: Optional[Dict] = None) -> None:
        """Add an error"""
        with self._lock:
            self.errors.append({
                "error": str(error),
                "type": type(error).__name__,
                "context": context or {},
                "timestamp": datetime.now().isoformat()
            })

    def get_results(self) -> list:
        """Get all results (thread-safe)"""
        with self._lock:
            return self.results.copy()

    def get_errors(self) -> list:
        """Get all errors (thread-safe)"""
        with self._lock:
            return self.errors.copy()

    def has_errors(self) -> bool:
        """Check if any errors occurred"""
        with self._lock:
            return len(self.errors) > 0

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        with self._lock:
            return {
                "total_results": len(self.results),
                "total_errors": len(self.errors),
                "success_rate": len(self.results) / (len(self.results) + len(self.errors))
                    if (len(self.results) + len(self.errors)) > 0 else 0.0
            }

__all__ = ['ThreadSafeResultCollector']
