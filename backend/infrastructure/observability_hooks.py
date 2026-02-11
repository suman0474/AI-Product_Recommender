"""
Observability Hooks for Circuit Breakers

Provides callbacks and metrics collection for monitoring circuit breaker behavior.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class CircuitBreakerMetrics:
    """
    Collect and manage metrics for circuit breakers.
    
    Features:
    - Event logging (circuit open/close)
    - Aggregated statistics
    - Callbacks for external monitoring systems
    """
    
    def __init__(self):
        self._events: List[Dict[str, Any]] = []
        self._callbacks: List[Callable] = []
        self._stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "opened_count": 0,
            "closed_count": 0
        })
    
    def register_callback(self, callback: Callable[[str, str, Dict[str, Any]], None]):
        """
        Register a callback for circuit events.
        
        Args:
            callback: Function(event_type, circuit_name, stats) -> None
        """
        self._callbacks.append(callback)
        logger.info(f"Registered observability callback: {callback.__name__}")
    
    def on_circuit_open(self, circuit_name: str, stats: Dict[str, Any]):
        """
        Called when circuit opens.
        
        Args:
            circuit_name: Name of the circuit
            stats: Circuit statistics
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event": "circuit_opened",
            "circuit": circuit_name,
            "failure_count": stats.get("current_failures", 0),
            "total_failures": stats.get("total_failures", 0),
        }
        self._events.append(event)
        self._stats[circuit_name]["opened_count"] += 1
        
        logger.error(
            f"🚨 [OBSERVABILITY] Circuit OPEN: {circuit_name} | "
            f"Failures: {stats.get('current_failures')}/{stats.get('failure_threshold')}"
        )
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback("circuit_opened", circuit_name, stats)
            except Exception as e:
                logger.error(f"Callback failed: {e}")
    
    def on_circuit_close(self, circuit_name: str, stats: Dict[str, Any]):
        """
        Called when circuit closes (recovered).
        
        Args:
            circuit_name: Name of the circuit
            stats: Circuit statistics
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event": "circuit_closed",
            "circuit": circuit_name,
            "success_count": stats.get("current_successes", 0),
        }
        self._events.append(event)
        self._stats[circuit_name]["closed_count"] += 1
        
        logger.info(
            f"✅ [OBSERVABILITY] Circuit CLOSED: {circuit_name} | "
            f"Recovered successfully"
        )
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback("circuit_closed", circuit_name, stats)
            except Exception as e:
                logger.error(f"Callback failed: {e}")
    
    def get_events(self, circuit_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recorded events.
        
        Args:
            circuit_name: Filter by circuit name (None = all)
            limit: Maximum number of events to return
        
        Returns:
            List of event dictionaries
        """
        events = self._events
        
        if circuit_name:
            events = [e for e in events if e.get("circuit") == circuit_name]
        
        return events[-limit:]
    
    def get_aggregated_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics across all circuits."""
        return dict(self._stats)
    
    def clear_events(self):
        """Clear event history (useful for testing)."""
        self._events.clear()
        logger.info("Cleared observability events")


# ============================================================================
# GLOBAL SINGLETON
# ============================================================================

_metrics: Optional[CircuitBreakerMetrics] = None


def get_metrics() -> CircuitBreakerMetrics:
    """Get or create global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = CircuitBreakerMetrics()
    return _metrics


def setup_observability_hooks():
    """
    Setup observability hooks for circuit breakers.
    Call this during application initialization.
    """
    metrics = get_metrics()
    
    # Example: Setup Prometheus metrics (if available)
    try:
        from prometheus_client import Counter
        
        circuit_opened = Counter(
            'circuit_breaker_opened_total',
            'Total number of times circuits have opened',
            ['circuit_name']
        )
        
        circuit_closed = Counter(
            'circuit_breaker_closed_total',
            'Total number of times circuits have closed',
            ['circuit_name']
        )
        
        def prometheus_callback(event_type: str, circuit_name: str, stats: Dict[str, Any]):
            if event_type == "circuit_opened":
                circuit_opened.labels(circuit_name=circuit_name).inc()
            elif event_type == "circuit_closed":
                circuit_closed.labels(circuit_name=circuit_name).inc()
        
        metrics.register_callback(prometheus_callback)
        logger.info("✓ Prometheus metrics enabled for circuit breakers")
    
    except ImportError:
        logger.info("Prometheus not available, skipping metrics setup")
    
    logger.info("Observability hooks configured")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'CircuitBreakerMetrics',
    'get_metrics',
    'setup_observability_hooks',
]
