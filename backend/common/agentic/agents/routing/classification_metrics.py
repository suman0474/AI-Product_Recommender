"""
Classification Metrics Tracking

Tracks intent classification accuracy, performance, and patterns to:
1. Identify misclassifications and improve prompts
2. Monitor classification layer performance (pattern vs semantic vs LLM)
3. Collect low-confidence samples for review
4. Export metrics for analysis

Usage:
    from classification_metrics import get_classification_metrics

    metrics = get_classification_metrics()
    metrics.record_classification(
        query="What is a pressure transmitter?",
        intent="chat",
        confidence=0.95,
        target_workflow="engenie_chat",
        classification_time_ms=250.5,
        layer_used="semantic"
    )

    # Get statistics
    stats = metrics.get_accuracy_by_layer()
    low_conf = metrics.get_low_confidence_samples(threshold=0.70)

    # Export for analysis
    metrics.export_to_json("metrics_report.json")
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from threading import Lock
from collections import defaultdict

logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """
    Track classification accuracy and performance metrics.

    Thread-safe singleton for collecting classification data.
    Automatically maintains last 1000 classifications in memory.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """Singleton pattern - only one instance exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize metrics tracking."""
        if self._initialized:
            return

        self._initialized = True
        self._classifications: List[Dict] = []
        self._data_lock = Lock()

        logger.info("[ClassificationMetrics] Initialized - Tracking enabled")

    def record_classification(
        self,
        query: str,
        intent: str,
        confidence: float,
        target_workflow: str,
        classification_time_ms: float,
        layer_used: str,  # "pattern", "semantic", "llm"
        is_solution: bool = False,
        extracted_info: Optional[Dict] = None,
        session_id: Optional[str] = None
    ):
        """
        Record a classification for analysis.

        Args:
            query: User query (truncated for privacy)
            intent: Classified intent
            confidence: Confidence score (0.0-1.0)
            target_workflow: Target workflow name
            classification_time_ms: Time taken to classify
            layer_used: Which layer classified (pattern, semantic, llm)
            is_solution: Whether classified as solution
            extracted_info: Additional metadata
            session_id: Session ID (truncated for privacy)
        """
        with self._data_lock:
            self._classifications.append({
                "timestamp": datetime.now().isoformat(),
                "query": query[:100],  # Truncate for privacy
                "intent": intent,
                "confidence": confidence,
                "target_workflow": target_workflow,
                "classification_time_ms": classification_time_ms,
                "layer_used": layer_used,
                "is_solution": is_solution,
                "extracted_info": extracted_info or {},
                "session_id": session_id[:8] if session_id else None  # Truncate for privacy
            })

            # Keep only last 1000 classifications in memory
            if len(self._classifications) > 1000:
                self._classifications = self._classifications[-1000:]

            logger.debug(
                f"[ClassificationMetrics] Recorded: intent={intent}, "
                f"workflow={target_workflow}, conf={confidence:.2f}, "
                f"layer={layer_used}, time={classification_time_ms:.1f}ms"
            )

    def get_accuracy_by_layer(self) -> Dict[str, Dict[str, float]]:
        """
        Get classification accuracy statistics by layer.

        Returns:
            {
                "pattern": {"count": 100, "avg_confidence": 0.95, "avg_time_ms": 2.5},
                "semantic": {"count": 200, "avg_confidence": 0.85, "avg_time_ms": 250.0},
                "llm": {"count": 50, "avg_confidence": 0.75, "avg_time_ms": 1500.0}
            }
        """
        with self._data_lock:
            if not self._classifications:
                return {}

            layer_stats = {}
            for layer in ["pattern", "semantic", "llm"]:
                layer_items = [c for c in self._classifications if c["layer_used"] == layer]

                if layer_items:
                    avg_confidence = sum(c["confidence"] for c in layer_items) / len(layer_items)
                    avg_time_ms = sum(c["classification_time_ms"] for c in layer_items) / len(layer_items)

                    layer_stats[layer] = {
                        "count": len(layer_items),
                        "avg_confidence": round(avg_confidence, 3),
                        "avg_time_ms": round(avg_time_ms, 2),
                        "percentage": round((len(layer_items) / len(self._classifications)) * 100, 1)
                    }

            return layer_stats

    def get_accuracy_by_workflow(self) -> Dict[str, Dict[str, Any]]:
        """
        Get classification statistics by workflow.

        Returns:
            {
                "engenie_chat": {"count": 300, "avg_confidence": 0.88, "avg_time_ms": 200},
                "instrument_identifier": {"count": 150, "avg_confidence": 0.82, ...},
                ...
            }
        """
        with self._data_lock:
            if not self._classifications:
                return {}

            workflow_stats = defaultdict(lambda: {"items": [], "count": 0})

            for c in self._classifications:
                workflow = c["target_workflow"]
                workflow_stats[workflow]["items"].append(c)
                workflow_stats[workflow]["count"] += 1

            result = {}
            for workflow, data in workflow_stats.items():
                items = data["items"]
                result[workflow] = {
                    "count": len(items),
                    "avg_confidence": round(sum(c["confidence"] for c in items) / len(items), 3),
                    "avg_time_ms": round(sum(c["classification_time_ms"] for c in items) / len(items), 2),
                    "percentage": round((len(items) / len(self._classifications)) * 100, 1)
                }

            return result

    def get_low_confidence_samples(self, threshold: float = 0.70) -> List[Dict]:
        """
        Get samples with low confidence for review.

        Args:
            threshold: Confidence threshold (default 0.70)

        Returns:
            List of low-confidence classifications with details
        """
        with self._data_lock:
            return [
                {
                    "query": c["query"],
                    "intent": c["intent"],
                    "confidence": c["confidence"],
                    "target_workflow": c["target_workflow"],
                    "layer_used": c["layer_used"],
                    "timestamp": c["timestamp"],
                    "reasoning": c["extracted_info"].get("reasoning", "N/A")
                }
                for c in self._classifications
                if c["confidence"] < threshold
            ]

    def get_misclassification_candidates(self) -> List[Dict]:
        """
        Get potential misclassifications based on heuristics.

        Heuristics:
        1. Knowledge questions ("what is", "how does") routed to SEARCH/SOLUTION
        2. Purchase intent ("i need", "looking for") routed to CHAT
        3. Design intent ("design", "implement") routed to CHAT/SEARCH

        Returns:
            List of potential misclassifications
        """
        with self._data_lock:
            candidates = []

            for c in self._classifications:
                query_lower = c["query"].lower()
                workflow = c["target_workflow"]
                intent = c["intent"]

                # Heuristic 1: Knowledge questions should go to CHAT
                knowledge_patterns = ["what is", "how does", "explain", "tell me about"]
                if any(p in query_lower for p in knowledge_patterns):
                    if workflow in ["instrument_identifier", "solution"]:
                        candidates.append({
                            **c,
                            "issue": "Knowledge question routed to non-CHAT workflow",
                            "expected_workflow": "engenie_chat"
                        })

                # Heuristic 2: Purchase intent should go to SEARCH
                purchase_patterns = ["i need", "looking for", "find me", "recommend"]
                if any(p in query_lower for p in purchase_patterns):
                    if workflow == "engenie_chat" and intent != "greeting":
                        candidates.append({
                            **c,
                            "issue": "Purchase intent routed to CHAT",
                            "expected_workflow": "instrument_identifier"
                        })

                # Heuristic 3: Design intent should go to SOLUTION
                design_patterns = ["design", "implement", "build", "complete system"]
                if any(p in query_lower for p in design_patterns):
                    if workflow in ["engenie_chat", "instrument_identifier"]:
                        candidates.append({
                            **c,
                            "issue": "Design intent routed to non-SOLUTION workflow",
                            "expected_workflow": "solution"
                        })

            return candidates

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics.

        Returns:
            {
                "total_classifications": 1000,
                "date_range": {"start": "2026-02-12T10:00:00", "end": "2026-02-12T18:00:00"},
                "by_layer": {...},
                "by_workflow": {...},
                "avg_confidence": 0.85,
                "avg_time_ms": 350.5,
                "low_confidence_count": 50
            }
        """
        with self._data_lock:
            if not self._classifications:
                return {
                    "total_classifications": 0,
                    "message": "No data collected yet"
                }

            timestamps = [c["timestamp"] for c in self._classifications]
            confidences = [c["confidence"] for c in self._classifications]
            times = [c["classification_time_ms"] for c in self._classifications]

            return {
                "total_classifications": len(self._classifications),
                "date_range": {
                    "start": min(timestamps),
                    "end": max(timestamps)
                },
                "by_layer": self.get_accuracy_by_layer(),
                "by_workflow": self.get_accuracy_by_workflow(),
                "avg_confidence": round(sum(confidences) / len(confidences), 3),
                "avg_time_ms": round(sum(times) / len(times), 2),
                "low_confidence_count": len(self.get_low_confidence_samples()),
                "misclassification_candidates": len(self.get_misclassification_candidates())
            }

    def export_to_json(self, filepath: str):
        """
        Export metrics to JSON file for analysis.

        Args:
            filepath: Output file path
        """
        with self._data_lock:
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "summary": self.get_summary_statistics(),
                "low_confidence_samples": self.get_low_confidence_samples(),
                "misclassification_candidates": self.get_misclassification_candidates(),
                "all_classifications": self._classifications
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"[ClassificationMetrics] Exported metrics to: {filepath}")

    def clear(self):
        """Clear all collected metrics."""
        with self._data_lock:
            self._classifications.clear()
            logger.info("[ClassificationMetrics] Metrics cleared")

    def get_recent_classifications(self, count: int = 10) -> List[Dict]:
        """Get most recent N classifications."""
        with self._data_lock:
            return self._classifications[-count:]


# Singleton instance
_classification_metrics = ClassificationMetrics()

def get_classification_metrics() -> ClassificationMetrics:
    """Get the global classification metrics instance."""
    return _classification_metrics
