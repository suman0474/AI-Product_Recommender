# agentic/deep_agent/stateless_orchestrator.py
"""
Stateless Enrichment Orchestrator

Coordinates distributed enrichment sessions:
1. Initializes session state in Redis
2. Pushes initial tasks to queue
3. Monitors progress via state polling
4. Returns final results

No state is held in orchestrator - everything in Redis.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import uuid4

from ..memory.state_manager import EnrichmentStateManager
from .task_queue import EnrichmentTaskQueue

logger = logging.getLogger(__name__)


# =============================================================================
# STATELESS ORCHESTRATOR
# =============================================================================

class StatelessEnrichmentOrchestrator:
    """
    Stateless orchestrator for distributed enrichment sessions.

    Key features:
    - No in-memory state (all in Redis)
    - Can be restarted without data loss
    - Multiple orchestrators can run in parallel
    - Supports both sync and async patterns
    """

    def __init__(
        self,
        state_manager: EnrichmentStateManager,
        task_queue: EnrichmentTaskQueue
    ):
        """
        Initialize orchestrator.

        Args:
            state_manager: Persistent state manager (Redis/Cosmos)
            task_queue: Distributed task queue (Redis)
        """
        self.state_manager = state_manager
        self.task_queue = task_queue

    def start_enrichment_session(
        self,
        items: List[Dict[str, Any]],
        session_id: Optional[str] = None
    ) -> str:
        """
        Start a new enrichment session for a batch of items.

        This is NON-BLOCKING - returns immediately after queuing tasks.
        Workers process tasks asynchronously.

        Args:
            items: List of items to enrich. Each item should have:
                   - name: Item name/description
                   - category: Optional category
                   - sample_input: Optional additional context
            session_id: Optional session ID (auto-generated if None)

        Returns:
            session_id for tracking progress
        """
        session_id = session_id or f"session-{uuid4().hex}"

        logger.info(
            f"[ORCHESTRATOR] Starting session {session_id} with {len(items)} items..."
        )

        # Initialize session state in persistent store
        self.state_manager.create_session(session_id, {
            "total_items": len(items),
            "status": "in_progress",
            "started_at": datetime.utcnow().isoformat()
        })

        # Push initial tasks for each item
        for i, item in enumerate(items):
            item_id = f"item-{i}"

            # Determine initial domains (use planner logic)
            initial_domains = self._plan_initial_domains(item)

            # Initialize item state
            self.state_manager.update_item_state(session_id, item_id, {
                "status": "pending",
                "specifications": {},
                "spec_count": 0,
                "iteration": 0,
                "created_at": datetime.utcnow().isoformat()
            })

            # Push initial enrichment task
            self.task_queue.push_task({
                "task_id": f"{session_id}:{item_id}:init",
                "task_type": "initial_enrichment",
                "session_id": session_id,
                "item_id": item_id,
                "payload": {
                    "user_requirement": item.get("name", "Unknown"),
                    "initial_domains": initial_domains,
                    "item_data": item
                },
                "priority": 1,  # High priority for initial tasks
                "created_at": datetime.utcnow().isoformat(),
                "retry_count": 0
            })

        logger.info(
            f"[ORCHESTRATOR] Session {session_id} initialized, "
            f"{len(items)} tasks queued"
        )

        return session_id

    def wait_for_session_completion(
        self,
        session_id: str,
        timeout: int = 600,
        poll_interval: int = 5
    ) -> Dict[str, Any]:
        """
        Wait for session completion (BLOCKING).

        Use this for synchronous API requests where you need to return
        results immediately.

        Args:
            session_id: Session to wait for
            timeout: Maximum seconds to wait (default: 10 min)
            poll_interval: Seconds between status checks (default: 5s)

        Returns:
            Final session results with all enriched items
        """
        logger.info(
            f"[ORCHESTRATOR] Waiting for session {session_id} "
            f"(timeout: {timeout}s)..."
        )

        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if complete
            if self.state_manager.is_session_complete(session_id):
                logger.info(f"[ORCHESTRATOR] Session {session_id} completed")
                return self._get_session_results(session_id)

            # Log progress
            status = self.get_session_status(session_id)
            logger.debug(
                f"[ORCHESTRATOR] Session {session_id} progress: "
                f"{status['progress_percentage']:.1f}% "
                f"({status['completed_items']}/{status['total_items']})"
            )

            time.sleep(poll_interval)

        # Timeout
        elapsed = time.time() - start_time
        logger.warning(
            f"[ORCHESTRATOR] Session {session_id} timed out after {elapsed:.1f}s"
        )

        return self._get_session_results(session_id, timeout=True)

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get current status of a session (NON-BLOCKING).

        Use this for async/polling patterns where client checks status periodically.

        Args:
            session_id: Session to check

        Returns:
            Status dict with progress information
        """
        session_state = self.state_manager.get_session_state(session_id)

        if not session_state:
            return {
                "error": "Session not found",
                "session_id": session_id
            }

        # Get item states
        item_ids = self.state_manager.get_session_item_ids(session_id)
        item_states = [
            self.state_manager.get_item_state(session_id, item_id)
            for item_id in item_ids
        ]

        # Count completed
        completed = sum(
            1 for s in item_states
            if s and s.get("status") == "completed"
        )

        # Calculate progress
        total_items = len(item_ids)
        progress_pct = (completed / total_items * 100) if total_items > 0 else 0

        return {
            "session_id": session_id,
            "status": session_state.get("status"),
            "total_items": total_items,
            "completed_items": completed,
            "in_progress_items": total_items - completed,
            "progress_percentage": progress_pct,
            "started_at": session_state.get("started_at"),
            "is_complete": completed == total_items
        }

    def cancel_session(self, session_id: str) -> Dict[str, Any]:
        """
        Cancel a running session.

        Note: Tasks already being processed will complete.
        This prevents new iterations from being queued.

        Args:
            session_id: Session to cancel

        Returns:
            Cancellation result
        """
        session_state = self.state_manager.get_session_state(session_id)

        if not session_state:
            return {"error": "Session not found"}

        # Mark session as cancelled
        session_state["status"] = "cancelled"
        session_state["cancelled_at"] = datetime.utcnow().isoformat()

        self.state_manager.create_session(session_id, session_state)

        logger.info(f"[ORCHESTRATOR] Cancelled session {session_id}")

        return {
            "session_id": session_id,
            "status": "cancelled",
            "message": "Session cancelled. In-progress tasks will complete."
        }

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    def _get_session_results(
        self,
        session_id: str,
        timeout: bool = False
    ) -> Dict[str, Any]:
        """
        Compile final session results from persistent state.

        Args:
            session_id: Session identifier
            timeout: Whether session timed out

        Returns:
            Complete session results with all item data
        """
        session_state = self.state_manager.get_session_state(session_id)
        item_ids = self.state_manager.get_session_item_ids(session_id)

        if not session_state:
            return {
                "success": False,
                "error": "Session not found",
                "session_id": session_id
            }

        # Collect enriched items
        enriched_items = []
        for item_id in item_ids:
            item_state = self.state_manager.get_item_state(session_id, item_id)

            if item_state:
                enriched_items.append({
                    "item_id": item_id,
                    "specifications": item_state.get("specifications", {}),
                    "spec_count": item_state.get("spec_count", 0),
                    "status": item_state.get("status"),
                    "iterations": item_state.get("iteration", 0),
                    "domains_analyzed": item_state.get("domains_analyzed", [])
                })

        # Calculate statistics
        completed_count = sum(
            1 for item in enriched_items
            if item["status"] == "completed"
        )

        # Calculate total time
        started_at = session_state.get("started_at")
        if started_at:
            from dateutil import parser
            start_dt = parser.isoparse(started_at)
            elapsed_ms = int((datetime.utcnow() - start_dt).total_seconds() * 1000)
        else:
            elapsed_ms = 0

        return {
            "success": not timeout and completed_count == len(item_ids),
            "session_id": session_id,
            "items": enriched_items,
            "timeout": timeout,
            "batch_metadata": {
                "total_items": len(item_ids),
                "completed_items": completed_count,
                "processing_time_ms": elapsed_ms,
                "started_at": started_at,
                "completed_at": datetime.utcnow().isoformat() if not timeout else None
            }
        }

    def _plan_initial_domains(self, item: Dict[str, Any]) -> List[str]:
        """
        Plan which domains to analyze for an item.

        Uses simple heuristics based on category/name.

        Args:
            item: Item dict with name/category

        Returns:
            List of domain names to analyze
        """
        # Get category or name
        category = item.get("category", "").lower()
        name = item.get("name", "").lower()
        text = f"{category} {name}"

        # Domain mapping based on keywords
        if any(kw in text for kw in ["pressure", "transmitter", "pt", "gauge"]):
            return ["safety", "pressure", "calibration"]
        elif any(kw in text for kw in ["temperature", "thermocouple", "rtd", "tt"]):
            return ["safety", "temperature", "calibration"]
        elif any(kw in text for kw in ["flow", "flowmeter", "ft"]):
            return ["safety", "flow", "calibration"]
        elif any(kw in text for kw in ["level", "lt", "radar"]):
            return ["safety", "level", "calibration"]
        elif any(kw in text for kw in ["valve", "control valve"]):
            return ["safety", "valves", "control"]
        else:
            # Default domains
            return ["safety", "accessories", "calibration"]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_orchestrator(use_redis: bool = True) -> StatelessEnrichmentOrchestrator:
    """
    Factory function to create orchestrator with appropriate backends.

    Args:
        use_redis: Use Redis for state/queue (default True)

    Returns:
        StatelessEnrichmentOrchestrator instance
    """
    from ..memory.state_manager import get_state_manager
    from .task_queue import get_task_queue

    state_manager = get_state_manager(use_redis=use_redis)
    task_queue = get_task_queue(use_redis=use_redis)

    return StatelessEnrichmentOrchestrator(state_manager, task_queue)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "StatelessEnrichmentOrchestrator",
    "get_orchestrator"
]
