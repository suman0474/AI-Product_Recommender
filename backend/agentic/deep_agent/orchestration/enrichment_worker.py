# agentic/deep_agent/enrichment_worker.py
"""
Enrichment Worker - Processes tasks from distributed queue

Stateless workers that:
1. Pull tasks from queue
2. Process enrichment (LLM/Standards)
3. Update persistent state
4. Push new tasks if needed (iterations)
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
from uuid import uuid4

from ..memory.state_manager import EnrichmentStateManager
from .task_queue import EnrichmentTaskQueue, EnrichmentTask

logger = logging.getLogger(__name__)


# =============================================================================
# ENRICHMENT WORKER
# =============================================================================

class EnrichmentWorker:
    """
    Stateless worker that processes enrichment tasks from queue.

    Workers can be run in parallel across multiple machines.
    All state is stored in Redis/Cosmos, not in worker memory.
    """

    def __init__(
        self,
        queue: EnrichmentTaskQueue,
        state_manager: EnrichmentStateManager,
        worker_id: Optional[str] = None
    ):
        """
        Initialize enrichment worker.

        Args:
            queue: Task queue to pull from
            state_manager: State manager for persistent storage
            worker_id: Unique worker identifier (auto-generated if None)
        """
        self.queue = queue
        self.state_manager = state_manager
        self.worker_id = worker_id or f"worker-{uuid4().hex[:8]}"
        self.running = False
        self.tasks_processed = 0
        self.tasks_failed = 0

    def start(self, max_tasks: Optional[int] = None):
        """
        Start worker loop.

        Args:
            max_tasks: Maximum tasks to process before stopping (None = infinite)
        """
        self.running = True
        logger.info(f"[WORKER:{self.worker_id}] Starting worker loop...")

        while self.running:
            # Check if reached max tasks
            if max_tasks and self.tasks_processed >= max_tasks:
                logger.info(
                    f"[WORKER:{self.worker_id}] Reached max tasks ({max_tasks}), stopping..."
                )
                break

            try:
                # Get next task from queue (blocking with timeout)
                task = self.queue.pop_task(timeout=30)

                if not task:
                    # Timeout - no tasks available
                    continue

                # Process task based on type
                try:
                    logger.info(
                        f"[WORKER:{self.worker_id}] Processing task {task['task_id']} "
                        f"(type: {task['task_type']})"
                    )

                    if task["task_type"] == "initial_enrichment":
                        self._process_initial_enrichment(task)
                    elif task["task_type"] == "iterative_enrichment":
                        self._process_iterative_enrichment(task)
                    elif task["task_type"] == "domain_extraction":
                        self._process_domain_extraction(task)
                    else:
                        logger.error(
                            f"[WORKER:{self.worker_id}] Unknown task type: {task['task_type']}"
                        )
                        self.queue.nack_task(task["task_id"], retry=False)
                        self.tasks_failed += 1
                        continue

                    # ACK on success
                    self.queue.ack_task(task["task_id"])
                    self.tasks_processed += 1

                    logger.info(
                        f"[WORKER:{self.worker_id}] Task {task['task_id']} completed "
                        f"(total: {self.tasks_processed})"
                    )

                except Exception as e:
                    logger.error(
                        f"[WORKER:{self.worker_id}] Task processing failed: {e}",
                        exc_info=True
                    )
                    self.queue.nack_task(task["task_id"], retry=True)
                    self.tasks_failed += 1

            except KeyboardInterrupt:
                logger.info(f"[WORKER:{self.worker_id}] Received shutdown signal...")
                self.running = False
            except Exception as e:
                logger.error(f"[WORKER:{self.worker_id}] Worker error: {e}", exc_info=True)
                time.sleep(5)  # Backoff on error

        logger.info(
            f"[WORKER:{self.worker_id}] Stopped. "
            f"Processed: {self.tasks_processed}, Failed: {self.tasks_failed}"
        )

    def stop(self):
        """Stop worker gracefully."""
        logger.info(f"[WORKER:{self.worker_id}] Stopping worker...")
        self.running = False

    # =========================================================================
    # TASK PROCESSORS
    # =========================================================================

    def _process_initial_enrichment(self, task: EnrichmentTask):
        """
        Process initial enrichment for an item.

        This is the first enrichment pass - extracts specs from initial domains.
        """
        session_id = task["session_id"]
        item_id = task["item_id"]
        payload = task["payload"]

        logger.info(f"[WORKER:{self.worker_id}] Initial enrichment: {item_id}")

        # Get or create item state
        item_state = self.state_manager.get_item_state(session_id, item_id) or {}

        # Extract initial specs
        user_requirement = payload["user_requirement"]
        initial_domains = payload.get("initial_domains", ["safety", "accessories"])

        try:
            # Call existing extraction function
            from ..standards_deep_agent import _extract_specs_from_domains_parallel

            results = _extract_specs_from_domains_parallel(
                user_requirement=user_requirement,
                domains=initial_domains,
                existing_specs={},
                specs_needed=30
            )

            # Aggregate specs
            all_specs = {}
            for domain, result in results.items():
                specs = result.get("specifications", {})
                all_specs.update(specs)

            # Count valid specs
            spec_count = self._count_valid_specs(all_specs)

            # Update state
            item_state.update({
                "specifications": all_specs,
                "spec_count": spec_count,
                "domains_analyzed": initial_domains,
                "iteration": 1,
                "status": "in_progress",
                "updated_at": datetime.utcnow().isoformat()
            })

            self.state_manager.update_item_state(session_id, item_id, item_state)

            # Check if minimum reached
            if spec_count >= 30:
                # Complete!
                self.state_manager.mark_item_complete(session_id, item_id, all_specs)
                logger.info(
                    f"[WORKER:{self.worker_id}] Item {item_id} completed "
                    f"with {spec_count} specs"
                )
            else:
                # Need more iterations - push iterative task
                self.queue.push_task({
                    "task_id": f"{session_id}:{item_id}:iter2",
                    "task_type": "iterative_enrichment",
                    "session_id": session_id,
                    "item_id": item_id,
                    "payload": {
                        "user_requirement": user_requirement,
                        "current_specs": all_specs,
                        "current_domains": initial_domains,
                        "specs_needed": 30 - spec_count,
                        "iteration": 2
                    },
                    "priority": 5,
                    "created_at": datetime.utcnow().isoformat(),
                    "retry_count": 0
                })

                logger.info(
                    f"[WORKER:{self.worker_id}] Item {item_id} needs more specs "
                    f"({spec_count}/30), queued iteration 2"
                )

        except Exception as e:
            logger.error(
                f"[WORKER:{self.worker_id}] Initial enrichment failed for {item_id}: {e}",
                exc_info=True
            )
            raise

    def _process_iterative_enrichment(self, task: EnrichmentTask):
        """
        Process iterative enrichment for an item.

        This continues enrichment from where it left off, trying additional domains.
        """
        session_id = task["session_id"]
        item_id = task["item_id"]
        payload = task["payload"]

        iteration = payload["iteration"]
        logger.info(
            f"[WORKER:{self.worker_id}] Iterative enrichment: {item_id} "
            f"(iteration {iteration})"
        )

        # Get current state
        item_state = self.state_manager.get_item_state(session_id, item_id)

        if not item_state:
            logger.error(f"[WORKER:{self.worker_id}] No state found for {item_id}")
            return

        current_specs = payload["current_specs"]
        current_domains = payload["current_domains"]
        specs_needed = payload["specs_needed"]

        # Find remaining domains
        from ..standards_deep_agent import FALLBACK_DOMAINS_ORDER, MAX_STANDARDS_ITERATIONS

        remaining_domains = [d for d in FALLBACK_DOMAINS_ORDER if d not in current_domains]

        if not remaining_domains or iteration > MAX_STANDARDS_ITERATIONS:
            # No more domains or max iterations reached
            self.state_manager.mark_item_complete(session_id, item_id, current_specs)

            logger.warning(
                f"[WORKER:{self.worker_id}] Item {item_id} completed (partial) "
                f"with {self._count_valid_specs(current_specs)} specs "
                f"(iteration {iteration})"
            )
            return

        # Try next domain(s)
        domains_to_try = remaining_domains[:3]

        try:
            from ..standards_deep_agent import _extract_specs_from_domains_parallel

            results = _extract_specs_from_domains_parallel(
                user_requirement=payload["user_requirement"],
                domains=domains_to_try,
                existing_specs=current_specs,
                specs_needed=specs_needed
            )

            # Merge new specs
            added_count = 0
            for domain, result in results.items():
                specs = result.get("specifications", {})

                for key, value in specs.items():
                    if key not in current_specs and value and str(value).lower() not in ["null", "none"]:
                        current_specs[key] = value
                        added_count += 1

            current_domains.extend(domains_to_try)

            # Update state
            spec_count = self._count_valid_specs(current_specs)
            item_state.update({
                "specifications": current_specs,
                "spec_count": spec_count,
                "domains_analyzed": current_domains,
                "iteration": iteration,
                "updated_at": datetime.utcnow().isoformat()
            })

            self.state_manager.update_item_state(session_id, item_id, item_state)

            # Check completion
            if spec_count >= 30:
                self.state_manager.mark_item_complete(session_id, item_id, current_specs)
                logger.info(
                    f"[WORKER:{self.worker_id}] Item {item_id} completed "
                    f"with {spec_count} specs after {iteration} iterations"
                )
            else:
                # Queue next iteration
                self.queue.push_task({
                    "task_id": f"{session_id}:{item_id}:iter{iteration+1}",
                    "task_type": "iterative_enrichment",
                    "session_id": session_id,
                    "item_id": item_id,
                    "payload": {
                        "user_requirement": payload["user_requirement"],
                        "current_specs": current_specs,
                        "current_domains": current_domains,
                        "specs_needed": 30 - spec_count,
                        "iteration": iteration + 1
                    },
                    "priority": 5,
                    "created_at": datetime.utcnow().isoformat(),
                    "retry_count": 0
                })

                logger.info(
                    f"[WORKER:{self.worker_id}] Item {item_id} iteration {iteration} "
                    f"added {added_count} specs (total: {spec_count}/30), "
                    f"queued iteration {iteration+1}"
                )

        except Exception as e:
            logger.error(
                f"[WORKER:{self.worker_id}] Iterative enrichment failed for {item_id}: {e}",
                exc_info=True
            )
            raise

    def _process_domain_extraction(self, task: EnrichmentTask):
        """
        Process domain-specific extraction task.

        This extracts specs from a single domain for a single item.
        Used for fine-grained parallelization.
        """
        session_id = task["session_id"]
        item_id = task["item_id"]
        payload = task["payload"]

        domain = payload["domain"]
        logger.info(
            f"[WORKER:{self.worker_id}] Domain extraction: {item_id} - {domain}"
        )

        # Implementation for domain-specific extraction
        # This is a future optimization - not critical for MVP
        logger.warning("[WORKER:{self.worker_id}] Domain extraction not yet implemented")

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _count_valid_specs(self, specs: Dict[str, Any]) -> int:
        """Count valid (non-null, non-empty) specifications."""
        if not specs:
            return 0

        count = 0
        for value in specs.values():
            if value and str(value).lower() not in ["null", "none", "n/a", "", "not specified"]:
                count += 1

        return count


# =============================================================================
# WORKER RUNNER (CLI Entry Point)
# =============================================================================

def run_worker(
    worker_id: Optional[str] = None,
    use_redis: bool = True,
    max_tasks: Optional[int] = None
):
    """
    Run a single enrichment worker.

    This is the CLI entry point for starting workers.

    Args:
        worker_id: Worker identifier (auto-generated if None)
        use_redis: Use Redis for queue/state (default True)
        max_tasks: Maximum tasks to process (None = infinite)
    """
    from ..memory.state_manager import get_state_manager
    from .task_queue import get_task_queue

    logger.info("="*70)
    logger.info("ENRICHMENT WORKER STARTING")
    logger.info(f"Worker ID: {worker_id or 'auto'}")
    logger.info(f"Using Redis: {use_redis}")
    logger.info(f"Max Tasks: {max_tasks or 'unlimited'}")
    logger.info("="*70)

    # Initialize components
    state_manager = get_state_manager(use_redis=use_redis)
    task_queue = get_task_queue(use_redis=use_redis)

    # Create and start worker
    worker = EnrichmentWorker(
        queue=task_queue,
        state_manager=state_manager,
        worker_id=worker_id
    )

    try:
        worker.start(max_tasks=max_tasks)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        worker.stop()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "EnrichmentWorker",
    "run_worker"
]
