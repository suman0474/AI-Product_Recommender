# solution/orchestration/solution_orchestrator.py
# =============================================================================
# SOLUTION ORCHESTRATOR — Identity-Aware Parallel Execution Engine
# =============================================================================
#
# Replaces bare ThreadPoolExecutor usage in workflow.py with a structured
# orchestrator that:
#   1. Mints a child OrchestrationContext per item.
#   2. Injects the context into the worker thread via contextvars.copy_context()
#      so it flows transparently into ALL nested calls.
#   3. Results are keyed by instance_id — preventing any positional mis-mapping.
#   4. Tracks all active tasks for observability / diagnostics.
#   5. Provides a module-level singleton via get_solution_orchestrator().
#
# =============================================================================

import contextvars
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from .orchestration_context import (
    OrchestrationContext,
    get_orchestration_logger,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# ActiveTask — lightweight tracking record for in-flight tasks
# ---------------------------------------------------------------------------

class ActiveTask:
    """Metadata about a running orchestration task."""

    __slots__ = ("instance_id", "parent_id", "label", "started_at", "future")

    def __init__(
        self,
        instance_id: str,
        parent_id: str,
        label: str,
        future: Future,
    ) -> None:
        self.instance_id = instance_id
        self.parent_id = parent_id
        self.label = label
        self.started_at = time.time()
        self.future = future

    def elapsed_ms(self) -> int:
        return int((time.time() - self.started_at) * 1000)


# ---------------------------------------------------------------------------
# SolutionOrchestrator
# ---------------------------------------------------------------------------

class SolutionOrchestrator:
    """
    Industry-grade parallel task executor for the Solution Deep Agent.

    Guarantees that each item runs in complete isolation through three layers:
      1. Unique `instance_id` per task        → structural namespace
      2. ContextVar injection per thread      → transparent propagation
      3. Keyed result map by instance_id      → no positional mis-mapping

    Usage:
        orchestrator = get_solution_orchestrator()

        root_ctx = OrchestrationContext.root(session_id="sess-123")

        results = orchestrator.run_parallel(
            fn=_enrich_item,           # fn(item: Dict) -> Dict
            items=instruments,
            root_ctx=root_ctx,
            label_fn=lambda i: f"enrich:{i.get('category', '?')}",
            timeout_seconds=120,
        )
        # results: Dict[instance_id, enriched_item_dict]

    Thread Safety:
        All public methods are thread-safe. The internal executor and task
        registry are guarded by a reentrant lock.
    """

    def __init__(self, max_workers: int = 8) -> None:
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="SolOrch",
        )
        self._active: Dict[str, ActiveTask] = {}
        self._lock = threading.RLock()
        logger.info(
            f"[SolutionOrchestrator] Initialized (max_workers={max_workers})"
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def run_parallel(
        self,
        fn: Callable[[Any], Any],
        items: List[Any],
        root_ctx: OrchestrationContext,
        label_fn: Optional[Callable[[Any], str]] = None,
        timeout_seconds: float = 120.0,
    ) -> Dict[str, Any]:
        """
        Run `fn(item)` for every `item` in parallel, each with its own
        child OrchestrationContext injected via contextvars.

        Args:
            fn              : Worker function. Signature: fn(item) -> Any.
                              Does NOT need to accept `ctx`; call
                              get_current_context() inside fn if needed.
            items           : Items to process.
            root_ctx        : Parent context. Each task inherits session_id
                              and zone from this, but gets its own instance_id.
            label_fn        : Optional function to derive a human-readable
                              thread_label from each item.
            timeout_seconds : Per-task timeout.

        Returns:
            Dict[instance_id, result_or_error_dict] — keyed by the child
            context's instance_id; preserves identity regardless of order.
        """
        if not items:
            return {}

        _label = label_fn or (lambda _: "task")
        futures: Dict[str, Future] = {}
        ctx_map: Dict[str, OrchestrationContext] = {}

        # --- Submit phase ---
        for item in items:
            child_ctx = root_ctx.child(label=_label(item))
            ctx_map[child_ctx.instance_id] = child_ctx

            fut = self._executor.submit(
                self._worker_wrapper,
                fn,
                item,
                child_ctx,
            )
            futures[child_ctx.instance_id] = fut

            with self._lock:
                self._active[child_ctx.instance_id] = ActiveTask(
                    instance_id=child_ctx.instance_id,
                    parent_id=child_ctx.parent_instance_id,
                    label=child_ctx.thread_label,
                    future=fut,
                )

        _log = get_orchestration_logger(logger)
        _log.info(
            f"[SolutionOrchestrator] Submitted {len(futures)} parallel tasks "
            f"under parent={root_ctx.instance_id}"
        )

        # --- Collect phase ---
        results: Dict[str, Any] = {}
        for iid, fut in futures.items():
            try:
                results[iid] = fut.result(timeout=timeout_seconds)
            except TimeoutError:
                results[iid] = {
                    "success": False,
                    "error": f"Task {iid} timed out after {timeout_seconds}s",
                    "instance_id": iid,
                }
                _log.error(f"[SolutionOrchestrator] Task {iid} timed out")
            except Exception as exc:
                results[iid] = {
                    "success": False,
                    "error": str(exc),
                    "instance_id": iid,
                }
                _log.error(
                    f"[SolutionOrchestrator] Task {iid} failed: {exc}",
                    exc_info=True,
                )
            finally:
                with self._lock:
                    task = self._active.pop(iid, None)
                    if task:
                        _log.debug(
                            f"[SolutionOrchestrator] Task {iid} "
                            f"({ctx_map[iid].thread_label}) "
                            f"completed in {task.elapsed_ms()}ms"
                        )

        return results

    def active_task_count(self) -> int:
        """Return the number of currently running tasks."""
        with self._lock:
            return len(self._active)

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully shut down the executor."""
        self._executor.shutdown(wait=wait)
        logger.info("[SolutionOrchestrator] Executor shutdown complete")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _worker_wrapper(
        fn: Callable[[Any], Any],
        item: Any,
        ctx: OrchestrationContext,
    ) -> Any:
        """
        Run in the ThreadPoolExecutor thread.

        1. Copies the current context (via contextvars.copy_context) so the
           ContextVar set here does not bleed into sibling threads.
        2. Sets the child OrchestrationContext into the thread-local slot.
        3. Calls fn(item) — fn can use get_current_context() transparently.
        """
        ctx_copy = contextvars.copy_context()

        def _run():
            ctx.set_current()
            return fn(item)

        return ctx_copy.run(_run)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_orchestrator_instance: Optional[SolutionOrchestrator] = None
_orchestrator_lock = threading.Lock()


def get_solution_orchestrator(max_workers: int = 8) -> SolutionOrchestrator:
    """
    Return the module-level SolutionOrchestrator singleton.

    The instance is created on first call and reused for the lifetime of the
    process. Thread-safe initialization via a double-checked lock.
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        with _orchestrator_lock:
            if _orchestrator_instance is None:
                _orchestrator_instance = SolutionOrchestrator(
                    max_workers=max_workers
                )
    return _orchestrator_instance
