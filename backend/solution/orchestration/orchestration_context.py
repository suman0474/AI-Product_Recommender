# solution/orchestration/orchestration_context.py
# =============================================================================
# ORCHESTRATION CONTEXT — Identity-Aware Execution Namespace
# =============================================================================
#
# Provides transparent propagation of (session_id, instance_id, thread_id)
# across thread boundaries using Python's contextvars.
#
# Key Design Decisions:
#   - frozen=True  → Identity cannot be mutated after creation
#   - ContextVar   → Propagates "invisibly" into worker threads via
#                    contextvars.copy_context().run(...)
#   - LoggerAdapter→ Every log line is automatically tagged with
#                    [session_id:instance_id] — zero boilerplate
#
# Usage:
#   root_ctx = OrchestrationContext.root(session_id)
#   child_ctx = root_ctx.child("enrich:pressure_transmitter")
#   # child_ctx.set_current() is called inside the thread by the orchestrator
#
#   # Any function — even nested helpers — can do:
#   ctx = get_current_context()
#   logger = get_orchestration_logger(logging.getLogger(__name__))
#   logger.info("Processing item")  # → [sess123:inst-abc] Processing item
#
# =============================================================================

import logging
import time
import uuid
import contextvars
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# ---------------------------------------------------------------------------
# Module-level ContextVar — one per interpreter, shared across all threads.
# Each thread sets its own copy via ContextVar.set(), so reads are isolated.
# ---------------------------------------------------------------------------
_ORCH_CTX: contextvars.ContextVar["OrchestrationContext"] = contextvars.ContextVar(
    "orchestration_context"
)


# ---------------------------------------------------------------------------
# OrchestrationContext
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrchestrationContext:
    """
    Immutable identity namespace for one unit of concurrent work.

    Hierarchy:
        ROOT (one per API request / run_solution_deep_agent call)
         └── CHILD (one per item processed in the ThreadPoolExecutor)
              └── GRANDCHILD (optional — for sub-tasks inside a node)

    Fields:
        session_id          : Flask/UI session — the user's conversation.
        instance_id         : UUID4 — unique per CHILD/GRANDCHILD task.
        parent_instance_id  : Points to the parent context's instance_id.
        thread_label        : Human-readable tag e.g. "enrich:pressure_xmtr".
        zone                : Geographic zone ("US-WEST", "EU", …).
        started_at          : Epoch float — used for TTL / latency tracking.
    """

    session_id: str
    instance_id: str
    parent_instance_id: str
    thread_label: str = "root"
    zone: str = "DEFAULT"
    started_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------------ #
    # Factory helpers                                                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def root(
        cls,
        session_id: str,
        zone: str = "DEFAULT",
    ) -> "OrchestrationContext":
        """Create the top-level context for a new workflow invocation."""
        root_id = f"root-{uuid.uuid4().hex[:8]}"
        return cls(
            session_id=session_id,
            instance_id=root_id,
            parent_instance_id="<none>",
            thread_label="root",
            zone=zone,
        )

    def child(self, label: str = "") -> "OrchestrationContext":
        """
        Mint a new child context inheriting session_id and zone from parent.

        Args:
            label: Human-readable tag for this task
                   e.g. "enrich:pressure_transmitter"

        Returns:
            A new, frozen OrchestrationContext with a fresh instance_id.
        """
        child_id = f"child-{uuid.uuid4().hex[:12]}"
        return OrchestrationContext(
            session_id=self.session_id,
            instance_id=child_id,
            parent_instance_id=self.instance_id,
            thread_label=label or f"child-of-{self.thread_label}",
            zone=self.zone,
        )

    # ------------------------------------------------------------------ #
    # ContextVar integration                                               #
    # ------------------------------------------------------------------ #

    def set_current(self) -> contextvars.Token:
        """
        Bind this context to the current thread's ContextVar slot.

        Returns the Token so callers can restore the previous value if needed.
        This MUST be called at the top of every worker thread target function
        (handled automatically by SolutionOrchestrator._worker_wrapper).
        """
        return _ORCH_CTX.set(self)

    def as_langgraph_config(self) -> Dict[str, Any]:
        """
        Return a LangGraph-compatible configurable block.

        Usage:
            compiled.invoke(initial_state, config=ctx.as_langgraph_config())
        """
        return {
            "configurable": {
                "thread_id": self.instance_id,
                "session_id": self.session_id,
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for state storage."""
        return {
            "session_id": self.session_id,
            "instance_id": self.instance_id,
            "parent_instance_id": self.parent_instance_id,
            "thread_label": self.thread_label,
            "zone": self.zone,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrchestrationContext":
        """Restore from a serialised dict (e.g., from LangGraph state)."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Module-level accessors
# ---------------------------------------------------------------------------

def get_current_context() -> Optional[OrchestrationContext]:
    """
    Return the OrchestrationContext bound to the current thread, or None.

    Any function that needs the current identity (e.g., to log it or pass it
    to a downstream call) should use this instead of receiving ctx as a param.
    """
    return _ORCH_CTX.get(None)


def get_orchestration_logger(base_logger: logging.Logger) -> logging.Logger:
    """
    Wrap a standard logger so every message is automatically tagged with
    [session_id:instance_id:thread_label].

    Usage (inside any enrichment helper):
        logger = get_orchestration_logger(logging.getLogger(__name__))
        logger.info("processing item")
        # Output → [sess123:child-ab12cd34ef56:enrich:pt] processing item

    Falls back to the base logger if no context is set.
    """
    ctx = get_current_context()
    if ctx is None:
        return base_logger

    tag = f"[{ctx.session_id}:{ctx.instance_id}:{ctx.thread_label}]"
    return logging.LoggerAdapter(base_logger, {"extra_id": tag})
