"""
Execution Context - Unified context for session/workflow/task isolation.

This module provides the ExecutionContext dataclass that flows through all
API → Workflow → Tool calls to ensure proper state isolation between
concurrent requests.

Hierarchy:
    Session (user-scoped)
        └── Workflow (execution-scoped)
                └── Task (operation-scoped)

Usage:
    from common.infrastructure.context import ExecutionContext

    ctx = ExecutionContext(
        session_id="main_user123_US_WEST_abc123_20260222",
        workflow_type="product_search"
    )

    # Pass to workflows and tools
    result = run_workflow(user_input=..., ctx=ctx)
"""

import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """
    Unified execution context for session/workflow/task isolation.

    This object MUST flow through all API → Workflow → Tool calls.
    It provides:
    - Unique identification at each level (session, workflow, task)
    - Proper cache key generation for isolated storage
    - Correlation ID for distributed tracing
    - Parent-child relationships for nested workflows

    Attributes:
        session_id: User session identifier (main_thread_id format)
        user_id: User identifier
        zone: Geographic zone (US_WEST, EU_CENTRAL, etc.)
        workflow_id: Workflow execution identifier
        workflow_type: Type of workflow (solution, product_search, etc.)
        parent_workflow_id: Parent workflow for nested workflows
        instance_id: Unique instance ID for deduplication
        task_id: Current task within workflow
        task_type: Type of current task
        created_at: Context creation timestamp
        correlation_id: For distributed tracing across services
    """

    # ═══════════════════════════════════════════════════════════════════
    # SESSION LAYER - User-scoped, persists across requests
    # ═══════════════════════════════════════════════════════════════════
    session_id: str           # main_thread_id format (required)
    user_id: str = "anonymous"
    zone: str = "DEFAULT"

    # ═══════════════════════════════════════════════════════════════════
    # WORKFLOW LAYER - Execution-scoped, one per workflow invocation
    # ═══════════════════════════════════════════════════════════════════
    workflow_id: str = ""     # workflow_thread_id format
    workflow_type: str = ""   # solution, product_search, etc.
    parent_workflow_id: Optional[str] = None  # For nested workflows
    instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # ═══════════════════════════════════════════════════════════════════
    # TASK LAYER - Operation-scoped, one per tool/node invocation
    # ═══════════════════════════════════════════════════════════════════
    task_id: Optional[str] = None
    task_type: Optional[str] = None

    # ═══════════════════════════════════════════════════════════════════
    # METADATA - Observability & tracing
    # ═══════════════════════════════════════════════════════════════════
    created_at: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self):
        """Validate required fields and generate IDs if missing."""
        if not self.session_id:
            raise ValueError("session_id is required for ExecutionContext")

        if not self.workflow_id and self.workflow_type:
            # Auto-generate workflow_id from session_id
            self.workflow_id = self._generate_workflow_id()

    def _generate_workflow_id(self) -> str:
        """Generate a workflow ID in the standard format."""
        # Try to use HierarchicalThreadManager if available
        try:
            from common.infrastructure.state.execution.thread_manager import (
                HierarchicalThreadManager, WorkflowThreadType
            )
            # Convert string workflow_type to enum if possible
            workflow_type_str = self.workflow_type or "unknown"
            try:
                workflow_type_enum = WorkflowThreadType(workflow_type_str)
            except ValueError:
                # Unknown workflow type - use fallback format
                raise ImportError("Unknown workflow type")

            return HierarchicalThreadManager.generate_workflow_thread_id(
                workflow_type=workflow_type_enum,
                main_thread_id=self.session_id
            )
        except (ImportError, ValueError):
            # Fallback format for unknown workflow types or missing imports
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:20]
            uid = uuid.uuid4().hex[:8]
            session_ref = self.session_id[-8:] if len(self.session_id) > 8 else self.session_id
            return f"{self.workflow_type}_{session_ref}_{uid}_{timestamp}"

    def _generate_task_id(self, task_type: str) -> str:
        """Generate a task ID in the standard format."""
        try:
            from common.infrastructure.state.execution.thread_manager import (
                HierarchicalThreadManager
            )
            return HierarchicalThreadManager.generate_item_thread_id(
                workflow_thread_id=self.workflow_id,
                item_type=task_type,
                item_name=task_type,
                item_number=1
            )
        except ImportError:
            # Fallback format
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:20]
            uid = uuid.uuid4().hex[:8]
            wf_ref = self.workflow_id[-8:] if self.workflow_id and len(self.workflow_id) > 8 else "none"
            return f"task_{wf_ref}_{task_type}_{uid}_{timestamp}"

    # ═══════════════════════════════════════════════════════════════════
    # CONVERSION METHODS
    # ═══════════════════════════════════════════════════════════════════

    def to_langgraph_config(self) -> Dict[str, Any]:
        """
        Convert to LangGraph config format for checkpointing.

        Returns:
            Config dict compatible with LangGraph's invoke() method
        """
        return {
            "configurable": {
                "thread_id": self.workflow_id or self.session_id,
                "checkpoint_ns": f"{self.session_id}:{self.workflow_type}"
            }
        }

    def to_cache_key(self, suffix: str = "") -> str:
        """
        Generate isolated cache key scoped to this context.

        Args:
            suffix: Additional suffix for the key (e.g., "enrichment:pressure_transmitter")

        Returns:
            Cache key in format: ctx:{session_id}:{workflow_id}:{suffix}
        """
        base = f"ctx:{self.session_id}:{self.workflow_id}"
        return f"{base}:{suffix}" if suffix else base

    def to_log_context(self) -> Dict[str, str]:
        """
        Convert to structured logging fields.

        Returns:
            Dict with truncated IDs for log readability
        """
        return {
            "sid": self.session_id[:16] if self.session_id else "-",
            "wid": self.workflow_id[:16] if self.workflow_id else "-",
            "tid": self.task_id[:16] if self.task_id else "-",
            "cid": self.correlation_id,
            "wtype": self.workflow_type or "-",
            "zone": self.zone
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for storage/transmission.

        Returns:
            Complete context as dict
        """
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "zone": self.zone,
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type,
            "parent_workflow_id": self.parent_workflow_id,
            "instance_id": self.instance_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "created_at": self.created_at.isoformat(),
            "correlation_id": self.correlation_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionContext":
        """
        Deserialize from dictionary.

        Args:
            data: Dict with context fields

        Returns:
            ExecutionContext instance
        """
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            session_id=data["session_id"],
            user_id=data.get("user_id", "anonymous"),
            zone=data.get("zone", "DEFAULT"),
            workflow_id=data.get("workflow_id", ""),
            workflow_type=data.get("workflow_type", ""),
            parent_workflow_id=data.get("parent_workflow_id"),
            instance_id=data.get("instance_id", str(uuid.uuid4())),
            task_id=data.get("task_id"),
            task_type=data.get("task_type"),
            created_at=created_at or datetime.utcnow(),
            correlation_id=data.get("correlation_id", uuid.uuid4().hex[:12])
        )

    # ═══════════════════════════════════════════════════════════════════
    # CHILD CONTEXT CREATION
    # ═══════════════════════════════════════════════════════════════════

    def create_child_workflow(
        self,
        workflow_type: str,
        workflow_id: Optional[str] = None
    ) -> "ExecutionContext":
        """
        Create a child workflow context (e.g., solution → product_search).

        Args:
            workflow_type: Type of the child workflow
            workflow_id: Optional explicit workflow ID

        Returns:
            New ExecutionContext for the child workflow
        """
        child = ExecutionContext(
            session_id=self.session_id,
            user_id=self.user_id,
            zone=self.zone,
            workflow_id=workflow_id or "",  # Will auto-generate in __post_init__
            workflow_type=workflow_type,
            parent_workflow_id=self.workflow_id,
            correlation_id=self.correlation_id  # Preserve correlation for tracing
        )

        logger.debug(
            f"[ExecutionContext] Created child workflow: "
            f"parent={self.workflow_id[:16]}, child={child.workflow_id[:16]}, "
            f"type={workflow_type}"
        )

        return child

    def create_task(
        self,
        task_type: str,
        task_id: Optional[str] = None
    ) -> "ExecutionContext":
        """
        Create a task context within this workflow.

        Args:
            task_type: Type of the task (validation, enrichment, etc.)
            task_id: Optional explicit task ID

        Returns:
            New ExecutionContext for the task (shares workflow context)
        """
        return ExecutionContext(
            session_id=self.session_id,
            user_id=self.user_id,
            zone=self.zone,
            workflow_id=self.workflow_id,
            workflow_type=self.workflow_type,
            parent_workflow_id=self.parent_workflow_id,
            instance_id=self.instance_id,
            task_id=task_id or self._generate_task_id(task_type),
            task_type=task_type,
            correlation_id=self.correlation_id
        )

    def with_task(self, task_type: str) -> "ExecutionContext":
        """
        Create a copy of this context with task information added.

        Convenience alias for create_task().

        Args:
            task_type: Type of the task

        Returns:
            New ExecutionContext with task info
        """
        return self.create_task(task_type)

    # ═══════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def is_child_of(self, parent_ctx: "ExecutionContext") -> bool:
        """
        Check if this context is a child of another context.

        Args:
            parent_ctx: Potential parent context

        Returns:
            True if this is a child workflow of parent_ctx
        """
        return self.parent_workflow_id == parent_ctx.workflow_id

    def same_session(self, other: "ExecutionContext") -> bool:
        """
        Check if two contexts belong to the same session.

        Args:
            other: Another context to compare

        Returns:
            True if both contexts have the same session_id
        """
        return self.session_id == other.session_id

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"ExecutionContext("
            f"session={self.session_id[:16]}..., "
            f"workflow={self.workflow_type}, "
            f"cid={self.correlation_id})"
        )

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"ExecutionContext("
            f"session_id='{self.session_id}', "
            f"workflow_id='{self.workflow_id}', "
            f"workflow_type='{self.workflow_type}', "
            f"instance_id='{self.instance_id}', "
            f"correlation_id='{self.correlation_id}')"
        )
