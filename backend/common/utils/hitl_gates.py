"""
Human-in-the-Loop Approval Gates

Provides utilities for implementing human approval gates in LangGraph workflows.
These can be used to pause execution and wait for user confirmation before
proceeding with critical operations.

Usage:
    from common.utils.hitl_gates import ApprovalGate, create_approval_node

    # In workflow definition:
    workflow.add_node("approval", create_approval_node("review_requirements"))
"""
import logging
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class ApprovalRequest:
    """Represents a request for human approval."""
    gate_id: str
    gate_type: str
    message: str
    data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None


class ApprovalGate:
    """
    Human-in-the-Loop approval gate for workflows.
    
    When auto_mode is disabled, this creates an interrupt point
    where the workflow pauses for human review.
    """
    
    def __init__(
        self,
        gate_id: str,
        gate_type: str = "generic",
        message: str = "Please review and approve to continue",
        auto_approve_condition: Optional[Callable[[Dict], bool]] = None
    ):
        """
        Initialize approval gate.
        
        Args:
            gate_id: Unique identifier for this gate
            gate_type: Type of approval (e.g., "requirements", "vendor", "budget")
            message: Message to display to user
            auto_approve_condition: Optional function that returns True to auto-approve
        """
        self.gate_id = gate_id
        self.gate_type = gate_type
        self.message = message
        self.auto_approve_condition = auto_approve_condition
        logger.info(f"[HITL] ApprovalGate initialized: {gate_id} ({gate_type})")
    
    def should_auto_approve(self, state: Dict[str, Any]) -> bool:
        """
        Check if this gate should auto-approve.
        
        Returns True if:
        - auto_mode is enabled in state
        - auto_approve_condition returns True
        """
        # Check auto_mode flag
        if state.get("auto_mode", False):
            logger.info(f"[HITL] Gate {self.gate_id}: Auto-approved (auto_mode=True)")
            return True
        
        # Check custom condition
        if self.auto_approve_condition and self.auto_approve_condition(state):
            logger.info(f"[HITL] Gate {self.gate_id}: Auto-approved (condition met)")
            return True
        
        return False
    
    def create_request(self, state: Dict[str, Any]) -> ApprovalRequest:
        """Create an approval request from current state."""
        return ApprovalRequest(
            gate_id=self.gate_id,
            gate_type=self.gate_type,
            message=self.message,
            data=self._extract_review_data(state)
        )
    
    def _extract_review_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant data for human review."""
        # Default: extract common fields
        review_data = {}
        
        # Requirements review
        if "structured_requirements" in state:
            review_data["requirements"] = state["structured_requirements"]
        
        if "product_type" in state:
            review_data["product_type"] = state["product_type"]
        
        if "validation_result" in state:
            review_data["validation"] = state["validation_result"]
        
        # Vendor analysis review
        if "vendor_matches" in state:
            review_data["vendor_count"] = len(state["vendor_matches"])
            review_data["top_vendors"] = state["vendor_matches"][:5]
        
        return review_data
    
    def check(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check the approval gate and return updated state.
        
        In non-auto mode, this sets an interrupt flag that the
        workflow should check and pause on.
        """
        # Check for auto-approve
        if self.should_auto_approve(state):
            state["approval_status"] = ApprovalStatus.APPROVED.value
            state["approval_gate"] = None
            return state
        
        # Create approval request
        request = self.create_request(state)
        
        logger.info(f"[HITL] Gate {self.gate_id}: Approval required")
        logger.info(f"[HITL]   Message: {self.message}")
        logger.info(f"[HITL]   Data keys: {list(request.data.keys())}")
        
        # Set interrupt state
        state["approval_status"] = ApprovalStatus.PENDING.value
        state["approval_gate"] = {
            "gate_id": self.gate_id,
            "gate_type": self.gate_type,
            "message": self.message,
            "data": request.data,
            "created_at": request.created_at.isoformat()
        }
        state["requires_approval"] = True
        
        return state


def create_approval_node(
    gate_id: str,
    gate_type: str = "generic",
    message: str = "Please review and approve to continue",
    auto_approve_condition: Optional[Callable[[Dict], bool]] = None
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Factory function to create an approval node for LangGraph workflows.
    
    Args:
        gate_id: Unique identifier for this gate
        gate_type: Type of approval
        message: Message to display
        auto_approve_condition: Optional auto-approve condition
        
    Returns:
        Node function for use with workflow.add_node()
        
    Example:
        workflow.add_node("review", create_approval_node(
            gate_id="requirements_review",
            gate_type="requirements",
            message="Please review the captured requirements"
        ))
    """
    gate = ApprovalGate(
        gate_id=gate_id,
        gate_type=gate_type,
        message=message,
        auto_approve_condition=auto_approve_condition
    )
    
    def node_function(state: Dict[str, Any]) -> Dict[str, Any]:
        return gate.check(state)
    
    return node_function


def approval_decision(state: Dict[str, Any]) -> str:
    """
    Conditional edge function for routing based on approval status.
    
    Returns:
        "proceed" if approved
        "wait" if pending (workflow should pause)
        "reject" if rejected
        
    Example:
        workflow.add_conditional_edges(
            "approval_gate",
            approval_decision,
            {"proceed": "next_step", "wait": END, "reject": "handle_rejection"}
        )
    """
    approval_status = state.get("approval_status", "pending")
    
    if approval_status == ApprovalStatus.APPROVED.value:
        logger.info("[HITL] Approval decision: PROCEED")
        return "proceed"
    elif approval_status == ApprovalStatus.REJECTED.value:
        logger.info("[HITL] Approval decision: REJECT")
        return "reject"
    else:
        logger.info("[HITL] Approval decision: WAIT")
        return "wait"


# Pre-configured gates for common use cases
REQUIREMENTS_REVIEW_GATE = ApprovalGate(
    gate_id="requirements_review",
    gate_type="requirements",
    message="Please review the captured requirements before proceeding to vendor analysis"
)

VENDOR_SELECTION_GATE = ApprovalGate(
    gate_id="vendor_selection",
    gate_type="vendor",
    message="Please review the vendor analysis results before final ranking"
)

BUDGET_APPROVAL_GATE = ApprovalGate(
    gate_id="budget_approval",
    gate_type="budget",
    message="This selection exceeds the standard budget. Please approve to continue."
)
