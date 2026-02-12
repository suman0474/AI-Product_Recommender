"""
Router Agent - Separate Router Component for Intent-Based Workflow Dispatch

Provides:
- Centralized routing logic separated from intent classification
- Parent-child ID propagation when spawning child workflows
- Session locking and intent carrying
- Integration with state storage and distributed locks

Usage:
    from agentic.agents.routing.router_agent import RouterAgent
    
    router = RouterAgent(
        intent_classifier=intent_classifier,
        state_storage=storage,
        distributed_lock=lock,
        instance_manager=instance_manager
    )
    
    result = router.route(
        session_id="session_abc",
        query="Design a flow measurement system",
        context={}
    )
"""

import uuid
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum

# Local imports
from .state_storage import (
    SessionState,
    InstanceState,
    StateStorage,
    InMemoryStateStorage,
    create_state_storage,
)

try:
    from observability.structured_logger import (
        StructuredLogger,
        set_correlation_context,
        get_correlation_id,
    )
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

try:
    from concurrency.distributed_lock import create_distributed_lock
    from concurrency.optimistic_lock import ConcurrentModificationError
    CONCURRENCY_AVAILABLE = True
except ImportError:
    CONCURRENCY_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# ROUTING RESULT
# =============================================================================

class WorkflowType(Enum):
    """Available workflow types for routing."""
    SOLUTION = "solution"
    PRODUCT_SEARCH = "product_search"
    INSTRUMENT_IDENTIFIER = "instrument_identifier"
    ENGENIE_CHAT = "engenie_chat"
    OUT_OF_DOMAIN = "out_of_domain"


@dataclass
class RoutingResult:
    """
    Result of routing decision.
    
    Contains all information needed to dispatch to a workflow.
    """
    session_id: str
    instance_id: Optional[str]
    workflow_type: WorkflowType
    intent: str
    confidence: float
    is_new_session: bool
    is_new_instance: bool
    is_locked: bool
    parent_instance_id: Optional[str]
    query: str
    reasoning: str
    classification_time_ms: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "instance_id": self.instance_id,
            "workflow_type": self.workflow_type.value,
            "intent": self.intent,
            "confidence": self.confidence,
            "is_new_session": self.is_new_session,
            "is_new_instance": self.is_new_instance,
            "is_locked": self.is_locked,
            "parent_instance_id": self.parent_instance_id,
            "query": self.query,
            "reasoning": self.reasoning,
            "classification_time_ms": self.classification_time_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# =============================================================================
# ROUTER AGENT
# =============================================================================

class RouterAgent:
    """
    Centralized router agent for intent-based workflow dispatch.
    
    Responsibilities:
    1. Get or create session with locking
    2. Check if session is locked to a workflow
    3. Classify intent if session not locked
    4. Create or reuse workflow instance
    5. Propagate parent-child IDs
    6. Return routing result with all context
    
    This is a SEPARATE COMPONENT from IntentClassificationRoutingAgent,
    focused purely on routing logic without the classification details.
    """
    
    def __init__(
        self,
        state_storage=None,
        distributed_lock=None,
        intent_classifier=None,
        connection_string: Optional[str] = None,
        use_memory_fallback: bool = False
    ):
        """
        Initialize router agent.
        
        Args:
            state_storage: StateStorage instance for persistence
            distributed_lock: Distributed lock manager
            intent_classifier: Intent classification function/object
            connection_string: Azure connection string (if storage not provided)
            use_memory_fallback: Use in-memory storage for testing
        """
        # Initialize state storage
        self.storage = state_storage or create_state_storage(
            connection_string=connection_string,
            use_memory_fallback=use_memory_fallback
        )
        
        # Initialize distributed lock
        if distributed_lock:
            self.lock = distributed_lock
        elif CONCURRENCY_AVAILABLE and connection_string:
            self.lock = create_distributed_lock(connection_string)
        else:
            # Use in-memory lock for development
            from concurrency.distributed_lock import InMemoryDistributedLock
            self.lock = InMemoryDistributedLock()
        
        # Intent classifier (can be injected or use default)
        self.intent_classifier = intent_classifier
        
        # Structured logger
        if OBSERVABILITY_AVAILABLE:
            self.log = StructuredLogger("RouterAgent", service="orchestrator")
        else:
            self.log = None
        
        self._log_info("Router agent initialized")
    
    def _log_info(self, message: str, **kwargs):
        """Log info message."""
        if self.log:
            self.log.info(message, **kwargs)
        else:
            logger.info(f"[RouterAgent] {message}")
    
    def _log_debug(self, message: str, **kwargs):
        """Log debug message."""
        if self.log:
            self.log.debug(message, **kwargs)
        else:
            logger.debug(f"[RouterAgent] {message}")
    
    def _log_error(self, message: str, error: Exception = None, **kwargs):
        """Log error message."""
        if self.log:
            self.log.error(message, error=error, **kwargs)
        else:
            logger.error(f"[RouterAgent] {message}: {error}")
    
    def route(
        self,
        session_id: str,
        query: str,
        user_id: str = "",
        context: Optional[Dict[str, Any]] = None,
        parent_instance_id: Optional[str] = None,
    ) -> RoutingResult:
        """
        Route a query to the appropriate workflow.
        
        Args:
            session_id: User's session ID
            query: User's query text
            user_id: User identifier
            context: Additional context (workflow_hint, current_step, etc.)
            parent_instance_id: Parent instance ID for child workflows
            
        Returns:
            RoutingResult with routing decision and instance info
        """
        start_time = datetime.utcnow()
        context = context or {}
        
        self._log_info(
            "Routing query",
            session_id=session_id[:16] + "..." if len(session_id) > 16 else session_id,
            query_preview=query[:50] + "..." if len(query) > 50 else query,
            has_parent=bool(parent_instance_id)
        )
        
        # Step 1: Get or create session
        session, is_new_session = self._get_or_create_session(
            session_id, user_id, context
        )
        
        # Step 2: Check if session is locked
        if session.locked and session.intent:
            # Session is locked to a workflow - skip classification
            workflow_type = self._intent_to_workflow_type(session.intent)
            
            self._log_info(
                "Session locked",
                locked_intent=session.intent,
                workflow_type=workflow_type.value
            )
            
            # Get or create instance for this workflow
            instance, is_new_instance = self._get_or_create_instance(
                session_id=session_id,
                workflow_type=workflow_type,
                intent=session.intent,
                trigger_source=query[:50],
                parent_instance_id=parent_instance_id
            )
            
            end_time = datetime.utcnow()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            return RoutingResult(
                session_id=session_id,
                instance_id=instance.instance_id if instance else None,
                workflow_type=workflow_type,
                intent=session.intent,
                confidence=1.0,
                is_new_session=is_new_session,
                is_new_instance=is_new_instance,
                is_locked=True,
                parent_instance_id=parent_instance_id,
                query=query,
                reasoning=f"Session locked to {session.intent} workflow",
                classification_time_ms=duration_ms
            )
        
        # Step 3: Classify intent
        intent, confidence, is_solution, solution_indicators = self._classify_intent(
            query, context
        )
        
        # Step 4: Determine workflow type
        workflow_type = self._determine_workflow_type(
            intent, is_solution, context
        )
        
        self._log_info(
            "Intent classified",
            intent=intent,
            confidence=confidence,
            workflow_type=workflow_type.value,
            is_solution=is_solution
        )
        
        # Step 5: Lock session to this workflow (for stateful workflows)
        if workflow_type in (WorkflowType.SOLUTION, WorkflowType.INSTRUMENT_IDENTIFIER):
            session.locked = True
            session.intent = intent
            session.workflow_type = workflow_type.value
            self.storage.save_session(session_id, session)
            
            self._log_debug("Session locked to workflow", workflow=workflow_type.value)
        
        # Step 6: Create or get instance
        if workflow_type == WorkflowType.ENGENIE_CHAT:
            # Chat is fire-and-forget, no instance tracking
            instance = None
            is_new_instance = False
        else:
            instance, is_new_instance = self._get_or_create_instance(
                session_id=session_id,
                workflow_type=workflow_type,
                intent=intent,
                trigger_source=query[:50],
                parent_instance_id=parent_instance_id
            )
        
        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        return RoutingResult(
            session_id=session_id,
            instance_id=instance.instance_id if instance else None,
            workflow_type=workflow_type,
            intent=intent,
            confidence=confidence,
            is_new_session=is_new_session,
            is_new_instance=is_new_instance,
            is_locked=session.locked,
            parent_instance_id=parent_instance_id,
            query=query,
            reasoning=self._build_reasoning(intent, workflow_type, is_solution),
            classification_time_ms=duration_ms,
            metadata={
                "is_solution": is_solution,
                "solution_indicators": solution_indicators,
            }
        )
    
    def route_child(
        self,
        parent_instance_id: str,
        workflow_type: WorkflowType,
        trigger_source: str,
        metadata: Optional[Dict] = None
    ) -> RoutingResult:
        """
        Route a child workflow spawned by a parent workflow.
        
        Used when Solution workflow spawns Product Search instances.
        
        Args:
            parent_instance_id: Parent instance ID
            workflow_type: Child workflow type (e.g., PRODUCT_SEARCH)
            trigger_source: What triggered this child (e.g., "flow_transmitter")
            metadata: Additional metadata
            
        Returns:
            RoutingResult with child instance info
        """
        start_time = datetime.utcnow()
        
        # Get parent instance
        parent = self.storage.get_instance(parent_instance_id)
        if not parent:
            self._log_error(
                "Parent instance not found",
                parent_id=parent_instance_id
            )
            raise ValueError(f"Parent instance not found: {parent_instance_id}")
        
        session_id = parent.session_id
        
        self._log_info(
            "Routing child workflow",
            parent_id=parent_instance_id,
            workflow_type=workflow_type.value,
            trigger=trigger_source
        )
        
        # Create child instance with parent link
        instance, is_new = self._get_or_create_instance(
            session_id=session_id,
            workflow_type=workflow_type,
            intent=parent.intent,  # Inherit intent from parent
            trigger_source=trigger_source,
            parent_instance_id=parent_instance_id
        )
        
        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        return RoutingResult(
            session_id=session_id,
            instance_id=instance.instance_id,
            workflow_type=workflow_type,
            intent=parent.intent or "inherited",
            confidence=1.0,
            is_new_session=False,
            is_new_instance=is_new,
            is_locked=True,
            parent_instance_id=parent_instance_id,
            query=trigger_source,
            reasoning=f"Child workflow spawned by parent {parent_instance_id}",
            classification_time_ms=duration_ms,
            metadata=metadata or {}
        )
    
    def unlock_session(self, session_id: str) -> bool:
        """
        Unlock a session (allow new intent classification).
        
        Called when user says "start over", "reset", etc.
        """
        session = self.storage.get_session(session_id)
        if session:
            session.locked = False
            session.intent = None
            session.workflow_type = None
            self.storage.save_session(session_id, session)
            
            self._log_info("Session unlocked", session_id=session_id)
            return True
        return False
    
    def _get_or_create_session(
        self,
        session_id: str,
        user_id: str,
        context: Dict
    ) -> tuple:
        """Get existing session or create new one."""
        with self.lock.acquire(f"session:{session_id}", timeout=5):
            existing = self.storage.get_session(session_id)
            
            if existing:
                # Update request count
                existing.request_count += 1
                self.storage.save_session(session_id, existing)
                return existing, False
            
            # Create new session
            session = SessionState(
                session_id=session_id,
                user_id=user_id,
                intent=context.get("workflow_hint"),
                locked=bool(context.get("workflow_hint")),
                request_count=1
            )
            self.storage.save_session(session_id, session)
            
            self._log_info("Created new session", session_id=session_id)
            return session, True
    
    def _get_or_create_instance(
        self,
        session_id: str,
        workflow_type: WorkflowType,
        intent: str,
        trigger_source: str,
        parent_instance_id: Optional[str] = None
    ) -> tuple:
        """Get existing instance or create new one (with deduplication)."""
        lock_key = f"instance:{session_id}:{workflow_type.value}:{trigger_source[:30]}"
        
        with self.lock.acquire(lock_key, timeout=5):
            # Check for existing (deduplication)
            existing = self.storage.get_instance_by_trigger(
                session_id=session_id,
                workflow_type=workflow_type.value,
                trigger_source=trigger_source
            )
            
            if existing:
                # Increment request count
                existing.request_count += 1
                self.storage.save_instance(existing)
                
                self._log_debug(
                    "Reusing existing instance",
                    instance_id=existing.instance_id,
                    request_count=existing.request_count
                )
                return existing, False
            
            # Create new instance
            instance = InstanceState(
                instance_id=str(uuid.uuid4()),
                session_id=session_id,
                workflow_type=workflow_type.value,
                parent_id=parent_instance_id,
                intent=intent,
                trigger_source=trigger_source,
                status="created"
            )
            self.storage.save_instance(instance)
            
            self._log_info(
                "Created new instance",
                instance_id=instance.instance_id,
                parent_id=parent_instance_id,
                workflow_type=workflow_type.value
            )
            return instance, True
    
    def _classify_intent(
        self,
        query: str,
        context: Dict
    ) -> tuple:
        """
        Classify user intent.
        
        Returns: (intent, confidence, is_solution, solution_indicators)
        """
        if self.intent_classifier:
            # Use injected classifier
            try:
                result = self.intent_classifier.classify(
                    query=query,
                    session_id=context.get("session_id", "default"),
                    context=context
                )
                return (
                    result.intent,
                    result.confidence,
                    result.is_solution,
                    getattr(result, "solution_indicators", [])
                )
            except Exception as e:
                self._log_error("Intent classification failed", error=e)
        
        # Fallback: simple keyword-based classification
        return self._fallback_classify(query)
    
    def _fallback_classify(self, query: str) -> tuple:
        """Simple fallback classification."""
        query_lower = query.lower()
        
        # Solution indicators
        solution_keywords = ["design", "system", "solution", "build", "create", "multiple"]
        if any(kw in query_lower for kw in solution_keywords):
            return ("solution", 0.7, True, ["keyword_match"])
        
        # Instrument identifier indicators
        instrument_keywords = ["need", "find", "sensor", "transmitter", "pressure", "temperature"]
        if any(kw in query_lower for kw in instrument_keywords):
            return ("requirements", 0.6, False, [])
        
        # Default to chat
        return ("question", 0.5, False, [])
    
    def _determine_workflow_type(
        self,
        intent: str,
        is_solution: bool,
        context: Dict
    ) -> WorkflowType:
        """Determine workflow type from intent."""
        # Solution forcing intents
        if intent in ("solution", "system", "design", "complex_system"):
            return WorkflowType.SOLUTION
        
        # is_solution flag override
        if is_solution:
            return WorkflowType.SOLUTION
        
        # Instrument identifier intents
        if intent in ("requirements", "instrument", "additional_specs", "accessories"):
            return WorkflowType.INSTRUMENT_IDENTIFIER
        
        # Out of domain
        if intent in ("chitchat", "unrelated"):
            return WorkflowType.OUT_OF_DOMAIN
        
        # Default to chat
        return WorkflowType.ENGENIE_CHAT
    
    def _intent_to_workflow_type(self, intent: str) -> WorkflowType:
        """Convert intent string to WorkflowType enum."""
        intent_map = {
            "solution": WorkflowType.SOLUTION,
            "system": WorkflowType.SOLUTION,
            "instrument_identifier": WorkflowType.INSTRUMENT_IDENTIFIER,
            "requirements": WorkflowType.INSTRUMENT_IDENTIFIER,
            "engenie_chat": WorkflowType.ENGENIE_CHAT,
            "question": WorkflowType.ENGENIE_CHAT,
            "product_search": WorkflowType.PRODUCT_SEARCH,
        }
        return intent_map.get(intent.lower(), WorkflowType.ENGENIE_CHAT)
    
    def _build_reasoning(
        self,
        intent: str,
        workflow_type: WorkflowType,
        is_solution: bool
    ) -> str:
        """Build human-readable reasoning."""
        if workflow_type == WorkflowType.SOLUTION:
            return f"Routed to Solution workflow (intent={intent}, is_solution={is_solution})"
        elif workflow_type == WorkflowType.INSTRUMENT_IDENTIFIER:
            return f"Routed to Instrument Identifier (intent={intent})"
        elif workflow_type == WorkflowType.ENGENIE_CHAT:
            return f"Routed to EnGenie Chat (intent={intent})"
        else:
            return f"Routed to {workflow_type.value}"


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_router_agent(
    connection_string: Optional[str] = None,
    use_memory_fallback: bool = False,
    intent_classifier=None
) -> RouterAgent:
    """
    Create a router agent instance.
    
    Args:
        connection_string: Azure connection string
        use_memory_fallback: Use in-memory for testing
        intent_classifier: Optional intent classifier
        
    Returns:
        RouterAgent instance
    """
    return RouterAgent(
        connection_string=connection_string,
        use_memory_fallback=use_memory_fallback,
        intent_classifier=intent_classifier
    )
