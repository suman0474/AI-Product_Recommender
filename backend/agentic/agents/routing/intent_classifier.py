"""
Intent Classification Routing Agent

Routes user input from the UI textarea to the appropriate agentic workflow:
1. Solution Workflow - Complex engineering challenges requiring multiple instruments
2. Instrument Identifier Workflow - Single product requirements
3. Product Info Workflow - Questions about products, standards, vendors

Also rejects out-of-domain queries (unrelated to industrial automation).

Usage:
    agent = IntentClassificationRoutingAgent()
    result = agent.classify(query="I need a pressure transmitter 0-100 PSI", session_id="abc123")
    # Returns: WorkflowRoutingResult with target workflow and reasoning
"""

import logging
import time
from typing import Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from threading import Lock

# Debug imports
from debug_flags import debug_log, timed_execution, is_debug_enabled

logger = logging.getLogger(__name__)

# Level 4.5: Import WorkflowRegistry for registry-based matching
try:
    from .workflow_registry import get_workflow_registry, WorkflowRegistry
    _REGISTRY_AVAILABLE = True
except ImportError:
    _REGISTRY_AVAILABLE = False
    logger.debug("[Router] WorkflowRegistry not available - using IntentConfig fallback")


# =============================================================================
# WORKFLOW STATE MEMORY (Session-based memory for workflow locking)
# =============================================================================

class WorkflowStateMemory:
    """
    CACHE for workflow state from frontend SessionManager.
    
    ARCHITECTURE: Frontend SessionManager is the SOURCE OF TRUTH.
    This class only:
    - Caches workflow hints from frontend for the current request
    - Validates hints against known workflows  
    - Clears state ONLY on explicit exit detection
    - Does NOT make independent workflow decisions
    
    The response contains target_workflow which frontend uses to update
    SessionManager. Frontend then passes workflow_hint on next request.
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._workflow_states: Dict[str, str] = {}
        self._workflow_timestamps: Dict[str, datetime] = {}
        self._state_lock = Lock()
        logger.info("[WorkflowStateMemory] Initialized - Backend workflow state tracking enabled")
    
    def get_workflow(self, session_id: str) -> Optional[str]:
        """Get current workflow for a session."""
        with self._state_lock:
            workflow = self._workflow_states.get(session_id)
            if workflow:
                logger.debug(f"[WorkflowStateMemory] Session {session_id[:8]}...: current workflow = {workflow}")
            return workflow
    
    def set_workflow(self, session_id: str, workflow: str) -> None:
        """Set workflow for a session."""
        with self._state_lock:
            old_workflow = self._workflow_states.get(session_id)
            self._workflow_states[session_id] = workflow
            self._workflow_timestamps[session_id] = datetime.now()
            logger.info(f"[WorkflowStateMemory] Session {session_id[:8]}...: workflow changed {old_workflow} -> {workflow}")
    
    def clear_workflow(self, session_id: str) -> None:
        """Clear workflow for a session (allows re-classification)."""
        with self._state_lock:
            old_workflow = self._workflow_states.pop(session_id, None)
            self._workflow_timestamps.pop(session_id, None)
            if old_workflow:
                logger.info(f"[WorkflowStateMemory] Session {session_id[:8]}...: workflow cleared (was {old_workflow})")
    
    def is_locked(self, session_id: str) -> bool:
        """Check if session has an active workflow."""
        return self.get_workflow(session_id) is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._state_lock:
            return {
                "active_sessions": len(self._workflow_states),
                "workflows": dict(self._workflow_states)
            }

# Global singleton instance
_workflow_memory = WorkflowStateMemory()

def get_workflow_memory() -> WorkflowStateMemory:
    """Get the global workflow state memory instance."""
    return _workflow_memory


# =============================================================================
# EXIT DETECTION
# =============================================================================

# Phrases that indicate user wants to exit current workflow (rule-based fallback)
EXIT_PHRASES = [
    "start over", "new search", "reset", "clear", "begin again", 
    "start new", "exit", "quit", "cancel", "back to start"
]

# Greetings that indicate new conversation (rule-based fallback)
GREETING_PHRASES = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

def _should_exit_rule_based(user_input: str) -> bool:
    """Rule-based fallback for exit detection (used when LLM is unavailable)."""
    lower_input = user_input.lower().strip()
    
    # Check for exit phrases
    if any(phrase in lower_input for phrase in EXIT_PHRASES):
        return True
    
    # Check for pure greetings (new conversation)
    if lower_input in GREETING_PHRASES:
        return True
    
    return False

def should_exit_workflow(user_input: str) -> bool:
    """
    Check if user wants to exit current workflow using LLM classification.
    
    Uses the classify_intent_tool with temperature 0.0 for deterministic results.
    Falls back to rule-based detection if LLM fails.
    """
    try:
        # Import here to avoid circular imports
        from tools.intent_tools import _classify_llm_fast
        
        result = _classify_llm_fast(user_input)
        
        if result is not None:
            intent = result.get("intent", "")
            # Exit if classified as 'exit' or 'greeting' (new conversation)
            if intent in {"exit", "greeting"}:
                logger.info(f"[ExitDetection] LLM detected exit intent: '{intent}'")
                return True
            return False
        
        # LLM returned 'unknown' - not an exit intent
        return False
        
    except Exception as e:
        logger.warning(f"[ExitDetection] LLM classification failed: {e}, using rule-based fallback")
        return _should_exit_rule_based(user_input)


# =============================================================================
# WORKFLOW TARGETS
# =============================================================================

class WorkflowTarget(Enum):
    """Available workflow routing targets."""
    SOLUTION_WORKFLOW = "solution"              # Complex systems, multiple instruments
    INSTRUMENT_IDENTIFIER = "instrument_identifier"  # Single product requirements
    ENGENIE_CHAT = "engenie_chat"               # Questions, greetings, RAG queries
    OUT_OF_DOMAIN = "out_of_domain"             # Unrelated queries


# =============================================================================
# WORKFLOW ROUTING RESULT
# =============================================================================

@dataclass
class WorkflowRoutingResult:
    """Result of workflow routing classification."""
    query: str                          # Original query
    target_workflow: WorkflowTarget     # Which workflow to route to
    intent: str                         # Raw intent from classify_intent_tool
    confidence: float                   # Confidence (0.0-1.0)
    reasoning: str                      # Explanation for routing decision
    is_solution: bool                   # Whether this is a solution-type request
    target_rag: Optional[str]           # For ProductInfo: standards_rag, strategy_rag, product_info_rag
    solution_indicators: list           # Indicators that triggered solution detection
    extracted_info: Dict                # Any extracted information
    classification_time_ms: float       # Time taken to classify
    timestamp: str                      # ISO timestamp
    reject_message: Optional[str]       # Message for out-of-domain queries

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "target_workflow": self.target_workflow.value,
            "intent": self.intent,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "reasoning": self.reasoning,
            "is_solution": self.is_solution,
            "target_rag": self.target_rag,
            "solution_indicators": self.solution_indicators,
            "extracted_info": self.extracted_info,
            "classification_time_ms": self.classification_time_ms,
            "timestamp": self.timestamp,
            "reject_message": self.reject_message
        }


# =============================================================================
# OUT OF DOMAIN RESPONSE
# =============================================================================

OUT_OF_DOMAIN_MESSAGE = """I'm EnGenie, your industrial automation assistant. I can help with:

✓ Industrial automation products (transmitters, sensors, valves, PLCs, controllers)
✓ Technical specifications and certifications  
✓ Standards and compliance (IEC, ISO, ATEX, SIL, API, ASME)
✓ Vendor information and product comparisons
✓ Process instrumentation and control systems

Your query appears to be outside my expertise area. Please ask about industrial automation equipment, technical standards, or related topics."""


# =============================================================================
# INTENT TO WORKFLOW MAPPING
# =============================================================================
# Unified intent-to-workflow mapping with validation and is_solution override logic.
# This is the SINGLE SOURCE OF TRUTH for intent routing decisions.

class IntentConfig:
    """
    Centralized intent-to-workflow mapping with validation.

    Handles the complex logic of routing based on both intent name and is_solution flag.
    PHASE 1 FIX: Replaces hardcoded INTENT_TO_WORKFLOW_MAP with robust configuration.
    """

    # Primary intent-to-workflow mappings
    # These are the ONLY valid intent values the LLM should return
    # Supports both uppercase (from new prompt) and lowercase (legacy) intent names
    INTENT_MAPPINGS = {
        # ===== NEW 4-INTENT ARCHITECTURE (uppercase from LLM, lowercase normalized) =====
        # Solution Workflow - Complex systems with multiple instruments
        "solution": WorkflowTarget.SOLUTION_WORKFLOW,

        # Instrument Identifier Workflow - Single product with clear specifications
        "search": WorkflowTarget.INSTRUMENT_IDENTIFIER,

        # EnGenie Chat Workflow - Knowledge, education, conversational queries
        "chat": WorkflowTarget.ENGENIE_CHAT,

        # Out of Domain - Unrelated to industrial automation
        "invalid_input": WorkflowTarget.OUT_OF_DOMAIN,

        # ===== LEGACY MAPPINGS (for backwards compatibility) =====
        # Solution Workflow variants
        "system": WorkflowTarget.SOLUTION_WORKFLOW,
        "systems": WorkflowTarget.SOLUTION_WORKFLOW,
        "complex_system": WorkflowTarget.SOLUTION_WORKFLOW,
        "design": WorkflowTarget.SOLUTION_WORKFLOW,

        # Instrument Identifier Workflow (legacy names)
        "requirements": WorkflowTarget.INSTRUMENT_IDENTIFIER,
        "additional_specs": WorkflowTarget.INSTRUMENT_IDENTIFIER,
        "instrument": WorkflowTarget.INSTRUMENT_IDENTIFIER,
        "accessories": WorkflowTarget.INSTRUMENT_IDENTIFIER,
        "spec_request": WorkflowTarget.INSTRUMENT_IDENTIFIER,

        # EnGenie Chat Workflow (legacy names)
        "question": WorkflowTarget.ENGENIE_CHAT,
        "productinfo": WorkflowTarget.ENGENIE_CHAT,
        "product_info": WorkflowTarget.ENGENIE_CHAT,
        "greeting": WorkflowTarget.ENGENIE_CHAT,
        "confirm": WorkflowTarget.ENGENIE_CHAT,
        "reject": WorkflowTarget.ENGENIE_CHAT,
        "standards": WorkflowTarget.ENGENIE_CHAT,
        "vendor_strategy": WorkflowTarget.ENGENIE_CHAT,
        "grounded_chat": WorkflowTarget.ENGENIE_CHAT,
        "comparison": WorkflowTarget.ENGENIE_CHAT,
        "productcomparison": WorkflowTarget.ENGENIE_CHAT,

        # Out of Domain (legacy names)
        "chitchat": WorkflowTarget.OUT_OF_DOMAIN,
        "unrelated": WorkflowTarget.OUT_OF_DOMAIN,
    }

    # Intents that FORCE Solution Workflow regardless of other signals
    # These are "strong indicators" that override even low confidence
    # Includes both new (4-intent) and legacy intent names
    SOLUTION_FORCING_INTENTS = {
        "solution",  # New 4-intent architecture
        "system", "systems", "complex_system", "design"  # Legacy names
    }

    @classmethod
    def get_workflow(cls, intent: str, is_solution: bool = False) -> WorkflowTarget:
        """
        Determine workflow target based on intent and is_solution flag.

        ROUTING LOGIC (in priority order):
        1. If intent is in SOLUTION_FORCING_INTENTS → SOLUTION_WORKFLOW
        2. If is_solution=True → SOLUTION_WORKFLOW (override via flag)
        3. Otherwise → Use intent mapping (default: ENGENIE_CHAT for unknown)

        Args:
            intent: Intent classification string from LLM or rule-based classifier
            is_solution: Boolean flag indicating complex system detection

        Returns:
            WorkflowTarget enum value

        Examples:
            get_workflow("solution", is_solution=False) → SOLUTION_WORKFLOW
            get_workflow("question", is_solution=True) → SOLUTION_WORKFLOW
            get_workflow("unknown_intent", is_solution=False) → ENGENIE_CHAT
        """
        intent_lower = intent.lower().strip() if intent else "unrelated"

        # Priority 1: Intent is a solution forcing intent
        if intent_lower in cls.SOLUTION_FORCING_INTENTS:
            logger.debug(f"[IntentConfig] Solution forcing intent: '{intent_lower}'")
            return WorkflowTarget.SOLUTION_WORKFLOW

        # Priority 2: is_solution flag is set (explicit complex system detection)
        if is_solution:
            logger.debug(f"[IntentConfig] is_solution flag set, routing to SOLUTION")
            return WorkflowTarget.SOLUTION_WORKFLOW

        # Priority 3: Use intent mapping (defaults to ENGENIE_CHAT for unknowns)
        workflow = cls.INTENT_MAPPINGS.get(intent_lower, WorkflowTarget.ENGENIE_CHAT)
        logger.debug(f"[IntentConfig] Intent '{intent_lower}' → {workflow.value}")
        return workflow

    @classmethod
    def is_known_intent(cls, intent: str) -> bool:
        """Check if intent is in the recognized list."""
        if not intent:
            return False
        return intent.lower().strip() in cls.INTENT_MAPPINGS

    @classmethod
    def get_all_valid_intents(cls) -> list:
        """Get list of all recognized intent values for validation and documentation."""
        return list(cls.INTENT_MAPPINGS.keys())

    @classmethod
    def print_config(cls):
        """Print configuration for debugging (logging)."""
        logger.info("[IntentConfig] ===== INTENT CONFIGURATION =====")
        logger.info(f"[IntentConfig] Total intents: {len(cls.INTENT_MAPPINGS)}")
        logger.info(f"[IntentConfig] Solution forcing intents: {cls.SOLUTION_FORCING_INTENTS}")
        for workflow in WorkflowTarget:
            intents = [i for i, w in cls.INTENT_MAPPINGS.items() if w == workflow]
            logger.info(f"[IntentConfig] {workflow.value}: {intents}")


# Legacy alias for backward compatibility (can be removed in next version)
INTENT_TO_WORKFLOW_MAP = IntentConfig.INTENT_MAPPINGS


# =============================================================================
# INTENT CLASSIFICATION ROUTING AGENT
# =============================================================================

class IntentClassificationRoutingAgent:
    """
    Agent that classifies user queries and routes to appropriate workflows.

    Uses classify_intent_tool as the core classifier and maps intents to
    workflow targets.
    
    WORKFLOW STATE LOCKING:
    - Tracks which workflow each session is in via WorkflowStateMemory
    - Once a session enters a workflow, subsequent queries stay in that workflow
    - User can exit workflow by saying "start over", "reset", etc.

    Workflow Routing:
    - solution → Solution Workflow (complex systems)
    - requirements, additional_specs → Instrument Identifier Workflow
    - question, productInfo, greeting, confirm, reject → EnGenie Chat Workflow
    - chitchat, unrelated → OUT_OF_DOMAIN (reject)
    """

    def __init__(self, name: str = "WorkflowRouter", use_registry: bool = True):
        """Initialize the agent.
        
        Args:
            name: Agent name for logging
            use_registry: If True, use WorkflowRegistry for matching (Level 4.5).
                         Falls back to IntentConfig if registry unavailable.
        """
        self.name = name
        self.classification_count = 0
        self.last_classification_time_ms = 0.0
        self._memory = get_workflow_memory()  # Use singleton workflow memory
        self._use_registry = use_registry and _REGISTRY_AVAILABLE
        
        if self._use_registry:
            self._registry = get_workflow_registry()
            logger.info(f"[{self.name}] Initialized with WorkflowRegistry (Level 4.5)")
        else:
            self._registry = None
            logger.info(f"[{self.name}] Initialized with IntentConfig fallback")

    def _is_query_relevant_to_workflow(self, query: str, workflow: str) -> bool:
        """
        Check if query is relevant to the currently locked workflow.
        
        This prevents knowledge/question queries from being incorrectly routed
        to SOLUTION workflow when session is locked.
        
        Args:
            query: User query string
            workflow: Currently locked workflow name
            
        Returns:
            True if query is relevant to workflow, False if it should break the lock
        """
        query_lower = query.lower().strip()
        
        # Knowledge/question patterns that are NOT relevant to solution workflow
        if workflow == "solution":
            # Question starters - these are knowledge queries, not solution continuation
            knowledge_patterns = [
                "what is", "what are", "what's the", "what does",
                "how does", "how do", "how is", "how are",
                "explain", "tell me about", "describe",
                "why is", "why does", "why do",
                "can you explain", "can you tell",
                "input current", "output range", "accuracy",
                "specification", "specs of", "datasheet"
            ]
            
            # Check for knowledge patterns
            if any(query_lower.startswith(p) or f" {p}" in query_lower for p in knowledge_patterns):
                logger.info(
                    f"[{self.name}] Query not relevant to solution workflow "
                    f"(knowledge pattern detected): '{query[:50]}...'"
                )
                return False
            
            # Solution-relevant patterns (should stay locked)
            solution_relevant = [
                "add", "include", "remove", "select", "confirm",
                "i've selected", "use this", "continue", "proceed",
                "next", "done", "finished", "submit"
            ]
            if any(p in query_lower for p in solution_relevant):
                return True
        
        # Default: consider relevant to respect workflow lock
        return True

    def _has_strong_solution_indicators(self, query: str) -> bool:
        """
        Check if query has strong indicators of a solution/system request.
        
        These indicators should preempt fast pattern matching for standards/keywords.
        Examples: "select and specify", "complete system", "design a system"
        """
        query_lower = query.lower().strip()
        
        strong_indicators = [
            "select and specify",
            "complete system",
            "design a system",
            "design a solution",
            "pump system",
            "measurement system",
            "control system",
            "instrumentation system",
            "complete instrumentation",
            "holistic solution",
            "full solution",
            "entire system"
        ]
        
        return any(indicator in query_lower for indicator in strong_indicators)

    @debug_log("INTENT_ROUTER", log_args=False)
    @timed_execution("INTENT_ROUTER", threshold_ms=2000)
    def classify(
        self,
        query: str,
        session_id: str = "default",
        context: Optional[Dict] = None
    ) -> WorkflowRoutingResult:
        """
        Classify a query and determine which workflow to route to.

        SMART LOCKING:
        - Helper logic checks if the user is stuck in a workflow.
        - Strong intents (greeting, new solution) BREAK the lock.
        - Contextual intents (questions, confirmations) RESPECT the lock.

        Args:
            query: User query string from UI textarea
            session_id: Session ID for workflow state tracking
            context: Optional context (current workflow step, conversation history)

        Returns:
            WorkflowRoutingResult with target workflow and details
        """
        start_time = datetime.now()
        
        logger.info(f"[{self.name}] Classifying: '{query[:80]}...' (session: {session_id[:8]}...)")

        # =====================================================================
        # STEP 1: CHECK FOR EXPRESS EXIT
        # =====================================================================
        if should_exit_workflow(query):
            self._memory.clear_workflow(session_id)
            logger.info(f"[{self.name}] Exit detected - clearing workflow state")
            # Proceed to classify as a fresh query (likely resulting in Greeting or ProductInfo)
        
        # =====================================================================
        # STEP 2: HANDLE WORKFLOW HINT FROM FRONTEND
        # =====================================================================
        context = context or {}
        workflow_hint = context.get('workflow_hint')
        valid_workflows = {'engenie_chat', 'solution', 'instrument_identifier'}
        
        if workflow_hint and workflow_hint not in valid_workflows:
            logger.warning(f"[{self.name}] Invalid workflow hint ignored: {workflow_hint}")
            workflow_hint = None
        
        # =====================================================================
        # STEP 3: CHECK WORKFLOW LOCK (with session type differentiation)
        # =====================================================================
        current_workflow = self._memory.get_workflow(session_id)
        
        # If no backend state but valid hint from frontend, restore it
        if not current_workflow and workflow_hint and not should_exit_workflow(query):
            logger.info(f"[{self.name}] Restoring workflow from frontend hint: {workflow_hint}")
            current_workflow = workflow_hint
            self._memory.set_workflow(session_id, current_workflow)
        
        if current_workflow and not should_exit_workflow(query):
            # FIX: Differentiate between main interface sessions and dedicated workflow sessions
            # Main interface sessions (e.g., "main_Daman_DEFAULT") should allow free workflow switching
            # Dedicated workflow sessions (e.g., "engenie_chat_1234567890") should stay locked
            
            is_main_interface = session_id.startswith("main_")
            
            if is_main_interface:
                # Main interface - allow re-classification for workflow switching
                logger.info(
                    f"[{self.name}] Main interface session detected (session: {session_id[:30]}...). "
                    f"Previous workflow: '{current_workflow}'. Will re-classify to allow workflow switching."
                )
                # Clear the workflow lock to allow re-classification
                self._memory.clear_workflow(session_id)
                # Continue to classification steps below
            else:
                # Dedicated workflow session - respect the workflow lock
                logger.info(
                    f"[{self.name}] WORKFLOW LOCKED: Dedicated session in '{current_workflow}' "
                    f"(session: {session_id[:30]}...)"
                )
                
                # Map workflow to target
                workflow_map = {
                    "engenie_chat": WorkflowTarget.ENGENIE_CHAT,
                    "instrument_identifier": WorkflowTarget.INSTRUMENT_IDENTIFIER,
                    "solution": WorkflowTarget.SOLUTION_WORKFLOW
                }
                
                target_workflow = workflow_map.get(current_workflow, WorkflowTarget.ENGENIE_CHAT)
                
                # Calculate time
                end_time = datetime.now()
                classification_time_ms = (end_time - start_time).total_seconds() * 1000
                
                return WorkflowRoutingResult(
                    query=query,
                    target_workflow=target_workflow,
                    intent="workflow_locked",
                    confidence=1.0,
                    reasoning=f"Dedicated session locked in {current_workflow} workflow",
                    is_solution=(current_workflow == "solution"),
                    target_rag=None,
                    solution_indicators=[],
                    extracted_info={
                        "workflow_locked": True, 
                        "current_workflow": current_workflow,
                        "session_type": "dedicated"
                    },
                    classification_time_ms=classification_time_ms,
                    timestamp=datetime.now().isoformat(),
                    reject_message=None
                )

        # =====================================================================
        # STEP 3.5: FAST PATTERN CLASSIFICATION (Regex - Zero Latency)
        # =====================================================================
        # Pre-check simple knowledge queries to avoid Semantic Embedding latency (200ms vs 0ms)
        
        # FIX: Check for strong solution indicators FIRST to prevent misrouting complex queries
        if self._has_strong_solution_indicators(query):
            logger.info(f"[{self.name}] Strong solution indicators detected - Skipping Fast Pattern Check to favor Solution Workflow")
        else:
            try:
                from agentic.workflows.engenie_chat.engenie_chat_intent_agent import classify_by_patterns, DataSource
                
                # Fast pattern check
                pattern_result = classify_by_patterns(query.lower().strip())
                
                if pattern_result:
                    source, conf, reason = pattern_result
                    # Only use if high confidence (which patterns usually are)
                    if conf >= 0.8:
                        logger.info(
                            f"[{self.name}] FAST PATTERN: {source.value} (conf={conf:.2f}) - "
                            f"Skipping Semantic Classifier"
                        )
                        
                        target_rag = None
                        intent_name = "question"
                        
                        # Map Source -> Target RAG & Intent
                        if source == DataSource.STANDARDS_RAG:
                            target_rag = "standards_rag" 
                            intent_name = "standards"
                        elif source == DataSource.STRATEGY_RAG:
                            target_rag = "strategy_rag"
                            intent_name = "vendor_strategy"
                        elif source == DataSource.INDEX_RAG:
                            target_rag = "product_info_rag"
                            intent_name = "product_info"
                        elif source == DataSource.DEEP_AGENT:
                            target_rag = "product_info_rag" # Deep agent is part of product info
                            intent_name = "deep_agent"
                            
                        # Calculate time
                        end_time = datetime.now()
                        classification_time_ms = (end_time - start_time).total_seconds() * 1000
                        
                        return WorkflowRoutingResult(
                            query=query,
                            target_workflow=WorkflowTarget.ENGENIE_CHAT,
                            intent=intent_name,
                            confidence=conf,
                            reasoning=f"Fast Pattern Match: {reason}",
                            is_solution=False,
                            target_rag=target_rag,
                            solution_indicators=[],
                            extracted_info={
                                "fast_pattern_match": True,
                                "source": source.value, 
                                "pattern_reason": reason
                            },
                            classification_time_ms=classification_time_ms,
                            timestamp=datetime.now().isoformat(),
                            reject_message=None
                        )
            except ImportError:
                pass # Ignore if module not found, proceed to semantic
            except Exception as e:
                logger.warning(f"[{self.name}] Fast pattern check failed: {e}")

        # =====================================================================
        # STEP 4: SEMANTIC CLASSIFICATION (PRIMARY - Embedding Similarity)
        # =====================================================================
        # Use semantic embeddings to classify intent based on similarity
        # to pre-defined workflow signature patterns
        
        try:
            from agentic.agents.routing.semantic_classifier import (
                get_semantic_classifier, 
                WorkflowType,
                ClassificationResult
            )
            
            semantic_classifier = get_semantic_classifier()
            semantic_result = semantic_classifier.classify(query)
            
            logger.info(
                f"[{self.name}] SEMANTIC: {semantic_result.workflow.value} "
                f"(conf={semantic_result.confidence:.3f}, match='{semantic_result.matched_signature[:50]}...')"
            )
            
            # High confidence semantic match - use directly
            # Use 0.70 threshold to capture simple product requests that should go to EnGenie
            if semantic_result.confidence >= 0.70:
                # Map semantic workflow to target
                semantic_workflow_map = {
                    WorkflowType.ENGENIE_CHAT: WorkflowTarget.ENGENIE_CHAT,
                    WorkflowType.SOLUTION_WORKFLOW: WorkflowTarget.SOLUTION_WORKFLOW,
                    WorkflowType.INSTRUMENT_IDENTIFIER: WorkflowTarget.INSTRUMENT_IDENTIFIER
                }
                target_workflow = semantic_workflow_map.get(
                    semantic_result.workflow, 
                    WorkflowTarget.ENGENIE_CHAT
                )
                is_solution = (semantic_result.workflow == WorkflowType.SOLUTION_WORKFLOW)
                
                # Set target RAG for EnGenie Chat
                target_rag = None
                if target_workflow == WorkflowTarget.ENGENIE_CHAT:
                    target_rag = "product_info_rag"
                
                # Set workflow state
                workflow_name = {
                    WorkflowTarget.ENGENIE_CHAT: "engenie_chat",
                    WorkflowTarget.INSTRUMENT_IDENTIFIER: "instrument_identifier",
                    WorkflowTarget.SOLUTION_WORKFLOW: "solution"
                }.get(target_workflow)
                
                # CONSOLIDATION: Don't set backend state here
                # Frontend receives target_workflow in response and updates SessionManager
                # Frontend then passes workflow_hint on next request
                logger.debug(f"[{self.name}] SEMANTIC: Target workflow: {workflow_name} (frontend will store)")
                
                end_time = datetime.now()
                classification_time_ms = (end_time - start_time).total_seconds() * 1000
                
                return WorkflowRoutingResult(
                    query=query,
                    target_workflow=target_workflow,
                    intent="semantic_match",
                    confidence=semantic_result.confidence,
                    reasoning=f"Semantic classification: {semantic_result.reasoning}",
                    is_solution=is_solution,
                    target_rag=target_rag,
                    solution_indicators=["semantic_match"] if is_solution else [],
                    extracted_info={
                        "semantic_classification": True,
                        "matched_signature": semantic_result.matched_signature,
                        "all_scores": semantic_result.all_scores
                    },
                    classification_time_ms=classification_time_ms,
                    timestamp=datetime.now().isoformat(),
                    reject_message=None
                )
            else:
                logger.info(
                    f"[{self.name}] SEMANTIC: Low confidence ({semantic_result.confidence:.3f}), "
                    f"falling back to rule-based classification"
                )
                
        except ImportError as e:
            logger.warning(f"[{self.name}] Semantic classifier not available: {e}")
        except Exception as e:
            logger.warning(f"[{self.name}] Semantic classification failed: {e}, using fallback")

        # =====================================================================
        # STEP 4: RULE-BASED CLASSIFICATION (FALLBACK)
        # =====================================================================
        
        # Import classify_intent_tool here to avoid circular imports
        try:
            from tools.intent_tools import classify_intent_tool
        except ImportError:
            logger.error("Could not import classify_intent_tool")
            return self._create_error_result(query, start_time, "Import error")

        # Get context values
        current_step = context.get("current_step") if context else None
        context_str = context.get("context") if context else None

        # Call the core classifier
        try:
            intent_result = classify_intent_tool.invoke({
                "user_input": query,
                "current_step": current_step,
                "context": context_str
            })
        except Exception as e:
            logger.error(f"[{self.name}] classify_intent_tool failed: {e}")
            return self._create_error_result(query, start_time, str(e))

        # Extract intent details
        intent = intent_result.get("intent", "unrelated")
        confidence = intent_result.get("confidence", 0.5)
        is_solution = intent_result.get("is_solution", False)
        solution_indicators = intent_result.get("solution_indicators", [])
        extracted_info = intent_result.get("extracted_info", {})

        # =================================================================
        # PHASE 3 FIX: Semantic validation before solution routing
        # =================================================================
        # If is_solution is True, verify with semantic classifier that this
        # is actually a design/build request, not a knowledge query
        
        # FIX: Don't run semantic override if we have strong solution indicators
        has_strong_indicators = self._has_strong_solution_indicators(query)
        
        if is_solution and not has_strong_indicators:
            try:
                from agentic.workflows.engenie_chat.engenie_chat_intent_agent import classify_query, DataSource
                semantic_source, semantic_conf, semantic_reason = classify_query(query, use_semantic_llm=False)
                
                rag_sources = {DataSource.INDEX_RAG, DataSource.STANDARDS_RAG, DataSource.STRATEGY_RAG, DataSource.DEEP_AGENT}
                
                if semantic_source in rag_sources and semantic_conf >= 0.7:
                    # Override: This is a RAG query, not a solution request
                    logger.info(
                        f"[{self.name}] PHASE 3 OVERRIDE: is_solution=True overridden to False. "
                        f"Semantic analysis: {semantic_source.value} (conf={semantic_conf:.2f})"
                    )
                    is_solution = False
                    solution_indicators = []
                    extracted_info["semantic_override"] = {
                        "original_is_solution": True,
                        "overridden_to": False,
                        "semantic_source": semantic_source.value,
                        "semantic_confidence": semantic_conf,
                        "reason": semantic_reason
                    }
                    # Also update intent to match semantic classification
                    if semantic_source == DataSource.STANDARDS_RAG:
                        intent = "standards"
                    elif semantic_source == DataSource.STRATEGY_RAG:
                        intent = "vendor_strategy"
                    else:
                        intent = "question"
                else:
                    logger.debug(f"[{self.name}] Phase 3 validation passed (Source: {semantic_source.value})")
                        
            except ImportError:
                logger.debug(f"[{self.name}] Semantic classifier not available for validation")
            except Exception as e:
                logger.warning(f"[{self.name}] Semantic validation failed: {e}")
        elif is_solution and has_strong_indicators:
             logger.info(f"[{self.name}] Phase 3 validation skipped (Strong solution indicators present)")

        # =================================================================
        # LEVEL 4.5: Registry-based workflow matching with fallback
        # =================================================================
        registry_match = None
        
        if self._use_registry and self._registry:
            try:
                # Use registry for intent matching
                registry_match = self._registry.match_intent(
                    intent=intent,
                    is_solution=is_solution,
                    confidence=confidence
                )
                
                if registry_match.workflow:
                    # Map registry workflow name to WorkflowTarget enum
                    workflow_name_to_target = {
                        "solution": WorkflowTarget.SOLUTION_WORKFLOW,
                        "instrument_identifier": WorkflowTarget.INSTRUMENT_IDENTIFIER,
                        "engenie_chat": WorkflowTarget.ENGENIE_CHAT
                    }
                    target_workflow = workflow_name_to_target.get(
                        registry_match.workflow.name, 
                        WorkflowTarget.ENGENIE_CHAT
                    )
                    logger.info(
                        f"[{self.name}] Registry match: {registry_match.workflow.name} "
                        f"(confidence={registry_match.confidence:.2f})"
                    )
                    
                    # Check if disambiguation is needed
                    if registry_match.needs_disambiguation:
                        logger.info(
                            f"[{self.name}] Low confidence match - disambiguation suggested: "
                            f"{registry_match.disambiguation_question}"
                        )
                        extracted_info["needs_disambiguation"] = True
                        extracted_info["disambiguation_question"] = registry_match.disambiguation_question
                else:
                    # No registry match - fall back to IntentConfig
                    logger.debug(f"[{self.name}] No registry match, using IntentConfig fallback")
                    target_workflow = IntentConfig.get_workflow(intent, is_solution)
                    
            except Exception as e:
                logger.warning(f"[{self.name}] Registry matching failed: {e}, using IntentConfig")
                target_workflow = IntentConfig.get_workflow(intent, is_solution)
        else:
            # FALLBACK: Use IntentConfig (Level 3 behavior)
            if not IntentConfig.is_known_intent(intent):
                logger.warning(
                    f"[{self.name}] Unknown intent '{intent}' from classifier. "
                    f"Valid intents: {IntentConfig.get_all_valid_intents()[:5]}... "
                    f"Mapping to 'unrelated'"
                )
                intent = "unrelated"
            target_workflow = IntentConfig.get_workflow(intent, is_solution)

        # Log the routing decision
        if is_solution or intent in IntentConfig.SOLUTION_FORCING_INTENTS:
            logger.info(
                f"[{self.name}] Solution detected: intent='{intent}', is_solution={is_solution}, "
                f"indicators={solution_indicators}"
            )

        # Determine Target RAG for Product Info
        target_rag = None
        if target_workflow == WorkflowTarget.ENGENIE_CHAT:
            if intent == "standards":
                target_rag = "standards_rag"
            elif intent == "vendor_strategy":
                target_rag = "strategy_rag"
            else:
                target_rag = "product_info_rag"

        # Override: Out of Domain Logic
        reject_message = None
        if target_workflow == WorkflowTarget.OUT_OF_DOMAIN:
            reject_message = "Invalid question. Please ask a domain-related question about industrial instrumentation, standards, or product selection."
            reasoning = f"Query classified as '{intent}' which is out of domain."
        else:
            reasoning = self._build_reasoning(intent, target_workflow, is_solution, solution_indicators)

        # =====================================================================
        # STEP 4: SET WORKFLOW STATE FOR SESSION
        # =====================================================================
        workflow_name = {
            WorkflowTarget.ENGENIE_CHAT: "engenie_chat",
            WorkflowTarget.INSTRUMENT_IDENTIFIER: "instrument_identifier",
            WorkflowTarget.SOLUTION_WORKFLOW: "solution"
        }.get(target_workflow)
        
        # CONSOLIDATION: Don't set backend state here
        # Frontend receives target_workflow in response and updates SessionManager
        # Frontend then passes workflow_hint on next request
        if workflow_name:
            logger.debug(f"[{self.name}] Target workflow: {workflow_name} (frontend will store)")

        # Prepare reject message for out-of-domain
        # reject_message logic handled above

        # Calculate classification time
        end_time = datetime.now()
        classification_time_ms = (end_time - start_time).total_seconds() * 1000
        self.last_classification_time_ms = classification_time_ms
        self.classification_count += 1

        result = WorkflowRoutingResult(
            query=query,
            target_workflow=target_workflow,
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            is_solution=is_solution,
            target_rag=target_rag,
            solution_indicators=solution_indicators,
            extracted_info=extracted_info,
            classification_time_ms=classification_time_ms,
            timestamp=datetime.now().isoformat(),
            reject_message=reject_message
        )

        logger.info(f"[{self.name}] Result: {target_workflow.value} (intent={intent}, conf={confidence:.2f}) in {classification_time_ms:.1f}ms")

        return result

    def _build_reasoning(
        self,
        intent: str,
        target_workflow: WorkflowTarget,
        is_solution: bool,
        solution_indicators: list
    ) -> str:
        """Build human-readable reasoning for the routing decision."""

        if target_workflow == WorkflowTarget.SOLUTION_WORKFLOW:
            if solution_indicators:
                return f"Solution detected: {', '.join(solution_indicators[:3])}"
            return "Complex system requiring multiple instruments detected"

        elif target_workflow == WorkflowTarget.INSTRUMENT_IDENTIFIER:
            return "Single product requirements detected"

        elif target_workflow == WorkflowTarget.ENGENIE_CHAT:
            if intent == "greeting":
                return "Greeting detected"
            elif intent == "confirm":
                return "User confirmation detected"
            elif intent == "reject":
                return "User rejection/cancellation detected"
            elif intent == "standards":
                return "Standards/compliance question detected"
            elif intent == "vendor_strategy":
                return "Vendor strategy question detected"
            return "Product knowledge question detected"

        elif target_workflow == WorkflowTarget.OUT_OF_DOMAIN:
            return f"Out of domain: '{intent}' is not related to industrial automation"

        return f"Classified as '{intent}'"

    def _create_error_result(self, query: str, start_time: datetime, error: str) -> WorkflowRoutingResult:
        """Create an error result."""
        end_time = datetime.now()
        classification_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return WorkflowRoutingResult(
            query=query,
            target_workflow=WorkflowTarget.OUT_OF_DOMAIN,
            intent="error",
            confidence=0.0,
            reasoning=f"Classification error: {error}",
            is_solution=False,
            target_rag=None,
            solution_indicators=[],
            extracted_info={},
            classification_time_ms=classification_time_ms,
            timestamp=datetime.now().isoformat(),
            reject_message=OUT_OF_DOMAIN_MESSAGE
        )

    def get_stats(self) -> Dict:
        """Get agent statistics."""
        stats = {
            "name": self.name,
            "classification_count": self.classification_count,
            "last_classification_time_ms": self.last_classification_time_ms,
            "using_registry": self._use_registry,
            "memory_stats": self._memory.get_stats()
        }
        
        # Add registry stats if available
        if self._use_registry and self._registry:
            stats["registry_stats"] = self._registry.get_stats()
        
        return stats


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def route_to_workflow(query: str, context: Optional[Dict] = None) -> WorkflowRoutingResult:
    """
    Convenience function for quick workflow routing.

    Args:
        query: User query string
        context: Optional context dict

    Returns:
        WorkflowRoutingResult
    """
    agent = IntentClassificationRoutingAgent()
    return agent.classify(query, context)


def get_workflow_target(query: str) -> str:
    """
    Get just the workflow target name.

    Args:
        query: User query string

    Returns:
        Workflow target value (e.g., "solution", "instrument_identifier")
    """
    result = route_to_workflow(query)
    return result.target_workflow.value


def is_valid_domain_query(query: str) -> bool:
    """
    Check if a query is within the valid domain.

    Args:
        query: User query string

    Returns:
        True if valid, False if out-of-domain
    """
    result = route_to_workflow(query)
    return result.target_workflow != WorkflowTarget.OUT_OF_DOMAIN


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'WorkflowTarget',
    'WorkflowRoutingResult',
    'IntentClassificationRoutingAgent',
    'WorkflowStateMemory',
    'get_workflow_memory',
    'should_exit_workflow',
    'route_to_workflow',
    'get_workflow_target',
    'is_valid_domain_query',
    'OUT_OF_DOMAIN_MESSAGE',
    'EXIT_PHRASES',
    'GREETING_PHRASES'
]
