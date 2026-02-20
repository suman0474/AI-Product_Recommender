# solution_N/solution_deep_agent.py
# =============================================================================
# SOLUTION DEEP AGENT - Core State & Orchestration
# =============================================================================
#
# Unified Deep Agent state definition for the Solution Workflow.
# Replaces the tool-based SolutionState with a memory-centric,
# context-aware, parallel-processing deep agent architecture.
#
# =============================================================================

import time
import logging
from typing import Dict, Any, List, Optional, TypedDict

from common.agentic.deep_agent.memory import ParallelEnrichmentResult
from .orchestration import OrchestrationContext

logger = logging.getLogger(__name__)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class SolutionDeepAgentState(TypedDict, total=False):
    """
    LangGraph state for Solution Deep Agent.

    Extends the base deep agent pattern with:
    - Semantic intent classification
    - Conversation memory (rolling window)
    - Personal context integration
    - Flash personality orchestration
    - Identity-namespaced orchestration context for structural isolation
    """

    # ---- Session ----
    session_id: str
    user_input: str
    user_id: str

    # ---- Conversation Context ----
    conversation_history: List[Dict[str, str]]  # Rolling window of messages
    personal_context: Dict[str, Any]  # User preferences, saved configs
    active_thread_context: Dict[str, Any]  # Current thread entities

    # ---- Intent Classification ----
    intent_classification: Dict[str, Any]  # Full classification result
    is_solution_workflow: bool  # Whether input matches solution intent
    intent_confidence: float  # 0.0 - 1.0
    intent_method: str  # "semantic", "keyword", "contextual"

    # ---- Modification & Interaction ----
    is_modification: bool
    modification_diff: Dict[str, Any]  # Changes made during modification
    clarification_needed: bool
    clarification_questions: List[str]
    reset_confirmed: bool

    # ---- Flash Personality ----
    personality_plan: Dict[str, Any]  # Planner output
    execution_strategy: str  # "full", "fast", "deep"
    response_tone: str  # "professional", "conversational", "technical"

    # ---- Solution Analysis ----
    solution_analysis: Dict[str, Any]  # Domain, process, safety, parameters
    solution_name: str
    instrument_context: str  # Enriched context for identification

    # ---- Identification ----
    identified_instruments: List[Dict[str, Any]]
    identified_accessories: List[Dict[str, Any]]
    all_items: List[Dict[str, Any]]  # Unified list with numbering
    total_items: int

    # ---- Taxonomy Normalization (Orchestrator Layer) ----
    standardized_instruments: List[Dict[str, Any]]  # canonical names post-taxonomy
    standardized_accessories: List[Dict[str, Any]]  # canonical names post-taxonomy
    taxonomy_normalization_applied: bool             # observability flag

    # ---- Requirements ----
    provided_requirements: Dict[str, Any]  # Extracted from user input
    standards_detected: bool
    standards_confidence: float
    standards_indicators: List[str]

    # ---- Enrichment ----
    enrichment_results: Dict[str, ParallelEnrichmentResult]  # Per-item (isolated)
    user_specified_specs: Dict[str, Dict[str, Any]]  # Per-item user specs
    llm_generated_specs: Dict[str, Dict[str, Any]]  # Per-item LLM specs
    standards_specifications: Dict[str, Dict[str, Any]]  # Per-item standards specs
    crossover_validation: Dict[str, Any]  # Validation results

    # ---- Memory ----
    # NOTE: DeepAgentMemory is NOT stored in state (contains threading.Lock → not
    # msgpack-serializable by LangGraph MemorySaver). It is held in a module-level
    # session registry in workflow.py, keyed by session_id.
    memory_stats: Dict[str, Any]

    # ---- Thread Management ----
    workflow_thread_id: str
    main_thread_id: str

    # ---- Orchestration Context ----
    # Serialised OrchestrationContext (frozen dataclass → always a plain dict
    # in state so LangGraph MemorySaver can serialise it without issue).
    # Reconstructed at runtime via OrchestrationContext.from_dict().
    orchestration_ctx: Dict[str, Any]

    # ---- Output ----
    response: str
    response_data: Dict[str, Any]
    messages: List[Dict[str, str]]  # System messages for tracking

    # ---- Tool Audit ----
    tools_called: List[str]               # Ordered list of tool names invoked
    tool_results_summary: Dict[str, Any]  # Per-tool key results
    quality_flags: List[str]              # Quality warnings and issues

    # ---- Quality Scoring ----
    identification_quality_score: int     # 0-100, computed after identify_items
    enrichment_quality_score: int         # 0-100, computed after enrich_specs
    identification_retry_count: int       # Retry counter for identification

    # ---- Workflow Tracking ----
    current_phase: str
    phases_completed: List[str]
    error: Optional[str]

    # ---- Timing ----
    start_time: float
    processing_time_ms: int


# =============================================================================
# STATE FACTORY
# =============================================================================

def create_solution_deep_agent_state(
    user_input: str,
    session_id: str = "default",
    user_id: str = "",
    conversation_history: Optional[List[Dict[str, str]]] = None,
    personal_context: Optional[Dict[str, Any]] = None,
    workflow_thread_id: Optional[str] = None,
    main_thread_id: Optional[str] = None,
    zone: str = "DEFAULT",
) -> SolutionDeepAgentState:
    """
    Create initial state for Solution Deep Agent workflow.

    Args:
        user_input: User's solution description
        session_id: Session identifier
        user_id: User identifier for personal context
        conversation_history: Previous messages for context continuity
        personal_context: User preferences and saved configurations
        workflow_thread_id: Thread ID for this workflow instance
        main_thread_id: Parent thread ID for session tracking
        zone: Geographic zone for orchestration routing
    """
    root_ctx = OrchestrationContext.root(session_id=session_id, zone=zone)
    return SolutionDeepAgentState(
        # Session
        session_id=session_id,
        user_input=user_input,
        user_id=user_id,

        # Conversation Context
        conversation_history=conversation_history or [],
        personal_context=personal_context or {},
        active_thread_context={},

        # Intent
        intent_classification={},
        is_solution_workflow=False,
        intent_confidence=0.0,
        intent_method="pending",

        # Modification & Interaction
        is_modification=False,
        modification_diff={},
        clarification_needed=False,
        clarification_questions=[],
        reset_confirmed=False,

        # Flash Personality
        personality_plan={},
        execution_strategy="full",
        response_tone="professional",

        # Solution Analysis
        solution_analysis={},
        solution_name="",
        instrument_context="",

        # Identification
        identified_instruments=[],
        identified_accessories=[],
        all_items=[],
        total_items=0,

        # Taxonomy Normalization
        standardized_instruments=[],
        standardized_accessories=[],
        taxonomy_normalization_applied=False,

        # Requirements
        provided_requirements={},
        standards_detected=False,
        standards_confidence=0.0,
        standards_indicators=[],

        # Enrichment (per-item isolation)
        enrichment_results={},
        user_specified_specs={},
        llm_generated_specs={},
        standards_specifications={},
        crossover_validation={},

        # Memory (DeepAgentMemory lives in module-level registry, not in state)
        memory_stats={},

        # Thread Management
        workflow_thread_id=workflow_thread_id or f"solution_{session_id}_{int(time.time())}",
        main_thread_id=main_thread_id or session_id,

        # Orchestration Context (serialised — reconstructed at runtime)
        orchestration_ctx=root_ctx.to_dict(),

        # Output
        response="",
        response_data={},
        messages=[],

        # Tool Audit
        tools_called=[],
        tool_results_summary={},
        quality_flags=[],

        # Quality Scoring
        identification_quality_score=0,
        enrichment_quality_score=0,
        identification_retry_count=0,

        # Tracking
        current_phase="init",
        phases_completed=[],
        error=None,

        # Timing
        start_time=time.time(),
        processing_time_ms=0,
    )


# =============================================================================
# PHASE TRACKING HELPERS
# =============================================================================

def mark_phase_complete(state: SolutionDeepAgentState, phase: str) -> None:
    """Mark a phase as completed and update timing."""
    completed = state.get("phases_completed", [])
    if phase not in completed:
        completed.append(phase)
    state["phases_completed"] = completed

    elapsed = int((time.time() - state.get("start_time", time.time())) * 1000)
    state["processing_time_ms"] = elapsed
    logger.info(f"[SolutionDeepAgent] Phase '{phase}' complete ({elapsed}ms elapsed)")


def add_system_message(state: SolutionDeepAgentState, content: str) -> None:
    """Add a system message to the state for tracking."""
    messages = state.get("messages", [])
    messages.append({"role": "system", "content": content})
    state["messages"] = messages
