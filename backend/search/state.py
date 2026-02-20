# search/state.py
# =============================================================================
# SEARCH DEEP AGENT STATE DEFINITION
# =============================================================================
#
# Centralized state TypedDict and factory function for the Search Deep Agent.
# Follows the Solution Deep Agent pattern with clean separation of concerns.
#
# Graph topology:
#     plan → validate → collect_requirements (HITL) → discover_params
#     → analyze_vendors → rank → respond
#
# =============================================================================

import time
import uuid
from typing import Dict, Any, List, Optional, TypedDict


class SearchDeepAgentState(TypedDict, total=False):
    """
    LangGraph state for the Search Deep Agent workflow.

    Organized into logical sections for clarity and maintenance.
    """

    # =========================================================================
    # SESSION & THREADING
    # =========================================================================
    session_id: str
    instance_id: str
    workflow_thread_id: str
    main_thread_id: Optional[str]
    parent_workflow_id: Optional[str]
    zone: Optional[str]

    # =========================================================================
    # INPUT CONFIGURATION
    # =========================================================================
    user_input: str
    expected_product_type: Optional[str]
    execution_mode: str                     # "auto" | "interactive"
    auto_mode: bool                         # Shorthand for execution_mode=="auto"
    enable_ppi: bool                        # Allow PPI schema generation
    skip_advanced_params: bool              # Skip advanced params discovery
    max_vendor_workers: int                 # ThreadPoolExecutor worker count

    # =========================================================================
    # PLANNING PHASE (Node 1)
    # =========================================================================
    execution_plan: Dict[str, Any]          # Full plan from SearchPlanner
    strategy: str                           # "fast" | "full" | "deep"
    quality_thresholds: Dict[str, int]      # Min scores for each phase
    phases_to_run: List[str]                # Ordered list of phases to execute

    # =========================================================================
    # VALIDATION PHASE (Node 2)
    # =========================================================================
    product_type: str
    original_product_type: str              # Before refinement
    product_type_refined: bool              # True if product_type was refined
    product_category: str                   # transmitter|analyzer|sensor|etc
    schema: Dict[str, Any]
    schema_source: str                      # "database" | "ppi" | "default"
    validation_result: Dict[str, Any]
    is_valid: bool
    missing_fields: List[str]               # Mandatory fields not provided
    optional_fields: List[str]              # Optional fields available
    provided_requirements: Dict[str, Any]   # Requirements extracted from input

    # =========================================================================
    # STANDARDS ENRICHMENT
    # =========================================================================
    standards_enrichment_applied: bool
    standards_info: Optional[Dict[str, Any]]
    enrichment_result: Optional[Dict[str, Any]]
    rag_invocations: Dict[str, Any]         # Tracking RAG calls
    has_safety_requirements: bool           # ATEX, SIL, IECEx detected

    # =========================================================================
    # HITL - COLLECT REQUIREMENTS PHASE (Node 3)
    # Immediate post-validation user interaction
    # =========================================================================
    hitl_phase: str                         # "awaitMissingInfo" | "awaitAdditionalSpecs" | "complete"
    missing_mandatory_fields: List[Dict[str, Any]]  # Detailed field info for user
    skipped_fields: List[str]               # Fields user chose to skip
    user_provided_values: Dict[str, Any]    # Values user provided for missing fields
    additional_specs_raw: str               # Free-text extra requirements from user
    additional_specs_parsed: Dict[str, Any] # LLM-parsed extra requirements
    additional_specs_collected: bool        # Flag for Phase 2 completion
    structured_requirements: Dict[str, Any] # Final merged requirements
    user_confirmed: bool                    # User has confirmed requirements
    awaiting_user_input: bool               # Workflow is paused for user
    hitl_response_type: str                 # "provide" | "skip" | "proceed_anyway"
    sales_agent_response: str               # Formatted prompt/response for user
    interrupt_reason: str                   # Reason for HITL interrupt

    # =========================================================================
    # ADVANCED PARAMETERS PHASE (Node 4)
    # =========================================================================
    advanced_params_result: Dict[str, Any]
    available_advanced_params: List[Dict[str, Any]]
    discovered_specifications: List[Dict[str, Any]]

    # =========================================================================
    # VENDOR ANALYSIS PHASE (Node 5)
    # =========================================================================
    vendor_analysis_result: Dict[str, Any]
    vendor_matches: List[Dict[str, Any]]
    strategy_context: Optional[Dict[str, Any]]
    vendors_analyzed: int
    original_vendor_count: int
    filtered_vendor_count: int
    excluded_by_strategy: List[str]

    # =========================================================================
    # RANKING PHASE (Node 6)
    # =========================================================================
    ranking_result: Dict[str, Any]
    overall_ranking: List[Dict[str, Any]]
    top_product: Optional[Dict[str, Any]]
    ranking_summary: str
    match_quality_score: float              # Quality score for matches
    judge_validation_score: int             # Judge validation result

    # =========================================================================
    # QUALITY TRACKING & RETRY
    # =========================================================================
    quality_flags: List[str]                # Quality issues detected
    relaxed_mode: bool                      # Running with relaxed thresholds
    retry_count: int                        # Number of retries attempted
    retry_history: List[Dict[str, Any]]     # History of retry attempts

    # =========================================================================
    # FINAL OUTPUT (Node 7)
    # =========================================================================
    analysis_result: Dict[str, Any]         # Full analysis for API response
    response_data: Dict[str, Any]           # UI-ready response (camelCase)
    success: bool

    # =========================================================================
    # WORKFLOW TRACKING
    # =========================================================================
    current_step: str                       # Current node name
    steps_completed: List[str]              # Completed step names
    messages: List[Dict[str, str]]          # System messages log
    error: Optional[str]                    # Error message if any
    error_type: Optional[str]               # Error type classification
    workflow_interrupted: bool              # Workflow is paused


def create_search_deep_agent_state(
    user_input: str,
    session_id: str = "default",
    expected_product_type: Optional[str] = None,
    execution_mode: Optional[str] = None,
    auto_mode: bool = True,
    enable_ppi: bool = True,
    skip_advanced_params: bool = False,
    max_vendor_workers: int = 10,
    instance_id: Optional[str] = None,
    workflow_thread_id: Optional[str] = None,
    main_thread_id: Optional[str] = None,
    parent_workflow_id: Optional[str] = None,
    zone: Optional[str] = None,
) -> SearchDeepAgentState:
    """
    Create initial state for the Search Deep Agent workflow.

    Args:
        user_input: Raw user requirement description
        session_id: Root session identifier
        expected_product_type: Hint for product type detection
        execution_mode: "auto" or "interactive" (HITL)
        auto_mode: Shorthand for execution_mode=="auto"
        enable_ppi: Allow PPI schema generation
        skip_advanced_params: Skip advanced params discovery
        max_vendor_workers: Max parallel workers for vendor analysis
        instance_id: Unique instance identifier (auto-generated if not provided)
        workflow_thread_id: LangGraph thread ID (auto-generated if not provided)
        main_thread_id: Parent thread ID (for nested workflows)
        parent_workflow_id: Parent workflow ID (for nested workflows)
        zone: Geographic zone for vendor filtering

    Returns:
        Initialized SearchDeepAgentState
    """
    # Determine execution mode
    resolved_mode = execution_mode or ("auto" if auto_mode else "interactive")

    return SearchDeepAgentState(
        # Session & Threading
        session_id=session_id,
        instance_id=instance_id or str(uuid.uuid4()),
        workflow_thread_id=workflow_thread_id or f"search_{session_id}_{int(time.time())}",
        main_thread_id=main_thread_id,
        parent_workflow_id=parent_workflow_id,
        zone=zone,

        # Input Configuration
        user_input=user_input,
        expected_product_type=expected_product_type,
        execution_mode=resolved_mode,
        auto_mode=(resolved_mode == "auto"),
        enable_ppi=enable_ppi,
        skip_advanced_params=skip_advanced_params,
        max_vendor_workers=max_vendor_workers,

        # Planning Phase
        execution_plan={},
        strategy="full",
        quality_thresholds={},
        phases_to_run=[],

        # Validation Phase
        product_type="",
        original_product_type="",
        product_type_refined=False,
        product_category="",
        schema={},
        schema_source="",
        validation_result={},
        is_valid=False,
        missing_fields=[],
        optional_fields=[],
        provided_requirements={},

        # Standards Enrichment
        standards_enrichment_applied=False,
        standards_info=None,
        enrichment_result=None,
        rag_invocations={},
        has_safety_requirements=False,

        # HITL - Collect Requirements
        hitl_phase="awaitMissingInfo",
        missing_mandatory_fields=[],
        skipped_fields=[],
        user_provided_values={},
        additional_specs_raw="",
        additional_specs_parsed={},
        additional_specs_collected=False,
        structured_requirements={},
        user_confirmed=False,
        awaiting_user_input=False,
        hitl_response_type="",
        sales_agent_response="",
        interrupt_reason="",

        # Advanced Parameters
        advanced_params_result={},
        available_advanced_params=[],
        discovered_specifications=[],

        # Vendor Analysis
        vendor_analysis_result={},
        vendor_matches=[],
        strategy_context=None,
        vendors_analyzed=0,
        original_vendor_count=0,
        filtered_vendor_count=0,
        excluded_by_strategy=[],

        # Ranking
        ranking_result={},
        overall_ranking=[],
        top_product=None,
        ranking_summary="",
        match_quality_score=0.0,
        judge_validation_score=0,

        # Quality Tracking
        quality_flags=[],
        relaxed_mode=False,
        retry_count=0,
        retry_history=[],

        # Final Output
        analysis_result={},
        response_data={},
        success=False,

        # Workflow Tracking
        current_step="plan",
        steps_completed=[],
        messages=[],
        error=None,
        error_type=None,
        workflow_interrupted=False,
    )


# =============================================================================
# STATE HELPER FUNCTIONS
# =============================================================================

def mark_step_complete(state: SearchDeepAgentState, step_name: str) -> None:
    """Mark a step as completed in the state."""
    if "steps_completed" not in state:
        state["steps_completed"] = []
    if step_name not in state["steps_completed"]:
        state["steps_completed"].append(step_name)


def add_system_message(
    state: SearchDeepAgentState,
    content: str,
    step: Optional[str] = None,
) -> None:
    """Add a system message to the state."""
    if "messages" not in state:
        state["messages"] = []
    state["messages"].append({
        "role": "system",
        "content": f"[{step or state.get('current_step', 'unknown')}] {content}",
    })


def set_error(
    state: SearchDeepAgentState,
    error: str,
    error_type: Optional[str] = None,
) -> None:
    """Set error state."""
    state["error"] = error
    state["error_type"] = error_type or "UnknownError"
    state["success"] = False


def is_hitl_interrupted(state: SearchDeepAgentState) -> bool:
    """Check if workflow is interrupted for HITL."""
    return state.get("workflow_interrupted", False) and state.get("awaiting_user_input", False)


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Alias for backward compatibility with existing code
ProductSearchDeepAgentState = SearchDeepAgentState
create_product_search_deep_agent_state = create_search_deep_agent_state
