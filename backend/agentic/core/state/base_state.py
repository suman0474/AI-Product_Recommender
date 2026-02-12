# agentic/base_state.py
# =============================================================================
# BASE RAG STATE - Shared TypedDict Fields for RAG Workflows
# =============================================================================
#
# This module provides base state definitions and factory functions for
# RAG workflow states. Common fields are defined here to ensure consistency
# across different RAG workflows:
#   - StandardsRAGState
#   - StrategyRAGState
#   - IndexRAGState
#
# Usage:
#   from agentic.core.state.base_state import BaseRAGStateFields, create_base_rag_state
#
#   class MyRAGState(TypedDict):
#       # Include all base fields
#       question: str
#       session_id: Optional[str]
#       # ... (copy from BaseRAGStateFields)
#
#       # Add domain-specific fields
#       my_custom_field: str
#
#   def create_my_rag_state(question: str, **kwargs) -> MyRAGState:
#       base = create_base_rag_state(question, **kwargs)
#       return MyRAGState(
#           **base,
#           my_custom_field=""
#       )
#
# =============================================================================

import time
from typing import Dict, Any, List, Optional, TypedDict


# =============================================================================
# BASE STATE FIELDS (for documentation and reference)
# =============================================================================

class BaseRAGStateFields(TypedDict, total=False):
    """
    Base fields shared across all RAG workflow states.

    This class documents the common fields. Subclass states should
    include these fields directly (TypedDict doesn't support inheritance well).

    Categories:
    - Input: User's question and session info
    - Processing: Intermediate processing data
    - Generation: LLM generation results
    - Validation: Response validation data
    - Output: Final response
    - Metadata: Timing and tracking
    """
    # =========================================================================
    # INPUT FIELDS
    # =========================================================================
    question: str                    # Original user question
    resolved_question: str           # After follow-up resolution
    is_follow_up: bool              # Whether this is a follow-up query
    session_id: Optional[str]       # Session identifier for memory
    top_k: int                      # Number of documents to retrieve

    # =========================================================================
    # PROCESSING FIELDS
    # =========================================================================
    question_valid: bool            # Whether question passed validation
    key_terms: List[str]            # Extracted key terms from question
    retrieved_docs: List[Dict]      # Retrieved documents from vector store
    context: str                    # Formatted context for LLM
    source_metadata: Dict           # Metadata about sources

    # =========================================================================
    # GENERATION FIELDS
    # =========================================================================
    answer: str                     # Generated answer
    citations: List[Dict]           # Citation information
    confidence: float               # Confidence score (0.0 - 1.0)
    sources_used: List[str]         # List of source names used
    generation_count: int           # Number of generation attempts

    # =========================================================================
    # VALIDATION FIELDS
    # =========================================================================
    validation_result: Optional[Dict]  # Detailed validation result
    is_valid: bool                     # Whether response passed validation
    retry_count: int                   # Current retry attempt
    max_retries: int                   # Maximum retry attempts
    validation_feedback: str           # Feedback for retry
    should_fail_fast: bool             # Skip retries for unrecoverable errors

    # =========================================================================
    # OUTPUT FIELDS
    # =========================================================================
    final_response: Dict            # Final structured response
    status: str                     # Workflow status (success/error/etc)
    error: Optional[str]            # Error message if failed

    # =========================================================================
    # METADATA FIELDS
    # =========================================================================
    start_time: float               # Workflow start timestamp
    processing_time_ms: int         # Total processing time in milliseconds


# =============================================================================
# STATE FACTORY FUNCTION
# =============================================================================

def create_base_rag_state(
    question: str,
    session_id: Optional[str] = None,
    top_k: int = 5,
    max_retries: int = 2
) -> Dict[str, Any]:
    """
    Create base state dictionary with common RAG fields.

    This function returns a dictionary (not TypedDict) that can be
    merged with domain-specific fields to create a complete state.

    Args:
        question: User's question
        session_id: Optional session identifier
        top_k: Number of documents to retrieve
        max_retries: Maximum validation retries

    Returns:
        Dictionary with all base RAG state fields initialized

    Example:
        def create_standards_rag_state(question: str, **kwargs) -> StandardsRAGState:
            base = create_base_rag_state(question, **kwargs)
            return StandardsRAGState(
                **base,
                standards_mentioned=[],  # domain-specific
                topics_discussed=[]      # domain-specific
            )
    """
    return {
        # Input
        "question": question,
        "resolved_question": question,
        "is_follow_up": False,
        "session_id": session_id or f"session-{int(time.time())}",
        "top_k": top_k,

        # Processing
        "question_valid": False,
        "key_terms": [],
        "retrieved_docs": [],
        "context": "",
        "source_metadata": {},

        # Generation
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "sources_used": [],
        "generation_count": 0,

        # Validation
        "validation_result": None,
        "is_valid": False,
        "retry_count": 0,
        "max_retries": max_retries,
        "validation_feedback": "",
        "should_fail_fast": False,

        # Output
        "final_response": {},
        "status": "",
        "error": None,

        # Metadata
        "start_time": time.time(),
        "processing_time_ms": 0
    }


def update_processing_time(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update the processing_time_ms field based on start_time.

    Args:
        state: State dictionary with start_time field

    Returns:
        Updated state with processing_time_ms calculated
    """
    start_time = state.get("start_time", time.time())
    state["processing_time_ms"] = int((time.time() - start_time) * 1000)
    return state


def build_final_response(
    state: Dict[str, Any],
    success: bool = True,
    additional_fields: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Build the final_response dictionary from state.

    Extracts common fields and adds status information.

    Args:
        state: Current state dictionary
        success: Whether workflow succeeded
        additional_fields: Extra fields to include

    Returns:
        Final response dictionary
    """
    response = {
        "success": success,
        "answer": state.get("answer", ""),
        "citations": state.get("citations", []),
        "confidence": state.get("confidence", 0.0),
        "sources_used": state.get("sources_used", []),
        "processing_time_ms": state.get("processing_time_ms", 0)
    }

    if state.get("error"):
        response["error"] = state["error"]

    if additional_fields:
        response.update(additional_fields)

    return response


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def should_retry(state: Dict[str, Any]) -> bool:
    """
    Check if workflow should retry based on state.

    Args:
        state: Current state dictionary

    Returns:
        True if retry should be attempted
    """
    if state.get("should_fail_fast", False):
        return False

    if state.get("is_valid", False):
        return False

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    return retry_count < max_retries


def increment_retry(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Increment retry count in state.

    Args:
        state: Current state dictionary

    Returns:
        Updated state with incremented retry_count
    """
    state["retry_count"] = state.get("retry_count", 0) + 1
    return state


def set_validation_result(
    state: Dict[str, Any],
    is_valid: bool,
    feedback: str = "",
    fail_fast: bool = False,
    validation_details: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Update validation fields in state.

    Args:
        state: Current state dictionary
        is_valid: Whether validation passed
        feedback: Validation feedback message
        fail_fast: Whether to skip retries
        validation_details: Detailed validation results

    Returns:
        Updated state with validation fields set
    """
    state["is_valid"] = is_valid
    state["validation_feedback"] = feedback
    state["should_fail_fast"] = fail_fast

    if validation_details:
        state["validation_result"] = validation_details

    return state


# =============================================================================
# STATE COPYING UTILITIES
# =============================================================================

def copy_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a shallow copy of state dictionary.

    For LangGraph compatibility (states should be immutable).

    Args:
        state: State dictionary to copy

    Returns:
        New dictionary with same values
    """
    return dict(state)


def merge_states(
    base_state: Dict[str, Any],
    updates: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge updates into base state (immutable).

    Args:
        base_state: Original state
        updates: Fields to update

    Returns:
        New state with updates applied
    """
    result = copy_state(base_state)
    result.update(updates)
    return result


# =============================================================================
# PHASE 2 FIX: STATE UPDATE/MERGE FUNCTIONS
# =============================================================================

def update_base_rag_state(
    existing_state: Dict[str, Any],
    **updates
) -> Dict[str, Any]:
    """
    Update existing state with new values - PRESERVES existing fields.

    PHASE 2 FIX: Use this instead of create_base_rag_state() when modifying state.
    This preserves workflow modifications instead of overwriting everything.

    Args:
        existing_state: Current state dictionary
        **updates: Fields to update

    Returns:
        Updated state with preserved fields

    Example:
        state = create_base_rag_state("question")
        state["resolved_question"] = "processed"
        state["answer"] = "partial answer"

        # Later, update confidence without losing previous fields:
        state = update_base_rag_state(state, confidence=0.9)
        # resolved_question and answer are still there!

    Raises:
        ValueError: If existing_state is None or empty
    """
    if not existing_state:
        raise ValueError("existing_state cannot be None or empty")

    # Create shallow copy to avoid mutating original
    updated_state = copy_state(existing_state)

    # Merge updates
    updated_state.update(updates)

    return updated_state


def merge_base_rag_state(
    base_state: Dict[str, Any],
    partial_state: Dict[str, Any],
    deep_merge_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Deep merge two state dictionaries - useful for merging workflow results.

    PHASE 2 FIX: Handles nested structures (dicts/lists) intelligently.

    Args:
        base_state: Base state dictionary
        partial_state: Partial updates to merge
        deep_merge_keys: Keys that should be deep-merged instead of replaced
                        (e.g., ['metadata', 'analysis_results'])

    Returns:
        Merged state with proper nesting

    Example:
        base = create_base_rag_state("question")
        base["metadata"] = {"source": "pinecone", "version": 1}

        partial = {
            "metadata": {"retrieved_docs": 5},  # Will merge, not replace
            "confidence": 0.85
        }

        merged = merge_base_rag_state(
            base,
            partial,
            deep_merge_keys=["metadata"]
        )
        # merged["metadata"] = {
        #     "source": "pinecone",
        #     "version": 1,
        #     "retrieved_docs": 5
        # }
    """
    if deep_merge_keys is None:
        deep_merge_keys = []

    merged = copy_state(base_state)

    for key, value in partial_state.items():
        if key in deep_merge_keys and key in merged:
            # Deep merge for specified keys
            if isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = {**merged[key], **value}
            elif isinstance(merged[key], list) and isinstance(value, list):
                # Extend lists instead of replacing
                merged[key] = merged[key] + value
            else:
                # Can't deep merge - replace
                merged[key] = value
        else:
            # Replace value normally
            merged[key] = value

    return merged


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    'BaseRAGStateFields',

    # Factory functions
    'create_base_rag_state',
    'build_final_response',

    # Update helpers
    'update_processing_time',
    'should_retry',
    'increment_retry',
    'set_validation_result',

    # Copy utilities
    'copy_state',
    'merge_states',

    # PHASE 2 FIX: New merge functions
    'update_base_rag_state',
    'merge_base_rag_state'
]
