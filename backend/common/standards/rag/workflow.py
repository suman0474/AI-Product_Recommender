# agentic/standards_rag_workflow.py
# Standards RAG Workflow with Validation and Retry Logic

import json
import logging
from typing import Dict, Any, List, Literal, Optional, TypedDict
from datetime import datetime
import time

from langgraph.graph import StateGraph, END

from .chat_agent import StandardsChatAgent, create_standards_chat_agent
from common.agentic.agents.shared.shared_agents import ResponseValidatorAgent
from common.infrastructure.state.checkpointing.local import compile_with_checkpointing

import os
from common.services.llm.fallback import create_llm_with_fallback

# Import new utility modules for Phase 1 & 2 gap fixes
try:
    from ..fast_fail import should_fail_fast, check_and_set_fast_fail
    from ..rag_cache import cache_get, cache_set
    from ..rag_logger import StandardsRAGLogger, set_trace_id
    from ..circuit_breaker import get_circuit_breaker, CircuitOpenError
    UTILITIES_AVAILABLE = True
except ImportError:
    UTILITIES_AVAILABLE = False

import threading

logger = logging.getLogger(__name__)


# ============================================================================
# [FIX #5] CACHED LLM FOR VALIDATION
# ============================================================================
# Avoid creating a new LLM instance for every validate_response_node call.
# This eliminates repeated "Primary LLM wrapped with Xs timeout" log spam
# and reduces init overhead (~1-2s per call saved).

_cached_validation_llm = None
_cached_validation_llm_lock = threading.Lock()


def _get_cached_validation_llm():
    """
    Get or create a cached LLM instance for validation.

    [FIX #5] Singleton pattern to avoid per-call LLM initialization overhead.
    """
    global _cached_validation_llm

    if _cached_validation_llm is not None:
        return _cached_validation_llm

    with _cached_validation_llm_lock:
        # Double-check inside lock
        if _cached_validation_llm is not None:
            return _cached_validation_llm

        logger.info("[FIX #5] Creating cached validation LLM (singleton)...")
        _cached_validation_llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            timeout=60  # Use the new 60s default
        )
        return _cached_validation_llm


# ============================================================================
# WORKFLOW STATE
# ============================================================================

class StandardsRAGState(TypedDict):
    """State for the Standards RAG Workflow (with conversation memory)"""
    # Input
    question: str
    resolved_question: str  # After follow-up resolution (NEW)
    is_follow_up: bool  # NEW
    session_id: Optional[str]
    top_k: int
    source_filter: Optional[List[str]]  # Domain-routed document filter

    # Processing
    question_valid: bool
    key_terms: List[str]
    retrieved_docs: List[Dict]
    context: str
    source_metadata: Dict

    # Generation
    answer: str
    citations: List[Dict]
    confidence: float
    sources_used: List[str]
    generation_count: int

    # Validation
    validation_result: Optional[Dict]
    is_valid: bool
    retry_count: int
    max_retries: int
    validation_feedback: str
    should_fail_fast: bool  # [FIX #A3] Detect unrecoverable errors, skip retries

    # Output
    final_response: Dict
    status: str
    error: Optional[str]

    # Metadata
    start_time: float
    processing_time_ms: int


def create_standards_rag_state(
    question: str,
    session_id: Optional[str] = None,
    top_k: int = 5,
    source_filter: Optional[List[str]] = None
) -> StandardsRAGState:
    """Create initial state for Standards RAG workflow (with memory support)"""
    return StandardsRAGState(
        # Input
        question=question,
        resolved_question=question,  # Will be updated if follow-up
        is_follow_up=False,  # Will be updated by validate_question_node
        session_id=session_id or f"session-{int(time.time())}",
        top_k=top_k,
        source_filter=source_filter,  # Domain-routed document filter

        # Processing
        question_valid=False,
        key_terms=[],
        retrieved_docs=[],
        context="",
        source_metadata={},

        # Generation
        answer="",
        citations=[],
        confidence=0.0,
        sources_used=[],
        generation_count=0,

        # Validation
        validation_result=None,
        is_valid=False,
        retry_count=0,
        max_retries=2,
        validation_feedback="",
        should_fail_fast=False,  # [FIX #A3] Detect unrecoverable errors, skip retries

        # Output
        final_response={},
        status="",
        error=None,

        # Metadata
        start_time=time.time(),
        processing_time_ms=0
    )


# ============================================================================
# WORKFLOW NODES
# ============================================================================

def validate_question_node(state: StandardsRAGState) -> StandardsRAGState:
    """
    Node 1: Validate the user's question.

    Checks if question is valid, extracts key terms, and resolves follow-ups.
    """
    logger.info("[Node 1] Validating question and resolving follow-ups...")

    question = state['question'].strip()

    # Import memory module
    try:
        from .memory import (
            standards_rag_memory,
            resolve_standards_follow_up
        )
        
        # Check if this is a follow-up and resolve it
        session_id = state.get('session_id')
        if session_id:
            resolved = resolve_standards_follow_up(session_id, question)
            if resolved != question:
                state['is_follow_up'] = True
                state['resolved_question'] = resolved
                logger.info(f"Resolved follow-up: '{question}' -> '{resolved}'")
            else:
                state['is_follow_up'] = False
                state['resolved_question'] = question
        else:
            state['is_follow_up'] = False
            state['resolved_question'] = question
    except ImportError:
        logger.warning("Memory module not available, skipping follow-up resolution")
        state['is_follow_up'] = False
        state['resolved_question'] = question

    # Use resolved question for validation
    question = state['resolved_question']

    # Basic validation
    if not question:
        logger.warning("Empty question received")
        state['question_valid'] = False
        state['error'] = "Question cannot be empty"
        return state

    if len(question) < 5:
        logger.warning("Question too short")
        state['question_valid'] = False
        state['error'] = "Question is too short (minimum 5 characters)"
        return state

    if len(question) > 4000:
        logger.warning("Question too long")
        state['question_valid'] = False
        state['error'] = "Question is too long (maximum 4000 characters)"
        return state

    # Extract key terms (simple tokenization)
    # Remove common words and extract meaningful terms
    stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'for', 'in', 'on', 'at', 'to', 'of'}
    words = question.lower().split()
    key_terms = [w for w in words if w not in stop_words and len(w) > 2]

    state['question_valid'] = True
    state['key_terms'] = key_terms[:10]  # Limit to 10 key terms

    logger.info(f"Question valid. Key terms: {state['key_terms']}, Follow-up: {state['is_follow_up']}")

    return state


def retrieve_standards_node(state: StandardsRAGState) -> StandardsRAGState:
    """
    Node 2: Retrieve relevant standards documents from Pinecone.

    Queries vector store and builds context with production-ready retry logic.

    OPTIMIZATION: When source_filter is provided, queries only the specified
    documents instead of all 12+ standards documents (90% reduction in scope).
    """
    source_filter = state.get('source_filter')
    if source_filter:
        logger.info(f"[Node 2] Retrieving standards from {len(source_filter)} filtered docs: {source_filter}")
    else:
        logger.info("[Node 2] Retrieving standards documents from Pinecone (all docs)...")

    try:
        # Create agent to use its retrieval method
        agent = create_standards_chat_agent()

        # Determine if parallel processing should be used (2026-02-11 Enhancement)
        question = state['question']
        question_lower = question.lower()

        # Use parallel processing if:
        # 1. Complex query (more than 10 words)
        # 2. Contains certification keywords (SIL, ATEX, IEC, etc.)
        # 3. Product type is present in state
        # 4. Multiple domains likely needed
        use_parallel = (
            len(question.split()) > 10 or  # Complex query
            any(kw in question_lower for kw in ["sil", "certification", "requirements", "standards", "compliance", "atex", "iecex"]) or
            bool(state.get('product_type'))  # Product type available
        )

        if use_parallel and hasattr(agent, 'retrieve_documents_parallel'):
            logger.info("[Node 2] Using PARALLEL processing (Deep Agent 3-worker mode)")
            search_results = agent.retrieve_documents_parallel(
                question=question,
                product_type=state.get('product_type'),
                standards_mentioned=state.get('standards_mentioned', []),
                top_k=state['top_k']
            )
            # Store retrieval mode for diagnostics
            state['retrieval_mode'] = search_results.get('processing_mode', 'parallel')
        else:
            logger.info("[Node 2] Using SEQUENTIAL processing")
            # Retrieve documents with optional source filter
            search_results = agent.retrieve_documents(
                question=question,
                top_k=state['top_k'],
                source_filter=source_filter
            )
            state['retrieval_mode'] = 'sequential'

        # Store results
        state['retrieved_docs'] = search_results.get('results', [])
        state['context'] = agent.build_context(search_results)

        # Extract source metadata
        sources = {}
        for doc in state['retrieved_docs']:
            metadata = doc.get('metadata', {})
            filename = metadata.get('filename', 'unknown')
            if filename not in sources:
                sources[filename] = {
                    'standard_type': metadata.get('standard_type', ''),
                    'standards_references': metadata.get('standards_references', []),
                    'relevance': doc.get('relevance_score', 0.0)
                }

        state['source_metadata'] = sources

        logger.info(f"Retrieved {len(state['retrieved_docs'])} documents")
        logger.info(f"Unique sources: {list(sources.keys())}")

        # ╔════════════════════════════════════════════════════════════════════════╗
        # ║  [FIX #1] FAST-FAIL ON ZERO DOCUMENTS                                 ║
        # ║  When vector store returns 0 results, retrying will NEVER help        ║
        # ║  This is especially true when in mock_mode (Pinecone INIT_ERROR)      ║
        # ║  Saves 15-30 seconds per call (3 retries × 5-10s LLM validation each) ║
        # ╚════════════════════════════════════════════════════════════════════════╝
        if len(state['retrieved_docs']) == 0:
            is_mock_mode = search_results.get('mock_mode', False)
            mock_reason = search_results.get('mock_reason', 'unknown')

            if is_mock_mode:
                logger.warning(f"[FIX #1] 🔴 ZERO DOCS + MOCK MODE DETECTED - Fast-failing!")
                logger.warning(f"[FIX #1]    Mock Reason: {mock_reason}")
                logger.warning(f"[FIX #1]    Retries Prevented: {state['max_retries']} (saves 15-30+ seconds)")
                state['should_fail_fast'] = True
                state['error'] = f"Vector store in mock mode ({mock_reason}). No documents available."
            else:
                # Even without mock_mode, 0 docs means retrying won't help
                logger.warning("[FIX #1] ⚠️ ZERO DOCS RETRIEVED - Setting fast-fail (retries won't produce new docs)")
                state['should_fail_fast'] = True
                state['error'] = "No relevant standards documents found for this question"

    except Exception as e:
        logger.error(f"Error retrieving documents: {e}", exc_info=True)
        state['error'] = f"Error retrieving documents: {str(e)}"

        # ╔════════════════════════════════════════════════════════════════════════╗
        # ║  [FIX #A3] FAST-FAIL ON UNRECOVERABLE ERRORS                         ║
        # ║  Detect network/timeout errors that won't be fixed by retrying       ║
        # ║  Prevents 4 wasteful retry attempts (saves 60-80 seconds!)           ║
        # ╚════════════════════════════════════════════════════════════════════════╝
        error_str = str(e).lower()
        unrecoverable_errors = [
            "connection refused",      # Network connection failed
            "connection reset",         # Connection was reset
            "connection timeout",       # Timeout
            "timed out",               # General timeout
            "winerror 10060",          # Windows connection timeout
            "econnrefused",            # Connection refused (Unix)
            "ehostunreach",            # Host unreachable
            "enetunreach",             # Network unreachable
            "no such host",            # DNS failure
            "name or service not known", # DNS failure (Linux)
            "getaddrinfo failed",      # DNS failure
            "connection attempt failed",  # Generic connection failure
        ]

        if any(unrecoverable in error_str for unrecoverable in unrecoverable_errors):
            logger.warning(f"[FIX #A3] 🔴 UNRECOVERABLE ERROR DETECTED - Failing fast instead of retrying!")
            logger.warning(f"[FIX #A3]    Error: {str(e)}")
            logger.warning(f"[FIX #A3]    Retries Prevented: {state['max_retries']} × ~20s = {state['max_retries'] * 20}+ seconds saved!")
            state['should_fail_fast'] = True
            print(f"\n{'='*70}")
            print(f"🔴 [FIX #A3] FAST-FAIL TRIGGERED")
            print(f"   Error: {type(e).__name__}")
            print(f"   Retries Skipped: {state['max_retries']} (saves {state['max_retries'] * 20}+ seconds)")
            print(f"{'='*70}\n")

    return state


def generate_answer_node(state: StandardsRAGState) -> StandardsRAGState:
    """
    Node 3: Generate answer using StandardsChatAgent.

    Passes question and context to Google Gemini.

    CRITICAL FIX: Increment retry_count here (not in retry_decision_node)
    because conditional edge functions should not modify state.
    """
    # Increment retry_count if this is a retry (generation_count > 0 means we've already generated once)
    if state['generation_count'] > 0:
        state['retry_count'] = state.get('retry_count', 0) + 1
        logger.info(f"[Node 3] RETRY #{state['retry_count']} - Generating answer (attempt {state['generation_count'] + 1})...")
    else:
        logger.info(f"[Node 3] Generating answer (attempt {state['generation_count'] + 1})...")

    try:
        # Check if we have documents
        if len(state['retrieved_docs']) == 0:
            # ROOT CAUSE FIX: Provide diagnostic message with actual error context
            error_msg = state.get('error', '')
            logger.warning(
                f"[Node 3] No documents retrieved. Error: {error_msg}. "
                "Check logs for [VS-ROOT-CAUSE] messages. "
                "Possible causes: Pinecone not initialized, index empty, or API key missing."
            )

            # Build user-friendly error message with context (ENHANCED 2026-02-11)
            # Check retrieval mode for better diagnostics
            retrieval_mode = state.get('retrieval_mode', 'unknown')

            if "BlobNotFound" in error_msg or "does not exist" in error_msg or "404" in error_msg:
                # Azure Blob documents don't exist
                user_message = (
                    "❌ Standards documents are not available in Azure Blob Storage.\n\n"
                    f"**Retrieval Mode:** {retrieval_mode}\n\n"
                    "**What this means:**\n"
                    "- The standards documents have not been uploaded to Azure Blob Storage yet\n"
                    "- Expected location: `standards-documents` container in Azure Blob\n\n"
                    "**Recommendations:**\n"
                    "1. Upload the 11 standards documents to Azure Blob Storage\n"
                    "2. Verify Deep Agent fallback is configured (should happen automatically)\n"
                    "3. Contact your administrator if this persists\n\n"
                    f"**Technical details:** {error_msg[:200]}"
                )
            elif "fallback" in error_msg.lower() or "Deep Agent" in error_msg:
                # Using Deep Agent fallback
                user_message = (
                    "⚠️ Azure Blob unavailable, using local Deep Agent standards.\n\n"
                    "**Note:** Results may be limited compared to full Azure Blob integration.\n\n"
                    "**Recommendation:** Upload standards documents to Azure Blob Storage for optimal performance.\n"
                    "Contact your administrator to complete the migration."
                )
            elif "mock mode" in error_msg.lower():
                # Pinecone failed to initialize
                user_message = (
                    "🔴 Standards database unavailable (mock mode active).\n\n"
                    "**Possible causes:**\n"
                    "- PINECONE_API_KEY not configured or invalid\n"
                    "- Azure Blob Storage connection failed\n"
                    "- Index not properly initialized\n\n"
                    "**Please verify:**\n"
                    "- PINECONE_API_KEY is set in environment\n"
                    "- AZURE_STORAGE_CONNECTION_STRING is configured\n"
                    "- Standards documents are uploaded\n\n"
                    "Contact your administrator to resolve this issue."
                )
            elif "no relevant standards documents found" in error_msg.lower():
                # Documents exist but query returned 0 results
                user_message = (
                    "🔍 I couldn't find any standards documents relevant to your question.\n\n"
                    "**This could mean:**\n"
                    "- The standards database may not contain documents covering this specific topic\n"
                    "- Your question might need to be rephrased to better match indexed content\n\n"
                    "**Suggestions:**\n"
                    "- Try asking about specific standards (e.g., 'IEC 61508', 'SIL-2 requirements')\n"
                    "- Mention the product type explicitly (e.g., 'pressure transmitters', 'control valves')\n"
                    "- Use standard certification keywords (e.g., 'ATEX', 'SIL-3', 'functional safety')"
                )
            else:
                # Generic fallback with diagnostics
                user_message = (
                    "⚠️ I don't have access to relevant standards documents for this question.\n\n"
                    f"**Retrieval Mode:** {retrieval_mode}\n"
                    f"**Technical details:** {error_msg[:200]}\n\n"
                    "**Troubleshooting:**\n"
                    "- Verify Azure Blob Storage is accessible\n"
                    "- Check if standards documents are uploaded\n"
                    "- Ensure Deep Agent fallback is working\n\n"
                    "Please contact your administrator if this issue persists."
                )

            state['answer'] = user_message
            state['citations'] = []
            state['confidence'] = 0.0
            state['sources_used'] = []
            state['generation_count'] += 1
            return state

        # Create agent
        agent = create_standards_chat_agent(temperature=0.1)

        # Generate answer using resolved question for follow-up support
        # FIX: Use resolved_question instead of raw question for proper context
        result = agent.run(
            question=state['resolved_question'] or state['question'],
            top_k=state['top_k']
        )

        # Update state
        state['answer'] = result.get('answer', '')
        state['citations'] = result.get('citations', [])
        state['confidence'] = result.get('confidence', 0.0)
        state['sources_used'] = result.get('sources_used', [])
        state['generation_count'] += 1

        logger.info(f"Answer generated with confidence: {state['confidence']:.2f}")
        logger.info(f"Sources used: {state['sources_used']}")

        # Add validation feedback if this is a retry
        if state['retry_count'] > 0 and state['validation_feedback']:
            logger.info(f"Retry based on feedback: {state['validation_feedback']}")

    except Exception as e:
        logger.error(f"Error generating answer: {e}", exc_info=True)
        state['error'] = f"Error generating answer: {str(e)}"
        state['answer'] = "An error occurred while generating the answer."
        state['citations'] = []
        state['confidence'] = 0.0

    return state


def validate_response_node(state: StandardsRAGState) -> StandardsRAGState:
    """
    Node 4: Validate the generated response.

    Uses ResponseValidatorAgent to check quality.

    OPTIMIZATION: Skip validation LLM call when:
    - Confidence from generation is high (>= 0.8)
    - Answer is substantial (>= 200 chars)
    - This saves 1 LLM call per request when answer quality is good
    """
    logger.info("[Node 4] Validating response...")

    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║  [FIX #1] SKIP VALIDATION WHEN FAST-FAIL IS SET                       ║
    # ║  No point validating an empty-context answer - it will always fail    ║
    # ║  Saves 1 LLM call (5-10 seconds) when vector store returned 0 docs    ║
    # ╚════════════════════════════════════════════════════════════════════════╝
    if state.get('should_fail_fast', False):
        logger.info("[Node 4] SKIP validation - should_fail_fast is set (no docs, retries won't help)")
        state['is_valid'] = True  # Mark as valid to proceed to finalize
        state['validation_result'] = {
            'skipped': True,
            'reason': 'Fast-fail mode - no documents retrieved, validation skipped',
            'overall_score': 0.0
        }
        return state

    # =========================================================================
    # OPTIMIZATION: Skip validation if confidence is high and answer is good
    # This saves 1 LLM API call when the answer is clearly adequate
    # =========================================================================
    confidence = state.get('confidence', 0.0)
    answer = state.get('answer', '')
    answer_length = len(answer)

    # Skip validation if confidence is high and answer is substantial
    if confidence >= 0.8 and answer_length >= 200:
        logger.info(f"[Node 4] SKIP validation - high confidence ({confidence:.2f}) and good answer length ({answer_length} chars)")
        state['is_valid'] = True
        state['validation_result'] = {
            'skipped': True,
            'reason': 'High confidence and substantial answer',
            'confidence': confidence,
            'answer_length': answer_length,
            'overall_score': confidence  # Use confidence as proxy score
        }
        return state

    # Skip validation if this is a retry (already validated once)
    current_retry = state.get('current_retry', 0)
    if current_retry > 0 and confidence >= 0.6:
        logger.info(f"[Node 4] SKIP validation on retry - confidence adequate ({confidence:.2f})")
        state['is_valid'] = True
        state['validation_result'] = {
            'skipped': True,
            'reason': 'Retry with adequate confidence',
            'confidence': confidence
        }
        return state

    try:
        # [FIX #5] Use cached LLM to avoid per-call init overhead
        llm = _get_cached_validation_llm()
        validator = ResponseValidatorAgent(llm=llm)

        # Validate
        validation_result = validator.run(
            user_question=state['question'],
            response=state['answer'],
            context=state['context']
        )

        # Store validation result (it's already a dict)
        state['validation_result'] = validation_result
        state['is_valid'] = validation_result.get('is_valid', False)

        # Build feedback for retry
        if not state['is_valid']:
            issues = ", ".join(validation_result.get('issues_found', []))
            suggestions = validation_result.get('suggestions', '')
            state['validation_feedback'] = f"Issues: {issues}. {suggestions}"

            logger.warning(f"Validation failed: {state['validation_feedback']}")
        else:
            overall_score = validation_result.get('overall_score', 0)
            logger.info(f"Validation passed with score: {overall_score:.2f}")

    except Exception as e:
        logger.error(f"Error validating response: {e}", exc_info=True)
        # If validation fails, assume response is valid to avoid blocking
        state['is_valid'] = True
        state['validation_result'] = {'error': str(e)}

    return state


def retry_decision_node(state: StandardsRAGState) -> Literal["finalize", "generate_answer"]:
    """
    Node 5: Decide whether to retry or finalize.

    Routes to finalize if valid or max retries reached, else retry generation.

    CRITICAL FIX: This is a conditional edge function - it should NOT modify state.
    State modifications should happen in actual nodes, not conditional routers.
    The retry_count is now incremented in generate_answer_node instead.
    """
    logger.info("[Node 5] Making retry decision...")

    # Get current state values
    should_fail_fast = state.get('should_fail_fast', False)
    is_valid = state.get('is_valid', False)
    retry_count = state.get('retry_count', 0)
    max_retries = state.get('max_retries', 2)

    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║  [FIX #A3] FAST-FAIL SKIP RETRIES                                    ║
    # ║  If unrecoverable error detected, skip all retries                    ║
    # ║  Prevents 60-80+ seconds of wasteful retry attempts                  ║
    # ╚════════════════════════════════════════════════════════════════════════╝
    if should_fail_fast:
        logger.warning("[FIX #A3] 🔴 FAST-FAIL TRIGGERED - Skipping all retries!")
        return "finalize"

    # If valid, finalize
    if is_valid:
        logger.info("Response valid - proceeding to finalize")
        return "finalize"

    # CRITICAL: Check max retries BEFORE routing to retry
    # This prevents the recursion limit from being hit
    if retry_count >= max_retries:
        logger.warning(f"Max retries ({max_retries}) reached (count={retry_count}) - proceeding to finalize")
        return "finalize"

    # Route to retry (generate_answer_node will increment retry_count)
    logger.info(f"Will retry (current count={retry_count}, max={max_retries})")
    return "generate_answer"


def finalize_node(state: StandardsRAGState) -> StandardsRAGState:
    """
    Node 6: Finalize the response.

    Formats final output with metadata and stores in conversation memory.
    """
    logger.info("[Node 6] Finalizing response...")

    # Calculate processing time
    state['processing_time_ms'] = int((time.time() - state['start_time']) * 1000)

    # Build final response
    final_response = {
        'answer': state['answer'],
        'citations': state['citations'],
        'confidence': state['confidence'],
        'sources_used': state['sources_used'],
        'metadata': {
            'processing_time_ms': state['processing_time_ms'],
            'retry_count': state['retry_count'],
            'validation_score': state['validation_result'].get('overall_score', 0.0) if state['validation_result'] else 0.0,
            'documents_retrieved': len(state['retrieved_docs']),
            'hallucination_detected': state['validation_result'].get('hallucination_detected', False) if state['validation_result'] else False,
            'generation_attempts': state['generation_count'],
            'is_follow_up': state.get('is_follow_up', False),
            'resolved_question': state.get('resolved_question', state['question'])
        }
    }

    # Add warning if validation failed
    if not state['is_valid'] and state['retry_count'] >= state['max_retries']:
        final_response['metadata']['warning'] = "Response may not meet quality standards. Validation issues: " + ", ".join(
            state['validation_result'].get('issues_found', []) if state['validation_result'] else []
        )

    state['final_response'] = final_response
    state['status'] = 'success' if not state['error'] else 'error'

    # Store in conversation memory for future follow-ups
    try:
        from .standards_rag_memory import add_to_standards_memory
        
        session_id = state.get('session_id')
        if session_id and state['answer']:
            add_to_standards_memory(
                session_id=session_id,
                question=state['question'],
                answer=state['answer'],
                citations=state['citations'],
                key_terms=state.get('key_terms', [])
            )
            logger.info(f"Stored Q&A in memory for session: {session_id}")
    except ImportError:
        logger.debug("Memory module not available, skipping Q&A storage")
    except Exception as e:
        logger.warning(f"Failed to store in memory: {e}")

    logger.info(f"Workflow complete in {state['processing_time_ms']}ms")

    return state


# ============================================================================
# WORKFLOW GRAPH CONSTRUCTION
# ============================================================================

def create_standards_rag_workflow():
    """
    Create the Standards RAG workflow graph.

    Graph structure:
    START → validate_question → retrieve_standards → generate_answer
                                                           ↓
                                                    validate_response
                                                           ↓
                                                     retry_decision
                                                       ↙        ↘
                                                 finalize    → generate_answer (loop)
                                                    ↓
                                                   END
    """
    logger.info("Creating Standards RAG workflow...")

    # Create graph
    workflow = StateGraph(StandardsRAGState)

    # Add nodes
    workflow.add_node("validate_question", validate_question_node)
    workflow.add_node("retrieve_standards", retrieve_standards_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("validate_response", validate_response_node)
    workflow.add_node("finalize", finalize_node)

    # Set entry point
    workflow.set_entry_point("validate_question")

    # Add edges
    workflow.add_edge("validate_question", "retrieve_standards")
    workflow.add_edge("retrieve_standards", "generate_answer")
    workflow.add_edge("generate_answer", "validate_response")

    # Conditional edge from retry_decision
    workflow.add_conditional_edges(
        "validate_response",
        retry_decision_node,
        {
            "finalize": "finalize",
            "generate_answer": "generate_answer"
        }
    )

    # Finalize to END
    workflow.add_edge("finalize", END)

    # Compile
    app = compile_with_checkpointing(workflow)

    logger.info("Standards RAG workflow created successfully")

    return app


# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================

# Create workflow instance
_standards_rag_app = None


def get_standards_rag_workflow():
    """Get or create the Standards RAG workflow instance (singleton)."""
    global _standards_rag_app
    if _standards_rag_app is None:
        _standards_rag_app = create_standards_rag_workflow()
    return _standards_rag_app


def run_standards_rag_workflow(
    question: str,
    session_id: Optional[str] = None,
    top_k: int = 5,
    recursion_limit: int = 15,
    source_filter: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run the Standards RAG workflow.

    Args:
        question: User's question
        session_id: Optional session ID for checkpointing
        top_k: Number of documents to retrieve
        recursion_limit: Maximum recursion depth for workflow (default: 15)
            - Reduced from 50 to prevent infinite retry loops
            - With max_retries=2, worst case is: validate(1) + retrieve(1) +
              generate(1) + validate(1) + retry_decision(1) × 3 iterations = ~15 nodes
            - If this limit is hit, it indicates a bug in the workflow logic
        source_filter: Optional list of document filenames to filter to
                      (e.g., ["instrumentation_pressure_standards.docx"])
                      When provided, queries only these documents instead of all 12+

    Returns:
        Final result dictionary
    """
    if source_filter:
        logger.info(f"Running Standards RAG workflow for: {question[:80]}... (filtered to {len(source_filter)} docs)")
    else:
        logger.info(f"Running Standards RAG workflow for question: {question[:100]}...")

    # Create initial state with optional source filter
    initial_state = create_standards_rag_state(
        question=question,
        session_id=session_id,
        top_k=top_k,
        source_filter=source_filter
    )

    # Get workflow
    app = get_standards_rag_workflow()

    # Execute workflow
    try:
        # Run with config for checkpointing and increased recursion limit
        # FIX: Increase recursion limit from LangGraph default of 25 to 50
        # This fixes "Recursion limit of 25 reached without hitting a stop condition" errors
        config = {
            "configurable": {"thread_id": session_id or "default"},
            "recursion_limit": recursion_limit
        }
        final_state = app.invoke(initial_state, config=config)

        logger.info(f"Workflow completed with status: {final_state.get('status', 'unknown')}")

        return final_state

    except Exception as e:
        logger.error(f"Error executing workflow: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'final_response': {
                'answer': f"An error occurred: {str(e)}",
                'citations': [],
                'confidence': 0.0,
                'sources_used': [],
                'metadata': {}
            }
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test the workflow
    logger.info("="*80)
    logger.info("TESTING STANDARDS RAG WORKFLOW")
    logger.info("="*80)

    test_questions = [
        "What are the SIL2 requirements for pressure transmitters?",
        "What calibration standards apply to temperature sensors?",
        "What are the ATEX requirements for flow meters?"
    ]

    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n[Test {i}] Question: {question}")
        logger.info("-"*80)

        result = run_standards_rag_workflow(
            question=question,
            session_id=f"test-{i}",
            top_k=5
        )

        if result['status'] == 'success':
            response = result['final_response']
            logger.info(f"\nAnswer: {response['answer'][:200]}...")
            logger.info(f"\nConfidence: {response['confidence']:.2f}")
            logger.info(f"Sources: {', '.join(response['sources_used'])}")
            logger.info(f"Processing time: {response['metadata']['processing_time_ms']}ms")
            logger.info(f"Retries: {response['metadata']['retry_count']}")
            logger.info(f"Validation score: {response['metadata']['validation_score']:.2f}")

            if response.get('citations'):
                logger.info(f"\nTop citations:")
                for j, cite in enumerate(response['citations'][:2], 1):
                    logger.info(f"  [{j}] {cite['source']}")
        else:
            logger.error(f"\n[ERROR] {result.get('error', 'Unknown error')}")

        logger.info("="*80)
