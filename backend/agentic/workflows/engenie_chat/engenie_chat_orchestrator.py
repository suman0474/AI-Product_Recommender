"""
EnGenie Chat Orchestrator

Handles parallel query execution across multiple RAG sources
and merges results into unified response.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from .engenie_chat_intent_agent import DataSource, classify_query, get_sources_for_hybrid, classify_questions_batch
from .engenie_chat_memory import (
    get_session, add_to_history, is_follow_up_query,
    resolve_follow_up, set_context, get_context
)
from .engenie_chat_question_splitter import (
    split_questions, ParsedQuestion, MultiQuestionInput,
    extract_refinery, extract_product_type
)

# Import prompt loader for dynamic response generation
from prompts_library.prompt_loader import load_prompt

# Import new utility modules
try:
    from ..circuit_breaker import get_circuit_breaker, CircuitOpenError
    from ..rag_cache import get_rag_cache, cache_get, cache_set
    from ..rate_limiter import get_rate_limiter, acquire_for_service
    from ..fast_fail import should_fail_fast, check_and_set_fast_fail
    from ..rag_logger import set_trace_id, get_trace_id, OrchestratorLogger
    UTILITIES_AVAILABLE = True
except ImportError as e:
    UTILITIES_AVAILABLE = False

# Import AgenticConfig for timeouts
try:
    from config.agentic_config import AgenticConfig
    ORCHESTRATOR_TIMEOUT = AgenticConfig.ORCHESTRATOR_TIMEOUT_SECONDS
except ImportError:
    ORCHESTRATOR_TIMEOUT = 1200  # Default 20 minutes

logger = logging.getLogger(__name__)

# Thread pool for parallel execution
_executor = ThreadPoolExecutor(max_workers=4)


def generate_no_results_message(query: str, session_id: str) -> str:
    """
    Generate a personable, context-aware message when no products are found.
    
    Uses LLM with a specialized prompt to create natural, helpful responses
    instead of robotic error messages.
    
    Args:
        query: Original user query
        session_id: Session ID (unused but kept for consistency)
        
    Returns:
        Personable "no results" message
    """
    try:
        from services.llm.fallback import create_llm_with_fallback, invoke_with_retry_fallback
        
        # Load the fallback prompt
        prompt_template = load_prompt("fallback_prompts", section="NO_RESULTS_FOUND")
        prompt = prompt_template.format(query=query)
        
        # Use Flash model for speed (this is a simple generation task)
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.7,  # Higher temp for more natural responses
            skip_test=True
        )
        
        # Generate the response
        response = invoke_with_retry_fallback(
            llm,
            prompt,
            max_retries=2,
            fallback_to_openai=True,
            model="gemini-2.5-flash",
            temperature=0.7
        )
        
        message = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"[ORCHESTRATOR] Generated personable no-results message ({len(message)} chars)")
        return message.strip()
        
    except Exception as e:
        # Fallback to a simple but helpful static message if LLM fails
        logger.warning(f"[ORCHESTRATOR] Failed to generate dynamic no-results message: {e}")
        return (
            f"I couldn't find any products matching '{query}' in our database. "
            "Try checking the spelling or searching for a broader category."
        )


def query_index_rag(query: str, session_id: str) -> Dict[str, Any]:
    """Query Index RAG for product information."""
    try:
        from ..index_rag.index_rag_workflow import run_index_rag_workflow
        
        result = run_index_rag_workflow(
            question=query,
            session_id=session_id
        )
        
        # Extract answer from output.summary (the workflow returns final_response)
        output = result.get("output", {})
        answer = output.get("summary", "") or result.get("response", "")

        # If no answer, generate a message based on what we found
        if not answer or len(answer.strip()) == 0:
            products = output.get("recommended_products", [])
            if products:
                answer = f"Found {len(products)} products matching your query. Please review the results below."
            else:
                # Generate personable, context-aware "no results" message using LLM
                answer = generate_no_results_message(query, session_id)

        # Check if we found any products
        found_in_database = result.get("success", False) and bool(output.get("recommended_products", []))
        
        return {
            "success": True,
            "source": "index_rag",
            "answer": answer,
            "found_in_database": found_in_database,
            "data": result
        }
    except ImportError:
        logger.warning("[ORCHESTRATOR] Index RAG not available")
        return {"success": False, "source": "index_rag", "error": "Index RAG not available"}
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Index RAG error: {e}")
        return {"success": False, "source": "index_rag", "error": str(e)}


def query_standards_rag(query: str, session_id: str) -> Dict[str, Any]:
    """Query Standards RAG for standards information."""
    try:
        from ..standards_rag.standards_rag_workflow import run_standards_rag_workflow
        
        result = run_standards_rag_workflow(
            question=query,
            session_id=session_id
        )
        
        # Extract answer from final_response (the workflow returns full state)
        final_response = result.get("final_response", {})
        answer = final_response.get("answer", "") or result.get("answer", "")
        
        return {
            "success": True,
            "source": "standards_rag",
            "answer": answer,
            "standards_cited": final_response.get("citations", []),
            "data": result
        }
    except ImportError:
        logger.warning("[ORCHESTRATOR] Standards RAG not available")
        return {"success": False, "source": "standards_rag", "error": "Standards RAG not available"}
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Standards RAG error: {e}")
        return {"success": False, "source": "standards_rag", "error": str(e)}


def query_strategy_rag(
    query: str,
    session_id: str,
    refinery: Optional[str] = None,
    product_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query Strategy RAG for vendor strategy information.

    Args:
        query: User query text
        session_id: Session ID for memory
        refinery: Optional pre-extracted refinery name for filtering
        product_type: Optional pre-extracted product type

    Returns:
        Dict with success, answer, preferred_vendors, etc.
    """
    try:
        from ..strategy_rag.strategy_rag_workflow import run_strategy_rag_workflow

        # Build workflow parameters
        workflow_params = {
            "question": query,
            "session_id": session_id
        }

        # Add refinery if provided (enables refinery-specific filtering)
        if refinery:
            workflow_params["refinery"] = refinery
            logger.info(f"[ORCHESTRATOR] Strategy RAG with refinery filter: {refinery}")

        # Add product_type if provided
        if product_type:
            workflow_params["product_type"] = product_type

        result = run_strategy_rag_workflow(**workflow_params)

        # Extract answer from final_response
        final_response = result.get("final_response", {})
        answer = final_response.get("answer", "")

        # Build response with refinery info
        response = {
            "success": result.get("status") == "success",
            "source": "strategy_rag",
            "answer": answer,
            "preferred_vendors": result.get("preferred_vendors", []),
            "data": result
        }

        # Add refinery match info if available
        if refinery:
            response["refinery_filter"] = refinery
            response["refinery_matched"] = result.get("refinery_matched", False)

        return response

    except ImportError:
        logger.warning("[ORCHESTRATOR] Strategy RAG not available")
        return {"success": False, "source": "strategy_rag", "error": "Strategy RAG not available"}
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Strategy RAG error: {e}")
        return {"success": False, "source": "strategy_rag", "error": str(e)}


def query_deep_agent(query: str, session_id: str) -> Dict[str, Any]:
    """Query Deep Agent for detailed spec extraction."""
    try:
        from ..deep_agent.standards_deep_agent import run_standards_deep_agent
        
        result = run_standards_deep_agent(
            query=query,
            context={"session_id": session_id}
        )
        
        return {
            "success": True,
            "source": "deep_agent",
            "answer": result.get("response", ""),
            "extracted_specs": result.get("extracted_specs", {}),
            "data": result
        }
    except ImportError:
        logger.warning("[ORCHESTRATOR] Deep Agent not available")
        return {"success": False, "source": "deep_agent", "error": "Deep Agent not available"}
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Deep Agent error: {e}")
        return {"success": False, "source": "deep_agent", "error": str(e)}


def query_llm_fallback(query: str, session_id: str) -> Dict[str, Any]:
    """
    Use LLM directly for general questions.

    Uses create_llm_with_fallback for automatic key rotation and OpenAI fallback
    on RESOURCE_EXHAUSTED errors.
    """
    try:
        from services.llm.fallback import create_llm_with_fallback, invoke_with_retry_fallback

        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.3,
            skip_test=True
        )

        # Use retry wrapper with automatic key rotation and OpenAI fallback
        response = invoke_with_retry_fallback(
            llm,
            query,  # LLM accepts string directly
            max_retries=3,
            fallback_to_openai=True,
            model="gemini-2.5-flash",
            temperature=0.3
        )

        return {
            "success": True,
            "source": "llm",
            "answer": response.content if hasattr(response, 'content') else str(response),
            "data": {}
        }
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] LLM fallback error: {e}")
        return {
            "success": False,
            "source": "llm",
            "error": str(e),
            "answer": "I'm sorry, I couldn't process your request. Please try again."
        }


def query_sources_parallel(
    query: str,
    sources: List[DataSource],
    session_id: str
) -> Dict[str, Dict[str, Any]]:
    """
    Query multiple sources in parallel.
    
    Args:
        query: User query
        sources: List of DataSource to query
        session_id: Session ID for memory
        
    Returns:
        Dict mapping source name to result
    """
    source_funcs = {
        DataSource.INDEX_RAG: query_index_rag,
        DataSource.STANDARDS_RAG: query_standards_rag,
        DataSource.STRATEGY_RAG: query_strategy_rag,
        DataSource.DEEP_AGENT: query_deep_agent,
        DataSource.LLM: query_llm_fallback
    }
    
    results = {}
    futures = {}
    
    for source in sources:
        if source in source_funcs:
            future = _executor.submit(source_funcs[source], query, session_id)
            futures[future] = source
    
    # Use try/except to handle timeout gracefully and use partial results
    from concurrent.futures import TimeoutError as FuturesTimeoutError
    
    try:
        # Use configurable timeout (default 20 minutes = 1200 seconds)
        for future in as_completed(futures, timeout=ORCHESTRATOR_TIMEOUT):
            source = futures[future]
            try:
                result = future.result()
                results[source.value] = result
            except Exception as e:
                logger.error(f"[ORCHESTRATOR] Error querying {source.value}: {e}")
                results[source.value] = {
                    "success": False,
                    "source": source.value,
                    "error": str(e)
                }
    except FuturesTimeoutError:
        # Handle timeout gracefully - add error results for unfinished futures
        timeout_mins = ORCHESTRATOR_TIMEOUT // 60
        logger.warning(f"[ORCHESTRATOR] Timeout ({timeout_mins} min) waiting for sources, using partial results")
        for future, source in futures.items():
            if source.value not in results:
                if future.done():
                    try:
                        results[source.value] = future.result()
                    except Exception as e:
                        results[source.value] = {
                            "success": False,
                            "source": source.value,
                            "error": str(e)
                        }
                else:
                    # Future not done - add timeout error
                    results[source.value] = {
                        "success": False,
                        "source": source.value,
                        "error": f"Query timed out after {ORCHESTRATOR_TIMEOUT} seconds ({timeout_mins} min)"
                    }
                    future.cancel()
    
    return results


def merge_results(
    results: Dict[str, Dict[str, Any]],
    primary_source: DataSource
) -> Dict[str, Any]:
    """
    Merge results from multiple sources into unified response.
    
    Args:
        results: Dict of source results
        primary_source: The primary source to prioritize
        
    Returns:
        Merged response dict
    """
    merged = {
        "success": False,
        "answer": "",
        "source": "unknown",
        "found_in_database": False,
        "sources_used": [],
        "metadata": {}
    }
    
    answer_parts = []
    sources_used = []
    primary_has_incomplete_answer = False
    
    logger.info(f"[ORCHESTRATOR] Merging results from {len(results)} sources")
    
    # Patterns indicating the source doesn't have the full answer
    INCOMPLETE_PATTERNS = [
        "i don't have",
        "i do not have",
        "not available in",
        "no information about",
        "cannot find",
        "no specific information",
        "unable to find"
    ]
    
    # Process primary source first
    primary_key = primary_source.value
    if primary_key in results and results[primary_key].get("success"):
        primary_result = results[primary_key]
        primary_answer = primary_result.get("answer", "").strip()
        
        logger.info(f"[ORCHESTRATOR] Primary source '{primary_key}' answer length: {len(primary_answer)}")
        
        # Check if primary answer indicates incomplete information
        if primary_answer:
            answer_lower = primary_answer.lower()
            primary_has_incomplete_answer = any(pattern in answer_lower for pattern in INCOMPLETE_PATTERNS)
            
            if primary_has_incomplete_answer:
                logger.info(f"[ORCHESTRATOR] Primary source '{primary_key}' has incomplete answer, will prefer LLM")
            
            answer_parts.append(primary_answer)
            sources_used.append(primary_key)
            merged["found_in_database"] = primary_result.get("found_in_database", False)
        else:
            logger.warning(f"[ORCHESTRATOR] Primary source '{primary_key}' returned empty answer")
            primary_has_incomplete_answer = True
            sources_used.append(primary_key)  # Still record as used
        merged["metadata"][primary_key] = primary_result.get("data", {})
    else:
        logger.warning(f"[ORCHESTRATOR] Primary source '{primary_key}' not found or failed")
        primary_has_incomplete_answer = True
        if primary_key in results:
            logger.warning(f"[ORCHESTRATOR]   Error: {results[primary_key].get('error', 'Unknown')}")
    
    # Add supplementary information from other sources
    llm_answer = None
    for source_key, result in results.items():
        if source_key == primary_key:
            continue
        if result.get("success"):
            supp_answer = result.get("answer", "").strip()
            if supp_answer:
                if source_key == "llm":
                    llm_answer = supp_answer
                    logger.info(f"[ORCHESTRATOR] LLM answer available, length: {len(supp_answer)}")
                else:
                    answer_parts.append(supp_answer)
                    sources_used.append(source_key)
                    logger.info(f"[ORCHESTRATOR] Supplementary source '{source_key}' answer length: {len(supp_answer)}")
            merged["metadata"][source_key] = result.get("data", {})
    
    # If primary has incomplete answer and LLM has a full answer, prefer LLM
    if primary_has_incomplete_answer and llm_answer:
        # Replace the primary answer with LLM answer, but keep citations from primary
        logger.info(f"[ORCHESTRATOR] Using LLM answer as primary due to incomplete RAG answer")
        answer_parts = [llm_answer]  # Replace with LLM answer
        sources_used.append("llm")
    elif llm_answer and not primary_has_incomplete_answer:
        # Add LLM as supplementary only if it provides additional value
        # Check if LLM adds significantly different content
        if len(llm_answer) > 100:  # Only add if substantial
            answer_parts.append(f"\n\n**Additional Information:**\n{llm_answer}")
            sources_used.append("llm")
            logger.info(f"[ORCHESTRATOR] Added LLM as supplementary information")
    
    # Build final answer
    if answer_parts:
        merged["answer"] = "\n\n".join(answer_parts)
        merged["success"] = True
        merged["source"] = "database" if merged["found_in_database"] else "llm"
        logger.info(f"[ORCHESTRATOR] Merged {len(answer_parts)} answer(s), total length: {len(merged['answer'])}")
    else:
        # No successful results - use first error or default message
        merged["answer"] = "I couldn't find specific information about your query."
        logger.warning("[ORCHESTRATOR] No answers found from any source")
        for source_key, result in results.items():
            if result.get("error"):
                logger.warning(f"[ORCHESTRATOR]   {source_key} error: {result['error']}")
    
    merged["sources_used"] = sources_used
    
    return merged


def run_engenie_chat_query(
    query: str,
    session_id: str
) -> Dict[str, Any]:
    """
    Main entry point for EnGenie Chat queries.

    OPTIMIZED: Uses SEQUENTIAL LLM fallback instead of parallel to reduce API calls.
    LLM is only called if RAG returns incomplete/empty answer.

    Handles:
    1. Memory resolution (follow-up detection)
    2. Intent classification
    3. Source selection
    4. Sequential querying (RAG first, then LLM if needed)
    5. Result merging

    Args:
        query: User query
        session_id: Session ID for memory tracking

    Returns:
        Unified response dict
    """
    logger.info(f"[ORCHESTRATOR] Processing query: {query[:100]}...")


    # ============================================================================
    # PHASE 3: ORCHESTRATOR PRE-LLM CHECK - ENABLED
    # Block invalid/out-of-domain queries BEFORE expensive RAG+LLM operations
    # BUT still return a helpful, user-friendly response message
    # ============================================================================
    from agentic.validators import validate_query_domain
    
    logger.info(f"[ORCHESTRATOR] Pre-flight validation for session {session_id}")
    
    # Validate query domain (NEVER raises exceptions)
    validation_result = validate_query_domain(
        query=query,
        session_id=session_id,
        context={},
        use_fast_path=True  # Use fast-path optimization
    )
    
    # Block OUT_OF_DOMAIN at orchestrator level (BEFORE LLM to save costs)
    if not validation_result.is_valid:
        logger.info(
            f"[ORCHESTRATOR] ⚠️ OUT_OF_DOMAIN blocked (LLM NOT called): {query[:50]}... "
            f"(confidence: {validation_result.confidence:.2f}, reasoning: {validation_result.reasoning[:80]})"
        )
        
        # Return helpful response WITHOUT calling LLM
        # User sees a clear explanation, but we save LLM costs
        return {
            'success': True,  # ✅ Success (user gets a response)
            'answer': validation_result.reject_message or 
                "I'm EnGenie, your industrial automation assistant. I can help with:\n\n"
                "• **Instrument Identification** - Finding the right products for your needs\n\n"
                "• **Product Search** - Searching for specific industrial instruments\n\n"
                "• **Standards & Compliance** - Questions about industrial standards (ISA, IEC, etc.)\n\n"
                "• **Technical Knowledge** - Industrial automation concepts and best practices\n\n"
                "Please ask a question related to industrial automation, instrumentation, or process control.",
            'source': 'validation_blocked',  # Indicates it was blocked at validation
            'found_in_database': False,
            'awaiting_confirmation': False,
            'sources_used': [],
            'is_follow_up': False,
            'classification': {
                'primary_source': 'out_of_domain',
                'confidence': validation_result.confidence,
                'reasoning': validation_result.reasoning
            },
            'blocked_reason': 'out_of_domain',  # Added field for debugging
            'note': 'Query was blocked before LLM to prevent processing invalid input'
        }
    
    logger.info(
        f"[ORCHESTRATOR] ✅ Valid query - proceeding with LLM: {validation_result.target_workflow} "
        f"(confidence: {validation_result.confidence:.2f})"
    )

    # Step 1: Memory resolution
    is_follow_up = is_follow_up_query(query, session_id)
    resolved_query = query

    if is_follow_up:
        logger.info(f"[ORCHESTRATOR] Detected follow-up query")
        resolved_query = resolve_follow_up(query, session_id)

    # Step 2: Intent classification
    primary_source, confidence, reasoning = classify_query(resolved_query)
    logger.info(f"[ORCHESTRATOR] Classification: {primary_source.value} ({confidence:.2f}) - {reasoning}")

    # Step 3: Determine sources to query
    # OPTIMIZATION: Don't add LLM in parallel - use sequential fallback instead
    if primary_source == DataSource.HYBRID:
        sources = get_sources_for_hybrid(resolved_query)
        # Remove LLM from hybrid sources - will use as fallback if needed
        sources = [s for s in sources if s != DataSource.LLM]
    else:
        sources = [primary_source]

    # Don't add LLM as parallel source - it will be used as sequential fallback
    logger.info(f"[ORCHESTRATOR] Primary sources to query: {[s.value for s in sources]}")

    # Step 4: Query primary sources (without LLM)
    results = query_sources_parallel(resolved_query, sources, session_id)

    # Step 5: Check if we need LLM fallback
    # OPTIMIZATION: Only call LLM if RAG didn't provide a good answer
    needs_llm_fallback = False
    primary_answer = ""

    primary_key = primary_source.value
    if primary_key in results:
        primary_result = results[primary_key]
        if primary_result.get("success"):
            primary_answer = primary_result.get("answer", "").strip()

    # Patterns indicating incomplete answer that needs LLM help
    INCOMPLETE_PATTERNS = [
        "i don't have", "i do not have", "not available in",
        "no information about", "cannot find", "no specific information",
        "unable to find", "couldn't find"
    ]

    if not primary_answer:
        needs_llm_fallback = True
        logger.info(f"[ORCHESTRATOR] RAG returned empty answer - will use LLM fallback")
    elif len(primary_answer) < 50:
        needs_llm_fallback = True
        logger.info(f"[ORCHESTRATOR] RAG answer too short ({len(primary_answer)} chars) - will use LLM fallback")
    elif any(pattern in primary_answer.lower() for pattern in INCOMPLETE_PATTERNS):
        needs_llm_fallback = True
        logger.info(f"[ORCHESTRATOR] RAG answer incomplete - will use LLM fallback")
    elif confidence < 0.5:
        needs_llm_fallback = True
        logger.info(f"[ORCHESTRATOR] Low confidence ({confidence:.2f}) - will use LLM fallback")

    # Step 5b: Query LLM only if needed (SEQUENTIAL, not parallel)
    if needs_llm_fallback:
        logger.info(f"[ORCHESTRATOR] Running LLM fallback sequentially...")
        llm_result = query_llm_fallback(resolved_query, session_id)
        results[DataSource.LLM.value] = llm_result
    else:
        logger.info(f"[ORCHESTRATOR] RAG answer sufficient - skipping LLM (saved 1 API call)")

    # Step 6: Merge results
    merged = merge_results(results, primary_source)

    # Step 7: Save to memory
    add_to_history(
        session_id=session_id,
        query=query,
        response=merged.get("answer", ""),
        sources_used=merged.get("sources_used", [])
    )

    # Add metadata
    merged["is_follow_up"] = is_follow_up
    merged["classification"] = {
        "primary_source": primary_source.value,
        "confidence": confidence,
        "reasoning": reasoning
    }
    merged["llm_fallback_used"] = needs_llm_fallback

    return merged


# =============================================================================
# MULTI-QUESTION QUERY HANDLER
# =============================================================================

def run_multi_question_query(
    query: str,
    session_id: str
) -> Dict[str, Any]:
    """
    Handle multi-question inputs by splitting, classifying, and executing in parallel.

    This is the main entry point for multi-question queries. For single questions,
    delegates to run_engenie_chat_query() for backward compatibility.

    Args:
        query: User input (may contain multiple numbered questions)
        session_id: Session ID for memory tracking

    Returns:
        Dict with:
        - success: bool
        - is_multi_question: bool
        - question_count: int
        - answer: str (combined markdown answer)
        - results: List[Dict] (per-question results)
        - sources_used: List[str]
        - total_processing_time_ms: int
    """
    import time
    start_time = time.time()

    logger.info(f"[ORCHESTRATOR] Processing query (checking for multi-question)...")

    # Step 1: Split questions
    parsed = split_questions(query)

    # Step 2: If single question, delegate to existing handler (backward compatible)
    if not parsed.is_multi_question or parsed.question_count <= 1:
        logger.info("[ORCHESTRATOR] Single question detected, using standard handler")
        result = run_engenie_chat_query(query, session_id)
        result["is_multi_question"] = False
        result["question_count"] = 1
        return result

    logger.info(f"[ORCHESTRATOR] Multi-question detected: {parsed.question_count} questions")

    # Step 3: Classify all questions in batch
    classifications = classify_questions_batch(parsed.questions)

    # Step 4: Execute questions in parallel
    question_results = _execute_questions_parallel(classifications, session_id)

    # Step 5: Aggregate results
    aggregated = _aggregate_multi_question_results(
        question_results,
        parsed.original_input,
        start_time
    )

    # Step 6: Store in memory (store as single combined entry)
    combined_questions = " | ".join([q.cleaned_text for q in parsed.questions])
    add_to_history(
        session_id=session_id,
        query=combined_questions,
        response=aggregated.get("answer", ""),
        sources_used=aggregated.get("sources_used", [])
    )

    return aggregated


def _execute_questions_parallel(
    classifications: List,
    session_id: str,
    per_question_timeout: int = 60
) -> List[Dict[str, Any]]:
    """
    Execute classified questions in parallel.

    Args:
        classifications: List of (ParsedQuestion, DataSource, confidence, reasoning)
        session_id: Session ID for memory
        per_question_timeout: Timeout per question in seconds

    Returns:
        List of result dicts, one per question
    """
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    results = []
    futures = {}

    # Map DataSource to query function
    source_funcs = {
        DataSource.INDEX_RAG: lambda q, ctx: query_index_rag(q, session_id),
        DataSource.STANDARDS_RAG: lambda q, ctx: query_standards_rag(q, session_id),
        DataSource.STRATEGY_RAG: lambda q, ctx: query_strategy_rag(
            q, session_id,
            refinery=ctx.get("refinery"),
            product_type=ctx.get("product_type")
        ),
        DataSource.DEEP_AGENT: lambda q, ctx: query_deep_agent(q, session_id),
        DataSource.LLM: lambda q, ctx: query_llm_fallback(q, session_id),
    }

    # Submit all questions to thread pool
    for question, source, confidence, reasoning in classifications:
        context = getattr(question, 'extracted_context', {}) or {}
        query_text = getattr(question, 'cleaned_text', str(question))

        # Get the appropriate query function
        query_func = source_funcs.get(source, source_funcs[DataSource.LLM])

        # Submit to executor
        future = _executor.submit(query_func, query_text, context)
        futures[future] = {
            "question": question,
            "source": source,
            "confidence": confidence,
            "reasoning": reasoning,
            "context": context
        }

    # Collect results with timeout
    total_timeout = per_question_timeout * len(classifications)
    try:
        for future in as_completed(futures, timeout=min(total_timeout, 300)):
            meta = futures[future]
            question = meta["question"]
            source = meta["source"]
            context = meta["context"]

            try:
                rag_result = future.result(timeout=per_question_timeout)

                results.append({
                    "question_number": getattr(question, 'question_number', 0),
                    "question_text": getattr(question, 'cleaned_text', str(question)),
                    "source": source.value,
                    "answer": rag_result.get("answer", ""),
                    "success": rag_result.get("success", False),
                    "confidence": meta["confidence"],
                    "refinery": context.get("refinery"),
                    "product_type": context.get("product_type"),
                    "preferred_vendors": rag_result.get("preferred_vendors", []),
                    "error": rag_result.get("error")
                })

            except Exception as e:
                logger.error(f"[ORCHESTRATOR] Question {getattr(question, 'question_number', '?')} failed: {e}")
                results.append({
                    "question_number": getattr(question, 'question_number', 0),
                    "question_text": getattr(question, 'cleaned_text', str(question)),
                    "source": source.value,
                    "answer": "",
                    "success": False,
                    "confidence": 0,
                    "error": str(e)
                })

    except FuturesTimeoutError:
        logger.warning("[ORCHESTRATOR] Multi-question timeout, collecting partial results")
        # Add timeout errors for unfinished futures
        for future, meta in futures.items():
            if not future.done():
                question = meta["question"]
                results.append({
                    "question_number": getattr(question, 'question_number', 0),
                    "question_text": getattr(question, 'cleaned_text', str(question)),
                    "source": meta["source"].value,
                    "answer": "",
                    "success": False,
                    "error": "Question timed out"
                })

    # Sort by question number
    results.sort(key=lambda r: r.get("question_number", 0))

    return results


def _aggregate_multi_question_results(
    results: List[Dict[str, Any]],
    original_input: str,
    start_time: float
) -> Dict[str, Any]:
    """
    Aggregate multiple question results into unified response.

    Args:
        results: List of per-question results
        original_input: Original user input
        start_time: Processing start time

    Returns:
        Aggregated response dict
    """
    import time

    # Build combined markdown answer
    answer_parts = []
    sources_used = set()
    successful = 0
    failed = 0

    for result in results:
        q_num = result.get("question_number", 0)
        q_text = result.get("question_text", "")
        answer = result.get("answer", "")
        source = result.get("source", "unknown")
        refinery = result.get("refinery")
        success = result.get("success", False)
        error = result.get("error")

        sources_used.add(source)

        # Format question header
        header = f"## Question {q_num}: {q_text}"
        if refinery:
            header += f"\n*Refinery: {refinery}*"

        if success and answer:
            successful += 1
            answer_parts.append(f"{header}\n\n{answer}\n\n*Source: {source}*")
        else:
            failed += 1
            error_msg = error or "No answer available"
            answer_parts.append(f"{header}\n\n**Error:** {error_msg}\n\n*Source: {source}*")

    combined_answer = "\n\n---\n\n".join(answer_parts)

    # Calculate processing time
    processing_time_ms = int((time.time() - start_time) * 1000)

    logger.info(f"[ORCHESTRATOR] Multi-question complete: {successful}/{len(results)} successful in {processing_time_ms}ms")

    return {
        "success": successful > 0,
        "is_multi_question": True,
        "question_count": len(results),
        "answer": combined_answer,
        "results": results,
        "sources_used": list(sources_used),
        "total_processing_time_ms": processing_time_ms,
        "successful_count": successful,
        "failed_count": failed
    }


# =============================================================================
# BACKWARD COMPATIBILITY ALIAS
# =============================================================================




# =============================================================================
# WORKFLOW REGISTRATION (Level 4.5 + Level 5)
# =============================================================================

def _register_workflow():
    """Register this workflow with the central registry."""
    try:
        from ..workflow_registry import get_workflow_registry, WorkflowMetadata
        
        registry = get_workflow_registry()
        
        # Register as "engenie_chat" (Primary)
        registry.register(WorkflowMetadata(
            name="engenie_chat",
            display_name="EnGenie Chat",
            description="Conversational knowledge assistant for product information, standards, and vendor strategies. Routes to appropriate RAG sources and handles follow-up conversations with memory.",
            keywords=[
                "question", "what is", "tell me", "explain", "compare", "difference",
                "standard", "certification", "vendor", "supplier", "specification",
                "how does", "why", "when", "which", "datasheet", "model",
                "help", "information", "about", "chat", "conversational"
            ],
            intents=[
                "question", "greeting", "chitchat", "confirm", "reject",
                "standards", "vendor_strategy", "grounded_chat", "comparison"
            ],
            capabilities=[
                "rag_routing",
                "hybrid_sources",
                "follow_up_detection",
                "conversation_memory",
                "parallel_query",
                "llm_fallback",
                "streaming"
            ],
            entry_function=run_engenie_chat_query,
            priority=50,  # Primary workflow
            tags=["core", "knowledge", "rag", "chat", "conversational"],
            min_confidence_threshold=0.35  # Flexible
        ))
        logger.info("[EnGenieChatOrchestrator] Registered as 'engenie_chat' (primary)")
        

        
    except ImportError as e:
        logger.debug(f"[EnGenieChatOrchestrator] Registry not available: {e}")
    except Exception as e:
        logger.warning(f"[EnGenieChatOrchestrator] Failed to register: {e}")

# Register on module load
_register_workflow()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'run_engenie_chat_query',
    'query_index_rag',
    'query_standards_rag',
    'query_strategy_rag',
    'query_deep_agent',
    'query_llm_fallback',
    'query_sources_parallel',
    'merge_results'
]
