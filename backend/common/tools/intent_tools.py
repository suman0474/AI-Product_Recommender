# tools/intent_tools.py
# Intent Classification and Requirements Extraction Tools

import json
import logging
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os
from common.services.llm.fallback import create_llm_with_fallback, invoke_with_retry_fallback

logger = logging.getLogger(__name__)


# ============================================================================
# TOOL INPUT SCHEMAS
# ============================================================================

class ClassifyIntentInput(BaseModel):
    """Input for intent classification"""
    user_input: str = Field(description="User's input message to classify")
    current_step: Optional[str] = Field(default=None, description="Current workflow step")
    context: Optional[str] = Field(default=None, description="Conversation context")


class ExtractRequirementsInput(BaseModel):
    """Input for requirements extraction"""
    user_input: str = Field(description="User's input containing requirements")


from common.prompts import INTENT_CLASSIFICATION_PROMPTS, INTENT_PROMPTS

# Load classification prompts (CLASSIFICATION + QUICK_CLASSIFICATION)
# Since the legacy loader used 'CLASSIFICATION' as default section, we map from 'DEFAULT' in new constants
INTENT_CLASSIFICATION_PROMPT = INTENT_CLASSIFICATION_PROMPTS.get("DEFAULT", "")
QUICK_CLASSIFICATION_PROMPT = INTENT_CLASSIFICATION_PROMPTS.get("QUICK_CLASSIFICATION", None)

# Load intent analysis prompts (REQUIREMENTS_EXTRACTION)
# INTENT_PROMPTS is a string containing the prompt content
REQUIREMENTS_EXTRACTION_PROMPT = INTENT_PROMPTS


# ============================================================================
# RULE-BASED CLASSIFICATION (OPTIMIZATION - avoid LLM for obvious intents)
# ============================================================================
# These patterns are checked BEFORE calling the LLM to save API calls.
# All callers of classify_intent_tool benefit automatically.

_GREETING_PHRASES = {'hi', 'hello', 'hey', 'hi there', 'hello there',
                     'good morning', 'good afternoon', 'good evening'}

_CONFIRM_PHRASES = {'yes', 'y', 'yep', 'yeah', 'sure', 'ok', 'okay',
                    'proceed', 'continue', 'go ahead', 'confirm', 'approved'}

_REJECT_PHRASES = {'no', 'n', 'nope', 'cancel', 'stop', 'reject', 'decline',
                   'never mind', 'nevermind', 'forget it'}

_KNOWLEDGE_STARTERS = [
    'what is ', 'what are ', 'what does ', "what's ",
    'how does ', 'how do ', 'how to ', 'how can ',
    'why is ', 'why does ', 'why do ', 'why are ',
    'tell me about ', 'explain ', 'define ', 'describe ',
    'difference between ', 'compare ', 'meaning of ', 'definition of '
]

_PRODUCT_REQUEST_STARTERS = [
    'i need a ', 'i need an ', 'looking for a ', 'looking for an ',
    'i want a ', 'i want an ', 'need a ', 'need an ',
    'find me a ', 'find a ', 'get me a ', 'show me ',
    'recommend a ', 'suggest a '
]


def _classify_rule_based(user_input: str) -> Optional[Dict[str, Any]]:
    """
    Attempt rule-based intent classification without calling the LLM.

    Returns a result dict if a rule matches, or None if LLM is needed.
    This saves 1 LLM API call for ~40-60% of common queries.
    """
    query = user_input.lower().strip()

    # Pure greeting
    if query in _GREETING_PHRASES:
        logger.info(f"[INTENT_RULE] Greeting detected: '{query}'")
        return {
            "success": True,
            "intent": "greeting",
            "confidence": 1.0,
            "next_step": "greeting",
            "extracted_info": {"rule_based": True},
            "is_solution": False,
            "solution_indicators": []
        }

    # Confirm/Reject
    if query in _CONFIRM_PHRASES:
        logger.info(f"[INTENT_RULE] Confirm detected: '{query}'")
        return {
            "success": True,
            "intent": "confirm",
            "confidence": 1.0,
            "next_step": None,
            "extracted_info": {"rule_based": True},
            "is_solution": False,
            "solution_indicators": []
        }

    if query in _REJECT_PHRASES:
        logger.info(f"[INTENT_RULE] Reject detected: '{query}'")
        return {
            "success": True,
            "intent": "reject",
            "confidence": 1.0,
            "next_step": None,
            "extracted_info": {"rule_based": True},
            "is_solution": False,
            "solution_indicators": []
        }

    # Knowledge questions
    if any(query.startswith(p) for p in _KNOWLEDGE_STARTERS):
        logger.info(f"[INTENT_RULE] Knowledge question detected: '{query[:50]}'")
        return {
            "success": True,
            "intent": "chat",  # New 4-intent architecture
            "confidence": 0.95,
            "next_step": None,
            "extracted_info": {"rule_based": True},
            "is_solution": False,
            "key_indicators": ["knowledge_question"],
            "reasoning": "Detected knowledge/educational query about industrial topic"
        }

    # Product requests - need to distinguish complex systems from simple requests (PHASE 3 FIX)
    if any(query.startswith(p) for p in _PRODUCT_REQUEST_STARTERS):
        # PHASE 3 FIX: Knowledge question indicators - these override solution detection
        # If query is asking ABOUT systems (knowledge), not building them
        knowledge_indicators = [
            'what is', 'what are', 'how does', 'how do', 'explain',
            'tell me about', 'describe', 'difference between', 'compare',
            'meaning of', 'definition of', 'what\'s the', 'can you explain'
        ]
        is_knowledge_query = any(kw in query for kw in knowledge_indicators)
        
        # PHASE 3 FIX: Stricter solution phrases - require ACTION verbs for building/designing
        # These must indicate the user is BUILDING something, not asking about it
        solution_action_phrases = [
            'i\'m designing', 'i\'m building', 'i\'m implementing',
            'i\'m creating', 'i\'m planning', 'i\'m developing',
            'design a system', 'build a system', 'create a system',
            'implement a solution', 'develop a solution',
            'need to design', 'need to build', 'need to implement',
            'planning to build', 'planning to design', 'planning to create'
        ]
        is_solution_action = any(phrase in query for phrase in solution_action_phrases)
        
        # PHASE 3 FIX: Complex system phrases (multi-instrument requirement)
        complex_system_phrases = [
            'multiple instruments', 'multiple transmitters', 'multiple sensors',
            'instrumentation package', 'full instrumentation', 'complete instrumentation',
            'monitoring system for', 'control system for', 'measurement system for',
            'plant instrumentation', 'process instrumentation', 'skid package'
        ]
        is_complex_system = any(phrase in query for phrase in complex_system_phrases)
        
        # PHASE 3 FIX: Semantic validation - use engenie_chat classifier
        # ONLY used to detect knowledge queries that should NOT go to solution
        # NOT used to override simple requirements to question
        is_rag_knowledge_query = False
        if is_knowledge_query:
            # Already a knowledge query based on keywords
            is_rag_knowledge_query = True
        elif is_solution_action or is_complex_system:
            # User wants to BUILD something, but might actually be asking about it
            # Use semantic classifier to double-check
            try:
                from chat.engenie_chat_intent_agent import classify_query, DataSource
                data_source, confidence, _ = classify_query(query, use_semantic_llm=False)
                # Standards and Strategy RAG indicate knowledge/info queries
                rag_knowledge_sources = {DataSource.STANDARDS_RAG, DataSource.STRATEGY_RAG}
                if data_source in rag_knowledge_sources and confidence >= 0.8:
                    is_rag_knowledge_query = True
                    logger.info(f"[INTENT_RULE] Semantic check: Knowledge RAG detected ({data_source.value}, conf={confidence:.2f})")
            except ImportError:
                logger.debug("[INTENT_RULE] Semantic classifier not available, using rule-based only")
            except Exception as e:
                logger.debug(f"[INTENT_RULE] Semantic check failed: {e}")
        
        # PHASE 3 FIX: Decision logic with semantic awareness
        # 1. Knowledge queries → EnGenie Chat (question)
        if is_knowledge_query or is_rag_knowledge_query:
            logger.info(f"[INTENT_RULE] Knowledge query detected: '{query[:50]}' (knowledge={is_knowledge_query}, rag_knowledge={is_rag_knowledge_query})")
            return {
                "success": True,
                "intent": "chat",  # New 4-intent architecture
                "confidence": 0.9,
                "next_step": None,
                "extracted_info": {
                    "rule_based": True,
                    "is_knowledge_query": is_knowledge_query,
                    "is_rag_knowledge_query": is_rag_knowledge_query
                },
                "is_solution": False,
                "key_indicators": ["knowledge_query"],
                "reasoning": "Detected educational/informational query"
            }
        
        # Solution action phrase OR complex system → Solution workflow
        if is_solution_action or is_complex_system:
            matched_indicators = []
            if is_solution_action:
                matched_indicators.append("solution_action_phrase")
            if is_complex_system:
                matched_indicators.append("complex_system_phrase")
            
            logger.info(
                f"[INTENT_RULE] Solution request detected: '{query[:50]}' "
                f"(action={is_solution_action}, complex={is_complex_system})"
            )
            return {
                "success": True,
                "intent": "solution",
                "confidence": 0.85,
                "next_step": None,
                "extracted_info": {
                    "rule_based": True,
                    "is_solution_action": is_solution_action,
                    "is_complex_system": is_complex_system
                },
                "is_solution": True,
                "solution_indicators": matched_indicators
            }
        
        # Default: Simple product request with specs → Search workflow
        logger.info(f"[INTENT_RULE] Simple product request detected: '{query[:50]}'")
        return {
            "success": True,
            "intent": "search",  # New 4-intent architecture (was 'requirements')
            "confidence": 0.9,
            "next_step": None,
            "extracted_info": {"rule_based": True},
            "is_solution": False,
            "key_indicators": ["product_request"],
            "reasoning": "Detected single product request with specifications"
        }

    # PHASE 3 FIX: Solution design phrases that don't start with product request starters
    # These are explicit "I'm building/designing" statements
    solution_design_starters = [
        "i'm designing", "i'm building", "i'm implementing",
        "i'm creating", "i'm planning", "i'm developing",
        "we're designing", "we're building", "we're implementing",
        "we are designing", "we are building", "we are implementing"
    ]
    if any(query.startswith(s) for s in solution_design_starters):
        logger.info(f"[INTENT_RULE] Solution design phrase detected: '{query[:50]}'")
        return {
            "success": True,
            "intent": "solution",
            "confidence": 0.9,
            "next_step": None,
            "extracted_info": {"rule_based": True, "is_design_statement": True},
            "is_solution": True,
            "solution_indicators": ["design_statement"]
        }

    # No rule matched - need LLM
    return None


# ============================================================================
# LLM-BASED FAST CLASSIFICATION (REPLACES RULE-BASED FOR SIMPLE INTENTS)
# ============================================================================
# Uses LLM with temperature=0.0 for deterministic classification of simple intents.
# This ensures consistent, intelligent classification while still being fast.

def _classify_llm_fast(user_input: str) -> Optional[Dict[str, Any]]:
    """
    LLM-based fast classification for simple intents using temperature 0.0.
    
    This replaces the rule-based classification with an LLM call that can
    intelligently handle greetings, confirmations, rejections, and exit phrases.
    
    Returns:
        - Result dict for greeting/confirm/reject/exit intents
        - None if classified as 'unknown' (requires full LLM classification)
    """
    # Check if prompt is available
    if not QUICK_CLASSIFICATION_PROMPT:
        logger.warning("[INTENT_LLM_FAST] QUICK_CLASSIFICATION prompt not available, falling back to rule-based")
        return _classify_rule_based(user_input)
    
    try:
        # Create LLM with temperature=0.0 for deterministic output
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.0,  # Deterministic output
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        prompt = ChatPromptTemplate.from_template(QUICK_CLASSIFICATION_PROMPT)
        parser = JsonOutputParser()
        
        chain = prompt | llm | parser
        
        # Use retry wrapper for reliability
        result = invoke_with_retry_fallback(
            chain,
            {"user_input": user_input},
            max_retries=2,  # Fewer retries for speed
            fallback_to_openai=True,
            model="gemini-2.5-flash",
            temperature=0.0
        )
        
        intent = result.get("intent", "unknown")
        confidence = result.get("confidence", 0.5)
        
        logger.info(f"[INTENT_LLM_FAST] Classified '{user_input[:40]}...' as '{intent}' (conf={confidence:.2f})")
        
        # If unknown, return None to trigger full classification
        if intent == "unknown":
            logger.debug("[INTENT_LLM_FAST] Unknown intent, deferring to full classification")
            return None
        
        # Map LLM result to standard response format
        intent_mapping = {
            "greeting": {
                "intent": "greeting",
                "next_step": "greeting",
                "is_solution": False,
                "solution_indicators": []
            },
            "confirm": {
                "intent": "confirm",
                "next_step": None,
                "is_solution": False,
                "solution_indicators": []
            },
            "reject": {
                "intent": "reject",
                "next_step": None,
                "is_solution": False,
                "solution_indicators": []
            },
            "exit": {
                "intent": "exit",
                "next_step": "reset",
                "is_solution": False,
                "solution_indicators": []
            }
        }
        
        if intent in intent_mapping:
            return {
                "success": True,
                "intent": intent_mapping[intent]["intent"],
                "confidence": confidence,
                "next_step": intent_mapping[intent]["next_step"],
                "extracted_info": {"llm_fast_classification": True, "temperature": 0.0},
                "is_solution": intent_mapping[intent]["is_solution"],
                "solution_indicators": intent_mapping[intent]["solution_indicators"]
            }
        
        # Unexpected intent value - defer to full classification
        logger.warning(f"[INTENT_LLM_FAST] Unexpected intent '{intent}', deferring to full classification")
        return None
        
    except Exception as e:
        logger.warning(f"[INTENT_LLM_FAST] LLM fast classification failed: {e}, falling back to rule-based")
        # Fallback to rule-based on LLM failure
        return _classify_rule_based(user_input)


# ============================================================================
# TOOLS
# ============================================================================

@tool("classify_intent", args_schema=ClassifyIntentInput)
def classify_intent_tool(
    user_input: str,
    current_step: Optional[str] = None,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Classify user intent for routing in the procurement workflow.
    Returns intent type, confidence, and suggested next step.

    APPROACH: Uses LLM-based classification with temperature 0.0 for ALL intents,
    ensuring consistent and intelligent classification for greetings, confirms,
    rejects, and exits. Falls back to full LLM classification for complex queries.
    """
    # =========================================================================
    # STEP 0: Pre-LLM Input Validation (reject invalid queries immediately)
    # =========================================================================
    # RECURSION FIX: Skip validation if we're already in a classification context
    # This prevents circular calls:
    # classify_intent_tool → validate_query_domain → route_to_workflow →
    # IntentClassificationRoutingAgent.classify → classify_intent_tool (recursion!)
    #
    # When called from IntentClassificationRoutingAgent, current_step is typically set
    # or context indicates we're already doing classification
    is_in_classification_context = (
        current_step is not None or
        (context and "classification" in str(context).lower())
    )

    if not is_in_classification_context:
        from common.validators import validate_query_domain

        validation_result = validate_query_domain(user_input, use_fast_path=True)
        if not validation_result.is_valid:
            logger.warning(f"[INTENT_CLASSIFY] Query rejected at validation: '{user_input[:60]}...'")
            return {
                "success": False,
                "intent": "INVALID_INPUT",
                "confidence": validation_result.confidence,
                "reasoning": validation_result.reject_message or "Query is outside system scope",
                "key_indicators": ["out_of_domain"],
                "rejected": True,
                "next_step": None,
                "extracted_info": {},
                "is_solution": False
            }
    else:
        logger.debug(f"[INTENT_CLASSIFY] Skipping validation (already in classification context)")
    
    # =========================================================================
    # STEP 1: Try LLM-based fast classification (temperature 0.0)
    # =========================================================================
    fast_result = _classify_llm_fast(user_input)
    if fast_result is not None:
        return fast_result

    # =========================================================================
    # STEP 2: Fall back to full LLM classification for complex queries
    # =========================================================================
    logger.info(f"[INTENT_LLM] Fast classification returned unknown, using full LLM for: '{user_input[:60]}'")

    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = ChatPromptTemplate.from_template(INTENT_CLASSIFICATION_PROMPT)
        parser = JsonOutputParser()

        chain = prompt | llm | parser

        # Use retry wrapper with automatic key rotation and OpenAI fallback
        result = invoke_with_retry_fallback(
            chain,
            {
                "user_input": user_input,
                "current_step": current_step or "start",
                "context": context or "New conversation"
            },
            max_retries=3,
            fallback_to_openai=True,
            model="gemini-2.5-flash",
            temperature=0.1
        )

        # Map new 4-intent architecture to response
        # Normalize intent to lowercase for internal consistency
        raw_intent = result.get("intent", "INVALID_INPUT")
        intent = raw_intent.lower() if raw_intent else "invalid_input"

        # Map intent names: INVALID_INPUT -> invalid_input, CHAT -> chat, etc.
        intent_name_mapping = {
            "invalid_input": "invalid_input",
            "chat": "chat",
            "search": "search",
            "solution": "solution",
            # Legacy mappings for backwards compatibility
            "requirements": "search",
            "question": "chat",
            "productinfo": "chat",
            "product_info": "chat",
            "chitchat": "invalid_input",
            "unrelated": "invalid_input"
        }
        intent = intent_name_mapping.get(intent, intent)

        # Get confidence - support both "confidence_score" (new) and "confidence" (old)
        confidence = result.get("confidence_score") or result.get("confidence", 0.5)
        if isinstance(confidence, str):
            # Map string confidence to numeric
            confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
            confidence = confidence_map.get(confidence.lower(), 0.7)

        return {
            "success": True,
            "intent": intent,
            "confidence": confidence,
            "next_step": result.get("next_step"),
            "extracted_info": result.get("extracted_info", {}),
            "is_solution": result.get("is_solution", False),
            "key_indicators": result.get("key_indicators", []),
            "reasoning": result.get("reasoning", ""),
            "product_category": result.get("product_category", "unknown"),
            "parent_instrument": result.get("parent_instrument"),
            "solution_indicators": result.get("solution_indicators", [])
        }

    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return {
            "success": False,
            "intent": "invalid_input",  # New 4-intent architecture
            "confidence": 0.0,
            "next_step": None,
            "error": str(e)
        }


@tool("extract_requirements", args_schema=ExtractRequirementsInput)
def extract_requirements_tool(user_input: str) -> Dict[str, Any]:
    """
    Extract structured technical requirements from user input.
    Identifies product type, specifications, and infers missing common specs.

    Uses invoke_with_retry_fallback for automatic retry, key rotation,
    and OpenAI fallback on RESOURCE_EXHAUSTED errors.
    """
    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = ChatPromptTemplate.from_template(REQUIREMENTS_EXTRACTION_PROMPT)
        parser = JsonOutputParser()

        chain = prompt | llm | parser

        # Use retry wrapper with automatic key rotation and OpenAI fallback
        result = invoke_with_retry_fallback(
            chain,
            {"user_input": user_input},
            max_retries=3,
            fallback_to_openai=True,
            model="gemini-2.5-flash",
            temperature=0.1
        )

        return {
            "success": True,
            "product_type": result.get("product_type"),
            "product_category": result.get("product_category", "instrument"),
            "parent_instrument": result.get("parent_instrument"),
            "specifications": result.get("specifications", {}),
            "inferred_specs": result.get("inferred_specs", {}),
            "raw_requirements_text": result.get("raw_requirements_text", user_input)
        }

    except Exception as e:
        error_str = str(e).lower()
        if "json" in error_str or "parse" in error_str or "parsing" in error_str or "output" in error_str:
            logger.warning(f"Requirements extraction JSON parse failed (likely conversational input): {e}")
            return {
                "success": True,
                "product_type": None,
                "is_conversational": True,
                "specifications": {},
                "raw_requirements_text": user_input
            }

        logger.error(f"Requirements extraction failed: {e}")
        return {
            "success": False,
            "product_type": None,
            "specifications": {},
            "error": str(e)
        }
