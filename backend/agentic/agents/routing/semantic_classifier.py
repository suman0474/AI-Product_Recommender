"""
Semantic Intent Classifier
Uses embedding similarity to classify user queries into workflows.

This replaces LLM-based classification with faster, deterministic semantic matching.
Leverages Google Gemini embeddings and cosine similarity.
"""
import os
import logging
import math
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Debug flag integration
try:
    from debug_flags import issue_debug
except ImportError:
    issue_debug = None


class WorkflowType(Enum):
    """Target workflow types for routing."""
    ENGENIE_CHAT = "engenie_chat"
    SOLUTION_WORKFLOW = "solution_workflow"
    INSTRUMENT_IDENTIFIER = "instrument_identifier"


@dataclass
class ClassificationResult:
    """Result of semantic classification."""
    workflow: WorkflowType
    confidence: float
    matched_signature: str
    all_scores: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""


# =============================================================================
# WORKFLOW SIGNATURE DEFINITIONS
# =============================================================================

# EnGenie Chat: Simple product requests, knowledge queries, single instrument needs
ENGENIE_CHAT_SIGNATURES = [
    # Simple product requests
    "Looking for a temperature sensor",
    "Find me a flow meter",
    "I want a level sensor",
    "I need a thermocouple",
    "Looking for a pressure gauge",
    "Find a control valve",
    "I need an RTD sensor",
    "Show me pressure transmitters",
    "Recommend a temperature sensor",
    
    # Knowledge queries
    "What is ATEX certification?",
    "Explain the difference between SIL 2 and SIL 3",
    "What is a thermowell?",
    "How does a pressure transmitter work?",
    "Tell me about RTD sensors",
    "What is the accuracy of Rosemount transmitters?",
    
    # Simple specifications
    "I need a pressure transmitter 0-100 bar",
    "Looking for temperature sensor for 200 degrees",
    "Find me a 4-20mA pressure transmitter",
    "I want a stainless steel temperature probe",
]

# Solution Workflow: Complex multi-requirement engineering challenges
SOLUTION_WORKFLOW_SIGNATURES = [
    # Complex system design - Temperature
    """Design a comprehensive temperature measurement system for a chemical reactor with hot oil heating circuit:
    Requirements: Reactor operating temperature 150-280°C, hot oil inlet/outlet measurement up to 320°C,
    differential temperature monitoring, redundant sensors for safety, 4-20mA and HART protocol.""",
    
    # Multi-tube reactor profiling
    """Implement a comprehensive temperature profiling system for a multi-tube catalytic reactor:
    Requirements: 48 tubes vertical fixed bed, operating temperature 200-350°C, 1500 psi pressure,
    skin temperature 8 tubes monitored, catalyst bed profile 4 depth measurements, 32 total points.""",
    
    # Pump system selection
    """Select and specify a complete positive displacement pump system for a process recycle line:
    Requirements: Flow rate 25 m³/hr, discharge pressure 250 psi, VFD control, pulsation dampener,
    relief valve, mechanical seal with flushing, jacket heating, ATEX Zone 1 compliance.""",
    
    # Pressure measurement system
    """Design a complete pressure monitoring system for a hydrocracker unit:
    Requirements: Multiple pressure points, high temperature operation, redundant transmitters,
    SIL 2 rated sensors, HART communication, DCS integration, safety alarms.""",
    
    # Flow measurement system
    """Implement a comprehensive flow measurement system for a refinery process unit:
    Requirements: Multiple flow streams, custody transfer accuracy, coriolis and ultrasonic meters,
    temperature compensation, API compliant, data logging capability.""",
    
    # Level measurement system  
    """Design a complete level measurement system for storage tank farm:
    Requirements: 12 tanks, radar level transmitters, overfill protection, inventory management,
    high accuracy for custody transfer, multiple outputs per tank.""",
    
    # Indicators of complex challenges
    "I'm designing a complete instrumentation package for a new process unit",
    "We need to implement a comprehensive monitoring system with multiple measurement points",
    "Design a multi-zone temperature control system with redundancy",
    "I'm building a complete measurement system for reactor temperature profiling",
]


class SemanticIntentClassifier:
    """
    Semantic embedding-based intent classifier.
    
    Uses cosine similarity between query embeddings and pre-computed
    workflow signature embeddings to determine routing.
    """
    
    # Classification thresholds
    SOLUTION_THRESHOLD = 0.75  # Minimum similarity for solution workflow
    ENGENIE_THRESHOLD = 0.70   # Minimum similarity for engenie chat
    MIN_CONFIDENCE = 0.60      # Below this, use fallback
    
    # Query length indicator for solution (long queries = likely solution)
    SOLUTION_LENGTH_THRESHOLD = 200  # Characters
    
    def __init__(self):
        """Initialize classifier with embedding model."""
        self._embeddings = None
        self._signature_cache: Dict[str, List[float]] = {}
        self._initialized = False
        self._signatures_precomputed = False
        
    def _get_embeddings(self):
        """Lazy-load embedding model with retry logic."""
        if self._embeddings is None:
            max_retries = 3
            retry_delay = 1.0  # seconds

            for attempt in range(max_retries):
                try:
                    from langchain_google_genai import GoogleGenerativeAIEmbeddings
                    self._embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/gemini-embedding-001",
                        google_api_key=os.getenv("GOOGLE_API_KEY")
                    )
                    logger.info("[SEMANTIC] Initialized Google Gemini embeddings with gemini-embedding-001")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"[SEMANTIC] Failed to initialize embeddings (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"[SEMANTIC] Failed to initialize embeddings after {max_retries} attempts: {e}")
                        raise
        return self._embeddings
    
    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for text, with caching and retry logic."""
        # Check cache
        if text in self._signature_cache:
            return self._signature_cache[text]

        # Also check the global embedding cache
        try:
            from agentic.infrastructure.caching.embedding_cache import get_embedding_cache
            cache = get_embedding_cache()
            cached = cache.get(text)
            if cached is not None:
                self._signature_cache[text] = cached
                return cached
        except ImportError:
            pass

        # Compute embedding with retry logic - track API call
        max_retries = 3
        retry_delay = 0.5  # seconds

        for attempt in range(max_retries):
            try:
                embeddings = self._get_embeddings()
                if issue_debug:
                    issue_debug.embedding_call("gemini-embedding-001", 1, "semantic_classifier")
                embedding = embeddings.embed_query(text)

                # Cache it
                self._signature_cache[text] = embedding
                try:
                    from agentic.infrastructure.caching.embedding_cache import get_embedding_cache
                    cache = get_embedding_cache()
                    cache.put(text, embedding)
                except ImportError:
                    pass

                return embedding

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"[SEMANTIC] Embedding computation failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"[SEMANTIC] Embedding computation failed after {max_retries} attempts: {e}")
                    raise
    
    def _precompute_signature_embeddings(self):
        """
        Pre-compute all signature embeddings once at startup.
        Uses batch processing to reduce latency.
        """
        if self._signatures_precomputed:
            return
        
        all_signatures = ENGENIE_CHAT_SIGNATURES + SOLUTION_WORKFLOW_SIGNATURES
        missing_signatures = []
        cached_count = 0
        
        # Check what's already cached
        for sig in all_signatures:
            # Check local cache first
            if sig in self._signature_cache:
                cached_count += 1
                continue
                
            # Check global cache
            try:
                from agentic.infrastructure.caching.embedding_cache import get_embedding_cache
                cache = get_embedding_cache()
                cached_emb = cache.get(sig)
                if cached_emb is not None:
                    self._signature_cache[sig] = cached_emb
                    cached_count += 1
                    continue
            except ImportError:
                pass
            
            # If not in either cache, add to missing list
            missing_signatures.append(sig)
            
        # Batch compute missing signatures with retry logic
        computed_count = 0
        if missing_signatures:
            logger.info(f"[SEMANTIC] Batch computing {len(missing_signatures)} missing signature embeddings...")

            max_retries = 3
            retry_delay = 1.0

            for attempt in range(max_retries):
                try:
                    embeddings_model = self._get_embeddings()

                    if issue_debug:
                        issue_debug.embedding_call("gemini-embedding-001", len(missing_signatures), "semantic_classifier_batch")

                    # Batch API call (much faster than sequential)
                    batch_embeddings = embeddings_model.embed_documents(missing_signatures)

                    # Store in caches
                    from agentic.infrastructure.caching.embedding_cache import get_embedding_cache
                    try:
                        global_cache = get_embedding_cache()
                    except ImportError:
                        global_cache = None

                    for sig, emb in zip(missing_signatures, batch_embeddings):
                        self._signature_cache[sig] = emb
                        if global_cache:
                            global_cache.put(sig, emb)
                        computed_count += 1

                    break  # Success, exit retry loop

                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(
                            f"[SEMANTIC] Batch embedding failed (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"[SEMANTIC] Batch embedding failed after {max_retries} attempts: {e}")
                        # Fallback to sequential if batch fails completely
                        logger.info("[SEMANTIC] Falling back to sequential embedding computation...")
                        for sig in missing_signatures:
                            try:
                                self._compute_embedding(sig)
                                computed_count += 1
                            except Exception as seq_err:
                                logger.warning(f"[SEMANTIC] Failed to compute signature {sig[:20]}...: {seq_err}")
        
        self._signatures_precomputed = True
        logger.info(
            f"[SEMANTIC] Signature embeddings ready: "
            f"{cached_count} cached, {computed_count} computed (batch)"
        )
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors using pure Python."""
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
    
    def _compute_workflow_similarity(
        self, 
        query_embedding: List[float],
        signatures: List[str]
    ) -> Tuple[float, str]:
        """
        Compute maximum similarity between query and workflow signatures.
        
        Returns:
            (max_similarity, best_matching_signature)
        """
        max_sim = 0.0
        best_sig = ""
        
        for sig in signatures:
            sig_embedding = self._compute_embedding(sig)
            sim = self._cosine_similarity(query_embedding, sig_embedding)
            if sim > max_sim:
                max_sim = sim
                best_sig = sig[:100] + "..." if len(sig) > 100 else sig
                
        return max_sim, best_sig
    
    def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        Analyze query for complexity indicators.
        
        Returns indicators that suggest solution workflow:
        - Length (long queries = complex requirements)
        - Multiple requirements (numbered lists, bullet points)
        - Technical specifications
        """
        query_lower = query.lower()
        
        # Length check
        is_long = len(query) > self.SOLUTION_LENGTH_THRESHOLD
        
        # Multi-requirement indicators
        multi_requirement_indicators = [
            "requirements:", "requirement:", "specs:", "specifications:",
            "1.", "2.", "3.", "•", "-", "multiple", "several",
            "and also", "additionally", "furthermore"
        ]
        has_multi_requirements = any(ind in query_lower for ind in multi_requirement_indicators)
        
        # System design indicators
        system_indicators = [
            "design a", "implement a", "comprehensive", "complete system",
            "measurement system", "monitoring system", "control system",
            "profiling system", "instrumentation package"
        ]
        has_system_design = any(ind in query_lower for ind in system_indicators)
        
        # Technical complexity
        technical_indicators = [
            "redundant", "sil", "atex", "iecex", "hart", "dcs",
            "safety", "alarm", "multiple points", "integration"
        ]
        has_technical_complexity = sum(1 for ind in technical_indicators if ind in query_lower)
        
        return {
            "is_long": is_long,
            "has_multi_requirements": has_multi_requirements,
            "has_system_design": has_system_design,
            "technical_complexity_score": has_technical_complexity,
            "complexity_boost": (
                0.1 if is_long else 0 +
                0.1 if has_multi_requirements else 0 +
                0.1 if has_system_design else 0 +
                0.02 * has_technical_complexity
            )
        }
    
    def classify(self, query: str) -> ClassificationResult:
        """
        Classify user query into a workflow using semantic similarity.
        
        Args:
            query: User input text
            
        Returns:
            ClassificationResult with workflow, confidence, and matched signature
        """
        try:
            # Pre-compute signature embeddings if not done yet
            if not self._signatures_precomputed:
                self._precompute_signature_embeddings()
            
            # Compute query embedding
            query_embedding = self._compute_embedding(query)
            
            # Compute similarity to each workflow
            engenie_sim, engenie_match = self._compute_workflow_similarity(
                query_embedding, ENGENIE_CHAT_SIGNATURES
            )
            solution_sim, solution_match = self._compute_workflow_similarity(
                query_embedding, SOLUTION_WORKFLOW_SIGNATURES
            )
            
            # Analyze query complexity for boosting
            complexity = self._analyze_query_complexity(query)
            
            # Apply complexity boost to solution score
            solution_sim_boosted = min(1.0, solution_sim + complexity["complexity_boost"])
            
            # Log similarities
            logger.info(
                f"[SEMANTIC] Query: '{query[:50]}...' | "
                f"EnGenie: {engenie_sim:.3f} | Solution: {solution_sim:.3f} "
                f"(boosted: {solution_sim_boosted:.3f})"
            )
            
            # Decision logic
            all_scores = {
                "engenie_chat": engenie_sim,
                "solution_workflow": solution_sim,
                "solution_boosted": solution_sim_boosted
            }
            
            # Solution workflow wins if:
            # 1. Boosted score > threshold AND
            # 2. Boosted score > engenie score
            if solution_sim_boosted > self.SOLUTION_THRESHOLD and solution_sim_boosted > engenie_sim:
                return ClassificationResult(
                    workflow=WorkflowType.SOLUTION_WORKFLOW,
                    confidence=solution_sim_boosted,
                    matched_signature=solution_match,
                    all_scores=all_scores,
                    reasoning=f"Solution similarity {solution_sim_boosted:.3f} > threshold {self.SOLUTION_THRESHOLD}"
                )
            
            # EnGenie Chat wins if score > threshold
            if engenie_sim > self.ENGENIE_THRESHOLD:
                return ClassificationResult(
                    workflow=WorkflowType.ENGENIE_CHAT,
                    confidence=engenie_sim,
                    matched_signature=engenie_match,
                    all_scores=all_scores,
                    reasoning=f"EnGenie similarity {engenie_sim:.3f} > threshold {self.ENGENIE_THRESHOLD}"
                )
            
            # Fallback: EnGenie Chat with lower confidence
            return ClassificationResult(
                workflow=WorkflowType.ENGENIE_CHAT,
                confidence=max(engenie_sim, solution_sim),
                matched_signature=engenie_match if engenie_sim > solution_sim else solution_match,
                all_scores=all_scores,
                reasoning=f"Fallback to EnGenie Chat (no score met threshold)"
            )
            
        except Exception as e:
            logger.error(f"[SEMANTIC] Classification failed: {e}")
            # Fallback to rule-based heuristics
            return self._classify_rule_based_fallback(query, str(e))
    
    def _classify_rule_based_fallback(self, query: str, error: str = "") -> ClassificationResult:
        """
        Rule-based fallback classification when embeddings are unavailable.
        Uses query complexity analysis to determine workflow.
        """
        complexity = self._analyze_query_complexity(query)
        query_lower = query.lower()
        
        # Solution indicators (require multiple matches for high confidence)
        solution_score = 0
        
        # Length-based
        if len(query) > 500:
            solution_score += 3
        elif len(query) > 200:
            solution_score += 2
        
        # Multi-requirement indicators
        if complexity["has_multi_requirements"]:
            solution_score += 2
        
        # System design phrases
        if complexity["has_system_design"]:
            solution_score += 2
            
        # Technical complexity
        solution_score += complexity["technical_complexity_score"]
        
        # Explicit design phrases
        design_phrases = [
            "design a complete", "implement a comprehensive", "design a system",
            "measurement system for", "monitoring system for", "control system for",
            "instrumentation package", "multiple instruments", "multiple measurement"
        ]
        for phrase in design_phrases:
            if phrase in query_lower:
                solution_score += 2
                
        # Decision
        if solution_score >= 5:
            return ClassificationResult(
                workflow=WorkflowType.SOLUTION_WORKFLOW,
                confidence=min(0.9, 0.5 + solution_score * 0.05),
                matched_signature="rule_based_match",
                all_scores={"solution_score": solution_score},
                reasoning=f"Rule-based fallback: solution_score={solution_score} (error: {error[:50]})"
            )
        else:
            return ClassificationResult(
                workflow=WorkflowType.ENGENIE_CHAT,
                confidence=0.75,
                matched_signature="rule_based_match",
                all_scores={"solution_score": solution_score},
                reasoning=f"Rule-based fallback: solution_score={solution_score} (error: {error[:50]})"
            )


# Global singleton
_classifier_instance: Optional[SemanticIntentClassifier] = None


def get_semantic_classifier() -> SemanticIntentClassifier:
    """Get or create the global semantic classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = SemanticIntentClassifier()
        logger.info("[SEMANTIC] Created global semantic classifier instance")
    return _classifier_instance


def classify_intent_semantic(query: str) -> Tuple[str, float, str]:
    """
    Convenience function for semantic classification.
    
    Returns:
        (workflow_name, confidence, matched_signature)
    """
    classifier = get_semantic_classifier()
    result = classifier.classify(query)
    return result.workflow.value, result.confidence, result.matched_signature
