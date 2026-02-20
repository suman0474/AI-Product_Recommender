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
    """
    Canonical workflow types for routing (SOURCE OF TRUTH).

    All other files should import this enum rather than defining their own.
    """
    ENGENIE_CHAT = "engenie_chat"
    SOLUTION_WORKFLOW = "solution"
    INSTRUMENT_IDENTIFIER = "instrument_identifier"
    OUT_OF_DOMAIN = "out_of_domain"  # Invalid/non-industrial queries
    GREETING = "greeting"  # Direct greeting response
    CONVERSATIONAL = "conversational"  # Farewell, gratitude, help, chitchat


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

# Instrument Identifier: Purchase/search intent for single products with specifications
INSTRUMENT_IDENTIFIER_SIGNATURES = [
    # Direct purchase intent
    "I need a pressure transmitter for 0-100 bar range",
    "Looking for a temperature sensor with 4-20mA output",
    "Find me a flowmeter for corrosive liquids",
    "I want an RTD sensor with HART protocol",
    "Recommend a level transmitter for tank monitoring",
    "Show me options for a differential pressure transmitter",

    # Product with specifications
    "I need a transmitter with 0.075% accuracy",
    "Looking for a sensor rated for 400 degrees Celsius",
    "Find a valve positioner with SIL 2 certification",
    "I want an ATEX-certified pressure gauge for Zone 1",

    # Vendor-specific product requests
    "Get me a Rosemount 3051S pressure transmitter",
    "I need a Yokogawa EJX series differential pressure transmitter",
    "Find a Fisher control valve for my application",
    "Looking for an Emerson magnetic flowmeter",

    # Specification-driven requests
    "I need an instrument that can measure 0-500 psi with HART",
    "Find a thermocouple probe for high-temperature furnace application",
    "Looking for a coriolis meter with custody transfer accuracy",
    "I need a control valve with fail-safe close functionality",

    # Product selection with constraints
    "What pressure transmitter works for corrosive chemicals?",
    "Which flowmeter has the best accuracy for custody transfer?",
    "Best thermocouple type for high temperature furnace",
    "Suggest a level sensor for hazardous area Zone 1",
]

# Out-of-Domain: Non-industrial queries that should be rejected
OUT_OF_DOMAIN_SIGNATURES = [
    # General knowledge (non-industrial)
    "What is the capital of France?",
    "Who is the president of United States?",
    "Tell me about World War 2",
    "What is quantum physics?",
    "Explain theory of relativity",
    "Who invented the telephone?",
    "What is the meaning of life?",
    "How does photosynthesis work?",
    
    # Entertainment & Media
    "Tell me a joke",
    "What's a good movie to watch?",
    "Who won the World Cup?",
    "Latest episode of Game of Thrones",
    "Best songs of 2024",
    "Who is the best actor?",
    "Netflix recommendations",
    "Video game tips and tricks",
    
    # Weather & Nature
    "What's the weather like today?",
    "Will it rain tomorrow?",
    "Climate change effects",
    "Earthquake prediction methods",
    "What causes hurricanes?",
    "Best time to plant tomatoes",
    
    # Personal & Health & Food
    "Best recipe for chocolate cake",
    "How to lose weight fast?",
    "Symptoms of flu",
    "Restaurant recommendations near me",
    "Dating advice for beginners",
    "How to cook pasta perfectly?",
    "Best diet for weight loss",
    "Yoga poses for back pain",
    
    # Consumer Tech (non-industrial)
    "How to code in Python?",
    "Best laptop for gaming?",
    "Install Windows 11 step by step",
    "Fix WiFi connection issues",
    "iPhone vs Android comparison",
    "How to edit photos in Photoshop?",
    "Smartphone camera settings",
    "Best antivirus software",
    
    # Politics & Current Events
    "Who is Donald Trump?",
    "Latest political news",
    "What is happening in Ukraine?",
    "Election results 2024",
    "Government policies explained",
    
    # Sports & Fitness
    "Who won the Super Bowl?",
    "Best exercises for abs",
    "How to improve running speed?",
    "Football match schedule",
    "NBA playoff predictions",
    
    # Travel & Lifestyle
    "Best places to visit in Europe",
    "How to pack for a vacation?",
    "Budget travel tips",
    "Hotel recommendations in Paris",
    "What to see in New York?",
    
    # Education (non-industrial)
    "How to study for exams?",
    "Best online courses for marketing",
    "What is algebra?",
    "How to write an essay?",
    "Grammar rules explained",
    
    # Edge cases with industrial terms but wrong context
    "Temperature for baking bread",
    "Pressure when diving underwater",
    "Flow of traffic on highway",
    "Level up in video game",
    "Sensor news and updates",
    "Automation in workflow management software",
    "Sensor data from smartphone accelerometer",
    "Temperature control in refrigerator",
    "Pressure cooker cooking times",
    "Flow rate of garden hose",
    "Water level in swimming pool",
    "Smart home sensor integration",
    "Body temperature measurement",
    "Blood pressure readings",
    "Air pressure in car tires",
]

# Strong reject keywords (immediate OUT_OF_DOMAIN, no embedding needed)
# These trigger fast-path rejection without computing embeddings
STRONG_REJECT_KEYWORDS = {
    "weather", "joke", "funny", "story", "poem",
    "recipe", "cooking", "food", "restaurant",
    "movie", "music", "celebrity", "sports", "game",
    "politics", "election", "president", "government",
    "lottery", "gambling", "casino", "betting",
    "dating", "relationship", "marriage",
}


class SemanticIntentClassifier:
    """
    Semantic embedding-based intent classifier.
    
    Uses cosine similarity between query embeddings and pre-computed
    workflow signature embeddings to determine routing.
    """
    
    # Classification thresholds
    # IMPROVED: Lowered SOLUTION_THRESHOLD from 0.75 to 0.70 for better coverage
    # Complex solution queries often have raw scores 0.70-0.75 before boosting
    SOLUTION_THRESHOLD = 0.70  # Minimum similarity for solution workflow
    INSTRUMENT_THRESHOLD = 0.70  # Minimum similarity for instrument identifier
    ENGENIE_THRESHOLD = 0.68   # Minimum similarity for engenie chat
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
            from common.infrastructure.caching.embedding_cache import get_embedding_cache
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
                    from common.infrastructure.caching.embedding_cache import get_embedding_cache
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
        
        all_signatures = (
            ENGENIE_CHAT_SIGNATURES +
            SOLUTION_WORKFLOW_SIGNATURES +
            INSTRUMENT_IDENTIFIER_SIGNATURES +
            OUT_OF_DOMAIN_SIGNATURES
        )
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
                from common.infrastructure.caching.embedding_cache import get_embedding_cache
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
                    from common.infrastructure.caching.embedding_cache import get_embedding_cache
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
                (0.1 if is_long else 0) +
                (0.1 if has_multi_requirements else 0) +
                (0.1 if has_system_design else 0) +
                (0.02 * has_technical_complexity)
            )
        }
    
    def classify(self, query: str) -> ClassificationResult:
        """
        Classify user query into a workflow using semantic similarity.

        Implements 4-way classification:
        - ENGENIE_CHAT: Knowledge queries, simple questions
        - INSTRUMENT_IDENTIFIER: Purchase/search intent for single products
        - SOLUTION_WORKFLOW: Complex multi-requirement system design
        - OUT_OF_DOMAIN: Non-industrial queries

        Args:
            query: User input text

        Returns:
            ClassificationResult with workflow, confidence, and matched signature
        """
        try:
            # FAST PATH: Check for strong reject keywords (no embedding needed)
            query_lower = query.lower()
            for keyword in STRONG_REJECT_KEYWORDS:
                if keyword in query_lower:
                    logger.info(f"[SEMANTIC] Strong reject keyword: '{keyword}'")
                    return ClassificationResult(
                        workflow=WorkflowType.OUT_OF_DOMAIN,
                        confidence=0.95,
                        matched_signature=f"strong_reject:{keyword}",
                        all_scores={"strong_reject": 0.95},
                        reasoning=f"Strong reject keyword detected: '{keyword}'"
                    )

            # Pre-compute signature embeddings if not done yet
            if not self._signatures_precomputed:
                self._precompute_signature_embeddings()

            # Compute query embedding
            query_embedding = self._compute_embedding(query)

            # Compute similarity to each workflow (TRUE 4-way classification)
            engenie_sim, engenie_match = self._compute_workflow_similarity(
                query_embedding, ENGENIE_CHAT_SIGNATURES
            )
            solution_sim, solution_match = self._compute_workflow_similarity(
                query_embedding, SOLUTION_WORKFLOW_SIGNATURES
            )
            instrument_sim, instrument_match = self._compute_workflow_similarity(
                query_embedding, INSTRUMENT_IDENTIFIER_SIGNATURES
            )
            out_of_domain_sim, ood_match = self._compute_workflow_similarity(
                query_embedding, OUT_OF_DOMAIN_SIGNATURES
            )
            
            # Analyze query complexity for boosting
            complexity = self._analyze_query_complexity(query)

            # Apply complexity boost to solution score
            # FIX: Cap maximum boost at 0.15 to prevent over-boosting (was allowing 0.3+)
            # This prevents queries from jumping from 0.75 to 1.0 which is too aggressive
            capped_boost = min(0.15, complexity["complexity_boost"])
            solution_sim_boosted = min(1.0, solution_sim + capped_boost)
            
            # Log similarities (4-way)
            logger.info(
                f"[SEMANTIC] Query: '{query[:50]}...' | "
                f"EnGenie: {engenie_sim:.3f} | Instrument: {instrument_sim:.3f} | "
                f"Solution: {solution_sim:.3f} (boosted: {solution_sim_boosted:.3f}) | "
                f"OOD: {out_of_domain_sim:.3f}"
            )

            # Decision logic with OUT_OF_DOMAIN priority (TRUE 4-way)
            all_scores = {
                "engenie_chat": engenie_sim,
                "instrument_identifier": instrument_sim,
                "solution": solution_sim,
                "solution_boosted": solution_sim_boosted,
                "out_of_domain": out_of_domain_sim
            }
            
            # PRIORITY 1: Reject if OUT_OF_DOMAIN confidence is very high
            # High threshold (0.80) to avoid false positives
            if out_of_domain_sim > 0.80:
                logger.warning(
                    f"[SEMANTIC] OUT_OF_DOMAIN detected (high confidence): {out_of_domain_sim:.3f}"
                )
                return ClassificationResult(
                    workflow=WorkflowType.OUT_OF_DOMAIN,
                    confidence=out_of_domain_sim,
                    matched_signature=ood_match,
                    all_scores=all_scores,
                    reasoning=f"Out-of-domain similarity {out_of_domain_sim:.3f} > 0.80 (high threshold)"
                )
            
            # PRIORITY 2: Solution workflow wins if:
            # 1. Boosted score > threshold AND
            # 2. Boosted score > other valid scores AND
            # 3. NOT strongly out-of-domain
            if (solution_sim_boosted > self.SOLUTION_THRESHOLD and
                solution_sim_boosted > engenie_sim and
                solution_sim_boosted > instrument_sim and
                out_of_domain_sim < 0.65):
                return ClassificationResult(
                    workflow=WorkflowType.SOLUTION_WORKFLOW,
                    confidence=solution_sim_boosted,
                    matched_signature=solution_match,
                    all_scores=all_scores,
                    reasoning=f"Solution similarity {solution_sim_boosted:.3f} > threshold {self.SOLUTION_THRESHOLD}"
                )

            # PRIORITY 3: Instrument Identifier wins if:
            # 1. Score > threshold AND
            # 2. Score > engenie score AND
            # 3. Score > raw solution score (not boosted - instrument is single product focus)
            # 4. NOT strongly out-of-domain
            if (instrument_sim > self.INSTRUMENT_THRESHOLD and
                instrument_sim > engenie_sim and
                instrument_sim > solution_sim and
                out_of_domain_sim < 0.65):
                return ClassificationResult(
                    workflow=WorkflowType.INSTRUMENT_IDENTIFIER,
                    confidence=instrument_sim,
                    matched_signature=instrument_match,
                    all_scores=all_scores,
                    reasoning=f"Instrument Identifier similarity {instrument_sim:.3f} > threshold {self.INSTRUMENT_THRESHOLD}"
                )

            # PRIORITY 4: EnGenie Chat wins if:
            # 1. Score > threshold AND
            # 2. NOT strongly out-of-domain
            if engenie_sim > self.ENGENIE_THRESHOLD and out_of_domain_sim < 0.65:
                return ClassificationResult(
                    workflow=WorkflowType.ENGENIE_CHAT,
                    confidence=engenie_sim,
                    matched_signature=engenie_match,
                    all_scores=all_scores,
                    reasoning=f"EnGenie similarity {engenie_sim:.3f} > threshold {self.ENGENIE_THRESHOLD}"
                )

            # PRIORITY 5: Ambiguous case - compare valid scores vs OUT_OF_DOMAIN
            max_valid_score = max(engenie_sim, instrument_sim, solution_sim_boosted)

            # If all valid scores are low AND OOD score is moderate, likely invalid
            if max_valid_score < 0.60 and out_of_domain_sim > 0.60:
                logger.warning(
                    f"[SEMANTIC] OUT_OF_DOMAIN detected (ambiguous): "
                    f"max_valid={max_valid_score:.3f}, ood={out_of_domain_sim:.3f}"
                )
                # Compute hybrid confidence: average of OOD score and (1 - max_valid_score)
                hybrid_confidence = (out_of_domain_sim + (1 - max_valid_score)) / 2
                return ClassificationResult(
                    workflow=WorkflowType.OUT_OF_DOMAIN,
                    confidence=hybrid_confidence,
                    matched_signature=ood_match,
                    all_scores=all_scores,
                    reasoning=f"Ambiguous query: low industrial similarity ({max_valid_score:.3f}) + moderate OOD ({out_of_domain_sim:.3f})"
                )
            
            # PRIORITY 6: Additional check - if OOD clearly beats valid scores
            if out_of_domain_sim > max_valid_score and out_of_domain_sim > 0.65:
                logger.warning(
                    f"[SEMANTIC] OUT_OF_DOMAIN wins comparison: "
                    f"ood={out_of_domain_sim:.3f} > max_valid={max_valid_score:.3f}"
                )
                return ClassificationResult(
                    workflow=WorkflowType.OUT_OF_DOMAIN,
                    confidence=out_of_domain_sim,
                    matched_signature=ood_match,
                    all_scores=all_scores,
                    reasoning=f"OOD score {out_of_domain_sim:.3f} clearly exceeds valid scores ({max_valid_score:.3f})"
                )
            
            # FALLBACK: Default to EnGenie Chat with lower confidence
            # Prefer to process rather than reject in uncertain cases
            return ClassificationResult(
                workflow=WorkflowType.ENGENIE_CHAT,
                confidence=max(engenie_sim, solution_sim),
                matched_signature=engenie_match if engenie_sim > solution_sim else solution_match,
                all_scores=all_scores,
                reasoning=f"Fallback to EnGenie Chat (no clear classification, max_valid={max_valid_score:.3f})"
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


# =============================================================================
# WORKFLOW CAPABILITY VALIDATION (Merged from domain_validator.py)
# =============================================================================

def validate_workflow_capability(
    query: str,
    target_workflow: str  # Use string to avoid circular import
) -> Tuple[bool, Optional[str]]:
    """
    Validate that target workflow can handle the query.

    Args:
        query: User query string
        target_workflow: Target workflow name (e.g., "engenie_chat", "instrument_identifier", "solution")

    Returns:
        (can_handle, reason_if_not)

    Examples:
        >>> validate_workflow_capability("What is a transmitter?", "engenie_chat")
        (True, None)

        >>> validate_workflow_capability("I need a transmitter", "engenie_chat")
        (False, "Purchase intent detected, should route to SEARCH")

        >>> validate_workflow_capability("What is a transmitter?", "solution")
        (False, "Knowledge question detected, should route to CHAT")
    """
    query_lower = query.lower()

    # Convert string to enum-like comparison
    if target_workflow in ["engenie_chat", "ENGENIE_CHAT"]:
        # CHAT cannot handle purchase/design intents
        purchase_keywords = ["i need", "looking for", "find me", "recommend", "suggest"]
        design_keywords = ["design a", "design an", "implement a", "build a", "create a"]

        # Check for purchase intent
        has_purchase = any(kw in query_lower for kw in purchase_keywords)
        has_instrument = any(inst in query_lower for inst in [
            "transmitter", "sensor", "valve", "actuator", "flowmeter", "analyzer"
        ])

        if has_purchase and has_instrument:
            logger.info(f"[WorkflowValidator] CHAT cannot handle purchase intent")
            return False, "Purchase intent detected, should route to SEARCH"

        # Check for design intent
        if any(kw in query_lower for kw in design_keywords):
            system_keywords = ["system", "package", "instrumentation"]
            if any(sk in query_lower for sk in system_keywords):
                logger.info(f"[WorkflowValidator] CHAT cannot handle design intent")
                return False, "Design intent detected, should route to SOLUTION"

    elif target_workflow in ["instrument_identifier", "INSTRUMENT_IDENTIFIER"]:
        # SEARCH cannot handle knowledge questions
        knowledge_keywords = ["what is", "how does", "explain", "tell me about", "describe"]

        if any(kw in query_lower for kw in knowledge_keywords):
            logger.info(f"[WorkflowValidator] SEARCH cannot handle knowledge question")
            return False, "Knowledge question detected, should route to CHAT"

        # SEARCH cannot handle multi-instrument systems
        system_keywords = ["complete system", "monitoring system", "control system",
                         "instrumentation package", "measurement system"]

        if any(kw in query_lower for kw in system_keywords):
            logger.info(f"[WorkflowValidator] SEARCH cannot handle system design")
            return False, "System design detected, should route to SOLUTION"

    elif target_workflow in ["solution", "SOLUTION_WORKFLOW"]:
        # SOLUTION cannot handle knowledge questions (unless "what is needed...")
        knowledge_keywords = ["what is", "how does", "explain", "tell me about", "describe"]

        # Exception: "what is needed" or "what do i need" are design questions
        design_exceptions = ["what is needed", "what do i need", "what instruments"]

        has_knowledge = any(kw in query_lower for kw in knowledge_keywords)
        has_exception = any(ex in query_lower for ex in design_exceptions)

        if has_knowledge and not has_exception:
            logger.info(f"[WorkflowValidator] SOLUTION cannot handle knowledge question")
            return False, "Knowledge question detected, should route to CHAT"

    # No capability issues detected
    logger.debug(f"[WorkflowValidator] Workflow '{target_workflow}' can handle query")
    return True, None


def get_reject_message(reason: str) -> str:
    """
    Get user-friendly reject message based on validation reason.

    Args:
        reason: Rejection reason from OUT_OF_DOMAIN classification

    Returns:
        User-friendly message explaining why query was rejected
    """
    # Map reasons to user-friendly messages
    if "plc" in reason.lower() or "scada" in reason.lower():
        return (
            "I specialize in instrumentation selection and system design, not PLC/SCADA programming. "
            "For control logic and programming, please consult your automation engineering team."
        )

    if "price" in reason.lower() or "cost" in reason.lower() or "pricing" in reason.lower():
        return (
            "I don't have access to pricing information. "
            "For pricing and quotes, please contact your procurement team or vendor sales representative."
        )

    if "troubleshoot" in reason.lower() or "error" in reason.lower():
        return (
            "I specialize in instrument selection and system design, not troubleshooting. "
            "For operational issues, please consult your maintenance team or vendor technical support."
        )

    if any(word in reason.lower() for word in ["motor", "pump", "compressor", "turbine"]):
        return (
            "I specialize in instrumentation and control devices. "
            "For mechanical equipment selection, please consult your mechanical engineering team."
        )

    # Default reject message
    return (
        "I specialize in industrial instrumentation and process measurement. I can help with:\n"
        "• Instrument specifications and selection (transmitters, sensors, valves, actuators)\n"
        "• Standards and compliance (IEC, ISO, API, ATEX, SIL)\n"
        "• Vendor information and approved suppliers\n"
        "• System design for instrumentation packages\n\n"
        f"Your query appears to be outside this scope. {reason}"
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
