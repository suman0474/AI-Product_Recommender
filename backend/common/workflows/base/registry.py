"""
Workflow Registry - Central registration and invocation system for agentic workflows.

This module provides a singleton registry where workflows self-register with their
metadata, enabling automatic intent-to-workflow matching and invocation.

Level 4 Features (Implemented):
- Workflow self-registration with metadata
- Intent-based workflow matching
- Automatic workflow invocation
- Dynamic workflow discovery

Level 5 Features (Stubs - Schema Ready):
- Semantic matching via embeddings (match_semantic)
- Self-healing with retry policies (invoke_with_retry)
- A/B testing support (version field)

Guardrails (Placeholder):
- Input validation before routing
- Confidence verification
- Disambiguation for low-confidence matches

Author: Agentic System
Version: 1.0.0
"""

import logging
import time
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from threading import Lock
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# RETRY POLICY (Level 5 Ready - Schema Only)
# =============================================================================

class RetryStrategy(Enum):
    """Retry strategies for self-healing invocation."""
    NONE = "none"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR = "linear"
    IMMEDIATE = "immediate"


@dataclass
class RetryPolicy:
    """
    Retry policy for workflow invocation (Level 5 Ready).
    
    Currently a placeholder - will be used when invoke_with_retry is implemented.
    """
    strategy: RetryStrategy = RetryStrategy.NONE
    max_retries: int = 3
    base_delay_ms: int = 1000
    max_delay_ms: int = 30000
    retryable_exceptions: List[str] = field(default_factory=lambda: ["RESOURCE_EXHAUSTED", "TIMEOUT"])


# =============================================================================
# EXPERIMENT (For A/B Testing)
# =============================================================================

@dataclass
class Experiment:
    """
    A/B testing experiment configuration.
    
    Attributes:
        id: Unique experiment identifier
        name: Human-readable experiment name
        base_workflow: Control workflow name
        variant_workflow: Treatment workflow name
        traffic_percentage: Percentage of traffic to variant (0.0-1.0)
        start_time: When the experiment started
        end_time: When the experiment ends (optional)
        is_active: Whether the experiment is currently running
    """
    id: str
    name: str
    base_workflow: str
    variant_workflow: str
    traffic_percentage: float = 0.5
    start_time: 'datetime' = None
    end_time: Optional['datetime'] = None
    is_active: bool = True
    
    def __post_init__(self):
        from datetime import datetime
        if self.start_time is None:
            self.start_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "base_workflow": self.base_workflow,
            "variant_workflow": self.variant_workflow,
            "traffic_percentage": self.traffic_percentage,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "is_active": self.is_active
        }


# =============================================================================
# GUARDRAIL RESULT (For Input Validation)
# =============================================================================

class GuardrailStatus(Enum):
    """Result status from guardrail checks."""
    PASSED = "passed"
    BLOCKED = "blocked"
    NEEDS_CLARIFICATION = "needs_clarification"
    LOW_CONFIDENCE = "low_confidence"


@dataclass
class GuardrailResult:
    """
    Result of guardrail validation.
    
    Attributes:
        status: Whether the input passed, was blocked, or needs clarification
        reason: Explanation for the decision
        suggested_response: Response to show user if blocked/unclear
        confidence: Confidence in the decision (0.0-1.0)
    """
    status: GuardrailStatus
    reason: str = ""
    suggested_response: Optional[str] = None
    confidence: float = 1.0
    
    @property
    def is_safe(self) -> bool:
        return self.status == GuardrailStatus.PASSED


# =============================================================================
# WORKFLOW METADATA
# =============================================================================

@dataclass
class WorkflowMetadata:
    """
    Comprehensive metadata for a registered workflow.
    
    Level 4 Fields (Implemented):
        name: Unique identifier (e.g., "solution")
        display_name: Human-readable name for UI
        description: What this workflow handles
        keywords: Matching keywords for fallback matching
        intents: Valid intent values that route to this workflow
        capabilities: Feature flags (e.g., "multi_instrument", "rag_routing")
        entry_function: The run_*_workflow function to invoke
        priority: Higher = preferred when multiple workflows match
    
    Level 5 Fields (Schema Ready - Stubs):
        embedding: Vector embedding of description for semantic matching
        retry_policy: Self-healing configuration
        version: For A/B testing different versions
        tags: Additional categorization for plugin filtering
        is_enabled: Runtime enable/disable without unregistration
    """
    # Level 4 - Implemented
    name: str
    display_name: str
    description: str
    keywords: List[str]
    intents: List[str]
    capabilities: List[str]
    entry_function: Callable
    priority: int = 50
    
    # Level 5 - Schema Ready (Optional fields with defaults)
    embedding: Optional[List[float]] = None
    retry_policy: Optional[RetryPolicy] = None
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    is_enabled: bool = True
    
    # Guardrail settings
    requires_confirmation: bool = False
    min_confidence_threshold: float = 0.5
    
    def matches_intent(self, intent: str) -> bool:
        """Check if this workflow handles the given intent."""
        return intent.lower().strip() in [i.lower() for i in self.intents]
    
    def matches_keyword(self, query: str) -> int:
        """Return count of matching keywords in query."""
        query_lower = query.lower()
        return sum(1 for kw in self.keywords if kw.lower() in query_lower)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "keywords": self.keywords,
            "intents": self.intents,
            "capabilities": self.capabilities,
            "priority": self.priority,
            "version": self.version,
            "tags": self.tags,
            "is_enabled": self.is_enabled,
            "has_embedding": self.embedding is not None,
            "has_retry_policy": self.retry_policy is not None,
            "requires_confirmation": self.requires_confirmation,
            "min_confidence_threshold": self.min_confidence_threshold
        }


# =============================================================================
# MATCH RESULT (Enhanced with Guardrail Info)
# =============================================================================

@dataclass
class MatchResult:
    """
    Result of workflow matching with confidence and alternatives.
    
    Provides more context than just returning WorkflowMetadata,
    enabling guardrails and disambiguation.
    """
    workflow: Optional[WorkflowMetadata]
    confidence: float
    reasoning: str
    alternatives: List[Tuple[WorkflowMetadata, float]] = field(default_factory=list)
    needs_disambiguation: bool = False
    disambiguation_question: Optional[str] = None
    
    @property
    def is_confident(self) -> bool:
        """Check if match is confident enough to proceed."""
        if not self.workflow:
            return False
        return self.confidence >= self.workflow.min_confidence_threshold
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow": self.workflow.name if self.workflow else None,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "needs_disambiguation": self.needs_disambiguation,
            "disambiguation_question": self.disambiguation_question,
            "alternatives": [
                {"name": w.name, "confidence": c} 
                for w, c in self.alternatives
            ]
        }


# =============================================================================
# WORKFLOW REGISTRY (Singleton)
# =============================================================================

class WorkflowRegistry:
    """
    Singleton registry for all agentic workflows.
    
    Provides:
    - Workflow registration at module load time
    - Intent-based workflow matching with confidence
    - Automatic workflow invocation
    - Dynamic workflow discovery
    - Guardrail validation (placeholder)
    
    Thread-safe for concurrent access.
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
        self._workflows: Dict[str, WorkflowMetadata] = {}
        self._intent_map: Dict[str, str] = {}  # intent -> workflow_name
        self._registry_lock = Lock()
        self._stats = {
            "total_registrations": 0,
            "total_invocations": 0,
            "total_matches": 0,
            "blocked_by_guardrails": 0
        }
        
        # Guardrail patterns (simple regex-based for now)
        self._blocked_patterns = [
            r"ignore\s+(previous|all)\s+instructions",
            r"you\s+are\s+now\s+",
            r"pretend\s+to\s+be",
            r"act\s+as\s+if",
            r"disregard\s+your\s+programming"
        ]
        
        logger.info("[WorkflowRegistry] Initialized - Ready for workflow registration")
    
    # =========================================================================
    # LEVEL 4: CORE REGISTRATION (Implemented)
    # =========================================================================
    
    def register(self, metadata: WorkflowMetadata) -> None:
        """
        Register a workflow with its metadata.
        
        Called by workflow modules at import/load time.
        Thread-safe for concurrent registration.
        
        Args:
            metadata: WorkflowMetadata with all workflow details
        
        Raises:
            ValueError: If workflow with same name already registered
        """
        with self._registry_lock:
            if metadata.name in self._workflows:
                logger.warning(f"[WorkflowRegistry] Workflow '{metadata.name}' already registered, updating...")
            
            self._workflows[metadata.name] = metadata
            
            # Build intent-to-workflow mapping
            for intent in metadata.intents:
                intent_lower = intent.lower().strip()
                if intent_lower in self._intent_map:
                    existing = self._intent_map[intent_lower]
                    existing_priority = self._workflows[existing].priority
                    if metadata.priority > existing_priority:
                        self._intent_map[intent_lower] = metadata.name
                        logger.debug(f"[WorkflowRegistry] Intent '{intent}' remapped: {existing} -> {metadata.name} (higher priority)")
                else:
                    self._intent_map[intent_lower] = metadata.name
            
            self._stats["total_registrations"] += 1
            
            logger.info(
                f"[WorkflowRegistry] Registered: {metadata.name} "
                f"(intents={len(metadata.intents)}, keywords={len(metadata.keywords)}, priority={metadata.priority})"
            )
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a workflow by name.
        
        Args:
            name: Workflow name to remove
            
        Returns:
            True if removed, False if not found
        """
        with self._registry_lock:
            if name not in self._workflows:
                return False
            
            workflow = self._workflows.pop(name)
            
            # Remove from intent map
            for intent in workflow.intents:
                intent_lower = intent.lower().strip()
                if self._intent_map.get(intent_lower) == name:
                    del self._intent_map[intent_lower]
            
            logger.info(f"[WorkflowRegistry] Unregistered: {name}")
            return True
    
    def get(self, name: str) -> Optional[WorkflowMetadata]:
        """Get workflow metadata by name."""
        return self._workflows.get(name)
    
    def list_all(self) -> List[WorkflowMetadata]:
        """Get all registered workflows."""
        return list(self._workflows.values())
    
    def list_enabled(self) -> List[WorkflowMetadata]:
        """Get all enabled workflows."""
        return [w for w in self._workflows.values() if w.is_enabled]
    
    def list_by_tag(self, tag: str) -> List[WorkflowMetadata]:
        """Get workflows with a specific tag."""
        return [w for w in self._workflows.values() if tag in w.tags]
    
    # =========================================================================
    # GUARDRAILS: INPUT VALIDATION (Placeholder + Basic Implementation)
    # =========================================================================
    
    def validate_input(self, query: str) -> GuardrailResult:
        """
        Validate user input before routing.
        
        Checks for:
        - Empty or too short input
        - Prompt injection patterns
        - Suspicious patterns
        
        Args:
            query: User input to validate
            
        Returns:
            GuardrailResult with status and reason
        """
        if not query or len(query.strip()) < 2:
            return GuardrailResult(
                status=GuardrailStatus.BLOCKED,
                reason="Input too short",
                suggested_response="Please provide more details about what you need."
            )
        
        if len(query) > 10000:
            return GuardrailResult(
                status=GuardrailStatus.BLOCKED,
                reason="Input too long",
                suggested_response="Please provide a shorter, more focused query."
            )
        
        # Check for prompt injection patterns
        query_lower = query.lower()
        for pattern in self._blocked_patterns:
            if re.search(pattern, query_lower):
                self._stats["blocked_by_guardrails"] += 1
                logger.warning(f"[WorkflowRegistry] Guardrail blocked potential injection: {pattern}")
                return GuardrailResult(
                    status=GuardrailStatus.BLOCKED,
                    reason="Suspicious input pattern detected",
                    suggested_response="I can help you with industrial automation and procurement questions. Please rephrase your request."
                )
        
        return GuardrailResult(
            status=GuardrailStatus.PASSED,
            reason="Input validation passed",
            confidence=1.0
        )
    
    # =========================================================================
    # LEVEL 4: INTENT MATCHING (Implemented with Confidence)
    # =========================================================================
    
    def match_intent(
        self, 
        intent: str, 
        is_solution: bool = False,
        confidence: float = 1.0
    ) -> MatchResult:
        """
        Find the best matching workflow for an intent.
        
        Matching priority:
        1. If is_solution=True, force match to "solution" workflow
        2. Exact intent match in intent_map
        3. Fallback to "product_info" for unknown intents
        
        Args:
            intent: Classification intent from classify_intent_tool
            is_solution: Override flag for complex system detection
            confidence: Confidence from classifier (0.0-1.0)
        
        Returns:
            MatchResult with workflow, confidence, and alternatives
        """
        self._stats["total_matches"] += 1
        
        # Priority 1: is_solution flag forces solution workflow
        if is_solution and "solution" in self._workflows:
            workflow = self._workflows["solution"]
            logger.debug(f"[WorkflowRegistry] is_solution=True, forcing 'solution' workflow")
            return MatchResult(
                workflow=workflow,
                confidence=max(confidence, 0.9),  # High confidence for explicit flag
                reasoning="Solution flag detected - routing to Solution Workflow"
            )
        
        # Priority 2: Exact intent match
        intent_lower = intent.lower().strip()
        if intent_lower in self._intent_map:
            workflow_name = self._intent_map[intent_lower]
            workflow = self._workflows.get(workflow_name)
            if workflow and workflow.is_enabled:
                logger.debug(f"[WorkflowRegistry] Intent '{intent}' matched to '{workflow_name}'")
                
                # Check if confidence is too low - may need disambiguation
                if confidence < workflow.min_confidence_threshold:
                    alternatives = self._find_alternatives(intent, exclude=workflow_name)
                    return MatchResult(
                        workflow=workflow,
                        confidence=confidence,
                        reasoning=f"Intent '{intent}' matched but confidence is low ({confidence:.2f})",
                        needs_disambiguation=len(alternatives) > 0,
                        alternatives=alternatives,
                        disambiguation_question=self._generate_disambiguation_question(workflow, alternatives)
                    )
                
                return MatchResult(
                    workflow=workflow,
                    confidence=confidence,
                    reasoning=f"Intent '{intent}' matched to '{workflow_name}'"
                )
        
        # Priority 3: Fallback to engenie_chat for unknown intents
        fallback = self._workflows.get("engenie_chat")
        if fallback and fallback.is_enabled:
            logger.debug(f"[WorkflowRegistry] Unknown intent '{intent}', falling back to 'engenie_chat'")
            return MatchResult(
                workflow=fallback,
                confidence=0.5,  # Lower confidence for fallback
                reasoning=f"Unknown intent '{intent}', defaulting to EnGenie Chat"
            )
        
        logger.warning(f"[WorkflowRegistry] No matching workflow for intent '{intent}'")
        return MatchResult(
            workflow=None,
            confidence=0.0,
            reasoning=f"No matching workflow found for intent '{intent}'"
        )
    
    def _find_alternatives(
        self, 
        intent: str, 
        exclude: str = None,
        max_alternatives: int = 2
    ) -> List[Tuple[WorkflowMetadata, float]]:
        """Find alternative workflows that might also handle this intent."""
        alternatives = []
        for workflow in self.list_enabled():
            if workflow.name == exclude:
                continue
            # Simple heuristic: check keyword overlap
            if workflow.matches_keyword(intent) > 0:
                alternatives.append((workflow, 0.3))
        return alternatives[:max_alternatives]
    
    def _generate_disambiguation_question(
        self,
        primary: WorkflowMetadata,
        alternatives: List[Tuple[WorkflowMetadata, float]]
    ) -> Optional[str]:
        """Generate a clarifying question when confidence is low."""
        if not alternatives:
            return None
        
        options = [primary.display_name] + [a[0].display_name for a in alternatives]
        options_str = ", ".join(options[:-1]) + f", or {options[-1]}"
        
        return f"I want to make sure I understand correctly. Are you looking for help with {options_str}?"
    
    # =========================================================================
    # LEVEL 4: WORKFLOW INVOCATION (Implemented)
    # =========================================================================
    
    def invoke(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Invoke a workflow by name with given arguments.
        
        Calls the workflow's entry_function with the provided kwargs.
        
        Args:
            name: Workflow name (e.g., "solution", "instrument_identifier")
            **kwargs: Arguments to pass to entry_function
        
        Returns:
            Result dict from the workflow
        
        Raises:
            ValueError: If workflow not found or disabled
        """
        workflow = self._workflows.get(name)
        
        if not workflow:
            logger.error(f"[WorkflowRegistry] Workflow '{name}' not found")
            raise ValueError(f"Workflow '{name}' not registered")
        
        if not workflow.is_enabled:
            logger.error(f"[WorkflowRegistry] Workflow '{name}' is disabled")
            raise ValueError(f"Workflow '{name}' is disabled")
        
        self._stats["total_invocations"] += 1
        
        start_time = time.time()
        logger.info(f"[WorkflowRegistry] Invoking '{name}' with keys: {list(kwargs.keys())}")
        
        try:
            result = workflow.entry_function(**kwargs)
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"[WorkflowRegistry] Workflow '{name}' completed in {elapsed_ms:.1f}ms")
            return result
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"[WorkflowRegistry] Workflow '{name}' failed after {elapsed_ms:.1f}ms: {e}")
            raise
    
    def invoke_safe(self, name: str, query: str, **kwargs) -> Dict[str, Any]:
        """
        Invoke a workflow with guardrail validation.
        
        Validates input before invoking the workflow.
        
        Args:
            name: Workflow name
            query: User query (for guardrail validation)
            **kwargs: Additional arguments for workflow
        
        Returns:
            Result dict with 'success' flag and either 'data' or 'error'
        """
        # Run guardrail check
        guardrail_result = self.validate_input(query)
        
        if not guardrail_result.is_safe:
            return {
                "success": False,
                "error": guardrail_result.reason,
                "suggested_response": guardrail_result.suggested_response,
                "guardrail_status": guardrail_result.status.value
            }
        
        try:
            # Add query to kwargs if not already present
            if "user_input" not in kwargs:
                kwargs["user_input"] = query
            
            result = self.invoke(name, **kwargs)
            return {
                "success": True,
                "data": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # =========================================================================
    # LEVEL 5: SEMANTIC MATCHING (Implemented)
    # =========================================================================
    
    def match_semantic(self, query: str, top_k: int = 3) -> List[MatchResult]:
        """
        Find workflows by semantic similarity to query.
        
        Uses Gemini text-embedding-004 to generate embeddings and compares
        with pre-computed workflow embeddings using cosine similarity.
        
        Falls back to keyword matching if embedding generation fails.
        
        Args:
            query: User query text
            top_k: Number of top matches to return
        
        Returns:
            List of MatchResult ordered by relevance (highest first)
        """
        self._stats["semantic_matches"] = self._stats.get("semantic_matches", 0) + 1
        
        # Generate embedding for the query
        query_embedding = self.generate_embedding(query)
        
        if not query_embedding:
            logger.debug("[WorkflowRegistry] Embedding generation failed, using keyword fallback")
            return self._keyword_fallback_match(query, top_k)
        
        # Ensure all workflows have embeddings
        self._ensure_workflow_embeddings()
        
        # Calculate similarity scores
        scored_workflows = []
        for workflow in self.list_enabled():
            if workflow.embedding and len(workflow.embedding) > 0:
                similarity = self._cosine_similarity(query_embedding, workflow.embedding)
                scored_workflows.append((workflow, similarity))
            else:
                # Use keyword matching as fallback for workflows without embeddings
                keyword_score = workflow.matches_keyword(query)
                if keyword_score > 0:
                    # Normalize keyword score to 0-1 range (rough approximation)
                    similarity = min(0.5 + (keyword_score * 0.1), 0.8)
                    scored_workflows.append((workflow, similarity))
        
        # Sort by similarity descending
        scored_workflows.sort(key=lambda x: (-x[1], -x[0].priority))
        
        results = []
        for workflow, similarity in scored_workflows[:top_k]:
            results.append(MatchResult(
                workflow=workflow,
                confidence=similarity,
                reasoning=f"Semantic similarity: {similarity:.3f}"
            ))
        
        if results:
            logger.info(
                f"[WorkflowRegistry] Semantic match: top result '{results[0].workflow.name}' "
                f"with confidence {results[0].confidence:.3f}"
            )
        
        return results
    
    def _keyword_fallback_match(self, query: str, top_k: int = 3) -> List[MatchResult]:
        """Fallback to keyword matching when embeddings unavailable."""
        scored_workflows = []
        for workflow in self.list_enabled():
            score = workflow.matches_keyword(query)
            if score > 0:
                scored_workflows.append((workflow, score))
        
        # Sort by score descending
        scored_workflows.sort(key=lambda x: (-x[1], -x[0].priority))
        
        results = []
        for workflow, score in scored_workflows[:top_k]:
            # Normalize score to 0-1 range
            confidence = min(score / 3.0, 1.0)
            results.append(MatchResult(
                workflow=workflow,
                confidence=confidence,
                reasoning=f"Keyword match score: {score}"
            ))
        
        return results
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using Gemini text-embedding-004 model.
        
        Uses the Google Generative AI library to generate embeddings.
        Falls back gracefully if API is unavailable.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector (768 dimensions for text-embedding-004)
            Empty list if generation fails
        """
        if not text or len(text.strip()) < 2:
            return []
        
        try:
            import google.generativeai as genai
            import os
            
            # Configure with API key if not already done
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
            
            # Generate embedding using Gemini
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text[:8000],  # Limit text length
                task_type="retrieval_document"
            )
            
            embedding = result.get('embedding', [])
            
            if embedding:
                logger.debug(f"[WorkflowRegistry] Generated embedding of dimension {len(embedding)}")
                self._stats["embeddings_generated"] = self._stats.get("embeddings_generated", 0) + 1
            
            return embedding
            
        except ImportError:
            logger.warning("[WorkflowRegistry] google-generativeai not installed, embedding unavailable")
            return []
        except Exception as e:
            logger.warning(f"[WorkflowRegistry] Embedding generation failed: {e}")
            self._stats["embedding_failures"] = self._stats.get("embedding_failures", 0) + 1
            return []
    
    def _ensure_workflow_embeddings(self) -> None:
        """Generate and cache embeddings for all registered workflows."""
        for workflow in self._workflows.values():
            if workflow.embedding is None or len(workflow.embedding) == 0:
                # Generate embedding from workflow description
                embedding_text = f"{workflow.display_name}: {workflow.description}"
                if workflow.keywords:
                    embedding_text += f". Keywords: {', '.join(workflow.keywords[:10])}"
                
                embedding = self.generate_embedding(embedding_text)
                if embedding:
                    workflow.embedding = embedding
                    logger.info(f"[WorkflowRegistry] Generated embedding for workflow '{workflow.name}'")
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            a: First embedding vector
            b: Second embedding vector
        
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not a or not b or len(a) != len(b):
            return 0.0
        
        import math
        
        # Compute dot product
        dot_product = sum(x * y for x, y in zip(a, b))
        
        # Compute magnitudes
        magnitude_a = math.sqrt(sum(x ** 2 for x in a))
        magnitude_b = math.sqrt(sum(x ** 2 for x in b))
        
        # Avoid division by zero
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        # Cosine similarity
        similarity = dot_product / (magnitude_a * magnitude_b)
        
        # Clamp to [0, 1] range (similarity can be negative for very dissimilar texts)
        return max(0.0, min(1.0, similarity))
    
    def refresh_embeddings(self) -> int:
        """
        Force regeneration of all workflow embeddings.
        
        Useful after updating workflow descriptions or switching models.
        
        Returns:
            Number of embeddings regenerated
        """
        count = 0
        for workflow in self._workflows.values():
            workflow.embedding = None  # Clear existing
        
        self._ensure_workflow_embeddings()
        
        for workflow in self._workflows.values():
            if workflow.embedding:
                count += 1
        
        logger.info(f"[WorkflowRegistry] Refreshed {count} workflow embeddings")
        return count
    
    # =========================================================================
    # LEVEL 5: SELF-HEALING INVOCATION (Implemented)
    # =========================================================================
    
    def invoke_with_retry(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Invoke a workflow with automatic retry on transient failures.
        
        Uses the workflow's RetryPolicy for backoff strategy. Automatically
        retries on RESOURCE_EXHAUSTED, TIMEOUT, and other configured exceptions.
        
        Args:
            name: Workflow name
            **kwargs: Arguments for workflow
        
        Returns:
            Result dict from the workflow
        
        Raises:
            ValueError: If workflow not found or disabled
            Exception: If all retries exhausted or non-retryable error
        """
        workflow = self.get(name)
        if not workflow:
            raise ValueError(f"Workflow '{name}' not registered")
        
        if not workflow.is_enabled:
            raise ValueError(f"Workflow '{name}' is disabled")
        
        # Get retry policy (use default if none configured)
        policy = workflow.retry_policy or RetryPolicy(strategy=RetryStrategy.NONE)
        
        # If no retry strategy, just invoke directly
        if policy.strategy == RetryStrategy.NONE:
            logger.debug(f"[WorkflowRegistry] No retry policy for '{name}', invoking directly")
            return self.invoke(name, **kwargs)
        
        last_error = None
        total_attempts = policy.max_retries + 1
        
        for attempt in range(total_attempts):
            try:
                start_time = time.time()
                result = self.invoke(name, **kwargs)
                
                if attempt > 0:
                    logger.info(
                        f"[WorkflowRegistry] Workflow '{name}' succeeded on attempt {attempt + 1}"
                    )
                
                return result
                
            except Exception as e:
                last_error = e
                error_str = str(e)
                error_type = type(e).__name__
                
                # Check if error is retryable
                is_retryable = self._is_retryable_error(e, policy)
                
                if not is_retryable:
                    logger.error(
                        f"[WorkflowRegistry] Non-retryable error in '{name}': {error_type}"
                    )
                    raise
                
                # Check if we have retries left
                if attempt >= policy.max_retries:
                    logger.error(
                        f"[WorkflowRegistry] Workflow '{name}' failed after {total_attempts} attempts: {e}"
                    )
                    self._stats["retry_exhausted"] = self._stats.get("retry_exhausted", 0) + 1
                    raise
                
                # Calculate backoff delay
                delay_ms = self._calculate_retry_delay(policy, attempt)
                
                logger.warning(
                    f"[WorkflowRegistry] Workflow '{name}' failed (attempt {attempt + 1}/{total_attempts}), "
                    f"retrying in {delay_ms}ms: {error_type} - {error_str[:100]}"
                )
                
                self._stats["total_retries"] = self._stats.get("total_retries", 0) + 1
                
                # Wait before retry
                time.sleep(delay_ms / 1000.0)
        
        # Should not reach here, but just in case
        raise last_error
    
    def _is_retryable_error(self, error: Exception, policy: RetryPolicy) -> bool:
        """Check if an error is retryable based on policy."""
        error_str = str(error).upper()
        error_type = type(error).__name__.upper()
        
        for pattern in policy.retryable_exceptions:
            pattern_upper = pattern.upper()
            if pattern_upper in error_str or pattern_upper in error_type:
                return True
        
        # Also check for common retryable HTTP status codes in error messages
        retryable_patterns = [
            "429", "503", "502", "504",  # HTTP status codes
            "RATE_LIMIT", "QUOTA", "THROTTL",  # Rate limiting
            "UNAVAILABLE", "OVERLOADED",  # Service unavailable
            "DEADLINE_EXCEEDED", "CONNECTION"  # Connection issues
        ]
        
        for pattern in retryable_patterns:
            if pattern in error_str:
                return True
        
        return False
    
    def _calculate_retry_delay(self, policy: RetryPolicy, attempt: int) -> int:
        """Calculate delay in milliseconds based on retry strategy."""
        if policy.strategy == RetryStrategy.IMMEDIATE:
            return 0
        
        elif policy.strategy == RetryStrategy.LINEAR:
            # Linear: base * (attempt + 1)
            delay = policy.base_delay_ms * (attempt + 1)
        
        elif policy.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            # Exponential: base * 2^attempt (with jitter)
            import random
            base_delay = policy.base_delay_ms * (2 ** attempt)
            # Add 0-25% jitter to prevent thundering herd
            jitter = random.uniform(0, 0.25) * base_delay
            delay = int(base_delay + jitter)
        
        else:
            delay = policy.base_delay_ms
        
        # Cap at max delay
        return min(delay, policy.max_delay_ms)
    
    # =========================================================================
    # LEVEL 5: A/B TESTING (Implemented)
    # =========================================================================
    
    def __init_experiments(self) -> None:
        """Initialize A/B testing experiment storage."""
        if not hasattr(self, '_experiments'):
            self._experiments: Dict[str, 'Experiment'] = {}
            self._session_assignments: Dict[str, Dict[str, str]] = {}  # session_id -> {workflow -> variant}
            self._experiment_metrics: Dict[str, Dict[str, Any]] = {}
            self._experiments_file = None
    
    def get_variant(
        self, 
        name: str, 
        session_id: str = None,
        experiment_id: str = None
    ) -> Optional[WorkflowMetadata]:
        """
        Get workflow variant for A/B testing.
        
        Selects the appropriate variant based on active experiments and
        session assignment. Uses consistent hashing to ensure the same
        session always gets the same variant.
        
        Args:
            name: Base workflow name
            session_id: Session ID for consistent assignment
            experiment_id: Specific experiment to use (optional)
        
        Returns:
            WorkflowMetadata for selected variant
        """
        self.__init_experiments()
        
        # If no session_id, return default workflow
        if not session_id:
            return self.get(name)
        
        # Check for active experiment on this workflow
        experiment = self._find_active_experiment(name, experiment_id)
        
        if not experiment:
            return self.get(name)
        
        # Check if session already has an assignment
        if session_id in self._session_assignments:
            if name in self._session_assignments[session_id]:
                variant_name = self._session_assignments[session_id][name]
                logger.debug(
                    f"[WorkflowRegistry] A/B: Session {session_id[:8]}... assigned to '{variant_name}' (cached)"
                )
                return self.get(variant_name)
        
        # Assign variant using consistent hashing
        variant_name = self._assign_variant(session_id, experiment)
        
        # Cache the assignment
        if session_id not in self._session_assignments:
            self._session_assignments[session_id] = {}
        self._session_assignments[session_id][name] = variant_name
        
        # Track assignment metrics
        self._track_experiment_event(experiment.id, "assignment", variant_name)
        
        logger.info(
            f"[WorkflowRegistry] A/B: Session {session_id[:8]}... assigned to '{variant_name}' "
            f"(experiment: {experiment.id})"
        )
        
        return self.get(variant_name)
    
    def _find_active_experiment(
        self, 
        workflow_name: str, 
        experiment_id: str = None
    ) -> Optional['Experiment']:
        """Find an active experiment for the given workflow."""
        self.__init_experiments()
        
        from datetime import datetime
        now = datetime.now()
        
        for exp in self._experiments.values():
            # Check if experiment is for this workflow
            if exp.base_workflow != workflow_name:
                continue
            
            # Check if specific experiment requested
            if experiment_id and exp.id != experiment_id:
                continue
            
            # Check if experiment is active
            if not exp.is_active:
                continue
            
            # Check time bounds
            if exp.start_time > now:
                continue
            if exp.end_time and exp.end_time < now:
                continue
            
            return exp
        
        return None
    
    def _assign_variant(self, session_id: str, experiment: 'Experiment') -> str:
        """Assign a variant based on session ID and traffic split."""
        import hashlib
        
        # Use consistent hashing for sticky assignment
        hash_input = f"{session_id}:{experiment.id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Determine if session is in treatment group
        bucket = hash_value % 100
        in_treatment = bucket < (experiment.traffic_percentage * 100)
        
        if in_treatment:
            return experiment.variant_workflow
        else:
            return experiment.base_workflow
    
    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        base_workflow: str,
        variant_workflow: str,
        traffic_percentage: float = 0.5,
        start_time: 'datetime' = None,
        end_time: 'datetime' = None
    ) -> 'Experiment':
        """
        Create a new A/B testing experiment.
        
        Args:
            experiment_id: Unique identifier for the experiment
            name: Human-readable name
            base_workflow: Control workflow name
            variant_workflow: Treatment workflow name
            traffic_percentage: Percentage of traffic to send to variant (0.0-1.0)
            start_time: When to start (defaults to now)
            end_time: When to end (optional)
        
        Returns:
            Created Experiment object
        """
        self.__init_experiments()
        
        from datetime import datetime
        
        # Validate workflows exist
        if not self.get(base_workflow):
            raise ValueError(f"Base workflow '{base_workflow}' not registered")
        if not self.get(variant_workflow):
            raise ValueError(f"Variant workflow '{variant_workflow}' not registered")
        
        experiment = Experiment(
            id=experiment_id,
            name=name,
            base_workflow=base_workflow,
            variant_workflow=variant_workflow,
            traffic_percentage=max(0.0, min(1.0, traffic_percentage)),
            start_time=start_time or datetime.now(),
            end_time=end_time,
            is_active=True
        )
        
        self._experiments[experiment_id] = experiment
        self._experiment_metrics[experiment_id] = {
            "assignments": {"control": 0, "treatment": 0},
            "invocations": {"control": 0, "treatment": 0},
            "successes": {"control": 0, "treatment": 0},
            "failures": {"control": 0, "treatment": 0},
            "latencies": {"control": [], "treatment": []}
        }
        
        logger.info(
            f"[WorkflowRegistry] A/B: Created experiment '{experiment_id}' "
            f"({base_workflow} vs {variant_workflow}, {traffic_percentage*100:.0f}% treatment)"
        )
        
        self._save_experiments()
        return experiment
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an active experiment."""
        self.__init_experiments()
        
        if experiment_id not in self._experiments:
            return False
        
        self._experiments[experiment_id].is_active = False
        logger.info(f"[WorkflowRegistry] A/B: Stopped experiment '{experiment_id}'")
        
        self._save_experiments()
        return True
    
    def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for an experiment."""
        self.__init_experiments()
        
        if experiment_id not in self._experiments:
            return None
        
        experiment = self._experiments[experiment_id]
        metrics = self._experiment_metrics.get(experiment_id, {})
        
        # Calculate statistics
        control_latencies = metrics.get("latencies", {}).get("control", [])
        treatment_latencies = metrics.get("latencies", {}).get("treatment", [])
        
        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "base_workflow": experiment.base_workflow,
            "variant_workflow": experiment.variant_workflow,
            "traffic_percentage": experiment.traffic_percentage,
            "is_active": experiment.is_active,
            "metrics": {
                "control": {
                    "assignments": metrics.get("assignments", {}).get("control", 0),
                    "invocations": metrics.get("invocations", {}).get("control", 0),
                    "successes": metrics.get("successes", {}).get("control", 0),
                    "failures": metrics.get("failures", {}).get("control", 0),
                    "avg_latency_ms": sum(control_latencies) / len(control_latencies) if control_latencies else 0
                },
                "treatment": {
                    "assignments": metrics.get("assignments", {}).get("treatment", 0),
                    "invocations": metrics.get("invocations", {}).get("treatment", 0),
                    "successes": metrics.get("successes", {}).get("treatment", 0),
                    "failures": metrics.get("failures", {}).get("treatment", 0),
                    "avg_latency_ms": sum(treatment_latencies) / len(treatment_latencies) if treatment_latencies else 0
                }
            }
        }
    
    def list_experiments(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """List all experiments."""
        self.__init_experiments()
        
        results = []
        for exp in self._experiments.values():
            if active_only and not exp.is_active:
                continue
            results.append({
                "id": exp.id,
                "name": exp.name,
                "base_workflow": exp.base_workflow,
                "variant_workflow": exp.variant_workflow,
                "traffic_percentage": exp.traffic_percentage,
                "is_active": exp.is_active
            })
        return results
    
    def _track_experiment_event(
        self, 
        experiment_id: str, 
        event_type: str, 
        variant: str,
        value: Any = None
    ) -> None:
        """Track an experiment event for metrics."""
        if experiment_id not in self._experiment_metrics:
            return
        
        metrics = self._experiment_metrics[experiment_id]
        
        # Determine group (control vs treatment)
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return
        
        group = "treatment" if variant == experiment.variant_workflow else "control"
        
        if event_type == "assignment":
            metrics["assignments"][group] = metrics["assignments"].get(group, 0) + 1
        elif event_type == "invocation":
            metrics["invocations"][group] = metrics["invocations"].get(group, 0) + 1
        elif event_type == "success":
            metrics["successes"][group] = metrics["successes"].get(group, 0) + 1
        elif event_type == "failure":
            metrics["failures"][group] = metrics["failures"].get(group, 0) + 1
        elif event_type == "latency" and value is not None:
            if group not in metrics["latencies"]:
                metrics["latencies"][group] = []
            metrics["latencies"][group].append(value)
            # Keep only last 100 latencies per group
            if len(metrics["latencies"][group]) > 100:
                metrics["latencies"][group] = metrics["latencies"][group][-100:]
    
    def _save_experiments(self) -> None:
        """Save experiments to JSON file for persistence."""
        if not self._experiments_file:
            import os
            self._experiments_file = os.path.join(
                os.path.dirname(__file__), 
                "experiments.json"
            )
        
        try:
            import json
            from datetime import datetime
            
            data = {
                "experiments": [
                    {
                        "id": exp.id,
                        "name": exp.name,
                        "base_workflow": exp.base_workflow,
                        "variant_workflow": exp.variant_workflow,
                        "traffic_percentage": exp.traffic_percentage,
                        "start_time": exp.start_time.isoformat(),
                        "end_time": exp.end_time.isoformat() if exp.end_time else None,
                        "is_active": exp.is_active
                    }
                    for exp in self._experiments.values()
                ],
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self._experiments_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"[WorkflowRegistry] Failed to save experiments: {e}")
    
    def _load_experiments(self) -> None:
        """Load experiments from JSON file."""
        if not self._experiments_file:
            import os
            self._experiments_file = os.path.join(
                os.path.dirname(__file__), 
                "experiments.json"
            )
        
        try:
            import json
            import os
            from datetime import datetime
            
            if not os.path.exists(self._experiments_file):
                return
            
            with open(self._experiments_file, 'r') as f:
                data = json.load(f)
            
            for exp_data in data.get("experiments", []):
                experiment = Experiment(
                    id=exp_data["id"],
                    name=exp_data["name"],
                    base_workflow=exp_data["base_workflow"],
                    variant_workflow=exp_data["variant_workflow"],
                    traffic_percentage=exp_data["traffic_percentage"],
                    start_time=datetime.fromisoformat(exp_data["start_time"]),
                    end_time=datetime.fromisoformat(exp_data["end_time"]) if exp_data.get("end_time") else None,
                    is_active=exp_data["is_active"]
                )
                self._experiments[experiment.id] = experiment
                self._experiment_metrics[experiment.id] = {
                    "assignments": {"control": 0, "treatment": 0},
                    "invocations": {"control": 0, "treatment": 0},
                    "successes": {"control": 0, "treatment": 0},
                    "failures": {"control": 0, "treatment": 0},
                    "latencies": {"control": [], "treatment": []}
                }
            
            logger.info(f"[WorkflowRegistry] Loaded {len(self._experiments)} experiments from file")
            
        except Exception as e:
            logger.warning(f"[WorkflowRegistry] Failed to load experiments: {e}")
    
    
    # =========================================================================
    # STATISTICS & UTILITIES
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "registered_workflows": len(self._workflows),
            "enabled_workflows": len(self.list_enabled()),
            "intent_mappings": len(self._intent_map),
            "total_registrations": self._stats["total_registrations"],
            "total_invocations": self._stats["total_invocations"],
            "total_matches": self._stats["total_matches"],
            "blocked_by_guardrails": self._stats["blocked_by_guardrails"],
            "workflows": [w.name for w in self._workflows.values()],
            "intent_map": dict(self._intent_map)
        }
    
    def clear(self) -> None:
        """Clear all registrations (for testing)."""
        with self._registry_lock:
            self._workflows.clear()
            self._intent_map.clear()
            self._stats = {
                "total_registrations": 0,
                "total_invocations": 0,
                "total_matches": 0,
                "blocked_by_guardrails": 0
            }
            logger.info("[WorkflowRegistry] Cleared all registrations")
    
    def enable_workflow(self, name: str) -> bool:
        """Enable a workflow."""
        workflow = self._workflows.get(name)
        if workflow:
            workflow.is_enabled = True
            logger.info(f"[WorkflowRegistry] Enabled workflow: {name}")
            return True
        return False
    
    def disable_workflow(self, name: str) -> bool:
        """Disable a workflow without unregistering."""
        workflow = self._workflows.get(name)
        if workflow:
            workflow.is_enabled = False
            logger.info(f"[WorkflowRegistry] Disabled workflow: {name}")
            return True
        return False


# =============================================================================
# GLOBAL SINGLETON ACCESS
# =============================================================================

_registry: Optional[WorkflowRegistry] = None

def get_workflow_registry() -> WorkflowRegistry:
    """Get the global workflow registry singleton."""
    global _registry
    if _registry is None:
        _registry = WorkflowRegistry()
    return _registry


# =============================================================================
# CONVENIENCE DECORATORS (Level 5 Ready)
# =============================================================================

def workflow(
    name: str,
    display_name: str,
    description: str,
    intents: List[str],
    keywords: List[str] = None,
    capabilities: List[str] = None,
    priority: int = 50,
    tags: List[str] = None,
    min_confidence: float = 0.5
):
    """
    Decorator to auto-register a workflow function.
    
    Usage:
        @workflow(
            name="solution",
            display_name="Solution Workflow",
            description="Handles complex multi-instrument systems",
            intents=["solution", "system", "design"],
            priority=100
        )
        def run_solution_workflow(user_input: str, session_id: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        metadata = WorkflowMetadata(
            name=name,
            display_name=display_name,
            description=description,
            keywords=keywords or [],
            intents=intents,
            capabilities=capabilities or [],
            entry_function=func,
            priority=priority,
            tags=tags or [],
            min_confidence_threshold=min_confidence
        )
        get_workflow_registry().register(metadata)
        return func
    return decorator


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'WorkflowMetadata',
    'WorkflowRegistry',
    'MatchResult',
    'RetryPolicy',
    'RetryStrategy',
    'GuardrailResult',
    'GuardrailStatus',
    'Experiment',
    'get_workflow_registry',
    'workflow'
]

