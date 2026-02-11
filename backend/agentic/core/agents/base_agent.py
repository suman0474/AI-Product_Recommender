# agentic/base_agent.py
# =============================================================================
# BASE RAG AGENT - Abstract Base Class for RAG Chat Agents
# =============================================================================
#
# This module provides a base class for RAG (Retrieval-Augmented Generation)
# agents that is extended by specific implementations:
#   - StandardsChatAgent
#   - StrategyChatAgent
#   - IndexRAGAgent
#
# Common features:
#   - LLM initialization with fallback
#   - Vector store integration
#   - Prompt template management
#   - JSON output parsing
#   - Document retrieval
#   - Context building
#   - Citation extraction
#   - Singleton pattern support
#
# Usage:
#   class MyRAGAgent(BaseRAGAgent):
#       COLLECTION_TYPE = "my_collection"
#       PROMPT_KEY = "MY_PROMPT"
#
#       def run(self, question: str, **kwargs) -> Dict[str, Any]:
#           # Custom implementation
#           pass
#
# =============================================================================

import json
import logging
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, TypeVar

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Type variable for agent subclasses
T = TypeVar('T', bound='BaseRAGAgent')


class BaseRAGAgent(ABC):
    """
    Abstract base class for RAG chat agents.

    Provides common functionality:
    - LLM initialization with fallback support
    - Vector store connection
    - Prompt/chain setup
    - Document retrieval
    - Context building
    - Citation extraction

    Subclasses must implement:
    - run(): Main execution method
    - COLLECTION_TYPE: Vector store collection to query
    - PROMPT_TEMPLATE or PROMPT_KEY: Prompt configuration

    Class Attributes:
        COLLECTION_TYPE: Vector store collection type (e.g., "standards")
        DEFAULT_MODEL: Default LLM model (default: "gemini-2.5-flash")
        DEFAULT_TEMPERATURE: Default temperature (default: 0.1)
        DEFAULT_MAX_TOKENS: Default max tokens (default: 2000)
        DEFAULT_TIMEOUT: Default timeout in seconds (default: 120)
        PROMPT_TEMPLATE: Static prompt template string (optional)
        PROMPT_KEY: Key for loading prompt from prompts_library (optional)
        PROMPT_SECTION: Section in prompts_library (default: "rag_prompts")
    """

    # Override in subclasses
    COLLECTION_TYPE: str = "default"
    DEFAULT_MODEL: str = "gemini-2.5-flash"
    DEFAULT_TEMPERATURE: float = 0.1
    DEFAULT_MAX_TOKENS: int = 2000
    DEFAULT_TIMEOUT: int = 120

    # Prompt configuration (override one of these)
    PROMPT_TEMPLATE: Optional[str] = None
    PROMPT_KEY: Optional[str] = None
    PROMPT_SECTION: str = "rag_prompts"

    def __init__(
        self,
        llm=None,
        temperature: float = None,
        max_tokens: int = None,
        model: str = None,
        timeout: int = None
    ):
        """
        Initialize the RAG agent.

        Args:
            llm: Pre-configured LLM instance (optional)
            temperature: LLM temperature (default: class DEFAULT_TEMPERATURE)
            max_tokens: Max tokens for generation (default: class DEFAULT_MAX_TOKENS)
            model: LLM model name (default: class DEFAULT_MODEL)
            timeout: Request timeout in seconds (default: class DEFAULT_TIMEOUT)
        """
        # Store configuration
        self.temperature = temperature if temperature is not None else self.DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT

        # Initialize LLM
        if llm is not None:
            self.llm = llm
        else:
            self.llm = self._create_llm()

        # Initialize vector store
        self.vector_store = self._get_vector_store()

        # Initialize prompt and chain
        self.prompt = self._create_prompt()
        self.parser = JsonOutputParser()
        self.chain = self.prompt | self.llm | self.parser

        logger.info(
            f"[{self.__class__.__name__}] Initialized: "
            f"model={self.model}, temp={self.temperature}, "
            f"collection={self.COLLECTION_TYPE}"
        )

    # =========================================================================
    # INITIALIZATION HELPERS
    # =========================================================================

    def _create_llm(self):
        """
        Create LLM instance with fallback support.

        Returns:
            Configured LLM instance
        """
        from services.llm.fallback import create_llm_with_fallback

        return create_llm_with_fallback(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout
        )

    def _get_vector_store(self):
        """
        Get vector store instance.

        Returns:
            Vector store instance
        """
        from agentic.rag.vector_store import get_vector_store
        return get_vector_store()

    def _create_prompt(self) -> ChatPromptTemplate:
        """
        Create prompt template.

        Uses PROMPT_TEMPLATE if defined, otherwise loads from prompts_library
        using PROMPT_KEY.

        Returns:
            ChatPromptTemplate instance
        """
        if self.PROMPT_TEMPLATE:
            return ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)

        if self.PROMPT_KEY:
            from prompts_library import load_prompt_sections
            prompts = load_prompt_sections(self.PROMPT_SECTION)
            template = prompts.get(self.PROMPT_KEY, "")
            if not template:
                raise ValueError(
                    f"Prompt key '{self.PROMPT_KEY}' not found in "
                    f"section '{self.PROMPT_SECTION}'"
                )
            return ChatPromptTemplate.from_template(template)

        raise ValueError(
            f"{self.__class__.__name__} must define either "
            "PROMPT_TEMPLATE or PROMPT_KEY"
        )

    # =========================================================================
    # ABSTRACT METHODS
    # =========================================================================

    @abstractmethod
    def run(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the RAG pipeline.

        Subclasses must implement this method to:
        1. Retrieve relevant documents
        2. Build context
        3. Generate response
        4. Return structured result

        Args:
            question: User's question
            **kwargs: Additional parameters

        Returns:
            Dictionary with response data
        """
        pass

    # =========================================================================
    # DOCUMENT RETRIEVAL
    # =========================================================================

    def retrieve_documents(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant documents from vector store.

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filters: Optional metadata filters

        Returns:
            Dictionary with retrieval results
        """
        try:
            search_results = self.vector_store.search(
                collection_type=self.COLLECTION_TYPE,
                query=query,
                top_k=top_k,
                filters=filters
            )

            logger.info(
                f"[{self.__class__.__name__}] Retrieved "
                f"{search_results.get('result_count', 0)} documents"
            )

            return search_results

        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Retrieval error: {e}")
            return {
                "results": [],
                "result_count": 0,
                "error": str(e)
            }

    # =========================================================================
    # CONTEXT BUILDING
    # =========================================================================

    def build_context(
        self,
        search_results: Dict[str, Any],
        max_chars: int = None
    ) -> str:
        """
        Build formatted context from search results.

        Override in subclasses for custom formatting.

        Args:
            search_results: Results from vector store search
            max_chars: Maximum context length (optional)

        Returns:
            Formatted context string
        """
        results = search_results.get('results', [])

        if not results:
            return "No relevant documents found."

        context_parts = []

        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            source = metadata.get('filename', metadata.get('source', 'unknown'))
            content = result.get('content', '')
            relevance = result.get('relevance_score', 0.0)

            doc_section = f"[Document {i}: {source}]\n"
            doc_section += f"Relevance: {relevance:.3f}\n"
            doc_section += f"\nContent:\n{content}"

            context_parts.append(doc_section)

        context = "\n\n" + "="*80 + "\n\n".join(context_parts)

        # Truncate if needed
        if max_chars and len(context) > max_chars:
            context = context[:max_chars] + "\n\n[Context truncated...]"

        return context

    # =========================================================================
    # CITATION EXTRACTION
    # =========================================================================

    def extract_citations(
        self,
        search_results: Dict[str, Any],
        max_content_preview: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Extract citation information from search results.

        Args:
            search_results: Results from vector store search
            max_content_preview: Max chars for content preview

        Returns:
            List of citation dictionaries
        """
        citations = []

        for result in search_results.get('results', []):
            metadata = result.get('metadata', {})
            content = result.get('content', '')

            # Truncate content for preview
            if len(content) > max_content_preview:
                content = content[:max_content_preview] + "..."

            citation = {
                'source': metadata.get('filename', metadata.get('source', 'unknown')),
                'content': content,
                'relevance': result.get('relevance_score', 0.0),
                'metadata': metadata
            }
            citations.append(citation)

        return citations

    # =========================================================================
    # RESPONSE PARSING
    # =========================================================================

    def parse_llm_response(
        self,
        raw_response,
        fallback_value: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Parse LLM response, handling various formats.

        Args:
            raw_response: Raw LLM response (string or object)
            fallback_value: Value to return on parse failure

        Returns:
            Parsed dictionary
        """
        from utils.json_utils import extract_json_from_response

        # Extract text from response object
        if hasattr(raw_response, 'content'):
            text = raw_response.content
        elif hasattr(raw_response, 'text'):
            text = raw_response.text
        else:
            text = str(raw_response)

        # Parse JSON
        result = extract_json_from_response(text)

        if result is None:
            logger.warning(
                f"[{self.__class__.__name__}] Failed to parse response, "
                "using fallback"
            )
            return fallback_value or {}

        return result

    # =========================================================================
    # ERROR HANDLING
    # =========================================================================

    def create_error_response(
        self,
        error: Exception,
        sources_used: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized error response.

        Args:
            error: Exception that occurred
            sources_used: List of sources that were retrieved

        Returns:
            Error response dictionary
        """
        return {
            'success': False,
            'answer': f"An error occurred: {str(error)}",
            'citations': [],
            'confidence': 0.0,
            'sources_used': sources_used or [],
            'error': str(error)
        }

    def create_no_results_response(
        self,
        message: str = None
    ) -> Dict[str, Any]:
        """
        Create a response for when no documents are found.

        Args:
            message: Custom message (optional)

        Returns:
            No-results response dictionary
        """
        default_message = (
            f"I don't have any relevant {self.COLLECTION_TYPE} documents "
            "to answer this question. Check the logs for [VS-ROOT-CAUSE] messages "
            "to diagnose vector store configuration issues."
        )
        return {
            'success': False,
            'answer': message or default_message,
            'citations': [],
            'confidence': 0.0,
            'sources_used': [],
            'error': 'No documents found'
        }


# =============================================================================
# SINGLETON PATTERN SUPPORT
# =============================================================================

class AgentSingletonMeta(type):
    """
    Metaclass for creating singleton agents.

    Usage:
        class MyAgent(BaseRAGAgent, metaclass=AgentSingletonMeta):
            pass

        # First call creates instance
        agent1 = MyAgent.get_instance()
        # Subsequent calls return same instance
        agent2 = MyAgent.get_instance()
        assert agent1 is agent2
    """

    _instances: Dict[type, Any] = {}
    _locks: Dict[type, threading.Lock] = {}

    def __call__(cls, *args, **kwargs):
        """Create new instance (use get_instance for singleton)."""
        return super().__call__(*args, **kwargs)

    def get_instance(cls: Type[T], **kwargs) -> T:
        """
        Get or create singleton instance.

        Args:
            **kwargs: Arguments for first initialization only

        Returns:
            Singleton instance
        """
        if cls not in AgentSingletonMeta._locks:
            AgentSingletonMeta._locks[cls] = threading.Lock()

        with AgentSingletonMeta._locks[cls]:
            if cls not in AgentSingletonMeta._instances:
                AgentSingletonMeta._instances[cls] = cls(**kwargs)
                logger.info(f"[{cls.__name__}] Singleton instance created")

        return AgentSingletonMeta._instances[cls]

    def reset_instance(cls: Type[T]) -> None:
        """Reset singleton instance (for testing)."""
        if cls in AgentSingletonMeta._instances:
            del AgentSingletonMeta._instances[cls]
            logger.info(f"[{cls.__name__}] Singleton instance reset")


def get_agent_singleton(
    agent_class: Type[T],
    instance_holder: Dict[str, Any],
    lock_holder: Dict[str, threading.Lock] = None,
    **kwargs
) -> T:
    """
    Factory function for creating singleton agents.

    Alternative to metaclass approach for more control.

    Args:
        agent_class: Agent class to instantiate
        instance_holder: Dict to store singleton (e.g., module-level dict)
        lock_holder: Optional dict to store lock
        **kwargs: Arguments for first initialization

    Returns:
        Singleton agent instance

    Example:
        _singleton = {}

        def get_my_agent(**kwargs) -> MyAgent:
            return get_agent_singleton(MyAgent, _singleton, **kwargs)
    """
    class_name = agent_class.__name__

    # Create lock holder if not provided
    if lock_holder is None:
        lock_holder = instance_holder

    lock_key = f"_lock_{class_name}"
    instance_key = f"_instance_{class_name}"

    if lock_key not in lock_holder:
        lock_holder[lock_key] = threading.Lock()

    with lock_holder[lock_key]:
        if instance_key not in instance_holder:
            instance_holder[instance_key] = agent_class(**kwargs)
            logger.info(f"[{class_name}] Singleton instance created")

    return instance_holder[instance_key]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BaseRAGAgent',
    'AgentSingletonMeta',
    'get_agent_singleton'
]
