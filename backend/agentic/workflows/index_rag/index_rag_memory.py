# agentic/index_rag/index_rag_memory.py
# =============================================================================
# INDEX RAG MEMORY - Conversation Memory for Product Index RAG
# =============================================================================
#
# Refactored to use BaseRAGMemory base class.
# Provides session-based memory for follow-up query resolution in index RAG.
#
# =============================================================================

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from agentic.core.agents.base_memory import BaseRAGMemory, create_memory_singleton

logger = logging.getLogger(__name__)


class IndexRAGMemory(BaseRAGMemory):
    """
    Conversation memory for Index RAG.

    Extends BaseRAGMemory with index-specific functionality:
    - Track product types, vendors, and models searched
    - LLM-based follow-up resolution
    - Result caching

    Features inherited from BaseRAGMemory:
    - Session-based storage with TTL
    - Automatic cleanup of expired sessions
    - Thread-safe operations
    - History tracking with pruning
    """

    # Cache for query results
    _result_cache: Dict[str, Dict[str, Any]] = {}
    _cache_max_size: int = 100

    def _get_context_keys(self) -> List[str]:
        """Define context keys tracked by index memory."""
        return ["product_types", "vendors", "models"]

    def add_to_memory(
        self,
        session_id: str,
        query: str,
        intent: str = None,
        product_type: str = None,
        vendors: List[str] = None,
        models: List[str] = None
    ) -> None:
        """
        Add a conversation turn to memory.

        Args:
            session_id: Session identifier
            query: User's query
            intent: Classified intent
            product_type: Detected product type
            vendors: Detected vendors
            models: Detected models
        """
        # Add history entry using base class method
        self._add_history_entry(session_id, {
            "query": query,
            "intent": intent,
            "product_type": product_type,
            "vendors": vendors or [],
            "models": models or []
        })

        # Update context
        if product_type:
            self._update_context(session_id, "product_types", [product_type])
        if vendors:
            self._update_context(session_id, "vendors", vendors)
        if models:
            self._update_context(session_id, "models", models)

        logger.debug(f"[IndexRAGMemory] Added turn for {session_id}")

    def get_last_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get the last conversation context for follow-up resolution.

        Returns:
            Dictionary with last query data and accumulated context
        """
        base_context = super().get_last_context(session_id)

        if not base_context:
            return {}

        last_entry = base_context.get("last_entry", {})

        return {
            "query": last_entry.get("query", ""),
            "product_type": last_entry.get("product_type", ""),
            "vendors": last_entry.get("vendors", []),
            "models": last_entry.get("models", []),
            "intent": last_entry.get("intent", ""),
            "all_product_types": base_context.get("product_types", []),
            "all_vendors": base_context.get("vendors", []),
            "all_models": base_context.get("models", [])
        }

    def resolve_follow_up(self, session_id: str, query: str) -> str:
        """
        Resolve follow-up queries using conversation history.

        Uses LLM for intelligent resolution when simple patterns don't match.

        Example:
        - Previous: "Honeywell pressure transmitters"
        - Current: "What about Emerson?"
        - Resolved: "Emerson pressure transmitters"

        Args:
            session_id: Session identifier
            query: User's current query

        Returns:
            Resolved query (unchanged if not a follow-up)
        """
        # Check if this is a follow-up
        if not self.is_follow_up_query(query):
            return query

        # Get previous context
        last_context = self.get_last_context(session_id)
        if not last_context:
            return query  # No history, can't resolve

        last_query = last_context.get('query', '')
        last_product_type = last_context.get('product_type', '')
        last_vendors = last_context.get('vendors', [])

        if not last_product_type and not last_vendors:
            return query  # No useful context

        # Try simple pattern-based resolution first
        query_lower = query.lower().strip()

        # Handle "What about [vendor]?" pattern
        if query_lower.startswith("what about ") or query_lower.startswith("how about "):
            prefix = "what about " if query_lower.startswith("what about ") else "how about "
            new_subject = query[len(prefix):].strip().rstrip("?")

            if last_product_type:
                resolved = f"{new_subject} {last_product_type}"
                logger.info(f"[IndexRAGMemory] Resolved follow-up: '{query}' -> '{resolved}'")
                return resolved

        # Use LLM for complex follow-up resolution
        try:
            resolved = self._llm_resolve_follow_up(
                query=query,
                last_query=last_query,
                product_type=last_product_type,
                vendors=last_vendors
            )
            if resolved and resolved != query:
                logger.info(f"[IndexRAGMemory] LLM resolved follow-up: '{query}' -> '{resolved}'")
                return resolved
        except Exception as e:
            logger.warning(f"[IndexRAGMemory] LLM follow-up resolution failed: {e}")

        return query

    def _llm_resolve_follow_up(
        self,
        query: str,
        last_query: str,
        product_type: str,
        vendors: List[str]
    ) -> str:
        """
        Use LLM to resolve complex follow-up queries.

        Args:
            query: Current follow-up query
            last_query: Previous query
            product_type: Last product type discussed
            vendors: Last vendors discussed

        Returns:
            Resolved query or original if resolution fails
        """
        from services.llm.fallback import create_llm_with_fallback
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            max_tokens=200
        )

        prompt = ChatPromptTemplate.from_template("""Resolve this follow-up question using the conversation context.

PREVIOUS QUESTION: {last_query}
PREVIOUS CONTEXT:
- Product type: {product_type}
- Vendors discussed: {vendors}

CURRENT FOLLOW-UP: {current_query}

INSTRUCTIONS:
- If the user is asking about a different vendor/product in the same context, expand the query
- Example: Previous "Honeywell pressure transmitters", Current "What about Emerson?" -> "Emerson pressure transmitters"
- If it's not clearly a follow-up, return the original query unchanged
- Return ONLY the resolved query text, nothing else

RESOLVED QUERY:""")

        chain = prompt | llm | StrOutputParser()

        resolved = chain.invoke({
            "last_query": last_query,
            "product_type": product_type or "not specified",
            "vendors": ", ".join(vendors) if vendors else "none",
            "current_query": query
        })

        return resolved.strip()

    # =========================================================================
    # RESULT CACHING
    # =========================================================================

    def cache_result(self, query_hash: str, result: Dict[str, Any]) -> None:
        """
        Cache a query result.

        Args:
            query_hash: Hash of the query
            result: Result to cache
        """
        if len(self._result_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest = next(iter(self._result_cache))
            del self._result_cache[oldest]

        self._result_cache[query_hash] = {
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }

    def get_cached_result(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached result if available.

        Args:
            query_hash: Hash of the query

        Returns:
            Cached result or None
        """
        cached = self._result_cache.get(query_hash)
        if cached:
            return cached.get("result")
        return None

    def clear_cache(self) -> int:
        """
        Clear the result cache.

        Returns:
            Number of entries cleared
        """
        count = len(self._result_cache)
        self._result_cache.clear()
        return count


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_singleton_holder: Dict[str, Any] = {}


def get_index_rag_memory() -> IndexRAGMemory:
    """Get the global IndexRAGMemory singleton instance."""
    return create_memory_singleton(
        IndexRAGMemory,
        _singleton_holder,
        session_timeout=3600  # 1 hour
    )


# Global instance for backward compatibility
index_rag_memory = get_index_rag_memory()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def resolve_follow_up_query(session_id: str, query: str) -> str:
    """Convenience function for follow-up resolution."""
    return get_index_rag_memory().resolve_follow_up(session_id, query)


def add_to_conversation_memory(
    session_id: str,
    query: str,
    intent: str = None,
    product_type: str = None,
    vendors: List[str] = None,
    models: List[str] = None
) -> None:
    """Convenience function for adding to memory."""
    get_index_rag_memory().add_to_memory(
        session_id=session_id,
        query=query,
        intent=intent,
        product_type=product_type,
        vendors=vendors,
        models=models
    )


def clear_conversation_memory(session_id: str) -> None:
    """Convenience function for clearing memory."""
    get_index_rag_memory().clear_session(session_id)


def get_session_summary(session_id: str) -> Dict[str, Any]:
    """Get summary of a session."""
    memory = get_index_rag_memory()
    info = memory.get_session_info(session_id)
    if not info:
        return {"exists": False}

    context = memory.get_last_context(session_id)
    return {
        "exists": True,
        **info,
        "product_types": context.get("all_product_types", []),
        "vendors": context.get("all_vendors", []),
        "models": context.get("all_models", [])
    }


def clear_session(session_id: str) -> bool:
    """Clear a specific session."""
    return get_index_rag_memory().clear_session(session_id)


def cleanup_expired_sessions() -> int:
    """Clean up expired sessions."""
    return get_index_rag_memory().cleanup_expired_sessions()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'IndexRAGMemory',
    'index_rag_memory',
    'get_index_rag_memory',
    'resolve_follow_up_query',
    'add_to_conversation_memory',
    'clear_conversation_memory',
    'get_session_summary',
    'clear_session',
    'cleanup_expired_sessions'
]
