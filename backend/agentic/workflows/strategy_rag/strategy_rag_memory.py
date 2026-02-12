# agentic/strategy_rag/strategy_rag_memory.py
# =============================================================================
# STRATEGY RAG MEMORY - Conversation Memory for Strategy Documentation Q&A
# =============================================================================
#
# Refactored to use BaseRAGMemory base class.
# Provides session-based memory for follow-up query resolution in strategy RAG.
#
# =============================================================================

import logging
from typing import Dict, Any, List

from agentic.core.agents.base_memory import BaseRAGMemory, create_memory_singleton

logger = logging.getLogger(__name__)


class StrategyRAGMemory(BaseRAGMemory):
    """
    Conversation memory for Strategy RAG.

    Extends BaseRAGMemory with strategy-specific functionality:
    - Track product types searched
    - Track vendors mentioned
    - Resolve follow-up queries about vendors/products

    Features inherited from BaseRAGMemory:
    - Session-based storage with TTL
    - Automatic cleanup of expired sessions
    - Thread-safe operations
    - History tracking with pruning
    """

    def _get_context_keys(self) -> List[str]:
        """Define context keys tracked by strategy memory."""
        return ["product_types_searched", "vendors_mentioned"]

    def add_to_memory(
        self,
        session_id: str,
        product_type: str,
        preferred_vendors: List[str] = None,
        strategy_notes: str = None
    ) -> None:
        """
        Add a strategy query to memory.

        Args:
            session_id: Session identifier
            product_type: Product type searched
            preferred_vendors: Vendors found
            strategy_notes: Strategy notes returned
        """
        # Add history entry using base class method
        self._add_history_entry(session_id, {
            "product_type": product_type,
            "preferred_vendors": preferred_vendors or [],
            "strategy_notes": strategy_notes or ""
        })

        # Update context
        if product_type:
            self._update_context(session_id, "product_types_searched", [product_type])

        if preferred_vendors:
            self._update_context(session_id, "vendors_mentioned", preferred_vendors)

        logger.debug(f"Added to strategy memory: session={session_id}")

    def get_last_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get the last query context for follow-up resolution.

        Returns:
            Dictionary with last product type, vendors, and accumulated context
        """
        base_context = super().get_last_context(session_id)

        if not base_context:
            return {}

        last_entry = base_context.get("last_entry", {})

        return {
            "last_product_type": last_entry.get("product_type", ""),
            "last_vendors": last_entry.get("preferred_vendors", []),
            "product_types_searched": base_context.get("product_types_searched", []),
            "vendors_mentioned": base_context.get("vendors_mentioned", [])
        }

    def resolve_follow_up(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        Resolve follow-up queries using session history.

        Args:
            session_id: Session identifier
            query: User's current query

        Returns:
            Dict with resolved product_type and any context
        """
        if not self.is_follow_up_query(query):
            return {"product_type": query, "is_follow_up": False}

        context = self.get_last_context(session_id)
        if not context:
            return {"product_type": query, "is_follow_up": False}

        query_lower = query.lower().strip()

        # Handle "What about [vendor]?" - return same product type, filter for vendor
        if query_lower.startswith("what about ") or query_lower.startswith("how about "):
            prefix = "what about " if query_lower.startswith("what about ") else "how about "
            subject = query[len(prefix):].strip().rstrip("?")

            # Check if it's asking about a vendor
            last_product_type = context.get("last_product_type", "")
            if last_product_type:
                return {
                    "product_type": last_product_type,
                    "vendor_filter": subject,
                    "is_follow_up": True,
                    "original_query": query
                }

        # Return the query as product type with follow-up flag
        return {
            "product_type": query,
            "is_follow_up": True,
            "context": context
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_singleton_holder: Dict[str, Any] = {}


def get_strategy_rag_memory() -> StrategyRAGMemory:
    """Get the global StrategyRAGMemory singleton instance."""
    return create_memory_singleton(
        StrategyRAGMemory,
        _singleton_holder,
        session_timeout=3600  # 1 hour
    )


# Global instance for backward compatibility
strategy_rag_memory = get_strategy_rag_memory()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def add_to_strategy_memory(
    session_id: str,
    product_type: str,
    preferred_vendors: List[str] = None,
    strategy_notes: str = None
) -> None:
    """Convenience function for adding to memory."""
    get_strategy_rag_memory().add_to_memory(
        session_id=session_id,
        product_type=product_type,
        preferred_vendors=preferred_vendors,
        strategy_notes=strategy_notes
    )


def clear_strategy_memory(session_id: str) -> None:
    """Convenience function for clearing memory."""
    get_strategy_rag_memory().clear_session(session_id)


def get_session_summary(session_id: str) -> Dict[str, Any]:
    """Get summary of a session."""
    memory = get_strategy_rag_memory()
    info = memory.get_session_info(session_id)
    if not info:
        return {"exists": False}

    context = memory.get_last_context(session_id)
    return {
        "exists": True,
        **info,
        "product_types_searched": context.get("product_types_searched", []),
        "vendors_mentioned": context.get("vendors_mentioned", [])
    }


def clear_session(session_id: str) -> bool:
    """Clear a specific session."""
    return get_strategy_rag_memory().clear_session(session_id)


def cleanup_expired_sessions() -> int:
    """Clean up expired sessions."""
    return get_strategy_rag_memory().cleanup_expired_sessions()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'StrategyRAGMemory',
    'strategy_rag_memory',
    'get_strategy_rag_memory',
    'add_to_strategy_memory',
    'clear_strategy_memory',
    'get_session_summary',
    'clear_session',
    'cleanup_expired_sessions'
]
