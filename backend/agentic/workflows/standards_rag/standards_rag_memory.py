# agentic/standards_rag/standards_rag_memory.py
# =============================================================================
# STANDARDS RAG MEMORY - Conversation Memory for Standards Documentation Q&A
# =============================================================================
#
# Refactored to use BaseRAGMemory base class.
# Provides session-based memory for follow-up query resolution in standards RAG.
#
# =============================================================================

import logging
from typing import Dict, Any, List

from agentic.core.agents.base_memory import BaseRAGMemory, create_memory_singleton

logger = logging.getLogger(__name__)


class StandardsRAGMemory(BaseRAGMemory):
    """
    Conversation memory for Standards RAG.

    Extends BaseRAGMemory with standards-specific functionality:
    - Track mentioned standards (IEC, ISO, API, ASME, SIL, ATEX)
    - Track discussed topics
    - Resolve follow-up queries about standards

    Features inherited from BaseRAGMemory:
    - Session-based storage with TTL
    - Automatic cleanup of expired sessions
    - Thread-safe operations
    - History tracking with pruning
    """

    def _get_context_keys(self) -> List[str]:
        """Define context keys tracked by standards memory."""
        return ["standards_mentioned", "topics_discussed"]

    def add_to_memory(
        self,
        session_id: str,
        question: str,
        answer: str = None,
        citations: List[Dict] = None,
        key_terms: List[str] = None
    ) -> None:
        """
        Add a conversation turn to memory.

        Args:
            session_id: Session identifier
            question: User's question
            answer: Generated answer
            citations: Source citations
            key_terms: Extracted key terms
        """
        # Add history entry using base class method
        self._add_history_entry(session_id, {
            "question": question,
            "answer": answer or "",
            "citations": citations or [],
            "key_terms": key_terms or []
        })

        # Update context with standards and topics
        if key_terms:
            standards = []
            topics = []

            for term in key_terms:
                term_upper = term.upper()
                # Extract standard codes
                if any(code in term_upper for code in ["IEC", "ISO", "API", "ASME", "SIL", "ATEX"]):
                    standards.append(term)
                else:
                    topics.append(term)

            if standards:
                self._update_context(session_id, "standards_mentioned", standards)
            if topics:
                self._update_context(session_id, "topics_discussed", topics)

        logger.debug(f"Added to standards memory: session={session_id}")

    def get_last_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get the last conversation context for follow-up resolution.

        Returns:
            Dictionary with last question/answer and accumulated context
        """
        base_context = super().get_last_context(session_id)

        if not base_context:
            return {}

        last_entry = base_context.get("last_entry", {})

        return {
            "last_question": last_entry.get("question", ""),
            "last_answer": last_entry.get("answer", ""),
            "key_terms": last_entry.get("key_terms", []),
            "standards_mentioned": base_context.get("standards_mentioned", []),
            "topics_discussed": base_context.get("topics_discussed", [])
        }

    def resolve_follow_up(self, session_id: str, question: str) -> str:
        """
        Resolve follow-up queries using conversation history.

        Example:
        - Previous: "What is SIL-2 certification?"
        - Current: "What about SIL-3?"
        - Resolved: "What is SIL-3 certification?"

        Args:
            session_id: Session identifier
            question: User's current question

        Returns:
            Resolved query (unchanged if not a follow-up)
        """
        if not self.is_follow_up_query(question):
            return question

        context = self.get_last_context(session_id)
        if not context:
            return question

        last_question = context.get("last_question", "")
        standards = context.get("standards_mentioned", [])
        topics = context.get("topics_discussed", [])

        question_lower = question.lower().strip()

        # Handle "What about X?" pattern
        if question_lower.startswith("what about ") or question_lower.startswith("how about "):
            prefix = "what about " if question_lower.startswith("what about ") else "how about "
            new_subject = question[len(prefix):].strip().rstrip("?")

            # If last question exists, substitute the subject
            if last_question:
                resolved = last_question
                for standard in standards:
                    if standard.lower() in last_question.lower():
                        resolved = last_question.replace(standard, new_subject)
                        break

                if resolved != last_question:
                    logger.info(f"Resolved follow-up: '{question}' -> '{resolved}'")
                    return resolved

            # Fallback: Create a question about the new subject
            return f"What is {new_subject}?"

        # Handle "And X?" pattern
        if question_lower.startswith("and "):
            new_subject = question[4:].strip().rstrip("?")
            if last_question:
                return f"{last_question.rstrip('?')} and {new_subject}?"

        # Handle comparison patterns
        if any(p in question_lower for p in ["versus", "vs", "compare"]):
            if standards:
                return f"{question} (in context of {', '.join(standards[:2])})"

        # If we have context, append it for clarity
        if standards or topics:
            context_hint = standards[:2] if standards else topics[:2]
            return f"{question} (regarding {', '.join(context_hint)})"

        return question


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_singleton_holder: Dict[str, Any] = {}


def get_standards_rag_memory() -> StandardsRAGMemory:
    """Get the global StandardsRAGMemory singleton instance."""
    return create_memory_singleton(
        StandardsRAGMemory,
        _singleton_holder,
        session_timeout=3600  # 1 hour
    )


# Global instance for backward compatibility
standards_rag_memory = get_standards_rag_memory()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def resolve_standards_follow_up(session_id: str, question: str) -> str:
    """Convenience function for follow-up resolution."""
    return get_standards_rag_memory().resolve_follow_up(session_id, question)


def add_to_standards_memory(
    session_id: str,
    question: str,
    answer: str = None,
    citations: List[Dict] = None,
    key_terms: List[str] = None
) -> None:
    """Convenience function for adding to memory."""
    get_standards_rag_memory().add_to_memory(
        session_id=session_id,
        question=question,
        answer=answer,
        citations=citations,
        key_terms=key_terms
    )


def clear_standards_memory(session_id: str) -> None:
    """Convenience function for clearing memory."""
    get_standards_rag_memory().clear_session(session_id)


def get_session_summary(session_id: str) -> Dict[str, Any]:
    """Get summary of a session."""
    memory = get_standards_rag_memory()
    info = memory.get_session_info(session_id)
    if not info:
        return {"exists": False}

    context = memory.get_last_context(session_id)
    return {
        "exists": True,
        **info,
        "standards_mentioned": context.get("standards_mentioned", []),
        "topics_discussed": context.get("topics_discussed", [])
    }


def clear_session(session_id: str) -> bool:
    """Clear a specific session."""
    return get_standards_rag_memory().clear_session(session_id)


def cleanup_expired_sessions() -> int:
    """Clean up expired sessions."""
    return get_standards_rag_memory().cleanup_expired_sessions()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'StandardsRAGMemory',
    'standards_rag_memory',
    'get_standards_rag_memory',
    'resolve_standards_follow_up',
    'add_to_standards_memory',
    'clear_standards_memory',
    'get_session_summary',
    'clear_session',
    'cleanup_expired_sessions'
]
