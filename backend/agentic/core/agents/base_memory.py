# agentic/base_memory.py
# =============================================================================
# BASE RAG MEMORY - Abstract Base Class for Conversation Memory
# =============================================================================
#
# This module provides a base class for session-based conversation memory
# that is extended by specific RAG memory implementations:
#   - StandardsRAGMemory
#   - StrategyRAGMemory
#   - IndexRAGMemory
#   - ProductInfoMemory
#
# Common features:
#   - Session-based storage with TTL
#   - Conversation history tracking
#   - Follow-up query detection
#   - Automatic cleanup of expired sessions
#   - Thread-safe operations
#
# Usage:
#   class MyRAGMemory(BaseRAGMemory):
#       def add_to_memory(self, session_id, **kwargs):
#           # Custom implementation
#           pass
#
# =============================================================================

import logging
import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, TypeVar, Generic
from datetime import datetime

logger = logging.getLogger(__name__)

# Type variable for memory entry type
T = TypeVar('T')


class BaseRAGMemory(ABC, Generic[T]):
    """
    Abstract base class for RAG conversation memory.

    Provides common session management functionality:
    - Session creation and cleanup
    - History retrieval
    - Follow-up query detection
    - Thread-safe operations
    - TTL-based expiration

    Subclasses must implement:
    - add_to_memory(): Add domain-specific data to session
    - _get_context_keys(): Define which keys to track in context
    """

    # Default session timeout (can be overridden by subclasses)
    SESSION_TIMEOUT_SECONDS: int = 3600  # 1 hour
    MAX_HISTORY_PER_SESSION: int = 20     # Prevent unbounded growth

    def __init__(self, session_timeout: int = None, max_history: int = None):
        """
        Initialize base memory.

        Args:
            session_timeout: Session TTL in seconds (default: 3600)
            max_history: Max history entries per session (default: 20)
        """
        self.session_timeout = session_timeout or self.SESSION_TIMEOUT_SECONDS
        self.max_history = max_history or self.MAX_HISTORY_PER_SESSION

        # Session storage: session_id -> session data
        self._sessions: Dict[str, Dict[str, Any]] = {}

        # Thread lock for thread-safe operations
        self._lock = threading.RLock()

        # Stats tracking
        self._stats = {
            "sessions_created": 0,
            "sessions_expired": 0,
            "total_entries": 0
        }

        logger.info(
            f"[{self.__class__.__name__}] Initialized: "
            f"timeout={self.session_timeout}s, max_history={self.max_history}"
        )

    # =========================================================================
    # ABSTRACT METHODS (must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def add_to_memory(self, session_id: str, **kwargs) -> None:
        """
        Add domain-specific data to session memory.

        Subclasses should:
        1. Call _ensure_session(session_id) to create/get session
        2. Add entry to session["history"]
        3. Update session["context"] with relevant data
        4. Call _update_session_timestamp(session_id)

        Args:
            session_id: Session identifier
            **kwargs: Domain-specific data to store
        """
        pass

    def _get_context_keys(self) -> List[str]:
        """
        Get the context keys tracked by this memory type.

        Override in subclasses to define domain-specific context keys.
        These keys are used for follow-up resolution and context building.

        Returns:
            List of context key names

        Example:
            return ["standards_mentioned", "topics_discussed"]
        """
        return ["topics"]

    # =========================================================================
    # SESSION MANAGEMENT (shared by all subclasses)
    # =========================================================================

    def _ensure_session(self, session_id: str) -> Dict[str, Any]:
        """
        Ensure a session exists, creating if necessary.

        Args:
            session_id: Session identifier

        Returns:
            Session dictionary
        """
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = self._create_session()
                self._stats["sessions_created"] += 1
                logger.debug(f"[{self.__class__.__name__}] Created session: {session_id}")
            return self._sessions[session_id]

    def _create_session(self) -> Dict[str, Any]:
        """
        Create a new session structure.

        Returns:
            New session dictionary
        """
        context = {key: [] for key in self._get_context_keys()}
        return {
            "created_at": time.time(),
            "last_updated": time.time(),
            "history": [],
            "context": context
        }

    def _update_session_timestamp(self, session_id: str) -> None:
        """Update the last_updated timestamp for a session."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["last_updated"] = time.time()

    def _add_history_entry(
        self,
        session_id: str,
        entry: Dict[str, Any]
    ) -> None:
        """
        Add an entry to session history with automatic pruning.

        Args:
            session_id: Session identifier
            entry: History entry to add
        """
        with self._lock:
            session = self._ensure_session(session_id)

            # Add timestamp if not present
            if "timestamp" not in entry:
                entry["timestamp"] = datetime.utcnow().isoformat()

            session["history"].append(entry)
            session["last_updated"] = time.time()
            self._stats["total_entries"] += 1

            # Prune history if over limit
            if len(session["history"]) > self.max_history:
                session["history"] = session["history"][-self.max_history:]
                logger.debug(
                    f"[{self.__class__.__name__}] Pruned history for {session_id}"
                )

    def _update_context(
        self,
        session_id: str,
        key: str,
        values: List[Any],
        max_items: int = 20
    ) -> None:
        """
        Update a context list with new values (deduped).

        Args:
            session_id: Session identifier
            key: Context key to update
            values: Values to add
            max_items: Max items to keep in list
        """
        with self._lock:
            session = self._ensure_session(session_id)
            context_list = session["context"].get(key, [])

            for value in values:
                if value and value not in context_list:
                    context_list.append(value)

            # Keep only recent items
            session["context"][key] = context_list[-max_items:]

    # =========================================================================
    # HISTORY RETRIEVAL
    # =========================================================================

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of history entries (empty if session doesn't exist)
        """
        with self._lock:
            session = self._sessions.get(session_id, {})
            return session.get("history", []).copy()

    def get_last_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get the last conversation context for follow-up resolution.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with last entry and accumulated context
        """
        with self._lock:
            session = self._sessions.get(session_id, {})
            history = session.get("history", [])
            context = session.get("context", {})

            if not history:
                return {}

            last = history[-1]
            result = {
                "last_entry": last,
                **{k: v.copy() for k, v in context.items()}
            }
            return result

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session metadata.

        Args:
            session_id: Session identifier

        Returns:
            Session info or None if not found
        """
        with self._lock:
            if session_id not in self._sessions:
                return None

            session = self._sessions[session_id]
            return {
                "session_id": session_id,
                "created_at": session.get("created_at"),
                "last_updated": session.get("last_updated"),
                "history_count": len(session.get("history", [])),
                "age_seconds": time.time() - session.get("created_at", time.time())
            }

    # =========================================================================
    # FOLLOW-UP DETECTION
    # =========================================================================

    def is_follow_up_query(self, query: str) -> bool:
        """
        Check if a query looks like a follow-up.

        Common follow-up indicators:
        - "What about X?"
        - "How about X?"
        - "And X?"
        - Short queries (< 5 words)
        - Pronouns referring to previous context

        Args:
            query: Query string to check

        Returns:
            True if query appears to be a follow-up
        """
        if not query:
            return False

        query_lower = query.lower().strip()

        # Explicit follow-up patterns
        follow_up_starts = [
            "what about", "how about", "and ", "also ",
            "what's the difference", "what is the difference",
            "compare", "versus", "vs", "or ",
            "can you explain more", "tell me more",
            "what if", "how does that", "show me",
            "same for", "what's the"
        ]

        for pattern in follow_up_starts:
            if query_lower.startswith(pattern):
                return True

        # Short queries are often follow-ups
        words = query_lower.split()
        if len(words) < 5:
            return True

        # Pronoun references to previous context
        pronouns = [
            "it", "they", "this", "that", "these", "those",
            "the standard", "the requirement", "them"
        ]
        for pronoun in pronouns:
            if pronoun in query_lower:
                return True

        return False

    def resolve_follow_up(self, session_id: str, query: str) -> str:
        """
        Resolve a follow-up query using session context.

        Override in subclasses for domain-specific resolution.
        Default implementation returns the query unchanged.

        Args:
            session_id: Session identifier
            query: Current query

        Returns:
            Resolved query (may be unchanged)
        """
        # Default: no resolution
        return query

    # =========================================================================
    # SESSION CLEANUP
    # =========================================================================

    def clear_session(self, session_id: str) -> bool:
        """
        Clear a specific session.

        Args:
            session_id: Session to clear

        Returns:
            True if session existed and was cleared
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"[{self.__class__.__name__}] Cleared session: {session_id}")
                return True
            return False

    def cleanup_expired_sessions(self) -> int:
        """
        Remove all expired sessions.

        Returns:
            Number of sessions removed
        """
        current_time = time.time()
        expired = []

        with self._lock:
            for session_id, session in self._sessions.items():
                last_updated = session.get("last_updated", 0)
                if current_time - last_updated > self.session_timeout:
                    expired.append(session_id)

            for session_id in expired:
                del self._sessions[session_id]
                self._stats["sessions_expired"] += 1

        if expired:
            logger.info(
                f"[{self.__class__.__name__}] Cleaned up {len(expired)} expired sessions"
            )

        return len(expired)

    def clear_all(self) -> int:
        """
        Clear all sessions.

        Returns:
            Number of sessions cleared
        """
        with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
            logger.info(f"[{self.__class__.__name__}] Cleared all {count} sessions")
            return count

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dictionary with stats
        """
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "sessions_created": self._stats["sessions_created"],
                "sessions_expired": self._stats["sessions_expired"],
                "total_entries": self._stats["total_entries"],
                "session_timeout_seconds": self.session_timeout,
                "max_history_per_session": self.max_history
            }


# =============================================================================
# CONVENIENCE FACTORY
# =============================================================================

def create_memory_singleton(
    memory_class: type,
    instance_holder: Dict[str, Any],
    instance_key: str = "_instance",
    lock_key: str = "_lock",
    **kwargs
) -> BaseRAGMemory:
    """
    Factory for creating singleton memory instances.

    Usage in subclass module:
        _singleton = {}

        def get_my_memory() -> MyRAGMemory:
            return create_memory_singleton(
                MyRAGMemory,
                _singleton,
                session_timeout=3600
            )

    Args:
        memory_class: Memory class to instantiate
        instance_holder: Dict to store singleton
        instance_key: Key for instance in holder
        lock_key: Key for lock in holder
        **kwargs: Arguments to pass to memory constructor

    Returns:
        Singleton memory instance
    """
    # Create lock if not exists
    if lock_key not in instance_holder:
        instance_holder[lock_key] = threading.Lock()

    with instance_holder[lock_key]:
        if instance_key not in instance_holder:
            instance_holder[instance_key] = memory_class(**kwargs)
            logger.info(f"[{memory_class.__name__}] Singleton instance created")

    return instance_holder[instance_key]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BaseRAGMemory',
    'create_memory_singleton'
]
