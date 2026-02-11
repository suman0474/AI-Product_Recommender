# agentic/deep_agent/state_manager.py
"""
State Persistence Layer for Distributed Enrichment

Provides persistent state management using Redis or Cosmos DB.
State survives process crashes and enables resume capability.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Set
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class EnrichmentStateManager(ABC):
    """
    Abstract base class for enrichment state management.

    Provides interface for storing and retrieving enrichment session state
    in a persistent store (Redis, Cosmos DB, etc.)
    """

    @abstractmethod
    def create_session(self, session_id: str, initial_data: Dict[str, Any]) -> None:
        """
        Initialize a new enrichment session.

        Args:
            session_id: Unique session identifier
            initial_data: Initial session metadata (total_items, started_at, etc.)
        """
        pass

    @abstractmethod
    def update_item_state(self, session_id: str, item_id: str, state: Dict[str, Any]) -> None:
        """
        Update state for a single item in the session.

        Args:
            session_id: Session identifier
            item_id: Item identifier
            state: Item state dict (specifications, iteration, status, etc.)
        """
        pass

    @abstractmethod
    def get_item_state(self, session_id: str, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve state for a single item.

        Args:
            session_id: Session identifier
            item_id: Item identifier

        Returns:
            Item state dict or None if not found
        """
        pass

    @abstractmethod
    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete session state/metadata.

        Args:
            session_id: Session identifier

        Returns:
            Session state dict or None if not found
        """
        pass

    @abstractmethod
    def mark_item_complete(self, session_id: str, item_id: str, final_specs: Dict[str, Any]) -> None:
        """
        Mark an item as completed with final specifications.

        Args:
            session_id: Session identifier
            item_id: Item identifier
            final_specs: Final specifications dict
        """
        pass

    @abstractmethod
    def is_session_complete(self, session_id: str) -> bool:
        """
        Check if all items in the session are complete.

        Args:
            session_id: Session identifier

        Returns:
            True if all items completed, False otherwise
        """
        pass

    @abstractmethod
    def get_session_item_ids(self, session_id: str) -> List[str]:
        """
        Get list of all item IDs in the session.

        Args:
            session_id: Session identifier

        Returns:
            List of item IDs
        """
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """
        Delete a session and all associated data.

        Args:
            session_id: Session identifier
        """
        pass


# =============================================================================
# REDIS IMPLEMENTATION
# =============================================================================

class RedisStateManager(EnrichmentStateManager):
    """
    Redis-based state manager for fast, in-memory persistent storage.

    Recommended for production use due to:
    - Low latency (<1ms reads/writes)
    - Built-in TTL for automatic cleanup
    - Atomic operations
    - High availability with Redis Cluster
    """

    def __init__(self, redis_url: Optional[str] = None, ttl: int = 3600):
        """
        Initialize Redis state manager.

        Args:
            redis_url: Redis connection URL (default: from env REDIS_URL)
            ttl: Time-to-live for keys in seconds (default: 1 hour)
        """
        try:
            import redis
        except ImportError:
            raise ImportError(
                "redis package not installed. Install with: pip install redis"
            )

        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.ttl = ttl

        try:
            self.redis = redis.from_url(
                self.redis_url,
                decode_responses=True,  # Auto-decode bytes to strings
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis.ping()
            logger.info(f"[StateManager] Connected to Redis: {self.redis_url}")
        except Exception as e:
            logger.error(f"[StateManager] Failed to connect to Redis: {e}")
            raise

    def create_session(self, session_id: str, initial_data: Dict[str, Any]) -> None:
        """Create a new enrichment session in Redis."""
        key = self._session_key(session_id)

        # Add metadata
        session_data = {
            **initial_data,
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat()
        }

        self.redis.setex(key, self.ttl, json.dumps(session_data))
        logger.info(f"[StateManager] Created session: {session_id}")

    def update_item_state(self, session_id: str, item_id: str, state: Dict[str, Any]) -> None:
        """Update item state in Redis."""
        key = self._item_key(session_id, item_id)

        # Protect completed items from being modified
        existing_data = self.redis.get(key)
        if existing_data:
            existing_state = json.loads(existing_data)
            if existing_state.get("status") == "completed":
                logger.debug(f"[StateManager] Cannot update completed item: {session_id}:{item_id}")
                return

        # Add metadata
        item_state = {
            **state,
            "item_id": item_id,
            "session_id": session_id,
            "updated_at": datetime.utcnow().isoformat()
        }

        self.redis.setex(key, self.ttl, json.dumps(item_state))

        # Add to session's item set
        self.redis.sadd(self._items_set_key(session_id), item_id)
        self.redis.expire(self._items_set_key(session_id), self.ttl)

        logger.debug(f"[StateManager] Updated item state: {session_id}:{item_id}")

    def get_item_state(self, session_id: str, item_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve item state from Redis."""
        key = self._item_key(session_id, item_id)
        data = self.redis.get(key)

        if data:
            return json.loads(data)

        logger.debug(f"[StateManager] Item state not found: {session_id}:{item_id}")
        return None

    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session metadata from Redis."""
        key = self._session_key(session_id)
        data = self.redis.get(key)

        if data:
            return json.loads(data)

        logger.debug(f"[StateManager] Session not found: {session_id}")
        return None

    def mark_item_complete(self, session_id: str, item_id: str, final_specs: Dict[str, Any]) -> None:
        """Mark item as completed in Redis."""
        # Update item state with completion marker
        self.update_item_state(session_id, item_id, {
            "status": "completed",
            "final_specs": final_specs,
            "completed_at": datetime.utcnow().isoformat()
        })

        # Add to completed set
        self.redis.sadd(self._completed_set_key(session_id), item_id)
        self.redis.expire(self._completed_set_key(session_id), self.ttl)

        logger.info(f"[StateManager] Marked complete: {session_id}:{item_id}")

    def is_session_complete(self, session_id: str) -> bool:
        """Check if all items in session are completed."""
        total_items = self.redis.scard(self._items_set_key(session_id))
        completed_items = self.redis.scard(self._completed_set_key(session_id))

        is_complete = total_items > 0 and total_items == completed_items

        logger.debug(
            f"[StateManager] Session {session_id} completion: "
            f"{completed_items}/{total_items} items"
        )

        return is_complete

    def get_session_item_ids(self, session_id: str) -> List[str]:
        """Get all item IDs in the session."""
        item_ids = self.redis.smembers(self._items_set_key(session_id))
        return list(item_ids) if item_ids else []

    def delete_session(self, session_id: str) -> None:
        """Delete session and all associated data."""
        # Get all item IDs
        item_ids = self.get_session_item_ids(session_id)

        # Delete all item keys
        for item_id in item_ids:
            self.redis.delete(self._item_key(session_id, item_id))

        # Delete sets and session
        self.redis.delete(self._items_set_key(session_id))
        self.redis.delete(self._completed_set_key(session_id))
        self.redis.delete(self._session_key(session_id))

        logger.info(f"[StateManager] Deleted session: {session_id} ({len(item_ids)} items)")

    # Helper methods for key generation
    def _session_key(self, session_id: str) -> str:
        """Generate Redis key for session metadata."""
        return f"enrichment:session:{session_id}"

    def _item_key(self, session_id: str, item_id: str) -> str:
        """Generate Redis key for item state."""
        return f"enrichment:session:{session_id}:item:{item_id}"

    def _items_set_key(self, session_id: str) -> str:
        """Generate Redis key for set of all items in session."""
        return f"enrichment:session:{session_id}:items"

    def _completed_set_key(self, session_id: str) -> str:
        """Generate Redis key for set of completed items."""
        return f"enrichment:session:{session_id}:completed"


# =============================================================================
# IN-MEMORY IMPLEMENTATION (For Testing)
# =============================================================================

class InMemoryStateManager(EnrichmentStateManager):
    """
    In-memory state manager for testing and development.

    WARNING: State is lost on process restart. Use only for testing.
    """

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.items: Dict[str, Dict[str, Dict[str, Any]]] = {}  # session_id -> item_id -> state
        self.completed: Dict[str, Set[str]] = {}  # session_id -> set of completed item_ids
        logger.warning("[StateManager] Using InMemoryStateManager - state not persistent!")

    def create_session(self, session_id: str, initial_data: Dict[str, Any]) -> None:
        self.sessions[session_id] = {
            **initial_data,
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat()
        }
        self.items[session_id] = {}
        self.completed[session_id] = set()

    def update_item_state(self, session_id: str, item_id: str, state: Dict[str, Any]) -> None:
        if session_id not in self.items:
            self.items[session_id] = {}

        # Protect completed items from being modified
        existing_state = self.items[session_id].get(item_id, {})
        if existing_state.get("status") == "completed":
            # Don't overwrite completed status, just log and return
            logger.debug(f"[StateManager] Cannot update completed item: {session_id}:{item_id}")
            return

        self.items[session_id][item_id] = {
            **state,
            "item_id": item_id,
            "session_id": session_id,
            "updated_at": datetime.utcnow().isoformat()
        }

    def get_item_state(self, session_id: str, item_id: str) -> Optional[Dict[str, Any]]:
        return self.items.get(session_id, {}).get(item_id)

    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)

    def mark_item_complete(self, session_id: str, item_id: str, final_specs: Dict[str, Any]) -> None:
        self.update_item_state(session_id, item_id, {
            "status": "completed",
            "final_specs": final_specs,
            "completed_at": datetime.utcnow().isoformat()
        })

        if session_id not in self.completed:
            self.completed[session_id] = set()

        self.completed[session_id].add(item_id)

    def is_session_complete(self, session_id: str) -> bool:
        total_items = len(self.items.get(session_id, {}))
        completed_items = len(self.completed.get(session_id, set()))
        return total_items > 0 and total_items == completed_items

    def get_session_item_ids(self, session_id: str) -> List[str]:
        return list(self.items.get(session_id, {}).keys())

    def delete_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        self.items.pop(session_id, None)
        self.completed.pop(session_id, None)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_state_manager(use_redis: bool = True) -> EnrichmentStateManager:
    """
    Factory function to get appropriate state manager.

    Args:
        use_redis: If True, use Redis (default). If False, use in-memory.

    Returns:
        EnrichmentStateManager instance
    """
    if use_redis:
        try:
            return RedisStateManager()
        except Exception as e:
            logger.warning(f"[StateManager] Failed to create Redis manager: {e}")
            logger.warning("[StateManager] Falling back to InMemoryStateManager")
            return InMemoryStateManager()
    else:
        return InMemoryStateManager()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "EnrichmentStateManager",
    "RedisStateManager",
    "InMemoryStateManager",
    "get_state_manager"
]
