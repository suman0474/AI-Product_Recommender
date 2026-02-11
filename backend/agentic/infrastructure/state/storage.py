"""
State Storage - Azure Blob Storage Backend for Session and Instance Persistence

Provides:
- Session state persistence to Azure Blob Storage
- Instance state persistence with parent-child relationships
- Automatic serialization/deserialization
- Cache layer for performance
- TTL-based cleanup

Usage:
    from agentic.infrastructure.state.storage import StateStorage
    
    storage = StateStorage(connection_string, container_name)
    
    # Save session
    storage.save_session("session_abc", session_data)
    
    # Get session
    session = storage.get_session("session_abc")
    
    # Save instance with parent link
    storage.save_instance("inst_123", instance_data, parent_id="inst_456")
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from threading import RLock
from enum import Enum

# Azure SDK imports
try:
    from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
    from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SessionState:
    """Persisted session state."""
    session_id: str
    user_id: str
    intent: Optional[str] = None
    locked: bool = False
    workflow_type: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    request_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class InstanceState:
    """Persisted instance state."""
    instance_id: str
    session_id: str
    workflow_type: str
    parent_id: Optional[str] = None
    intent: Optional[str] = None
    trigger_source: Optional[str] = None
    status: str = "created"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    request_count: int = 1
    children: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstanceState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# STATE STORAGE
# =============================================================================

class StateStorage:
    """
    Azure Blob Storage backend for session and instance state persistence.
    
    Blob structure:
        sessions/{session_id}.json
        instances/{instance_id}.json
        indexes/session_instances/{session_id}.json  (list of instance IDs)
        indexes/trigger/{session_id}_{workflow_type}_{trigger}.json  (dedup index)
    """
    
    def __init__(
        self,
        connection_string: str,
        container_name: str = "orchestrator-state",
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 60
    ):
        """
        Initialize state storage.
        
        Args:
            connection_string: Azure Blob Storage connection string
            container_name: Container for state blobs
            cache_enabled: Enable in-memory caching
            cache_ttl_seconds: Cache TTL in seconds
        """
        if not AZURE_AVAILABLE:
            raise ImportError(
                "Azure Storage SDK not available. "
                "Install with: pip install azure-storage-blob"
            )
        
        self.connection_string = connection_string
        self.container_name = container_name
        self.cache_enabled = cache_enabled
        self.cache_ttl_seconds = cache_ttl_seconds
        
        # Initialize Azure clients
        self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service.get_container_client(container_name)
        
        # Ensure container exists
        self._ensure_container_exists()
        
        # In-memory cache
        self._cache: Dict[str, tuple] = {}  # key -> (data, timestamp)
        self._cache_lock = RLock()
        
        logger.info(
            f"[StateStorage] Initialized with container: {container_name}, "
            f"cache_enabled={cache_enabled}"
        )
    
    def _ensure_container_exists(self) -> None:
        """Create container if it doesn't exist."""
        try:
            self.container_client.create_container()
            logger.info(f"[StateStorage] Created container: {self.container_name}")
        except ResourceExistsError:
            pass
    
    def _get_blob_client(self, path: str) -> BlobClient:
        """Get blob client for a path."""
        return self.container_client.get_blob_client(path)
    
    def _cache_get(self, key: str) -> Optional[Dict]:
        """Get from cache if valid."""
        if not self.cache_enabled:
            return None
        
        with self._cache_lock:
            if key in self._cache:
                data, timestamp = self._cache[key]
                if time.time() - timestamp < self.cache_ttl_seconds:
                    return data
                else:
                    del self._cache[key]
        return None
    
    def _cache_set(self, key: str, data: Dict) -> None:
        """Set cache entry."""
        if not self.cache_enabled:
            return
        
        with self._cache_lock:
            self._cache[key] = (data, time.time())
    
    def _cache_delete(self, key: str) -> None:
        """Delete cache entry."""
        with self._cache_lock:
            self._cache.pop(key, None)
    
    # =========================================================================
    # SESSION OPERATIONS
    # =========================================================================
    
    def save_session(self, session_id: str, session: SessionState) -> bool:
        """
        Save session state to Azure Blob.
        
        Args:
            session_id: Session ID
            session: SessionState to save
            
        Returns:
            True if saved successfully
        """
        path = f"sessions/{session_id}.json"
        blob_client = self._get_blob_client(path)
        
        # Update timestamp
        session.updated_at = datetime.utcnow().isoformat()
        data = session.to_dict()
        
        try:
            blob_client.upload_blob(
                json.dumps(data, default=str),
                overwrite=True
            )
            
            # Update cache
            self._cache_set(f"session:{session_id}", data)
            
            logger.debug(f"[StateStorage] Saved session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"[StateStorage] Failed to save session {session_id}: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        Get session state from Azure Blob.
        
        Args:
            session_id: Session ID
            
        Returns:
            SessionState or None if not found
        """
        # Check cache first
        cached = self._cache_get(f"session:{session_id}")
        if cached:
            return SessionState.from_dict(cached)
        
        path = f"sessions/{session_id}.json"
        blob_client = self._get_blob_client(path)
        
        try:
            content = blob_client.download_blob().readall()
            data = json.loads(content)
            
            # Cache the result
            self._cache_set(f"session:{session_id}", data)
            
            return SessionState.from_dict(data)
            
        except ResourceNotFoundError:
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session state."""
        path = f"sessions/{session_id}.json"
        blob_client = self._get_blob_client(path)
        
        try:
            blob_client.delete_blob()
            self._cache_delete(f"session:{session_id}")
            logger.info(f"[StateStorage] Deleted session: {session_id}")
            return True
        except ResourceNotFoundError:
            return True
    
    def session_exists(self, session_id: str) -> bool:
        """Check if session exists."""
        if self._cache_get(f"session:{session_id}"):
            return True
        
        path = f"sessions/{session_id}.json"
        blob_client = self._get_blob_client(path)
        return blob_client.exists()
    
    # =========================================================================
    # INSTANCE OPERATIONS
    # =========================================================================
    
    def save_instance(self, instance: InstanceState) -> bool:
        """
        Save instance state to Azure Blob.
        
        Also updates:
        - Parent's children list (if parent_id set)
        - Session's instance index
        - Trigger deduplication index
        """
        path = f"instances/{instance.instance_id}.json"
        blob_client = self._get_blob_client(path)
        
        # Update timestamp
        instance.updated_at = datetime.utcnow().isoformat()
        data = instance.to_dict()
        
        try:
            blob_client.upload_blob(
                json.dumps(data, default=str),
                overwrite=True
            )
            
            # Update cache
            self._cache_set(f"instance:{instance.instance_id}", data)
            
            # Update trigger index for deduplication
            if instance.trigger_source:
                self._save_trigger_index(
                    instance.session_id,
                    instance.workflow_type,
                    instance.trigger_source,
                    instance.instance_id
                )
            
            # Update parent's children list
            if instance.parent_id:
                self._add_child_to_parent(instance.parent_id, instance.instance_id)
            
            logger.debug(f"[StateStorage] Saved instance: {instance.instance_id}")
            return True
            
        except Exception as e:
            logger.error(
                f"[StateStorage] Failed to save instance {instance.instance_id}: {e}"
            )
            raise
    
    def get_instance(self, instance_id: str) -> Optional[InstanceState]:
        """Get instance state from Azure Blob."""
        # Check cache
        cached = self._cache_get(f"instance:{instance_id}")
        if cached:
            return InstanceState.from_dict(cached)
        
        path = f"instances/{instance_id}.json"
        blob_client = self._get_blob_client(path)
        
        try:
            content = blob_client.download_blob().readall()
            data = json.loads(content)
            
            self._cache_set(f"instance:{instance_id}", data)
            return InstanceState.from_dict(data)
            
        except ResourceNotFoundError:
            return None
    
    def get_instance_by_trigger(
        self,
        session_id: str,
        workflow_type: str,
        trigger_source: str
    ) -> Optional[InstanceState]:
        """
        Get instance by trigger source (for deduplication).
        
        Uses the trigger index for O(1) lookup.
        """
        instance_id = self._get_trigger_index(
            session_id, workflow_type, trigger_source
        )
        
        if instance_id:
            return self.get_instance(instance_id)
        return None
    
    def delete_instance(self, instance_id: str) -> bool:
        """Delete instance state."""
        # Get instance first to clean up indexes
        instance = self.get_instance(instance_id)
        
        path = f"instances/{instance_id}.json"
        blob_client = self._get_blob_client(path)
        
        try:
            blob_client.delete_blob()
            self._cache_delete(f"instance:{instance_id}")
            
            # Clean up trigger index
            if instance and instance.trigger_source:
                self._delete_trigger_index(
                    instance.session_id,
                    instance.workflow_type,
                    instance.trigger_source
                )
            
            logger.info(f"[StateStorage] Deleted instance: {instance_id}")
            return True
            
        except ResourceNotFoundError:
            return True
    
    # =========================================================================
    # TRIGGER INDEX (For Deduplication)
    # =========================================================================
    
    def _get_trigger_index_path(
        self,
        session_id: str,
        workflow_type: str,
        trigger_source: str
    ) -> str:
        """Get blob path for trigger index."""
        safe_trigger = trigger_source.replace(":", "_").replace("/", "_")[:50]
        return f"indexes/trigger/{session_id}_{workflow_type}_{safe_trigger}.json"
    
    def _save_trigger_index(
        self,
        session_id: str,
        workflow_type: str,
        trigger_source: str,
        instance_id: str
    ) -> None:
        """Save trigger -> instance_id mapping."""
        path = self._get_trigger_index_path(session_id, workflow_type, trigger_source)
        blob_client = self._get_blob_client(path)
        
        try:
            blob_client.upload_blob(
                json.dumps({"instance_id": instance_id}),
                overwrite=True
            )
        except Exception as e:
            logger.warning(f"[StateStorage] Failed to save trigger index: {e}")
    
    def _get_trigger_index(
        self,
        session_id: str,
        workflow_type: str,
        trigger_source: str
    ) -> Optional[str]:
        """Get instance_id for a trigger."""
        path = self._get_trigger_index_path(session_id, workflow_type, trigger_source)
        blob_client = self._get_blob_client(path)
        
        try:
            content = blob_client.download_blob().readall()
            data = json.loads(content)
            return data.get("instance_id")
        except ResourceNotFoundError:
            return None
    
    def _delete_trigger_index(
        self,
        session_id: str,
        workflow_type: str,
        trigger_source: str
    ) -> None:
        """Delete trigger index."""
        path = self._get_trigger_index_path(session_id, workflow_type, trigger_source)
        blob_client = self._get_blob_client(path)
        
        try:
            blob_client.delete_blob()
        except ResourceNotFoundError:
            pass
    
    # =========================================================================
    # PARENT-CHILD RELATIONSHIPS
    # =========================================================================
    
    def _add_child_to_parent(self, parent_id: str, child_id: str) -> None:
        """Add child instance to parent's children list."""
        parent = self.get_instance(parent_id)
        if parent:
            if child_id not in parent.children:
                parent.children.append(child_id)
                self.save_instance(parent)
    
    def get_children(self, parent_id: str) -> List[InstanceState]:
        """Get all child instances of a parent."""
        parent = self.get_instance(parent_id)
        if not parent:
            return []
        
        children = []
        for child_id in parent.children:
            child = self.get_instance(child_id)
            if child:
                children.append(child)
        
        return children


# =============================================================================
# IN-MEMORY FALLBACK
# =============================================================================

class InMemoryStateStorage:
    """
    In-memory state storage for development/testing.
    
    WARNING: Data is not persistent!
    """
    
    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}
        self._instances: Dict[str, InstanceState] = {}
        self._trigger_index: Dict[str, str] = {}  # trigger_key -> instance_id
        self._lock = RLock()
        
        logger.warning("[InMemoryStateStorage] Using in-memory storage - NOT persistent!")
    
    def save_session(self, session_id: str, session: SessionState) -> bool:
        with self._lock:
            session.updated_at = datetime.utcnow().isoformat()
            self._sessions[session_id] = session
            return True
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        self._sessions.pop(session_id, None)
        return True
    
    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions
    
    def save_instance(self, instance: InstanceState) -> bool:
        with self._lock:
            instance.updated_at = datetime.utcnow().isoformat()
            self._instances[instance.instance_id] = instance
            
            # Trigger index
            if instance.trigger_source:
                key = f"{instance.session_id}_{instance.workflow_type}_{instance.trigger_source}"
                self._trigger_index[key] = instance.instance_id
            
            # Parent children
            if instance.parent_id and instance.parent_id in self._instances:
                parent = self._instances[instance.parent_id]
                if instance.instance_id not in parent.children:
                    parent.children.append(instance.instance_id)
            
            return True
    
    def get_instance(self, instance_id: str) -> Optional[InstanceState]:
        return self._instances.get(instance_id)
    
    def get_instance_by_trigger(
        self,
        session_id: str,
        workflow_type: str,
        trigger_source: str
    ) -> Optional[InstanceState]:
        key = f"{session_id}_{workflow_type}_{trigger_source}"
        instance_id = self._trigger_index.get(key)
        if instance_id:
            return self._instances.get(instance_id)
        return None
    
    def delete_instance(self, instance_id: str) -> bool:
        instance = self._instances.pop(instance_id, None)
        if instance and instance.trigger_source:
            key = f"{instance.session_id}_{instance.workflow_type}_{instance.trigger_source}"
            self._trigger_index.pop(key, None)
        return True
    
    def get_children(self, parent_id: str) -> List[InstanceState]:
        parent = self._instances.get(parent_id)
        if not parent:
            return []
        return [self._instances[cid] for cid in parent.children if cid in self._instances]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_state_storage(
    connection_string: Optional[str] = None,
    container_name: str = "orchestrator-state",
    use_memory_fallback: bool = False
):
    """
    Create a state storage instance.
    
    Args:
        connection_string: Azure connection string (None for in-memory)
        container_name: Azure container name
        use_memory_fallback: Force in-memory for testing
        
    Returns:
        StateStorage or InMemoryStateStorage
    """
    if use_memory_fallback or not connection_string:
        return InMemoryStateStorage()
    
    return StateStorage(connection_string, container_name)
