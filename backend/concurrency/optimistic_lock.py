"""
Optimistic Lock - ETag-Based Optimistic Locking for Concurrent Updates

Provides:
- Optimistic locking using Azure Blob ETags
- Version tracking for state updates
- Conflict detection without blocking
- Retry logic for transient failures

Usage:
    from concurrency.optimistic_lock import OptimisticLockManager, VersionedState
    
    manager = OptimisticLockManager(blob_service, container)
    
    # Read with version
    state = manager.read_with_version("session:abc123")
    
    # Modify and write back
    new_data = {**state.data, "updated": True}
    success = manager.write_if_unchanged("session:abc123", state, new_data)
    
    if not success:
        # Concurrent modification detected - retry or handle conflict
        raise ConcurrentModificationError()
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from enum import Enum

# Azure SDK imports
try:
    from azure.storage.blob import BlobServiceClient, BlobClient
    from azure.core.exceptions import ResourceModifiedError, ResourceNotFoundError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class ConcurrentModificationError(Exception):
    """Raised when a concurrent modification is detected."""
    def __init__(self, key: str, message: str = None):
        self.key = key
        self.message = message or f"Concurrent modification detected for key: {key}"
        super().__init__(self.message)


class StateNotFoundError(Exception):
    """Raised when state is not found."""
    def __init__(self, key: str):
        self.key = key
        super().__init__(f"State not found for key: {key}")


# =============================================================================
# VERSIONED STATE
# =============================================================================

@dataclass
class VersionedState:
    """
    State with version information for optimistic locking.
    
    Attributes:
        key: State key
        data: Actual state data
        etag: Azure Blob ETag for compare-and-swap
        version: Application-level version number
    """
    key: str
    data: Dict[str, Any]
    etag: str
    version: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "data": self.data,
            "etag": self.etag,
            "version": self.version
        }


# =============================================================================
# OPTIMISTIC LOCK MANAGER
# =============================================================================

class OptimisticLockManager:
    """
    Optimistic locking using Azure Blob ETags.
    
    How it works:
    1. Read state with ETag (version identifier)
    2. Modify state locally
    3. Write back with "if-match: ETag" condition
    4. If ETag mismatch → concurrent modification detected
    
    This pattern allows high concurrency without blocking.
    """
    
    def __init__(
        self,
        connection_string: str,
        container_name: str = "orchestrator-state",
        state_prefix: str = "state"
    ):
        """
        Initialize optimistic lock manager.
        
        Args:
            connection_string: Azure Blob Storage connection string
            container_name: Container for state blobs
            state_prefix: Prefix for state blob paths
        """
        if not AZURE_AVAILABLE:
            raise ImportError(
                "Azure Storage SDK not available. "
                "Install with: pip install azure-storage-blob"
            )
        
        self.connection_string = connection_string
        self.container_name = container_name
        self.state_prefix = state_prefix
        
        self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service.get_container_client(container_name)
        
        logger.info(f"[OptimisticLock] Initialized with container: {container_name}")
    
    def _get_state_path(self, key: str) -> str:
        """Get blob path for a state key."""
        safe_key = key.replace(":", "_").replace("/", "_")
        return f"{self.state_prefix}/{safe_key}.json"
    
    def read_with_version(self, key: str) -> VersionedState:
        """
        Read state with version info for later compare-and-swap.
        
        Args:
            key: State key
            
        Returns:
            VersionedState with data and ETag
            
        Raises:
            StateNotFoundError: If state doesn't exist
        """
        blob_path = self._get_state_path(key)
        blob_client = self.container_client.get_blob_client(blob_path)
        
        try:
            # Get blob properties (includes ETag)
            props = blob_client.get_blob_properties()
            
            # Download content
            content = blob_client.download_blob().readall()
            data = json.loads(content)
            
            version = data.get("_version", 1)
            
            logger.debug(
                f"[OptimisticLock] Read state: {key} "
                f"(version={version}, etag={props.etag[:16]}...)"
            )
            
            return VersionedState(
                key=key,
                data=data,
                etag=props.etag,
                version=version
            )
            
        except ResourceNotFoundError:
            raise StateNotFoundError(key)
    
    def write_if_unchanged(
        self,
        key: str,
        current_state: VersionedState,
        new_data: Dict[str, Any]
    ) -> bool:
        """
        Update state only if ETag matches (no concurrent modification).
        
        Args:
            key: State key
            current_state: State read earlier with read_with_version()
            new_data: New data to write
            
        Returns:
            True if write succeeded, False if concurrent modification
        """
        blob_path = self._get_state_path(key)
        blob_client = self.container_client.get_blob_client(blob_path)
        
        # Increment version
        new_data["_version"] = current_state.version + 1
        content = json.dumps(new_data, default=str)
        
        try:
            blob_client.upload_blob(
                content,
                overwrite=True,
                etag=current_state.etag,
                match_condition="IfMatch"
            )
            
            logger.info(
                f"[OptimisticLock] Updated state: {key} "
                f"(version {current_state.version} -> {new_data['_version']})"
            )
            return True
            
        except ResourceModifiedError:
            logger.warning(
                f"[OptimisticLock] Concurrent modification detected: {key}"
            )
            return False
    
    def write_new(
        self,
        key: str,
        data: Dict[str, Any],
        fail_if_exists: bool = False
    ) -> VersionedState:
        """
        Write new state (initial creation).
        
        Args:
            key: State key
            data: Data to write
            fail_if_exists: If True, fail if state already exists
            
        Returns:
            VersionedState of created state
        """
        blob_path = self._get_state_path(key)
        blob_client = self.container_client.get_blob_client(blob_path)
        
        # Set initial version
        data["_version"] = 1
        content = json.dumps(data, default=str)
        
        try:
            blob_client.upload_blob(
                content,
                overwrite=not fail_if_exists
            )
            
            # Get the new ETag
            props = blob_client.get_blob_properties()
            
            logger.info(f"[OptimisticLock] Created state: {key}")
            
            return VersionedState(
                key=key,
                data=data,
                etag=props.etag,
                version=1
            )
            
        except Exception as e:
            logger.error(f"[OptimisticLock] Failed to create state {key}: {e}")
            raise
    
    def delete(self, key: str, current_state: Optional[VersionedState] = None) -> bool:
        """
        Delete state, optionally with version check.
        
        Args:
            key: State key
            current_state: If provided, only delete if ETag matches
            
        Returns:
            True if deleted, False if concurrent modification
        """
        blob_path = self._get_state_path(key)
        blob_client = self.container_client.get_blob_client(blob_path)
        
        try:
            if current_state:
                blob_client.delete_blob(
                    etag=current_state.etag,
                    match_condition="IfMatch"
                )
            else:
                blob_client.delete_blob()
            
            logger.info(f"[OptimisticLock] Deleted state: {key}")
            return True
            
        except ResourceModifiedError:
            logger.warning(
                f"[OptimisticLock] Concurrent modification on delete: {key}"
            )
            return False
        except ResourceNotFoundError:
            logger.debug(f"[OptimisticLock] State already deleted: {key}")
            return True
    
    def exists(self, key: str) -> bool:
        """Check if state exists."""
        blob_path = self._get_state_path(key)
        blob_client = self.container_client.get_blob_client(blob_path)
        return blob_client.exists()


# =============================================================================
# IN-MEMORY FALLBACK (For development/testing)
# =============================================================================

class InMemoryOptimisticLockManager:
    """
    In-memory optimistic lock manager for development/testing.
    
    WARNING: Data is not persistent!
    """
    
    def __init__(self):
        import threading
        self._store: Dict[str, Tuple[Dict, int]] = {}  # key -> (data, version)
        self._lock = threading.RLock()
        logger.warning(
            "[InMemoryOptimisticLock] Using in-memory storage - NOT persistent!"
        )
    
    def read_with_version(self, key: str) -> VersionedState:
        """Read state with version."""
        with self._lock:
            if key not in self._store:
                raise StateNotFoundError(key)
            
            data, version = self._store[key]
            return VersionedState(
                key=key,
                data=data.copy(),
                etag=f"v{version}",
                version=version
            )
    
    def write_if_unchanged(
        self,
        key: str,
        current_state: VersionedState,
        new_data: Dict[str, Any]
    ) -> bool:
        """Update if version matches."""
        with self._lock:
            if key not in self._store:
                return False
            
            _, current_version = self._store[key]
            if current_version != current_state.version:
                return False
            
            new_data["_version"] = current_version + 1
            self._store[key] = (new_data, current_version + 1)
            return True
    
    def write_new(
        self,
        key: str,
        data: Dict[str, Any],
        fail_if_exists: bool = False
    ) -> VersionedState:
        """Write new state."""
        with self._lock:
            if fail_if_exists and key in self._store:
                raise Exception(f"State already exists: {key}")
            
            data["_version"] = 1
            self._store[key] = (data, 1)
            
            return VersionedState(
                key=key,
                data=data,
                etag="v1",
                version=1
            )
    
    def delete(self, key: str, current_state: Optional[VersionedState] = None) -> bool:
        """Delete state."""
        with self._lock:
            if key not in self._store:
                return True
            
            if current_state:
                _, version = self._store[key]
                if version != current_state.version:
                    return False
            
            del self._store[key]
            return True
    
    def exists(self, key: str) -> bool:
        """Check if exists."""
        return key in self._store


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_optimistic_lock_manager(
    connection_string: Optional[str] = None,
    container_name: str = "orchestrator-state",
    use_memory_fallback: bool = False
):
    """
    Create an optimistic lock manager instance.
    
    Args:
        connection_string: Azure connection string (None for in-memory)
        container_name: Azure container name
        use_memory_fallback: Force in-memory for testing
        
    Returns:
        OptimisticLockManager or InMemoryOptimisticLockManager
    """
    if use_memory_fallback or not connection_string:
        return InMemoryOptimisticLockManager()
    
    return OptimisticLockManager(connection_string, container_name)
