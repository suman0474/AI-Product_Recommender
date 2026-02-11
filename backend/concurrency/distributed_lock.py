"""
Distributed Lock - Azure Blob Storage Lease-Based Distributed Locking

Provides:
- Distributed locks using Azure Blob Storage leases
- Automatic lock release on context exit
- Timeout handling with retry logic
- Thread-safe lock acquisition

Usage:
    from concurrency.distributed_lock import AzureBlobDistributedLock
    
    lock = AzureBlobDistributedLock(connection_string, container_name)
    
    with lock.acquire("session:abc123", lease_duration=30, timeout=10):
        # Only ONE process can execute this block at a time
        create_session()
    # Lock automatically released
"""

import time
import logging
from contextlib import contextmanager
from typing import Optional, Generator
from dataclasses import dataclass
from enum import Enum

# Azure SDK imports
try:
    from azure.storage.blob import BlobServiceClient, BlobLeaseClient, BlobClient
    from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# LOCK STATUS
# =============================================================================

class LockStatus(Enum):
    """Status of a distributed lock."""
    ACQUIRED = "acquired"
    RELEASED = "released"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class LockResult:
    """Result of a lock acquisition attempt."""
    status: LockStatus
    lock_name: str
    lease_id: Optional[str] = None
    duration_ms: float = 0.0
    error: Optional[str] = None


# =============================================================================
# DISTRIBUTED LOCK
# =============================================================================

class AzureBlobDistributedLock:
    """
    Distributed lock using Azure Blob Storage leases.
    
    Azure Blob leases provide:
    - Mutual exclusion across multiple processes/machines
    - Automatic lease expiry if holder crashes
    - Atomic acquire/release operations
    
    Lock keys are stored as blobs in the format: locks/{lock_name}
    """
    
    def __init__(
        self,
        connection_string: str,
        container_name: str = "orchestrator-state",
        locks_prefix: str = "locks"
    ):
        """
        Initialize distributed lock manager.
        
        Args:
            connection_string: Azure Blob Storage connection string
            container_name: Container to store lock blobs
            locks_prefix: Prefix for lock blob paths
        """
        if not AZURE_AVAILABLE:
            raise ImportError(
                "Azure Storage SDK not available. "
                "Install with: pip install azure-storage-blob"
            )
        
        self.connection_string = connection_string
        self.container_name = container_name
        self.locks_prefix = locks_prefix
        
        # Initialize blob service client
        self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service.get_container_client(container_name)
        
        # Ensure container exists
        self._ensure_container_exists()
        
        logger.info(f"[DistributedLock] Initialized with container: {container_name}")
    
    def _ensure_container_exists(self) -> None:
        """Create container if it doesn't exist."""
        try:
            self.container_client.create_container()
            logger.info(f"[DistributedLock] Created container: {self.container_name}")
        except ResourceExistsError:
            pass  # Container already exists
    
    def _get_lock_blob_path(self, lock_name: str) -> str:
        """Get blob path for a lock name."""
        # Sanitize lock name (replace colons with underscores for blob path)
        safe_name = lock_name.replace(":", "_").replace("/", "_")
        return f"{self.locks_prefix}/{safe_name}"
    
    def _ensure_lock_blob_exists(self, blob_client: BlobClient) -> None:
        """Create lock blob if it doesn't exist."""
        try:
            blob_client.upload_blob(b"lock", overwrite=False)
        except ResourceExistsError:
            pass  # Blob already exists
    
    @contextmanager
    def acquire(
        self,
        lock_name: str,
        lease_duration: int = 30,
        timeout: int = 10,
        retry_interval: float = 0.1
    ) -> Generator[str, None, None]:
        """
        Acquire distributed lock with automatic release.
        
        Args:
            lock_name: Unique name for the lock (e.g., "session:abc123")
            lease_duration: How long to hold the lease (15-60 seconds, or -1 for infinite)
            timeout: Maximum time to wait for lock acquisition
            retry_interval: Time between retry attempts
            
        Yields:
            Lease ID if successful
            
        Raises:
            TimeoutError: If lock cannot be acquired within timeout
        """
        blob_path = self._get_lock_blob_path(lock_name)
        blob_client = self.container_client.get_blob_client(blob_path)
        
        # Ensure lock blob exists
        self._ensure_lock_blob_exists(blob_client)
        
        lease_client = BlobLeaseClient(blob_client)
        start_time = time.time()
        lease_id = None
        
        logger.debug(f"[DistributedLock] Attempting to acquire lock: {lock_name}")
        
        while True:
            try:
                # Try to acquire lease
                lease_id = lease_client.acquire(lease_duration=lease_duration)
                duration_ms = (time.time() - start_time) * 1000
                
                logger.info(
                    f"[DistributedLock] Acquired lock: {lock_name} "
                    f"(lease_id={lease_id[:8]}..., duration_ms={duration_ms:.2f})"
                )
                
                try:
                    yield lease_id
                finally:
                    # Always release the lease
                    try:
                        lease_client.release()
                        logger.debug(f"[DistributedLock] Released lock: {lock_name}")
                    except Exception as e:
                        logger.warning(
                            f"[DistributedLock] Failed to release lock {lock_name}: {e}"
                        )
                return
                
            except Exception as e:
                elapsed = time.time() - start_time
                
                if elapsed >= timeout:
                    logger.error(
                        f"[DistributedLock] Timeout acquiring lock: {lock_name} "
                        f"after {elapsed:.2f}s"
                    )
                    raise TimeoutError(
                        f"Could not acquire lock '{lock_name}' within {timeout}s"
                    )
                
                # Wait and retry
                logger.debug(
                    f"[DistributedLock] Lock busy, retrying: {lock_name} "
                    f"(elapsed={elapsed:.2f}s)"
                )
                time.sleep(retry_interval)
    
    def try_acquire(
        self,
        lock_name: str,
        lease_duration: int = 30
    ) -> Optional[str]:
        """
        Try to acquire lock without waiting.
        
        Args:
            lock_name: Unique name for the lock
            lease_duration: How long to hold the lease
            
        Returns:
            Lease ID if acquired, None if lock is busy
        """
        blob_path = self._get_lock_blob_path(lock_name)
        blob_client = self.container_client.get_blob_client(blob_path)
        
        self._ensure_lock_blob_exists(blob_client)
        
        lease_client = BlobLeaseClient(blob_client)
        
        try:
            lease_id = lease_client.acquire(lease_duration=lease_duration)
            logger.info(f"[DistributedLock] Acquired lock (non-blocking): {lock_name}")
            return lease_id
        except Exception:
            logger.debug(f"[DistributedLock] Lock busy (non-blocking): {lock_name}")
            return None
    
    def release(self, lock_name: str, lease_id: str) -> bool:
        """
        Manually release a lock.
        
        Args:
            lock_name: Lock name
            lease_id: Lease ID from acquire
            
        Returns:
            True if released, False if failed
        """
        blob_path = self._get_lock_blob_path(lock_name)
        blob_client = self.container_client.get_blob_client(blob_path)
        lease_client = BlobLeaseClient(blob_client, lease_id=lease_id)
        
        try:
            lease_client.release()
            logger.info(f"[DistributedLock] Manually released lock: {lock_name}")
            return True
        except Exception as e:
            logger.error(f"[DistributedLock] Failed to release lock {lock_name}: {e}")
            return False


# =============================================================================
# IN-MEMORY FALLBACK (For development/testing)
# =============================================================================

class InMemoryDistributedLock:
    """
    In-memory distributed lock for development/testing.
    
    WARNING: This only works within a single process!
    Use AzureBlobDistributedLock for production.
    """
    
    def __init__(self):
        import threading
        self._locks: dict = {}
        self._lock = threading.RLock()
        logger.warning(
            "[InMemoryLock] Using in-memory locks - NOT suitable for production!"
        )
    
    @contextmanager
    def acquire(
        self,
        lock_name: str,
        lease_duration: int = 30,
        timeout: int = 10,
        retry_interval: float = 0.1
    ) -> Generator[str, None, None]:
        """Acquire in-memory lock."""
        import threading
        
        start_time = time.time()
        
        while True:
            with self._lock:
                if lock_name not in self._locks:
                    # Lock is free, acquire it
                    lock = threading.RLock()
                    self._locks[lock_name] = lock
                    lock.acquire()
                    
                    try:
                        yield f"inmem-{lock_name}"
                    finally:
                        lock.release()
                        del self._locks[lock_name]
                    return
            
            # Lock is busy
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(f"Could not acquire lock '{lock_name}'")
            
            time.sleep(retry_interval)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_distributed_lock(
    connection_string: Optional[str] = None,
    container_name: str = "orchestrator-state",
    use_memory_fallback: bool = False
):
    """
    Create a distributed lock instance.
    
    Args:
        connection_string: Azure connection string (None for in-memory)
        container_name: Azure container name
        use_memory_fallback: Force in-memory lock for testing
        
    Returns:
        Lock instance (Azure or in-memory)
    """
    if use_memory_fallback or not connection_string:
        return InMemoryDistributedLock()
    
    return AzureBlobDistributedLock(connection_string, container_name)
