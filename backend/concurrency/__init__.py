"""
Concurrency Package - Production-grade distributed locking and optimistic concurrency

Provides:
- Distributed locks using Azure Blob Storage leases
- Optimistic locking using ETags
- In-memory fallbacks for development/testing
"""

from .distributed_lock import (
    AzureBlobDistributedLock,
    InMemoryDistributedLock,
    LockStatus,
    LockResult,
    create_distributed_lock,
)

from .optimistic_lock import (
    OptimisticLockManager,
    InMemoryOptimisticLockManager,
    VersionedState,
    ConcurrentModificationError,
    StateNotFoundError,
    create_optimistic_lock_manager,
)

__all__ = [
    # Distributed locks
    "AzureBlobDistributedLock",
    "InMemoryDistributedLock",
    "LockStatus",
    "LockResult",
    "create_distributed_lock",
    # Optimistic locks
    "OptimisticLockManager",
    "InMemoryOptimisticLockManager",
    "VersionedState",
    "ConcurrentModificationError",
    "StateNotFoundError",
    "create_optimistic_lock_manager",
]
