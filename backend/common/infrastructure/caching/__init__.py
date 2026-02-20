# Caching Module
# =============================================================================
# Consolidated Caching Infrastructure
# =============================================================================

from .base_cache import BaseLRUCache, HashKeyCache, CompositeKeyCache, create_cache_singleton
from .bounded_cache import BoundedCache, BoundedCacheManager, get_or_create_cache, get_cache, cleanup_all_caches, clear_all_caches
from .embedding_cache import EmbeddingCacheManager, get_embedding_cache
from .llm_response_cache import LLMResponseCache, get_llm_response_cache
from .rag_cache import RAGCache, get_rag_cache, cache_get, cache_set
from .schema_cache import SchemaCache, get_schema_cache
from .cache_cleanup_task import start_cache_cleanup, stop_cache_cleanup
from .workflow_state_cache import BoundedWorkflowStateManager, get_workflow_state_manager, stop_workflow_state_manager
from .failure_memory_cache import SchemaFailureMemory, get_schema_failure_memory, FailureType, RecoveryAction, reset_failure_memory, FailureEntry, SuccessEntry, FailurePattern

__all__ = [
    'BaseLRUCache',
    'HashKeyCache',
    'CompositeKeyCache',
    'create_cache_singleton',
    'BoundedCache',
    'BoundedCacheManager',
    'get_or_create_cache',
    'get_cache',
    'cleanup_all_caches',
    'clear_all_caches',
    'EmbeddingCacheManager',
    'get_embedding_cache',
    'LLMResponseCache',
    'get_llm_response_cache',
    'RAGCache',
    'get_rag_cache',
    'cache_get',
    'cache_set',
    'SchemaCache',
    'get_schema_cache',
    'start_cache_cleanup',
    'stop_cache_cleanup',
    'BoundedWorkflowStateManager',
    'get_workflow_state_manager',
    'stop_workflow_state_manager',
    'SchemaFailureMemory',
    'get_schema_failure_memory',
    'FailureType',
    'RecoveryAction',
    'reset_failure_memory',
    'FailureEntry',
    'SuccessEntry',
    'FailurePattern'
]

