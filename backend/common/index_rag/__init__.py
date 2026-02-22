# common/index_rag/__init__.py
# Backward-compatibility shim — canonical location is now common.rag.index

from common.rag.index import (
    IndexRAGState,
    create_index_rag_state,
    create_index_rag_workflow,
    get_index_rag_workflow,
    run_index_rag_workflow,
    IndexRAGAgent,
    create_index_rag_agent,
    run_index_rag,
    IndexRAGMemory,
    index_rag_memory,
    get_index_rag_memory,
    resolve_follow_up_query,
    add_to_conversation_memory,
    clear_conversation_memory,
)

__all__ = [
    'IndexRAGState', 'create_index_rag_state', 'create_index_rag_workflow',
    'get_index_rag_workflow', 'run_index_rag_workflow',
    'IndexRAGAgent', 'create_index_rag_agent', 'run_index_rag',
    'IndexRAGMemory', 'index_rag_memory', 'get_index_rag_memory',
    'resolve_follow_up_query', 'add_to_conversation_memory', 'clear_conversation_memory',
]
