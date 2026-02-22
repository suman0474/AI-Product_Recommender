# common/strategy_rag/__init__.py
# Backward-compatibility shim — canonical location is now common.rag.strategy

from common.rag.strategy import (
    StrategyRAGState,
    create_strategy_rag_state,
    create_strategy_rag_workflow,
    run_strategy_rag_workflow,
    get_strategy_for_product,
    get_strategy_rag_workflow,
    StrategyChatAgent,
    create_strategy_chat_agent,
    get_strategy_chat_agent,
    load_strategy_from_mongodb,
    filter_vendors_by_strategy,
    get_vendor_strategy_info,
    get_strategy_filter,
    enrich_with_strategy_rag,
    get_strategy_with_auto_fallback,
    enrich_schema_with_strategy,
    filter_vendors_by_strategy_data,
    is_strategy_related_question,
    route_strategy_question,
    enrich_with_strategy_llm_fallback,
    StrategyRAGMemory,
    strategy_rag_memory,
    get_strategy_rag_memory,
    add_to_strategy_memory,
    clear_strategy_memory,
)

__all__ = [
    'StrategyRAGState', 'create_strategy_rag_state', 'create_strategy_rag_workflow',
    'run_strategy_rag_workflow', 'get_strategy_for_product', 'get_strategy_rag_workflow',
    'StrategyChatAgent', 'create_strategy_chat_agent', 'get_strategy_chat_agent',
    # MongoDB Loader (replaces CSV Filter)
    'load_strategy_from_mongodb', 'filter_vendors_by_strategy', 'get_vendor_strategy_info',
    'get_strategy_filter', 'enrich_with_strategy_rag', 'enrich_schema_with_strategy',
    'get_strategy_with_auto_fallback', 'filter_vendors_by_strategy_data',
    'is_strategy_related_question', 'route_strategy_question', 'enrich_with_strategy_llm_fallback',
    'StrategyRAGMemory', 'strategy_rag_memory', 'get_strategy_rag_memory',
    'add_to_strategy_memory', 'clear_strategy_memory',
]
