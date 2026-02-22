# common/rag/strategy/__init__.py
# Strategy RAG – filter and prioritize vendors from procurement strategy CSV / MongoDB

from .workflow import (
    StrategyRAGState,
    create_strategy_rag_state,
    create_strategy_rag_workflow,
    get_strategy_rag_workflow,
    run_strategy_rag_workflow,
    get_strategy_for_product,
)
from .chat_agent import (
    StrategyChatAgent,
    create_strategy_chat_agent,
    get_strategy_chat_agent,
)
from .mongodb_loader import (
    load_strategy_from_mongodb,
    filter_vendors_by_strategy,
    get_vendor_strategy_info,
    get_strategy_filter,
)
from .enrichment import (
    enrich_with_strategy_rag,
    get_strategy_with_auto_fallback,
    enrich_schema_with_strategy,
    filter_vendors_by_strategy_data,
    is_strategy_related_question,
    route_strategy_question,
    enrich_with_strategy_llm_fallback,
)
from .memory import (
    StrategyRAGMemory,
    strategy_rag_memory,
    get_strategy_rag_memory,
    add_to_strategy_memory,
    clear_strategy_memory,
)
__all__ = [
    # Workflow
    "StrategyRAGState",
    "create_strategy_rag_state",
    "create_strategy_rag_workflow",
    "get_strategy_rag_workflow",
    "run_strategy_rag_workflow",
    "get_strategy_for_product",
    # Chat Agent
    "StrategyChatAgent",
    "create_strategy_chat_agent",
    "get_strategy_chat_agent",
    # MongoDB Loader (replaces CSV Filter)
    "load_strategy_from_mongodb",
    "filter_vendors_by_strategy",
    "get_vendor_strategy_info",
    "get_strategy_filter",
    # Enrichment
    "enrich_with_strategy_rag",
    "get_strategy_with_auto_fallback",
    "enrich_schema_with_strategy",
    "filter_vendors_by_strategy_data",
    "is_strategy_related_question",
    "route_strategy_question",
    "enrich_with_strategy_llm_fallback",
    # Memory
    "StrategyRAGMemory",
    "strategy_rag_memory",
    "get_strategy_rag_memory",
    "add_to_strategy_memory",
    "clear_strategy_memory",
]
