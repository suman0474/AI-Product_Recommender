# common/rag/__init__.py
# =============================================================================
# Unified RAG Package
# =============================================================================
#
# Sub-packages:
#   common.rag.shared    - Shared infrastructure (vector store, logger, aggregator, enrichment)
#   common.rag.standards - Standards RAG (IEC, ISO, API, SIL, ATEX …)
#   common.rag.strategy  - Strategy RAG (procurement CSV / MongoDB vendor filtering)
#   common.rag.index     - Index RAG (product search: DB + PDF + web)
#
# Backward-compatibility re-exports so existing code that imports from
# common.rag.* (old flat paths) continues to work without changes.
# =============================================================================

# ── Shared infrastructure ─────────────────────────────────────────────────────
from .shared.vector_store import get_vector_store
from .shared.logger import (
    RAGLogger,
    IndexRAGLogger,
    StandardsRAGLogger,
    StrategyRAGLogger,
    OrchestratorLogger,
    set_trace_id,
    get_trace_id,
    clear_trace_id,
    get_rag_logger,
    log_node_timing,
)
from .shared.components import (
    RAGAggregator,
    StrategyFilter,
    create_rag_aggregator,
    create_strategy_filter,
)
from .shared.enrichment import enrich_product_with_60_plus_specs

# ── Standards RAG ─────────────────────────────────────────────────────────────
from .standards import (
    StandardsRAGState,
    create_standards_rag_state,
    create_standards_rag_workflow,
    get_standards_rag_workflow,
    run_standards_rag_workflow,
    StandardsChatAgent,
    create_standards_chat_agent,
    get_standards_chat_agent,
    StandardsRAGMemory,
    standards_rag_memory,
    get_standards_rag_memory,
    resolve_standards_follow_up,
    add_to_standards_memory,
    clear_standards_memory,
    enrich_identified_items_with_standards,
    validate_items_against_domain_standards,
    is_standards_related_question,
    route_standards_question,
    StandardsBlobRetriever,
    get_standards_blob_retriever,
)

# ── Strategy RAG ──────────────────────────────────────────────────────────────
from .strategy import (
    StrategyRAGState,
    create_strategy_rag_state,
    create_strategy_rag_workflow,
    get_strategy_rag_workflow,
    run_strategy_rag_workflow,
    get_strategy_for_product,
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

# ── Index RAG ─────────────────────────────────────────────────────────────────
from .index import (
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
    # Shared
    "get_vector_store",
    "RAGLogger", "IndexRAGLogger", "StandardsRAGLogger", "StrategyRAGLogger",
    "OrchestratorLogger", "set_trace_id", "get_trace_id", "clear_trace_id",
    "get_rag_logger", "log_node_timing",
    "RAGAggregator", "StrategyFilter", "create_rag_aggregator", "create_strategy_filter",
    "enrich_product_with_60_plus_specs",
    # Standards
    "StandardsRAGState", "create_standards_rag_state", "create_standards_rag_workflow",
    "get_standards_rag_workflow", "run_standards_rag_workflow",
    "StandardsChatAgent", "create_standards_chat_agent", "get_standards_chat_agent",
    "StandardsRAGMemory", "standards_rag_memory", "get_standards_rag_memory",
    "resolve_standards_follow_up", "add_to_standards_memory", "clear_standards_memory",
    "enrich_identified_items_with_standards", "validate_items_against_domain_standards",
    "is_standards_related_question", "route_standards_question",
    "StandardsBlobRetriever", "get_standards_blob_retriever",
    # Strategy
    "StrategyRAGState", "create_strategy_rag_state", "create_strategy_rag_workflow",
    "get_strategy_rag_workflow", "run_strategy_rag_workflow", "get_strategy_for_product",
    "StrategyChatAgent", "create_strategy_chat_agent", "get_strategy_chat_agent",
    "load_strategy_from_mongodb", "filter_vendors_by_strategy", "get_vendor_strategy_info",
    "get_strategy_filter", "enrich_with_strategy_rag", "enrich_schema_with_strategy",
    "get_strategy_with_auto_fallback", "filter_vendors_by_strategy_data",
    "is_strategy_related_question", "route_strategy_question", "enrich_with_strategy_llm_fallback",
    "StrategyRAGMemory", "strategy_rag_memory", "get_strategy_rag_memory",
    "add_to_strategy_memory", "clear_strategy_memory",
    # Index
    "IndexRAGState", "create_index_rag_state", "create_index_rag_workflow",
    "get_index_rag_workflow", "run_index_rag_workflow",
    "IndexRAGAgent", "create_index_rag_agent", "run_index_rag",
    "IndexRAGMemory", "index_rag_memory", "get_index_rag_memory",
    "resolve_follow_up_query", "add_to_conversation_memory", "clear_conversation_memory",
]
