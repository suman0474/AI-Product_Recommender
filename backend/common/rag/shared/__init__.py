# common/rag/shared/__init__.py
# Shared RAG infrastructure - vector store, logger, enrichment, aggregator

from .vector_store import get_vector_store
from .logger import (
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
from .components import RAGAggregator, StrategyFilter, create_rag_aggregator, create_strategy_filter
from .enrichment import enrich_product_with_60_plus_specs

__all__ = [
    # Vector Store
    "get_vector_store",
    # Logger
    "RAGLogger",
    "IndexRAGLogger",
    "StandardsRAGLogger",
    "StrategyRAGLogger",
    "OrchestratorLogger",
    "set_trace_id",
    "get_trace_id",
    "clear_trace_id",
    "get_rag_logger",
    "log_node_timing",
    # Components / Aggregator
    "RAGAggregator",
    "StrategyFilter",
    "create_rag_aggregator",
    "create_strategy_filter",
    # Enrichment
    "enrich_product_with_60_plus_specs",
]
