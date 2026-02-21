# agentic/strategy_rag/strategy_vector_store.py
# =============================================================================
# STRATEGY VECTOR STORE - Pinecone-based semantic search for strategy data
# =============================================================================
#
# PURPOSE: Replace hardcoded PRODUCT_TO_CATEGORY_MAP with semantic matching
# using vector embeddings for flexible product-to-strategy mapping.
#
# MIGRATION: Migrated from ChromaDB to Pinecone for production readiness
#
# =============================================================================
#
# DEPRECATED: This file is deprecated. Please use strategy_vector_store_pinecone.py
# This wrapper maintains backward compatibility during migration.
#
# =============================================================================

import os
import logging

logger = logging.getLogger(__name__)

# Path to strategy CSV (kept for compatibility)
STRATEGY_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "instrumentation_procurement_strategy.csv"
)

# Import the new Pinecone-based implementation
try:
    from .strategy_vector_store_pinecone import (
        StrategyVectorStore,
        get_strategy_vector_store,
        init_strategy_store,
        search_strategy,
        match_product_category,
        get_vendor_score,
        PINECONE_AVAILABLE
    )
    logger.info("[StrategyVectorStore] Using Pinecone-based implementation")
    
    # Legacy compatibility flags
    CHROMADB_AVAILABLE = False  # Deprecated
    SENTENCE_TRANSFORMERS_AVAILABLE = False  # Handled in Pinecone implementation
    
except ImportError as e:
    logger.error(f"[StrategyVectorStore] Failed to import Pinecone implementation: {e}")
    raise


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'StrategyVectorStore',
    'get_strategy_vector_store',
    'init_strategy_store',
    'search_strategy',
    'match_product_category',
    'get_vendor_score',
    'STRATEGY_CSV_PATH',
    'CHROMADB_AVAILABLE',  # Deprecated, always False
    'PINECONE_AVAILABLE'
]
