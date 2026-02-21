# rag/vector_store.py
# Vector Store Implementation - Pinecone Only
# FAISS and ChromaDB have been completely removed
#
# Available backends:
# 1. Pinecone - Primary (managed cloud service, works with Python 3.14)
#
# ChromaDB has been deprecated and removed due to Python 3.14 compatibility issues.

import os
import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from common.infrastructure.caching.embedding_processor import get_batch_processor
from common.infrastructure.caching.embedding_cache import get_embedding_cache
logger = logging.getLogger(__name__)

# Environment configuration
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "pinecone").lower()

# ChromaDB settings (DEPRECATED - no longer used)
# CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
# CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", "8000"))

# Pinecone Settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", None)
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-quickstart-test")

# Collection names for different RAG types
COLLECTIONS = {
    "strategy": "strategy_documents",
    "standards": "standards_documents",
    "inventory": "inventory_documents",
    "general": "general_documents",
    "taxonomy": "taxonomy_documents"
}

# Text splitting configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ============================================================================
# BASE DOCUMENT STORE (Abstract)
# ============================================================================

class BaseDocumentStore:
    """Base class for all document stores."""
    
    def __init__(self):
        # gemini-embedding-001 produces 3072-dim vectors.
        # The Pinecone index must be created with dimension=3072 to match.
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def add_document(self, collection_type: str, content: str, 
                     metadata: Optional[Dict] = None, doc_id: Optional[str] = None) -> Dict:
        raise NotImplementedError
    
    def search(self, collection_type: str, query: str, top_k: int = 5,
               filter_metadata: Optional[Dict] = None) -> Dict:
        raise NotImplementedError
    
    def list_documents(self, collection_type: str, limit: int = 100) -> Dict:
        raise NotImplementedError
    
    def delete_document(self, collection_type: str, doc_id: str) -> Dict:
        raise NotImplementedError
    
    def get_collection_stats(self) -> Dict:
        raise NotImplementedError
    
    def clear_collection(self, collection_type: str) -> Dict:
        raise NotImplementedError

    def is_healthy(self) -> bool:
        """
        Check if the vector store is healthy and can return real results.

        [FIX #2] Used by callers to skip standards_rag calls when vector store
        is in mock mode (INIT_ERROR). Saves 15-30+ seconds per item by avoiding
        the full retrieval -> generation -> validation -> retry cycle.

        Returns:
            True if healthy (can return real results), False if in mock/fallback mode
        """
        raise NotImplementedError

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get detailed health status of the vector store.

        Returns:
            Dict with 'healthy' bool, 'reason' string, and any additional details
        """
        raise NotImplementedError


# ============================================================================
# PINECONE DOCUMENT STORE (Primary)
# ============================================================================

class PineconeDocumentStore(BaseDocumentStore):
    """
    Pinecone Vector Database Document Store.
    
    This is the PRIMARY vector store implementation.
    Works with Python 3.14 and provides managed cloud infrastructure.
    """
    
    def __init__(self, api_key: str = None, index_name: str = None):
        super().__init__()

        self.api_key = api_key or PINECONE_API_KEY
        self.index_name = index_name or PINECONE_INDEX_NAME

        # ROOT CAUSE FIX: Better diagnostics for Mock mode activation
        self._mock_reason = None

        if not self.api_key:
            logger.warning(
                "[VS-ROOT-CAUSE] PINECONE_API_KEY environment variable not set. "
                "Vector store falling back to Mock Document Store (returns 0 results). "
                "ACTION: Set PINECONE_API_KEY environment variable to enable real document retrieval."
            )
            self._use_mock = True
            self._mock_reason = "MISSING_API_KEY"
        else:
            try:
                from pinecone import Pinecone

                self.pc = Pinecone(api_key=self.api_key)
                self.index = self.pc.Index(self.index_name)
                self._use_mock = False

                logger.info(f"[VS] PineconeDocumentStore initialized successfully with index: {self.index_name}")

            except ImportError as e:
                logger.error(f"[VS-ROOT-CAUSE] Failed to import Pinecone package: {e}")
                self._use_mock = True
                self._mock_reason = "IMPORT_ERROR"
            except Exception as e:
                logger.error(
                    f"[VS-ROOT-CAUSE] Failed to initialize Pinecone: {e} "
                    f"(Index: {self.index_name}). Falling back to Mock Document Store. "
                    f"ACTION: Verify PINECONE_API_KEY is valid and index '{self.index_name}' exists."
                )
                self._use_mock = True
                self._mock_reason = f"INIT_ERROR: {type(e).__name__}"

    def _get_namespace(self, collection_type: str) -> str:
        """Map collection type to Pinecone namespace."""
        return COLLECTIONS.get(collection_type, COLLECTIONS["general"])
    
    def add_document(self, collection_type: str, content: str,
                     metadata: Optional[Dict] = None, doc_id: Optional[str] = None) -> Dict:
        """Add a document to Pinecone (or Mock)."""
        if getattr(self, '_use_mock', False):
            logger.warning("[VS] Mock add_document called (no persistence)")
            return {"success": True, "doc_id": doc_id or "mock_id", "message": "Mock added"}

        try:
            doc_id = doc_id or str(uuid.uuid4())
            metadata = metadata or {}
            metadata.update({
                "doc_id": doc_id,
                "collection_type": collection_type,
                "ingested_at": datetime.now().isoformat(),
                "content_length": len(content)
            })
            
            chunks = self.text_splitter.split_text(content)
            namespace = self._get_namespace(collection_type)

            # Batch embed all chunks for better performance
            processor = get_batch_processor(self.embeddings)
            chunk_embeddings = processor.embed_documents_batch(chunks)

            vectors = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                chunk_metadata["text"] = chunk

                embedding = chunk_embeddings[i]

                vectors.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": chunk_metadata
                })
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
            
            logger.info(f"Added document {doc_id} with {len(chunks)} chunks to {collection_type}")
            
            return {
                "success": True,
                "doc_id": doc_id,
                "collection": collection_type,
                "chunk_count": len(chunks),
                "doc_ids": [v["id"] for v in vectors],
                "message": f"Document added with {len(chunks)} chunks"
            }
        except Exception as e:
            logger.error(f"Failed to add document to Pinecone: {e}")
            return {"success": False, "error": str(e)}
    
    def search(self, collection_type: str, query: str, top_k: int = 5,
               filter_metadata: Optional[Dict] = None) -> Dict:
        """Search for documents in Pinecone with embedding caching."""
        if getattr(self, '_use_mock', False):
            # ROOT CAUSE FIX: Include diagnostic information in mock search result
            mock_reason = getattr(self, '_mock_reason', 'UNKNOWN')
            logger.warning(
                f"[VS-MOCK-SEARCH] Search returning 0 results because: {mock_reason}. "
                f"Query: '{query[:50]}...' Collection: {collection_type}"
            )
            return {
                "success": True,
                "results": [],
                "result_count": 0,
                "collection": collection_type,
                "mock_mode": True,
                "mock_reason": mock_reason
            }

        try:
            namespace = self._get_namespace(collection_type)

            # Check cache first, then compute if needed
            cache = get_embedding_cache()
            query_embedding = cache.get(query)
            if query_embedding is None:
                query_embedding = self.embeddings.embed_query(query)
                cache.put(query, query_embedding)
            
            results = self.index.query(
                namespace=namespace,
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_metadata
            )
            
            formatted_results = []
            for match in results.get("matches", []):
                formatted_results.append({
                    "id": match["id"],
                    "content": match.get("metadata", {}).get("text", ""),
                    "metadata": match.get("metadata", {}),
                    "relevance_score": match.get("score", 0)
                })
            
            return {
                "success": True,
                "results": formatted_results,
                "result_count": len(formatted_results),
                "collection": collection_type
            }
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return {"success": False, "results": [], "result_count": 0, "error": str(e)}
    
    def search_all_collections(self, query: str, top_k_per_collection: int = 3) -> Dict:
        """Search across all collections in parallel for better performance."""
        all_results = []

        def search_single_collection(collection_type: str) -> tuple:
            """Helper to search a single collection and return results with type."""
            try:
                result = self.search(collection_type, query, top_k=top_k_per_collection)
                if result.get("results"):
                    for r in result["results"]:
                        r["collection_type"] = collection_type
                    return collection_type, result["results"]
            except Exception as e:
                logger.warning(f"Search in {collection_type} failed: {e}")
            return collection_type, []

        # Execute searches in parallel (4 collections = 4 threads)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(search_single_collection, coll_type): coll_type
                for coll_type in COLLECTIONS.keys()
            }

            for future in as_completed(futures):
                try:
                    _, results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    coll_type = futures[future]
                    logger.warning(f"Parallel search failed for {coll_type}: {e}")

        all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return {"success": True, "results": all_results, "result_count": len(all_results)}
    
    def list_documents(self, collection_type: str, limit: int = 100) -> Dict:
        """List documents - limited support in Pinecone."""
        return {
            "success": True, 
            "documents": [], 
            "count": 0, 
            "message": "Pinecone doesn't support listing all documents. Use search instead."
        }
    
    def delete_document(self, collection_type: str, doc_id: str) -> Dict:
        """Delete a document from Pinecone."""
        if getattr(self, '_use_mock', False):
            return {"success": True, "message": f"Mock deleted {doc_id}"}
            
        try:
            namespace = self._get_namespace(collection_type)
            # Delete all chunks with this doc_id
            self.index.delete(filter={"doc_id": doc_id}, namespace=namespace)
            return {"success": True, "message": f"Deleted document {doc_id}"}
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_collection_stats(self) -> Dict:
        """Get Pinecone index statistics."""
        if getattr(self, '_use_mock', False):
            return {"success": True, "backend": "mock", "collections": {}}

        try:
            stats = self.index.describe_index_stats()
            
            collections = {}
            for ns, data in (stats.namespaces or {}).items():
                # Reverse lookup collection type from namespace
                coll_type = next((k for k, v in COLLECTIONS.items() if v == ns), ns)
                collections[coll_type] = {
                    "document_count": data.vector_count,
                    "status": "active" if data.vector_count > 0 else "empty"
                }
            
            return {
                "success": True,
                "backend": "pinecone",
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "collections": collections
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"success": False, "error": str(e)}
    
    def clear_collection(self, collection_type: str) -> Dict:
        """Clear all vectors in a namespace."""
        if getattr(self, '_use_mock', False):
            return {"success": True, "message": "Mock cleared"}

        try:
            namespace = self._get_namespace(collection_type)
            self.index.delete(delete_all=True, namespace=namespace)
            return {"success": True, "message": f"Cleared collection {collection_type}"}
        except Exception as e:
            logger.error(f"Clear failed: {e}")
            return {"success": False, "error": str(e)}

    def is_healthy(self) -> bool:
        """
        Check if the vector store is healthy and can return real results.

        [FIX #2] Used by callers to skip standards_rag calls when vector store
        is in mock mode (INIT_ERROR). Saves 15-30+ seconds per item by avoiding
        the full retrieval -> generation -> validation -> retry cycle.

        Returns:
            True if healthy (can return real results), False if in mock/fallback mode
        """
        return not getattr(self, '_use_mock', False)

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get detailed health status of the vector store.

        [FIX #2] Provides diagnostic info for why the store is unhealthy.

        Returns:
            Dict with 'healthy' bool, 'reason' string, and additional details
        """
        is_mock = getattr(self, '_use_mock', False)
        mock_reason = getattr(self, '_mock_reason', None)

        if not is_mock:
            return {
                'healthy': True,
                'backend': 'pinecone',
                'index_name': self.index_name,
                'reason': 'Connected to Pinecone successfully'
            }
        else:
            return {
                'healthy': False,
                'backend': 'mock',
                'reason': mock_reason or 'Unknown initialization error',
                'action_required': 'Check PINECONE_API_KEY environment variable and Pinecone index availability'
            }


# ============================================================================
# SINGLETON INSTANCE AND FACTORY
# ============================================================================

_store_instance: Optional[BaseDocumentStore] = None


def get_vector_store() -> BaseDocumentStore:
    """
    Get the singleton vector store instance.
    
    Uses Pinecone as the only supported backend.
    ChromaDB support has been removed.
    """
    global _store_instance
    
    if _store_instance is None:
        store_type = VECTOR_STORE_TYPE
        
        # Warn if trying to use deprecated ChromaDB
        if store_type in ("chromadb", "chromadb_http"):
            logger.warning(
                f"VECTOR_STORE_TYPE='{store_type}' is deprecated. "
                "ChromaDB support has been removed. Using Pinecone instead."
            )
        
        # Always use Pinecone
        logger.info("Initializing Pinecone document store")
        try:
            _store_instance = PineconeDocumentStore()
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            # Don't raise, let it use the mock fallback inside PineconeDocumentStore
            _store_instance = PineconeDocumentStore()
    
    return _store_instance


def create_vector_store(store_type: str = "pinecone", **kwargs) -> BaseDocumentStore:
    """
    Create a new vector store instance (not singleton).
    
    Only Pinecone is supported. ChromaDB has been deprecated.
    """
    store_type = store_type.lower()
    
    if store_type in ("chromadb", "chromadb_http"):
        logger.warning(
            f"ChromaDB (store_type='{store_type}') is deprecated and no longer supported. "
            "Using Pinecone instead."
        )
    
    return PineconeDocumentStore(**kwargs)


# Backward compatibility aliases (DEPRECATED)
ChromaDocumentStore = PineconeDocumentStore  # ChromaDB deprecated, use Pinecone
get_chroma_store = get_vector_store  # DEPRECATED: Use get_vector_store
create_chroma_store = create_vector_store  # DEPRECATED: Use create_vector_store
