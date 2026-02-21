# agentic/strategy_rag/strategy_vector_store_pinecone.py
# =============================================================================
# STRATEGY VECTOR STORE - Pinecone-based semantic search for strategy data
# =============================================================================
#
# PURPOSE: Replace ChromaDB with Pinecone for production-ready semantic matching
# using vector embeddings for flexible product-to-strategy mapping.
#
# MIGRATION: Replaces strategy_vector_store.py ChromaDB implementation
#
# =============================================================================

import os
import csv
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

# Path to strategy CSV
STRATEGY_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "instrumentation_procurement_strategy.csv"
)

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", None)
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-quickstart-test")
STRATEGY_NAMESPACE = "strategy_documents"

# Try to import Pinecone
PINECONE_AVAILABLE = False
Pinecone = None

try:
    from pinecone import Pinecone as _Pinecone
    Pinecone = _Pinecone
    PINECONE_AVAILABLE = True
    logger.info("[StrategyVectorStore] Pinecone SDK available")
except ImportError:
    logger.warning("[StrategyVectorStore] pinecone-client not installed. Install with: pip install pinecone-client")
except Exception as e:
    logger.warning(f"[StrategyVectorStore] Pinecone unavailable: {type(e).__name__}: {e}")

# Try to import embedding models
GOOGLE_EMBEDDINGS_AVAILABLE = False
GoogleGenerativeAIEmbeddings = None

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings as _GoogleEmbeddings
    GoogleGenerativeAIEmbeddings = _GoogleEmbeddings
    GOOGLE_EMBEDDINGS_AVAILABLE = True
except ImportError:
    logger.warning("[StrategyVectorStore] Google embeddings not available")

# Fallback to sentence-transformers
SENTENCE_TRANSFORMERS_AVAILABLE = False
SentenceTransformer = None

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    SentenceTransformer = _SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("[StrategyVectorStore] sentence-transformers not installed. Using fallback.")
except Exception as e:
    logger.warning(f"[StrategyVectorStore] sentence-transformers unavailable: {type(e).__name__}")


# =============================================================================
# PINECONE VECTOR STORE CLASS
# =============================================================================

class StrategyVectorStore:
    """
    Pinecone-based vector store for strategy data.
    
    Features:
    - Index CSV rows with embeddings into Pinecone
    - Semantic search by product type / vendor / category
    - Similarity-based ranking
    - Production-ready with cloud storage
    - Auto-fallback to mock mode if Pinecone unavailable
    """
    
    EMBEDDING_MODEL_GOOGLE = "models/gemini-embedding-001"
    EMBEDDING_MODEL_LOCAL = "all-MiniLM-L6-v2"  # Fast local model fallback
    
    def __init__(self, api_key: str = None, index_name: str = None, namespace: str = None):
        """
        Initialize the Pinecone vector store.
        
        Args:
            api_key: Pinecone API key (defaults to env var)
            index_name: Pinecone index name (defaults to env var)
            namespace: Namespace for strategy data (defaults to STRATEGY_NAMESPACE)
        """
        self.api_key = api_key or PINECONE_API_KEY
        self.index_name = index_name or PINECONE_INDEX_NAME
        self.namespace = namespace or STRATEGY_NAMESPACE
        
        self._pc = None
        self._index = None
        self._embedding_model = None
        self._is_indexed = False
        self._index_count = 0
        self._use_mock = False
        self._mock_reason = None
        
        # Initialize Pinecone
        self._initialize_pinecone()
        
        # Initialize embeddings
        self._initialize_embeddings()
        
        logger.info(f"[StrategyVectorStore] Initialized (Pinecone: {PINECONE_AVAILABLE}, Mock: {self._use_mock})")
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and index."""
        if not self.api_key:
            logger.warning("[StrategyVectorStore] PINECONE_API_KEY not set. Using mock mode.")
            self._use_mock = True
            self._mock_reason = "MISSING_API_KEY"
            return
        
        if not PINECONE_AVAILABLE:
            logger.warning("[StrategyVectorStore] Pinecone SDK not available. Using mock mode.")
            self._use_mock = True
            self._mock_reason = "SDK_NOT_AVAILABLE"
            return
        
        try:
            self._pc = Pinecone(api_key=self.api_key)
            self._index = self._pc.Index(self.index_name)
            logger.info(f"[StrategyVectorStore] Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"[StrategyVectorStore] Failed to initialize Pinecone: {e}")
            self._use_mock = True
            self._mock_reason = f"INIT_ERROR: {type(e).__name__}"
    
    def _initialize_embeddings(self):
        """Initialize embedding model (Google Gemini preferred, SentenceTransformers fallback)."""
        # Try Google Gemini first
        if GOOGLE_EMBEDDINGS_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
            try:
                self._embedding_model = GoogleGenerativeAIEmbeddings(
                    model=self.EMBEDDING_MODEL_GOOGLE,
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
                self._embedding_type = "google"
                logger.info(f"[StrategyVectorStore] Using Google Gemini embeddings: {self.EMBEDDING_MODEL_GOOGLE}")
                return
            except Exception as e:
                logger.warning(f"[StrategyVectorStore] Google embeddings failed: {e}")
        
        # Fallback to local SentenceTransformers
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._embedding_model = SentenceTransformer(self.EMBEDDING_MODEL_LOCAL)
                self._embedding_type = "local"
                logger.info(f"[StrategyVectorStore] Using local embeddings: {self.EMBEDDING_MODEL_LOCAL}")
                return
            except Exception as e:
                logger.warning(f"[StrategyVectorStore] SentenceTransformers failed: {e}")
        
        # Ultimate fallback to simple hash-based embeddings
        logger.warning("[StrategyVectorStore] No embedding model available, using hash-based fallback")
        self._embedding_type = "hash"
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if self._embedding_type == "google":
            return self._embedding_model.embed_query(text)
        elif self._embedding_type == "local":
            return self._embedding_model.encode(text).tolist()
        else:
            # Hash-based fallback (384 dimensions to match all-MiniLM-L6-v2)
            import hashlib
            hash_val = int(hashlib.md5(text.lower().encode()).hexdigest(), 16)
            return [(hash_val >> (i * 8)) % 256 / 255.0 for i in range(384)]
    
    def _create_document_text(self, row: Dict[str, str]) -> str:
        """Create searchable text from CSV row."""
        parts = []
        
        # Category and subcategory
        if row.get('category'):
            parts.append(f"Category: {row['category']}")
        if row.get('subcategory'):
            parts.append(f"Subcategory: {row['subcategory']}")
        
        # Vendor
        if row.get('vendor'):
            parts.append(f"Vendor: {row['vendor']}")
        
        # Product types
        if row.get('typical_product_types'):
            parts.append(f"Products: {row['typical_product_types']}")
        
        # Key terms
        if row.get('key_terms'):
            parts.append(f"Terms: {row['key_terms']}")
        
        return " | ".join(parts)
    
    def index_csv(self, csv_path: str = None, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Index strategy CSV into Pinecone.
        
        Args:
            csv_path: Path to CSV file (defaults to standard location)
            force_reindex: If True, delete existing data and reindex
        
        Returns:
            Dict with success status and stats
        """
        if self._use_mock:
            logger.warning(f"[StrategyVectorStore] Mock mode ({self._mock_reason}), skipping indexing")
            return {"success": False, "error": f"Mock mode: {self._mock_reason}"}
        
        csv_path = csv_path or STRATEGY_CSV_PATH
        
        if not os.path.exists(csv_path):
            logger.error(f"[StrategyVectorStore] CSV not found: {csv_path}")
            return {"success": False, "error": f"CSV not found: {csv_path}"}
        
        # Check if already indexed
        try:
            stats = self._index.describe_index_stats()
            namespace_stats = stats.namespaces.get(self.namespace, None)
            if namespace_stats and namespace_stats.vector_count > 0 and not force_reindex:
                self._index_count = namespace_stats.vector_count
                self._is_indexed = True
                logger.info(f"[StrategyVectorStore] Already indexed ({self._index_count} docs)")
                return {"success": True, "already_indexed": True, "count": self._index_count}
        except Exception as e:
            logger.warning(f"[StrategyVectorStore] Could not check index stats: {e}")
        
        # Force reindex: delete existing namespace data
        if force_reindex:
            try:
                self._index.delete(delete_all=True, namespace=self.namespace)
                logger.info("[StrategyVectorStore] Cleared namespace for reindex")
            except Exception as e:
                logger.warning(f"[StrategyVectorStore] Could not clear namespace: {e}")
        
        logger.info(f"[StrategyVectorStore] Indexing CSV: {csv_path}")
        
        # Read CSV
        documents = []
        metadatas = []
        ids = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                doc_text = self._create_document_text(row)
                doc_id = f"strategy_{idx}"
                
                documents.append(doc_text)
                metadatas.append({
                    "category": row.get('category', ''),
                    "subcategory": row.get('subcategory', ''),
                    "vendor": row.get('vendor', ''),
                    "typical_product_types": row.get('typical_product_types', ''),
                    "key_terms": row.get('key_terms', ''),
                    "indexed_at": datetime.now().isoformat()
                })
                ids.append(doc_id)
        
        if not documents:
            logger.warning(f"[StrategyVectorStore] No documents found in CSV")
            return {"success": False, "error": "No documents in CSV"}
        
        # Generate embeddings in batches
        logger.info(f"[StrategyVectorStore] Generating embeddings for {len(documents)} documents...")
        embeddings = []
        
        if self._embedding_type == "google":
            # Google embeddings API doesn't support batch, so we do it sequentially
            # (Could be optimized with concurrent futures)
            for doc in documents:
                embeddings.append(self._generate_embedding(doc))
        elif self._embedding_type == "local":
            # SentenceTransformers supports batch encoding
            embeddings = [emb.tolist() for emb in self._embedding_model.encode(documents)]
        else:
            # Hash fallback
            embeddings = [self._generate_embedding(doc) for doc in documents]
        
        # Upsert to Pinecone in batches
        logger.info(f"[StrategyVectorStore] Upserting {len(documents)} vectors to Pinecone...")
        batch_size = 100
        vectors = []
        
        for i in range(len(documents)):
            vectors.append({
                "id": ids[i],
                "values": embeddings[i],
                "metadata": metadatas[i]
            })
        
        # Batch upsert
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            try:
                self._index.upsert(vectors=batch, namespace=self.namespace)
            except Exception as e:
                logger.error(f"[StrategyVectorStore] Batch upsert failed: {e}")
                return {"success": False, "error": f"Upsert failed: {e}"}
        
        self._index_count = len(documents)
        self._is_indexed = True
        
        logger.info(f"[StrategyVectorStore] Successfully indexed {self._index_count} documents")
        
        return {
            "success": True,
            "count": self._index_count,
            "csv_path": csv_path,
            "namespace": self.namespace
        }
    
    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        filter_category: str = None,
        filter_vendor: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar strategy entries using semantic search.
        
        Args:
            query: Search query (e.g., "chromatography equipment")
            top_k: Number of results to return
            filter_category: Optional category filter
            filter_vendor: Optional vendor filter
        
        Returns:
            List of matching documents with similarity scores
        """
        if self._use_mock:
            logger.warning(f"[StrategyVectorStore] Mock search returning empty results ({self._mock_reason})")
            return []
        
        # Ensure indexed
        if not self._is_indexed:
            self.index_csv()
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Build metadata filter
        filter_dict = {}
        if filter_category:
            filter_dict["category"] = {"$eq": filter_category}
        if filter_vendor:
            filter_dict["vendor"] = {"$eq": filter_vendor}
        
        # Query Pinecone
        try:
            results = self._index.query(
                namespace=self.namespace,
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
        except Exception as e:
            logger.error(f"[StrategyVectorStore] Search failed: {e}")
            return []
        
        # Format results
        formatted = []
        for match in results.get('matches', []):
            metadata = match.get('metadata', {})
            score = match.get('score', 0)
            
            formatted.append({
                "id": match['id'],
                "document": f"{metadata.get('category', '')} | {metadata.get('subcategory', '')} | {metadata.get('vendor', '')}",
                "metadata": metadata,
                "category": metadata.get('category', ''),
                "subcategory": metadata.get('subcategory', ''),
                "vendor": metadata.get('vendor', ''),
                "similarity": round(score, 4),
                "score": round(score, 4)
            })
        
        logger.debug(f"[StrategyVectorStore] Search '{query[:30]}...' returned {len(formatted)} results")
        return formatted
    
    def match_product_to_category(self, product_type: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Find best matching category for a product type using semantic search.
        
        Args:
            product_type: Product type string (e.g., "HPLC")
        
        Returns:
            Tuple of (category, subcategory) or (None, None) if no match
        """
        if not product_type:
            return None, None
        
        results = self.search_similar(product_type, top_k=1)
        
        if results and len(results) > 0:
            best_match = results[0]
            if best_match.get('similarity', 0) > 0.5:  # Confidence threshold
                return best_match.get('category'), best_match.get('subcategory')
        
        return None, None
    
    def get_vendor_similarity(self, vendor_name: str, product_type: str) -> float:
        """
        Calculate similarity score between vendor and product type.
        
        Args:
            vendor_name: Vendor name
            product_type: Product type
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not vendor_name or not product_type:
            return 0.0
        
        # Search for vendor + product combination
        query = f"{vendor_name} {product_type}"
        results = self.search_similar(query, top_k=1)
        
        if results and len(results) > 0:
            return results[0].get('similarity', 0.0)
        
        return 0.0
    
    def add_entry(self, category: str, subcategory: str = None, vendor: str = None,
                  typical_product_types: str = None, key_terms: str = None) -> Dict[str, Any]:
        """
        Add a new strategy entry to the vector store.
        
        Args:
            category: Main category
            subcategory: Subcategory (optional)
            vendor: Vendor name (optional)
            typical_product_types: Comma-separated product types (optional)
            key_terms: Comma-separated key terms (optional)
        
        Returns:
            Dict with success status
        """
        if self._use_mock:
            return {"success": False, "error": f"Mock mode: {self._mock_reason}"}
        
        # Create document
        row = {
            'category': category or '',
            'subcategory': subcategory or '',
            'vendor': vendor or '',
            'typical_product_types': typical_product_types or '',
            'key_terms': key_terms or ''
        }
        
        doc_text = self._create_document_text(row)
        doc_id = f"strategy_custom_{uuid.uuid4().hex[:8]}"
        
        # Generate embedding
        embedding = self._generate_embedding(doc_text)
        
        # Metadata
        metadata = {
            "category": category or '',
            "subcategory": subcategory or '',
            "vendor": vendor or '',
            "typical_product_types": typical_product_types or '',
            "key_terms": key_terms or '',
            "indexed_at": datetime.now().isoformat()
        }
        
        # Upsert to Pinecone
        try:
            self._index.upsert(
                vectors=[{"id": doc_id, "values": embedding, "metadata": metadata}],
                namespace=self.namespace
            )
        except Exception as e:
            logger.error(f"[StrategyVectorStore] Add entry failed: {e}")
            return {"success": False, "error": str(e)}
        
        logger.info(f"[StrategyVectorStore] Added new entry: {doc_id}")
        
        return {"success": True, "id": doc_id}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if self._use_mock:
            return {
                "success": False,
                "mock_mode": True,
                "mock_reason": self._mock_reason,
                "indexed": False,
                "count": 0
            }
        
        try:
            stats = self._index.describe_index_stats()
            namespace_stats = stats.namespaces.get(self.namespace, None)
            
            return {
                "success": True,
                "backend": "pinecone",
                "index_name": self.index_name,
                "namespace": self.namespace,
                "embedding_model": self.EMBEDDING_MODEL_GOOGLE if self._embedding_type == "google" else self.EMBEDDING_MODEL_LOCAL,
                "embedding_type": self._embedding_type,
                "indexed": namespace_stats is not None and namespace_stats.vector_count > 0,
                "count": namespace_stats.vector_count if namespace_stats else 0
            }
        except Exception as e:
            logger.error(f"[StrategyVectorStore] Get stats failed: {e}")
            return {"success": False, "error": str(e)}
    
    def is_healthy(self) -> bool:
        """Check if the vector store is healthy and operational."""
        return not self._use_mock


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_strategy_store_instance: Optional[StrategyVectorStore] = None


def get_strategy_vector_store() -> StrategyVectorStore:
    """Get or create singleton StrategyVectorStore instance."""
    global _strategy_store_instance
    
    if _strategy_store_instance is None:
        _strategy_store_instance = StrategyVectorStore()
        
        # Auto-index if needed
        try:
            _strategy_store_instance.index_csv()
        except Exception as e:
            logger.warning(f"[StrategyVectorStore] Auto-index failed: {e}")
    
    return _strategy_store_instance


# Convenience alias
strategy_store = None


def init_strategy_store():
    """Initialize the global strategy store."""
    global strategy_store
    strategy_store = get_strategy_vector_store()
    return strategy_store


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def search_strategy(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Convenience function for semantic strategy search."""
    return get_strategy_vector_store().search_similar(query, top_k)


def match_product_category(product_type: str) -> Tuple[Optional[str], Optional[str]]:
    """Convenience function for product-to-category matching."""
    return get_strategy_vector_store().match_product_to_category(product_type)


def get_vendor_score(vendor_name: str, product_type: str) -> float:
    """Convenience function for vendor similarity score."""
    return get_strategy_vector_store().get_vendor_similarity(vendor_name, product_type)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'StrategyVectorStore',
    'get_strategy_vector_store',
    'init_strategy_store',
    'search_strategy',
    'match_product_category',
    'get_vendor_score'
]
