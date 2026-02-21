"""
Incremental Indexer for Strategy RAG
=====================================
Implements incremental indexing to avoid full re-index on every update.

PROBLEM: Current implementation (strategy_vector_store_pinecone.py) performs
         full re-index when CSV changes, which is slow and expensive.

SOLUTION: Track document versions and only index new/changed documents.

Strategy:
1. Store document hash in Pinecone metadata
2. Compare CSV row hashes with existing index
3. Only upsert changed/new documents
4. Optionally delete removed documents

Usage:
    from common.strategy_rag.incremental_indexer import IncrementalStrategyIndexer
    
    indexer = IncrementalStrategyIndexer()
    result = indexer.index_csv_incremental(csv_path)
    print(f"Added: {result['added']}, Updated: {result['updated']}, Deleted: {result['deleted']}")
"""

import os
import csv
import logging
import hashlib
from typing import Dict, Any, List, Tuple, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)

# Pinecone imports
try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("[IncrementalIndexer] Pinecone not installed")

# Gemini embeddings
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Fallback to sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class IncrementalStrategyIndexer:
    """
    Incremental indexer for Strategy RAG that only updates changed documents.
    
    Features:
    - Content-based hashing to detect changes
    - Batch upsert for efficiency
    - Optional garbage collection of deleted entries
    - Progress tracking and statistics
    """
    
    def __init__(self, 
                 api_key: str = None, 
                 index_name: str = None,
                 namespace: str = "strategy",
                 batch_size: int = 100):
        """
        Initialize incremental indexer.
        
        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            index_name: Pinecone index name
            namespace: Namespace within index
            batch_size: Number of documents to upsert per batch
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "agentic-quickstart-test")
        self.namespace = namespace
        self.batch_size = batch_size
        
        self._index = None
        self._embedding_model = None
        self._use_gemini = False
        
        # Initialize
        self._init_pinecone()
        self._init_embeddings()
    
    def _init_pinecone(self):
        """Initialize Pinecone connection"""
        if not PINECONE_AVAILABLE:
            raise RuntimeError("Pinecone not installed. Run: pip install pinecone")
        
        if not self.api_key:
            raise RuntimeError("PINECONE_API_KEY not set")
        
        try:
            pc = Pinecone(api_key=self.api_key)
            self._index = pc.Index(self.index_name)
            logger.info(f"[IncrementalIndexer] Connected to index: {self.index_name}")
        except Exception as e:
            logger.error(f"[IncrementalIndexer] Failed to connect to Pinecone: {e}")
            raise
    
    def _init_embeddings(self):
        """Initialize embedding model (Gemini or SentenceTransformers)"""
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if GEMINI_AVAILABLE and gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self._use_gemini = True
                logger.info("[IncrementalIndexer] Using Gemini embeddings")
                return
            except Exception as e:
                logger.warning(f"[IncrementalIndexer] Gemini init failed: {e}")
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("[IncrementalIndexer] Using SentenceTransformers")
            except Exception as e:
                logger.error(f"[IncrementalIndexer] SentenceTransformers init failed: {e}")
                raise
        else:
            raise RuntimeError("No embedding model available (need Gemini or SentenceTransformers)")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if self._use_gemini:
            try:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )
                return result['embedding']
            except Exception as e:
                logger.error(f"[IncrementalIndexer] Gemini embedding failed: {e}")
                raise
        else:
            return self._embedding_model.encode(text).tolist()
    
    def _compute_document_hash(self, row: Dict[str, str]) -> str:
        """
        Compute content hash for a CSV row to detect changes.
        
        Args:
            row: Dictionary from CSV DictReader
        
        Returns:
            SHA256 hash of document content
        """
        # Create deterministic string representation
        content_parts = [
            row.get('category', ''),
            row.get('subcategory', ''),
            row.get('vendor', ''),
            row.get('typical_product_types', ''),
            row.get('key_terms', '')
        ]
        content = '|'.join(content_parts)
        
        # Compute hash
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _create_document_text(self, row: Dict[str, str]) -> str:
        """Create searchable text from CSV row (same as original)"""
        parts = []
        
        if row.get('category'):
            parts.append(f"Category: {row['category']}")
        
        if row.get('subcategory'):
            parts.append(f"Subcategory: {row['subcategory']}")
        
        if row.get('vendor'):
            parts.append(f"Vendor: {row['vendor']}")
        
        if row.get('typical_product_types'):
            parts.append(f"Products: {row['typical_product_types']}")
        
        if row.get('key_terms'):
            parts.append(f"Terms: {row['key_terms']}")
        
        return " | ".join(parts)
    
    def _fetch_existing_hashes(self) -> Dict[str, str]:
        """
        Fetch all existing document hashes from Pinecone.
        
        Returns:
            Dict mapping document_id -> content_hash
        """
        existing_hashes = {}
        
        try:
            # Fetch all vectors in namespace
            # Note: Pinecone doesn't have a "list all" API, so we use a dummy query
            # with a high top_k to retrieve all documents
            stats = self._index.describe_index_stats()
            namespace_stats = stats.namespaces.get(self.namespace, None)
            
            if not namespace_stats or namespace_stats.vector_count == 0:
                logger.info("[IncrementalIndexer] No existing documents in namespace")
                return existing_hashes
            
            vector_count = namespace_stats.vector_count
            logger.info(f"[IncrementalIndexer] Fetching {vector_count} existing document hashes...")
            
            # Fetch in batches using query with dummy vector
            # (Pinecone limitation: no direct "list all" API)
            dummy_embedding = [0.0] * 768  # Adjust dimension as needed
            
            # Query to get all IDs (we'll fetch metadata separately)
            results = self._index.query(
                vector=dummy_embedding,
                top_k=min(vector_count, 10000),  # Max Pinecone allows
                include_metadata=True,
                namespace=self.namespace
            )
            
            for match in results.matches:
                doc_id = match.id
                metadata = match.metadata or {}
                content_hash = metadata.get('content_hash', '')
                existing_hashes[doc_id] = content_hash
            
            logger.info(f"[IncrementalIndexer] Fetched {len(existing_hashes)} document hashes")
            
        except Exception as e:
            logger.error(f"[IncrementalIndexer] Failed to fetch existing hashes: {e}")
        
        return existing_hashes
    
    def index_csv_incremental(self, 
                              csv_path: str,
                              delete_removed: bool = False) -> Dict[str, Any]:
        """
        Incrementally index CSV file, only updating changed documents.
        
        Args:
            csv_path: Path to CSV file
            delete_removed: If True, delete documents that no longer exist in CSV
        
        Returns:
            Dict with statistics:
            {
                'success': bool,
                'added': int,
                'updated': int,
                'deleted': int,
                'unchanged': int,
                'total_indexed': int
            }
        """
        if not os.path.exists(csv_path):
            logger.error(f"[IncrementalIndexer] CSV not found: {csv_path}")
            return {"success": False, "error": f"CSV not found: {csv_path}"}
        
        logger.info(f"[IncrementalIndexer] Starting incremental index: {csv_path}")
        
        # Fetch existing documents
        existing_hashes = self._fetch_existing_hashes()
        
        # Track changes
        stats = {
            'added': 0,
            'updated': 0,
            'deleted': 0,
            'unchanged': 0,
            'total_indexed': 0
        }
        
        # Read CSV and detect changes
        csv_doc_ids = set()
        vectors_to_upsert = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for idx, row in enumerate(reader):
                doc_id = f"strategy_{idx}"
                csv_doc_ids.add(doc_id)
                
                # Compute hash
                content_hash = self._compute_document_hash(row)
                
                # Check if changed
                existing_hash = existing_hashes.get(doc_id)
                
                if existing_hash == content_hash:
                    # Unchanged
                    stats['unchanged'] += 1
                    continue
                
                # Document is new or changed
                is_new = existing_hash is None
                
                if is_new:
                    stats['added'] += 1
                    logger.debug(f"[IncrementalIndexer] NEW: {doc_id}")
                else:
                    stats['updated'] += 1
                    logger.debug(f"[IncrementalIndexer] UPDATED: {doc_id}")
                
                # Generate embedding
                doc_text = self._create_document_text(row)
                embedding = self._generate_embedding(doc_text)
                
                # Prepare metadata
                metadata = {
                    "category": row.get('category', ''),
                    "subcategory": row.get('subcategory', ''),
                    "vendor": row.get('vendor', ''),
                    "typical_product_types": row.get('typical_product_types', ''),
                    "key_terms": row.get('key_terms', ''),
                    "content_hash": content_hash,
                    "indexed_at": datetime.now().isoformat()
                }
                
                # Add to upsert batch
                vectors_to_upsert.append({
                    "id": doc_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
                # Upsert in batches
                if len(vectors_to_upsert) >= self.batch_size:
                    self._upsert_batch(vectors_to_upsert)
                    vectors_to_upsert = []
        
        # Upsert remaining
        if vectors_to_upsert:
            self._upsert_batch(vectors_to_upsert)
        
        # Delete removed documents
        if delete_removed:
            removed_ids = set(existing_hashes.keys()) - csv_doc_ids
            if removed_ids:
                logger.info(f"[IncrementalIndexer] Deleting {len(removed_ids)} removed documents")
                self._delete_batch(list(removed_ids))
                stats['deleted'] = len(removed_ids)
        
        stats['total_indexed'] = stats['added'] + stats['updated']
        stats['success'] = True
        
        logger.info(
            f"[IncrementalIndexer] Complete: "
            f"Added={stats['added']}, Updated={stats['updated']}, "
            f"Deleted={stats['deleted']}, Unchanged={stats['unchanged']}"
        )
        
        return stats
    
    def _upsert_batch(self, vectors: List[Dict]):
        """Upsert a batch of vectors"""
        try:
            self._index.upsert(vectors=vectors, namespace=self.namespace)
            logger.debug(f"[IncrementalIndexer] Upserted batch of {len(vectors)}")
        except Exception as e:
            logger.error(f"[IncrementalIndexer] Batch upsert failed: {e}")
            raise
    
    def _delete_batch(self, ids: List[str]):
        """Delete a batch of documents"""
        try:
            self._index.delete(ids=ids, namespace=self.namespace)
            logger.debug(f"[IncrementalIndexer] Deleted batch of {len(ids)}")
        except Exception as e:
            logger.error(f"[IncrementalIndexer] Batch delete failed: {e}")
            raise


# Convenience functions
def index_strategy_csv_incremental(csv_path: str, delete_removed: bool = False) -> Dict[str, Any]:
    """
    Convenience function for incremental indexing.
    
    Args:
        csv_path: Path to CSV file
        delete_removed: If True, delete documents no longer in CSV
    
    Returns:
        Statistics dict
    """
    indexer = IncrementalStrategyIndexer()
    return indexer.index_csv_incremental(csv_path, delete_removed=delete_removed)
