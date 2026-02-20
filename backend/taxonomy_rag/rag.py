import json
import logging
import os
from typing import Dict, Any, List, Optional

from common.rag.vector_store import get_vector_store
from common.services.llm.fallback import create_llm_with_fallback
from common.config import AgenticConfig

logger = logging.getLogger(__name__)

class TaxonomyRAG:
    """
    RAG Service for Taxonomy.
    Handles indexing and retrieval of taxonomy terms.
    """

    def __init__(self):
        self.vector_store = get_vector_store()
        self.collection_type = "taxonomy"

    def index_taxonomy(self, taxonomy_data: Dict[str, Any]) -> None:
        """
        Index the taxonomy into the vector store.
        Skips if already populated (checked via collection stats).
        """
        try:
            stats = self.vector_store.get_collection_stats()
            if stats.get("success"):
                cols = stats.get("collections", {})
                if self.collection_type in cols and cols[self.collection_type].get("document_count", 0) > 0:
                    logger.info("[TaxonomyRAG] Index already populated. Skipping ingestion.")
                    return

            logger.info("[TaxonomyRAG] Indexing taxonomy...")

            for item in taxonomy_data.get("instruments", []):
                doc_content = f"Instrument: {item['name']}\n"
                doc_content += f"Category: {item.get('category', 'General')}\n"
                doc_content += f"Definition: {item.get('definition', '')}\n"
                if item.get("aliases"):
                    doc_content += f"Aliases: {', '.join(item['aliases'])}\n"

                metadata = {
                    "type": "instrument",
                    "name": item['name'],
                    "aliases": item.get('aliases', []),
                    "category": item.get('category')
                }

                self.vector_store.add_document(
                    collection_type=self.collection_type,
                    content=doc_content,
                    metadata=metadata,
                    doc_id=f"tax_inst_{item['name'].replace(' ', '_')}"
                )

            for item in taxonomy_data.get("accessories", []):
                doc_content = f"Accessory: {item['name']}\n"
                doc_content += f"Category: {item.get('category', 'General')}\n"
                if item.get("related_instruments"):
                    doc_content += f"Related Instruments: {', '.join(item['related_instruments'])}\n"
                if item.get("aliases"):
                    doc_content += f"Aliases: {', '.join(item['aliases'])}\n"

                metadata = {
                    "type": "accessory",
                    "name": item['name'],
                    "aliases": item.get('aliases', []),
                    "related_instruments": item.get('related_instruments', [])
                }

                self.vector_store.add_document(
                    collection_type=self.collection_type,
                    content=doc_content,
                    metadata=metadata,
                    doc_id=f"tax_acc_{item['name'].replace(' ', '_')}"
                )

            logger.info(
                f"[TaxonomyRAG] Successfully indexed {len(taxonomy_data.get('instruments', []))} instruments and {len(taxonomy_data.get('accessories', []))} accessories."
            )

        except Exception as e:
            logger.error(f"[TaxonomyRAG] Failed to index taxonomy: {e}")

    def retrieve(self, query: str, top_k: int = 5, item_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant taxonomy terms for a query."""
        try:
            filters = {}
            if item_type:
                filters = {"type": item_type}

            search_result = self.vector_store.search(
                collection_type=self.collection_type,
                query=query,
                top_k=top_k,
                filter_metadata=filters if filters else None
            )

            if not search_result.get("success"):
                logger.warning(f"[TaxonomyRAG] Search failed: {search_result.get('error')}")
                return []

            results = []
            for item in search_result.get("results", []):
                meta = item.get("metadata", {})
                results.append({
                    "name": meta.get("name"),
                    "aliases": meta.get("aliases", []),
                    "type": meta.get("type"),
                    "score": item.get("relevance_score"),
                    "content": item.get("content")
                })

            return results

        except Exception as e:
            logger.error(f"[TaxonomyRAG] Retrieval failed: {e}")
            return []


_taxonomy_rag_instance = None


def get_taxonomy_rag() -> TaxonomyRAG:
    global _taxonomy_rag_instance
    if _taxonomy_rag_instance is None:
        _taxonomy_rag_instance = TaxonomyRAG()
    return _taxonomy_rag_instance
