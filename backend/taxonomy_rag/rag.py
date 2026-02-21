import json
import logging
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

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

    def get_top_files_by_similarity(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve top files by cosine similarity to use for specification extraction.
        Defaults to searching the 'documents' collection if it exists, otherwise searches whatever is available.
        """
        try:
            search_result = self.vector_store.search(
                collection_type="documents",  # Assuming files are indexed here
                query=query,
                top_k=top_k
            )

            if not search_result.get("success"):
                logger.warning(f"[TaxonomyRAG] File search failed: {search_result.get('error')}")
                return ""

            content = []
            for item in search_result.get("results", []):
                content.append(item.get("content", ""))

            return "\n\n---\n\n".join(content)
        except Exception as e:
            logger.error(f"[TaxonomyRAG] File retrieval failed: {e}")
            return ""

    def extract_specifications_from_files(self, product_name: str, files_content: str) -> dict:
        """
        Extract technical specifications from the downloaded file contents using an LLM.
        """
        if not files_content:
            return {}

        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import JsonOutputParser
            import json

            llm = create_llm_with_fallback(temperature=0.0)

            prompt_template = """
You are an expert technical data extractor parsing industrial documents. 
Given the following technical document contents, extract all relevant technical specifications, 
operating parameters, dimensions, materials, and constraints for the product: "{product_name}".

Return ONLY a valid JSON dictionary mapping specification names (e.g., "temperature_range", "material", "accuracy") to their extracted values. 
If you cannot find any specifications, return an empty dictionary {{}}. Do not hallucinate or make up data.

Document Contents:
{files_content}

Output JSON Format:
"""
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | llm | JsonOutputParser()

            # Trim files_content to avoid token limits if too large.
            # 20k chars is usually safe for Gemini Flash.
            if len(files_content) > 30000:
                files_content = files_content[:30000] + "... (truncated)"

            result = chain.invoke({
                "product_name": product_name,
                "files_content": files_content
            })

            if isinstance(result, dict):
                logger.info(f"[TaxonomyRAG] Extracted {len(result)} specs for {product_name} from files.")
                return result
            return {}
        except Exception as e:
            logger.error(f"[TaxonomyRAG] Specification extraction from files failed: {e}")
            return {}




_taxonomy_rag_instance = None


def get_taxonomy_rag() -> TaxonomyRAG:
    global _taxonomy_rag_instance
    if _taxonomy_rag_instance is None:
        _taxonomy_rag_instance = TaxonomyRAG()
    return _taxonomy_rag_instance

class SpecificationRetriever:
    """
    Retrieves specifications for normalized product types from MongoDB or JSON files.
    
    Supports two modes:
    1. MongoDB mode: Queries a MongoDB collection for product specifications
    2. JSON file mode: Loads specifications from JSON catalog files
    """
    
    def __init__(
        self, 
        mongodb_uri: Optional[str] = None,
        mongodb_database: str = None,
        mongodb_collection: str = None,
        json_catalog_path: Optional[str] = None
    ):
        """
        Initialize the specification retriever.
        
        Args:
            mongodb_uri: MongoDB connection string (optional, uses MONGODB_URI env var)
            mongodb_database: Database name (optional, uses MONGODB_DATABASE env var or "engenie")
            mongodb_collection: Collection name (optional, uses MONGODB_COLLECTION env var or "product_specifications")
            json_catalog_path: Path to JSON catalog file or directory (optional)
        """
        self.mongodb_uri = mongodb_uri or os.getenv("MONGODB_URI")
        self.mongodb_database = mongodb_database or os.getenv("MONGODB_DATABASE", "engenie")
        self.mongodb_collection = mongodb_collection or os.getenv("MONGODB_COLLECTION", "product_specifications")
        self.json_catalog_path = json_catalog_path
        
        self._mongo_client = None
        self._json_catalog = None
        
        # Determine which mode to use
        self.mode = self._determine_mode()
        
        logger.info(f"[SpecRetriever] Initialized in {self.mode} mode")
    
    def _determine_mode(self) -> str:
        """Determine whether to use MongoDB or JSON file mode."""
        if self.mongodb_uri:
            return "mongodb"
        elif self.json_catalog_path:
            return "json"
        else:
            # Default to JSON mode with default path
            default_path = os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "data", 
                "product_catalog.json"
            )
            self.json_catalog_path = default_path
            return "json"
    
    def _get_mongo_client(self):
        """Lazy initialization of MongoDB client."""
        if self._mongo_client is None and self.mongodb_uri:
            try:
                from pymongo import MongoClient
                self._mongo_client = MongoClient(self.mongodb_uri)
                # Test connection
                self._mongo_client.admin.command('ping')
                logger.info("[SpecRetriever] MongoDB connection established")
            except Exception as e:
                logger.error(f"[SpecRetriever] MongoDB connection failed: {e}")
                self._mongo_client = None
        
        return self._mongo_client
    
    def _load_json_catalog(self) -> Dict[str, Any]:
        """Load JSON catalog from file."""
        if self._json_catalog is not None:
            return self._json_catalog
        
        if not self.json_catalog_path:
            logger.warning("[SpecRetriever] No JSON catalog path specified")
            return {}
        
        try:
            catalog_path = Path(self.json_catalog_path)
            
            if not catalog_path.exists():
                logger.warning(f"[SpecRetriever] JSON catalog not found: {catalog_path}")
                return {}
            
            if catalog_path.is_file():
                # Single JSON file
                with open(catalog_path, 'r', encoding='utf-8-sig') as f:
                    self._json_catalog = json.load(f)
                logger.info(f"[SpecRetriever] Loaded catalog from {catalog_path}")
            
            elif catalog_path.is_dir():
                # Directory of JSON files - merge them
                self._json_catalog = {}
                for json_file in catalog_path.glob("*.json"):
                    with open(json_file, 'r', encoding='utf-8-sig') as f:
                        data = json.load(f)
                        self._json_catalog.update(data)
                logger.info(f"[SpecRetriever] Loaded catalog from directory {catalog_path}")
            
            return self._json_catalog or {}
            
        except Exception as e:
            logger.error(f"[SpecRetriever] Failed to load JSON catalog: {e}")
            return {}
    
    def get_specification(self, canonical_name: str) -> Optional[Dict[str, Any]]:
        """
        Get specification for a single product by canonical name.
        
        Args:
            canonical_name: Normalized product name (e.g., "Temperature Transmitter")
            
        Returns:
            Dictionary with product specifications or None if not found
        """
        if self.mode == "mongodb":
            return self._get_spec_from_mongodb(canonical_name)
        else:
            return self._get_spec_from_json(canonical_name)
    
    def _get_spec_from_mongodb(self, canonical_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve specification from MongoDB."""
        try:
            client = self._get_mongo_client()
            if not client:
                return None
            
            db = client[self.mongodb_database]
            collection = db[self.mongodb_collection]
            
            # Query by canonical name (case-insensitive)
            spec = collection.find_one(
                {"canonical_name": {"$regex": f"^{canonical_name}$", "$options": "i"}}
            )
            
            if spec:
                # Remove MongoDB _id field
                spec.pop('_id', None)
                logger.debug(f"[SpecRetriever] Found spec for '{canonical_name}' in MongoDB")
                return spec
            else:
                logger.debug(f"[SpecRetriever] No spec found for '{canonical_name}' in MongoDB")
                return None
                
        except Exception as e:
            logger.error(f"[SpecRetriever] MongoDB query failed for '{canonical_name}': {e}")
            return None
    
    def _get_spec_from_json(self, canonical_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve specification from JSON catalog."""
        catalog = self._load_json_catalog()
        
        if not catalog:
            return None
        
        # Try exact match first
        if canonical_name in catalog:
            logger.debug(f"[SpecRetriever] Found spec for '{canonical_name}' in JSON (exact)")
            return catalog[canonical_name]
        
        # Try case-insensitive match
        canonical_lower = canonical_name.lower()
        for key, value in catalog.items():
            if key.lower() == canonical_lower:
                logger.debug(f"[SpecRetriever] Found spec for '{canonical_name}' in JSON (case-insensitive)")
                return value
        
        logger.debug(f"[SpecRetriever] No spec found for '{canonical_name}' in JSON")
        return None
    
    def get_specifications_batch(
        self, 
        normalized_items: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve specifications for all normalized items in batch.
        
        Args:
            normalized_items: List of items with 'canonical_name' field
            
        Returns:
            Dictionary mapping item index/ID to specification data
        """
        logger.info(f"[SpecRetriever] Batch retrieval for {len(normalized_items)} items")
        
        results = {}
        found_count = 0
        
        for idx, item in enumerate(normalized_items):
            item_key = f"item_{idx}"
            canonical_name = item.get("canonical_name", "")
            original_name = item.get("product_name") or item.get("name") or canonical_name
            
            if not canonical_name:
                logger.warning(f"[SpecRetriever] Item {idx} has no canonical_name, skipping")
                results[item_key] = {
                    "canonical_name": "",
                    "original_name": original_name,
                    "specifications": {},
                    "spec_found": False,
                    "error": "No canonical name"
                }
                continue
            
            # Retrieve specification
            spec = self.get_specification(canonical_name)
            
            if spec:
                found_count += 1
                results[item_key] = {
                    "canonical_name": canonical_name,
                    "original_name": original_name,
                    "specifications": spec,
                    "spec_found": True,
                    "category": item.get("category", "unknown"),
                    "quantity": item.get("quantity", 1)
                }
            else:
                results[item_key] = {
                    "canonical_name": canonical_name,
                    "original_name": original_name,
                    "specifications": {},
                    "spec_found": False,
                    "category": item.get("category", "unknown"),
                    "quantity": item.get("quantity", 1)
                }
        
        logger.info(
            f"[SpecRetriever] Batch complete: {found_count}/{len(normalized_items)} "
            f"specifications found ({found_count/len(normalized_items)*100:.1f}%)"
        )
        
        return results
    
    def close(self):
        """Close MongoDB connection if open."""
        if self._mongo_client:
            self._mongo_client.close()
            logger.info("[SpecRetriever] MongoDB connection closed")
