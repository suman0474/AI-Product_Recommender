"""
Schema Service
===============
Service layer for product schema management.

Replaces direct blob/collection access with MongoDB queries.
Provides multi-strategy search for flexible schema retrieval.

Usage:
    from services.schema_service import schema_service

    # Get schema
    schema = schema_service.get_schema("Pressure Transmitter")

    # Save schema
    schema_service.save_schema("Pressure Transmitter", {
        "mandatory": ["range", "accuracy"],
        "optional": ["output_signal"]
    })

    # List all product types
    types = schema_service.get_all_product_types()
"""
import os
import sys
from typing import Optional, Dict, List, Any
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mongodb_manager import mongodb_manager, is_mongodb_available


class SchemaService:
    """
    Manages product type schemas.

    Provides:
    - Multi-strategy search (exact, normalized, regex)
    - Schema CRUD operations
    - Product type listing
    """

    def __init__(self):
        self._collection = None

    @property
    def collection(self):
        """Lazy collection access"""
        if self._collection is None:
            self._collection = mongodb_manager.get_collection('specs')
        return self._collection

    def _normalize_product_type(self, product_type: str) -> str:
        """Normalize product type for consistent matching"""
        return product_type.lower().replace(" ", "").replace("_", "").replace("-", "")

    def get_schema(self, product_type: str) -> Optional[Dict]:
        """
        Get product schema with multi-strategy search.

        Search strategies (in order):
        1. Exact match on product_type
        2. Normalized match on metadata.normalized_product_type
        3. Regex case-insensitive match

        Args:
            product_type: Product type name (e.g., "Pressure Transmitter")

        Returns:
            Schema dict with mandatory/optional fields, or None if not found

        Example:
            >>> schema = schema_service.get_schema("Pressure Transmitter")
            >>> print(schema.get('mandatory'))
            ['measurement_range', 'accuracy', 'output_signal']
        """
        if self.collection is None:
            return self._get_schema_fallback(product_type)

        normalized = self._normalize_product_type(product_type)

        # Strategy 1: Exact match
        schema_doc = self.collection.find_one({'product_type': product_type})

        if not schema_doc:
            # Strategy 2: Normalized match
            schema_doc = self.collection.find_one({
                'metadata.normalized_product_type': normalized
            })

        if not schema_doc:
            # Strategy 3: Regex case-insensitive
            schema_doc = self.collection.find_one({
                '$or': [
                    {'product_type': {'$regex': f'^{product_type}$', '$options': 'i'}},
                    {'metadata.product_type': {'$regex': f'^{product_type}$', '$options': 'i'}}
                ]
            })

        if not schema_doc:
            # Strategy 4: Partial match
            schema_doc = self.collection.find_one({
                '$or': [
                    {'product_type': {'$regex': product_type, '$options': 'i'}},
                    {'metadata.product_type': {'$regex': product_type, '$options': 'i'}}
                ]
            })

        if schema_doc:
            return self._extract_schema_data(schema_doc)

        return None

    def _extract_schema_data(self, schema_doc: Dict) -> Dict:
        """Extract schema data from document"""
        # If document has 'data' field, return it
        if 'data' in schema_doc:
            return schema_doc['data']

        # Otherwise, return document without metadata fields
        return {
            k: v for k, v in schema_doc.items()
            if k not in ['_id', 'metadata', 'product_type', 'created_at', 'updated_at']
        }

    def _get_schema_fallback(self, product_type: str) -> Optional[Dict]:
        """Fallback to Azure Blob if MongoDB not available"""
        try:
            from azure_blob_config import get_azure_blob_connection
            conn = get_azure_blob_connection()
            blob_collection = conn['collections']['specs']

            schema_doc = blob_collection.find_one({'product_type': product_type})
            if schema_doc:
                return self._extract_schema_data(schema_doc)
        except Exception as e:
            print(f"Schema fallback error: {e}")

        return None

    def save_schema(self, product_type: str, schema_data: Dict) -> bool:
        """
        Save or update product schema.

        Args:
            product_type: Product type name
            schema_data: Schema with mandatory/optional fields and descriptions

        Returns:
            True if successful

        Example:
            >>> schema_service.save_schema("Pressure Transmitter", {
            ...     "mandatory": ["range", "accuracy"],
            ...     "optional": ["housing_material"],
            ...     "field_descriptions": {"range": "Operating pressure range"}
            ... })
            True
        """
        if self.collection is None:
            print("⚠️ MongoDB not available - cannot save schema")
            return False

        normalized = self._normalize_product_type(product_type)

        document = {
            "product_type": product_type,
            "metadata": {
                "normalized_product_type": normalized,
                "product_type": product_type
            },
            "data": schema_data,
            "updated_at": datetime.utcnow()
        }

        result = self.collection.update_one(
            {'metadata.normalized_product_type': normalized},
            {
                '$set': document,
                '$setOnInsert': {'created_at': datetime.utcnow()}
            },
            upsert=True
        )

        return result.acknowledged

    def delete_schema(self, product_type: str) -> bool:
        """
        Delete product schema.

        Args:
            product_type: Product type name

        Returns:
            True if deleted
        """
        if self.collection is None:
            return False

        normalized = self._normalize_product_type(product_type)

        result = self.collection.delete_one({
            '$or': [
                {'product_type': product_type},
                {'metadata.normalized_product_type': normalized}
            ]
        })

        return result.deleted_count > 0

    def get_all_product_types(self) -> List[str]:
        """
        Get list of all available product types.

        Returns:
            List of product type names

        Example:
            >>> types = schema_service.get_all_product_types()
            >>> print(types)
            ['Pressure Transmitter', 'Temperature Transmitter', ...]
        """
        if self.collection is None:
            return self._get_all_product_types_fallback()

        return self.collection.distinct('product_type')

    def _get_all_product_types_fallback(self) -> List[str]:
        """Fallback to Azure Blob"""
        try:
            from azure_blob_config import get_azure_blob_connection
            conn = get_azure_blob_connection()
            blob_collection = conn['collections']['specs']
            return blob_collection.distinct('product_type', {})
        except Exception as e:
            print(f"Product types fallback error: {e}")
            return []

    def search_schemas(self, query: str) -> List[Dict]:
        """
        Search schemas by product type (partial match).

        Args:
            query: Search query string

        Returns:
            List of matching schema documents

        Example:
            >>> results = schema_service.search_schemas("Transmitter")
            >>> print([r['product_type'] for r in results])
            ['Pressure Transmitter', 'Temperature Transmitter', ...]
        """
        if self.collection is None:
            return []

        cursor = self.collection.find({
            '$or': [
                {'product_type': {'$regex': query, '$options': 'i'}},
                {'metadata.product_type': {'$regex': query, '$options': 'i'}}
            ]
        })

        results = []
        for doc in cursor:
            doc['_id'] = str(doc['_id'])
            results.append(doc)

        return results

    def get_field_descriptions(self, product_type: str) -> Dict[str, str]:
        """
        Get field descriptions for a product type.

        Args:
            product_type: Product type name

        Returns:
            Dict mapping field names to descriptions
        """
        schema = self.get_schema(product_type)
        if schema:
            return schema.get('field_descriptions', {})
        return {}

    def get_mandatory_fields(self, product_type: str) -> List[str]:
        """
        Get mandatory fields for a product type.

        Args:
            product_type: Product type name

        Returns:
            List of mandatory field names
        """
        schema = self.get_schema(product_type)
        if schema:
            return schema.get('mandatory', [])
        return []

    def get_optional_fields(self, product_type: str) -> List[str]:
        """
        Get optional fields for a product type.

        Args:
            product_type: Product type name

        Returns:
            List of optional field names
        """
        schema = self.get_schema(product_type)
        if schema:
            return schema.get('optional', [])
        return []

    def schema_exists(self, product_type: str) -> bool:
        """
        Check if schema exists for product type.

        Args:
            product_type: Product type name

        Returns:
            True if schema exists
        """
        return self.get_schema(product_type) is not None

    def get_schema_count(self) -> int:
        """Get total number of schemas"""
        if self.collection is None:
            return 0
        return self.collection.count_documents({})

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on schema service"""
        return {
            'service': 'SchemaService',
            'status': 'healthy' if self.collection is not None else 'unavailable',
            'mongodb_available': is_mongodb_available(),
            'schema_count': self.get_schema_count()
        }


# Singleton instance
schema_service = SchemaService()
