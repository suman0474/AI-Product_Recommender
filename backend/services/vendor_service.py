"""
Vendor Service
===============
Service layer for vendor product catalog management.

Provides fast, indexed queries for vendor/product search.
Replaces slow blob scanning with MongoDB indexed queries.

Usage:
    from services.vendor_service import vendor_service

    # Search products
    results = vendor_service.search_products(
        product_type="Pressure Transmitter",
        vendors=["Emerson", "Honeywell"]
    )

    # Get vendors by product type
    vendors = vendor_service.get_vendors_by_product_type("Temperature Transmitter")

    # Get model families
    models = vendor_service.get_model_families_by_vendor("Emerson")
"""
import os
import sys
from typing import Optional, Dict, List, Any
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mongodb_manager import mongodb_manager, is_mongodb_available


class VendorService:
    """
    Manages vendor product catalogs.

    Provides:
    - Product search with filters
    - Vendor listing by product type
    - Model family discovery
    - Category expansion for flexible matching
    """

    # Related categories for search expansion
    CATEGORY_EXPANSIONS = {
        'transmitter': ['transmitter', 'sensor', 'transducer'],
        'sensor': ['sensor', 'transmitter', 'detector'],
        'gauge': ['gauge', 'indicator', 'meter'],
        'valve': ['valve', 'actuator', 'positioner'],
        'flow': ['flow meter', 'flow transmitter', 'flowmeter'],
        'pressure': ['pressure transmitter', 'pressure gauge', 'pressure sensor'],
        'temperature': ['temperature transmitter', 'thermocouple', 'rtd', 'thermometer'],
        'level': ['level transmitter', 'level sensor', 'level gauge'],
    }

    def __init__(self):
        self._collection = None

    @property
    def collection(self):
        """Lazy collection access"""
        if self._collection is None:
            self._collection = mongodb_manager.get_collection('vendors')
        return self._collection

    def _normalize_name(self, name: str) -> str:
        """Normalize vendor/model name"""
        return name.lower().strip().replace(' ', '_').replace('-', '_')

    def _get_search_categories(self, product_type: str) -> List[str]:
        """
        Expand product type to include related categories.

        Args:
            product_type: Base product type

        Returns:
            List of related categories for search

        Example:
            >>> _get_search_categories("Pressure Transmitter")
            ['Pressure Transmitter', 'Pressure Sensor', 'Pressure Transducer']
        """
        categories = [product_type]

        # Check for category expansions
        product_lower = product_type.lower()

        for keyword, expansions in self.CATEGORY_EXPANSIONS.items():
            if keyword in product_lower:
                for exp in expansions:
                    # Create variant by replacing the keyword
                    variant = product_type.lower().replace(keyword, exp)
                    if variant not in [c.lower() for c in categories]:
                        categories.append(variant.title())

        return categories

    def search_products(
        self,
        product_type: str,
        vendors: Optional[List[str]] = None,
        model_families: Optional[List[str]] = None,
        limit: int = 100,
        expand_categories: bool = True
    ) -> List[Dict]:
        """
        Search vendor products with filters.

        Args:
            product_type: Product type to search (e.g., "Pressure Transmitter")
            vendors: List of vendor names to filter by (optional)
            model_families: List of model families to filter by (optional)
            limit: Maximum results to return
            expand_categories: Whether to include related categories

        Returns:
            List of matching product documents

        Example:
            >>> results = vendor_service.search_products(
            ...     product_type="Pressure Transmitter",
            ...     vendors=["Emerson", "Honeywell"]
            ... )
        """
        if not self.collection:
            return self._search_products_fallback(product_type, vendors, model_families)

        # Build product type query with category expansion
        if expand_categories:
            categories = self._get_search_categories(product_type)
            product_query = {
                '$or': [
                    {'product_type': {'$regex': cat, '$options': 'i'}}
                    for cat in categories
                ]
            }
        else:
            product_query = {'product_type': {'$regex': product_type, '$options': 'i'}}

        # Build full query
        query = product_query

        if vendors:
            query['vendor_name'] = {'$in': vendors}

        if model_families:
            query['model_family'] = {'$in': model_families}

        # Execute query
        cursor = self.collection.find(query).limit(limit)

        results = []
        for doc in cursor:
            doc['_id'] = str(doc['_id'])
            results.append(doc)

        return results

    def _search_products_fallback(
        self,
        product_type: str,
        vendors: Optional[List[str]] = None,
        model_families: Optional[List[str]] = None
    ) -> List[Dict]:
        """Fallback to Azure Blob"""
        try:
            from azure_blob_config import get_azure_blob_connection
            conn = get_azure_blob_connection()
            blob_collection = conn['collections']['vendors']

            # Simple query
            all_docs = list(blob_collection.find({'product_type': product_type}))

            # Filter in memory
            results = []
            for doc in all_docs:
                if vendors and doc.get('vendor_name') not in vendors:
                    continue
                if model_families and doc.get('model_family') not in model_families:
                    continue
                results.append(doc)

            return results
        except Exception as e:
            print(f"Vendor search fallback error: {e}")
            return []

    def get_vendors_by_product_type(self, product_type: str) -> List[str]:
        """
        Get unique vendor names for a product type.

        Args:
            product_type: Product type name

        Returns:
            List of vendor names

        Example:
            >>> vendors = vendor_service.get_vendors_by_product_type("Pressure Transmitter")
            >>> print(vendors)
            ['Emerson', 'Honeywell', 'Siemens', ...]
        """
        if not self.collection:
            return self._get_vendors_fallback(product_type)

        return self.collection.distinct('vendor_name', {
            'product_type': {'$regex': product_type, '$options': 'i'}
        })

    def _get_vendors_fallback(self, product_type: str) -> List[str]:
        """Fallback to Azure Blob"""
        try:
            from azure_blob_config import get_azure_blob_connection
            conn = get_azure_blob_connection()
            blob_collection = conn['collections']['vendors']
            return blob_collection.distinct('vendor_name', {'product_type': product_type})
        except Exception:
            return []

    def get_model_families_by_vendor(
        self,
        vendor_name: str,
        product_type: Optional[str] = None
    ) -> List[str]:
        """
        Get model families for a vendor.

        Args:
            vendor_name: Vendor name
            product_type: Optional product type filter

        Returns:
            List of model family names

        Example:
            >>> models = vendor_service.get_model_families_by_vendor("Emerson", "Pressure Transmitter")
            >>> print(models)
            ['Rosemount 3051', 'Rosemount 2088', ...]
        """
        if not self.collection:
            return []

        query = {'vendor_name': vendor_name}
        if product_type:
            query['product_type'] = {'$regex': product_type, '$options': 'i'}

        return self.collection.distinct('model_family', query)

    def get_all_vendors(self) -> List[str]:
        """Get all unique vendor names"""
        if not self.collection:
            return []
        return self.collection.distinct('vendor_name')

    def get_all_product_types(self) -> List[str]:
        """Get all unique product types in vendor catalog"""
        if not self.collection:
            return []
        return self.collection.distinct('product_type')

    def get_vendor_product(
        self,
        vendor_name: str,
        model_family: str,
        product_type: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get specific vendor product by vendor and model.

        Args:
            vendor_name: Vendor name
            model_family: Model family name
            product_type: Optional product type

        Returns:
            Product document or None
        """
        if not self.collection:
            return None

        query = {
            'vendor_name': vendor_name,
            'model_family': model_family
        }
        if product_type:
            query['product_type'] = {'$regex': product_type, '$options': 'i'}

        doc = self.collection.find_one(query)
        if doc:
            doc['_id'] = str(doc['_id'])
        return doc

    def save_vendor_product(self, product_data: Dict) -> str:
        """
        Save vendor product to catalog.

        Args:
            product_data: Product document

        Returns:
            Inserted document ID
        """
        if not self.collection:
            raise RuntimeError("MongoDB not available")

        # Add normalized names for indexing
        product_data['vendor_name_normalized'] = self._normalize_name(
            product_data.get('vendor_name', '')
        )
        product_data['model_family_normalized'] = self._normalize_name(
            product_data.get('model_family', '')
        )
        product_data['updated_at'] = datetime.utcnow()

        result = self.collection.insert_one(product_data)
        return str(result.inserted_id)

    def update_vendor_product(
        self,
        vendor_name: str,
        model_family: str,
        update_data: Dict
    ) -> bool:
        """
        Update vendor product.

        Args:
            vendor_name: Vendor name
            model_family: Model family
            update_data: Fields to update

        Returns:
            True if updated
        """
        if not self.collection:
            return False

        update_data['updated_at'] = datetime.utcnow()

        result = self.collection.update_one(
            {'vendor_name': vendor_name, 'model_family': model_family},
            {'$set': update_data}
        )
        return result.modified_count > 0

    def delete_vendor_product(self, vendor_name: str, model_family: str) -> bool:
        """Delete vendor product"""
        if not self.collection:
            return False

        result = self.collection.delete_one({
            'vendor_name': vendor_name,
            'model_family': model_family
        })
        return result.deleted_count > 0

    def get_vendor_count(self) -> int:
        """Get total number of unique vendors"""
        if not self.collection:
            return 0
        return len(self.collection.distinct('vendor_name'))

    def get_product_count(self) -> int:
        """Get total number of products in catalog"""
        if not self.collection:
            return 0
        return self.collection.count_documents({})

    def search_by_specs(
        self,
        product_type: str,
        specifications: Dict[str, Any],
        limit: int = 50
    ) -> List[Dict]:
        """
        Search products by specifications.

        Args:
            product_type: Product type
            specifications: Dict of spec requirements
            limit: Max results

        Returns:
            Matching products
        """
        if not self.collection:
            return []

        query = {'product_type': {'$regex': product_type, '$options': 'i'}}

        # Add specification filters
        for spec_name, spec_value in specifications.items():
            query[f'specifications.{spec_name}'] = spec_value

        cursor = self.collection.find(query).limit(limit)

        results = []
        for doc in cursor:
            doc['_id'] = str(doc['_id'])
            results.append(doc)

        return results

    def get_submodel_mapping(self, product_type: str) -> Dict[str, List[str]]:
        """
        Get mapping of model families to sub-models.

        Args:
            product_type: Product type

        Returns:
            Dict mapping model_family to list of sub_models
        """
        if not self.collection:
            return {}

        products = self.search_products(product_type, limit=1000)

        mapping = {}
        for product in products:
            model_family = product.get('model_family')
            sub_models = product.get('data', {}).get('models', [])

            if model_family and sub_models:
                if model_family not in mapping:
                    mapping[model_family] = []

                for model in sub_models:
                    for sub in model.get('sub_models', []):
                        sub_name = sub.get('name')
                        if sub_name and sub_name not in mapping[model_family]:
                            mapping[model_family].append(sub_name)

        return mapping

    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'service': 'VendorService',
            'status': 'healthy' if self.collection else 'unavailable',
            'mongodb_available': is_mongodb_available(),
            'vendor_count': self.get_vendor_count(),
            'product_count': self.get_product_count()
        }


# Singleton instance
vendor_service = VendorService()
