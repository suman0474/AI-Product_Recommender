"""
Image Service
==============
Service layer for product image management.

Implements cache-first pattern:
- Check MongoDB cache first (instant)
- Fall back to fetch/generate if not cached
- Auto-expire old cache entries (TTL indexes)

Usage:
    from services.image_service import image_service

    # Get cached image
    image = image_service.get_cached_image(
        vendor_name="Emerson",
        model_family="Rosemount 3051",
        product_type="Pressure Transmitter"
    )

    # Cache new image
    cached = image_service.cache_image(
        vendor_name="Emerson",
        model_family="Rosemount 3051",
        product_type="Pressure Transmitter",
        image_bytes=image_data
    )
"""
import os
import sys
from typing import Optional, Dict, List, Any
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mongodb_manager import mongodb_manager, is_mongodb_available
from core.azure_blob_file_manager import azure_blob_file_manager
from core.sas_utils import generate_sas_url


class ImageService:
    """
    Manages product images with cache-first pattern.

    Collections:
    - images: Product model images (vendor + model specific)
    - generic_images: AI-generated product type images
    - vendor_logos: Vendor company logos
    """

    def __init__(self):
        self._images_collection = None
        self._generic_images_collection = None
        self._vendor_logos_collection = None

    @property
    def images_collection(self):
        """Lazy images collection access"""
        if self._images_collection is None:
            self._images_collection = mongodb_manager.get_collection('images')
        return self._images_collection

    @property
    def generic_images_collection(self):
        """Lazy generic_images collection access"""
        if self._generic_images_collection is None:
            self._generic_images_collection = mongodb_manager.get_collection('generic_images')
        return self._generic_images_collection

    @property
    def vendor_logos_collection(self):
        """Lazy vendor_logos collection access"""
        if self._vendor_logos_collection is None:
            self._vendor_logos_collection = mongodb_manager.get_collection('vendor_logos')
        return self._vendor_logos_collection

    def _normalize_name(self, name: str) -> str:
        """Normalize name for consistent cache keys"""
        return name.strip().lower().replace(' ', '_').replace('-', '_')

    # ============================================================
    # Product Images (Vendor + Model Specific)
    # ============================================================

    def get_cached_image(
        self,
        vendor_name: str,
        model_family: str,
        product_type: str
    ) -> Optional[Dict]:
        """
        Get cached product image from MongoDB.

        Args:
            vendor_name: Vendor name
            model_family: Model family name
            product_type: Product type

        Returns:
            Image document with SAS URL or None if not cached

        Example:
            >>> image = image_service.get_cached_image(
            ...     "Emerson", "Rosemount 3051", "Pressure Transmitter"
            ... )
            >>> if image:
            ...     print(image['sas_url'])
        """
        if not self.images_collection:
            return None

        normalized_vendor = self._normalize_name(vendor_name)
        normalized_model = self._normalize_name(model_family)
        normalized_product = self._normalize_name(product_type)

        # Build cache key
        cache_id = f"product_image_{normalized_product}_{normalized_vendor}_{normalized_model}"

        cached = self.images_collection.find_one({
            'id': cache_id,
            'status': 'active'
        })

        if cached:
            cached['_id'] = str(cached['_id'])

            # Add SAS URL for secure access
            blob_url = cached.get('image', {}).get('blobUrl') or cached.get('image', {}).get('blob_url')
            if blob_url:
                cached['sas_url'] = generate_sas_url(blob_url, expiry_hours=24)

            return cached

        return None

    def cache_image(
        self,
        vendor_name: str,
        model_family: str,
        product_type: str,
        image_bytes: bytes,
        content_type: str = 'image/jpeg'
    ) -> Dict:
        """
        Upload image to Azure Blob and cache metadata in MongoDB.

        Args:
            vendor_name: Vendor name
            model_family: Model family name
            product_type: Product type
            image_bytes: Image file bytes
            content_type: MIME type

        Returns:
            Cached image document with SAS URL
        """
        # Upload to Azure Blob
        blob_info = azure_blob_file_manager.upload_product_image(
            image_bytes=image_bytes,
            product_type=product_type,
            vendor_name=vendor_name,
            model_family=model_family
        )

        # Normalize for cache key
        normalized_vendor = self._normalize_name(vendor_name)
        normalized_model = self._normalize_name(model_family)
        normalized_product = self._normalize_name(product_type)

        # Create cache document
        cache_doc = {
            "id": f"product_image_{normalized_product}_{normalized_vendor}_{normalized_model}",
            "type": "product_model_image",
            "productType": product_type,
            "product_type_normalized": normalized_product,
            "vendor_name": vendor_name,
            "vendor_name_normalized": normalized_vendor,
            "model_family": model_family,
            "model_family_normalized": normalized_model,
            "image": {
                "storage": "azure_blob",
                "container": blob_info.get('container'),
                "blobName": blob_info.get('blob_path'),
                "blobUrl": blob_info.get('blob_url')
            },
            "contentType": content_type,
            "fileSize": len(image_bytes),
            "createdAt": datetime.utcnow(),
            "status": "active"
        }

        # Upsert to cache
        if self.images_collection:
            try:
                self.images_collection.update_one(
                    {'id': cache_doc['id']},
                    {'$set': cache_doc},
                    upsert=True
                )
            except Exception as e:
                print(f"Image cache error: {e}")

        # Return with SAS URL
        cache_doc['sas_url'] = generate_sas_url(blob_info['blob_url'], expiry_hours=24)
        return cache_doc

    def invalidate_image_cache(
        self,
        vendor_name: str,
        model_family: str,
        product_type: str
    ) -> bool:
        """
        Invalidate cached image.

        Args:
            vendor_name: Vendor name
            model_family: Model family
            product_type: Product type

        Returns:
            True if invalidated
        """
        if not self.images_collection:
            return False

        normalized_vendor = self._normalize_name(vendor_name)
        normalized_model = self._normalize_name(model_family)
        normalized_product = self._normalize_name(product_type)

        cache_id = f"product_image_{normalized_product}_{normalized_vendor}_{normalized_model}"

        result = self.images_collection.update_one(
            {'id': cache_id},
            {'$set': {'status': 'invalidated', 'invalidatedAt': datetime.utcnow()}}
        )
        return result.modified_count > 0

    # ============================================================
    # Generic Images (Product Type Only)
    # ============================================================

    def get_generic_image(self, product_type: str) -> Optional[Dict]:
        """
        Get cached generic product image.

        Args:
            product_type: Product type

        Returns:
            Generic image document with SAS URL or None
        """
        if not self.generic_images_collection:
            return None

        normalized = self._normalize_name(product_type)

        cached = self.generic_images_collection.find_one({
            'product_type_normalized': normalized,
            'status': 'active'
        })

        if cached:
            cached['_id'] = str(cached['_id'])

            blob_url = cached.get('image', {}).get('blobUrl') or cached.get('image', {}).get('blob_url')
            if blob_url:
                cached['sas_url'] = generate_sas_url(blob_url, expiry_hours=24)

            return cached

        return None

    def cache_generic_image(
        self,
        product_type: str,
        image_bytes: bytes,
        content_type: str = 'image/png'
    ) -> Dict:
        """
        Cache generic product image.

        Args:
            product_type: Product type
            image_bytes: Image bytes
            content_type: MIME type

        Returns:
            Cached image document with SAS URL
        """
        # Upload to Azure Blob
        blob_info = azure_blob_file_manager.upload_generic_image(
            image_bytes=image_bytes,
            product_type=product_type
        )

        normalized = self._normalize_name(product_type)

        cache_doc = {
            "id": f"generic_image_{normalized}",
            "type": "generic_product_image",
            "product_type": product_type,
            "product_type_normalized": normalized,
            "image": {
                "storage": "azure_blob",
                "container": blob_info.get('container'),
                "blobName": blob_info.get('blob_path'),
                "blobUrl": blob_info.get('blob_url')
            },
            "contentType": content_type,
            "fileSize": len(image_bytes),
            "createdAt": datetime.utcnow(),
            "status": "active"
        }

        if self.generic_images_collection:
            try:
                self.generic_images_collection.update_one(
                    {'product_type_normalized': normalized},
                    {'$set': cache_doc},
                    upsert=True
                )
            except Exception as e:
                print(f"Generic image cache error: {e}")

        cache_doc['sas_url'] = generate_sas_url(blob_info['blob_url'], expiry_hours=24)
        return cache_doc

    # ============================================================
    # Vendor Logos
    # ============================================================

    def get_vendor_logo(self, vendor_name: str) -> Optional[Dict]:
        """
        Get cached vendor logo.

        Args:
            vendor_name: Vendor name

        Returns:
            Logo document with SAS URL or None
        """
        if not self.vendor_logos_collection:
            return None

        normalized = self._normalize_name(vendor_name)

        cached = self.vendor_logos_collection.find_one({
            'vendor_name_normalized': normalized,
            'status': 'active'
        })

        if cached:
            cached['_id'] = str(cached['_id'])

            blob_url = cached.get('image', {}).get('blobUrl') or cached.get('image', {}).get('blob_url')
            if blob_url:
                cached['sas_url'] = generate_sas_url(blob_url, expiry_hours=24)

            return cached

        return None

    def cache_vendor_logo(
        self,
        vendor_name: str,
        image_bytes: bytes,
        content_type: str = 'image/png'
    ) -> Dict:
        """
        Cache vendor logo.

        Args:
            vendor_name: Vendor name
            image_bytes: Logo image bytes
            content_type: MIME type

        Returns:
            Cached logo document with SAS URL
        """
        normalized = self._normalize_name(vendor_name)
        container_name = azure_blob_file_manager.CONTAINERS['vendor_logos']
        blob_path = f"{normalized}_logo.png"

        blob_url = azure_blob_file_manager.upload_file(
            file_bytes=image_bytes,
            blob_path=blob_path,
            content_type=content_type,
            container_name=container_name
        )

        cache_doc = {
            "id": f"vendor_logo_{normalized}",
            "vendor_name": vendor_name,
            "vendor_name_normalized": normalized,
            "image": {
                "storage": "azure_blob",
                "container": container_name,
                "blobName": blob_path,
                "blobUrl": blob_url
            },
            "contentType": content_type,
            "fileSize": len(image_bytes),
            "createdAt": datetime.utcnow(),
            "status": "active"
        }

        if self.vendor_logos_collection:
            try:
                self.vendor_logos_collection.update_one(
                    {'vendor_name_normalized': normalized},
                    {'$set': cache_doc},
                    upsert=True
                )
            except Exception as e:
                print(f"Vendor logo cache error: {e}")

        cache_doc['sas_url'] = generate_sas_url(blob_url, expiry_hours=24)
        return cache_doc

    # ============================================================
    # Batch Operations
    # ============================================================

    def get_images_for_products(
        self,
        products: List[Dict]
    ) -> Dict[str, Optional[Dict]]:
        """
        Get cached images for multiple products.

        Args:
            products: List of product dicts with vendor_name, model_family, product_type

        Returns:
            Dict mapping product key to image info
        """
        results = {}

        for product in products:
            vendor = product.get('vendor_name', '')
            model = product.get('model_family', '')
            ptype = product.get('product_type', '')

            key = f"{vendor}_{model}_{ptype}"

            image = self.get_cached_image(vendor, model, ptype)
            results[key] = image

        return results

    def get_generic_images_batch(
        self,
        product_types: List[str]
    ) -> Dict[str, Optional[Dict]]:
        """
        Get generic images for multiple product types.

        Args:
            product_types: List of product types

        Returns:
            Dict mapping product_type to image info
        """
        results = {}

        for product_type in product_types:
            image = self.get_generic_image(product_type)
            results[product_type] = image

        return results

    # ============================================================
    # Statistics and Health
    # ============================================================

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        stats = {
            'product_images': 0,
            'generic_images': 0,
            'vendor_logos': 0
        }

        if self.images_collection:
            stats['product_images'] = self.images_collection.count_documents({'status': 'active'})

        if self.generic_images_collection:
            stats['generic_images'] = self.generic_images_collection.count_documents({'status': 'active'})

        if self.vendor_logos_collection:
            stats['vendor_logos'] = self.vendor_logos_collection.count_documents({'status': 'active'})

        return stats

    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        stats = self.get_cache_stats()

        return {
            'service': 'ImageService',
            'status': 'healthy' if self.images_collection else 'unavailable',
            'mongodb_available': is_mongodb_available(),
            'blob_available': azure_blob_file_manager.is_connected(),
            'cache_stats': stats
        }


# Singleton instance
image_service = ImageService()
