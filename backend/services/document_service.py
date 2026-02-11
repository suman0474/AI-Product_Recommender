"""
Document Service
=================
Service layer for document management (strategy/standards).

Implements simplified pattern:
- File → Azure Blob
- Metadata + URL → MongoDB
- SAS URLs for secure access

Usage:
    from services.document_service import document_service

    # Upload strategy document
    result = document_service.upload_strategy_document(
        file_bytes=file.read(),
        filename="strategy.pdf",
        user_id=123,
        content_type="application/pdf"
    )

    # Get documents with SAS URLs
    docs = document_service.get_strategy_documents(user_id=123)
"""
import os
import sys
from typing import Optional, Dict, List, Any
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from bson import ObjectId
    BSON_AVAILABLE = True
except ImportError:
    BSON_AVAILABLE = False
    import uuid
    class ObjectId:
        def __init__(self, oid=None):
            self.oid = oid or str(uuid.uuid4()).replace('-', '')[:24]
        def __str__(self):
            return self.oid

try:
    from werkzeug.utils import secure_filename
except ImportError:
    def secure_filename(filename):
        return filename.replace(' ', '_').replace('/', '_').replace('\\', '_')

from core.mongodb_manager import mongodb_manager, is_mongodb_available
from core.azure_blob_file_manager import azure_blob_file_manager
from core.sas_utils import add_sas_to_documents, generate_sas_url
from cosmosdb_config import CosmosDBConnection, CosmosContainers


class DocumentService:
    """
    Manages strategy and standards documents.

    Pattern:
    - Upload: File → Blob, Metadata → MongoDB
    - List: MongoDB query + SAS URL generation
    - Download: Direct SAS URL access
    """

    def __init__(self):
        self._strategy_collection = None
        self._standards_collection = None
        self._cosmos_strategy_container = None
        self._cosmos_standards_container = None

    @property
    def strategy_collection(self):
        """Lazy strategy collection access"""
        if self._strategy_collection is None:
            try:
                self._strategy_collection = mongodb_manager.get_collection('stratergy')
                if self._strategy_collection:
                    print("✓ MongoDB strategy collection connected")
            except Exception as e:
                print(f"⚠️ Failed to get strategy collection: {e}")
                self._strategy_collection = None
        return self._strategy_collection

    @property
    def standards_collection(self):
        """Lazy standards collection access"""
        if self._standards_collection is None:
            try:
                self._standards_collection = mongodb_manager.get_collection('standards')
                if self._standards_collection:
                    print("✓ MongoDB standards collection connected")
            except Exception as e:
                print(f"⚠️ Failed to get standards collection: {e}")
                self._standards_collection = None
        return self._standards_collection

    @property
    def cosmos_strategy_container(self):
        """Lazy CosmosDB strategy container access"""
        if self._cosmos_strategy_container is None:
            try:
                cosmos_conn = CosmosDBConnection.get_instance()
                if cosmos_conn.client:
                    self._cosmos_strategy_container = cosmos_conn.get_or_create_container(
                        CosmosContainers.STRATEGY_DOCUMENTS.value
                    )
                    if self._cosmos_strategy_container:
                        print("✓ CosmosDB strategy container connected")
            except Exception as e:
                print(f"⚠️ Failed to get CosmosDB strategy container: {e}")
                self._cosmos_strategy_container = None
        return self._cosmos_strategy_container

    @property
    def cosmos_standards_container(self):
        """Lazy CosmosDB standards container access"""
        if self._cosmos_standards_container is None:
            try:
                cosmos_conn = CosmosDBConnection.get_instance()
                if cosmos_conn.client:
                    self._cosmos_standards_container = cosmos_conn.get_or_create_container(
                        CosmosContainers.STANDARDS_DOCUMENTS.value
                    )
                    if self._cosmos_standards_container:
                        print("✓ CosmosDB standards container connected")
            except Exception as e:
                print(f"⚠️ Failed to get CosmosDB standards container: {e}")
                self._cosmos_standards_container = None
        return self._cosmos_standards_container

    # ============================================================
    # Strategy Documents
    # ============================================================

    def upload_strategy_document(
        self,
        file_bytes: bytes,
        filename: str,
        user_id: int,
        content_type: str = 'application/pdf',
        username: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Upload strategy document.

        Flow:
        1. Upload file to Azure Blob
        2. Store metadata in MongoDB
        3. Generate SAS URL for immediate access

        Args:
            file_bytes: File content
            filename: Original filename
            user_id: User ID
            content_type: MIME type
            username: Username (optional)
            metadata: Additional metadata (optional)

        Returns:
            Dict with document_id, filename, sas_url
        """
        filename = secure_filename(filename)

        # 1. Upload to Azure Blob
        blob_info = azure_blob_file_manager.upload_strategy_document(
            file_bytes=file_bytes,
            filename=filename,
            user_id=user_id,
            content_type=content_type
        )

        # 2. Store metadata in MongoDB
        document = {
            "user_id": user_id,
            "filename": filename,
            "file_type": content_type,
            "file_size": len(file_bytes),
            "uploaded_at": datetime.utcnow().isoformat(),
            "uploaded_by_username": username,
            "storage": blob_info,
            "metadata": metadata or {}
        }

        document_id = None
        metadata_stored = False

        # Try MongoDB first
        if self.strategy_collection:
            try:
                result = self.strategy_collection.insert_one(document)
                document_id = str(result.inserted_id)
                print(f"✓ Strategy document metadata stored in MongoDB: {document_id}")
                metadata_stored = True
            except Exception as e:
                print(f"❌ MongoDB strategy insert error: {e}")
                import traceback
                traceback.print_exc()

        # Try CosmosDB as fallback
        if not metadata_stored and self.cosmos_strategy_container:
            try:
                # Add id field for CosmosDB
                cosmos_doc = {
                    "id": str(ObjectId()),
                    **document
                }
                result = self.cosmos_strategy_container.create_item(body=cosmos_doc)
                document_id = result.get('id')
                print(f"✓ Strategy document metadata stored in CosmosDB: {document_id}")
                metadata_stored = True
            except Exception as e:
                print(f"❌ CosmosDB strategy insert error: {e}")
                import traceback
                traceback.print_exc()

        if not metadata_stored:
            print(f"⚠️ Neither MongoDB nor CosmosDB available - metadata not stored (file is in blob storage)")

        # 3. Generate SAS URL
        sas_url = generate_sas_url(blob_info['blob_url'], expiry_hours=24)

        return {
            "success": True,
            "document_id": document_id,
            "filename": filename,
            "file_size": len(file_bytes),
            "blob_url": blob_info['blob_url'],
            "sas_url": sas_url
        }

    def get_strategy_documents(
        self,
        user_id: int,
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict]:
        """
        Get all strategy documents for user with SAS URLs.

        Args:
            user_id: User ID
            limit: Max documents
            skip: Pagination offset

        Returns:
            List of documents with sas_url field
        """
        if not self.strategy_collection:
            return []

        cursor = self.strategy_collection.find(
            {"user_id": user_id}
        ).sort("uploaded_at", -1).skip(skip).limit(limit)

        documents = []
        for doc in cursor:
            doc['_id'] = str(doc['_id'])
            documents.append(doc)

        # Add SAS URLs
        return add_sas_to_documents(documents, "storage.blob_url", expiry_hours=1)

    def delete_strategy_document(self, document_id: str, user_id: int) -> bool:
        """
        Delete strategy document.

        Args:
            document_id: Document ID
            user_id: User ID (security check)

        Returns:
            True if deleted
        """
        if not self.strategy_collection:
            return False

        try:
            doc = self.strategy_collection.find_one({
                '_id': ObjectId(document_id),
                'user_id': user_id
            })

            if not doc:
                return False

            # Delete blob
            blob_info = doc.get('storage', {})
            blob_path = blob_info.get('blob_path')
            container_name = blob_info.get('container')
            if blob_path:
                try:
                    azure_blob_file_manager.delete_file(blob_path, container_name=container_name)
                except Exception as e:
                    print(f"Blob delete error: {e}")

            # Delete metadata
            self.strategy_collection.delete_one({'_id': ObjectId(document_id)})
            return True

        except Exception as e:
            print(f"Strategy delete error: {e}")
            return False

    def get_strategy_document_count(self, user_id: int) -> int:
        """Get strategy document count for user"""
        if not self.strategy_collection:
            return 0
        return self.strategy_collection.count_documents({"user_id": user_id})

    # ============================================================
    # Standards Documents
    # ============================================================

    def upload_standards_document(
        self,
        file_bytes: bytes,
        filename: str,
        user_id: int,
        content_type: str = 'application/pdf',
        username: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Upload standards document.

        Same flow as strategy documents.
        """
        filename = secure_filename(filename)

        # 1. Upload to Azure Blob
        blob_info = azure_blob_file_manager.upload_standards_document(
            file_bytes=file_bytes,
            filename=filename,
            user_id=user_id,
            content_type=content_type
        )

        # 2. Store metadata in MongoDB
        document = {
            "user_id": user_id,
            "filename": filename,
            "file_type": content_type,
            "file_size": len(file_bytes),
            "uploaded_at": datetime.utcnow().isoformat(),
            "uploaded_by_username": username,
            "storage": blob_info,
            "metadata": metadata or {}
        }

        document_id = None
        metadata_stored = False

        # Try MongoDB first
        if self.standards_collection:
            try:
                result = self.standards_collection.insert_one(document)
                document_id = str(result.inserted_id)
                print(f"✓ Standards document metadata stored in MongoDB: {document_id}")
                metadata_stored = True
            except Exception as e:
                print(f"❌ MongoDB standards insert error: {e}")
                import traceback
                traceback.print_exc()

        # Try CosmosDB as fallback
        if not metadata_stored and self.cosmos_standards_container:
            try:
                # Add id field for CosmosDB
                cosmos_doc = {
                    "id": str(ObjectId()),
                    **document
                }
                result = self.cosmos_standards_container.create_item(body=cosmos_doc)
                document_id = result.get('id')
                print(f"✓ Standards document metadata stored in CosmosDB: {document_id}")
                metadata_stored = True
            except Exception as e:
                print(f"❌ CosmosDB standards insert error: {e}")
                import traceback
                traceback.print_exc()

        if not metadata_stored:
            print(f"⚠️ Neither MongoDB nor CosmosDB available - metadata not stored (file is in blob storage)")

        # 3. Generate SAS URL
        sas_url = generate_sas_url(blob_info['blob_url'], expiry_hours=24)

        return {
            "success": True,
            "document_id": document_id,
            "filename": filename,
            "file_size": len(file_bytes),
            "blob_url": blob_info['blob_url'],
            "sas_url": sas_url
        }

    def get_standards_documents(
        self,
        user_id: int,
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict]:
        """Get all standards documents for user with SAS URLs"""
        if not self.standards_collection:
            return []

        cursor = self.standards_collection.find(
            {"user_id": user_id}
        ).sort("uploaded_at", -1).skip(skip).limit(limit)

        documents = []
        for doc in cursor:
            doc['_id'] = str(doc['_id'])
            documents.append(doc)

        return add_sas_to_documents(documents, "storage.blob_url", expiry_hours=1)

    def delete_standards_document(self, document_id: str, user_id: int) -> bool:
        """Delete standards document"""
        if not self.standards_collection:
            return False

        try:
            doc = self.standards_collection.find_one({
                '_id': ObjectId(document_id),
                'user_id': user_id
            })

            if not doc:
                return False

            # Delete blob
            blob_info = doc.get('storage', {})
            blob_path = blob_info.get('blob_path')
            container_name = blob_info.get('container')
            if blob_path:
                try:
                    azure_blob_file_manager.delete_file(blob_path, container_name=container_name)
                except Exception as e:
                    print(f"Blob delete error: {e}")

            # Delete metadata
            self.standards_collection.delete_one({'_id': ObjectId(document_id)})
            return True

        except Exception as e:
            print(f"Standards delete error: {e}")
            return False

    def get_standards_document_count(self, user_id: int) -> int:
        """Get standards document count for user"""
        if not self.standards_collection:
            return 0
        return self.standards_collection.count_documents({"user_id": user_id})

    # ============================================================
    # Common Methods
    # ============================================================

    def get_document_by_id(
        self,
        document_id: str,
        user_id: int,
        doc_type: str = 'strategy'
    ) -> Optional[Dict]:
        """
        Get single document by ID.

        Args:
            document_id: Document ID
            user_id: User ID
            doc_type: 'strategy' or 'standards'

        Returns:
            Document with SAS URL or None
        """
        collection = self.strategy_collection if doc_type == 'strategy' else self.standards_collection

        if not collection:
            return None

        try:
            doc = collection.find_one({
                '_id': ObjectId(document_id),
                'user_id': user_id
            })

            if doc:
                doc['_id'] = str(doc['_id'])
                blob_url = doc.get('storage', {}).get('blob_url')
                if blob_url:
                    doc['sas_url'] = generate_sas_url(blob_url, expiry_hours=24)
                return doc

        except Exception as e:
            print(f"Get document error: {e}")

        return None

    def refresh_sas_url(self, document_id: str, doc_type: str = 'strategy') -> Optional[str]:
        """
        Generate fresh SAS URL for document.

        Args:
            document_id: Document ID
            doc_type: 'strategy' or 'standards'

        Returns:
            New SAS URL or None
        """
        collection = self.strategy_collection if doc_type == 'strategy' else self.standards_collection

        if not collection:
            return None

        try:
            doc = collection.find_one({'_id': ObjectId(document_id)})
            if doc:
                blob_url = doc.get('storage', {}).get('blob_url')
                if blob_url:
                    return generate_sas_url(blob_url, expiry_hours=24)
        except Exception as e:
            print(f"SAS refresh error: {e}")

        return None

    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'service': 'DocumentService',
            'status': 'healthy' if self.strategy_collection else 'unavailable',
            'mongodb_available': is_mongodb_available(),
            'blob_available': azure_blob_file_manager.is_connected()
        }


# Singleton instance
document_service = DocumentService()
