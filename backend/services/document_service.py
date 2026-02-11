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
                    # Verify collection is accessible by checking collection name
                    try:
                        coll_name = self._standards_collection.name
                        print(f"   - Collection name: {coll_name}")
                        # Try a simple operation to verify connection
                        count = self._standards_collection.count_documents({})
                        print(f"   - Current document count: {count}")
                    except Exception as verify_error:
                        print(f"⚠️ Collection verification failed: {verify_error}")
                        self._standards_collection = None
                else:
                    print("❌ MongoDB standards collection is None")
            except Exception as e:
                print(f"⚠️ Failed to get standards collection: {e}")
                import traceback
                traceback.print_exc()
                self._standards_collection = None
        return self._standards_collection


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

        # Store metadata in MongoDB (Cosmos DB for MongoDB API)
        if self.strategy_collection:
            try:
                result = self.strategy_collection.insert_one(document)
                document_id = str(result.inserted_id)
                print(f"✓ Strategy document metadata stored in MongoDB: {document_id}")
            except Exception as e:
                print(f"❌ MongoDB strategy insert error: {e}")
                import traceback
                traceback.print_exc()
                print(f"⚠️ Metadata not stored in MongoDB (file is in blob storage)")
        else:
            print(f"⚠️ MongoDB not available - metadata not stored (file is in blob storage)")

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

        # Store metadata in MongoDB (Cosmos DB for MongoDB API)
        # CRITICAL: MongoDB storage is MANDATORY - fail if it doesn't work
        if not self.standards_collection:
            error_msg = "MongoDB standards collection not available. Cannot save document metadata."
            print(f"❌ {error_msg}")
            # Optionally delete the blob file if metadata storage fails
            try:
                azure_blob_file_manager.delete_file(
                    blob_info['blob_path'],
                    container_name=blob_info.get('container')
                )
                print(f"🗑️ Cleaned up blob file: {blob_info['blob_path']}")
            except Exception as cleanup_error:
                print(f"⚠️ Failed to cleanup blob: {cleanup_error}")

            return {
                "success": False,
                "error": error_msg,
                "details": "MongoDB connection not available. Please contact administrator."
            }

        try:
            # Attempt MongoDB insertion
            result = self.standards_collection.insert_one(document)
            document_id = str(result.inserted_id)
            print(f"✓ Standards document metadata stored in MongoDB: {document_id}")
            print(f"   - Collection: standards")
            print(f"   - User ID: {user_id}")
            print(f"   - Filename: {filename}")
            print(f"   - Blob URL: {blob_info['blob_url']}")
        except Exception as e:
            error_msg = f"Failed to store document metadata in MongoDB: {str(e)}"
            print(f"❌ MongoDB standards insert error: {e}")
            import traceback
            traceback.print_exc()

            # Cleanup: Delete the blob file since metadata storage failed
            try:
                azure_blob_file_manager.delete_file(
                    blob_info['blob_path'],
                    container_name=blob_info.get('container')
                )
                print(f"🗑️ Cleaned up blob file after MongoDB failure: {blob_info['blob_path']}")
            except Exception as cleanup_error:
                print(f"⚠️ Failed to cleanup blob: {cleanup_error}")

            return {
                "success": False,
                "error": error_msg,
                "details": str(e)
            }

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
