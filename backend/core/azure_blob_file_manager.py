"""
Azure Blob File Manager
========================
Handles Azure Blob Storage operations for FILES ONLY.

This separates file storage concerns from data/metadata queries:
- File storage: Azure Blob (this module)
- Metadata queries: MongoDB (mongodb_manager.py)

Usage:
    from core.azure_blob_file_manager import azure_blob_file_manager

    # Upload file
    blob_url = azure_blob_file_manager.upload_file(
        file_bytes=data,
        blob_path="Product-Recommender/files/abc.pdf",
        content_type="application/pdf"
    )

    # Download file
    data = azure_blob_file_manager.download_file("Product-Recommender/files/abc.pdf")

    # Generate SAS URL for secure access
    sas_url = azure_blob_file_manager.generate_sas_url(blob_url, expiry_hours=1)
"""
import os
import uuid
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, BinaryIO, Union
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

# Try to import Azure SDK
try:
    from azure.storage.blob import (
        BlobServiceClient,
        ContentSettings,
        generate_blob_sas,
        BlobSasPermissions
    )
    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False
    print("⚠️ azure-storage-blob not installed. Run: pip install azure-storage-blob>=12.19.0")


class AzureBlobFileManager:
    """
    Manages Azure Blob Storage for FILE operations only.

    Features:
    - Lazy initialization
    - Thread-safe singleton
    - SAS URL generation
    - GridFS-compatible interface
    - Container auto-creation
    """

    _instance: Optional['AzureBlobFileManager'] = None
    _lock = threading.Lock()

    # Container configuration
    CONTAINERS = {
        'product_images': 'product-images',
        'generic_images': 'generic-images',
        'vendor_logos': 'vendor-logos',
        'strategy_documents': 'strategy-documents',
        'standards_documents': 'standards-documents',
        'user_projects': 'user-projects',
        'product_documents': 'product-documents',
        'files': 'files',  # General file storage
        'default': 'product-documents'  # Main container
    }

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._connected = False
            self._connection_error = None
            self._blob_service_client = None
            self._container_clients = {}

            # Configuration from environment
            self.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            self.account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
            self.account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
            self.container_name = 'product-documents'
            self.base_path = os.getenv("AZURE_BLOB_BASE_PATH", "Product-Recommender")

            # Extract account name from connection string if not set
            if not self.account_name and self.connection_string:
                self._extract_account_info()

    def _extract_account_info(self):
        """Extract account name and key from connection string"""
        if not self.connection_string:
            return

        parts = dict(part.split('=', 1) for part in self.connection_string.split(';') if '=' in part)
        self.account_name = parts.get('AccountName')
        self.account_key = parts.get('AccountKey')

    def _setup_connection(self) -> bool:
        """Initialize Azure Blob connection (lazy)"""
        if not AZURE_SDK_AVAILABLE:
            self._connection_error = "azure-storage-blob not installed"
            return False

        if not self.connection_string:
            self._connection_error = "AZURE_STORAGE_CONNECTION_STRING not set"
            print("⚠️ AZURE_STORAGE_CONNECTION_STRING not set in environment")
            return False

        try:
            print("🔄 Connecting to Azure Blob Storage...")

            self._blob_service_client = BlobServiceClient.from_connection_string(
                self.connection_string,
                connection_timeout=10,
                read_timeout=30
            )

            # Test connection by listing containers
            containers = list(self._blob_service_client.list_containers(max_results=1))
            self._connected = True
            self._connection_error = None

            print(f"✅ Azure Blob Storage connected")
            print(f"   Account: {self.account_name}")
            print(f"   Default container: {self.container_name}")

            return True

        except Exception as e:
            self._connection_error = str(e)
            print(f"❌ Azure Blob connection error: {str(e)}")
            return False

    @property
    def blob_service_client(self):
        """Get blob service client (lazy connection)"""
        if self._blob_service_client is None:
            self._setup_connection()
        return self._blob_service_client

    def _get_container_client(self, container_name: str = None):
        """Get container client, creating container if needed"""
        container_name = container_name or self.container_name

        if container_name not in self._container_clients:
            client = self.blob_service_client
            if client is None:
                return None

            container_client = client.get_container_client(container_name)

            # Create container if it doesn't exist
            try:
                if not container_client.exists():
                    container_client.create_container()
                    print(f"✅ Created container: {container_name}")
            except Exception as e:
                print(f"⚠️ Container check failed: {e}")

            self._container_clients[container_name] = container_client

        return self._container_clients[container_name]

    # ============================================================
    # Core File Operations
    # ============================================================

    def upload_file(
        self,
        file_bytes: bytes,
        blob_path: str,
        content_type: str = 'application/octet-stream',
        metadata: Optional[Dict[str, str]] = None,
        container_name: str = None
    ) -> str:
        """
        Upload file to Azure Blob Storage.

        Args:
            file_bytes: File content as bytes
            blob_path: Full path in blob (e.g., "Product-Recommender/files/abc.pdf")
            content_type: MIME type
            metadata: Additional metadata dict
            container_name: Optional container override

        Returns:
            Blob URL
        """
        container_name = container_name or self.container_name
        container_client = self._get_container_client(container_name)

        if container_client is None:
            raise ConnectionError("Azure Blob Storage not connected")

        blob_client = container_client.get_blob_client(blob_path)

        # Upload with content settings
        blob_client.upload_blob(
            file_bytes,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
            metadata=metadata or {}
        )

        return blob_client.url

    def upload_stream(
        self,
        stream: BinaryIO,
        blob_path: str,
        content_type: str = 'application/octet-stream',
        metadata: Optional[Dict[str, str]] = None,
        container_name: str = None
    ) -> str:
        """Upload file stream to Azure Blob Storage"""
        file_bytes = stream.read()
        return self.upload_file(file_bytes, blob_path, content_type, metadata, container_name)

    def download_file(self, blob_path: str, container_name: str = None) -> bytes:
        """
        Download file from Azure Blob Storage.

        Args:
            blob_path: Path to blob
            container_name: Optional container override

        Returns:
            File content as bytes
        """
        container_name = container_name or self.container_name
        container_client = self._get_container_client(container_name)

        if container_client is None:
            raise ConnectionError("Azure Blob Storage not connected")

        blob_client = container_client.get_blob_client(blob_path)
        return blob_client.download_blob().readall()

    def delete_file(self, blob_path: str, container_name: str = None) -> bool:
        """
        Delete file from Azure Blob Storage.

        Args:
            blob_path: Path to blob
            container_name: Optional container override

        Returns:
            True if deleted successfully
        """
        container_name = container_name or self.container_name
        container_client = self._get_container_client(container_name)

        if container_client is None:
            raise ConnectionError("Azure Blob Storage not connected")

        try:
            blob_client = container_client.get_blob_client(blob_path)
            blob_client.delete_blob()
            return True
        except Exception as e:
            print(f"Error deleting blob: {e}")
            return False

    def file_exists(self, blob_path: str, container_name: str = None) -> bool:
        """Check if file exists in Azure Blob Storage"""
        container_name = container_name or self.container_name
        container_client = self._get_container_client(container_name)

        if container_client is None:
            return False

        try:
            blob_client = container_client.get_blob_client(blob_path)
            return blob_client.exists()
        except Exception:
            return False

    def list_files(
        self,
        prefix: str = None,
        container_name: str = None,
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List files in Azure Blob Storage.

        Args:
            prefix: Filter by path prefix
            container_name: Optional container override
            include_metadata: Include blob metadata in results

        Returns:
            List of blob info dicts
        """
        container_name = container_name or self.container_name
        container_client = self._get_container_client(container_name)

        if container_client is None:
            return []

        include = ['metadata'] if include_metadata else []
        blobs = container_client.list_blobs(name_starts_with=prefix, include=include)

        results = []
        for blob in blobs:
            info = {
                'name': blob.name,
                'size': blob.size,
                'content_type': blob.content_settings.content_type if blob.content_settings else None,
                'last_modified': blob.last_modified,
                'url': f"https://{self.account_name}.blob.core.windows.net/{container_name}/{blob.name}"
            }
            if include_metadata:
                info['metadata'] = blob.metadata or {}
            results.append(info)

        return results

    # ============================================================
    # SAS URL Generation
    # ============================================================

    def generate_sas_url(
        self,
        blob_url: str,
        expiry_hours: int = 1,
        permissions: str = 'r'
    ) -> Optional[str]:
        """
        Generate time-limited SAS URL for secure blob access.

        Args:
            blob_url: Full blob URL
            expiry_hours: Hours until SAS token expires (default: 1)
            permissions: SAS permissions (default: 'r' for read)

        Returns:
            URL with SAS token appended, or None on error
        """
        if not self.account_name or not self.account_key:
            print("⚠️ Cannot generate SAS URL: account credentials not available")
            return None

        try:
            # Parse blob URL to extract container and blob name
            parsed_url = urlparse(blob_url)
            path_parts = parsed_url.path.lstrip('/').split('/', 1)

            if len(path_parts) < 2:
                return None

            container_name = path_parts[0]
            blob_name = path_parts[1]

            # Build permissions
            sas_permissions = BlobSasPermissions(
                read='r' in permissions,
                write='w' in permissions,
                delete='d' in permissions,
                list='l' in permissions
            )

            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=self.account_name,
                container_name=container_name,
                blob_name=blob_name,
                account_key=self.account_key,
                permission=sas_permissions,
                expiry=datetime.utcnow() + timedelta(hours=expiry_hours)
            )

            return f"{blob_url}?{sas_token}"

        except Exception as e:
            print(f"Error generating SAS URL: {str(e)}")
            return None

    def generate_sas_url_for_path(
        self,
        blob_path: str,
        container_name: str = None,
        expiry_hours: int = 1
    ) -> Optional[str]:
        """Generate SAS URL from blob path (not full URL)"""
        container_name = container_name or self.container_name
        blob_url = f"https://{self.account_name}.blob.core.windows.net/{container_name}/{blob_path}"
        return self.generate_sas_url(blob_url, expiry_hours)

    # ============================================================
    # GridFS-Compatible Interface
    # ============================================================

    def put(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str = 'application/octet-stream',
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Upload file with GridFS-compatible interface.

        Args:
            file_bytes: File content
            filename: Original filename
            content_type: MIME type
            metadata: Additional metadata

        Returns:
            file_id (UUID string)
        """
        file_id = str(uuid.uuid4())
        blob_path = f"{self.base_path}/files/{file_id}_{filename}"

        full_metadata = metadata or {}
        full_metadata.update({
            'file_id': file_id,
            'filename': filename,
            'upload_date': datetime.utcnow().isoformat(),
            'content_type': content_type,
            'file_size': str(len(file_bytes))
        })

        self.upload_file(file_bytes, blob_path, content_type, full_metadata)
        return file_id

    def get(self, file_id: str) -> bytes:
        """
        Download file by ID (GridFS-compatible).

        Args:
            file_id: File ID returned by put()

        Returns:
            File content as bytes

        Raises:
            FileNotFoundError: If file not found
        """
        # Search for blob with file_id prefix
        blobs = self.list_files(
            prefix=f"{self.base_path}/files/{file_id}_",
            include_metadata=True
        )

        if blobs:
            return self.download_file(blobs[0]['name'])

        # Try searching by metadata
        all_files = self.list_files(
            prefix=f"{self.base_path}/files/",
            include_metadata=True
        )

        for blob in all_files:
            if blob.get('metadata', {}).get('file_id') == file_id:
                return self.download_file(blob['name'])

        raise FileNotFoundError(f"File not found: {file_id}")

    def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find file by query (GridFS-compatible).

        Args:
            query: Query dict with metadata fields

        Returns:
            File info dict or None
        """
        files = self.list_files(
            prefix=f"{self.base_path}/files/",
            include_metadata=True
        )

        for file_info in files:
            metadata = file_info.get('metadata', {})

            # Check if all query conditions match
            match = True
            for key, value in query.items():
                if metadata.get(key) != value:
                    match = False
                    break

            if match:
                return {
                    'file_id': metadata.get('file_id'),
                    'filename': metadata.get('filename'),
                    'content_type': metadata.get('content_type'),
                    'file_size': int(metadata.get('file_size', 0)),
                    'upload_date': metadata.get('upload_date'),
                    'blob_path': file_info['name'],
                    'url': file_info['url']
                }

        return None

    # ============================================================
    # Specialized Upload Methods
    # ============================================================

    def upload_strategy_document(
        self,
        file_bytes: bytes,
        filename: str,
        user_id: Union[int, str],
        content_type: str = 'application/pdf'
    ) -> Dict[str, str]:
        """Upload strategy document"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        blob_path = f"{self.base_path}/strategy_documents/user_{user_id}/{timestamp}_{filename}"

        blob_url = self.upload_file(
            file_bytes=file_bytes,
            blob_path=blob_path,
            content_type=content_type,
            metadata={
                'user_id': str(user_id),
                'filename': filename,
                'upload_date': datetime.utcnow().isoformat()
            }
        )

        return {
            "storage": "azure_blob",
            "container": self.container_name,
            "blob_path": blob_path,
            "blob_url": blob_url
        }

    def upload_standards_document(
        self,
        file_bytes: bytes,
        filename: str,
        user_id: Union[int, str],
        content_type: str = 'application/pdf'
    ) -> Dict[str, str]:
        """Upload standards document"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        blob_path = f"{self.base_path}/standards_documents/user_{user_id}/{timestamp}_{filename}"

        blob_url = self.upload_file(
            file_bytes=file_bytes,
            blob_path=blob_path,
            content_type=content_type,
            metadata={
                'user_id': str(user_id),
                'filename': filename,
                'upload_date': datetime.utcnow().isoformat()
            }
        )

        return {
            "storage": "azure_blob",
            "container": self.container_name,
            "blob_path": blob_path,
            "blob_url": blob_url
        }

    def upload_project_data(
        self,
        project_json_bytes: bytes,
        user_id: str,
        project_id: str
    ) -> Dict[str, str]:
        """Upload project JSON data"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        blob_path = f"{self.base_path}/user_projects/{user_id}/{project_id}_{timestamp}.json"

        blob_url = self.upload_file(
            file_bytes=project_json_bytes,
            blob_path=blob_path,
            content_type='application/json',
            metadata={
                'user_id': user_id,
                'project_id': project_id,
                'upload_date': datetime.utcnow().isoformat()
            }
        )

        return {
            "storage": "azure_blob",
            "container": self.container_name,
            "blob_path": blob_path,
            "blob_url": blob_url
        }

    def upload_product_image(
        self,
        image_bytes: bytes,
        product_type: str,
        vendor_name: str,
        model_family: str
    ) -> Dict[str, str]:
        """Upload product image"""
        product_normalized = product_type.lower().replace(' ', '_')
        vendor_normalized = vendor_name.lower().replace(' ', '_')
        model_normalized = model_family.lower().replace(' ', '_')

        blob_path = f"{self.base_path}/images/{product_normalized}/{vendor_normalized}/{model_normalized}.jpeg"

        blob_url = self.upload_file(
            file_bytes=image_bytes,
            blob_path=blob_path,
            content_type='image/jpeg',
            metadata={
                'product_type': product_type,
                'vendor_name': vendor_name,
                'model_family': model_family,
                'upload_date': datetime.utcnow().isoformat()
            }
        )

        return {
            "storage": "azure_blob",
            "container": self.container_name,
            "blob_path": blob_path,
            "blob_url": blob_url
        }

    def upload_generic_image(
        self,
        image_bytes: bytes,
        product_type: str
    ) -> Dict[str, str]:
        """Upload generic product image"""
        product_normalized = product_type.lower().replace(' ', '_')
        blob_path = f"{self.base_path}/generic_images/generic_{product_normalized}.png"

        blob_url = self.upload_file(
            file_bytes=image_bytes,
            blob_path=blob_path,
            content_type='image/png',
            metadata={
                'product_type': product_type,
                'upload_date': datetime.utcnow().isoformat()
            }
        )

        return {
            "storage": "azure_blob",
            "container": self.container_name,
            "blob_path": blob_path,
            "blob_url": blob_url
        }

    # ============================================================
    # Health Check
    # ============================================================

    def is_connected(self) -> bool:
        """Check if Azure Blob is connected"""
        return self._connected and self._blob_service_client is not None

    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        result = {
            'service': 'Azure Blob Storage',
            'status': 'unknown',
            'connected': False,
            'account': self.account_name,
            'container': self.container_name,
            'error': None
        }

        if not AZURE_SDK_AVAILABLE:
            result['status'] = 'error'
            result['error'] = 'azure-storage-blob not installed'
            return result

        if not self.connection_string:
            result['status'] = 'error'
            result['error'] = 'AZURE_STORAGE_CONNECTION_STRING not configured'
            return result

        try:
            if self.is_connected() or self._setup_connection():
                result['status'] = 'healthy'
                result['connected'] = True
            else:
                result['status'] = 'disconnected'
                result['error'] = self._connection_error
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)

        return result


# Singleton instance
azure_blob_file_manager = AzureBlobFileManager()


# ============================================================
# Module-level utilities
# ============================================================

def get_blob_url(blob_path: str, container_name: str = None) -> str:
    """Construct full blob URL from path"""
    manager = azure_blob_file_manager
    container = container_name or manager.container_name
    return f"https://{manager.account_name}.blob.core.windows.net/{container}/{blob_path}"
