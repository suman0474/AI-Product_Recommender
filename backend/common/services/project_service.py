"""
Project Service
================
Service layer for user project management.

Implements hybrid pattern:
- MongoDB: Fast metadata queries (project list, search)
- Azure Blob: Full project JSON storage (load on demand)

Usage:
    from common.services.project_service import project_service

    # Save project
    project_id = project_service.save_project(user_id, project_data)

    # List projects (fast - metadata only)
    projects = project_service.get_user_projects(user_id)

    # Get full project (loads from blob)
    project = project_service.get_project_details(project_id, user_id)
"""
import os
import sys
import json
from typing import Optional, Dict, List, Any
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from bson import ObjectId
    from bson.errors import InvalidId
    BSON_AVAILABLE = True
except ImportError:
    BSON_AVAILABLE = False
    # Fallback ObjectId
    import uuid
    class ObjectId:
        def __init__(self, oid=None):
            self.oid = oid or str(uuid.uuid4()).replace('-', '')[:24]
        def __str__(self):
            return self.oid

from common.core.mongodb_manager import mongodb_manager, is_mongodb_available
from common.core.azure_blob_file_manager import azure_blob_file_manager
from common.core.sas_utils import add_sas_to_document


class ProjectService:
    """
    Manages user projects with hybrid storage.

    Pattern:
    - List views: MongoDB metadata (instant)
    - Detail views: Azure Blob JSON (on demand)
    - Saves: Write to both MongoDB + Blob
    """

    def __init__(self):
        self._collection = None

    @property
    def collection(self):
        """Lazy collection access"""
        if self._collection is None:
            self._collection = mongodb_manager.get_collection('user_projects')
        return self._collection

    def _generate_project_id(self) -> str:
        """Generate new project ID"""
        return str(ObjectId())

    def _validate_project_id(self, project_id: str) -> bool:
        """Validate project ID format"""
        if BSON_AVAILABLE:
            try:
                ObjectId(project_id)
                return True
            except (InvalidId, TypeError):
                return False
        return len(project_id) >= 12

    def save_project(self, user_id: str, project_data: Dict) -> str:
        """
        Save project with hybrid pattern.

        Flow:
        1. Serialize full project to JSON bytes
        2. Upload to Azure Blob
        3. Store metadata in MongoDB

        Args:
            user_id: User ID
            project_data: Complete project data

        Returns:
            Project ID

        Example:
            >>> project_id = project_service.save_project("user123", {
            ...     "project_name": "Refinery Instrumentation",
            ...     "product_type": "Pressure Transmitter",
            ...     "identified_instruments": [...],
            ... })
        """
        # Generate or get project ID
        project_id = project_data.get('project_id') or project_data.get('id') or self._generate_project_id()

        # Ensure timestamps
        now = datetime.utcnow()
        if 'created_at' not in project_data:
            project_data['created_at'] = now.isoformat()
        project_data['updated_at'] = now.isoformat()
        project_data['project_id'] = project_id

        # 1. Serialize full project to JSON bytes
        project_bytes = json.dumps(project_data, default=str).encode('utf-8')

        # 2. Upload to Azure Blob
        blob_info = azure_blob_file_manager.upload_project_data(
            project_json_bytes=project_bytes,
            user_id=str(user_id),
            project_id=project_id
        )

        # 3. Store metadata in MongoDB
        if self.collection:
            metadata_doc = {
                'user_id': str(user_id),
                'project_name': project_data.get('project_name'),
                'project_description': project_data.get('project_description', ''),
                'product_type': project_data.get('product_type'),
                'initial_requirements': project_data.get('initial_requirements', ''),
                'blob_info': blob_info,
                'storage_format': 'azure_blob',
                'project_status': project_data.get('project_status', 'active'),
                'instruments_count': len(project_data.get('identified_instruments', [])),
                'accessories_count': len(project_data.get('identified_accessories', [])),
                'search_tabs_count': len(project_data.get('search_tabs', [])),
                'created_at': project_data.get('created_at'),
                'updated_at': now
            }

            try:
                self.collection.update_one(
                    {'_id': ObjectId(project_id)},
                    {'$set': metadata_doc},
                    upsert=True
                )
            except Exception as e:
                print(f"MongoDB save error (falling back to blob-only): {e}")

        return project_id

    def get_user_projects(
        self,
        user_id: str,
        status: Optional[str] = None,
        product_type: Optional[str] = None,
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict]:
        """
        Get list of user projects (metadata only - fast).

        This is the main performance improvement - no blob loading!

        Args:
            user_id: User ID
            status: Filter by project status (optional)
            product_type: Filter by product type (optional)
            limit: Maximum results
            skip: Number of results to skip (pagination)

        Returns:
            List of project metadata (no full data)

        Example:
            >>> projects = project_service.get_user_projects("user123")
            >>> for p in projects:
            ...     print(f"{p['project_name']}: {p['instruments_count']} instruments")
        """
        if not self.collection:
            return self._get_user_projects_fallback(user_id)

        # Build query
        query = {'user_id': str(user_id)}

        if status:
            query['project_status'] = status

        if product_type:
            query['product_type'] = {'$regex': product_type, '$options': 'i'}

        # Execute query with sorting and pagination
        cursor = self.collection.find(query).sort(
            'updated_at', -1
        ).skip(skip).limit(limit)

        projects = []
        for doc in cursor:
            doc['_id'] = str(doc['_id'])
            doc['project_id'] = doc['_id']
            projects.append(doc)

        return projects

    def _get_user_projects_fallback(self, user_id: str) -> List[Dict]:
        """Fallback to Azure Blob (cosmos_manager pattern)"""
        try:
            from common.services.azure.cosmos_manager import cosmos_project_manager
            return cosmos_project_manager.get_user_projects(user_id)
        except Exception as e:
            print(f"Project list fallback error: {e}")
            return []

    def get_project_details(
        self,
        project_id: str,
        user_id: str
    ) -> Optional[Dict]:
        """
        Get full project details (loads from Azure Blob).

        This loads the complete project data on demand.

        Args:
            project_id: Project ID
            user_id: User ID (for security check)

        Returns:
            Complete project data or None

        Example:
            >>> project = project_service.get_project_details("abc123", "user123")
            >>> print(project['identified_instruments'])
            [{"name": "PT-001", "specs": {...}}, ...]
        """
        # First try MongoDB metadata
        if self.collection:
            try:
                project_meta = self.collection.find_one({
                    '_id': ObjectId(project_id),
                    'user_id': str(user_id)
                })

                if project_meta and project_meta.get('storage_format') == 'azure_blob':
                    # Load full data from Azure Blob
                    blob_info = project_meta.get('blob_info', {})
                    blob_path = blob_info.get('blob_path')

                    if blob_path:
                        container_name = blob_info.get('container')
                        blob_data = azure_blob_file_manager.download_file(blob_path, container_name=container_name)
                        project_data = json.loads(blob_data.decode('utf-8'))

                        # Add metadata fields
                        project_data['project_id'] = str(project_meta['_id'])
                        project_data['created_at'] = project_meta.get('created_at')
                        project_data['updated_at'] = project_meta.get('updated_at')

                        return project_data
            except Exception as e:
                print(f"MongoDB project load error: {e}")

        # Fallback to blob-only storage
        return self._get_project_details_fallback(project_id, user_id)

    def _get_project_details_fallback(
        self,
        project_id: str,
        user_id: str
    ) -> Optional[Dict]:
        """Fallback to Azure Blob (cosmos_manager pattern)"""
        try:
            from common.services.azure.cosmos_manager import cosmos_project_manager
            return cosmos_project_manager.get_project_details(project_id, user_id)
        except Exception as e:
            print(f"Project details fallback error: {e}")
            return None

    def delete_project(self, project_id: str, user_id: str) -> bool:
        """
        Delete project (both metadata and blob).

        Args:
            project_id: Project ID
            user_id: User ID (for security check)

        Returns:
            True if deleted

        Example:
            >>> success = project_service.delete_project("abc123", "user123")
        """
        # Get metadata first
        if self.collection:
            try:
                project_meta = self.collection.find_one({
                    '_id': ObjectId(project_id),
                    'user_id': str(user_id)
                })

                if not project_meta:
                    return False

                # Delete blob
                if project_meta.get('storage_format') == 'azure_blob':
                    blob_info = project_meta.get('blob_info', {})
                    blob_path = blob_info.get('blob_path')
                    container_name = blob_info.get('container')
                    if blob_path:
                        try:
                            azure_blob_file_manager.delete_file(blob_path, container_name=container_name)
                        except Exception as e:
                            print(f"Blob delete error: {e}")

                # Delete metadata
                self.collection.delete_one({'_id': ObjectId(project_id)})
                return True

            except Exception as e:
                print(f"Project delete error: {e}")

        # Fallback
        return self._delete_project_fallback(project_id, user_id)

    def _delete_project_fallback(self, project_id: str, user_id: str) -> bool:
        """Fallback to cosmos_manager"""
        try:
            from common.services.azure.cosmos_manager import cosmos_project_manager
            return cosmos_project_manager.delete_project(project_id, user_id)
        except Exception:
            return False

    def update_project_status(
        self,
        project_id: str,
        user_id: str,
        status: str
    ) -> bool:
        """
        Update project status.

        Args:
            project_id: Project ID
            user_id: User ID
            status: New status ("active", "archived", "deleted")

        Returns:
            True if updated
        """
        if not self.collection:
            return False

        result = self.collection.update_one(
            {'_id': ObjectId(project_id), 'user_id': str(user_id)},
            {
                '$set': {
                    'project_status': status,
                    'updated_at': datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0

    def search_projects(
        self,
        user_id: str,
        query: str,
        limit: int = 20
    ) -> List[Dict]:
        """
        Search user projects by name.

        Args:
            user_id: User ID
            query: Search query
            limit: Max results

        Returns:
            Matching projects (metadata only)
        """
        if not self.collection:
            return []

        cursor = self.collection.find({
            'user_id': str(user_id),
            'project_name': {'$regex': query, '$options': 'i'}
        }).sort('updated_at', -1).limit(limit)

        projects = []
        for doc in cursor:
            doc['_id'] = str(doc['_id'])
            doc['project_id'] = doc['_id']
            projects.append(doc)

        return projects

    def get_project_count(self, user_id: str) -> int:
        """Get total project count for user"""
        if not self.collection:
            return 0
        return self.collection.count_documents({'user_id': str(user_id)})

    def get_recent_projects(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Get most recently updated projects"""
        return self.get_user_projects(user_id, limit=limit)

    def project_exists(self, project_id: str, user_id: str) -> bool:
        """Check if project exists"""
        if not self.collection:
            return False

        count = self.collection.count_documents({
            '_id': ObjectId(project_id),
            'user_id': str(user_id)
        })
        return count > 0

    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        blob_healthy = azure_blob_file_manager.is_connected()

        return {
            'service': 'ProjectService',
            'status': 'healthy' if self.collection and blob_healthy else 'degraded',
            'mongodb_available': is_mongodb_available(),
            'blob_available': blob_healthy
        }


# Singleton instance
project_service = ProjectService()
