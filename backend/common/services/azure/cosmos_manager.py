"""
Hybrid Project Management Module
================================
Uses MongoDB for metadata (fast queries) + Azure Blob for full project JSON.
 
Storage Strategy:
- MongoDB: Lightweight metadata for project listing, search, and filtering
- Azure Blob: Complete project JSON with all details, images URLs, etc.
- Images: Already in respective containers (product-images, generic-images, vendor-logos)
  Only URLs are stored in project data - no duplicate uploads
 
Structure:
- MongoDB collection: user_projects (metadata only)
- Blob path: user-projects/{user_id}/{project_id}.json (full data)
"""
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import uuid
import urllib.parse
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import ContentSettings, BlobClient
 
from common.core.azure_blob_file_manager import azure_blob_file_manager
from common.core.mongodb_manager import mongodb_manager
 
class CosmosProjectManager:
    """
    Hybrid Project Manager - MongoDB + Azure Blob Storage
 
    Architecture:
    - MongoDB: Stores lightweight metadata for fast queries
      Collection: user_projects
      Fields: project_id, user_id, project_name, product_type, counts, timestamps
 
    - Azure Blob: Stores complete project JSON
      Container: user-projects
      Path: {user_id}/{project_id}.json
      Content: Full project data including all fields, images URLs, etc.
 
    - Images: Already stored in their respective blob containers
      Containers: product-images, generic-images, vendor-logos
      Project data only stores the blob URLs (no duplicate uploads)
    """
 
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.blob_manager = azure_blob_file_manager
        self.mongodb = mongodb_manager
        self.projects_collection = None
 
    def _get_projects_collection(self):
        """Get MongoDB projects collection (lazy initialization)"""
        if self.projects_collection is None:
            db = self.mongodb.database
            if db is not None:
                self.projects_collection = db['user_projects']
                self.logger.info("✅ MongoDB user_projects collection connected")
 
                # Verify we can actually write to the collection
                try:
                    # Test write permission
                    self.projects_collection.count_documents({}, limit=1)
                    self.logger.info("✅ MongoDB collection is readable")
                except Exception as test_error:
                    self.logger.error(f"❌ MongoDB collection access test failed: {test_error}")
            else:
                self.logger.warning("⚠️ MongoDB not available - metadata queries will be limited")
                self.logger.warning(f"   MONGODB_URI: {os.getenv('MONGODB_URI', 'NOT SET')[:50]}...")
        return self.projects_collection
 
    def check_mongodb_health(self) -> dict:
        """Check MongoDB connection health and return status"""
        health_info = {
            'connected': False,
            'database': None,
            'collection': None,
            'can_read': False,
            'can_write': False,
            'error': None
        }
 
        try:
            # Check if MongoDB manager is connected
            if self.mongodb.is_connected():
                health_info['connected'] = True
                health_info['database'] = self.mongodb._db_name
 
                # Try to get collection
                collection = self._get_projects_collection()
                if collection is not None:
                    health_info['collection'] = 'user_projects'
 
                    # Test read
                    try:
                        count = collection.count_documents({})
                        health_info['can_read'] = True
                        health_info['document_count'] = count
                    except Exception as read_error:
                        health_info['error'] = f"Read failed: {str(read_error)}"
 
                    # Test write (insert and immediately delete a test document)
                    try:
                        test_id = f"health_check_{uuid.uuid4()}"
                        test_doc = {'_id': test_id, 'test': True}
                        collection.insert_one(test_doc)
                        collection.delete_one({'_id': test_id})
                        health_info['can_write'] = True
                    except Exception as write_error:
                        health_info['error'] = f"Write failed: {str(write_error)}"
                else:
                    health_info['error'] = "Collection is None"
            else:
                health_info['error'] = "MongoDB not connected"
 
        except Exception as e:
            health_info['error'] = str(e)
 
        return health_info
 
    @property
    def container_client(self):
        """Get the user-projects container client for blob storage"""
        container_name = self.blob_manager.CONTAINERS['user_projects']
        return self.blob_manager._get_container_client(container_name)
 
    @property
    def base_path(self):
        """Base path is empty since we're using dedicated container"""
        return ""
 
    def _get_project_blob_path(self, user_id: str, project_id: str) -> str:
        """Construct the blob path: {user_id}/{project_id}.json (no base path needed)"""
        # Ensure user_id is clean
        safe_user_id = str(user_id).strip()
        safe_project_id = str(project_id).strip()
        # No base_path prefix since we're using dedicated user-projects container
        if self.base_path:
            return f"{self.base_path}/{safe_user_id}/{safe_project_id}.json"
        else:
            return f"{safe_user_id}/{safe_project_id}.json"
 
    def save_project(self, user_id: str, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save project using hybrid approach:
        1. Save metadata to MongoDB (fast queries)
        2. Save full JSON to Azure Blob (complete data)
        3. Images already in their containers - only URLs stored
        """
        try:
            current_time = datetime.utcnow().isoformat()
 
            project_id = project_data.get('project_id')
            if not project_id:
                project_id = str(uuid.uuid4())
                is_new = True
            else:
                is_new = False
 
            # Ensure we have essential metadata
            project_name = (project_data.get('project_name') or 'Untitled Project').strip()
            detected_product_type = project_data.get('detected_product_type')
            incoming_product_type = (project_data.get('product_type') or '').strip()
 
            if detected_product_type:
                product_type = detected_product_type.strip()
            else:
                if incoming_product_type and project_name and incoming_product_type.lower() == project_name.lower():
                    product_type = ''
                else:
                    product_type = incoming_product_type
 
            # Calculate counts for metadata
            instruments_list = project_data.get('identified_instruments', [])
            accessories_list = project_data.get('identified_accessories', [])
            search_tabs = project_data.get('search_tabs', [])
            conversations_count = project_data.get('user_interactions', {}).get('conversations_count', 0)
 
            # MongoDB metadata document (lightweight for fast queries)
            mongodb_metadata = {
                '_id': project_id,  # Use project_id as MongoDB _id for easy lookup
                'project_id': project_id,
                'user_id': str(user_id),
                'project_name': project_name,
                'product_type': product_type,
                'project_description': project_data.get('project_description', ''),
                'project_status': 'active',
                'created_at': project_data.get('created_at', current_time),
                'updated_at': current_time,
                'instruments_count': len(instruments_list),
                'accessories_count': len(accessories_list),
                'search_tabs_count': len(search_tabs),
                'conversations_count': conversations_count,
                'blob_path': f"{user_id}/{project_id}.json",  # Reference to blob location
                'storage_format': 'hybrid_v1'
            }
 
            # Azure Blob metadata (for backward compatibility if needed)
            blob_metadata = {
                'user_id': str(user_id),
                'project_id': project_id,
                'project_name': urllib.parse.quote(project_name),
                'product_type': urllib.parse.quote(product_type),
                'project_status': 'active',
                'created_at': project_data.get('created_at', current_time),
                'updated_at': current_time,
                'instruments_count': str(len(instruments_list)),
                'accessories_count': str(len(accessories_list)),
                'search_tabs_count': str(len(search_tabs)),
                'conversations_count': str(conversations_count)
            }
 
            # Prepare complete project data (for Blob storage)
            # Images are already in their containers - only URLs are stored here
            complete_project_data = {
                'id': project_id,  # Compatibility
                'project_id': project_id,
                'user_id': str(user_id),
                'project_name': project_name,
                'project_description': project_data.get('project_description', ''),
                'initial_requirements': project_data.get('initial_requirements', ''),
                'product_type': product_type,
                'pricing': project_data.get('pricing', {}),
                'identified_instruments': instruments_list,  # Contains image URLs
                'identified_accessories': accessories_list,  # Contains image URLs
                'search_tabs': search_tabs,
                'conversation_histories': project_data.get('conversation_histories', {}),
                'collected_data': project_data.get('collected_data', {}),
                'generic_images': project_data.get('generic_images', {}),  # Image URLs only
                'feedback_entries': project_data.get('feedback_entries', project_data.get('feedback', [])),
                'current_step': project_data.get('current_step', ''),
                'active_tab': project_data.get('active_tab', ''),
                'analysis_results': project_data.get('analysis_results', {}),
                'field_descriptions': project_data.get('field_descriptions', {}),
                'workflow_position': project_data.get('workflow_position', {}),
                'workflow_state': project_data.get('workflow_state', {}),  # Includes right_panel_tab
                'user_interactions': project_data.get('user_interactions', {}),
                'embedded_media': project_data.get('embedded_media', {}),
                'project_metadata': {
                    'schema_version': '4.0',
                    'storage_format': 'hybrid_mongodb_blob',
                    'last_updated_by': 'ai_product_recommender_system'
                },
                'created_at': blob_metadata['created_at'],
                'updated_at': blob_metadata['updated_at'],
                'project_status': 'active'
            }
 
            # 1. Save metadata to MongoDB (for fast listing/search)
            projects_collection = self._get_projects_collection()
            if projects_collection is not None:
                try:
                    self.logger.info(f"🔄 Attempting to save metadata to MongoDB for project: {project_id}")
                    self.logger.info(f"   User ID: {user_id}")
                    self.logger.info(f"   Project Name: {project_name}")
 
                    result = projects_collection.replace_one(
                        {'_id': project_id},
                        mongodb_metadata,
                        upsert=True
                    )
 
                    if result.upserted_id:
                        self.logger.info(f"✅ INSERTED new metadata to MongoDB: {project_id}")
                    elif result.modified_count > 0:
                        self.logger.info(f"✅ UPDATED existing metadata in MongoDB: {project_id}")
                    else:
                        self.logger.info(f"✅ Metadata already exists (no changes): {project_id}")
 
                except Exception as mongo_error:
                    self.logger.error(f"❌ MongoDB save FAILED for {project_id}: {str(mongo_error)}")
                    import traceback
                    self.logger.error(f"   Full error: {traceback.format_exc()}")
                    self.logger.warning(f"   Continuing with blob storage only...")
            else:
                self.logger.warning(f"⚠️ MongoDB collection is None - cannot save metadata")
                self.logger.warning(f"   Check MongoDB connection: MONGODB_URI={os.getenv('MONGODB_URI', 'NOT SET')}")
 
            # 2. Save full JSON to Azure Blob (complete project data)
            blob_path = self._get_project_blob_path(user_id, project_id)
            blob_client = self.container_client.get_blob_client(blob_path)
 
            blob_client.upload_blob(
                json.dumps(complete_project_data, indent=2),
                overwrite=True,
                metadata=blob_metadata,
                content_settings=ContentSettings(content_type='application/json')
            )
 
            self.logger.info(f"✅ Saved full project to Blob: {blob_path}")
 
            # Return structure compatible with frontend
            return {
                'project_id': project_id,
                'project_name': project_name,
                'project_description': complete_project_data['project_description'],
                'product_type': product_type,
                'pricing': complete_project_data['pricing'],
                'feedback_entries': complete_project_data['feedback_entries'],
                'created_at': complete_project_data['created_at'],
                'updated_at': complete_project_data['updated_at'],
                'project_status': 'active'
            }
           
        except Exception as e:
            self.logger.error(f"Failed to save project for user {user_id}: {e}")
            raise
 
    def get_user_projects(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all projects for a user from MongoDB (fast metadata queries).
        Falls back to blob listing if MongoDB unavailable.
        """
        try:
            # Try MongoDB first (fast)
            projects_collection = self._get_projects_collection()
            if projects_collection is not None:
                try:
                    cursor = projects_collection.find(
                        {'user_id': str(user_id), 'project_status': 'active'},
                        projection={
                            '_id': 0,
                            'project_id': 1,
                            'project_name': 1,
                            'product_type': 1,
                            'project_description': 1,
                            'instruments_count': 1,
                            'accessories_count': 1,
                            'search_tabs_count': 1,
                            'conversations_count': 1,
                            'project_status': 1,
                            'created_at': 1,
                            'updated_at': 1
                        }
                    ).sort('updated_at', -1)
 
                    project_list = []
                    for doc in cursor:
                        project_summary = {
                            'id': doc.get('project_id'),
                            'project_name': doc.get('project_name', 'Untitled'),
                            'product_type': doc.get('product_type', ''),
                            'project_description': doc.get('project_description', ''),
                            'instruments_count': doc.get('instruments_count', 0),
                            'accessories_count': doc.get('accessories_count', 0),
                            'search_tabs_count': doc.get('search_tabs_count', 0),
                            'conversations_count': doc.get('conversations_count', 0),
                            'project_status': doc.get('project_status', 'active'),
                            'created_at': doc.get('created_at'),
                            'updated_at': doc.get('updated_at'),
                            # Optional fields
                            'project_phase': 'unknown',
                            'has_analysis': doc.get('instruments_count', 0) > 0,
                            'requirements_preview': ''
                        }
                        project_list.append(project_summary)
 
                    self.logger.info(f"✅ Fetched {len(project_list)} projects from MongoDB for user {user_id}")
                    return project_list
 
                except Exception as mongo_error:
                    self.logger.warning(f"⚠️ MongoDB query failed, falling back to blob: {mongo_error}")
 
            # Fallback: List from blob storage (slower)
            self.logger.info("Using blob storage fallback for project listing")
            if self.base_path:
                prefix = f"{self.base_path}/{user_id}/"
            else:
                prefix = f"{user_id}/"
 
            blobs = self.container_client.list_blobs(
                name_starts_with=prefix,
                include=['metadata']
            )
 
            project_list = []
            for blob in blobs:
                if not blob.name.endswith('.json'):
                    continue
 
                meta = blob.metadata or {}
                p_name = urllib.parse.unquote(meta.get('project_name', 'Untitled'))
                p_type = urllib.parse.unquote(meta.get('product_type', ''))
 
                project_summary = {
                    'id': meta.get('project_id') or blob.name.split('/')[-1].replace('.json', ''),
                    'project_name': p_name,
                    'product_type': p_type,
                    'instruments_count': int(meta.get('instruments_count', 0)),
                    'accessories_count': int(meta.get('accessories_count', 0)),
                    'search_tabs_count': int(meta.get('search_tabs_count', 0)),
                    'conversations_count': int(meta.get('conversations_count', 0)),
                    'project_status': meta.get('project_status', 'active'),
                    'created_at': meta.get('created_at'),
                    'updated_at': meta.get('updated_at', blob.last_modified.isoformat() if blob.last_modified else None),
                    'project_description': '',
                    'project_phase': 'unknown',
                    'has_analysis': False,
                    'requirements_preview': ''
                }
                project_list.append(project_summary)
 
            project_list.sort(key=lambda x: x.get('updated_at') or '', reverse=True)
            return project_list
 
        except Exception as e:
            self.logger.error(f"Failed to get projects for user {user_id}: {e}")
            return []
 
    def get_project_details(self, project_id: str, user_id: str) -> Dict[str, Any]:
        """Get full project details from Blob"""
        try:
            blob_path = self._get_project_blob_path(user_id, project_id)
            blob_client = self.container_client.get_blob_client(blob_path)
           
            if not blob_client.exists():
                raise ValueError("Project not found")
               
            file_data = blob_client.download_blob().readall()
            project_data = json.loads(file_data.decode('utf-8'))
           
            return project_data
           
        except ResourceNotFoundError:
            raise ValueError("Project not found")
        except Exception as e:
            self.logger.error(f"Failed to get project {project_id}: {e}")
            raise
 
    def append_feedback_to_project(self, project_id: str, user_id: str, feedback_entry: Dict[str, Any]) -> bool:
        """Append feedback"""
        try:
            project_data = self.get_project_details(project_id, user_id)
           
            if 'feedback_entries' not in project_data:
                project_data['feedback_entries'] = []
            project_data['feedback_entries'].append(feedback_entry)
           
            self.save_project(user_id, project_data)
            return True
        except Exception as e:
            self.logger.error(f"Failed to append feedback: {e}")
            raise
 
    def delete_project(self, project_id: str, user_id: str) -> bool:
        """
        Delete project from both MongoDB and Blob storage.
        Images remain in their containers (may be used by other projects).
        """
        try:
            deleted = False
 
            # 1. Delete from MongoDB
            projects_collection = self._get_projects_collection()
            if projects_collection is not None:
                try:
                    result = projects_collection.delete_one({'_id': project_id, 'user_id': str(user_id)})
                    if result.deleted_count > 0:
                        self.logger.info(f"✅ Deleted metadata from MongoDB: {project_id}")
                        deleted = True
                except Exception as mongo_error:
                    self.logger.warning(f"⚠️ MongoDB delete failed: {mongo_error}")
 
            # 2. Delete from Blob storage
            blob_path = self._get_project_blob_path(user_id, project_id)
            blob_client = self.container_client.get_blob_client(blob_path)
 
            if blob_client.exists():
                blob_client.delete_blob()
                self.logger.info(f"✅ Deleted blob: {blob_path}")
                deleted = True
 
            return deleted
 
        except Exception as e:
            self.logger.error(f"Failed to delete project: {e}")
            raise
 
# Global Instance
cosmos_project_manager = CosmosProjectManager()
 
 