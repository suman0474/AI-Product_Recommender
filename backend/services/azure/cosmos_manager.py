"""
Blob Storage Project Management Module
Handles project storage using strictly Azure Blob Storage.
Replaces CosmosDB implementation for pure folder-based storage.
Structure: projects/{user_id}/{project_id}.json
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import uuid
import urllib.parse
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import ContentSettings, BlobClient

from services.azure.blob_utils import azure_blob_file_manager

class CosmosProjectManager:
    """
    Manages project operations using ONLY Azure Blob Storage.
    Note: kept class name 'CosmosProjectManager' to maintain compatibility with main.py imports,
    but internal implementation is pure Blob Storage.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.blob_manager = azure_blob_file_manager
        
    @property
    def container_client(self):
        return self.blob_manager.container_client

    @property
    def base_path(self):
        return self.blob_manager.base_path

    def _get_project_blob_path(self, user_id: str, project_id: str) -> str:
        """Construct the blob path: projects/{user_id}/{project_id}.json"""
        # Ensure user_id is clean
        safe_user_id = str(user_id).strip()
        safe_project_id = str(project_id).strip()
        return f"{self.base_path}/projects/{safe_user_id}/{safe_project_id}.json"

    def save_project(self, user_id: str, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save or update a project to Azure Blob Storage.
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

            # Metadata to store on the Blob (for fast listing)
            # Note: Azure Blob Metadata key nomenclature must be C# valid identifiers (no spaces/weird chars)
            blob_metadata = {
                'user_id': str(user_id),
                'project_id': project_id,
                'project_name': urllib.parse.quote(project_name), # Encode to ensure valid header characters
                'product_type': urllib.parse.quote(product_type),
                'project_status': 'active',
                'created_at': project_data.get('created_at', current_time),
                'updated_at': current_time,
                'instruments_count': str(len(instruments_list)),
                'accessories_count': str(len(accessories_list)),
                'search_tabs_count': str(len(search_tabs)),
                'conversations_count': str(conversations_count)
            }

            # Prepare the full Data content
            complete_project_data = {
                'id': project_id, # Compatibility
                'project_id': project_id,
                'user_id': str(user_id),
                'project_name': project_name,
                'project_description': project_data.get('project_description', ''),
                'initial_requirements': project_data.get('initial_requirements', ''),
                'product_type': product_type,
                'pricing': project_data.get('pricing', {}),
                'identified_instruments': instruments_list,
                'identified_accessories': accessories_list,
                'search_tabs': search_tabs,
                'conversation_histories': project_data.get('conversation_histories', {}),
                'collected_data': project_data.get('collected_data', {}),
                'generic_images': project_data.get('generic_images', {}),
                'feedback_entries': project_data.get('feedback_entries', project_data.get('feedback', [])),
                'current_step': project_data.get('current_step', ''),
                'active_tab': project_data.get('active_tab', ''),
                'analysis_results': project_data.get('analysis_results', {}),
                'field_descriptions': project_data.get('field_descriptions', {}),
                'workflow_position': project_data.get('workflow_position', {}),
                'user_interactions': project_data.get('user_interactions', {}),
                'embedded_media': project_data.get('embedded_media', {}),
                'project_metadata': {
                    'schema_version': '3.0',
                    'storage_format': 'blob_folder',
                    'last_updated_by': 'ai_product_recommender_system'
                },
                'created_at': blob_metadata['created_at'],
                'updated_at': blob_metadata['updated_at'],
                'project_status': 'active'
            }

            # Upload to Blob
            blob_path = self._get_project_blob_path(user_id, project_id)
            blob_client = self.container_client.get_blob_client(blob_path)
            
            blob_client.upload_blob(
                json.dumps(complete_project_data, indent=2),
                overwrite=True,
                metadata=blob_metadata,
                content_settings=ContentSettings(content_type='application/json')
            )
            
            self.logger.info(f"Saved project {project_id} to Blob Storage: {blob_path}")

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
        """Get all projects for a user by listing blobs in their folder"""
        try:
            # Prefix: projects/{user_id}/
            prefix = f"{self.base_path}/projects/{user_id}/"
            
            blobs = self.container_client.list_blobs(
                name_starts_with=prefix,
                include=['metadata']
            )
            
            project_list = []
            for blob in blobs:
                # Basic validation
                if not blob.name.endswith('.json'):
                    continue
                
                meta = blob.metadata or {}
                
                # Retrieve and decode metadata
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
                    # Fields not in metadata but nice to have defaults
                    'project_description': '',
                    'project_phase': 'unknown',
                    'has_analysis': False,
                    'requirements_preview': ''
                }
                project_list.append(project_summary)
            
            # Sort by updated_at desc
            project_list.sort(key=lambda x: x.get('updated_at') or '', reverse=True)
            
            return project_list
            
        except Exception as e:
            self.logger.error(f"Failed to get projects for user {user_id}: {e}")
            # Don't crash, return empty list if container issue
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
        """Delete project blob"""
        try:
            blob_path = self._get_project_blob_path(user_id, project_id)
            blob_client = self.container_client.get_blob_client(blob_path)
            
            if blob_client.exists():
                blob_client.delete_blob()
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete project: {e}")
            raise

# Global Instance
cosmos_project_manager = CosmosProjectManager()
