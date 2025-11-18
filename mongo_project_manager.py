"""
MongoDB Project Management Module
Handles project storage and retrieval in MongoDB
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from mongodb_utils import MongoDBFileManager
from bson import ObjectId
import json

class MongoProjectManager:
    """Manages project operations in MongoDB"""
    
    def __init__(self):
        self.file_manager = MongoDBFileManager()
        self.collection_name = 'user_projects'
        self.logger = logging.getLogger(__name__)
    
    def _serialize_project_data(self, data: Any) -> str:
        """Convert project data to JSON string for storage"""
        if data is None:
            return None
        if isinstance(data, (dict, list)):
            return json.dumps(data)
        return str(data)
    
    def _deserialize_project_data(self, data: str) -> Any:
        """Convert JSON string back to Python objects"""
        if data is None:
            return None
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data
        return data
    
    def save_project(self, user_id: str, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save or update a project in MongoDB
        
        Args:
            user_id: ID of the user who owns the project
            project_data: Project data to save
            
        Returns:
            Dictionary with project ID and metadata
        """
        try:
            project_id = project_data.get('project_id')
            current_time = datetime.utcnow()
            
            # Ensure detected product type is used, and avoid accidentally saving project_name as product_type
            project_name = (project_data.get('project_name') or '').strip()
            detected_product_type = project_data.get('detected_product_type')
            incoming_product_type = (project_data.get('product_type') or '').strip()

            if detected_product_type:
                product_type = detected_product_type.strip()
            else:
                # If incoming product_type exactly matches project_name, ignore it
                if incoming_product_type and project_name and incoming_product_type.lower() == project_name.lower():
                    self.logger.warning(f"Incoming product_type matches project_name ('{incoming_product_type}'). Clearing product_type to avoid incorrect storage for user {user_id}.")
                    product_type = ''
                else:
                    product_type = incoming_product_type

            project_doc = {
                'user_id': str(user_id),
                'project_name': project_data.get('project_name', ''),
                'project_description': project_data.get('project_description', ''),
                'initial_requirements': project_data.get('initial_requirements', ''),
                'product_type': product_type,
                'identified_instruments': self._serialize_project_data(project_data.get('identified_instruments')),
                'identified_accessories': self._serialize_project_data(project_data.get('identified_accessories')),
                'search_tabs': self._serialize_project_data(project_data.get('search_tabs')),
                'conversation_histories': self._serialize_project_data(project_data.get('conversation_histories')),
                'collected_data': self._serialize_project_data(project_data.get('collected_data')),
                'current_step': project_data.get('current_step', ''),
                'active_tab': project_data.get('active_tab', ''),
                'analysis_results': self._serialize_project_data(project_data.get('analysis_results')),
                'field_descriptions': self._serialize_project_data(project_data.get('field_descriptions')),
                'workflow_position': self._serialize_project_data(project_data.get('workflow_position')),
                'user_interactions': self._serialize_project_data(project_data.get('user_interactions')),
                'project_metadata': {
                    'schema_version': '1.0',
                    'data_structure_description': 'Complete project state including conversations, analysis, and workflow position',
                    'supported_features': ['multi_tab_search', 'conversation_persistence', 'state_restoration', 'analysis_results'],
                    'last_updated_by': 'ai_product_recommender_system'
                },
                'project_status': 'active',
                'updated_at': current_time
            }
            
            # Log the product_type we plan to save for debugging
            self.logger.info(f"Saving project - detected_product_type='{detected_product_type}' resolved_product_type='{product_type}' for user {user_id}")

            if project_id:
                # Update existing project
                try:
                    object_id = ObjectId(project_id)
                    existing_project = self.file_manager.db[self.collection_name].find_one({
                        '_id': object_id,
                        'user_id': str(user_id)
                    })
                    
                    if not existing_project:
                        raise ValueError("Project not found or access denied")
                    
                    # Update the document
                    result = self.file_manager.db[self.collection_name].update_one(
                        {'_id': object_id, 'user_id': str(user_id)},
                        {'$set': project_doc}
                    )
                    
                    if result.matched_count == 0:
                        raise ValueError("Failed to update project")
                    
                    final_project_id = str(object_id)
                    self.logger.info(f"Updated project {final_project_id} for user {user_id}")
                    # Read back saved document to verify stored product_type
                    try:
                        saved_doc = self.file_manager.db[self.collection_name].find_one({'_id': object_id, 'user_id': str(user_id)})
                        self.logger.info(f"Post-update stored product_type for {final_project_id}: '{saved_doc.get('product_type') if saved_doc else None}'")
                    except Exception as e:
                        self.logger.warning(f"Failed to read back project after update: {e}")
                    
                except (ValueError, Exception) as e:
                    self.logger.error(f"Failed to update project {project_id}: {e}")
                    raise
            else:
                # Create new project
                project_doc['created_at'] = current_time
                
                result = self.file_manager.db[self.collection_name].insert_one(project_doc)
                final_project_id = str(result.inserted_id)
                self.logger.info(f"Created new project {final_project_id} for user {user_id}")
                # Read back saved document to verify stored product_type
                try:
                    saved_doc = self.file_manager.db[self.collection_name].find_one({'_id': result.inserted_id, 'user_id': str(user_id)})
                    self.logger.info(f"Post-insert stored product_type for {final_project_id}: '{saved_doc.get('product_type') if saved_doc else None}'")
                except Exception as e:
                    self.logger.warning(f"Failed to read back project after insert: {e}")
            
            return {
                'project_id': final_project_id,
                'project_name': project_doc['project_name'],
                'project_description': project_doc['project_description'],
                'product_type': project_doc.get('product_type', ''),
                'created_at': project_doc.get('created_at', current_time).isoformat(),
                'updated_at': project_doc['updated_at'].isoformat(),
                'project_status': project_doc['project_status']
            }
            
        except Exception as e:
            self.logger.error(f"Failed to save project for user {user_id}: {e}")
            raise
    
    def get_user_projects(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all projects for a user
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of project summaries
        """
        try:
            projects_cursor = self.file_manager.db[self.collection_name].find({
                'user_id': str(user_id),
                'project_status': 'active'
            }).sort('updated_at', -1)  # Most recently updated first
            
            project_list = []
            for project in projects_cursor:
                # Parse JSON fields safely for summary
                try:
                    identified_instruments = self._deserialize_project_data(project.get('identified_instruments')) or []
                    identified_accessories = self._deserialize_project_data(project.get('identified_accessories')) or []
                    search_tabs = self._deserialize_project_data(project.get('search_tabs')) or []
                except Exception as e:
                    self.logger.warning(f"Failed to parse project {project['_id']} JSON fields: {e}")
                    identified_instruments = []
                    identified_accessories = []
                    search_tabs = []
                
                # Create requirements preview
                requirements = project.get('initial_requirements', '')
                requirements_preview = requirements[:200] + "..." if len(requirements) > 200 else requirements
                
                # Extract workflow position and user interactions
                workflow_position = self._deserialize_project_data(project.get('workflow_position')) or {}
                user_interactions = self._deserialize_project_data(project.get('user_interactions')) or {}
                project_metadata = project.get('project_metadata') or {}
                
                project_summary = {
                    'id': str(project['_id']),
                    'project_name': project.get('project_name', ''),
                    'project_description': project.get('project_description', ''),
                    'product_type': project.get('product_type', ''),
                    'instruments_count': len(identified_instruments) if isinstance(identified_instruments, list) else 0,
                    'accessories_count': len(identified_accessories) if isinstance(identified_accessories, list) else 0,
                    'search_tabs_count': len(search_tabs) if isinstance(search_tabs, list) else 0,
                    'current_step': project.get('current_step', ''),
                    'active_tab': project.get('active_tab', ''),   
                    'project_phase': workflow_position.get('project_phase', 'unknown'),
                    'conversations_count': user_interactions.get('conversations_count', 0),
                    'has_analysis': user_interactions.get('has_analysis', False),
                    'schema_version': project_metadata.get('schema_version', 'unknown'),
                    'project_status': project.get('project_status', 'active'),
                    'created_at': project.get('created_at', datetime.utcnow()).isoformat(),
                    'updated_at': project.get('updated_at', datetime.utcnow()).isoformat(),
                    'requirements_preview': requirements_preview,
                    'field_descriptions_available': bool(project.get('field_descriptions')),
                    'workflow_metadata': {
                        'has_workflow_position': bool(workflow_position),
                        'has_user_interactions': bool(user_interactions),
                        'total_data_fields': len(project.keys())
                    }
                }
                project_list.append(project_summary)
            
            self.logger.info(f"Retrieved {len(project_list)} projects for user {user_id}")
            return project_list
            
        except Exception as e:
            self.logger.error(f"Failed to get projects for user {user_id}: {e}")
            raise
    
    def get_project_details(self, project_id: str, user_id: str) -> Dict[str, Any]:
        """
        Get full project details for loading
        
        Args:
            project_id: ID of the project
            user_id: ID of the user (for security)
            
        Returns:
            Complete project data
        """
        try:
            object_id = ObjectId(project_id)
            project = self.file_manager.db[self.collection_name].find_one({
                '_id': object_id,
                'user_id': str(user_id)
            })
            
            if not project:
                raise ValueError("Project not found or access denied")
            
            # Parse all JSON fields
            project_details = {
                'id': str(project['_id']),
                'project_name': project.get('project_name', ''),
                'project_description': project.get('project_description', ''),
                'initial_requirements': project.get('initial_requirements', ''),
                'product_type': project.get('product_type', ''),
                'identified_instruments': self._deserialize_project_data(project.get('identified_instruments')) or [],
                'identified_accessories': self._deserialize_project_data(project.get('identified_accessories')) or [],
                'search_tabs': self._deserialize_project_data(project.get('search_tabs')) or [],
                'conversation_histories': self._deserialize_project_data(project.get('conversation_histories')) or {},
                'collected_data': self._deserialize_project_data(project.get('collected_data')) or {},
                'current_step': project.get('current_step', ''),
                'active_tab': project.get('active_tab', ''),
                'analysis_results': self._deserialize_project_data(project.get('analysis_results')) or {},
                'field_descriptions': self._deserialize_project_data(project.get('field_descriptions')) or {},
                'workflow_position': self._deserialize_project_data(project.get('workflow_position')) or {},
                'user_interactions': self._deserialize_project_data(project.get('user_interactions')) or {},
                'project_metadata': project.get('project_metadata') or {},
                'project_status': project.get('project_status', 'active'),
                'created_at': project.get('created_at', datetime.utcnow()).isoformat(),
                'updated_at': project.get('updated_at', datetime.utcnow()).isoformat()
            }
            
            self.logger.info(f"Retrieved project {project_id} details for user {user_id}")
            return project_details
            
        except Exception as e:
            self.logger.error(f"Failed to get project {project_id} for user {user_id}: {e}")
            raise
    
    def delete_project(self, project_id: str, user_id: str) -> bool:
        """
        Delete (archive) a project
        
        Args:
            project_id: ID of the project
            user_id: ID of the user (for security)
            
        Returns:
            True if successful
        """
        try:
            object_id = ObjectId(project_id)
            
            result = self.file_manager.db[self.collection_name].update_one(
                {
                    '_id': object_id,
                    'user_id': str(user_id)
                },
                {
                    '$set': {
                        'project_status': 'archived',
                        'updated_at': datetime.utcnow()
                    }
                }
            )
            
            if result.matched_count == 0:
                raise ValueError("Project not found or access denied")
            
            self.logger.info(f"Archived project {project_id} for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete project {project_id} for user {user_id}: {e}")
            raise

# Create global instance
mongo_project_manager = MongoProjectManager()