"""
MongoDB Helper Utilities for File Operations
Provides high-level functions for storing and retrieving files using GridFS
"""

import os
import json
import io
import logging
from typing import Dict, Any, List, Optional, Union, BinaryIO
from datetime import datetime
from bson import ObjectId
from gridfs.errors import NoFile
from pymongo.errors import DuplicateKeyError

from mongodb_config import get_mongodb_connection, Collections

logger = logging.getLogger(__name__)

class MongoDBFileManager:
    """High-level file management for MongoDB with GridFS"""
    
    def __init__(self):
        self.conn = get_mongodb_connection()
        self.gridfs = self.conn['gridfs']
        self.collections = self.conn['collections']
    
    # ==================== UPLOAD OPERATIONS ====================
    
    def upload_to_mongodb(self, file_path_or_data: Union[str, bytes, BinaryIO], 
                         metadata: Dict[str, Any]) -> str:
        """
        Upload file to MongoDB GridFS with metadata
        
        Args:
            file_path_or_data: File path, bytes, or file-like object
            metadata: File metadata including collection_type, product_type, vendor_name, etc.
            
        Returns:
            str: GridFS file ID
        """
        try:
            # Determine file data
            if isinstance(file_path_or_data, str):
                # File path
                with open(file_path_or_data, 'rb') as f:
                    file_data = f.read()
                filename = os.path.basename(file_path_or_data)
            elif isinstance(file_path_or_data, bytes):
                # Bytes data
                file_data = file_path_or_data
                filename = metadata.get('filename', 'unknown_file')
            else:
                # File-like object
                file_data = file_path_or_data.read()
                filename = metadata.get('filename', 'unknown_file')
            
            # Add standard metadata
            upload_metadata = {
                'filename': filename,
                'upload_date': datetime.utcnow(),
                'file_size': len(file_data),
                **metadata
            }
            
            # Upload to GridFS
            file_id = self.gridfs.put(file_data, **upload_metadata)
            
            # Store metadata in appropriate collection
            self._store_file_metadata(str(file_id), upload_metadata)
            
            logger.info(f"Successfully uploaded file to MongoDB: {filename} (ID: {file_id})")
            return str(file_id)
            
        except Exception as e:
            logger.error(f"Failed to upload file to MongoDB: {e}")
            raise
    
    def upload_json_data(self, json_data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """
        Upload JSON data to MongoDB
        
        Args:
            json_data: JSON data to store
            metadata: Metadata including collection_type, product_type, vendor_name, etc.
            
        Returns:
            str: Document ID
        """
        try:
            collection_type = metadata.get('collection_type', 'documents')
            collection = self.collections[collection_type]
            
            # Add standard metadata
            document = {
                'data': json_data,
                'metadata': {
                    'upload_date': datetime.utcnow(),
                    **metadata
                },
                'product_type': metadata.get('product_type'),
                'vendor_name': metadata.get('vendor_name'),
                'file_type': 'json'
            }
            
            # Insert document
            result = collection.insert_one(document)
            doc_id = str(result.inserted_id)
            
            logger.info(f"Successfully uploaded JSON data to MongoDB collection '{collection_type}' (ID: {doc_id})")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to upload JSON data to MongoDB: {e}")
            raise
    
    def _store_file_metadata(self, file_id: str, metadata: Dict[str, Any]):
        """Store file metadata in appropriate metadata collection"""
        try:
            collection_type = metadata.get('collection_type', 'documents')
            metadata_collection_name = f"{collection_type}_metadata"
            
            if metadata_collection_name in self.collections:
                metadata_doc = {
                    'file_id': file_id,
                    'created_at': datetime.utcnow(),
                    **metadata
                }
                
                self.collections[metadata_collection_name].insert_one(metadata_doc)
                
        except Exception as e:
            logger.warning(f"Failed to store file metadata: {e}")
    
    # ==================== RETRIEVAL OPERATIONS ====================
    
    def get_file_from_mongodb(self, collection_name: str, query: Dict[str, Any]) -> Optional[bytes]:
        """
        Retrieve file from MongoDB GridFS
        
        Args:
            collection_name: Collection type (documents, vendors, static, specs)
            query: Query to find the file
            
        Returns:
            bytes: File content or None if not found
        """
        try:
            # Strategy 1: Search GridFS directly (migration stores metadata at top level)
            grid_file = self.gridfs.find_one(query)
            if grid_file:
                return grid_file.read()
            
            # Strategy 2: Try to find in metadata collection
            metadata_collection = self.collections.get(f"{collection_name}_metadata")
            if metadata_collection is not None:
                metadata_doc = metadata_collection.find_one(query)
                if metadata_doc and 'file_id' in metadata_doc:
                    # Handle both string and ObjectId file_id
                    file_id = metadata_doc['file_id']
                    if isinstance(file_id, str):
                        file_id = ObjectId(file_id)
                    
                    grid_file = self.gridfs.get(file_id)
                    return grid_file.read()
            
            # Strategy 3: Enhanced GridFS search with collection_type filter
            if collection_name in ['documents', 'vendors', 'specs']:
                enhanced_query = {**query, 'collection_type': collection_name}
                grid_file = self.gridfs.find_one(enhanced_query)
                if grid_file:
                    return grid_file.read()
            
            return None
            
        except NoFile:
            logger.warning(f"File not found in MongoDB: {query}")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve file from MongoDB: {e}")
            raise
    
    def get_json_data_from_mongodb(self, collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve JSON data from MongoDB collection
        
        Args:
            collection_name: Collection name
            query: Query to find the document
            
        Returns:
            Dict: JSON data or None if not found
        """
        try:
            collection = self.collections.get(collection_name)
            if collection is None:
                logger.error(f"Collection '{collection_name}' not found")
                return None
            
            document = collection.find_one(query)
            if document:
                # Handle both new format (with 'data' field) and legacy format
                if 'data' in document:
                    return document['data']
                else:
                    # Legacy format or direct storage - return the document minus MongoDB metadata
                    return {k: v for k, v in document.items() if k not in ['_id', 'metadata']}
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve JSON data from MongoDB: {e}")
            raise
    
    def get_schema_from_mongodb(self, product_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve schema from MongoDB specs collection
        
        Args:
            product_type: Product type to get schema for
            
        Returns:
            Dict: Schema data or None if not found
        """
        try:
            # Normalize product type for search
            normalized_type = product_type.lower().replace(" ", "").replace("_", "")
            
            # Try multiple search strategies to find schema
            specs_collection = self.collections['specs']
            
            # Strategy 1: Exact match on top-level product_type
            query = {'product_type': product_type}
            schema_doc = specs_collection.find_one(query)
            
            if not schema_doc:
                # Strategy 2: Normalized match on metadata
                query = {'metadata.normalized_product_type': normalized_type}
                schema_doc = specs_collection.find_one(query)
            
            if not schema_doc:
                # Strategy 3: Case-insensitive regex match on multiple fields
                cursor = specs_collection.find({
                    '$or': [
                        {'product_type': {'$regex': product_type, '$options': 'i'}},
                        {'metadata.product_type': {'$regex': product_type, '$options': 'i'}},
                        {'metadata.normalized_product_type': {'$regex': normalized_type, '$options': 'i'}}
                    ]
                })
                
                for doc in cursor:
                    schema_doc = doc
                    break
            
            if schema_doc:
                # Handle both new format (with 'data' field) and legacy format
                if 'data' in schema_doc:
                    return schema_doc['data']
                else:
                    # Legacy format - return document minus MongoDB metadata
                    return {k: v for k, v in schema_doc.items() if k not in ['_id', 'metadata']}
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve schema from MongoDB: {e}")
            raise
    
    # ==================== QUERY OPERATIONS ====================
    
    def list_files(self, collection_name: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List files in a collection with optional filters
        
        Args:
            collection_name: Collection name
            filters: Optional filters to apply
            
        Returns:
            List of file metadata
        """
        try:
            metadata_collection_name = f"{collection_name}_metadata"
            metadata_collection = self.collections.get(metadata_collection_name)
            
            if metadata_collection is None:
                return []
            
            query = filters or {}
            cursor = metadata_collection.find(query)
            
            return list(cursor)
            
        except Exception as e:
            logger.error(f"Failed to list files from MongoDB: {e}")
            return []
    
    def file_exists(self, collection_name: str, query: Dict[str, Any]) -> bool:
        """
        Check if file exists in MongoDB
        
        Args:
            collection_name: Collection name
            query: Query to check
            
        Returns:
            bool: True if file exists
        """
        try:
            metadata_collection_name = f"{collection_name}_metadata"
            metadata_collection = self.collections.get(metadata_collection_name)
            
            if metadata_collection is not None:
                return metadata_collection.find_one(query) is not None
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check file existence in MongoDB: {e}")
            return False
    
    # ==================== DELETE OPERATIONS ====================
    
    def delete_file(self, collection_name: str, query: Dict[str, Any]) -> bool:
        """
        Delete file from MongoDB
        
        Args:
            collection_name: Collection name
            query: Query to find file to delete
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            metadata_collection_name = f"{collection_name}_metadata"
            metadata_collection = self.collections.get(metadata_collection_name)
            
            if metadata_collection is not None:
                metadata_doc = metadata_collection.find_one(query)
                if metadata_doc and 'file_id' in metadata_doc:
                    # Delete from GridFS
                    file_id = ObjectId(metadata_doc['file_id'])
                    self.gridfs.delete(file_id)
                    
                    # Delete metadata
                    metadata_collection.delete_one({'_id': metadata_doc['_id']})
                    
                    logger.info(f"Successfully deleted file from MongoDB: {query}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete file from MongoDB: {e}")
            return False

# Global instance
mongodb_file_manager = MongoDBFileManager()

# ==================== CONVENIENCE FUNCTIONS ====================

def upload_to_mongodb(file_path_or_data: Union[str, bytes, BinaryIO], metadata: Dict[str, Any]) -> str:
    """Convenience function for uploading files"""
    return mongodb_file_manager.upload_to_mongodb(file_path_or_data, metadata)

def get_file_from_mongodb(collection_name: str, query: Dict[str, Any]) -> Optional[bytes]:
    """Convenience function for retrieving files"""
    return mongodb_file_manager.get_file_from_mongodb(collection_name, query)

def get_schema_from_mongodb(product_type: str) -> Optional[Dict[str, Any]]:
    """Convenience function for retrieving schemas"""
    return mongodb_file_manager.get_schema_from_mongodb(product_type)

def upload_json_to_mongodb(json_data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    """Convenience function for uploading JSON data"""
    return mongodb_file_manager.upload_json_data(json_data, metadata)

def get_json_from_mongodb(collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convenience function for retrieving JSON data"""
    return mongodb_file_manager.get_json_data_from_mongodb(collection_name, query)
