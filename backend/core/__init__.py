"""
Core Infrastructure Module
===========================
Provides database connections and utilities.

Main Components:
- mongodb_manager: MongoDB connection singleton
- azure_blob_file_manager: Azure Blob file operations
- db_indexes: Database index management
- sas_utils: SAS URL generation

Usage:
    from core.mongodb_manager import mongodb_manager, get_mongodb_connection
    from core.azure_blob_file_manager import azure_blob_file_manager
    from core.sas_utils import generate_sas_url, add_sas_to_document
    from core.db_indexes import ensure_indexes
"""

# Version
__version__ = '1.0.0'

# Lazy imports to avoid circular dependencies
def get_mongodb_manager():
    from core.mongodb_manager import mongodb_manager
    return mongodb_manager

def get_azure_blob_file_manager():
    from core.azure_blob_file_manager import azure_blob_file_manager
    return azure_blob_file_manager

# Convenience re-exports (for common usage)
try:
    from core.mongodb_manager import mongodb_manager, get_mongodb_connection, is_mongodb_available
    from core.azure_blob_file_manager import azure_blob_file_manager
    from core.sas_utils import generate_sas_url, add_sas_to_document, add_sas_to_documents
    from core.db_indexes import ensure_indexes
except ImportError as e:
    # Handle case where dependencies not yet installed
    print(f"⚠️ Core module import warning: {e}")
    mongodb_manager = None
    azure_blob_file_manager = None
