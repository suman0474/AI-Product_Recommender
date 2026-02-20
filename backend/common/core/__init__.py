# Agentic Core Module
# Contains base classes and fundamental structures

# Lazy imports to avoid circular dependencies
# Import from submodules only when needed

def get_mongodb_manager():
    from .mongodb_manager import mongodb_manager
    return mongodb_manager

def get_azure_blob_file_manager():
    from .azure_blob_file_manager import azure_blob_file_manager
    return azure_blob_file_manager

# Convenience re-exports (for common usage)
try:
    from .mongodb_manager import mongodb_manager, get_mongodb_connection, is_mongodb_available
    from .azure_blob_file_manager import azure_blob_file_manager
    from .sas_utils import generate_sas_url, add_sas_to_document, add_sas_to_documents
    from .db_indexes import ensure_indexes
except ImportError as e:
    # Handle case where dependencies not yet installed
    print(f"⚠️ Core module import warning: {e}")
    mongodb_manager = None
    azure_blob_file_manager = None

# --- Merged from common.core ---
def __getattr__(name):
    """Lazy import to avoid circular dependencies for agents and nodes"""
    # Original core dependencies
    if name == 'base_agent':
        from .agents import base_agent
        return base_agent
    elif name == 'base_memory':
        from .agents import base_memory
        return base_memory
    elif name == 'base_state':
        from .agents import base_state
        return base_state
    elif name == 'common_nodes':
        from . import common_nodes
        return common_nodes
    
    # Fallback/Error for unknown attributes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'mongodb_manager', 'get_mongodb_connection', 'is_mongodb_available',
    'azure_blob_file_manager',
    'generate_sas_url', 'add_sas_to_document', 'add_sas_to_documents',
    'ensure_indexes',
    'base_agent', 'base_memory', 'base_state', 'common_nodes'
]
