# Agentic Core Module
# Contains base classes and fundamental structures

# Lazy imports to avoid circular dependencies
# Import from submodules only when needed

__all__ = [
    'base_agent',
    'base_memory', 
    'base_state',
    'base_cache',
    'common_nodes'
]

def __getattr__(name):
    """Lazy import to avoid circular dependencies"""
    if name == 'base_agent':
        from .agents import base_agent
        return base_agent
    elif name == 'base_memory':
        from .agents import base_memory
        return base_memory
    elif name == 'base_state':
        from .state import base_state
        return base_state
    elif name == 'base_cache':
        from .state import base_cache
        return base_cache
    elif name == 'common_nodes':
        from . import common_nodes
        return common_nodes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
