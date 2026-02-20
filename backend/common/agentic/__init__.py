# Agentic AI Module
# LangChain Tools, Agents, and LangGraph Workflow Implementation

# IMPORTANT: This module uses lazy imports to avoid circular dependencies
# Import specific items only when needed, not at module load time

# Core models are safe to import
from .models import (
    WorkflowState,
    IntentType,
    WorkflowStep,
    IntentClassification,
    RequirementValidation,
    VendorMatch,
    VendorAnalysis,
    ProductRanking,
    OverallRanking,
    InstrumentIdentification,
    create_initial_state,
    RequestMode,
    ConstraintContext,
    VendorModelCandidate,
    ScoringBreakdown,
    RankedComparisonProduct,
    ComparisonMatrix,
    RAGQueryResult,
    SpecObject,
    ComparisonType,
    ComparisonInput,
    ComparisonState,
    SolutionState,
    InstrumentIdentifierState,
    PotentialProductIndexState,
    create_comparison_state,
    create_solution_state,
    create_instrument_identifier_state,
    create_potential_product_index_state
)

# Everything else uses lazy imports via __getattr__
__all__ = [
    # Models
    'WorkflowState', 'IntentType', 'WorkflowStep', 'IntentClassification',
    'RequirementValidation', 'VendorMatch', 'VendorAnalysis', 'ProductRanking',
    'OverallRanking', 'InstrumentIdentification', 'create_initial_state',
    'RequestMode', 'ConstraintContext', 'VendorModelCandidate', 'ScoringBreakdown',
    'RankedComparisonProduct', 'ComparisonMatrix', 'RAGQueryResult', 'SpecObject',
    'ComparisonType', 'ComparisonInput', 'ComparisonState', 'SolutionState',
    'InstrumentIdentifierState', 'PotentialProductIndexState',
    'create_comparison_state', 'create_solution_state',
    'create_instrument_identifier_state', 'create_potential_product_index_state',
    
    # API
    'agentic_bp',
    
    # Lazy loaded items listed here for IDE support
    'login_required', 'admin_required', 'optional_auth',
    'api_response', 'handle_errors', 'validate_request_json', 'validate_query_params',
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies at module load time"""
    
    # Auth decorators
    if name in ('login_required', 'admin_required', 'optional_auth'):
        from .infrastructure.utils.auth_decorators import login_required, admin_required, optional_auth
        return globals().setdefault(name, locals()[name])
    
    # API utilities
    if name in ('api_response', 'handle_errors', 'validate_request_json', 'validate_query_params'):
        from .infrastructure.api.utils import api_response, handle_errors, validate_request_json, validate_query_params
        return globals().setdefault(name, locals()[name])
    
    # API Blueprint
    if name == 'agentic_bp':
        from .infrastructure.api.main_api import agentic_bp
        return agentic_bp
    
    # Deep Agent Blueprint
    if name == 'deep_agent_bp':
        from .deep_agent.api import deep_agent_bp
        return deep_agent_bp
    
    # If not found, raise AttributeError
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
