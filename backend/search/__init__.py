# search/__init__.py
"""
Search Deep Agent - Product Search Workflow
============================================

Pure LangGraph StateGraph deep agent for product search.

Entry points
------------
run_product_search_workflow()            -- full 7-step pipeline (auto or HITL) with thread locking
run_product_search_workflow_stream()     -- streaming variant with progress callbacks
run_single_product_workflow()            -- alias for auto-mode full pipeline
run_analysis_only()                     -- skip validation; start at vendor analysis
run_validation_only()                   -- run only the validate step
run_advanced_params_only()              -- run only the discover_params step
process_from_instrument_identifier()    -- batch, called from instrument identifier
process_from_solution_workflow()        -- batch, called from solution workflow
product_search_workflow()               -- top-level convenience wrapper
resume_product_search_workflow()        -- resume a HITL-interrupted graph

Utility / schema functions
--------------------------
generate_comparison_table()            -- build UI comparison table from ranking
get_schema_only()                      -- load/generate schema without validation
validate_with_schema()                 -- validate input against provided schema
validate_multiple_products_parallel()  -- parallel schema gen for N products
enrich_schema_parallel()               -- parallel standards RAG enrichment
get_or_generate_schema_async()         -- async complete schema lifecycle
get_or_generate_schemas_batch_async()  -- async batch schema lifecycle

State / factory
---------------
SearchDeepAgentState                   -- TypedDict for the graph state
create_search_deep_agent_state         -- factory function

Workflow
--------
create_search_workflow_graph           -- create compiled LangGraph app
create_product_search_graph            -- alias for backward compatibility
"""

# =============================================================================
# CORE ENTRY POINTS
# =============================================================================

from .entry_points import (
    run_product_search_workflow,
    run_product_search_workflow_stream,
    run_single_product_workflow,
    run_analysis_only,
    run_validation_only,
    run_advanced_params_only,
    process_from_instrument_identifier,
    process_from_solution_workflow,
    product_search_workflow,
    resume_product_search_workflow,
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

from .entry_points import (
    generate_comparison_table,
    get_schema_only,
    validate_with_schema,
    validate_multiple_products_parallel,
    enrich_schema_parallel,
    get_or_generate_schema_async,
    get_or_generate_schemas_batch_async,
)

# =============================================================================
# STATE & FACTORY
# =============================================================================

from .state import (
    SearchDeepAgentState,
    create_search_deep_agent_state,
    # Backward compatibility aliases
    ProductSearchDeepAgentState,
    create_product_search_deep_agent_state,
)

# =============================================================================
# WORKFLOW
# =============================================================================

from .workflow import (
    create_search_workflow_graph,
    create_product_search_graph,  # Backward compatibility alias
)

# =============================================================================
# BACKWARD-COMPATIBLE CLASS WRAPPERS
# =============================================================================

from .compat import (
    ProductSearchWorkflow,
    ValidationTool,
    AdvancedParametersTool,
    VendorAnalysisTool,
    RankingTool,
    analyze_vendors,
    rank_products,
)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Core entry points
    "run_product_search_workflow",
    "run_product_search_workflow_stream",
    "run_single_product_workflow",
    "run_analysis_only",
    "run_validation_only",
    "run_advanced_params_only",
    "process_from_instrument_identifier",
    "process_from_solution_workflow",
    "product_search_workflow",
    "resume_product_search_workflow",
    # Utility functions
    "generate_comparison_table",
    "get_schema_only",
    "validate_with_schema",
    "validate_multiple_products_parallel",
    "enrich_schema_parallel",
    "get_or_generate_schema_async",
    "get_or_generate_schemas_batch_async",
    # State
    "SearchDeepAgentState",
    "create_search_deep_agent_state",
    "ProductSearchDeepAgentState",  # Backward compat
    "create_product_search_deep_agent_state",  # Backward compat
    # Workflow
    "create_search_workflow_graph",
    "create_product_search_graph",  # Backward compat
    # Backward-compatible class wrappers
    "ProductSearchWorkflow",
    "ValidationTool",
    "AdvancedParametersTool",
    "VendorAnalysisTool",
    "RankingTool",
    "analyze_vendors",
    "rank_products",
]
