"""
Product Search Deep Agent Workflow
====================================

Pure LangGraph StateGraph deep agent for product search.

Entry points
------------
run_product_search_workflow()          -- full 5-step pipeline (auto or HITL)
run_single_product_workflow()          -- alias for auto-mode full pipeline
run_analysis_only()                    -- skip validation; start at vendor analysis
run_validation_only()                  -- run only the validate_product node
run_advanced_params_only()             -- run only the discover_advanced_params node  [P2-A]
process_from_instrument_identifier()   -- batch, called from instrument identifier
process_from_solution_workflow()       -- batch, called from solution workflow
product_search_workflow()              -- top-level convenience wrapper
resume_product_search_workflow()       -- resume a HITL-interrupted graph

Utility / schema functions
--------------------------
generate_comparison_table()            -- build UI comparison table from ranking  [P2-B]
get_schema_only()                      -- load/generate schema without validation  [P2-C]
validate_with_schema()                 -- validate input against provided schema  [P2-C]
validate_multiple_products_parallel()  -- parallel schema gen for N products  [P3-C]
enrich_schema_parallel()               -- parallel standards RAG enrichment  [P3-C]
get_or_generate_schema_async()         -- async complete schema lifecycle  [P3-D]
get_or_generate_schemas_batch_async()  -- async batch schema lifecycle  [P3-D]

State / factory
---------------
ProductSearchDeepAgentState            -- TypedDict for the graph state
create_product_search_deep_agent_state -- factory function
"""

from product_search.product_search_workflow import (
    # Core entry points
    run_product_search_workflow,
    run_single_product_workflow,
    run_analysis_only,
    run_validation_only,
    run_advanced_params_only,                   # P2-A
    process_from_instrument_identifier,
    process_from_solution_workflow,
    product_search_workflow,
    resume_product_search_workflow,
    create_product_search_graph,
    # Utility functions
    generate_comparison_table,                  # P2-B
    get_schema_only,                            # P2-C
    validate_with_schema,                       # P2-C
    validate_multiple_products_parallel,        # P3-C
    enrich_schema_parallel,                     # P3-C
    get_or_generate_schema_async,               # P3-D
    get_or_generate_schemas_batch_async,        # P3-D
)

from common.agentic.models import (
    ProductSearchDeepAgentState,
    create_product_search_deep_agent_state,
)

# Backward-compatible class wrappers (replaces old product_search_workflow package)
from product_search.compat import (
    ProductSearchWorkflow,
    ValidationTool,
    AdvancedParametersTool,
    VendorAnalysisTool,
    RankingTool,
    analyze_vendors,
    rank_products,
)

__all__ = [
    # Core entry points
    "run_product_search_workflow",
    "run_single_product_workflow",
    "run_analysis_only",
    "run_validation_only",
    "run_advanced_params_only",
    "process_from_instrument_identifier",
    "process_from_solution_workflow",
    "product_search_workflow",
    "resume_product_search_workflow",
    "create_product_search_graph",
    # Utility functions
    "generate_comparison_table",
    "get_schema_only",
    "validate_with_schema",
    "validate_multiple_products_parallel",
    "enrich_schema_parallel",
    "get_or_generate_schema_async",
    "get_or_generate_schemas_batch_async",
    # State
    "ProductSearchDeepAgentState",
    "create_product_search_deep_agent_state",
    # Backward-compatible class wrappers
    "ProductSearchWorkflow",
    "ValidationTool",
    "AdvancedParametersTool",
    "VendorAnalysisTool",
    "RankingTool",
    "analyze_vendors",
    "rank_products",
]
