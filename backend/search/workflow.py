# search/workflow.py
"""
Product Search Workflow - Tool-Based Architecture
==================================================

This file provides backward compatibility for old imports.
The actual workflow is now implemented using tool classes:
- ValidationTool
- AdvancedSpecificationAgent
- VendorAnalysisTool
- RankingTool
- SalesAgentTool

For the main workflow functions, import from search module:
    from search import run_product_search_workflow

For individual tools:
    from search.validation_tool import ValidationTool
    from search.vendor_analysis_deep_agent import VendorAnalysisDeepAgent
    from search.ranking_tool import RankingTool
"""

import logging

logger = logging.getLogger(__name__)

# Re-export workflow functions for backward compatibility
from . import (
    run_product_search_workflow,
    run_validation_only,
    run_advanced_params_only,
    run_analysis_only,
    process_from_solution_workflow,
    get_schema_only,
    validate_with_schema,
)

__all__ = [
    "run_product_search_workflow",
    "run_validation_only",
    "run_advanced_params_only",
    "run_analysis_only",
    "process_from_solution_workflow",
    "get_schema_only",
    "validate_with_schema",
]
