# tools/__init__.py
# LangChain Tools for Agentic AI Workflow

from .intent_tools import (
    classify_intent_tool,
    extract_requirements_tool
)

from .analysis_tools import (
    analyze_vendor_match_tool,
    calculate_match_score_tool,
    extract_specifications_tool
)

from .ranking_tools import (
    rank_products_tool,
    judge_analysis_tool
)

from .standards_enrichment_tool import (
    get_applicable_standards,
    enrich_schema_with_standards,
    validate_requirements_against_standards
)


# Index RAG Tools
from common.services.extraction.metadata_filter import (
    filter_by_hierarchy,
    extract_product_types,
    extract_vendors,
    extract_models,
    extract_metadata_with_llm
)

from common.services.extraction.parallel_indexer import (
    run_parallel_indexing,
    index_database,
    index_web_search
)

from .specs_filter import (
    apply_specs_and_strategy_filter,
    apply_standards_filter,
    apply_strategy_filter
)

__all__ = [
    # Intent Tools
    'classify_intent_tool',
    'extract_requirements_tool',
    # Analysis Tools
    'analyze_vendor_match_tool',
    'calculate_match_score_tool',
    'extract_specifications_tool',
    # Ranking Tools
    'rank_products_tool',
    'judge_analysis_tool',
    # Standards Tools
    'get_applicable_standards',
    'enrich_schema_with_standards',
    'validate_requirements_against_standards',
    # Index RAG Tools
    'filter_by_hierarchy',
    'extract_product_types',
    'extract_vendors',
    'extract_models',
    'extract_metadata_with_llm',
    'run_parallel_indexing',
    'index_database',
    'index_web_search',
    'apply_specs_and_strategy_filter',
    'apply_standards_filter',
    'apply_strategy_filter',
]
