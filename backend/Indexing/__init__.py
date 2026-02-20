"""
Indexing Package (Indexing Agent)
=============================
LangGraph-powered product indexing and schema retrieval system.
Historically known as PPI Agent.
"""

from .state import IndexingState
from .graph import (
    build_indexing_graph,
    create_indexing_workflow,
    run_indexing_workflow,
    run_indexing_workflow_streaming,
    run_indexing_workflow_legacy_compatible,
    run_potential_product_indexing_workflow,
)

# Backward-compatible aliases
PPIAgentState = IndexingState
PPIState = IndexingState
build_ppi_graph = build_indexing_graph
create_ppi_workflow = create_indexing_workflow
run_ppi_workflow = run_indexing_workflow
run_ppi_workflow_streaming = run_indexing_workflow_streaming
run_ppi_workflow_legacy_compatible = run_indexing_workflow_legacy_compatible
run_potential_product_index_workflow = run_potential_product_indexing_workflow

__version__ = "3.1.0"

__all__ = [
    # Core workflow
    "build_indexing_graph",
    "create_indexing_workflow",
    "run_indexing_workflow",
    "run_indexing_workflow_streaming",
    "run_indexing_workflow_legacy_compatible",
    "run_potential_product_indexing_workflow",

    # State
    "IndexingState",
    "PPIAgentState",
    "PPIState",

    # Aliases
    "build_ppi_graph",
    "create_ppi_workflow",
    "run_ppi_workflow",
    "run_ppi_workflow_streaming",
    "run_ppi_workflow_legacy_compatible",
    "run_potential_product_index_workflow",
]
