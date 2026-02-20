# agentic/deep_agent/specifications/generation/llm_generator.py
# =============================================================================
# LLM SPECIFICATION GENERATOR — DELEGATION FACADE
# =============================================================================
#
# This module is a thin re-export facade.
# The canonical implementation lives in:
#   common/standards/generation/llm_generator.py
#
# All symbols are re-exported here to preserve backward-compatible imports
# for callers that use `agentic.deep_agent.specifications.generation.llm_generator`.
#
# =============================================================================

from common.standards.generation.llm_generator import (
    # Main generation functions
    generate_llm_specs,
    generate_llm_specs_batch,
    generate_specs_with_discovery,

    # Dynamic key discovery
    discover_specification_keys,

    # User specification extraction
    extract_user_specified_specs,
    extract_user_specs_batch,

    # Configuration constants
    MIN_LLM_SPECS_COUNT,
    MAX_LLM_ITERATIONS,
    SPECS_PER_ITERATION,
    MAX_PARALLEL_WORKERS,
    ENABLE_PARALLEL_ITERATIONS,
    ENABLE_DYNAMIC_DISCOVERY,

    # Model names
    LLM_SPECS_MODEL,
    REASONING_MODEL,
)

__all__ = [
    # Main generation functions
    "generate_llm_specs",
    "generate_llm_specs_batch",
    "generate_specs_with_discovery",

    # Dynamic key discovery
    "discover_specification_keys",

    # User specification extraction
    "extract_user_specified_specs",
    "extract_user_specs_batch",

    # Configuration constants
    "MIN_LLM_SPECS_COUNT",
    "MAX_LLM_ITERATIONS",
    "SPECS_PER_ITERATION",
    "MAX_PARALLEL_WORKERS",
    "ENABLE_PARALLEL_ITERATIONS",
    "ENABLE_DYNAMIC_DISCOVERY",

    # Model names
    "LLM_SPECS_MODEL",
    "REASONING_MODEL",
]
