
# prompts/__init__.py
# =============================================================================
# PROMPTS PACKAGE
# =============================================================================
#
# This package contains all LLM prompts used throughout the AIPR application.
# Prompts are stored in individual .txt files and loaded at runtime.
#
# Usage:
#   from common.prompts import load_prompt, ANALYSIS_TOOL_VENDOR_ANALYSIS_PROMPT
#   
# =============================================================================

from .prompt_loader import (
    load_prompt,
    load_prompt_sections,
    get_prompt_metadata,
    list_available_prompts,
    reload_prompt,
    clear_prompt_cache,
    get_deep_agent_prompts,
    get_shared_agent_prompts,
    get_rag_prompts,
)

# Load constraints and prompts
ANALYSIS_TOOL_VENDOR_ANALYSIS_PROMPT = load_prompt("analysis_tool_vendor_analysis_prompt")
INDEX_RAG_PROMPTS = load_prompt_sections("index_rag_prompts")
INDEXING_AGENT_PROMPTS = load_prompt_sections("indexing_agent_prompts")
INTENT_CLASSIFICATION_PROMPTS = load_prompt_sections("intent_classification_prompts")
INTENT_PROMPTS = load_prompt("intent_prompts")
RAG_PROMPTS = load_prompt_sections("rag_prompts")
RANKING_PROMPTS = load_prompt_sections("ranking_prompts")
SCHEMA_VALIDATION_PROMPT = load_prompt("schema_validation_prompt")
SEARCH_DEEP_AGENT_PROMPTS = load_prompt_sections("search_deep_agent_prompts")
SOLUTION_DEEP_AGENT_PROMPTS = load_prompt_sections("solution_deep_agent_prompts")
STANDARDS_DEEP_AGENT_PROMPTS = load_prompt_sections("standards_deep_agent_prompts")

def get_vendor_prompt(
    vendor: str,
    structured_requirements: str,
    products_json: str,
    pdf_content_json: str,
    format_instructions: str,
    applicable_standards=None,
    standards_specs=None
) -> str:
    """
    Build a complete vendor analysis prompt combining the base system template
    with the dynamic per-call context (vendor, requirements, products, PDFs).

    Used by common/core/chaining.py::invoke_vendor_chain().
    """
    standards_section = ""
    if applicable_standards:
        standards_section += f"\n\nAPPLICABLE STANDARDS:\n{applicable_standards}"
    if standards_specs:
        standards_section += f"\n\nSTANDARDS SPECIFICATIONS:\n{standards_specs}"

    return (
        ANALYSIS_TOOL_VENDOR_ANALYSIS_PROMPT
        + f"\n\nVENDOR: {vendor}\n\n"
        + f"USER REQUIREMENTS:\n{structured_requirements}\n\n"
        + f"AVAILABLE PRODUCTS (JSON):\n{products_json}\n\n"
        + f"PDF DATASHEET CONTENT:\n{pdf_content_json}\n\n"
        + f"FORMAT INSTRUCTIONS:\n{format_instructions}"
        + standards_section
    )


def get_ranking_prompt(vendor_analysis: str, format_instructions: str) -> str:
    """
    Build a complete ranking prompt combining the base ranking template
    with the vendor analysis results and format instructions.

    Used by common/core/chaining.py::invoke_ranking_chain().
    """
    ranking_template = RANKING_PROMPTS.get("RANKING", RANKING_PROMPTS.get("DEFAULT", ""))
    return (
        ranking_template
        + f"\n\nVENDOR ANALYSIS RESULTS:\n{vendor_analysis}\n\n"
        + f"FORMAT INSTRUCTIONS:\n{format_instructions}"
    )

def get_validation_prompt(user_input: str, schema: str, format_instructions: str) -> str:
    """Build validation prompt using SCHEMA_VALIDATION_PROMPT"""
    return (
        SCHEMA_VALIDATION_PROMPT
        + f"\n\nUser Input: {user_input}\n"
        + f"Schema: {schema}\n"
        + f"Format Instructions: {format_instructions}"
    )

def get_requirements_prompt(user_input: str) -> str:
    """Build requirements prompt using INTENT_PROMPTS as base if needed, or structured differently"""
    # Assuming INTENT_PROMPTS is the base for requirements extraction as seen in prompts.py view
    return (
        INTENT_PROMPTS
        .replace("{user_input}", user_input)
    )

def get_additional_requirements_prompt(user_input, product_type, schema, format_instructions) -> str:
    # Placeholder implementation if not found in git show
    # Assuming functionality based on naming
    return f"Additional Requirements Analysis for {product_type}\nInput: {user_input}\nSchema: {schema}\n{format_instructions}"

def get_schema_description_prompt(field_name, product_type) -> str:
    return f"Provide description for schema field {field_name} for product {product_type}."

__all__ = [
    "load_prompt",
    "load_prompt_sections",
    "get_prompt_metadata",
    "list_available_prompts",
    "reload_prompt",
    "clear_prompt_cache",
    "get_deep_agent_prompts",
    "get_shared_agent_prompts",
    "get_rag_prompts",
    "ANALYSIS_TOOL_VENDOR_ANALYSIS_PROMPT",
    "INDEX_RAG_PROMPTS",
    "INDEXING_AGENT_PROMPTS",
    "INTENT_CLASSIFICATION_PROMPTS",
    "INTENT_PROMPTS",
    "RAG_PROMPTS",
    "RANKING_PROMPTS",
    "SCHEMA_VALIDATION_PROMPT",
    "SEARCH_DEEP_AGENT_PROMPTS",
    "SOLUTION_DEEP_AGENT_PROMPTS",
    "STANDARDS_DEEP_AGENT_PROMPTS",
    "get_vendor_prompt",
    "get_ranking_prompt",
    "get_validation_prompt",
    "get_requirements_prompt",
    "get_additional_requirements_prompt",
    "get_schema_description_prompt",
]
