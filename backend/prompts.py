DEPRECATED COMPATIBILITY SHIM FOR OLD PROMPTS MODULE

This module provides backward compatibility for code still referencing the old
'prompts' module pattern (e.g., prompts.sales_agent_greeting_prompt).

ALL NEW CODE SHOULD USE:
    from common.prompts_library import load_prompt
    PROMPT = load_prompt("prompt_name")

This file will be removed in a future version.
"""

import warnings
from common.prompts import (
    INDEXING_AGENT_PROMPTS,
    INTENT_PROMPTS,
    RANKING_PROMPTS,
    RAG_PROMPTS,
    SOLUTION_DEEP_AGENT_PROMPTS,
    ANALYSIS_TOOL_VENDOR_ANALYSIS_PROMPT,
    SCHEMA_VALIDATION_PROMPT
)
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, List

# Show deprecation warning when this module is imported
warnings.warn(
    "The 'prompts' module is deprecated. Please use 'common.prompts' constants instead.",
    DeprecationWarning,
    stacklevel=2
)

# Load consolidated prompt files
# Sales agent prompts removed as they are legacy
# _SALES_AGENT_PROMPTS = load_prompt_sections("sales_agent_prompts", default_section="SALES_CONSULTANT")
# _SALES_AGENT_MAIN_PROMPT = _SALES_AGENT_PROMPTS["SALES_CONSULTANT"]
# _SALES_WORKFLOW_PROMPTS = load_prompt_sections("sales_agent_prompts", default_section="GREETING_GENERATION")
# NOTE: Intent prompts have been reorganized:
# - intent_classification_prompts.txt -> CLASSIFICATION, QUICK_CLASSIFICATION (for routing)
# - intent_prompts.txt -> REQUIREMENTS_EXTRACTION (for intent analysis)
# This file (prompts.py) is DEPRECATED/Legacy. New code should use tools/intent_tools.py.

# Using INTENT_PROMPTS string directly
_INTENT_CLASSIFICATION_PROMPT = INTENT_PROMPTS

# PPI prompts loaded from combined file
_PPI_PROMPTS = INDEXING_AGENT_PROMPTS
_PPI_VENDOR_DISCOVERY_PROMPT = _PPI_PROMPTS.get("VENDOR_DISCOVERY", "")
_RANKING_PROMPTS = RANKING_PROMPTS
_STRATEGY_STANDARDIZATION_PROMPTS = RAG_PROMPTS


# ========================================================================================
# SALES AGENT PROMPTS (Legacy - for main.py compatibility)
# These should be refactored to use the new sales_agent_tools.py
# ========================================================================================

# Legacy sales agent prompts - use the main prompt for all
# Legacy sales agent prompts removed
sales_agent_knowledge_question_prompt = ""
sales_agent_initial_input_prompt = ""
sales_agent_no_additional_specs_prompt = ""
sales_agent_yes_additional_specs_prompt = ""
sales_agent_acknowledge_additional_specs_prompt = ""
sales_agent_default_prompt = ""
sales_agent_advanced_specs_yes_prompt = ""
sales_agent_advanced_specs_no_prompt = ""
sales_agent_advanced_specs_display_prompt = ""
sales_agent_confirm_after_missing_info_prompt = ""
sales_agent_confirm_after_missing_info_with_params_prompt = ""
sales_agent_show_summary_proceed_prompt = ""
sales_agent_show_summary_intro_prompt = ""
sales_agent_final_analysis_prompt = ""
sales_agent_analysis_error_prompt = ""
sales_agent_greeting_prompt = ""

# ========================================================================================
# FEEDBACK PROMPTS (Legacy)
# ========================================================================================

feedback_positive_prompt = "Thank you for your positive feedback!"
feedback_negative_prompt = "Thank you for your feedback. We'll use it to improve."
feedback_comment_prompt = "Thank you for your comment: {comment}"

# ========================================================================================
# IDENTIFICATION PROMPTS (Legacy - replaced by identify_instruments_tool)
# ========================================================================================
_PRODUCT_ID_PROMPTS = SOLUTION_DEEP_AGENT_PROMPTS

identify_classification_prompt = _INTENT_CLASSIFICATION_PROMPT #["CLASSIFICATION"]
identify_greeting_prompt = ""
identify_instrument_prompt = _PRODUCT_ID_PROMPTS.get("INSTRUMENT_IDENTIFICATION", "")  # Unified prompt
identify_unrelated_prompt = _INTENT_CLASSIFICATION_PROMPT #["CLASSIFICATION"]
identify_fallback_prompt = _INTENT_CLASSIFICATION_PROMPT #["CLASSIFICATION"]
identify_unexpected_prompt = _INTENT_CLASSIFICATION_PROMPT #["CLASSIFICATION"]

# ========================================================================================
# MANUFACTURER/VENDOR PROMPTS (Legacy)
# ========================================================================================

manufacturer_domain_prompt = _PPI_VENDOR_DISCOVERY_PROMPT

# ========================================================================================
# VALIDATION PROMPTS (Legacy - replaced by schema_tools.py)
# ========================================================================================

validation_prompt = SCHEMA_VALIDATION_PROMPT # Still individual
validation_alert_initial_prompt = "Please review the validation results."
validation_alert_repeat_prompt = "Please address the validation issues."

# ========================================================================================
# REQUIREMENT EXPLANATION PROMPTS (Legacy)
# ========================================================================================

requirement_explanation_prompt = _INTENT_CLASSIFICATION_PROMPT #["REQUIREMENTS_EXTRACTION"]

# ========================================================================================
# ADVANCED PARAMETER PROMPTS (Legacy)
# ========================================================================================

advanced_parameter_selection_prompt = ""

# ========================================================================================
# LANGCHAIN CHAT PROMPT TEMPLATES (Legacy - for chaining.py)
# These are ChatPromptTemplate objects for direct use with LangChain
# ========================================================================================

# Validation prompt for requirement extraction
validation_prompt_template = ChatPromptTemplate.from_template("""
You are Engenie - an expert assistant for industrial requisitioners and buyers. Your job is to validate technical product requirements in a way that helps procurement professionals make informed decisions.
 
**IMPORTANT: Think step-by-step through your validation process.**
 
Before providing your final validation:
1. First, analyze the user input to identify key technical terms and specifications
2. Then, determine what physical parameter is being measured or controlled
3. Next, identify the appropriate device type based on industrial standards
4. Finally, extract and categorize the requirements (mandatory vs optional)
 
User Input:
{user_input}
 
Requirements Schema:
{schema}
 
Tasks:
1. Intelligently identify the CORE PRODUCT CATEGORY from user input
2. Extract the requirements that were provided, focusing on what matters to buyers
 
CRITICAL: Dynamic Product Type Intelligence:
Your job is to determine the most appropriate and standardized product category based on the user's input. Use your knowledge of industrial instruments and measurement devices to:
 
1. **Identify the core measurement function** - What is being measured? (pressure, temperature, flow, level, pH, etc.)
2. **Determine the appropriate device type** - What type of instrument is needed? (sensor, transmitter, meter, gauge, controller, valve, etc.)
3. **Remove technology-specific modifiers** - Focus on function over implementation (remove terms like "differential", "vortex", "radar", "smart", etc.)
4. **Standardize terminology** - Use consistent, industry-standard naming conventions
 
EXAMPLES (learn the pattern, don't memorize):
- "differential pressure transmitter" → analyze: measures pressure + transmits signal → "pressure transmitter"
- "vortex flow meter" → analyze: measures flow + meter device → "flow meter"
- "RTD temperature sensor" → analyze: measures temperature + sensing function → "temperature sensor"
- "smart level indicator" → analyze: measures level + indicates/transmits → "level transmitter"
- "pH electrode" → analyze: measures pH + sensing function → "ph sensor"
- "Isolation Valve" → analyze: valve used for isolation → "isolation valve"
 
YOUR APPROACH:
1. Analyze what physical parameter is being measured
2. Determine what type of industrial device is most appropriate
3. Use standard industrial terminology
4. Focus on procurement-relevant categories that buyers understand
5. Be consistent - similar requests should get similar categorizations
 
Remember: The goal is to create logical, searchable categories that help procurement teams find the right products efficiently. Use your expertise to make intelligent decisions about standardization.
 
{format_instructions}
Validate the outputs and adherence to the output structure.
 
""")

# Requirements extraction prompt
requirements_prompt = ChatPromptTemplate.from_template("""
You are Engenie - an expert assistant for industrial requisitioners and buyers. Extract and structure the key requirements from this user input so a procurement professional can quickly understand what is needed and why.
 
 
User Input:
{user_input}
 
Focus on:
- Technical specifications (pressure ranges, accuracy, etc.)
- Connection types and standards
- Application context and environment
- Performance requirements
- Compliance or certification needs
- Any business or operational considerations relevant to buyers
 
Return a clear, structured summary of requirements, using language that is actionable and easy for buyers to use in procurement. Only include sections and details for which information is explicitly present in the user's input. Do not add any inferred requirements or placeholders for missing information.
Validate the outputs and adherence to the output structure.
 
""")

# Standardization prompt
standardization_prompt = ChatPromptTemplate.from_template("""
You are Engenie — an expert in industrial instrumentation and procurement standardization.
Your task is to standardize naming conventions for industrial products, vendors, and specifications
to create consistency across procurement systems.
 
Context: {context}
 
Data to standardize:
Vendor: {vendor}
Product Type: {product_type}
Model Family: {model_family}
Specifications: {specifications}
 
Instructions:
 
1. **Vendor Standardization**
   - Normalize vendor names to proper corporate naming conventions.
   - Resolve common spelling variations and abbreviations.
   - If uncertain, keep the original vendor name without guessing.
 
2. **Product Type Standardization**
   - Preserve all descriptors that meaningfully distinguish the instrument's measurement technology, method, or operating principle.
     *Do NOT remove or generalize technology descriptors.*
     Examples of descriptors to keep (conceptually, not limited to specific words):
       • measurement principle
       • sensing technology
       • operating method
       • functional variant that affects procurement category
   - The goal is to produce a standardized, clean, procurement-friendly name
     *while retaining every detail needed to map to the correct procurement subcategory*.
   - Do not simplify or collapse product types into broader categories
     if doing so would lose the technology or method.
 
   Correct behavior:
     - Keep distinctions such as different flow meter technologies.
     - Keep transmitter technology variations.
     - Keep level measurement principles.
     - Keep sensor type variations.
 
3. **Model Family Standardization**
   - Clean up model naming while keeping essential identifiers.
   - Normalize spacing, dashes, and capitalization.
   - Never create model names that are not present or infer missing ones.
 
4. **Specification Standardization**
   - Standardize units, formatting, and terminology.
   - Preserve meaningful parameter names.
   - Do not invent specifications that were not provided.
 
General Requirements:
- Output must be consistent with industrial instrumentation conventions.
- Keep all meaningful differentiators that affect procurement classification.
- Avoid removing technology indicators or collapsing categories.
- Final names must be precise, standardized, and searchable in procurement databases.
 
Response format:
{{
    "vendor": "[standardized vendor name]",
    "product_type": "[standardized product type]",
    "model_family": "[standardized model family]",
    "specifications": {{[standardized specifications]}}
}}
 
Validate adherence to the output structure.
""")

# Schema key description prompt
schema_description_prompt = ChatPromptTemplate.from_template("""
You are Engenie - an expert assistant helping users understand technical product specification fields in an easy, non-technical way.
 
Your task is to generate a short, human-readable description for the schema field: '{field}'
 
Guidelines:
- Write a clear, concise description that non-technical users can understand
- Focus on what this field represents and why it's important for product selection
- Do NOT mention the specific product type in the description
- Include 2-3 realistic example values that would be typical for this field
- Keep the entire description to 1-2 sentences maximum
- Use plain language, avoiding technical jargon where possible
 
Examples of good descriptions:
- For 'accuracy': "The precision of measurements, typically expressed as a percentage. Examples: ±0.1%, ±0.25%, ±1.0%"
- For 'outputSignal': "The type of electrical signal the device sends to control systems. Examples: 4-20mA, 0-10V, Digital"
- For 'operatingTemperature': "The temperature range where the device can function properly. Examples: -40°C to 85°C, 0°C to 60°C"
 
Field to describe: {field}
Context Product Type: {product_type}
 
Generate description:
Validate the outputs and adherence to the output structure.
 
""")


# ========================================================================================
# PROMPT GETTER FUNCTIONS (Required by chaining.py)
# These functions format prompts with variables for the analysis chain
# ========================================================================================

def get_validation_prompt(user_input: str, schema: str, format_instructions: str) -> str:
    """
    Build validation prompt for requirement extraction.
    Used by chaining.py invoke_validation_chain().
    """
    template = SCHEMA_VALIDATION_PROMPT # Still individual
    # Template uses: {user_input}, {product_type}, {schema}
    # format_instructions not used in this template, but we need product_type
    return template.format(
        user_input=user_input,
        product_type="",  # Will be detected from input
        schema=schema,
        format_instructions=format_instructions
    )


def get_requirements_prompt(user_input: str) -> str:
    """
    Build requirements extraction prompt.
    Used by chaining.py invoke_requirements_chain().
    """
    # Use the loaded prompt string and create a template from it, because it's just a string now
    from langchain_core.prompts import PromptTemplate
    template_str = _INTENT_CLASSIFICATION_PROMPT #["REQUIREMENTS_EXTRACTION"]
    # The prompt typically has {user_input} in it.
    # Note: load_prompt returned a string, so we need to format it or use PromptTemplate
    # The original code likely used load_prompt which returned a string, then .format() on string?
    # Yes, str.format()works if the string has {placeholders}.
    return template_str.format(user_input=user_input)


def get_vendor_prompt(
    vendor: str,
    structured_requirements: str,
    products_json: str,
    pdf_content_json: str,
    format_instructions: str,
    applicable_standards: Optional[List[str]] = None,
    standards_specs: Optional[str] = None
) -> str:
    """
    Build vendor analysis prompt.
    Used by chaining.py invoke_vendor_chain().
    
    Args:
        vendor: Name of the vendor being analyzed
        structured_requirements: Formatted user requirements string
        products_json: JSON string of product catalog data
        pdf_content_json: JSON string of PDF datasheet content
        format_instructions: Output format instructions
        applicable_standards: List of applicable engineering standards
        standards_specs: Standards specifications from user's standards documents
    
    Returns:
        Formatted prompt string ready for LLM
    """
    template = ANALYSIS_TOOL_VENDOR_ANALYSIS_PROMPT
    
    # Handle optional standards parameters
    if applicable_standards is None:
        applicable_standards = []
    if standards_specs is None:
        standards_specs = "No specific standards requirements provided."
    
    # Format standards list for prompt
    standards_list = "\n".join(f"- {std}" for std in applicable_standards) if applicable_standards else "No specific standards specified."
    
    # Template uses the exact parameter names from the prompt template
    return template.format(
        vendor=vendor,
        structured_requirements=structured_requirements,
        applicable_standards=standards_list,
        standards_specs=standards_specs,
        pdf_content_json=pdf_content_json,
        products_json=products_json,
        format_instructions=format_instructions
    )


def get_ranking_prompt(vendor_analysis: str, format_instructions: str) -> str:
    """
    Build ranking prompt for product comparison.
    Used by chaining.py invoke_ranking_chain().
    """
    template = _RANKING_PROMPTS["RANKING"]
    # Template uses: {requirements}, {vendor_matches}
    return template.format(
        requirements="See vendor analysis for original requirements",
        vendor_matches=vendor_analysis
    )


_ADDITIONAL_REQUIREMENTS_PROMPT = """You are EnGenie, an expert in Industrial Process Control Systems. Your task is to extract ADDITIONAL technical requirements from the user's input and map them to the provided schema fields.

PRODUCT TYPE: {product_type}

AVAILABLE SCHEMA FIELDS:
{schema}

USER INPUT: {user_input}

INSTRUCTIONS:
1. Extract ONLY the values that the user explicitly provides in their input
2. Map each value to the appropriate field from the schema above
3. Use the exact camelCase field names from the schema
4. Do NOT infer or assume values that aren't explicitly stated
5. If a value doesn't match any schema field, ignore it
6. Return empty dictionary for provided_requirements if no valid requirements are found

CRITICAL RULES:
- Only extract fields that exist in the provided schema
- Use exact schema field names (camelCase)
- Extract the exact values the user provides
- Do NOT add default values or inferences

EXPECTED OUTPUT FORMAT:
{{
  "provided_requirements": {{
    "<schemaFieldName>": "<userProvidedValue>",
    "<anotherSchemaField>": "<anotherValue>"
  }},
  "product_type": "{product_type}"
}}

EXAMPLE:
User says: "Class V seat leakage, Equal Percentage flow characteristic, Cv 25, Fail-Close"
For a Control Valve schema, output:
{{
  "provided_requirements": {{
    "seatLeakageClass": "Class V",
    "flowCharacteristic": "Equal Percentage",
    "ratedCvKvRange": "25",
    "actuatorAction": "Fail-Close"
  }},
  "product_type": "Control Valve"
}}

Now extract requirements from the user input above. Respond with ONLY valid JSON."""


def get_additional_requirements_prompt(
    user_input: str,
    product_type: str,
    schema: str,
    format_instructions: str
) -> str:
    """
    Build additional requirements extraction prompt.
    Used by chaining.py invoke_additional_requirements_chain().
    """
    return _ADDITIONAL_REQUIREMENTS_PROMPT.format(
        user_input=user_input,
        product_type=product_type,
        schema=schema,
        format_instructions=format_instructions
    )


def get_schema_description_prompt(field_name: str, product_type: str) -> str:
    """
    Build schema field description prompt.
    Used by chaining.py invoke_schema_description_chain().
    """
    return f"""You are Engenie - an expert assistant helping users understand technical product specification fields in an easy, non-technical way.

Your task is to generate a short, human-readable description for the schema field: '{field_name}'

Guidelines:
- Write a clear, concise description that non-technical users can understand
- Focus on what this field represents and why it's important for product selection
- Do NOT mention the specific product type in the description
- Include 2-3 realistic example values that would be typical for this field
- Keep the entire description to 1-2 sentences maximum
- Use plain language, avoiding technical jargon where possible

Examples of good descriptions:
- For 'accuracy': "The precision of measurements, typically expressed as a percentage. Examples: ±0.1%, ±0.25%, ±1.0%"
- For 'outputSignal': "The type of electrical signal the device sends to control systems. Examples: 4-20mA, 0-10V, Digital"
- For 'operatingTemperature': "The temperature range where the device can function properly. Examples: -40°C to 85°C, 0°C to 60°C"

Field to describe: {field_name}
Context Product Type: {product_type}

Generate description:
Validate the outputs and adherence to the output structure."""


def get_standardization_prompt(
    vendor: str,
    product_type: str,
    model_family: str,
    specifications: dict,
    context: str = ""
) -> str:
    """
    Build standardization prompt for naming conventions.
    Used for standardizing vendor/product names across procurement systems.
    """
    import json
    return standardization_prompt.format(
        context=context,
        vendor=vendor,
        product_type=product_type,
        model_family=model_family,
        specifications=json.dumps(specifications, indent=2) if specifications else "{}"
    )


def get_strategy_standardization_prompt(prompt_type: str) -> str:
    """
    Get strategy standardization prompt by type.

    Args:
        prompt_type: "category", "subcategory", "strategy_keywords", or "vendor"

    Returns:
        Prompt template string
    """
    section_map = {
        "category": "CATEGORY_STANDARDIZATION",
        "subcategory": "SUBCATEGORY_STANDARDIZATION",
        "strategy_keywords": "STRATEGY_KEYWORD_EXTRACTION",
        "strategy": "STRATEGY_KEYWORD_EXTRACTION",  # Alias
        "vendor": "VENDOR_NAME_STANDARDIZATION"
    }

    section = section_map.get(prompt_type, "CATEGORY_STANDARDIZATION")
    return _STRATEGY_STANDARDIZATION_PROMPTS.get(section, "")


# ========================================================================================
# Note: This compatibility layer maps old prompt names to new prompts_library names
# Where exact matches don't exist, we use the closest equivalent prompt
# ========================================================================================

