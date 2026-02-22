# tools/schema_tools.py
# Adapter layer for backward compatibility
# Canonical implementation is in search.schema_agent.SchemaAgent (DEPRECATED)
# Now using common.services.schema_service and local implementations

import logging
from typing import Dict, Any, Optional, List
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from common.services.schema_service import schema_service

logger = logging.getLogger(__name__)

# =============================================================================
# INPUT MODELS
# =============================================================================

class LoadSchemaInput(BaseModel):
    product_type: str = Field(description="The type of product to load schema for")
    enable_ppi: bool = Field(default=True, description="Whether to enable Potential Product Indexing if schema not found")

class ValidateRequirementsInput(BaseModel):
    user_input: str = Field(description="The user's raw input describing requirements")
    product_type: str = Field(description="The type of product")
    product_schema: Optional[Dict[str, Any]] = Field(default=None, description="The schema to validate against")

class GetMissingFieldsInput(BaseModel):
    provided_requirements: Dict[str, Any] = Field(description="Requirements extracted/provided so far")
    product_schema: Dict[str, Any] = Field(description="The product schema defining mandatory fields")

# =============================================================================
# TOOLS
# =============================================================================

@tool("load_schema", args_schema=LoadSchemaInput)
def load_schema_tool(product_type: str, enable_ppi: bool = True) -> Dict[str, Any]:
    """
    Load the requirements schema for a specific product type.
    """
    logger.info(f"[load_schema_tool] Loading schema for '{product_type}' (PPI={enable_ppi})")
    
    try:
        # 1. Try schema service (MongoDB/Azure)
        schema = schema_service.get_schema(product_type)
        
        if schema is not None:
            return {
                "schema": schema,
                "source": "database",
                "ppi_used": False,
                "from_database": True
            }
            
        # 2. If not found and PPI enabled, we would typically trigger generation.
        # For now, return a basic default schema or indicate failure to allow
        # the caller to handle it (e.g. valid_product_node might handle generation).
        # In the future, this should invoke the Indexing workflow.
        
        logger.warning(f"[load_schema_tool] Schema not found for '{product_type}'. PPI/Generation extraction is required.")
        
        return {
            "schema": {},
            "source": "not_found",
            "ppi_used": False, # Should be True if we actually generated it
            "from_database": False,
            "error": "Schema not found in database"
        }

    except Exception as e:
        logger.error(f"[load_schema_tool] Failed: {e}")
        return {
            "schema": {},
            "source": "error",
            "error": str(e)
        }

@tool("validate_requirements", args_schema=ValidateRequirementsInput)
def validate_requirements_tool(
    user_input: str,
    product_type: str,
    product_schema: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate user requirements against the product schema.
    """
    import json
    import os
    from common.services.llm.fallback import create_llm_with_fallback
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    
    logger.info(f"[validate_requirements_tool] Parsing against schema for '{product_type}'...")
    
    if not product_schema:
        logger.warning("[validate_requirements_tool] No schema provided, cannot validate.")
        return {
            "is_valid": False,
            "missing_fields": [],
            "optional_fields": [],
            "provided_requirements": {},
            "refined_product_type": product_type
        }
    
    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        prompt = ChatPromptTemplate.from_template("""
        You are an industrial engineer validating user requirements against a product schema.
        
        Product Type: {product_type}
        User Input:
        {user_input}
        
        Schema Mandatory Fields:
        {mandatory_fields}
        
        Analyze what the user requested. If the user input mentions specifications that map to the schema, extract them.
        Identify which of the schema's mandatory fields are NOT provided or mentioned by the user at all.
        A field is missing only if the user input implies nothing about it.

        Return ONLY a JSON object with this exact structure:
        {{
            "provided_requirements": {{"field_name": {{"value": "extracted value", "unit": "extracted unit"}}}},
            "missing_fields": ["missing_mandatory_field_1", "missing_mandatory_field_2"],
            "optional_fields": [],
            "refined_product_type": "{product_type}",
            "is_valid": true or false
        }}
        Set is_valid to true ONLY if missing_fields is empty.
        """)
        
        mandatory_reqs = product_schema.get("mandatory_requirements") or product_schema.get("mandatoryRequirements") or product_schema.get("mandatory", {})
        optional_reqs = product_schema.get("optional_requirements") or product_schema.get("optionalRequirements") or product_schema.get("optional", {})

        # ── Fix #1: Extract LEAF field names, not section names ───────────────
        # After schema generation, mandatory_reqs looks like:
        #   { "Performance": { "Accuracy": {...}, "OutputSignal": {...} }, "Electrical": {...} }
        # Building the list from mandatory_reqs.keys() would give ["Performance", "Electrical"],
        # NOT the real field names the LLM needs to match against the user's input.
        # This helper recursively collects all leaf field names.
        def _flatten_schema_keys(schema_section: Any, max_depth: int = 3) -> List[str]:
            """Recursively extract leaf field names from a potentially nested schema section."""
            keys = []
            if not isinstance(schema_section, dict):
                return keys
            for k, v in schema_section.items():
                if k.startswith('_'):
                    continue
                if isinstance(v, dict):
                    if 'value' in v or 'suggested_values' in v or 'source' in v:
                        # This is a field metadata object — k is the field name
                        keys.append(k)
                    elif max_depth > 1:
                        # Recurse into nested section
                        nested = _flatten_schema_keys(v, max_depth - 1)
                        if nested:
                            keys.extend(nested)
                        else:
                            # Treat k itself as a field (flat schema)
                            keys.append(k)
                    else:
                        keys.append(k)
                else:
                    # Primitive value — k is a field name
                    keys.append(k)
            return keys

        if isinstance(mandatory_reqs, dict):
            mandatory_list = _flatten_schema_keys(mandatory_reqs)
            if not mandatory_list:
                mandatory_list = list(mandatory_reqs.keys())
        else:
            mandatory_list = mandatory_reqs if isinstance(mandatory_reqs, list) else []

        if isinstance(optional_reqs, dict):
            optional_list = _flatten_schema_keys(optional_reqs)
            if not optional_list:
                optional_list = list(optional_reqs.keys())
        else:
            optional_list = optional_reqs if isinstance(optional_reqs, list) else []

        if not mandatory_list and not optional_list:
            props = product_schema.get("properties", {})
            mandatory_list = [k for k, v in props.items() if isinstance(v, dict) and (v.get("required") or str(v.get("importance")).lower() == "high")]
            optional_list = [k for k, v in props.items() if k not in mandatory_list]

        logger.info(
            "[validate_requirements_tool] Schema fields — mandatory: %d, optional: %d",
            len(mandatory_list), len(optional_list)
        )

        prompt = ChatPromptTemplate.from_template("""
        You are an industrial engineer validating user requirements against a product schema.
        
        Product Type: {product_type}
        User Input:
        {user_input}
        
        Schema Mandatory Fields:
        {mandatory_fields}

        Schema Optional Fields:
        {optional_fields}
        
        TASK:
        1. Extract specifications from User Input that map to the Schema Fields.
        2. Identify which Mandatory Fields are NOT mentioned or implied by the user.
        3. CRITICAL: Use the EXACT field name strings from the Schema Fields lists as keys in your response.
           If the schema field is "Accuracy", use "Accuracy" — NOT "accuracy" or "measurement_accuracy".
           If the schema field is "OutputSignal", use "OutputSignal" — NOT "output" or "output_signal".
           Only use keys that exist in the Schema Fields lists above.
        4. If a value has a unit (like bar, mA, barG, °C), extract the numeric part as value and unit separately.

        Return ONLY a JSON object:
        {{
            "provided_requirements": {{"ExactSchemaFieldName": {{"value": "extracted value", "unit": "extracted unit or empty"}}}},
            "missing_fields": ["ExactSchemaFieldName1", ...],
            "optional_fields": ["ExactSchemaFieldName1", ...],
            "refined_product_type": "{product_type}",
            "is_valid": true or false
        }}
        Set is_valid to true ONLY if missing_fields is empty.
        """)
        
        mandatory_fields = json.dumps(mandatory_list)
        optional_fields = json.dumps(optional_list)
        chain = prompt | llm | JsonOutputParser()
        
        result = chain.invoke({
            "product_type": product_type,
            "user_input": user_input,
            "mandatory_fields": mandatory_fields,
            "optional_fields": optional_fields
        })
        
        is_valid = len(result.get("missing_fields", [])) == 0
        
        return {
            "is_valid": is_valid,
            "missing_fields": result.get("missing_fields", []),
            "optional_fields": result.get("optional_fields", []),
            "provided_requirements": result.get("provided_requirements", {}),
            "refined_product_type": result.get("refined_product_type", product_type)
        }
        
    except Exception as e:
        logger.error(f"[validate_requirements_tool] Validation fallback error: {e}")
        return {
            "is_valid": False, 
            "missing_fields": mandatory_list if 'mandatory_list' in locals() and mandatory_list else ["All required specifications"],
            "optional_fields": [],
            "provided_requirements": {},
            "refined_product_type": product_type
        }

@tool("get_missing_fields", args_schema=GetMissingFieldsInput)
def get_missing_fields_tool(
    provided_requirements: Dict[str, Any],
    product_schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Identify missing mandatory and optional fields.
    """
    logger.info("[get_missing_fields_tool] Identifying missing fields")
    
    mandatory_fields = product_schema.get("mandatory", [])
    optional_fields = product_schema.get("optional", [])
    
    # Normalize keys for comparison
    provided_keys = set(k.lower() for k in provided_requirements.keys())
    
    missing_mandatory = [
        field for field in mandatory_fields 
        if field.lower() not in provided_keys
    ]
    
    missing_optional = [
        field for field in optional_fields
        if field.lower() not in provided_keys
    ]
    
    return {
        "missing_mandatory": missing_mandatory,
        "missing_optional": missing_optional,
        "is_complete": len(missing_mandatory) == 0
    }

