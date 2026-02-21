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
        
        mandatory_list = product_schema.get("mandatory", [])
        if not mandatory_list:
            mandatory_reqs = product_schema.get("mandatoryRequirements", {})
            if mandatory_reqs:
                if isinstance(mandatory_reqs, dict):
                    mandatory_list = list(mandatory_reqs.keys())
                elif isinstance(mandatory_reqs, list):
                    mandatory_list = mandatory_reqs
            else:
                props = product_schema.get("properties", {})
                mandatory_list = [k for k, v in props.items() if isinstance(v, dict) and (v.get("required") or str(v.get("importance")).lower() == "high")]
        
        mandatory_fields = json.dumps(mandatory_list)
        chain = prompt | llm | JsonOutputParser()
        
        result = chain.invoke({
            "product_type": product_type,
            "user_input": user_input,
            "mandatory_fields": mandatory_fields
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

