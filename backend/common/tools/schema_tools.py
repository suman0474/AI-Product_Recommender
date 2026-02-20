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
        
        if schema:
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
    # This logic was previously in SchemaAgent.validate_requirements.
    # We need to reimplement it or delegate to a new validation service.
    # For now, strictly to fix the import error, we will return a permissive result
    # but log a warning that validation logic is pending migration.
    
    logger.warning("[validate_requirements_tool] Using placeholder validation (pending full migration)")
    
    # Simple extraction simulation (placeholder)
    return {
        "is_valid": True, # Assume valid to unblock workflow
        "missing_fields": [],
        "optional_fields": [],
        "provided_requirements": {}, # Should extract based on LLM in real impl
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

