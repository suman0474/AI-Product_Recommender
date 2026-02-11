# agentic/api_utils.py
# =============================================================================
# CONSOLIDATED API UTILITIES
# =============================================================================
#
# This module consolidates common API utilities that were previously
# duplicated across multiple API modules:
#   - api_response() function
#   - handle_errors() decorator
#
# Previously duplicated in:
#   - agentic/api.py
#   - agentic/session_api.py
#   - agentic/strategy_rag/strategy_admin_api.py
#   - tools/api.py
#
# Usage:
#   from agentic.infrastructure.api.utils import api_response, handle_errors
#
#   @app.route('/endpoint')
#   @handle_errors
#   def my_endpoint():
#       return api_response(True, data={'key': 'value'})
#
# =============================================================================

import logging
from functools import wraps
from typing import Any, Optional, Dict
from flask import jsonify
import re
import copy
import json

logger = logging.getLogger(__name__)


def api_response(
    success: bool,
    data: Any = None,
    error: Optional[str] = None,
    status_code: int = 200,
    tags: Any = None,
    **extra_fields
) -> tuple:
    """
    Create a standardized API response.

    Provides consistent response format across all API endpoints:
    {
        "success": bool,
        "data": any,
        "error": string | null,
        "tags": object | null  (optional)
    }

    Args:
        success: Whether the request was successful
        data: Response data (any JSON-serializable type)
        error: Error message (if any)
        status_code: HTTP status code (default 200)
        tags: Optional tags object for frontend routing/UI hints
        **extra_fields: Additional fields to include in response

    Returns:
        Tuple of (Flask JSON response, status_code)
    """
    response = {
        "success": success,
        "data": data,
        "error": error
    }

    # Add tags if provided (backward compatible)
    if tags is not None:
        # Handle both dict and object with .dict() method
        if hasattr(tags, 'dict'):
            response["tags"] = tags.dict()
        elif hasattr(tags, 'model_dump'):
            response["tags"] = tags.model_dump()
        else:
            response["tags"] = tags

    # Add any extra fields
    response.update(extra_fields)

    return jsonify(response), status_code


def handle_errors(func=None, *, log_prefix: str = "API"):
    """
    Decorator for standardized error handling in API endpoints.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                logger.error(f"{log_prefix} Error in {f.__name__}: {e}", exc_info=True)
                return api_response(
                    success=False,
                    error=str(e),
                    status_code=500
                )
        return decorated_function

    # Handle both @handle_errors and @handle_errors(...)
    if func is not None:
        return decorator(func)
    return decorator


def validate_request_json(required_fields: list = None, optional_fields: list = None):
    """
    Decorator to validate JSON request body.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import request
            data = request.get_json()

            if not data:
                return api_response(
                    success=False,
                    error="Request body is required",
                    status_code=400
                )

            if required_fields:
                missing = [field for field in required_fields if not data.get(field)]
                if missing:
                    return api_response(
                        success=False,
                        error=f"Missing required fields: {', '.join(missing)}",
                        status_code=400
                    )

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def validate_query_params(required_params: list = None):
    """
    Decorator to validate query parameters.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import request
            if required_params:
                missing = [param for param in required_params if not request.args.get(param)]
                if missing:
                    return api_response(
                        success=False,
                        error=f"Missing required query parameters: {', '.join(missing)}",
                        status_code=400
                    )

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def convert_keys_to_camel_case(obj):
    """Recursively converts dictionary keys from snake_case to camelCase."""
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            camel_key = re.sub(r'_([a-z])', lambda m: m.group(1).upper(), key)
            new_dict[camel_key] = convert_keys_to_camel_case(value)
        return new_dict
    elif isinstance(obj, list):
        return [convert_keys_to_camel_case(item) for item in obj]
    return obj


def clean_empty_values(data):
    """Recursively replaces 'Not specified', etc., with empty strings."""
    if isinstance(data, dict):
        return {k: clean_empty_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_empty_values(item) for item in data]
    elif isinstance(data, str) and data.lower().strip() in ["not specified", "not requested", "none specified", "n/a", "na"]:
        return ""
    return data


def map_provided_to_schema(detected_schema: dict, provided: dict) -> dict:
    """Maps providedRequirements into the schema structure."""
    if not isinstance(detected_schema, dict):
        return {}
    mapped = copy.deepcopy(detected_schema)
    if not isinstance(provided, dict):
        return mapped
        
    if "mandatoryRequirements" in provided or "optionalRequirements" in provided:
        for section in ["mandatoryRequirements", "optionalRequirements"]:
            if section in provided and section in mapped and isinstance(provided[section], dict):
                for key, value in provided[section].items():
                    if key in mapped[section]:
                        mapped[section][key] = value
        return mapped
        
    for key, value in provided.items():
        if key in mapped.get("mandatoryRequirements", {}):
            mapped["mandatoryRequirements"][key] = value
        elif key in mapped.get("optionalRequirements", {}):
            mapped["optionalRequirements"][key] = value
    return mapped


def get_missing_mandatory_fields(provided: dict, schema: dict) -> list:
    """Identify missing mandatory fields by comparing provided data against schema."""
    missing = []
    if not isinstance(schema, dict) or not isinstance(provided, dict):
        return missing
        
    mandatory_schema = schema.get("mandatoryRequirements", {})
    provided_mandatory = provided.get("mandatoryRequirements", {})

    def traverse_and_check(schema_node, provided_node):
        if not isinstance(schema_node, dict):
            return
            
        for key, schema_value in schema_node.items():
            if isinstance(schema_value, dict):
                traverse_and_check(schema_value, provided_node.get(key, {}) if isinstance(provided_node, dict) else {})
            else:
                provided_value = provided_node.get(key) if isinstance(provided_node, dict) else None
                if provided_value is None or str(provided_value).strip() in ["", ","]:
                    missing.append(key)

    traverse_and_check(mandatory_schema, provided_mandatory)
    return missing


def friendly_field_name(field: str) -> str:
    """Convert camelCase or snake_case field name to a human-friendly Title Case."""
    if not field:
        return ""
    s1 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', field)
    return s1.replace("_", " ").title()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'api_response',
    'handle_errors',
    'validate_request_json',
    'validate_query_params',
    'convert_keys_to_camel_case',
    'clean_empty_values',
    'map_provided_to_schema',
    'get_missing_mandatory_fields',
    'friendly_field_name'
]
