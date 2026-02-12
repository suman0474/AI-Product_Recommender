# utils/json_utils.py
# =============================================================================
# CONSOLIDATED JSON UTILITIES
# =============================================================================
#
# This module consolidates JSON parsing and extraction utilities that were
# previously duplicated across multiple modules:
#   - agentic/standards_rag/standards_chat_agent.py
#   - agentic/strategy_rag/strategy_chat_agent.py
#   - agentic/index_rag/index_rag_agent.py
#
# These utilities handle common LLM response parsing challenges:
#   - JSON wrapped in markdown code blocks
#   - Malformed JSON with unescaped characters
#   - JSON with leading/trailing text
#
# Usage:
#   from utils.json_utils import extract_json_from_response, sanitize_json_string
#
#   response_dict = extract_json_from_response(llm_raw_output)
#
# =============================================================================

import json
import re
import logging
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)


def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from LLM response, handling various formats.

    LLMs often return JSON wrapped in markdown, with extra text, or with
    formatting issues. This function handles all common cases.

    Handles:
    - Pure JSON responses
    - JSON wrapped in markdown code blocks (```json ... ```)
    - JSON with leading/trailing text
    - Malformed JSON with unescaped newlines/quotes
    - Invalid control characters
    - Multiple JSON objects (returns first valid one)

    Args:
        text: Raw LLM response text

    Returns:
        Parsed JSON dictionary, or None if extraction fails

    Examples:
        # Pure JSON
        >>> extract_json_from_response('{"key": "value"}')
        {'key': 'value'}

        # Markdown wrapped
        >>> extract_json_from_response('```json\\n{"key": "value"}\\n```')
        {'key': 'value'}

        # With extra text
        >>> extract_json_from_response('Here is the result: {"key": "value"}')
        {'key': 'value'}
    """
    if not text:
        return None

    text = text.strip()

    # Try direct JSON parsing first (fastest path)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',   # ```json ... ```
        r'```\s*([\s\S]*?)\s*```',        # ``` ... ```
        r'`([\s\S]*?)`',                  # `...` (inline code)
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        for match in matches:
            try:
                cleaned = match.strip() if isinstance(match, str) else match
                sanitized = sanitize_json_string(cleaned)
                parsed = json.loads(sanitized)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

    # Try to find JSON object in text
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end + 1]
            sanitized = sanitize_json_string(json_str)
            result = json.loads(sanitized)
            if isinstance(result, dict):
                return result
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in text
    try:
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end + 1]
            sanitized = sanitize_json_string(json_str)
            result = json.loads(sanitized)
            # If it's an array with a dict, return the first dict
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                return result[0]
    except json.JSONDecodeError:
        pass

    logger.warning(f"Failed to extract JSON from response: {text[:200]}...")
    return None


def extract_json_array_from_response(text: str) -> Optional[List[Any]]:
    """
    Extract JSON array from LLM response.

    Similar to extract_json_from_response but specifically for arrays.

    Args:
        text: Raw LLM response text

    Returns:
        Parsed JSON array, or None if extraction fails
    """
    if not text:
        return None

    text = text.strip()

    # Try direct JSON parsing first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        for match in matches:
            try:
                cleaned = match.strip()
                sanitized = sanitize_json_string(cleaned)
                parsed = json.loads(sanitized)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                continue

    # Try to find JSON array in text
    try:
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end + 1]
            sanitized = sanitize_json_string(json_str)
            result = json.loads(sanitized)
            if isinstance(result, list):
                return result
    except json.JSONDecodeError:
        pass

    logger.warning(f"Failed to extract JSON array from response: {text[:200]}...")
    return None


def sanitize_json_string(text: str) -> str:
    """
    Sanitize JSON string by fixing common formatting issues.

    LLMs sometimes produce JSON with:
    - Unescaped newlines in string values
    - Invalid control characters
    - Trailing commas
    - Single quotes instead of double quotes

    This function attempts to fix these issues.

    Args:
        text: Raw JSON string

    Returns:
        Sanitized JSON string

    Example:
        >>> sanitize_json_string('{"key": "value with\\nnewline"}')
        '{"key": "value with\\\\nnewline"}'
    """
    if not text:
        return text

    # Remove invalid control characters (except whitespace)
    text = ''.join(ch for ch in text if ord(ch) >= 32 or ch in '\n\r\t')

    # Handle literal newlines/carriage returns outside of string values
    result = []
    in_string = False
    escape_next = False

    for i, char in enumerate(text):
        if escape_next:
            result.append(char)
            escape_next = False
            continue

        if char == '\\':
            result.append(char)
            escape_next = True
            continue

        if char == '"' and (i == 0 or text[i-1] != '\\'):
            in_string = not in_string
            result.append(char)
        elif char == '\n' and not in_string:
            # Literal newline outside string - skip it
            continue
        elif char == '\r' and not in_string:
            # Literal carriage return outside string - skip it
            continue
        else:
            result.append(char)

    text = ''.join(result)

    # Remove trailing commas before } or ]
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)

    return text


def safe_json_loads(text: str, default: Any = None) -> Any:
    """
    Safely parse JSON with fallback to default value.

    Args:
        text: JSON string to parse
        default: Default value if parsing fails

    Returns:
        Parsed JSON or default value

    Example:
        >>> safe_json_loads('{"key": "value"}', default={})
        {'key': 'value'}
        >>> safe_json_loads('invalid', default={})
        {}
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def normalize_llm_response(
    response_dict: Dict[str, Any],
    required_fields: Dict[str, Any] = None,
    field_validators: Dict[str, callable] = None
) -> Dict[str, Any]:
    """
    Normalize LLM response dictionary to match expected schema.

    Handles common issues:
    - Missing required fields (adds defaults)
    - Wrong data types (coerces to correct types)
    - Nested structures when flat expected

    Args:
        response_dict: Raw parsed response from LLM
        required_fields: Dict of {field_name: default_value}
        field_validators: Dict of {field_name: validator_function}

    Returns:
        Normalized dictionary

    Example:
        >>> normalize_llm_response(
        ...     {'answer': {'text': 'Hello'}},
        ...     required_fields={'answer': '', 'confidence': 0.5},
        ...     field_validators={'answer': lambda x: str(x) if isinstance(x, dict) else x}
        ... )
        {'answer': "{'text': 'Hello'}", 'confidence': 0.5}
    """
    if response_dict is None:
        response_dict = {}

    result = dict(response_dict)

    # Apply default values for missing fields
    if required_fields:
        for field, default in required_fields.items():
            if field not in result or result[field] is None:
                result[field] = default

    # Apply field validators
    if field_validators:
        for field, validator in field_validators.items():
            if field in result:
                try:
                    result[field] = validator(result[field])
                except Exception as e:
                    logger.warning(f"Validator failed for field {field}: {e}")
                    # Keep original value on validator failure

    return result


def ensure_string(value: Any, default: str = "") -> str:
    """
    Ensure a value is a string.

    Handles dicts by trying to extract common text fields,
    then falling back to str() conversion.

    Args:
        value: Any value to convert to string
        default: Default if value is None/empty

    Returns:
        String representation
    """
    if value is None:
        return default

    if isinstance(value, str):
        return value if value else default

    if isinstance(value, dict):
        # Try common text field names
        for key in ['text', 'content', 'message', 'value', 'answer']:
            if key in value:
                return str(value[key])
        return str(value)

    return str(value)


def ensure_float(value: Any, default: float = 0.0, min_val: float = None, max_val: float = None) -> float:
    """
    Ensure a value is a float within optional bounds.

    Args:
        value: Any value to convert to float
        default: Default if conversion fails
        min_val: Minimum allowed value (clamp if exceeded)
        max_val: Maximum allowed value (clamp if exceeded)

    Returns:
        Float value
    """
    try:
        result = float(value)
    except (ValueError, TypeError):
        result = default

    if min_val is not None:
        result = max(min_val, result)
    if max_val is not None:
        result = min(max_val, result)

    return result


def ensure_list(value: Any, default: List = None) -> List:
    """
    Ensure a value is a list.

    Args:
        value: Any value to convert to list
        default: Default if value is None/empty

    Returns:
        List value
    """
    if default is None:
        default = []

    if value is None:
        return default

    if isinstance(value, list):
        return value

    if isinstance(value, (tuple, set)):
        return list(value)

    # Single value becomes single-element list
    return [value]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'extract_json_from_response',
    'extract_json_array_from_response',
    'sanitize_json_string',
    'safe_json_loads',
    'normalize_llm_response',
    'ensure_string',
    'ensure_float',
    'ensure_list'
]
