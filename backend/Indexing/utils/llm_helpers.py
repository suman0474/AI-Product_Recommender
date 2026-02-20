"""
LLM Helpers for Deep Agent PPI
================================
Provides utilities for LLM interactions, JSON parsing, and response validation.

Canonical implementations are delegated to:
  - LLM creation  → common.services.llm.fallback.create_llm_with_fallback
  - JSON parsing  → common.utils.json_utils.extract_json_from_response
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from common.services.llm.fallback import create_llm_with_fallback
from common.utils.json_utils import extract_json_from_response as _extract_json

logger = logging.getLogger(__name__)

# Model configurations (kept here as PPI-specific constants)
DEFAULT_MODEL = "gemini-2.0-flash-exp"
FALLBACK_MODEL = "gemini-1.5-flash"
REASONING_MODEL = "gemini-2.0-flash-thinking-exp"


def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_retries: int = 3,
    timeout: int = 60
):
    """
    Get configured LLM instance.

    Delegates to `common.services.llm.fallback.create_llm_with_fallback`
    which provides rate limiting, fallback, and timeout support.

    Args:
        model: Model name to use
        temperature: Sampling temperature (0.0-1.0)
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds

    Returns:
        Configured LLM instance
    """
    return create_llm_with_fallback(
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        timeout=timeout,
    )


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from LLM response.

    Delegates to `common.utils.json_utils.extract_json_from_response`
    — the canonical multi-strategy JSON parser.

    Args:
        response: Raw LLM response

    Returns:
        Parsed JSON dict or None if parsing fails
    """
    return _extract_json(response)


def invoke_llm_with_prompt(
    llm,
    system_prompt: str,
    user_prompt: str,
    format_vars: Optional[Dict[str, Any]] = None
) -> str:
    """
    Invoke LLM with system and user prompts.

    Args:
        llm: LLM instance
        system_prompt: System instruction
        user_prompt: User message
        format_vars: Variables to format into prompts

    Returns:
        LLM response content
    """
    try:
        if format_vars:
            system_prompt = system_prompt.format(**format_vars)
            user_prompt = user_prompt.format(**format_vars)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        raise


def extract_json_field(
    response: str,
    field: str,
    default: Any = None
) -> Any:
    """
    Extract specific field from JSON response.

    Args:
        response: Raw LLM response
        field: Field name to extract
        default: Default value if extraction fails

    Returns:
        Field value or default
    """
    parsed = parse_json_response(response)
    if parsed and field in parsed:
        return parsed[field]
    return default


def validate_required_fields(
    data: Dict[str, Any],
    required_fields: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate that required fields are present in data.

    Args:
        data: Data dictionary to validate
        required_fields: List of required field names

    Returns:
        Tuple of (is_valid, missing_fields)
    """
    missing = [f for f in required_fields if f not in data or not data[f]]
    return len(missing) == 0, missing


def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for processing.

    Args:
        text: Input text
        chunk_size: Maximum chunk size in characters
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            para_break = text.rfind('\n\n', start, end)
            if para_break > start:
                end = para_break
            else:
                sent_break = text.rfind('. ', start, end)
                if sent_break > start:
                    end = sent_break + 1

        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def estimate_token_count(text: str) -> int:
    """Rough estimation of token count (~4 chars per token)."""
    return len(text) // 4


def truncate_to_token_limit(text: str, max_tokens: int = 30000) -> str:
    """Truncate text to fit within token limit."""
    if estimate_token_count(text) <= max_tokens:
        return text
    return text[:max_tokens * 4] + "\n\n[... truncated ...]"
