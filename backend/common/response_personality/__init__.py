"""
Response Personality Module

Provides domain-aware response personality management for EnGenie.
This module ensures consistent, context-appropriate responses across all chat interactions.

Usage:
    from common.response_personality import PersonalityEngine, ResponseDomain

    # Get personality config for a domain
    engine = PersonalityEngine()
    config = engine.get_config(ResponseDomain.PRODUCTS)

    # Use config.temperature for LLM calls
    # Use config.tone_instructions for prompt enhancement
"""

from .personality_engine import (
    PersonalityEngine,
    ResponseDomain,
    PersonalityConfig,
    get_personality_for_source,
)
from .error_suggestions import (
    ErrorSuggestionGenerator,
    generate_suggestions_for_query,
)

__all__ = [
    "PersonalityEngine",
    "ResponseDomain",
    "PersonalityConfig",
    "get_personality_for_source",
    "ErrorSuggestionGenerator",
    "generate_suggestions_for_query",
]
