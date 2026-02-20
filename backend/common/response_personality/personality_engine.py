"""
Personality Engine for EnGenie Responses

Provides domain-aware personality configuration for natural, context-appropriate responses.
Each domain (standards, products, strategy, error) has optimized settings for:
- Temperature (creativity vs precision)
- Tone instructions (formal vs friendly)
- Response structure preferences
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ResponseDomain(Enum):
    """
    Response domains with distinct personality requirements.

    Each domain represents a different type of user query and requires
    a tailored response style.
    """
    STANDARDS = "standards"          # Technical standards queries (IEC, ISA, etc.)
    PRODUCTS = "products"            # Product search and specifications
    STRATEGY = "strategy"            # Vendor strategy and procurement
    COMPARISON = "comparison"        # Product/vendor comparisons
    GENERAL = "general"              # General chat and information
    ERROR = "error"                  # Error messages and fallbacks
    EXTRACTION = "extraction"        # Data extraction tasks


@dataclass
class PersonalityConfig:
    """
    Configuration for response personality in a specific domain.

    Attributes:
        domain: The response domain
        temperature: LLM temperature setting (0.0 = precise, 1.0 = creative)
        tone: Descriptive tone label
        tone_instructions: Instructions to include in prompts
        include_suggestions: Whether to include actionable suggestions
        include_alternatives: Whether to suggest alternatives on failure
        max_response_length: Approximate max response length (words)
        emoji_allowed: Whether to use emojis in responses
    """
    domain: ResponseDomain
    temperature: float
    tone: str
    tone_instructions: str
    include_suggestions: bool = True
    include_alternatives: bool = True
    max_response_length: int = 200
    emoji_allowed: bool = False
    citation_style: str = "inline"  # inline, footnote, or none


# Domain-specific personality configurations
DOMAIN_CONFIGS: Dict[ResponseDomain, PersonalityConfig] = {
    ResponseDomain.STANDARDS: PersonalityConfig(
        domain=ResponseDomain.STANDARDS,
        temperature=0.2,
        tone="technical",
        tone_instructions=(
            "Respond in a professional, technically precise manner. "
            "Reference specific standard codes and sections when available. "
            "Be authoritative but accessible."
        ),
        include_suggestions=True,
        include_alternatives=True,
        max_response_length=300,
        emoji_allowed=False,
        citation_style="inline"
    ),

    ResponseDomain.PRODUCTS: PersonalityConfig(
        domain=ResponseDomain.PRODUCTS,
        temperature=0.5,
        tone="helpful",
        tone_instructions=(
            "Respond in a friendly, helpful manner. "
            "Focus on practical information and key specifications. "
            "Be descriptive but concise."
        ),
        include_suggestions=True,
        include_alternatives=True,
        max_response_length=250,
        emoji_allowed=False,
        citation_style="none"
    ),

    ResponseDomain.STRATEGY: PersonalityConfig(
        domain=ResponseDomain.STRATEGY,
        temperature=0.3,
        tone="advisory",
        tone_instructions=(
            "Respond in a professional, advisory tone. "
            "Provide strategic recommendations based on data. "
            "Be objective and business-focused."
        ),
        include_suggestions=True,
        include_alternatives=True,
        max_response_length=300,
        emoji_allowed=False,
        citation_style="footnote"
    ),

    ResponseDomain.COMPARISON: PersonalityConfig(
        domain=ResponseDomain.COMPARISON,
        temperature=0.3,
        tone="analytical",
        tone_instructions=(
            "Respond in an analytical, balanced manner. "
            "Present comparisons objectively with clear criteria. "
            "Highlight key differences and trade-offs."
        ),
        include_suggestions=True,
        include_alternatives=True,
        max_response_length=400,
        emoji_allowed=False,
        citation_style="inline"
    ),

    ResponseDomain.GENERAL: PersonalityConfig(
        domain=ResponseDomain.GENERAL,
        temperature=0.7,
        tone="conversational",
        tone_instructions=(
            "Respond in a warm, conversational manner. "
            "Be friendly and approachable while remaining informative. "
            "Adapt to the user's communication style."
        ),
        include_suggestions=False,
        include_alternatives=False,
        max_response_length=200,
        emoji_allowed=False,
        citation_style="none"
    ),

    ResponseDomain.ERROR: PersonalityConfig(
        domain=ResponseDomain.ERROR,
        temperature=0.7,
        tone="empathetic",
        tone_instructions=(
            "Respond with empathy and helpfulness. "
            "Acknowledge the user's intent, explain what happened, "
            "and provide clear, actionable next steps. "
            "Never be apologetic or negative - be solution-focused."
        ),
        include_suggestions=True,
        include_alternatives=True,
        max_response_length=150,
        emoji_allowed=False,
        citation_style="none"
    ),

    ResponseDomain.EXTRACTION: PersonalityConfig(
        domain=ResponseDomain.EXTRACTION,
        temperature=0.0,
        tone="precise",
        tone_instructions=(
            "Extract information precisely and consistently. "
            "Follow the exact output format required. "
            "Do not add explanations or commentary."
        ),
        include_suggestions=False,
        include_alternatives=False,
        max_response_length=100,
        emoji_allowed=False,
        citation_style="none"
    ),
}


class PersonalityEngine:
    """
    Manages response personality based on query context and domain.

    This engine provides consistent personality configuration across
    all EnGenie response generation, ensuring natural and appropriate
    responses for each type of query.
    """

    def __init__(self, custom_configs: Optional[Dict[ResponseDomain, PersonalityConfig]] = None):
        """
        Initialize the personality engine.

        Args:
            custom_configs: Optional custom configurations to override defaults
        """
        self.configs = DOMAIN_CONFIGS.copy()
        if custom_configs:
            self.configs.update(custom_configs)

    def get_config(self, domain: ResponseDomain) -> PersonalityConfig:
        """
        Get personality configuration for a domain.

        Args:
            domain: The response domain

        Returns:
            PersonalityConfig for the domain
        """
        return self.configs.get(domain, self.configs[ResponseDomain.GENERAL])

    def get_temperature(self, domain: ResponseDomain) -> float:
        """Get temperature setting for a domain."""
        return self.get_config(domain).temperature

    def get_tone_instructions(self, domain: ResponseDomain) -> str:
        """Get tone instructions for prompt enhancement."""
        return self.get_config(domain).tone_instructions

    def detect_domain(self, query: str, primary_source: Optional[str] = None) -> ResponseDomain:
        """
        Detect the appropriate domain from query content and source.

        Args:
            query: User query text
            primary_source: Primary data source being used

        Returns:
            Detected ResponseDomain
        """
        query_lower = query.lower()

        # Check primary source first
        if primary_source:
            source_mapping = {
                "standards_rag": ResponseDomain.STANDARDS,
                "index_rag": ResponseDomain.PRODUCTS,
                "strategy_rag": ResponseDomain.STRATEGY,
                "deep_agent": ResponseDomain.EXTRACTION,
            }
            if primary_source in source_mapping:
                return source_mapping[primary_source]

        # Keyword-based detection
        standards_keywords = [
            "standard", "iec", "isa", "api", "iso", "atex", "sil",
            "certification", "compliance", "regulation"
        ]
        if any(kw in query_lower for kw in standards_keywords):
            return ResponseDomain.STANDARDS

        strategy_keywords = [
            "vendor", "supplier", "procurement", "strategy", "preferred",
            "refinery", "facility", "approved"
        ]
        if any(kw in query_lower for kw in strategy_keywords):
            return ResponseDomain.STRATEGY

        comparison_keywords = [
            "compare", "comparison", "versus", "vs", "difference",
            "better", "which one", "pros and cons"
        ]
        if any(kw in query_lower for kw in comparison_keywords):
            return ResponseDomain.COMPARISON

        product_keywords = [
            "transmitter", "sensor", "valve", "meter", "gauge",
            "instrument", "product", "model", "specification"
        ]
        if any(kw in query_lower for kw in product_keywords):
            return ResponseDomain.PRODUCTS

        return ResponseDomain.GENERAL

    def enhance_prompt(self, prompt: str, domain: ResponseDomain) -> str:
        """
        Enhance a prompt with personality-specific instructions.

        Args:
            prompt: Original prompt
            domain: Target domain

        Returns:
            Enhanced prompt with tone instructions
        """
        config = self.get_config(domain)
        tone_prefix = f"[Response Style: {config.tone}]\n{config.tone_instructions}\n\n"
        return tone_prefix + prompt


def get_personality_for_source(source: str) -> PersonalityConfig:
    """
    Convenience function to get personality config for a data source.

    Args:
        source: Data source name (e.g., "standards_rag", "index_rag")

    Returns:
        PersonalityConfig for the source
    """
    engine = PersonalityEngine()

    source_mapping = {
        "standards_rag": ResponseDomain.STANDARDS,
        "index_rag": ResponseDomain.PRODUCTS,
        "strategy_rag": ResponseDomain.STRATEGY,
        "deep_agent": ResponseDomain.EXTRACTION,
        "llm": ResponseDomain.GENERAL,
    }

    domain = source_mapping.get(source, ResponseDomain.GENERAL)
    return engine.get_config(domain)
