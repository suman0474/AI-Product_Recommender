"""
Error Suggestion Generator

Generates context-aware, helpful suggestions when queries fail to find results.
This module analyzes the query and available context to provide actionable
alternatives instead of generic error messages.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    """Analysis of a user query for suggestion generation."""
    original_query: str
    detected_vendor: Optional[str] = None
    detected_product_type: Optional[str] = None
    detected_model: Optional[str] = None
    detected_standard: Optional[str] = None
    query_type: str = "general"  # product, standard, vendor, comparison


# Common vendor names for detection
KNOWN_VENDORS = [
    "honeywell", "siemens", "emerson", "rosemount", "yokogawa", "abb",
    "endress hauser", "endress+hauser", "e+h", "fisher", "foxboro",
    "krohne", "vega", "gefran", "wika", "ashcroft", "dwyer",
    "schneider", "rockwell", "allen bradley", "danfoss", "parker",
    "swagelok", "flowserve", "samson", "masoneilan", "metso", "neles"
]

# Common product types for detection
PRODUCT_TYPES = [
    "transmitter", "sensor", "valve", "gauge", "meter", "thermocouple",
    "rtd", "flow meter", "level sensor", "pressure sensor", "analyzer",
    "controller", "positioner", "actuator", "switch", "indicator"
]

# Common standards prefixes
STANDARD_PATTERNS = [
    r"IEC\s*\d+", r"ISA\s*\d+", r"API\s*\d+", r"ISO\s*\d+",
    r"ASME\s*\w+", r"ATEX", r"SIL\s*\d", r"NFPA\s*\d+"
]


class ErrorSuggestionGenerator:
    """
    Generates context-aware suggestions for failed queries.

    Instead of returning generic "no results found" messages, this class
    analyzes the query to provide helpful, actionable alternatives.
    """

    def __init__(self):
        self.vendors = KNOWN_VENDORS
        self.product_types = PRODUCT_TYPES

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a query to extract key components.

        Args:
            query: User query text

        Returns:
            QueryAnalysis with detected components
        """
        query_lower = query.lower()
        analysis = QueryAnalysis(original_query=query)

        # Detect vendor
        for vendor in self.vendors:
            if vendor in query_lower:
                analysis.detected_vendor = vendor.title()
                break

        # Detect product type
        for product_type in self.product_types:
            if product_type in query_lower:
                analysis.detected_product_type = product_type.title()
                break

        # Detect standards
        for pattern in STANDARD_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                analysis.detected_standard = match.group().upper()
                break

        # Detect model numbers (alphanumeric patterns)
        model_pattern = r'\b[A-Z]{1,4}[-\s]?\d{2,5}[A-Z]?\b'
        model_match = re.search(model_pattern, query, re.IGNORECASE)
        if model_match:
            analysis.detected_model = model_match.group().upper()

        # Determine query type
        if analysis.detected_standard:
            analysis.query_type = "standard"
        elif "compare" in query_lower or "versus" in query_lower or " vs " in query_lower:
            analysis.query_type = "comparison"
        elif analysis.detected_vendor and not analysis.detected_product_type:
            analysis.query_type = "vendor"
        else:
            analysis.query_type = "product"

        return analysis

    def generate_suggestions(
        self,
        query: str,
        error_type: str = "no_results",
        sources_tried: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate helpful suggestions based on query analysis.

        Args:
            query: Original user query
            error_type: Type of error (no_results, partial, timeout)
            sources_tried: List of data sources that were queried

        Returns:
            List of suggestion strings
        """
        analysis = self.analyze_query(query)
        suggestions = []

        if analysis.query_type == "product":
            suggestions.extend(self._product_suggestions(analysis))
        elif analysis.query_type == "standard":
            suggestions.extend(self._standard_suggestions(analysis))
        elif analysis.query_type == "vendor":
            suggestions.extend(self._vendor_suggestions(analysis))
        elif analysis.query_type == "comparison":
            suggestions.extend(self._comparison_suggestions(analysis))

        # Add general suggestions if needed
        if len(suggestions) < 2:
            suggestions.extend(self._general_suggestions(analysis))

        return suggestions[:4]  # Return max 4 suggestions

    def _product_suggestions(self, analysis: QueryAnalysis) -> List[str]:
        """Generate suggestions for product queries."""
        suggestions = []

        if analysis.detected_model:
            suggestions.append(
                f"Try searching for the product category instead of model '{analysis.detected_model}'"
            )

        if analysis.detected_vendor:
            if analysis.detected_product_type:
                suggestions.append(
                    f"Browse all {analysis.detected_vendor} {analysis.detected_product_type}s in our catalog"
                )
            else:
                suggestions.append(
                    f"Try specifying a product type (e.g., '{analysis.detected_vendor} temperature transmitter')"
                )

        if analysis.detected_product_type:
            suggestions.append(
                f"Search for '{analysis.detected_product_type}' to see all available options"
            )

        suggestions.append("Check the spelling of vendor or product names")

        return suggestions

    def _standard_suggestions(self, analysis: QueryAnalysis) -> List[str]:
        """Generate suggestions for standards queries."""
        suggestions = []

        if analysis.detected_standard:
            suggestions.append(
                f"Search for the exact standard code: '{analysis.detected_standard}'"
            )

        suggestions.extend([
            "Try searching for the standard category (e.g., 'safety standards' or 'pressure measurement standards')",
            "Browse our standards library by domain: IEC, ISA, API, ISO, SIL, ATEX",
            "Ask about specific requirements the standard addresses"
        ])

        return suggestions

    def _vendor_suggestions(self, analysis: QueryAnalysis) -> List[str]:
        """Generate suggestions for vendor queries."""
        suggestions = []

        if analysis.detected_vendor:
            suggestions.append(
                f"Specify what {analysis.detected_vendor} products you're looking for"
            )
            suggestions.append(
                f"Ask about {analysis.detected_vendor} product categories we carry"
            )

        suggestions.append(
            "Try asking about vendor options for a specific instrument type"
        )

        return suggestions

    def _comparison_suggestions(self, analysis: QueryAnalysis) -> List[str]:
        """Generate suggestions for comparison queries."""
        suggestions = [
            "Specify the key criteria you want to compare (accuracy, price, certifications)",
            "Try comparing specific models or product lines",
            "Ask about differences in technical specifications"
        ]

        if analysis.detected_product_type:
            suggestions.insert(0,
                f"Compare specific {analysis.detected_product_type} models by name"
            )

        return suggestions

    def _general_suggestions(self, analysis: QueryAnalysis) -> List[str]:
        """Generate general fallback suggestions."""
        return [
            "Try using more specific product or vendor names",
            "Ask about a specific instrument category (pressure, temperature, flow, level)",
            "Specify the application context for better recommendations"
        ]

    def format_suggestions_as_text(self, suggestions: List[str]) -> str:
        """
        Format suggestions as a readable text block.

        Args:
            suggestions: List of suggestion strings

        Returns:
            Formatted text with bullet points
        """
        if not suggestions:
            return ""

        lines = ["Here are some suggestions:"]
        for suggestion in suggestions:
            lines.append(f"  - {suggestion}")

        return "\n".join(lines)


def generate_suggestions_for_query(
    query: str,
    error_type: str = "no_results",
    sources_tried: Optional[List[str]] = None
) -> Tuple[List[str], str]:
    """
    Convenience function to generate suggestions for a query.

    Args:
        query: User query text
        error_type: Type of error encountered
        sources_tried: List of sources that were queried

    Returns:
        Tuple of (suggestions list, formatted text)
    """
    generator = ErrorSuggestionGenerator()
    suggestions = generator.generate_suggestions(query, error_type, sources_tried)
    formatted = generator.format_suggestions_as_text(suggestions)
    return suggestions, formatted
