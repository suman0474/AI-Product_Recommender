# search/agents/ranking_agent.py
# =============================================================================
# RANKING AGENT
# =============================================================================
#
# Handles product ranking using LLM or score-based fallback.
# Normalizes ranking output and enriches with vendor data.
#
# =============================================================================

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class RankingResult:
    """Result from the ranking agent."""

    success: bool
    overall_ranking: List[Dict[str, Any]]
    top_product: Optional[Dict[str, Any]]
    total_ranked: int
    ranking_summary: str
    ranking_method: str  # "llm" | "score_fallback"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "success": self.success,
            "overall_ranking": self.overall_ranking,
            "top_product": self.top_product,
            "total_ranked": self.total_ranked,
            "ranking_summary": self.ranking_summary,
            "ranking_method": self.ranking_method,
            "error": self.error,
        }


class RankingAgent:
    """
    Handles product ranking using LLM or score-based fallback.

    This agent takes vendor analysis results and produces a ranked
    list of products based on how well they match the requirements.
    """

    def __init__(self):
        """Initialize the RankingAgent."""
        self._components = None

    def rank(
        self,
        vendor_analysis: Dict[str, Any],
        requirements: Optional[Dict[str, Any]] = None,
        use_llm: bool = True,
        session_id: Optional[str] = None,
    ) -> RankingResult:
        """
        Rank products from vendor analysis.

        Args:
            vendor_analysis: Vendor analysis result with matches
            requirements: Original requirements for context
            use_llm: Whether to use LLM ranking (vs score-based)
            session_id: Session identifier

        Returns:
            RankingResult with ranked products
        """
        logger.info("[RankingAgent] Starting product ranking")

        try:
            # Get vendor matches
            vendor_matches = vendor_analysis.get("vendor_matches", [])

            if not vendor_matches:
                return RankingResult(
                    success=True,
                    overall_ranking=[],
                    top_product=None,
                    total_ranked=0,
                    ranking_summary="No products to rank",
                    ranking_method="none",
                )

            # Try LLM ranking first
            if use_llm:
                try:
                    ranked = self._llm_rank(vendor_matches, requirements)
                    if ranked:
                        return self._build_result(ranked, "llm")
                except Exception as exc:
                    logger.warning("[RankingAgent] LLM ranking failed, using fallback: %s", exc)

            # Fallback to score-based ranking
            ranked = self._score_fallback(vendor_matches)
            return self._build_result(ranked, "score_fallback")

        except Exception as exc:
            logger.error("[RankingAgent] Ranking failed: %s", exc, exc_info=True)
            return RankingResult(
                success=False,
                overall_ranking=[],
                top_product=None,
                total_ranked=0,
                ranking_summary=f"Ranking failed: {str(exc)}",
                ranking_method="error",
                error=str(exc),
            )

    def _llm_rank(
        self,
        vendor_matches: List[Dict[str, Any]],
        requirements: Optional[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Use LLM to rank products."""
        logger.debug("[RankingAgent] Attempting LLM ranking")

        try:
            from common.core.chaining import invoke_ranking_chain, setup_langchain_components

            # Setup components
            components = setup_langchain_components()

            # Invoke ranking chain
            result = invoke_ranking_chain(
                components=components,
                vendor_matches=vendor_matches,
                requirements=requirements or {},
            )

            if not result:
                return None

            # Extract ranked products
            # Key fix: ranking returns "overall_ranking" NOT "ranked_results"
            ranked = None
            for key in ["overall_ranking", "ranked_products", "rankedProducts", "results"]:
                if key in result and isinstance(result[key], list):
                    ranked = result[key]
                    break

            if not ranked:
                logger.warning("[RankingAgent] No ranked products in LLM result")
                return None

            # Normalize the ranked products
            normalized = self._normalize_ranked_products(ranked, vendor_matches)
            logger.info("[RankingAgent] LLM ranked %d products", len(normalized))

            return normalized

        except ImportError:
            logger.warning("[RankingAgent] Chaining module not available")
            return None
        except Exception as exc:
            logger.warning("[RankingAgent] LLM ranking error: %s", exc)
            return None

    def _score_fallback(
        self,
        vendor_matches: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Score-based fallback ranking."""
        logger.info("[RankingAgent] Using score-based fallback ranking")

        # Sort by matchScore or overallScore
        sorted_matches = sorted(
            vendor_matches,
            key=lambda x: x.get("overallScore") or x.get("matchScore", 0),
            reverse=True,
        )

        # Add rank numbers and normalize
        ranked = []
        for i, match in enumerate(sorted_matches):
            ranked.append({
                "rank": i + 1,
                "vendor": match.get("vendor", "Unknown"),
                "productName": match.get("productName") or match.get("model", "Unknown"),
                "model": match.get("model") or match.get("productName", ""),
                "overallScore": match.get("overallScore") or match.get("matchScore", 0),
                "matchScore": match.get("matchScore", 0),
                "strengths": match.get("strengths", []),
                "concerns": match.get("concerns", []),
                "recommendation": match.get("recommendation", ""),
                "matchedRequirements": match.get("matchedRequirements", []),
                "unmatchedRequirements": match.get("unmatchedRequirements", []),
                "specifications": match.get("specifications", {}),
                "strategyPriority": match.get("strategyPriority", 0),
                "isPreferredVendor": match.get("isPreferredVendor", False),
            })

        return ranked

    def _normalize_ranked_products(
        self,
        ranking_list: List[Dict[str, Any]],
        vendor_matches: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Normalize LLM ranking output to standard format.

        The LLM ranking may use different field names, so we normalize
        and enrich with data from vendor_matches.
        """
        # Build lookup from vendor matches
        match_lookup = {}
        for match in vendor_matches:
            vendor = match.get("vendor", "")
            model = match.get("model") or match.get("productName", "")
            key = f"{vendor}:{model}".lower()
            match_lookup[key] = match

        normalized = []
        for i, item in enumerate(ranking_list):
            # Extract product info - handle various field names
            vendor = item.get("vendor") or item.get("vendorName", "Unknown")
            product_name = (
                item.get("productName") or
                item.get("model") or
                item.get("name") or
                item.get("product", "Unknown")
            )
            model = (
                item.get("model") or
                item.get("productName") or
                item.get("name", "")
            )

            # Try to find matching vendor data
            lookup_key = f"{vendor}:{model}".lower()
            original_match = match_lookup.get(lookup_key, {})

            # Build normalized entry
            normalized.append({
                "rank": item.get("rank", i + 1),
                "vendor": vendor,
                "productName": product_name,
                "model": model,
                "overallScore": (
                    item.get("overallScore") or
                    item.get("overall_score") or
                    item.get("score") or
                    original_match.get("overallScore", 0)
                ),
                "matchScore": (
                    item.get("matchScore") or
                    item.get("match_score") or
                    original_match.get("matchScore", 0)
                ),
                "strengths": (
                    item.get("strengths") or
                    item.get("keyStrengths") or
                    original_match.get("strengths", [])
                ),
                "concerns": (
                    item.get("concerns") or
                    item.get("limitations") or
                    original_match.get("concerns", [])
                ),
                "recommendation": (
                    item.get("recommendation") or
                    item.get("summary") or
                    original_match.get("recommendation", "")
                ),
                "matchedRequirements": original_match.get("matchedRequirements", []),
                "unmatchedRequirements": original_match.get("unmatchedRequirements", []),
                "specifications": original_match.get("specifications", {}),
                "strategyPriority": original_match.get("strategyPriority", 0),
                "isPreferredVendor": original_match.get("isPreferredVendor", False),
                # Additional LLM insights
                "reasoning": self._normalize_reasoning(item.get("reasoning", "")),
            })

        # Sort by rank (should already be sorted, but ensure)
        normalized.sort(key=lambda x: x.get("rank", 999))

        return normalized

    def _normalize_reasoning(
        self,
        field_data: Union[str, List[Any], Dict[str, Any]],
    ) -> str:
        """Convert complex LLM reasoning output to clean markdown."""
        if isinstance(field_data, str):
            return field_data

        if isinstance(field_data, list):
            lines = []
            for item in field_data:
                if isinstance(item, dict):
                    # Handle dict items with various keys
                    param = item.get("parameter", "")
                    value = item.get("input_value", "")
                    explanation = item.get("holistic_explanation") or item.get("explanation", "")

                    if param and explanation:
                        lines.append(f"**{param}**: {value} - {explanation}")
                    elif explanation:
                        lines.append(f"- {explanation}")

                elif isinstance(item, str):
                    lines.append(f"- {item}")

            return "\n".join(lines)

        if isinstance(field_data, dict):
            # Try to extract meaningful text from dict
            for key in ["explanation", "reasoning", "summary", "text"]:
                if key in field_data:
                    return str(field_data[key])

        return str(field_data)

    def _build_result(
        self,
        ranked: List[Dict[str, Any]],
        method: str,
    ) -> RankingResult:
        """Build the ranking result from ranked list."""
        if not ranked:
            return RankingResult(
                success=True,
                overall_ranking=[],
                top_product=None,
                total_ranked=0,
                ranking_summary="No products ranked",
                ranking_method=method,
            )

        # Get top product
        top_product = ranked[0] if ranked else None

        # Build summary
        if top_product:
            summary = (
                f"Ranked {len(ranked)} products. "
                f"Top recommendation: {top_product.get('productName', 'Unknown')} "
                f"by {top_product.get('vendor', 'Unknown')} "
                f"with {top_product.get('overallScore', 0)}% match score."
            )
        else:
            summary = f"Ranked {len(ranked)} products."

        return RankingResult(
            success=True,
            overall_ranking=ranked,
            top_product=top_product,
            total_ranked=len(ranked),
            ranking_summary=summary,
            ranking_method=method,
        )
