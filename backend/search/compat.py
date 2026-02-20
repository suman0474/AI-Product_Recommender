# search/compat.py
# =============================================================================
# BACKWARD COMPATIBILITY WRAPPERS
# =============================================================================
#
# Provides class-based wrappers that delegate to the new functional API.
# This maintains compatibility with code that uses the old class-based interface.
#
# =============================================================================

import logging
from typing import Dict, Any, List, Optional

from .entry_points import (
    run_product_search_workflow,
    run_validation_only,
    run_advanced_params_only,
    run_analysis_only,
    process_from_instrument_identifier,
    process_from_solution_workflow,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CLASS WRAPPERS
# =============================================================================

class ProductSearchWorkflow:
    """
    Backward-compatible class wrapper for product search workflow.

    Delegates all operations to the new functional API.
    """

    def __init__(
        self,
        enable_ppi_workflow: bool = True,
        auto_mode: bool = True,
        max_vendor_workers: int = 5,
    ):
        """
        Initialize the ProductSearchWorkflow.

        Args:
            enable_ppi_workflow: Enable PPI schema generation
            auto_mode: Run in auto mode (skip HITL)
            max_vendor_workers: Max parallel workers for vendor analysis
        """
        self.enable_ppi_workflow = enable_ppi_workflow
        self.auto_mode = auto_mode
        self.max_vendor_workers = max_vendor_workers

    def run_vendor_analysis_step(
        self,
        product_type: str,
        requirements: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run vendor analysis step only."""
        return run_analysis_only(
            structured_requirements=requirements,
            product_type=product_type,
            schema=schema,
            session_id=session_id,
            max_vendor_workers=self.max_vendor_workers,
        )

    def run_ranking_step(
        self,
        product_type: str,
        requirements: Dict[str, Any],
        vendor_analysis: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run ranking step only (included in analysis)."""
        # Ranking is now part of analysis_only
        return run_analysis_only(
            structured_requirements=requirements,
            product_type=product_type,
            session_id=session_id,
            max_vendor_workers=self.max_vendor_workers,
        )

    def run_single_product_workflow(
        self,
        user_input: str,
        expected_product_type: Optional[str] = None,
        session_id: Optional[str] = None,
        skip_advanced_params: bool = False,
    ) -> Dict[str, Any]:
        """Run full workflow for single product."""
        return run_product_search_workflow(
            user_input=user_input,
            session_id=session_id,
            expected_product_type=expected_product_type,
            auto_mode=self.auto_mode,
            skip_advanced_params=skip_advanced_params,
            max_vendor_workers=self.max_vendor_workers,
        )

    def run_analysis_only(
        self,
        structured_requirements: Dict[str, Any],
        product_type: str,
        schema: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run analysis only (vendor + ranking)."""
        return run_analysis_only(
            structured_requirements=structured_requirements,
            product_type=product_type,
            schema=schema,
            session_id=session_id,
            max_vendor_workers=self.max_vendor_workers,
        )

    def process_from_instrument_identifier(
        self,
        identifier_output: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process from instrument identifier output."""
        return process_from_instrument_identifier(
            identifier_output=identifier_output,
            session_id=session_id,
        )

    def process_from_solution_workflow(
        self,
        solution_output: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process from solution workflow output."""
        return process_from_solution_workflow(
            solution_output=solution_output,
            session_id=session_id,
        )


class ValidationTool:
    """Backward-compatible validation tool wrapper."""

    def __init__(self, enable_ppi: bool = True):
        """Initialize ValidationTool."""
        self.enable_ppi = enable_ppi

    def validate(
        self,
        user_input: str,
        expected_product_type: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate user input."""
        return run_validation_only(
            user_input=user_input,
            expected_product_type=expected_product_type,
            session_id=session_id,
            enable_ppi=self.enable_ppi,
        )


class AdvancedParametersTool:
    """Backward-compatible advanced parameters tool wrapper."""

    def __init__(self):
        """Initialize AdvancedParametersTool."""
        pass

    def discover(
        self,
        product_type: str,
        session_id: Optional[str] = None,
        existing_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Discover advanced parameters."""
        return run_advanced_params_only(
            product_type=product_type,
            session_id=session_id,
            schema=existing_schema,
        )


class VendorAnalysisTool:
    """Backward-compatible vendor analysis tool wrapper."""

    def __init__(self, max_workers: int = 5):
        """Initialize VendorAnalysisTool."""
        self.max_workers = max_workers

    def analyze(
        self,
        product_type: str,
        structured_requirements: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze vendors."""
        result = run_analysis_only(
            structured_requirements=structured_requirements,
            product_type=product_type,
            schema=schema,
            session_id=session_id,
            max_vendor_workers=self.max_workers,
        )
        return result.get("vendor_analysis", result)


class RankingTool:
    """Backward-compatible ranking tool wrapper."""

    def __init__(self):
        """Initialize RankingTool."""
        pass

    def rank(
        self,
        product_type: str,
        structured_requirements: Dict[str, Any],
        vendor_analysis: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Rank products (included in analysis)."""
        result = run_analysis_only(
            structured_requirements=structured_requirements,
            product_type=product_type,
            session_id=session_id,
        )
        return {
            "overall_ranking": result.get("overall_ranking", []),
            "top_product": result.get("top_product"),
            "ranking_summary": result.get("ranking_summary", ""),
        }


# =============================================================================
# MODULE FUNCTION WRAPPERS
# =============================================================================

def analyze_vendors(
    product_type: str,
    structured_requirements: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    max_workers: int = 5,
) -> Dict[str, Any]:
    """Convenience function for vendor analysis."""
    result = run_analysis_only(
        structured_requirements=structured_requirements,
        product_type=product_type,
        schema=schema,
        session_id=session_id,
        max_vendor_workers=max_workers,
    )
    return result.get("vendor_analysis", result)


def rank_products(
    vendor_analysis: Dict[str, Any],
    structured_requirements: Dict[str, Any],
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function for product ranking."""
    from .agents import RankingAgent

    agent = RankingAgent()
    result = agent.rank(
        vendor_analysis=vendor_analysis,
        requirements=structured_requirements,
    )

    return result.to_dict()
