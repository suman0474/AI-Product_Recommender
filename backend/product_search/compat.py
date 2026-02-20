"""
agentic/workflows/product_search/compat.py
==========================================

Backward-compatibility shims for callers that used the old
``product_search_workflow`` class-based API.

These thin wrappers delegate to the new functional deep-agent API so that
existing call-sites continue to work with zero changes.

Importable as:
    from agentic.workflows.product_search.compat import (
        ProductSearchWorkflow,
        ValidationTool,
        AdvancedParametersTool,
        VendorAnalysisTool,
        RankingTool,
        analyze_vendors,
        rank_products,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from product_search.product_search_workflow import (
    run_product_search_workflow,
    run_single_product_workflow,
    run_analysis_only,
    run_validation_only,
    run_advanced_params_only,
    process_from_instrument_identifier,
    process_from_solution_workflow,
    product_search_workflow,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ProductSearchWorkflow  ← replaces the old workflow.py class
# ---------------------------------------------------------------------------
class ProductSearchWorkflow:
    """
    Backward-compatible class wrapper around the new functional deep-agent API.

    Constructor signature is intentionally identical to the old class.
    """

    def __init__(
        self,
        enable_ppi_workflow: bool = True,
        auto_mode: bool = True,
        max_vendor_workers: int = 5,
    ):
        self.enable_ppi_workflow = enable_ppi_workflow
        self.auto_mode = auto_mode
        self.max_vendor_workers = max_vendor_workers
        logger.info(
            "[ProductSearchWorkflow(compat)] Initialized — "
            f"ppi={enable_ppi_workflow} auto={auto_mode} workers={max_vendor_workers}"
        )

    def run_vendor_analysis_step(
        self,
        product_type: str,
        requirements: Dict[str, Any],
        schema: Optional[Dict] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Shim for backward compatibility."""
        return run_analysis_only(
            structured_requirements=requirements,
            product_type=product_type,
            schema=schema,
            session_id=session_id,
        )

    def run_ranking_step(
        self,
        product_type: str,
        requirements: Dict[str, Any],
        vendor_analysis: Optional[Dict] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Shim for backward compatibility."""
        # Note: vendor_analysis is ignored as run_analysis_only re-runs internal analysis or uses checkpoint
        return run_analysis_only(
            structured_requirements=requirements,
            product_type=product_type,
            session_id=session_id,
        )

    def run_single_product_workflow(
        self,
        user_input: str,
        expected_product_type: Optional[str] = None,
        session_id: Optional[str] = None,
        skip_advanced_params: bool = False,
    ) -> Dict[str, Any]:
        """Run full pipeline for a single product (auto mode)."""
        return run_single_product_workflow(
            user_input=user_input,
            session_id=session_id,
            auto_mode=self.auto_mode,
            enable_ppi=self.enable_ppi_workflow,
            skip_advanced_params=skip_advanced_params,
        )

    def run_analysis_only(
        self,
        structured_requirements: Dict[str, Any],
        product_type: str,
        schema: Optional[Dict] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Skip validation; run vendor analysis + ranking only (steps 4-5)."""
        return run_analysis_only(
            structured_requirements=structured_requirements,
            product_type=product_type,
            schema=schema,
            session_id=session_id,
        )

    def process_from_instrument_identifier(
        self,
        identifier_output: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Batch entry point from Instruments Identifier Workflow."""
        return process_from_instrument_identifier(
            identifier_output=identifier_output,
            session_id=session_id,
        )

    def process_from_solution_workflow(
        self,
        solution_output: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Batch entry point from Solution Workflow."""
        return process_from_solution_workflow(
            solution_output=solution_output,
            session_id=session_id,
        )


# ---------------------------------------------------------------------------
# ValidationTool  ← replaces the old validation_tool.py class
# ---------------------------------------------------------------------------
class ValidationTool:
    """Backward-compatible wrapper around ``run_validation_only()``."""

    def __init__(self, enable_ppi: bool = True, **kwargs):
        self.enable_ppi = enable_ppi

    def validate(
        self,
        user_input: str,
        expected_product_type: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate user input and return schema + missing fields."""
        return run_validation_only(
            user_input=user_input,
            session_id=session_id,
            enable_ppi=self.enable_ppi,
        )


# ---------------------------------------------------------------------------
# AdvancedParametersTool  ← replaces the old advanced_parameters_tool.py class
# ---------------------------------------------------------------------------
class AdvancedParametersTool:
    """Backward-compatible wrapper around ``run_advanced_params_only()``."""

    def __init__(self, **kwargs):
        pass

    def discover(
        self,
        product_type: str,
        session_id: Optional[str] = None,
        existing_schema: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Discover advanced vendor specs for a product type."""
        return run_advanced_params_only(
            product_type=product_type,
            session_id=session_id,
            existing_schema=existing_schema,
        )


# ---------------------------------------------------------------------------
# VendorAnalysisTool  ← replaces the old vendor_analysis_tool.py class
# ---------------------------------------------------------------------------
class VendorAnalysisTool:
    """
    Backward-compatible wrapper.

    Vendor analysis is now embedded inside ``run_analysis_only()`` (steps 4-5).
    """

    def __init__(self, max_workers: int = 5, **kwargs):
        self.max_workers = max_workers

    def analyze(
        self,
        product_type: str,
        structured_requirements: Dict[str, Any],
        schema: Optional[Dict] = None,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return run_analysis_only(
            structured_requirements=structured_requirements,
            product_type=product_type,
            schema=schema,
            session_id=session_id,
        )


def analyze_vendors(
    product_type: str,
    structured_requirements: Dict[str, Any],
    schema: Optional[Dict] = None,
    session_id: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Module-level convenience function — delegates to ``run_analysis_only()``."""
    return run_analysis_only(
        structured_requirements=structured_requirements,
        product_type=product_type,
        schema=schema,
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# RankingTool  ← replaces the old ranking_tool.py class
# ---------------------------------------------------------------------------
class RankingTool:
    """
    Backward-compatible wrapper.

    Ranking is now embedded inside ``run_analysis_only()`` (steps 4-5).
    """

    def __init__(self, **kwargs):
        pass

    def rank(
        self,
        product_type: str,
        structured_requirements: Dict[str, Any],
        vendor_analysis: Optional[Dict] = None,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return run_analysis_only(
            structured_requirements=structured_requirements,
            product_type=product_type,
            session_id=session_id,
        )


def rank_products(
    product_type: str,
    structured_requirements: Dict[str, Any],
    session_id: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Module-level convenience function — delegates to ``run_analysis_only()``."""
    return run_analysis_only(
        structured_requirements=structured_requirements,
        product_type=product_type,
        session_id=session_id,
    )


__all__ = [
    "ProductSearchWorkflow",
    "ValidationTool",
    "AdvancedParametersTool",
    "VendorAnalysisTool",
    "RankingTool",
    "analyze_vendors",
    "rank_products",
]
