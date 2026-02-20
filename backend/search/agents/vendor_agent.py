# search/agents/vendor_agent.py
# =============================================================================
# VENDOR ANALYSIS AGENT
# =============================================================================
#
# Handles vendor analysis orchestration including:
# - Loading vendors from Azure Blob
# - Applying Strategy RAG filtering
# - Parallel vendor analysis with ThreadPoolExecutor
# - Result enrichment with images, logos, pricing
#
# =============================================================================

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VendorAnalysisResult:
    """Result from the vendor analysis agent."""

    success: bool
    vendor_matches: List[Dict[str, Any]]
    vendor_run_details: List[Dict[str, Any]]
    strategy_context: Optional[Dict[str, Any]]
    total_matches: int
    vendors_analyzed: int
    original_vendor_count: int
    filtered_vendor_count: int
    excluded_by_strategy: List[str]
    analysis_summary: str
    error: Optional[str] = None
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "success": self.success,
            "vendor_matches": self.vendor_matches,
            "vendor_run_details": self.vendor_run_details,
            "strategy_context": self.strategy_context,
            "total_matches": self.total_matches,
            "vendors_analyzed": self.vendors_analyzed,
            "original_vendor_count": self.original_vendor_count,
            "filtered_vendor_count": self.filtered_vendor_count,
            "excluded_by_strategy": self.excluded_by_strategy,
            "analysis_summary": self.analysis_summary,
            "error": self.error,
            "cached": self.cached,
        }


class VendorAgent:
    """
    Handles vendor analysis orchestration.

    This agent manages the complete vendor analysis pipeline including
    loading data, filtering, parallel analysis, and result enrichment.
    """

    def __init__(self, max_workers: int = 10, max_retries: int = 3):
        """
        Initialize the VendorAgent.

        Args:
            max_workers: Maximum parallel workers for vendor analysis
            max_retries: Maximum retries per vendor on rate limit
        """
        self.max_workers = max_workers
        self.max_retries = max_retries
        self._components = None

    def analyze(
        self,
        requirements: Dict[str, Any],
        product_type: str,
        schema: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> VendorAnalysisResult:
        """
        Run the complete vendor analysis pipeline.

        Args:
            requirements: Structured requirements dictionary
            product_type: The product type being searched
            schema: Optional schema for context
            session_id: Session identifier for caching
            use_cache: Whether to use response cache

        Returns:
            VendorAnalysisResult with all analysis data
        """
        logger.info("[VendorAgent] Starting vendor analysis for: %s", product_type)
        start_time = time.time()

        try:
            # Check cache first
            if use_cache:
                from ..caching import get_vendor_response_cache, compute_vendor_cache_key

                cache = get_vendor_response_cache()
                cache_key = compute_vendor_cache_key(product_type, requirements)
                cached_result = cache.get(cache_key)

                if cached_result:
                    logger.info("[VendorAgent] Cache HIT - returning cached result")
                    cached_result["cached"] = True
                    return VendorAnalysisResult(**cached_result)

            # 1. Load vendors
            all_vendors = self._load_vendors(product_type)
            original_count = len(all_vendors)

            if not all_vendors:
                return VendorAnalysisResult(
                    success=False,
                    vendor_matches=[],
                    vendor_run_details=[],
                    strategy_context=None,
                    total_matches=0,
                    vendors_analyzed=0,
                    original_vendor_count=0,
                    filtered_vendor_count=0,
                    excluded_by_strategy=[],
                    analysis_summary="No vendors found for product type",
                    error="No vendors available",
                )

            # 2. Apply strategy filter
            filtered_vendors, excluded, strategy_context = self._apply_strategy_filter(
                all_vendors, product_type, requirements
            )

            # 3. Load product data
            products_data = self._load_product_data(filtered_vendors, product_type)

            # 4. Setup LLM components
            components = self._setup_components()

            # 5. Build requirements string
            requirements_str = self._format_requirements(requirements)

            # 6. Parallel vendor analysis
            vendor_matches, vendor_run_details = self._analyze_vendors_parallel(
                filtered_vendors,
                products_data,
                requirements_str,
                components,
            )

            # 7. Enrich results
            vendor_matches = self._enrich_results(
                vendor_matches, strategy_context, products_data
            )

            # Build summary
            elapsed = time.time() - start_time
            summary = (
                f"Analyzed {len(filtered_vendors)} vendors in {elapsed:.1f}s. "
                f"Found {len(vendor_matches)} matches."
            )

            result = VendorAnalysisResult(
                success=True,
                vendor_matches=vendor_matches,
                vendor_run_details=vendor_run_details,
                strategy_context=strategy_context,
                total_matches=len(vendor_matches),
                vendors_analyzed=len(filtered_vendors),
                original_vendor_count=original_count,
                filtered_vendor_count=len(filtered_vendors),
                excluded_by_strategy=excluded,
                analysis_summary=summary,
            )

            # Cache result
            if use_cache:
                cache.set(cache_key, result.to_dict())

            logger.info("[VendorAgent] Analysis complete: %s", summary)
            return result

        except Exception as exc:
            logger.error("[VendorAgent] Analysis failed: %s", exc, exc_info=True)
            return VendorAnalysisResult(
                success=False,
                vendor_matches=[],
                vendor_run_details=[],
                strategy_context=None,
                total_matches=0,
                vendors_analyzed=0,
                original_vendor_count=0,
                filtered_vendor_count=0,
                excluded_by_strategy=[],
                analysis_summary=f"Analysis failed: {str(exc)}",
                error=str(exc),
            )

    def _load_vendors(self, product_type: str) -> List[str]:
        """Load vendors for product type from Azure Blob."""
        try:
            from common.services.azure.blob_utils import get_vendors_for_product_type

            vendors = get_vendors_for_product_type(product_type)
            logger.info("[VendorAgent] Loaded %d vendors", len(vendors))
            return vendors

        except ImportError:
            logger.warning("[VendorAgent] Azure blob utils not available")
            return []
        except Exception as exc:
            logger.error("[VendorAgent] Failed to load vendors: %s", exc)
            return []

    def _apply_strategy_filter(
        self,
        vendors: List[str],
        product_type: str,
        requirements: Dict[str, Any],
    ) -> Tuple[List[str], List[str], Optional[Dict[str, Any]]]:
        """Apply Strategy RAG filtering to vendor list."""
        try:
            from common.strategy_rag.strategy_rag_enrichment import (
                get_strategy_with_auto_fallback,
                filter_vendors_by_strategy,
            )

            # Get strategy context
            strategy_context = get_strategy_with_auto_fallback(
                product_type, requirements, top_k=7
            )

            if not strategy_context or not strategy_context.get("success"):
                logger.info("[VendorAgent] No strategy context, using all vendors")
                return (vendors, [], None)

            # Filter vendors
            filter_result = filter_vendors_by_strategy(vendors, strategy_context)
            accepted = filter_result.get("accepted_vendors", vendors)
            excluded = filter_result.get("excluded_vendors", [])

            logger.info(
                "[VendorAgent] Strategy filter: %d accepted, %d excluded",
                len(accepted),
                len(excluded),
            )

            return (accepted, excluded, strategy_context)

        except ImportError:
            logger.warning("[VendorAgent] Strategy RAG not available")
            return (vendors, [], None)
        except Exception as exc:
            logger.warning("[VendorAgent] Strategy filtering failed: %s", exc)
            return (vendors, [], None)

    def _load_product_data(
        self,
        vendors: List[str],
        product_type: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Load product data for vendors."""
        try:
            from common.services.azure.blob_utils import get_products_for_vendors

            products_data = get_products_for_vendors(vendors, product_type)
            logger.info("[VendorAgent] Loaded product data for %d vendors", len(products_data))
            return products_data

        except ImportError:
            logger.warning("[VendorAgent] Azure blob utils not available")
            return {}
        except Exception as exc:
            logger.error("[VendorAgent] Failed to load product data: %s", exc)
            return {}

    def _setup_components(self) -> Dict[str, Any]:
        """Setup LangChain components for vendor analysis."""
        if self._components is not None:
            return self._components

        try:
            from common.core.chaining import setup_langchain_components

            self._components = setup_langchain_components()
            return self._components

        except ImportError:
            logger.warning("[VendorAgent] Chaining module not available")
            return {}
        except Exception as exc:
            logger.warning("[VendorAgent] Failed to setup components: %s", exc)
            return {}

    def _format_requirements(self, requirements: Dict[str, Any]) -> str:
        """Format requirements dictionary into prompt string."""
        lines = ["**Requirements:**"]

        # Group by category
        mandatory = {}
        optional = {}

        for key, value in requirements.items():
            if key.startswith("_"):
                continue

            if key in ["mandatoryRequirements", "mandatory"]:
                if isinstance(value, dict):
                    mandatory.update(value)
            elif key in ["optionalRequirements", "optional"]:
                if isinstance(value, dict):
                    optional.update(value)
            else:
                mandatory[key] = value

        if mandatory:
            lines.append("\n**Mandatory:**")
            for key, value in mandatory.items():
                lines.append(f"- {self._format_key(key)}: {value}")

        if optional:
            lines.append("\n**Optional:**")
            for key, value in optional.items():
                lines.append(f"- {self._format_key(key)}: {value}")

        return "\n".join(lines)

    def _format_key(self, key: str) -> str:
        """Format camelCase key for display."""
        import re
        name = re.sub(r'([A-Z])', r' \1', key)
        return name.strip().title()

    def _analyze_vendors_parallel(
        self,
        vendors: List[str],
        products_data: Dict[str, List[Dict[str, Any]]],
        requirements_str: str,
        components: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run parallel vendor analysis using ThreadPoolExecutor."""
        vendor_matches = []
        vendor_run_details = []

        actual_workers = min(self.max_workers, len(vendors))
        logger.info("[VendorAgent] Starting parallel analysis with %d workers", actual_workers)

        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Submit all vendor analysis tasks
            futures = {
                executor.submit(
                    self._analyze_single_vendor,
                    vendor,
                    products_data.get(vendor, []),
                    requirements_str,
                    components,
                ): vendor
                for vendor in vendors
            }

            # Collect results
            for future in as_completed(futures):
                vendor = futures[future]
                try:
                    result, run_detail = future.result()

                    if result:
                        vendor_matches.extend(result if isinstance(result, list) else [result])

                    vendor_run_details.append(run_detail)

                except Exception as exc:
                    logger.warning("[VendorAgent] Vendor %s failed: %s", vendor, exc)
                    vendor_run_details.append({
                        "vendor": vendor,
                        "success": False,
                        "error": str(exc),
                    })

        logger.info(
            "[VendorAgent] Parallel analysis complete: %d matches from %d vendors",
            len(vendor_matches),
            len(vendors),
        )

        return (vendor_matches, vendor_run_details)

    def _analyze_single_vendor(
        self,
        vendor: str,
        products: List[Dict[str, Any]],
        requirements_str: str,
        components: Dict[str, Any],
    ) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        """Analyze a single vendor with retry logic."""
        start_time = time.time()
        run_detail = {
            "vendor": vendor,
            "products_count": len(products),
            "success": False,
        }

        if not products:
            run_detail["error"] = "No products available"
            return (None, run_detail)

        for attempt in range(self.max_retries):
            try:
                from common.core.chaining import invoke_vendor_chain

                result = invoke_vendor_chain(
                    components=components,
                    vendor=vendor,
                    products=products,
                    requirements=requirements_str,
                )

                if result:
                    # Normalize result
                    matches = self._normalize_vendor_result(result, vendor)

                    run_detail["success"] = True
                    run_detail["matches_count"] = len(matches)
                    run_detail["elapsed_ms"] = int((time.time() - start_time) * 1000)

                    return (matches, run_detail)

            except Exception as exc:
                error_str = str(exc).lower()

                # Check for rate limit errors
                if any(term in error_str for term in ["429", "rate limit", "quota", "503"]):
                    if attempt < self.max_retries - 1:
                        wait_time = 15 * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            "[VendorAgent] Rate limit for %s, waiting %ds (attempt %d/%d)",
                            vendor, wait_time, attempt + 1, self.max_retries
                        )
                        time.sleep(wait_time)
                        continue

                run_detail["error"] = str(exc)
                logger.warning("[VendorAgent] Vendor %s analysis failed: %s", vendor, exc)
                break

        return (None, run_detail)

    def _normalize_vendor_result(
        self,
        result: Any,
        vendor: str,
    ) -> List[Dict[str, Any]]:
        """Normalize vendor analysis result to standard format."""
        matches = []

        if isinstance(result, dict):
            # Extract matches from various possible keys
            for key in ["matches", "products", "vendor_matches", "results"]:
                if key in result and isinstance(result[key], list):
                    matches = result[key]
                    break

            if not matches and "matchScore" in result:
                # Single match result
                matches = [result]

        elif isinstance(result, list):
            matches = result

        # Normalize each match
        normalized = []
        for match in matches:
            if not isinstance(match, dict):
                continue

            normalized.append({
                "vendor": vendor,
                "productName": match.get("productName") or match.get("name") or match.get("model", "Unknown"),
                "model": match.get("model") or match.get("productName", ""),
                "matchScore": match.get("matchScore", 0),
                "overallScore": match.get("overallScore") or match.get("matchScore", 0),
                "matchedRequirements": match.get("matchedRequirements", []),
                "unmatchedRequirements": match.get("unmatchedRequirements", []),
                "specifications": match.get("specifications", {}),
                "strengths": match.get("strengths", []),
                "concerns": match.get("concerns", []),
                "recommendation": match.get("recommendation", ""),
            })

        return normalized

    def _enrich_results(
        self,
        matches: List[Dict[str, Any]],
        strategy_context: Optional[Dict[str, Any]],
        products_data: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Enrich matches with strategy priority and additional data."""
        if not matches:
            return matches

        # Get preferred vendors from strategy
        preferred_vendors = set()
        if strategy_context:
            preferred_vendors = set(strategy_context.get("preferred_vendors", []))

        for match in matches:
            vendor = match.get("vendor", "")

            # Add strategy priority
            if vendor in preferred_vendors:
                match["strategyPriority"] = 1
                match["isPreferredVendor"] = True
            else:
                match["strategyPriority"] = 0
                match["isPreferredVendor"] = False

            # Try to add additional product info
            vendor_products = products_data.get(vendor, [])
            model = match.get("model", "")

            for product in vendor_products:
                if product.get("model") == model or product.get("name") == model:
                    # Add image if available
                    if "image" in product or "imageUrl" in product:
                        match["imageUrl"] = product.get("image") or product.get("imageUrl")

                    # Add pricing if available
                    if "price" in product or "pricing" in product:
                        match["pricing"] = product.get("price") or product.get("pricing")

                    break

        # Sort by overall score
        matches.sort(key=lambda x: x.get("overallScore", 0), reverse=True)

        return matches
