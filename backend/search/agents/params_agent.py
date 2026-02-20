# search/agents/params_agent.py
# =============================================================================
# ADVANCED PARAMETERS AGENT
# =============================================================================
#
# Handles discovery of vendor-specific advanced parameters.
# Queries vendor product data to identify specifications beyond the base schema.
#
# =============================================================================

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ParamsResult:
    """Result from the advanced parameters discovery agent."""

    success: bool
    product_type: str
    unique_specifications: List[Dict[str, Any]]
    total_unique_specifications: int
    existing_specifications_filtered: int
    vendors_searched: List[str]
    discovery_successful: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "success": self.success,
            "product_type": self.product_type,
            "unique_specifications": self.unique_specifications,
            "total_unique_specifications": self.total_unique_specifications,
            "existing_specifications_filtered": self.existing_specifications_filtered,
            "vendors_searched": self.vendors_searched,
            "discovery_successful": self.discovery_successful,
            "error": self.error,
        }


class ParamsAgent:
    """
    Handles discovery of vendor-specific advanced parameters.

    This agent queries vendor product catalogs to identify specifications
    that aren't in the base schema but could be relevant for matching.
    """

    # Common specification fields across vendors
    COMMON_SPEC_FIELDS: Set[str] = {
        "measurementRange", "pressureRange", "temperatureRange",
        "accuracy", "outputSignal", "communicationProtocol",
        "processConnection", "material", "wetted_material",
        "ingressProtection", "certification", "displayType",
        "supplyVoltage", "responseTime", "overPressureLimit",
    }

    def __init__(self):
        """Initialize the ParamsAgent."""
        pass

    def discover(
        self,
        product_type: str,
        session_id: Optional[str] = None,
        existing_schema: Optional[Dict[str, Any]] = None,
    ) -> ParamsResult:
        """
        Discover advanced parameters from vendor data.

        Args:
            product_type: The product type to discover params for
            session_id: Session identifier for caching
            existing_schema: Current schema to avoid duplicates

        Returns:
            ParamsResult with discovered specifications
        """
        logger.info("[ParamsAgent] Discovering advanced params for: %s", product_type)

        try:
            # Get existing spec keys to filter out
            existing_keys = self._get_existing_spec_keys(existing_schema)
            logger.debug("[ParamsAgent] Existing spec keys: %d", len(existing_keys))

            # Load vendor data
            vendors, vendor_specs = self._load_vendor_specifications(product_type)

            if not vendors:
                logger.warning("[ParamsAgent] No vendor data found")
                return ParamsResult(
                    success=True,
                    product_type=product_type,
                    unique_specifications=[],
                    total_unique_specifications=0,
                    existing_specifications_filtered=0,
                    vendors_searched=[],
                    discovery_successful=False,
                    error="No vendor data available",
                )

            # Identify unique specifications
            unique_specs, filtered_count = self._identify_unique_specs(
                vendor_specs, existing_keys
            )

            # Build specification details
            spec_details = self._build_spec_details(unique_specs, vendor_specs)

            logger.info(
                "[ParamsAgent] Discovered %d unique specs from %d vendors",
                len(spec_details),
                len(vendors),
            )

            return ParamsResult(
                success=True,
                product_type=product_type,
                unique_specifications=spec_details,
                total_unique_specifications=len(spec_details),
                existing_specifications_filtered=filtered_count,
                vendors_searched=vendors,
                discovery_successful=True,
            )

        except Exception as exc:
            logger.error("[ParamsAgent] Discovery failed: %s", exc, exc_info=True)
            return ParamsResult(
                success=False,
                product_type=product_type,
                unique_specifications=[],
                total_unique_specifications=0,
                existing_specifications_filtered=0,
                vendors_searched=[],
                discovery_successful=False,
                error=str(exc),
            )

    def _get_existing_spec_keys(
        self,
        schema: Optional[Dict[str, Any]],
    ) -> Set[str]:
        """Extract existing specification keys from schema."""
        keys = set()

        if not schema:
            return keys

        # Check various schema sections
        sections = [
            "mandatory_requirements",
            "optional_requirements",
            "mandatory",
            "optional",
            "specifications",
        ]

        for section in sections:
            if section in schema and isinstance(schema[section], dict):
                keys.update(schema[section].keys())

        # Normalize keys
        normalized = set()
        for key in keys:
            normalized.add(key.lower().replace("_", "").replace(" ", ""))

        return normalized

    def _load_vendor_specifications(
        self,
        product_type: str,
    ) -> tuple:
        """Load specifications from vendor product data."""
        vendors = []
        vendor_specs = {}

        try:
            from common.services.azure.blob_utils import (
                get_vendors_for_product_type,
                get_products_for_vendors,
            )

            # Get vendors for this product type
            vendors = get_vendors_for_product_type(product_type)

            if not vendors:
                return ([], {})

            # Limit to first 5 vendors for performance
            sample_vendors = vendors[:5]

            # Get product data
            products_data = get_products_for_vendors(sample_vendors, product_type)

            # Extract specifications from products
            for vendor, products in products_data.items():
                specs = set()
                for product in products:
                    if isinstance(product, dict):
                        # Extract specification keys
                        for key in product.keys():
                            if self._is_spec_field(key):
                                specs.add(key)

                        # Check nested specifications
                        if "specifications" in product and isinstance(product["specifications"], dict):
                            specs.update(product["specifications"].keys())

                vendor_specs[vendor] = specs

            return (sample_vendors, vendor_specs)

        except ImportError:
            logger.warning("[ParamsAgent] Azure blob utils not available")
            return ([], {})
        except Exception as exc:
            logger.warning("[ParamsAgent] Failed to load vendor data: %s", exc)
            return ([], {})

    def _is_spec_field(self, field_name: str) -> bool:
        """Check if a field name is likely a specification."""
        # Skip common non-spec fields
        skip_fields = {
            "id", "name", "model", "vendor", "price", "image", "url",
            "description", "category", "created", "updated", "status",
        }

        field_lower = field_name.lower()

        if field_lower in skip_fields:
            return False

        # Check if it matches common spec patterns
        spec_indicators = [
            "range", "accuracy", "output", "input", "signal",
            "material", "connection", "protocol", "rating",
            "temperature", "pressure", "voltage", "current",
            "protection", "certification", "compliance",
        ]

        return any(ind in field_lower for ind in spec_indicators)

    def _identify_unique_specs(
        self,
        vendor_specs: Dict[str, Set[str]],
        existing_keys: Set[str],
    ) -> tuple:
        """Identify specifications not in existing schema."""
        all_specs = set()
        for specs in vendor_specs.values():
            all_specs.update(specs)

        # Filter out existing
        filtered_count = 0
        unique_specs = set()

        for spec in all_specs:
            normalized = spec.lower().replace("_", "").replace(" ", "")
            if normalized in existing_keys:
                filtered_count += 1
            else:
                unique_specs.add(spec)

        return (unique_specs, filtered_count)

    def _build_spec_details(
        self,
        unique_specs: Set[str],
        vendor_specs: Dict[str, Set[str]],
    ) -> List[Dict[str, Any]]:
        """Build detailed specification entries."""
        spec_details = []

        for spec in sorted(unique_specs):
            # Count vendors that have this spec
            vendor_count = sum(
                1 for specs in vendor_specs.values()
                if spec in specs
            )

            spec_details.append({
                "field_name": spec,
                "display_name": self._format_display_name(spec),
                "vendor_coverage": vendor_count,
                "total_vendors": len(vendor_specs),
                "relevance_score": vendor_count / len(vendor_specs) if vendor_specs else 0,
            })

        # Sort by relevance (vendor coverage)
        spec_details.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Limit to top 10 most relevant
        return spec_details[:10]

    def _format_display_name(self, field_name: str) -> str:
        """Convert field name to display name."""
        import re

        # Handle camelCase
        name = re.sub(r'([A-Z])', r' \1', field_name)
        # Handle snake_case
        name = name.replace("_", " ")
        # Capitalize
        return name.strip().title()
