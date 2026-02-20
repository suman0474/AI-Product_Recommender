# services/azure/vendor_utils.py
"""
Vendor Search Utilities - Product Type Alias Expansion

Provides fuzzy matching and alias expansion for product type searches to improve
vendor matching rates in Azure Blob Storage.

Problem: Vendor search returns 0 results because of strict category matching
Solution: Expand product types to include all possible aliases and search across variants

Created: 2026-02-11
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# PRODUCT TYPE ALIAS MAPPINGS
# ============================================================================
# Maps canonical product types to all their possible variants
# This enables finding vendors even when product type naming varies

PRODUCT_TYPE_ALIASES = {
    "level_transmitter": [
        "level_transmitter",
        "level transmitter",
        "level sensor",
        "level sensors",
        "radar level",
        "radar level sensor",
        "radar level sensors",
        "ultrasonic level",
        "ultrasonic level sensor",
        "ultrasonic level sensors",
        "guided wave radar",
        "guided wave radar level",
        "level instrument",
        "level instruments",
        "level measurement",
        "tank level",
        "tank level sensor",
    ],

    "pressure_transmitter": [
        "pressure_transmitter",
        "pressure transmitter",
        "pressure sensor",
        "pressure sensors",
        "pressure gauge",
        "pressure gauges",
        "pressure instrument",
        "pressure instruments",
        "pressure transducer",
        "pressure transducers",
        "differential pressure",
        "differential pressure transmitter",
        "gauge pressure",
        "absolute pressure",
        "pressure measurement",
    ],

    "temperature_transmitter": [
        "temperature_transmitter",
        "temperature transmitter",
        "temperature sensor",
        "temperature sensors",
        "temperature instrument",
        "temperature instruments",
        "thermocouple",
        "thermocouples",
        "rtd",
        "rtds",
        "resistance temperature detector",
        "resistance temperature detectors",
        "temperature probe",
        "temperature probes",
        "temperature measurement",
        "thermometer",
        "thermometers",
    ],

    "flow_meter": [
        "flow_meter",
        "flow meter",
        "flowmeter",
        "flowmeters",
        "flow sensor",
        "flow sensors",
        "coriolis meter",
        "coriolis meters",
        "coriolis flow meter",
        "magnetic flowmeter",
        "magnetic flowmeters",
        "mag meter",
        "ultrasonic flowmeter",
        "ultrasonic flowmeters",
        "vortex flowmeter",
        "vortex flowmeters",
        "turbine flowmeter",
        "turbine flowmeters",
        "flow instrument",
        "flow instruments",
        "flow measurement",
        "mass flow meter",
        "volumetric flow meter",
    ],

    "control_valves": [
        "control_valves",
        "control valve",
        "control valves",
        "valve",
        "valves",
        "ball valve",
        "ball valves",
        "globe valve",
        "globe valves",
        "butterfly valve",
        "butterfly valves",
        "gate valve",
        "gate valves",
        "plug valve",
        "plug valves",
        "actuated valve",
        "actuated valves",
        "modulating valve",
        "modulating valves",
        "on-off valve",
        "shutoff valve",
        "isolation valve",
    ],

    "actuators": [
        "actuators",
        "actuator",
        "valve actuator",
        "valve actuators",
        "pneumatic actuator",
        "pneumatic actuators",
        "electric actuator",
        "electric actuators",
        "hydraulic actuator",
        "hydraulic actuators",
        "rotary actuator",
        "rotary actuators",
        "linear actuator",
        "linear actuators",
        "smart actuator",
        "smart actuators",
    ],

    "gas_detectors": [
        "gas_detectors",
        "gas detector",
        "gas detectors",
        "gas sensor",
        "gas sensors",
        "combustible gas detector",
        "toxic gas detector",
        "flame detector",
        "flame detectors",
        "fire detector",
        "fire detectors",
        "h2s detector",
        "co detector",
        "ch4 detector",
        "methane detector",
    ],

    "analytical_instruments": [
        "analytical_instruments",
        "analytical instrument",
        "analytical instruments",
        "analyzer",
        "analyzers",
        "gas chromatograph",
        "gas chromatographs",
        "gc",
        "ph meter",
        "ph meters",
        "ph sensor",
        "conductivity meter",
        "conductivity meters",
        "conductivity sensor",
        "dissolved oxygen",
        "do meter",
        "turbidity meter",
        "orp meter",
        "toc analyzer",
        "moisture analyzer",
        "oxygen analyzer",
        "nox analyzer",
        "sox analyzer",
    ],

    "positioners": [
        "positioners",
        "positioner",
        "valve positioner",
        "valve positioners",
        "smart positioner",
        "smart positioners",
        "pneumatic positioner",
        "electro-pneumatic positioner",
        "digital positioner",
        "digital positioners",
    ],

    "safety_instruments": [
        "safety_instruments",
        "safety instrument",
        "safety instrumented system",
        "sis",
        "safety system",
        "emergency shutdown",
        "esd",
        "fire and gas",
        "f&g",
        "burner management",
        "bms",
        "high integrity pressure protection",
        "hipps",
    ],
}


def expand_product_type_aliases(product_type: str) -> List[str]:
    """
    Expand a product type to include all possible aliases.

    This enables fuzzy matching by returning all known variants of a product type.
    For example, "level sensor" expands to include "level_transmitter", "radar level", etc.

    Args:
        product_type: Product type string (e.g., "level sensor", "pressure transmitter")

    Returns:
        List of all aliases for this product type, including the original

    Examples:
        >>> expand_product_type_aliases("level sensor")
        ["level_transmitter", "level transmitter", "level sensor", "radar level", ...]

        >>> expand_product_type_aliases("unknown product")
        ["unknown product"]  # Returns original if no mapping found
    """
    if not product_type:
        return []

    # Normalize the input
    pt_lower = product_type.lower().strip()
    pt_normalized = pt_lower.replace("-", "_").replace(" ", "_")

    # Search for matching canonical type
    for canonical, alias_list in PRODUCT_TYPE_ALIASES.items():
        # Check if input matches canonical type
        if pt_normalized == canonical:
            logger.debug(f"[ProductTypeAlias] Exact canonical match: '{product_type}' → {len(alias_list)} aliases")
            return alias_list

        # Check if input matches any alias
        normalized_aliases = [a.lower().replace("-", "_").replace(" ", "_") for a in alias_list]
        if pt_lower in [a.lower() for a in alias_list] or pt_normalized in normalized_aliases:
            logger.debug(f"[ProductTypeAlias] Alias match: '{product_type}' → canonical '{canonical}' → {len(alias_list)} aliases")
            return alias_list

    # No match found - return original
    logger.debug(f"[ProductTypeAlias] No match for '{product_type}', returning original only")
    return [product_type]


def search_vendors_with_aliases(
    product_type: str,
    vendor_name: Optional[str] = None,
    search_function: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Search for vendors using product type alias expansion.

    This function expands the product type to all known aliases and searches
    for each variant, then deduplicates results.

    Args:
        product_type: Product type to search for
        vendor_name: Optional vendor name filter
        search_function: Function to call for each alias (should accept product_category and vendor_filter)

    Returns:
        Deduplicated list of vendor matches

    Example:
        >>> def my_search(product_category, vendor_filter=None):
        ...     return azure_blob_search(category=product_category, vendor=vendor_filter)
        >>> results = search_vendors_with_aliases("level sensor", "Yokogawa", my_search)
    """
    if not search_function:
        logger.warning("[VendorUtils] No search function provided, cannot search")
        return []

    all_aliases = expand_product_type_aliases(product_type)
    logger.info(f"[VendorUtils] Searching across {len(all_aliases)} aliases for: {product_type}")

    all_results = []
    for alias in all_aliases:
        try:
            results = search_function(
                product_category=alias,
                vendor_filter=vendor_name
            )
            if results:
                all_results.extend(results)
                logger.debug(f"[VendorUtils] Alias '{alias}': {len(results)} results")
        except Exception as e:
            logger.debug(f"[VendorUtils] Search failed for alias '{alias}': {e}")
            continue

    # Deduplicate by (vendor_name, product_name) tuple
    seen = set()
    unique_results = []
    for vendor in all_results:
        # Create unique key from vendor and product names
        vendor_name_key = vendor.get("vendor_name", "").lower().strip()
        product_name_key = vendor.get("product_name", "").lower().strip()
        key = (vendor_name_key, product_name_key)

        if key not in seen and vendor_name_key and product_name_key:
            seen.add(key)
            unique_results.append(vendor)

    logger.info(
        f"[VendorUtils] Found {len(unique_results)} unique vendors "
        f"from {len(all_results)} total results across {len(all_aliases)} aliases"
    )

    return unique_results


def get_canonical_product_type(product_type: str) -> str:
    """
    Get the canonical (standardized) product type name.

    Args:
        product_type: Any product type variant

    Returns:
        Canonical product type name, or original if not found

    Examples:
        >>> get_canonical_product_type("radar level sensor")
        "level_transmitter"

        >>> get_canonical_product_type("mag meter")
        "flow_meter"
    """
    if not product_type:
        return product_type

    pt_lower = product_type.lower().strip()

    # Check each canonical type's aliases
    for canonical, alias_list in PRODUCT_TYPE_ALIASES.items():
        if pt_lower in [a.lower() for a in alias_list]:
            return canonical

    # Return original if no match
    return product_type


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_all_product_type_variants(product_type: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a product type and its variants.

    Returns:
        Dictionary with canonical name, all aliases, and match statistics
    """
    canonical = get_canonical_product_type(product_type)
    aliases = expand_product_type_aliases(product_type)

    return {
        "input": product_type,
        "canonical": canonical,
        "aliases": aliases,
        "alias_count": len(aliases),
        "is_canonical": canonical == product_type.lower().replace("-", "_").replace(" ", "_")
    }


if __name__ == "__main__":
    # Test the alias expansion
    test_cases = [
        "level sensor",
        "pressure transmitter",
        "flow meter",
        "radar level",
        "mag meter",
        "unknown product"
    ]

    print("Product Type Alias Expansion Test")
    print("=" * 60)

    for test in test_cases:
        aliases = expand_product_type_aliases(test)
        canonical = get_canonical_product_type(test)
        print(f"\nInput: '{test}'")
        print(f"Canonical: '{canonical}'")
        print(f"Aliases ({len(aliases)}): {aliases[:5]}...")  # Show first 5
