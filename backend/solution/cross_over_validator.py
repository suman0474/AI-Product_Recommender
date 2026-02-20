# solution_N/cross_over_validator.py
# =============================================================================
# CROSS-OVER VALIDATOR & SAMPLE INPUT GENERATOR
# =============================================================================
#
# Ensures specification isolation between items:
# - No spec values leak from one item to another
# - Each item's sample_input uses ONLY its own specifications
# - Validates post-enrichment that specs are correctly scoped
#
# =============================================================================

import logging
import re
from typing import Dict, Any, List, Tuple, Set

logger = logging.getLogger(__name__)


# =============================================================================
# CROSS-OVER VALIDATION
# =============================================================================

def validate_no_spec_crossover(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate that no specification values have leaked between items.

    Checks:
    1. Each item's specs are consistent with its category
    2. No exact duplicate spec dictionaries across different item types
    3. Category-specific specs match the item type

    Args:
        items: List of enriched items with specifications

    Returns:
        Dict with:
            valid: bool - whether validation passed
            issues: List[str] - any issues found
            items_checked: int
            crossover_count: int
    """
    issues = []
    crossover_count = 0

    # Build per-item spec fingerprints
    item_fingerprints: Dict[int, Set[str]] = {}
    item_categories: Dict[int, str] = {}

    for item in items:
        item_num = item.get("number", 0)
        category = item.get("category", "").lower()
        item_categories[item_num] = category
        specs = item.get("specifications", {})

        if isinstance(specs, dict):
            # Create fingerprint: set of "key=value" pairs
            fingerprint = set()
            for k, v in specs.items():
                # Strip source labels for comparison
                v_clean = str(v).replace("[INFERRED]", "").replace("[STANDARDS]", "").strip()
                fingerprint.add(f"{k.lower()}={v_clean.lower()}")
            item_fingerprints[item_num] = fingerprint

    # Check for exact duplicate spec dictionaries
    fingerprint_list = list(item_fingerprints.items())
    for i in range(len(fingerprint_list)):
        for j in range(i + 1, len(fingerprint_list)):
            num_a, fp_a = fingerprint_list[i]
            num_b, fp_b = fingerprint_list[j]

            # Different item types should not have identical specs
            cat_a = item_categories.get(num_a, "")
            cat_b = item_categories.get(num_b, "")

            if cat_a != cat_b and fp_a == fp_b and len(fp_a) > 5:
                issues.append(
                    f"Item #{num_a} ({cat_a}) and Item #{num_b} ({cat_b}) "
                    f"have identical specifications ({len(fp_a)} specs) - possible crossover"
                )
                crossover_count += 1

    # Check for category-inappropriate specs
    CATEGORY_SPEC_MAP = {
        "pressure": ["pressure_range", "process_pressure", "overpressure"],
        "temperature": ["temperature_range", "sensor_type", "thermocouple", "rtd", "probe_length"],
        "flow": ["flow_range", "flow_rate", "pipe_size", "line_size", "fluid_type"],
        "level": ["level_range", "tank_type", "antenna_type", "radar"],
        "valve": ["cv", "actuator", "fail_action", "trim_material", "seat_material"],
    }

    for item in items:
        item_num = item.get("number", 0)
        category = item.get("category", "").lower()
        item_type = item.get("type", "")
        specs = item.get("specifications", {})

        if not isinstance(specs, dict):
            continue

        spec_keys = {k.lower() for k in specs.keys()}

        # Check if specs belong to a different category
        for cat, cat_specific_keys in CATEGORY_SPEC_MAP.items():
            if cat in category:
                continue  # This is the right category
            # Check if this item has specs specific to another category
            foreign_specs = [k for k in cat_specific_keys if any(k in sk for sk in spec_keys)]
            if len(foreign_specs) >= 3:  # 3+ foreign specs = likely crossover
                issues.append(
                    f"Item #{item_num} ({category}) has {len(foreign_specs)} specs "
                    f"typical of '{cat}' category: {foreign_specs[:3]} - possible crossover"
                )
                crossover_count += 1

    validation_result = {
        "valid": crossover_count == 0,
        "issues": issues,
        "items_checked": len(items),
        "crossover_count": crossover_count,
    }

    if crossover_count > 0:
        logger.warning(
            f"[CrossOverValidator] Found {crossover_count} potential crossover issues "
            f"across {len(items)} items"
        )
    else:
        logger.info(f"[CrossOverValidator] All {len(items)} items passed crossover validation")

    return validation_result


# =============================================================================
# ISOLATED SAMPLE INPUT GENERATION
# =============================================================================

# Keys to skip in sample input (too verbose or internal)
_SKIP_IN_SAMPLE = {
    "extraction_method", "source_method", "confidence", "enrichment_source",
    "inferred_specs", "standards_info", "applicable_standards",
    "certifications", "standards_specs", "standards_summary",
}


def generate_isolated_sample_input(item: Dict[str, Any]) -> str:
    """
    Generate sample_input for a single item using ONLY its own specifications.

    Pattern: "[Product] with [spec1], [spec2]..., suitable for [application]"

    Strict isolation: Only uses specs from this item's specifications dict.
    No cross-referencing with other items or global state.

    Args:
        item: Single item dict with name, category, specifications, etc.

    Returns:
        Generated sample_input string
    """
    name = item.get("name", item.get("product_name", "Industrial Instrument"))
    category = item.get("category", "")
    specs = item.get("specifications", {})
    purpose = item.get("purpose", item.get("solution_purpose", ""))

    # Build spec parts from this item ONLY
    spec_parts = []

    if isinstance(specs, dict):
        for key, value in specs.items():
            key_lower = key.lower()

            # Skip internal/meta keys
            if key_lower in _SKIP_IN_SAMPLE:
                continue

            # Clean value
            val_str = str(value).strip()
            # Remove source labels
            val_str = val_str.replace("[INFERRED]", "").replace("[STANDARDS]", "").strip()

            if not val_str or val_str.lower() in ("n/a", "null", "none", "not specified", ""):
                continue

            # Format as readable spec
            key_readable = key.replace("_", " ").title()
            spec_parts.append(f"{key_readable}: {val_str}")

    # Build sample input
    parts = [name]

    if spec_parts:
        # Include up to 10 most important specs
        parts.append("with " + ", ".join(spec_parts[:10]))

    if purpose:
        parts.append(f"for {purpose}")
    elif category:
        parts.append(f"for {category} applications")

    sample_input = " ".join(parts)

    # Trim to reasonable length (max 300 chars)
    if len(sample_input) > 300:
        sample_input = sample_input[:297] + "..."

    return sample_input


def generate_all_sample_inputs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate sample_input for all items with strict isolation.

    Each item gets its sample_input generated independently using
    ONLY its own specifications. No cross-referencing.

    Args:
        items: List of items to generate sample_inputs for

    Returns:
        Same list with sample_input field updated
    """
    for item in items:
        item["sample_input"] = generate_isolated_sample_input(item)

    logger.info(f"[SampleInputGenerator] Generated sample_inputs for {len(items)} items")
    return items
