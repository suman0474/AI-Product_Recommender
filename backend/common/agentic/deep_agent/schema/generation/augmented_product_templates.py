# agentic/deep_agent/schema/generation/augmented_product_templates.py
# =============================================================================
# AUGMENTED PRODUCT TEMPLATES
# =============================================================================
#
# Purpose: Wrapper around the specification templates that provides an
# augmented interface for the schema generation deep agent.
#
# This module bridges the existing templates system with the deep agent's
# requirements for augmented templates with default values and metadata.
#
# =============================================================================

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import from the existing templates module
try:
    from common.agentic.deep_agent.specifications.templates.templates import (
        get_template_for_product_type,
        get_all_specs_for_product_type,
        SpecificationDefinition,
        PRODUCT_TYPE_TEMPLATES
    )
    TEMPLATES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[AUGMENTED_TEMPLATES] Could not import templates: {e}")
    TEMPLATES_AVAILABLE = False
    PRODUCT_TYPE_TEMPLATES = {}


class AugmentedSpecification:
    """
    Augmented specification with additional metadata for schema generation.
    """
    def __init__(self, spec_def: 'SpecificationDefinition'):
        self.key = spec_def.key
        self.category = spec_def.category
        self.description = spec_def.description
        self.unit = spec_def.unit
        self.data_type = spec_def.data_type
        self.default_value = spec_def.typical_value
        self.typical_values = spec_def.options if spec_def.options else []
        self.importance = spec_def.importance
        self.source_priority = getattr(spec_def, 'source_priority', 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "category": self.category,
            "description": self.description,
            "unit": self.unit,
            "data_type": self.data_type,
            "default_value": self.default_value,
            "typical_values": self.typical_values,
            "importance": self.importance.name if hasattr(self.importance, 'name') else str(self.importance)
        }


class AugmentedProductTemplates:
    """
    Provides augmented product templates for the schema generation deep agent.

    This class wraps the existing template system and adds:
    - Unified interface via get_augmented_template()
    - Fallback to generic template when specific not found
    - Additional metadata for deep agent consumption
    """

    # Mapping of common product type variations to normalized names
    PRODUCT_TYPE_ALIASES = {
        "temperature sensor": "temperature_sensor",
        "temp sensor": "temperature_sensor",
        "rtd": "temperature_sensor",
        "thermocouple": "temperature_sensor",
        "pressure transmitter": "pressure_transmitter",
        "pressure sensor": "pressure_transmitter",
        "flow meter": "flow_meter",
        "flowmeter": "flow_meter",
        "level transmitter": "level_transmitter",
        "level sensor": "level_transmitter",
        "control valve": "control_valve",
        "valve": "control_valve",
        "thermowell": "thermowell",
        "analyzer": "analyzer",
        "cable gland": "cable_gland",
        "enclosure": "enclosure",
        "mounting kit": "mounting_kit",
    }

    @classmethod
    def normalize_product_type(cls, product_type: str) -> str:
        """Normalize product type to match template keys."""
        normalized = product_type.lower().strip()

        # Check aliases first
        if normalized in cls.PRODUCT_TYPE_ALIASES:
            return cls.PRODUCT_TYPE_ALIASES[normalized]

        # Replace spaces with underscores
        normalized = normalized.replace(" ", "_")

        return normalized

    @classmethod
    def get_augmented_template(cls, product_type: str) -> Optional[Dict[str, AugmentedSpecification]]:
        """
        Get augmented template for a product type.

        Args:
            product_type: The product type (e.g., "Temperature Sensor", "Pressure Transmitter")

        Returns:
            Dictionary of field_key -> AugmentedSpecification, or None if not found
        """
        if not TEMPLATES_AVAILABLE:
            logger.warning("[AUGMENTED_TEMPLATES] Templates not available")
            return None

        # Normalize the product type
        normalized_type = cls.normalize_product_type(product_type)

        # Try to get the template
        specs = get_all_specs_for_product_type(normalized_type)

        if not specs:
            # Try with the original type
            specs = get_all_specs_for_product_type(product_type)

        if not specs:
            logger.debug(f"[AUGMENTED_TEMPLATES] No template found for: {product_type}")
            return None

        # Convert to augmented specifications
        augmented_specs = {}
        for key, spec_def in specs.items():
            try:
                augmented_specs[key] = AugmentedSpecification(spec_def)
            except Exception as e:
                logger.warning(f"[AUGMENTED_TEMPLATES] Failed to augment spec {key}: {e}")
                continue

        logger.info(f"[AUGMENTED_TEMPLATES] Loaded {len(augmented_specs)} specs for {product_type}")
        return augmented_specs

    @classmethod
    def get_available_product_types(cls) -> list:
        """Get list of all available product types with templates."""
        if not TEMPLATES_AVAILABLE:
            return []
        return list(PRODUCT_TYPE_TEMPLATES.keys())

    @classmethod
    def get_generic_template(cls) -> Dict[str, AugmentedSpecification]:
        """
        Get a generic template with common specifications applicable to most products.
        Used as fallback when specific template is not found.
        """
        # Common specifications that apply to most industrial instruments
        generic_specs = {
            "product_name": AugmentedSpecification(_create_generic_spec(
                "product_name", "General", "Product name or model number", "string"
            )),
            "manufacturer": AugmentedSpecification(_create_generic_spec(
                "manufacturer", "General", "Manufacturer/vendor name", "string"
            )),
            "model_number": AugmentedSpecification(_create_generic_spec(
                "model_number", "General", "Model number", "string"
            )),
            "operating_temperature_min": AugmentedSpecification(_create_generic_spec(
                "operating_temperature_min", "Environment", "Minimum operating temperature", "number", "°C", "-40"
            )),
            "operating_temperature_max": AugmentedSpecification(_create_generic_spec(
                "operating_temperature_max", "Environment", "Maximum operating temperature", "number", "°C", "85"
            )),
            "ip_rating": AugmentedSpecification(_create_generic_spec(
                "ip_rating", "Environment", "Ingress protection rating", "string", None, "IP65"
            )),
            "power_supply_voltage": AugmentedSpecification(_create_generic_spec(
                "power_supply_voltage", "Electrical", "Power supply voltage", "string", "V", "24 VDC"
            )),
            "output_signal": AugmentedSpecification(_create_generic_spec(
                "output_signal", "Electrical", "Output signal type", "string", None, "4-20 mA"
            )),
            "hazardous_area_certification": AugmentedSpecification(_create_generic_spec(
                "hazardous_area_certification", "Certification", "Hazardous area certification", "string"
            )),
            "material_wetted_parts": AugmentedSpecification(_create_generic_spec(
                "material_wetted_parts", "Materials", "Material of wetted parts", "string", None, "316L SS"
            )),
        }
        return generic_specs


def _create_generic_spec(key: str, category: str, description: str,
                         data_type: str = "string", unit: str = None,
                         typical_value: Any = None) -> 'SpecificationDefinition':
    """Helper to create a generic SpecificationDefinition-like object."""
    class GenericSpec:
        def __init__(self, key, category, description, data_type, unit, typical_value):
            self.key = key
            self.category = category
            self.description = description
            self.data_type = data_type
            self.unit = unit
            self.typical_value = typical_value
            self.options = None
            self.importance = "OPTIONAL"
            self.source_priority = 10

    return GenericSpec(key, category, description, data_type, unit, typical_value)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "AugmentedProductTemplates",
    "AugmentedSpecification",
]
