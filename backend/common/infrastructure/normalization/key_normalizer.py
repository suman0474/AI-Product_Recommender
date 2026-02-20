# agentic/infrastructure/normalization/key_normalizer.py
# =============================================================================
# CENTRALIZED KEY NORMALIZATION
# =============================================================================
#
# Single source of truth for specification key normalization.
# Consolidated from:
# - standards/generation/normalizer.py
# - deep_agent/specifications/aggregator.py
# - standards/shared/enrichment.py
#
# =============================================================================

import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)


# =============================================================================
# STANDARD KEY MAPPINGS
# =============================================================================

STANDARD_KEY_MAPPINGS: Dict[str, str] = {
    # Common variations - SIL Rating
    "silRating": "sil_rating",
    "sil_rating": "sil_rating",
    "Sil rating": "sil_rating",
    "SIL Rating": "sil_rating",
    "SIL_Rating": "sil_rating",

    # Hazardous area
    "hazardousAreaRating": "hazardous_area_approval",
    "hazardous_area_rating": "hazardous_area_approval",
    "Hazardous area approval": "hazardous_area_approval",
    "hazardousAreaApproval": "hazardous_area_approval",

    # Material wetted
    "materialWetted": "material_wetted",
    "Material wetted": "material_wetted",
    "Material_wetted": "material_wetted",

    # Material housing
    "materialHousing": "material_housing",
    "Material housing": "material_housing",
    "Material_housing": "material_housing",

    # Process connection
    "processConnection": "process_connection",
    "Process connection": "process_connection",
    "Process_connection": "process_connection",

    # Output signal
    "outputSignal": "output_signal",
    "Output signal": "output_signal",
    "Output_signal": "output_signal",

    # Supply voltage
    "supplyVoltage": "supply_voltage",
    "Supply voltage": "supply_voltage",
    "Supply_voltage": "supply_voltage",

    # Protection rating
    "protectionRating": "protection_rating",
    "Protection rating": "protection_rating",
    "Protection_rating": "protection_rating",

    # Temperature range
    "temperatureRange": "temperature_range",
    "Temperature range": "temperature_range",
    "Temperature_range": "temperature_range",

    # Pressure range
    "pressureRange": "pressure_range",
    "Pressure range": "pressure_range",
    "Pressure_range": "pressure_range",

    # Ambient temperature
    "ambientTemperature": "ambient_temperature",
    "Ambient temperature": "ambient_temperature",
    "Ambient_temperature": "ambient_temperature",

    # Process temperature
    "processTemperature": "process_temperature",
    "Process temperature": "process_temperature",
    "Process_temperature": "process_temperature",

    # Communication protocol
    "communicationProtocol": "communication_protocol",
    "Communication protocol": "communication_protocol",
    "Communication_protocol": "communication_protocol",

    # Response time
    "responseTime": "response_time",
    "Response time": "response_time",
    "Response_time": "response_time",

    # Calibration interval
    "calibrationInterval": "calibration_interval",
    "Calibration interval": "calibration_interval",
    "Calibration_interval": "calibration_interval",

    # Power consumption
    "powerConsumption": "power_consumption",
    "Power consumption": "power_consumption",
    "Power_consumption": "power_consumption",

    # Humidity range
    "humidityRange": "humidity_range",
    "Humidity range": "humidity_range",
    "Humidity_range": "humidity_range",

    # Measurement range
    "measurementRange": "measurement_range",
    "Measurement range": "measurement_range",
    "Measurement_range": "measurement_range",

    # Wake frequency calculation
    "wakeFrequencyCalculation": "wake_frequency_calculation",
    "Wake frequency calculation": "wake_frequency_calculation",

    # Shank style
    "shankStyle": "shank_style",
    "Shank style": "shank_style",

    # Pressure rating
    "pressureRating": "pressure_rating",
    "Pressure rating": "pressure_rating",

    # Terminal block type
    "terminalBlockType": "terminal_block_type",
    "Terminal block type": "terminal_block_type",

    # Cable entry
    "cableEntry": "cable_entry",
    "Cable entry": "cable_entry",

    # Conductor gauge
    "conductorGauge": "conductor_gauge",
    "Conductor gauge": "conductor_gauge",

    # Conductor material
    "conductorMaterial": "conductor_material",
    "Conductor material": "conductor_material",

    # Insulation resistance
    "insulationResistance": "insulation_resistance",
    "Insulation resistance": "insulation_resistance",

    # Color coding
    "colorCoding": "color_coding",
    "Color coding": "color_coding",

    # Pin material
    "pinMaterial": "pin_material",
    "Pin material": "pin_material",

    # Terminal type
    "terminalType": "terminal_type",
    "Terminal type": "terminal_type",

    # Thermocouple type compatibility
    "thermocoupleTypeCompatibility": "thermocouple_type_compatibility",
    "Thermocouple type compatibility": "thermocouple_type_compatibility",
}


# =============================================================================
# KEY NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_key(key: str) -> str:
    """
    Normalize a specification key to snake_case.

    Handles:
    - Direct mappings from STANDARD_KEY_MAPPINGS
    - CamelCase to snake_case conversion
    - Space to underscore conversion
    - Multiple underscore cleanup

    Args:
        key: The original key in any format

    Returns:
        Normalized snake_case key
    """
    if not key:
        return ""

    # Check direct mapping first
    if key in STANDARD_KEY_MAPPINGS:
        return STANDARD_KEY_MAPPINGS[key]

    # Convert camelCase to snake_case
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', key)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)

    # Replace spaces and multiple underscores
    result = re.sub(r'[\s]+', '_', s2)
    result = re.sub(r'_+', '_', result)

    return result.lower().strip('_')


def normalize_spec_key(key: str) -> str:
    """
    Normalize a specification key for deduplication purposes.

    Simpler version that just lowercases and normalizes separators.

    Args:
        key: The key to normalize

    Returns:
        Normalized key for comparison
    """
    if not key:
        return ""
    return key.strip().lower().replace(" ", "_").replace("-", "_")


def camel_to_snake(name: str) -> str:
    """
    Convert camelCase to snake_case.

    Args:
        name: CamelCase string

    Returns:
        snake_case string
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def snake_to_camel(name: str) -> str:
    """
    Convert snake_case to camelCase.

    Args:
        name: snake_case string

    Returns:
        camelCase string
    """
    components = name.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def get_canonical_key(key: str) -> str:
    """
    Get the canonical (standardized) version of a key.

    First checks STANDARD_KEY_MAPPINGS, then falls back to normalize_key.

    Args:
        key: Original key

    Returns:
        Canonical key name
    """
    # Check direct mapping
    if key in STANDARD_KEY_MAPPINGS:
        return STANDARD_KEY_MAPPINGS[key]

    # Check if normalized version is in mappings
    normalized = normalize_key(key)
    for mapped_key, canonical in STANDARD_KEY_MAPPINGS.items():
        if normalize_key(mapped_key) == normalized:
            return canonical

    return normalized


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "STANDARD_KEY_MAPPINGS",
    "normalize_key",
    "normalize_spec_key",
    "camel_to_snake",
    "snake_to_camel",
    "get_canonical_key",
]
