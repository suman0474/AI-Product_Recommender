# agentic/deep_agent_integration.py
# =============================================================================
# DEEP AGENT INTEGRATION MODULE
# =============================================================================
#
# PURPOSE: Integration layer for using Deep Agent with Solution and Instrument
# Identifier workflows. Provides helper functions to run Deep Agent and populate
# SCHEMA FIELD VALUES from standards specifications.
#
# KEY FUNCTIONALITY:
# 1. Run Deep Agent to analyze standards documents
# 2. Extract specifications from standards for each product type
# 3. POPULATE SCHEMA FIELD VALUES from the extracted specifications
# 4. Map specifications to schema fields (mandatory and optional)
#
# USAGE:
#   from .deep_agent_integration import integrate_deep_agent_specifications
#   enriched_items = integrate_deep_agent_specifications(
#       all_items=state["all_items"],
#       user_input=state["user_input"],
#       solution_context=state.get("solution_analysis")
#   )
#
# =============================================================================

import json
import logging
from typing import Dict, Any, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMA FIELD MAPPING
# =============================================================================

# Map Deep Agent specification keys to schema field names
# This mapping handles variations between snake_case spec keys and schema field names
# ROOT CAUSE FIX: Added Title Case with space variations to match UI display format
SPEC_TO_SCHEMA_MAPPING = {
    # =========================================================================
    # MEASUREMENT & ACCURACY
    # =========================================================================
    "accuracy": ["accuracy", "Accuracy", "measurementAccuracy", "measurement_accuracy", "Measurement Accuracy"],
    "repeatability": ["repeatability", "Repeatability", "measurement_repeatability", "Measurement Repeatability"],
    "resolution": ["resolution", "Resolution", "display_resolution", "Display Resolution"],
    "linearity": ["linearity", "Linearity", "measurement_linearity", "Measurement Linearity"],
    "hysteresis": ["hysteresis", "Hysteresis"],
    "stability": ["stability", "Stability", "long_term_stability", "Long Term Stability"],
    "drift": ["drift", "Drift", "temperature_drift", "zero_drift", "Temperature Drift", "Zero Drift"],
    "turn_down_ratio": ["turnDownRatio", "turn_down_ratio", "rangeability", "Turn Down Ratio", "Turndown Ratio"],
    "rangeability": ["rangeability", "Rangeability", "turnDownRatio", "Turn Down Ratio"],

    # =========================================================================
    # PRESSURE SPECIFICATIONS
    # =========================================================================
    "pressure_range": ["pressureRange", "pressure_range", "measurementRange", "range", "Pressure Range", "Measurement Range"],
    "max_pressure": ["maxPressure", "max_pressure", "maximum_pressure", "burstPressure", "Max Pressure", "Maximum Pressure"],
    "min_pressure": ["minPressure", "min_pressure", "minimum_pressure", "Min Pressure", "Minimum Pressure"],
    "burst_pressure": ["burstPressure", "burst_pressure", "max_pressure_rating", "Burst Pressure"],
    "proof_pressure": ["proofPressure", "proof_pressure", "Proof Pressure"],
    "overpressure_limit": ["overpressureLimit", "overpressure_limit", "Overpressure Limit"],
    "static_pressure_effect": ["staticPressureEffect", "static_pressure_effect", "Static Pressure Effect", "Static Pressure Effect On Accuracy"],
    "line_pressure_effect": ["linePressureEffect", "line_pressure_effect", "Line Pressure Effect"],

    # =========================================================================
    # TEMPERATURE SPECIFICATIONS
    # =========================================================================
    "temperature_range": ["temperatureRange", "temperature_range", "measurementRange", "Temperature Range", "Measurement Range"],
    "temperatureRange": ["temperatureRange", "temperature_range", "Temperature Range"],
    "process_temperature": ["processTemperature", "process_temperature", "fluid_temperature", "Process Temperature", "Process Temperature Limit"],
    "ambient_temperature": ["ambientTemperature", "ambient_temperature", "operatingTemperature", "Ambient Temperature", "Operating Temperature"],
    "storage_temperature": ["storageTemperature", "storage_temperature", "Storage Temperature"],
    "temperature_compensation": ["temperatureCompensation", "temperature_compensation", "Temperature Compensation"],
    "temperature_effect": ["temperatureEffect", "temperature_effect", "Temperature Effect", "Temperature Effect On Accuracy"],

    # =========================================================================
    # FLOW SPECIFICATIONS
    # =========================================================================
    "flowType": ["flowType", "flow_type", "measurementType"],
    "flowRange": ["flowRange", "flow_range", "measurementRange"],
    "flow_rate_range": ["flowRateRange", "flow_rate_range", "flowRange"],
    "flow_velocity_range": ["flowVelocityRange", "flow_velocity_range"],
    "minimum_flow_rate": ["minimumFlowRate", "minimum_flow_rate", "minFlow"],
    "maximum_flow_rate": ["maximumFlowRate", "maximum_flow_rate", "maxFlow"],
    "flow_port_type": ["flowPortType", "flow_port_type", "connectionType"],
    "flow_port_size": ["flowPortSize", "flow_port_size", "portSize"],
    "flow_port_material": ["flowPortMaterial", "flow_port_material", "portMaterial"],
    "flow_direction_marking": ["flowDirectionMarking", "flow_direction_marking"],
    "zero_cutoff": ["zeroCutoff", "zero_cutoff", "lowFlowCutoff"],
    "empty_pipe_detection": ["emptyPipeDetection", "empty_pipe_detection"],
    "batch_control": ["batchControl", "batch_control", "batching"],
    "totalizer": ["totalizer", "Totalizer", "totalization"],
    "totalizer_function": ["totalizerFunction", "totalizer_function"],

    # =========================================================================
    # LEVEL SPECIFICATIONS
    # =========================================================================
    "measurementType": ["measurementType", "measurement_type", "technologyType"],
    "levelRange": ["measurementRange", "levelRange", "range", "level_range"],
    "measurement_range": ["measurementRange", "measurement_range", "levelRange"],
    "tank_height": ["tankHeight", "tank_height"],
    "dead_zone": ["deadZone", "dead_zone", "blocking_distance"],
    "blocking_distance": ["blockingDistance", "blocking_distance", "deadZone"],
    "beam_angle": ["beamAngle", "beam_angle"],
    "dielectric_constant": ["dielectricConstant", "dielectric_constant"],

    # =========================================================================
    # PROCESS CONNECTION & PORTS
    # =========================================================================
    "processConnection": ["processConnection", "process_connection", "connection"],
    "process_connection_type": ["processConnectionType", "process_connection_type"],
    "process_connection_size": ["processConnectionSize", "process_connection_size"],
    "flange_rating": ["flangeRating", "flange_rating", "pressureRating"],
    "flange_type": ["flangeType", "flange_type"],
    "flange_size": ["flangeSize", "flange_size"],
    "pipe_size": ["pipeSize", "pipe_size", "lineSize", "line_size"],
    "seal_material": ["sealMaterial", "seal_material", "sealType"],
    "gasket_material": ["gasketMaterial", "gasket_material"],
    "bolting_specification": ["boltingSpecification", "bolting_specification"],
    "strainer_required": ["strainerRequired", "strainer_required"],
    "maintenance_access": ["maintenanceAccess", "maintenance_access"],

    # =========================================================================
    # WETTED MATERIALS (ROOT CAUSE FIX: Added Title Case variations)
    # =========================================================================
    "wettedParts": ["wettedParts", "wetted_parts", "material", "wettedMaterial", "Wetted Parts", "Wetted Material"],
    "wetted_material": ["wettedMaterial", "wetted_material", "wettedParts", "Wetted Material", "Wetted Parts"],
    "sensor_material": ["sensorMaterial", "sensor_material", "Sensor Material"],
    "electrode_material": ["electrodeMaterial", "electrode_material", "Electrode Material"],
    "liner_material": ["linerMaterial", "liner_material", "Liner Material"],
    "diaphragm_material": ["diaphragmMaterial", "diaphragm_material", "Diaphragm Material"],
    "fill_fluid": ["fillFluid", "fill_fluid", "Fill Fluid"],
    "o_ring_material": ["oRingMaterial", "o_ring_material", "O-Ring Material", "O Ring Material"],

    # =========================================================================
    # HUMIDITY & ENVIRONMENTAL (ROOT CAUSE FIX: Added missing fields)
    # =========================================================================
    "humidity_range": ["humidityRange", "humidity_range", "Humidity Range", "Operating Humidity"],
    "humidity": ["humidity", "Humidity", "relativeHumidity", "Relative Humidity"],
    "altitude_range": ["altitudeRange", "altitude_range", "Altitude Range", "Operating Altitude"],
    "emi_immunity": ["emiImmunity", "emi_immunity", "EMI Immunity", "EMC Immunity"],
    "galvanic_isolation": ["galvanicIsolation", "galvanic_isolation", "Galvanic Isolation"],
    "reverse_polarity_protection": ["reversePolarityProtection", "reverse_polarity_protection", "Reverse Polarity Protection"],
    "load_resistance": ["loadResistance", "load_resistance", "Load Resistance"],
    "measurement_update_rate": ["measurementUpdateRate", "measurement_update_rate", "Measurement Update Rate", "Update Rate"],
    "mounting_orientation": ["mountingOrientation", "mounting_orientation", "Mounting Orientation"],
    "mounting_type": ["mountingType", "mounting_type", "Mounting Type"],
    "noise_emission_level": ["noiseEmissionLevel", "noise_emission_level", "Noise Emission Level"],
    "reach_compliance": ["reachCompliance", "reach_compliance", "REACH Compliance", "Reach Compliance"],
    "shock_resistance": ["shockResistance", "shock_resistance", "Shock Resistance"],

    # =========================================================================
    # HOUSING & ENCLOSURE (ROOT CAUSE FIX: Added Title Case variations)
    # =========================================================================
    "housing_material": ["housingMaterial", "housing_material", "enclosureMaterial", "Housing Material", "Enclosure Material"],
    "enclosure_rating": ["enclosureRating", "enclosure_rating", "ipRating", "Enclosure Rating"],
    "enclosure_material": ["enclosureMaterial", "enclosure_material", "housingMaterial", "Enclosure Material"],
    "ingressProtection": ["ingressProtection", "ipRating", "IP", "ip_rating", "Ingress Protection", "IP Rating"],
    "ip_rating": ["ipRating", "ip_rating", "ingressProtection", "IP Rating", "Ingress Protection"],
    "nema_rating": ["nemaRating", "nema_rating", "NEMA Rating"],
    "conduit_entry": ["conduitEntry", "conduit_entry", "Conduit Entry"],
    "cable_entry": ["cableEntry", "cable_entry", "Cable Entry"],
    "cable_gland_material": ["cableGlandMaterial", "cable_gland_material", "Cable Gland Material"],
    "display_type": ["displayType", "display_type", "Display Type"],
    "local_display": ["localDisplay", "local_display", "Local Display"],
    "display_rotation": ["displayRotation", "display_rotation", "Display Rotation"],
    "mounting_bracket_material": ["mountingBracketMaterial", "mounting_bracket_material", "Mounting Bracket Material"],
    "external_fastener_material": ["externalFastenerMaterial", "external_fastener_material", "External Fastener Material"],
    "gasket_material": ["gasketMaterial", "gasket_material", "Gasket Material"],

    # =========================================================================
    # OUTPUT SIGNAL & COMMUNICATION (ROOT CAUSE FIX: Added Title Case variations)
    # =========================================================================
    "output": ["outputSignal", "output_signal", "output", "communicationProtocol", "Output Signal"],
    "outputSignal": ["outputSignal", "output_signal", "output", "Output Signal"],
    "output_signal": ["outputSignal", "output_signal", "primaryOutput", "Output Signal"],
    "primary_output_signal": ["primaryOutputSignal", "primary_output_signal", "Primary Output Signal"],
    "secondary_output": ["secondaryOutput", "secondary_output", "Secondary Output"],
    "protocol": ["protocol", "communicationProtocol", "digital_protocol", "Communication Protocol"],
    "communication_protocol": ["communicationProtocol", "communication_protocol", "protocol", "Communication Protocol"],
    "hart_protocol": ["hartProtocol", "hart_protocol", "HART", "HART Protocol"],
    "fieldbus_protocol": ["fieldbusProtocol", "fieldbus_protocol", "Fieldbus Protocol"],
    "wireless_protocol": ["wirelessProtocol", "wireless_protocol", "Wireless Protocol"],
    "modbus_address": ["modbusAddress", "modbus_address", "Modbus Address"],
    "alarm_outputs": ["alarmOutputs", "alarm_outputs", "Alarm Outputs"],
    "relay_output": ["relayOutput", "relay_output", "Relay Output"],
    "pulse_output": ["pulseOutput", "pulse_output", "Pulse Output"],
    "frequency_output": ["frequencyOutput", "frequency_output", "Frequency Output"],
    "electrical_connection": ["electricalConnection", "electrical_connection", "Electrical Connection"],

    # =========================================================================
    # POWER SUPPLY (ROOT CAUSE FIX: Added Title Case variations)
    # =========================================================================
    "powerSupply": ["powerSupply", "power_supply", "supplyVoltage", "Power Supply"],
    "power_supply": ["powerSupply", "power_supply", "supplyVoltage", "Power Supply"],
    "supply_voltage": ["supplyVoltage", "supply_voltage", "powerSupply", "Supply Voltage"],
    "power_consumption": ["powerConsumption", "power_consumption", "Power Consumption"],
    "loop_power": ["loopPower", "loop_power", "Loop Power"],
    "ex_power_supply": ["exPowerSupply", "ex_power_supply", "Ex Power Supply"],
    "power_supply_variation_effect": ["powerSupplyVariationEffect", "power_supply_variation_effect", "Power Supply Variation Effect On Accuracy"],

    # =========================================================================
    # SAFETY & CERTIFICATIONS (ROOT CAUSE FIX: Added Title Case with space variations)
    # =========================================================================
    "silRating": ["safetyRating", "silRating", "sil_rating", "SIL", "SIL Rating", "Safety Integrity Level"],
    "sil_rating": ["silRating", "sil_rating", "safetyRating", "SIL", "SIL Rating", "Safety Integrity Level"],
    "sil_capability": ["silCapability", "sil_capability", "SIL Capability"],
    "hazardousAreaRating": ["hazardousAreaRating", "atexRating", "hazardous_area", "ATEX", "Hazardous Area Rating"],
    "hazardous_area_rating": ["hazardousAreaRating", "hazardous_area_rating", "atexRating", "Hazardous Area Rating"],
    "atex_rating": ["atexRating", "atex_rating", "hazardousAreaRating", "ATEX Rating"],
    "iecex_rating": ["iecexRating", "iecex_rating", "IECEx Rating", "IEC Ex Rating"],
    "explosion_protection": ["explosionProtection", "explosion_protection", "Explosion Protection", "Explosion Proof"],
    "protection_type": ["protectionType", "protection_type", "Protection Type", "Protection Rating"],
    "certifications": ["certifications", "certification", "approvals", "Certifications", "Certification", "Approvals"],
    "ce_marking": ["ceMarking", "ce_marking", "CE Marking", "CE Mark"],
    "ul_listing": ["ulListing", "ul_listing", "UL Listing"],
    "csa_certification": ["csaCertification", "csa_certification", "CSA Certification"],
    "pressure_equipment_directive": ["pressureEquipmentDirective", "pressure_equipment_directive", "PED", "Pressure Equipment Directive"],

    # =========================================================================
    # TEMPERATURE SENSOR SPECIFIC
    # =========================================================================
    "sensorType": ["sensorType", "sensor_type", "type"],
    "sensor_type": ["sensorType", "sensor_type", "elementType"],
    "element_type": ["elementType", "element_type", "sensorType"],
    "rtd_type": ["rtdType", "rtd_type"],
    "thermocouple_type": ["thermocoupleType", "thermocouple_type"],
    "class_accuracy": ["classAccuracy", "class_accuracy"],
    "responseTime": ["responseTime", "response_time"],
    "response_time": ["responseTime", "response_time"],
    "sheathMaterial": ["sheathMaterial", "sheath_material", "probeMaterial"],
    "sheath_material": ["sheathMaterial", "sheath_material"],
    "probe_diameter": ["probeDiameter", "probe_diameter"],
    "probe_length": ["probeLength", "probe_length"],
    "insertion_length": ["insertionLength", "insertion_length"],
    "immersion_length": ["immersionLength", "immersion_length"],
    "thermowell_material": ["thermowellMaterial", "thermowell_material"],
    "wake_frequency": ["wakeFrequency", "wake_frequency"],

    # =========================================================================
    # DAMPING & RESPONSE
    # =========================================================================
    "damping": ["damping", "Damping", "responseDamping"],
    "damping_type": ["dampingType", "damping_type"],
    "damping_time": ["dampingTime", "damping_time"],
    "update_rate": ["updateRate", "update_rate"],
    "sampling_rate": ["samplingRate", "sampling_rate"],

    # =========================================================================
    # DIAGNOSTICS & FEATURES
    # =========================================================================
    "advanced_diagnostics": ["advancedDiagnostics", "advanced_diagnostics", "diagnostics"],
    "self_diagnostics": ["selfDiagnostics", "self_diagnostics"],
    "device_diagnostics": ["deviceDiagnostics", "device_diagnostics"],
    "configuration_backup": ["configurationBackup", "configuration_backup"],
    "simulation_mode": ["simulationMode", "simulation_mode"],
    "historian_function": ["historianFunction", "historian_function"],

    # =========================================================================
    # FLUID PROPERTIES
    # =========================================================================
    "fluidType": ["fluidType", "fluid_type", "medium"],
    "fluid_type": ["fluidType", "fluid_type", "processFluid"],
    "compatible_fluid_types": ["compatibleFluidTypes", "compatible_fluid_types"],
    "viscosity_range": ["viscosityRange", "viscosity_range"],
    "specific_gravity_range": ["specificGravityRange", "specific_gravity_range"],
    "density_range": ["densityRange", "density_range"],
    "density_compensation": ["densityCompensation", "density_compensation"],
    "conductivity_range": ["conductivityRange", "conductivity_range"],
    "particle_contamination_limit": ["particleContaminationLimit", "particle_contamination_limit"],
    "water_content_limit": ["waterContentLimit", "water_content_limit"],

    # =========================================================================
    # CALIBRATION & VERIFICATION (ROOT CAUSE FIX: Added Title Case variations)
    # =========================================================================
    "calibration_interval": ["calibrationInterval", "calibration_interval", "Calibration Interval"],
    "calibration_certificate": ["calibrationCertificate", "calibration_certificate", "Calibration Certificate"],
    "traceable_calibration": ["traceableCalibration", "traceable_calibration", "Traceable Calibration"],
    "verification_function": ["verificationFunction", "verification_function", "Verification Function"],
    "zero_adjustment": ["zeroAdjustment", "zero_adjustment", "Zero Adjustment"],
    "span_adjustment": ["spanAdjustment", "span_adjustment", "Span Adjustment"],

    # =========================================================================
    # MECHANICAL & MOUNTING (ROOT CAUSE FIX: Added Title Case variations)
    # =========================================================================
    "mounting_position": ["mountingPosition", "mounting_position", "Mounting Position"],
    "mounting_type": ["mountingType", "mounting_type", "Mounting Type"],
    "orientation": ["orientation", "Orientation"],
    "weight": ["weight", "Weight"],
    "dimensions": ["dimensions", "Dimensions"],
    "vibration_resistance": ["vibrationResistance", "vibration_resistance", "Vibration Resistance"],
    "shock_resistance": ["shockResistance", "shock_resistance", "Shock Resistance"],
    "vibration_damping": ["vibrationDamping", "vibration_damping", "Vibration Damping"],
    "process_connection": ["processConnection", "process_connection", "Process Connection"],
    "response_time": ["responseTime", "response_time", "Response Time"],

    # =========================================================================
    # SERVICE & WARRANTY
    # =========================================================================
    "warranty": ["warranty", "Warranty"],
    "expected_lifetime": ["expectedLifetime", "expected_lifetime"],
    "mtbf": ["mtbf", "MTBF", "mean_time_between_failure"],
    "spare_parts_availability": ["sparePartsAvailability", "spare_parts_availability"],

    # =========================================================================
    # STANDARDS COMPLIANCE
    # =========================================================================
    "applicable_standards": ["applicableStandards", "applicable_standards"],
    "iec_standards": ["iecStandards", "iec_standards"],
    "iso_standards": ["isoStandards", "iso_standards"],
    "namur_compliance": ["namurCompliance", "namur_compliance"],
    "ne107_compliance": ["ne107Compliance", "ne107_compliance"],
}


def _normalize_field_name(field_name: str) -> str:
    """Normalize field name for matching - removes underscores, dashes, spaces and lowercases."""
    return field_name.lower().replace("_", "").replace("-", "").replace(" ", "")


def _to_title_case_with_space(field_name: str) -> str:
    """
    Convert field name to Title Case with spaces.
    ROOT CAUSE FIX: This creates the UI-friendly display format.

    Examples:
        "ambient_temperature" -> "Ambient Temperature"
        "sil_rating" -> "SIL Rating"
        "pressureRange" -> "Pressure Range"
    """
    # Handle snake_case
    if '_' in field_name:
        parts = field_name.split('_')
        return ' '.join(p.title() if p.lower() != 'sil' else 'SIL' for p in parts)

    # Handle camelCase
    result = []
    current_word = []
    for i, char in enumerate(field_name):
        if char.isupper() and i > 0:
            if current_word:
                word = ''.join(current_word)
                result.append(word.upper() if word.lower() == 'sil' else word.title())
            current_word = [char]
        else:
            current_word.append(char)

    if current_word:
        word = ''.join(current_word)
        result.append(word.upper() if word.lower() == 'sil' else word.title())

    return ' '.join(result) if result else field_name.title()


def _snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split('_')
    return components[0].lower() + ''.join(x.title() for x in components[1:])


def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    result = []
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            result.append('_')
        result.append(char.lower())
    return ''.join(result)


def _get_field_variations(field_name: str) -> List[str]:
    """
    Generate all common variations of a field name for matching.

    ROOT CAUSE FIX: Now includes Title Case with space variation to match UI display names.

    Args:
        field_name: The field name to generate variations for

    Returns:
        List of field name variations
    """
    variations = [field_name]

    # Add snake_case and camelCase conversions
    if '_' in field_name:
        # It's snake_case, add camelCase
        variations.append(_snake_to_camel(field_name))
    else:
        # It might be camelCase, add snake_case
        variations.append(_camel_to_snake(field_name))

    # Add Title Case variation (no spaces)
    variations.append(field_name.replace('_', ' ').title().replace(' ', ''))

    # ROOT CAUSE FIX: Add Title Case WITH spaces (matches UI display format)
    variations.append(_to_title_case_with_space(field_name))

    # Add lowercase and uppercase
    variations.append(field_name.lower())
    variations.append(field_name.upper())

    # Add normalized version
    variations.append(_normalize_field_name(field_name))

    # ROOT CAUSE FIX: Add space-separated versions
    if '_' in field_name:
        variations.append(field_name.replace('_', ' '))
        variations.append(field_name.replace('_', ' ').title())
        variations.append(field_name.replace('_', ' ').lower())

    return list(set(variations))


def _find_schema_field_match(spec_key: str, schema_fields: List[str]) -> Optional[str]:
    """
    Find the matching schema field for a specification key.
    Uses multiple matching strategies for robust field mapping.

    Args:
        spec_key: The specification key from Deep Agent
        schema_fields: List of available schema field names

    Returns:
        Matching schema field name or None
    """
    if not spec_key or not schema_fields:
        return None

    # Strategy 1: Try direct mapping first (fastest)
    if spec_key in SPEC_TO_SCHEMA_MAPPING:
        for possible_match in SPEC_TO_SCHEMA_MAPPING[spec_key]:
            if possible_match in schema_fields:
                return possible_match
            # Try normalized match against mapped values
            normalized_possible = _normalize_field_name(possible_match)
            for schema_field in schema_fields:
                if _normalize_field_name(schema_field) == normalized_possible:
                    return schema_field

    # Strategy 2: Try exact match with case variations
    spec_variations = _get_field_variations(spec_key)
    for variation in spec_variations:
        if variation in schema_fields:
            return variation

    # Strategy 3: Try normalized exact match
    normalized_spec = _normalize_field_name(spec_key)
    for schema_field in schema_fields:
        if _normalize_field_name(schema_field) == normalized_spec:
            return schema_field

    # Strategy 4: Try matching with common suffixes/prefixes removed
    # e.g., "accuracy" should match "measurementAccuracy"
    common_prefixes = ["measurement", "process", "ambient", "operating", "primary", "secondary"]

    for schema_field in schema_fields:
        normalized_schema = _normalize_field_name(schema_field)

        # Check if spec is a suffix of schema field
        if normalized_schema.endswith(normalized_spec):
            return schema_field

        # Check if schema is a suffix of spec
        if normalized_spec.endswith(normalized_schema):
            return schema_field

        # Check with common prefixes removed
        for prefix in common_prefixes:
            if normalized_schema.startswith(prefix) and normalized_schema[len(prefix):] == normalized_spec:
                return schema_field
            if normalized_spec.startswith(prefix) and normalized_spec[len(prefix):] == normalized_schema:
                return schema_field

    # Strategy 5: Partial substring match (more lenient, used as fallback)
    # Only match if the spec is a significant part of the field name (>50% overlap)
    for schema_field in schema_fields:
        normalized_schema = _normalize_field_name(schema_field)

        # Calculate overlap significance
        if len(normalized_spec) >= 4 and len(normalized_schema) >= 4:
            if normalized_spec in normalized_schema:
                overlap_ratio = len(normalized_spec) / len(normalized_schema)
                if overlap_ratio >= 0.5:  # At least 50% of schema field is the spec
                    return schema_field

            if normalized_schema in normalized_spec:
                overlap_ratio = len(normalized_schema) / len(normalized_spec)
                if overlap_ratio >= 0.5:  # At least 50% of spec is the schema field
                    return schema_field

    return None


def _extract_all_schema_fields(schema: Dict[str, Any]) -> List[str]:
    """
    Recursively extract all field names from a schema.

    Args:
        schema: Schema dictionary

    Returns:
        List of all field names
    """
    fields = []

    def extract_from_dict(obj: Any):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.startswith("_") or key in ["standards", "normalized_category"]:
                    continue
                fields.append(key)
                if isinstance(value, dict):
                    extract_from_dict(value)

    # Extract from all sections
    for section in ["mandatory", "mandatory_requirements", "optional", "optional_requirements",
                    "Compliance", "Electrical", "Mechanical", "Performance", "Environmental", "Features"]:
        if section in schema:
            extract_from_dict(schema[section])

    # Also extract from root level
    extract_from_dict(schema)

    # Deduplicate
    return list(set(fields))


def _get_spec_value(spec_data: Any) -> Dict[str, Any]:
    """
    Extract the specification data with full metadata from a specification entry.
    Returns a dictionary with value, source, confidence, and standards_referenced.
    """
    from ..schema_field_extractor import extract_standards_from_value
    
    if isinstance(spec_data, dict):
        value = spec_data.get("value", "")
        if not value and not spec_data.get("confidence"):
            # This is just a nested dict, convert it to string
            value = str(spec_data)
        
        # Extract standards from value if not already present
        existing_refs = spec_data.get("standards_referenced", [])
        if not existing_refs and value:
            existing_refs = extract_standards_from_value(str(value))
        
        return {
            "value": value if value else str(spec_data),
            "source": spec_data.get("source", "standards_specifications"),
            "confidence": spec_data.get("confidence", 0.8),
            "standards_referenced": existing_refs
        }
    
    # For plain string values
    value_str = str(spec_data)
    standards_refs = extract_standards_from_value(value_str)
    return {
        "value": value_str,
        "source": "standards_specifications",
        "confidence": 0.8,
        "standards_referenced": standards_refs
    }


# =============================================================================
# SCHEMA LOADING
# =============================================================================

def load_schema_for_product(product_type: str) -> Dict[str, Any]:
    """
    Load the schema for a product type.

    Args:
        product_type: Product type name

    Returns:
        Schema dictionary with mandatory and optional fields
    """
    try:
        from common.tools.schema_tools import load_schema_tool

        result = load_schema_tool.invoke({
            "product_type": product_type,
            "enable_ppi": False  # Don't generate new schema, use existing
        })

        if result.get("success") and result.get("schema"):
            logger.info(f"[DEEP_AGENT_INTEGRATION] Loaded schema for: {product_type}")
            return result["schema"]
        else:
            logger.warning(f"[DEEP_AGENT_INTEGRATION] No schema found for: {product_type}")
            return {}

    except Exception as e:
        logger.error(f"[DEEP_AGENT_INTEGRATION] Failed to load schema for {product_type}: {e}")
        return {}


# =============================================================================
# SCHEMA POPULATION
# =============================================================================

def populate_schema_from_deep_agent(
    product_type: str,
    schema: Dict[str, Any],
    deep_agent_specs: Dict[str, Any],
    applicable_standards: List[Dict[str, Any]] = None,
    certifications: List[str] = None
) -> Dict[str, Any]:
    """
    Populate schema field VALUES from Deep Agent specifications.

    This is the KEY function that fills in schema fields with values
    extracted from standards documents by the Deep Agent.

    Args:
        product_type: Product type
        schema: Schema with empty or partial field values
        deep_agent_specs: Specifications extracted by Deep Agent
        applicable_standards: List of applicable standards codes
        certifications: List of certifications

    Returns:
        Schema with populated field values from Deep Agent specifications
    """
    logger.info(f"[DEEP_AGENT_INTEGRATION] Populating schema fields for: {product_type}")

    if not schema:
        logger.warning("[DEEP_AGENT_INTEGRATION] Empty schema provided")
        return schema

    # Create deep copy
    populated_schema = json.loads(json.dumps(schema))

    # Get all available schema fields
    schema_fields = _extract_all_schema_fields(schema)
    logger.info(f"[DEEP_AGENT_INTEGRATION] Found {len(schema_fields)} schema fields")

    # Track populated fields
    fields_populated = 0

    # Get the specifications from Deep Agent
    # Handle both nested (mandatory/safety) and flat structure
    all_specs = {}
    if isinstance(deep_agent_specs, dict):
        # Extract from nested structures
        for key, value in deep_agent_specs.items():
            if isinstance(value, dict):
                # Nested section (like "mandatory", "safety", "optional")
                for sub_key, sub_value in value.items():
                    all_specs[sub_key] = sub_value
            else:
                all_specs[key] = value

    logger.info(f"[DEEP_AGENT_INTEGRATION] Processing {len(all_specs)} specifications from Deep Agent")

    # Build a reverse lookup: normalized spec key -> (original key, value)
    spec_lookup = {}
    for spec_key, spec_value in all_specs.items():
        normalized = _normalize_field_name(spec_key)
        spec_lookup[normalized] = (spec_key, spec_value)
        # Also add variations
        for variation in _get_field_variations(spec_key):
            norm_var = _normalize_field_name(variation)
            if norm_var not in spec_lookup:
                spec_lookup[norm_var] = (spec_key, spec_value)

    def _is_empty_or_not_specified(value: Any) -> bool:
        """Check if a value is empty or 'not specified'."""
        if value is None:
            return True
        if isinstance(value, str):
            v = value.strip().lower()
            return not v or v in ["not specified", "n/a", "none", "null", ""]
        if isinstance(value, dict):
            # Check if nested dict has empty value
            inner_val = value.get("value", "")
            return _is_empty_or_not_specified(inner_val)
        return False

    def _is_nested_schema_field(field_value: dict) -> bool:
        """
        Check if a dict represents a nested schema field (with value/description structure)
        vs a section containing more fields.
        """
        if not isinstance(field_value, dict):
            return False
        # Nested schema fields typically have 'value', 'description', or 'source' keys
        schema_markers = {"value", "description", "source", "confidence", "standards_referenced"}
        field_keys = set(field_value.keys())
        return bool(field_keys & schema_markers)

    # Populate schema fields from Deep Agent specifications
    def update_schema_section(section: Dict[str, Any], section_name: str):
        nonlocal fields_populated

        for field_key, field_value in list(section.items()):
            if field_key.startswith("_"):
                continue

            if isinstance(field_value, dict):
                if _is_nested_schema_field(field_value):
                    # This is a nested schema field like {value: "", description: "...", source: ""}
                    # The field_key IS the spec name we want to match
                    current_value = field_value.get("value", "")
                    if _is_empty_or_not_specified(current_value):
                        # Try to find matching spec using the field_key (e.g., "accuracy")
                        matched_spec = None

                        # Direct lookup via normalized field key
                        normalized_field = _normalize_field_name(field_key)
                        if normalized_field in spec_lookup:
                            matched_spec = spec_lookup[normalized_field]

                        # Try via mapping
                        if not matched_spec:
                            for spec_key, spec_value in all_specs.items():
                                if _find_schema_field_match(spec_key, [field_key]):
                                    matched_spec = (spec_key, spec_value)
                                    break

                        if matched_spec:
                            orig_key, spec_value = matched_spec
                            spec_data = _get_spec_value(spec_value)
                            # Update the nested structure
                            section[field_key]["value"] = spec_data.get("value", "")
                            section[field_key]["source"] = spec_data.get("source", "standards_specifications")
                            if "confidence" in section[field_key] or "confidence" in spec_data:
                                section[field_key]["confidence"] = spec_data.get("confidence", 0.8)
                            if "standards_referenced" in section[field_key] or spec_data.get("standards_referenced"):
                                section[field_key]["standards_referenced"] = spec_data.get("standards_referenced", [])
                            fields_populated += 1
                            logger.debug(f"[DEEP_AGENT_INTEGRATION] Populated nested {field_key} = {spec_data.get('value', '')[:50]}")
                else:
                    # This is a section containing more fields - recurse
                    update_schema_section(field_value, f"{section_name}.{field_key}")

            elif isinstance(field_value, str):
                # Simple string field - check if value is empty or not specified
                if _is_empty_or_not_specified(field_value):
                    # Try to find matching spec from Deep Agent
                    matched_spec = None

                    # Direct lookup via normalized field key
                    normalized_field = _normalize_field_name(field_key)
                    if normalized_field in spec_lookup:
                        matched_spec = spec_lookup[normalized_field]

                    # Try via mapping
                    if not matched_spec:
                        for spec_key, spec_value in all_specs.items():
                            if _find_schema_field_match(spec_key, [field_key]):
                                matched_spec = (spec_key, spec_value)
                                break

                    if matched_spec:
                        orig_key, spec_value = matched_spec
                        # Replace simple string with nested structure for consistency
                        section[field_key] = _get_spec_value(spec_value)
                        fields_populated += 1
                        logger.debug(f"[DEEP_AGENT_INTEGRATION] Populated simple {field_key}")

    # Update mandatory requirements
    for section_key in ["mandatory", "mandatory_requirements"]:
        if section_key in populated_schema and isinstance(populated_schema[section_key], dict):
            update_schema_section(populated_schema[section_key], section_key)

    # Update optional requirements
    for section_key in ["optional", "optional_requirements"]:
        if section_key in populated_schema and isinstance(populated_schema[section_key], dict):
            update_schema_section(populated_schema[section_key], section_key)

    # Update other sections (Compliance, Electrical, etc.)
    for section_key in ["Compliance", "Electrical", "Mechanical", "Performance", "Environmental", 
                        "Features", "Integration", "MechanicalOptions", "ServiceAndSupport", "Certifications"]:
        if section_key in populated_schema and isinstance(populated_schema[section_key], dict):
            update_schema_section(populated_schema[section_key], section_key)

    # =========================================================================
    # SECONDARY PASS: Use Schema Field Extractor for remaining empty fields
    # This ensures all fields get populated with standards-based values
    # =========================================================================
    try:
        from ..schema_field_extractor import extract_schema_field_values_from_standards
        
        logger.info(f"[DEEP_AGENT_INTEGRATION] Running secondary extraction for remaining fields...")
        populated_schema = extract_schema_field_values_from_standards(
            product_type=product_type,
            schema=populated_schema
        )
        
        # Get secondary pass stats
        secondary_stats = populated_schema.get("_schema_field_extraction", {})
        secondary_fields = secondary_stats.get("fields_populated", 0)
        fields_populated += secondary_fields
        
        if secondary_fields > 0:
            logger.info(f"[DEEP_AGENT_INTEGRATION] Secondary extraction populated {secondary_fields} additional fields")
    except Exception as e:
        logger.warning(f"[DEEP_AGENT_INTEGRATION] Secondary extraction failed: {e}")

    # Add standards section if not present
    if applicable_standards and "standards" not in populated_schema:
        populated_schema["standards"] = {
            "applicable_standards": [
                s.get("code", s) if isinstance(s, dict) else s
                for s in applicable_standards
            ],
            "source": "deep_agent"
        }

    # Add certifications to schema
    if certifications:
        # Find appropriate place for certifications
        for section_key in ["optional", "optional_requirements", "Compliance", "Certifications"]:
            if section_key in populated_schema:
                if isinstance(populated_schema[section_key], dict):
                    if "certifications" not in populated_schema[section_key]:
                        populated_schema[section_key]["certifications"] = ", ".join(certifications)
                    # Also update Suggested Values if present
                    if "Suggested Values" in populated_schema[section_key]:
                        populated_schema[section_key]["Suggested Values"] = ", ".join(certifications)
                break

    # Add population metadata
    populated_schema["_deep_agent_population"] = {
        "product_type": product_type,
        "fields_populated": fields_populated,
        "total_specs_from_deep_agent": len(all_specs),
        "standards_count": len(applicable_standards) if applicable_standards else 0,
        "certifications_count": len(certifications) if certifications else 0,
        "source": "deep_agent_standards"
    }

    logger.info(f"[DEEP_AGENT_INTEGRATION] Populated {fields_populated} schema fields (Deep Agent + Standards Defaults)")

    return populated_schema


# =============================================================================
# INPUT PREPARATION
# =============================================================================

def prepare_deep_agent_input(
    all_items: List[Dict[str, Any]],
    user_input: str,
    solution_context: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None
) -> Dict[str, Any]:
    """
    Prepare input data for Deep Agent from identified items and context.

    Args:
        all_items: List of identified instruments and accessories
        user_input: User's original input/requirements
        solution_context: Solution analysis context (optional)
        domain: Domain/industry context (optional)

    Returns:
        Deep Agent input dictionary
    """
    logger.info("[DEEP_AGENT_INTEGRATION] Preparing Deep Agent input")

    # Extract safety requirements from solution context
    safety_requirements = {}
    environmental_conditions = {}

    if solution_context:
        safety_req = solution_context.get("safety_requirements", {})
        safety_requirements = {
            "sil_level": safety_req.get("sil_level"),
            "hazardous_area": safety_req.get("hazardous_area"),
            "atex_zone": safety_req.get("atex_zone")
        }

        env_context = solution_context.get("environmental", {})
        environmental_conditions = {
            "location": env_context.get("location"),
            "conditions": env_context.get("conditions"),
            "temperature_range": solution_context.get("key_parameters", {}).get("temperature_range"),
            "pressure_range": solution_context.get("key_parameters", {}).get("pressure_range")
        }

    # Convert all_items to Deep Agent format
    identified_items = []
    for item in all_items:
        item_data = {
            "product_type": item.get("name") or item.get("product_name", "Unknown"),
            "category": item.get("category", "Instrument" if item.get("type") == "instrument" else "Accessory"),
            "type": item.get("type", "instrument"),
            "quantity": item.get("quantity", 1),
            "sample_input": item.get("sample_input", ""),
            "specifications": item.get("specifications", {}),
            "purpose": item.get("purpose", "")
        }
        identified_items.append(item_data)

    deep_agent_input = {
        "user_input": user_input,
        "domain": domain or (solution_context.get("industry") if solution_context else "General"),
        "identified_items": identified_items,
        "safety_requirements": safety_requirements,
        "environmental_conditions": environmental_conditions,
        "solution_context": solution_context or {}
    }

    logger.info(f"[DEEP_AGENT_INTEGRATION] Prepared input with {len(identified_items)} items")
    return deep_agent_input


# =============================================================================
# MAIN INTEGRATION FUNCTION
# =============================================================================

def integrate_deep_agent_specifications(
    all_items: List[Dict[str, Any]],
    user_input: str,
    solution_context: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
    enable_schema_population: bool = True
) -> List[Dict[str, Any]]:
    """
    Run Deep Agent and extract technical specifications from standards documents.

    This function:
    1. Runs Deep Agent to analyze standards documents
    2. Extracts technical specifications for each product type
    3. Optionally loads schema and populates field values (if enable_schema_population=True)
    4. Returns items with extracted specifications

    Args:
        all_items: List of identified instruments and accessories
        user_input: User's original input/requirements
        solution_context: Solution analysis context (optional)
        domain: Domain/industry context (optional)
        enable_schema_population: If True, load schemas and populate fields.
                                  If False, only extract raw specifications
                                  (faster, for Solution Workflow).

    Returns:
        Enriched items list with extracted specifications (and optionally populated schemas)
    """
    logger.info(f"[DEEP_AGENT_INTEGRATION] Starting Deep Agent specification integration")
    logger.info(f"[DEEP_AGENT_INTEGRATION] Schema population: {'ENABLED' if enable_schema_population else 'DISABLED (specs-only mode)'}")

    try:
        # Import here to avoid circular imports
        from ..workflows.workflow import run_deep_agent_workflow

        # Generate session ID for this Deep Agent execution
        session_id = f"integration_{uuid4().hex[:12]}"

        # Prepare input for Deep Agent
        deep_agent_input = prepare_deep_agent_input(
            all_items=all_items,
            user_input=user_input,
            solution_context=solution_context,
            domain=domain
        )

        # Convert identified_items to separate instruments and accessories lists
        identified_instruments = []
        identified_accessories = []
        for item in deep_agent_input.get("identified_items", []):
            if item.get("type") == "accessory":
                identified_accessories.append({
                    "category": item.get("category", "Accessory"),
                    "accessory_name": item.get("product_type", "Unknown Accessory"),
                    "quantity": item.get("quantity", 1),
                    "specifications": item.get("specifications", {}),
                    "related_instrument": item.get("related_instrument", "")
                })
            else:
                identified_instruments.append({
                    "category": item.get("category", "Instrument"),
                    "product_name": item.get("product_type", "Unknown Instrument"),
                    "quantity": item.get("quantity", 1),
                    "specifications": item.get("specifications", {}),
                    "sample_input": item.get("sample_input", "")
                })

        # Run Deep Agent workflow synchronously
        logger.info(f"[DEEP_AGENT_INTEGRATION] Running Deep Agent for session {session_id}")
        logger.info(f"[DEEP_AGENT_INTEGRATION] Items: {len(identified_instruments)} instruments, {len(identified_accessories)} accessories")
        final_state = run_deep_agent_workflow(
            user_input=deep_agent_input["user_input"],
            session_id=session_id,
            identified_instruments=identified_instruments,
            identified_accessories=identified_accessories
        )

        # Extract specifications from Deep Agent output
        # The Deep Agent workflow returns standard_specifications_json with instruments and accessories
        standard_specs_json = final_state.get("standard_specifications_json", {})
        
        # Combine instruments and accessories into a single list for iteration
        generated_specs = []
        if standard_specs_json:
            instrument_specs = standard_specs_json.get("instruments", [])
            accessory_specs = standard_specs_json.get("accessories", [])
            generated_specs = instrument_specs + accessory_specs
            logger.info(f"[DEEP_AGENT_INTEGRATION] Extracted {len(instrument_specs)} instrument specs and {len(accessory_specs)} accessory specs")
        else:
            # Fallback: try legacy format
            generated_specs = final_state.get("generated_specifications", [])
            logger.info(f"[DEEP_AGENT_INTEGRATION] Using legacy format: {len(generated_specs)} specs")

        # Merge specifications with original items AND populate schema fields
        enriched_items = []
        for i, item in enumerate(all_items):
            enriched_item = item.copy()
            product_type = item.get("name") or item.get("product_name", "Unknown")

            # Find corresponding spec from Deep Agent
            if i < len(generated_specs):
                deep_agent_spec = generated_specs[i]

                # Get Deep Agent specifications
                deep_specs = deep_agent_spec.get("specifications", {})
                applicable_standards = deep_agent_spec.get("applicable_standards", [])
                certifications = deep_agent_spec.get("certifications", [])
                guidelines = deep_agent_spec.get("guidelines", [])
                sources_used = deep_agent_spec.get("sources_used", [])

                # ==========================================================
                # CONDITIONAL: Load and populate schema only if enabled
                # For Solution Workflow, we skip this to save time
                # ==========================================================

                if enable_schema_population:
                    # Load schema for this product type
                    product_schema = load_schema_for_product(product_type)

                    if product_schema:
                        # POPULATE SCHEMA FIELD VALUES from Deep Agent specifications
                        populated_schema = populate_schema_from_deep_agent(
                            product_type=product_type,
                            schema=product_schema,
                            deep_agent_specs=deep_specs,
                            applicable_standards=applicable_standards,
                            certifications=certifications
                        )

                        # Store populated schema in item
                        enriched_item["schema"] = populated_schema
                        enriched_item["schema_populated"] = True

                        logger.info(
                            f"[DEEP_AGENT_INTEGRATION] Populated schema for {product_type}: "
                            f"{populated_schema.get('_deep_agent_population', {}).get('fields_populated', 0)} fields"
                        )
                    else:
                        # No schema found - store specifications directly
                        enriched_item["schema"] = {
                            "specifications_from_standards": deep_specs,
                            "_note": "No predefined schema found, using raw specifications"
                        }
                        enriched_item["schema_populated"] = False
                else:
                    # Schema population disabled - just store extracted specifications
                    # This is much faster and suitable for Solution Workflow
                    enriched_item["schema_populated"] = False
                    logger.debug(f"[DEEP_AGENT_INTEGRATION] Skipping schema population for {product_type} (disabled)")

                # Always store raw Deep Agent specifications for reference
                enriched_item["deep_agent_specifications"] = deep_specs
                enriched_item["applicable_standards"] = applicable_standards
                enriched_item["certifications"] = certifications
                enriched_item["guidelines"] = guidelines
                enriched_item["sources_used"] = sources_used
                enriched_item["enrichment_status"] = "success"

            else:
                enriched_item["enrichment_status"] = "no_match"
                enriched_item["schema_populated"] = False
                logger.warning(f"[DEEP_AGENT_INTEGRATION] No spec found for item {i}: {item.get('name')}")

            enriched_items.append(enriched_item)

        # Count successful enrichments
        specs_extracted = sum(1 for item in enriched_items if item.get("enrichment_status") == "success")
        schemas_populated = sum(1 for item in enriched_items if item.get("schema_populated", False))

        logger.info(
            f"[DEEP_AGENT_INTEGRATION] Successfully enriched {len(enriched_items)} items: "
            f"{specs_extracted} specs extracted, {schemas_populated} schemas populated"
        )

        return enriched_items

    except Exception as e:
        logger.error(f"[DEEP_AGENT_INTEGRATION] Deep Agent integration failed: {e}")
        import traceback
        traceback.print_exc()

        # Return items with error status
        for item in all_items:
            item["enrichment_status"] = "failed"
            item["enrichment_error"] = str(e)
            item["schema_populated"] = False

        return all_items


# =============================================================================
# ASYNC VERSION
# =============================================================================

async def run_deep_agent_for_specifications(
    all_items: List[Dict[str, Any]],
    user_input: str,
    solution_context: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None
) -> Dict[str, Any]:
    """
    Async version: Run Deep Agent workflow to generate specifications.

    Args:
        all_items: List of identified instruments and accessories
        user_input: User's original input/requirements
        solution_context: Solution analysis context (optional)
        domain: Domain/industry context (optional)

    Returns:
        Dictionary with enriched items and Deep Agent results
    """
    logger.info("[DEEP_AGENT_INTEGRATION] Starting async Deep Agent workflow")

    try:
        from ..workflows.workflow import get_deep_agent_workflow, create_deep_agent_state

        session_id = f"integration_{uuid4().hex[:12]}"

        deep_agent_input = prepare_deep_agent_input(
            all_items=all_items,
            user_input=user_input,
            solution_context=solution_context,
            domain=domain
        )

        # Convert identified_items to separate instruments and accessories lists
        identified_instruments = []
        identified_accessories = []
        for item in deep_agent_input.get("identified_items", []):
            if item.get("type") == "accessory":
                identified_accessories.append({
                    "category": item.get("category", "Accessory"),
                    "accessory_name": item.get("product_type", "Unknown Accessory"),
                    "quantity": item.get("quantity", 1),
                    "specifications": item.get("specifications", {}),
                    "related_instrument": item.get("related_instrument", "")
                })
            else:
                identified_instruments.append({
                    "category": item.get("category", "Instrument"),
                    "product_name": item.get("product_type", "Unknown Instrument"),
                    "quantity": item.get("quantity", 1),
                    "specifications": item.get("specifications", {}),
                    "sample_input": item.get("sample_input", "")
                })

        # Create initial state
        initial_state = create_deep_agent_state(
            user_input=deep_agent_input["user_input"],
            session_id=session_id,
            identified_instruments=identified_instruments,
            identified_accessories=identified_accessories
        )

        workflow = get_deep_agent_workflow()

        logger.info(f"[DEEP_AGENT_INTEGRATION] Running async Deep Agent for session {session_id}")
        final_state = await workflow.ainvoke(
            initial_state,
            {"recursion_limit": 100, "configurable": {"thread_id": session_id}}
        )

        # Extract specifications from the correct key
        standard_specs_json = final_state.get("standard_specifications_json", {})
        specifications = []
        if standard_specs_json:
            specifications = standard_specs_json.get("instruments", []) + standard_specs_json.get("accessories", [])

        return {
            "success": True,
            "session_id": session_id,
            "specifications": specifications,
            "aggregated_specs": standard_specs_json,
            "thread_results": final_state.get("thread_results", {}),
            "processing_time_ms": final_state.get("processing_time_ms", 0),
            "documents_analyzed": standard_specs_json.get("metadata", {}).get("documents_analyzed", 0)
        }

    except Exception as e:
        logger.error(f"[DEEP_AGENT_INTEGRATION] Async Deep Agent workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "specifications": [],
            "aggregated_specs": {}
        }


# =============================================================================
# DISPLAY FORMATTING
# =============================================================================

def format_deep_agent_specs_for_display(
    items_with_specs: List[Dict[str, Any]]
) -> str:
    """
    Format Deep Agent specifications for user display.

    Args:
        items_with_specs: Items enriched with Deep Agent specifications

    Returns:
        Formatted markdown string for display
    """
    lines = []
    lines.append("## Deep Agent Specifications Analysis\n")

    for i, item in enumerate(items_with_specs, 1):
        lines.append(f"### {i}. {item.get('name', 'Unknown')}")
        lines.append(f"   **Type:** {item.get('type', 'Unknown').capitalize()}")
        lines.append(f"   **Category:** {item.get('category', 'Unknown')}")
        lines.append(f"   **Quantity:** {item.get('quantity', 1)}")
        lines.append(f"   **Schema Populated:** {'Yes' if item.get('schema_populated') else 'No'}\n")

        # Show populated schema
        schema = item.get("schema", {})
        if schema and item.get("schema_populated"):
            population_info = schema.get("_deep_agent_population", {})
            lines.append(f"   **Schema Fields Populated:** {population_info.get('fields_populated', 0)}")

            # Show mandatory requirements
            mandatory = schema.get("mandatory_requirements", schema.get("mandatory", {}))
            if mandatory:
                lines.append("   **Mandatory Requirements (from standards):**")
                for key, value in mandatory.items():
                    if not key.startswith("_") and value:
                        lines.append(f"   - {key}: {value}")

            # Show optional requirements
            optional = schema.get("optional_requirements", schema.get("optional", {}))
            if optional:
                lines.append("   **Optional Requirements (from standards):**")
                for key, value in list(optional.items())[:5]:
                    if not key.startswith("_") and value:
                        lines.append(f"   - {key}: {value}")
            lines.append("")

        # Show applicable standards
        standards = item.get("applicable_standards", [])
        if standards:
            codes = [s.get("code", s) if isinstance(s, dict) else s for s in standards[:5]]
            lines.append(f"   **Applicable Standards:** {', '.join(codes)}")
            if len(standards) > 5:
                lines.append(f"   ... and {len(standards) - 5} more")
            lines.append("")

        # Show certifications
        certs = item.get("certifications", [])
        if certs:
            lines.append(f"   **Certifications:** {', '.join(certs[:5])}")
            lines.append("")

        # Show guidelines
        guidelines = item.get("guidelines", [])
        if guidelines:
            lines.append(f"   **Guidelines:** {len(guidelines)} items available")
            for j, guideline in enumerate(guidelines[:3], 1):
                guideline_str = str(guideline)[:80]
                lines.append(f"   {j}. {guideline_str}...")
            if len(guidelines) > 3:
                lines.append(f"   ... and {len(guidelines) - 3} more")
            lines.append("")

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_schema_population_stats(enriched_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about schema field population.

    Args:
        enriched_items: Items enriched with Deep Agent specifications

    Returns:
        Dictionary with population statistics
    """
    total_items = len(enriched_items)
    schemas_populated = sum(1 for item in enriched_items if item.get("schema_populated", False))
    total_fields_populated = 0
    total_standards = 0
    total_certifications = 0

    for item in enriched_items:
        schema = item.get("schema", {})
        population_info = schema.get("_deep_agent_population", {})
        total_fields_populated += population_info.get("fields_populated", 0)
        total_standards += len(item.get("applicable_standards", []))
        total_certifications += len(item.get("certifications", []))

    return {
        "total_items": total_items,
        "schemas_populated": schemas_populated,
        "population_rate": (schemas_populated / total_items * 100) if total_items > 0 else 0,
        "total_fields_populated": total_fields_populated,
        "avg_fields_per_item": total_fields_populated / total_items if total_items > 0 else 0,
        "total_standards_codes": total_standards,
        "total_certifications": total_certifications
    }
