# agentic/standards/keywords.py
# =============================================================================
# STANDARDS MODULE - UNIFIED KEYWORDS & DOMAIN MAPPINGS
# =============================================================================
#
# This file consolidates all domain keywords and mappings from:
# - domain_classifier.py (StandardsDomain enum, DOMAIN_KEYWORDS, DOMAIN_DOCUMENT_ROUTING)
# - standards_detector.py (SAFETY_STANDARDS, PROCESS_STANDARDS, DOMAIN_KEYWORDS)
# - standards_deep_agent.py (STANDARD_DOMAINS, STANDARD_FILES)
# - standards_rag_enrichment.py (STANDARDS_KEYWORDS)
#
# =============================================================================

from enum import Enum
from typing import Dict, List

# =============================================================================
# STANDARDS DOMAIN ENUM
# =============================================================================
# Source: domain_classifier.py:26-42, with additions from standards_deep_agent.py

class StandardsDomain(Enum):
    """Standards document domains for routing."""
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    FLOW = "flow"
    LEVEL = "level"
    SAFETY = "safety"
    CONTROL = "control"
    ANALYTICAL = "analytical"
    COMMUNICATION = "communication"
    CALIBRATION = "calibration"
    VALVES = "valves"
    MONITORING = "monitoring"
    ACCESSORIES = "accessories"  # From standards_deep_agent.py


# =============================================================================
# UNIFIED DOMAIN KEYWORDS
# =============================================================================
# Merged from: domain_classifier.py:91-155, standards_deep_agent.py:174-234,
#              standards_detector.py:40-96

DOMAIN_KEYWORDS: Dict[StandardsDomain, List[str]] = {
    StandardsDomain.SAFETY: [
        # From standards_detector.py SAFETY_STANDARDS
        "sil", "sil1", "sil2", "sil3", "sil4", "iec 61508", "iec 61511", "isa 84",
        "atex", "iecex", "hazloc", "iec 60079",
        "ul listed", "fm approved", "csa", "ce marking", "rohs", "reach",
        "iso 9001", "iso 14001", "iso 17025", "nace", "nema", "ansi",
        # From domain_classifier.py
        "sis", "esd", "emergency shutdown", "bursting", "rupture disc",
        "zone 1", "zone 2", "zone 0", "functional safety", "safety integrity",
        "intrinsically safe", "explosion proof", "flameproof",
        "increased safety", "ex ia", "ex ib", "ex d", "ex e", "ex n",
        # From standards_deep_agent.py
        "hazardous", "protection", "sif",
    ],
    StandardsDomain.PRESSURE: [
        # From domain_classifier.py
        "pressure", "transmitter", "gauge", "psi", "bar", "kpa", "mpa",
        "differential", "dp", "absolute", "gauge pressure", "relief valve",
        "prv", "pressure sensor", "pressure switch", "manometer",
        "differential pressure", "static pressure", "burst disc",
        # From standards_deep_agent.py
        "pascal",
        # From standards_detector.py PROCESS_STANDARDS
        "api", "api 520", "api 521", "api 526", "api 2000",
        "asme", "asme b31", "asme viii", "pressure rating",
    ],
    StandardsDomain.TEMPERATURE: [
        # From domain_classifier.py
        "temperature", "rtd", "thermocouple", "pt100", "pt1000", "thermowell",
        "celsius", "fahrenheit", "kelvin", "thermal", "temp", "thermometer",
        "temperature sensor", "temperature transmitter", "pyrometer",
        "type k", "type j", "type t", "type e", "type n", "type s", "type r",
        # From standards_deep_agent.py
        "cryogenic", "high temp", "extreme temperature",
    ],
    StandardsDomain.FLOW: [
        # From domain_classifier.py
        "flow", "meter", "coriolis", "magnetic", "ultrasonic", "vortex",
        "turbine", "gpm", "m3/h", "lpm", "volumetric", "mass flow",
        "flowmeter", "flow meter", "flow transmitter", "orifice plate",
        "venturi", "pitot", "thermal mass", "positive displacement",
    ],
    StandardsDomain.LEVEL: [
        # From domain_classifier.py
        "level", "radar", "ultrasonic level", "guided wave", "capacitance",
        "hydrostatic", "tank level", "gwr", "level transmitter", "level sensor",
        "level gauge", "displacer", "float", "magnetostrictive", "laser level",
        # From standards_deep_agent.py
        "tank", "silo", "vessel",
    ],
    StandardsDomain.CONTROL: [
        # From domain_classifier.py
        "control valve", "actuator", "positioner", "pid", "controller",
        "regulator", "modulating", "on-off", "control system", "dcs",
        "plc", "scada", "loop control", "cascade", "feedforward",
        # From standards_deep_agent.py
        "feedback", "loop",
    ],
    StandardsDomain.ANALYTICAL: [
        # From domain_classifier.py
        "analyzer", "ph", "conductivity", "dissolved oxygen", "turbidity",
        "gas analyzer", "moisture", "chromatograph", "toc", "orp",
        "spectroscopy", "colorimeter", "orp sensor", "chlorine analyzer",
        "silica analyzer", "oxygen analyzer", "nox", "sox", "co", "co2",
    ],
    StandardsDomain.COMMUNICATION: [
        # From domain_classifier.py
        "hart", "fieldbus", "profibus", "modbus", "wireless", "foundation",
        "profinet", "ethernet/ip", "isa100", "wirelesshart", "ff",
        "4-20ma", "4-20 ma", "protocol", "communication", "io-link",
        "devicenet", "as-interface", "serial", "rs-485", "rs-232",
        # From standards_deep_agent.py
        "signal",
    ],
    StandardsDomain.CALIBRATION: [
        # From domain_classifier.py
        "calibration", "calibrator", "traceability", "uncertainty",
        "measurement", "accuracy", "precision", "metrology", "nist",
        "test equipment", "verification", "validation", "drift",
        # From standards_deep_agent.py
        "maintenance", "tolerance",
    ],
    StandardsDomain.VALVES: [
        # From domain_classifier.py
        "ball valve", "globe valve", "butterfly", "gate valve", "check valve",
        "solenoid", "isolation", "plug valve", "needle valve", "diaphragm valve",
        "pinch valve", "knife gate", "three-way valve", "valve body",
        "valve trim", "cv", "kvs", "valve sizing",
        # From standards_deep_agent.py
        "pneumatic", "hydraulic",
    ],
    StandardsDomain.MONITORING: [
        # From domain_classifier.py
        "condition monitoring", "vibration", "predictive maintenance",
        "asset monitoring", "machinery health", "bearing", "temperature monitoring",
        "online monitoring", "continuous monitoring",
        # From standards_deep_agent.py
        "diagnostic", "health",
    ],
    StandardsDomain.ACCESSORIES: [
        # From standards_deep_agent.py
        "mounting", "installation", "wiring", "cable", "enclosure",
        "protection", "ip rating", "ip65", "ip66", "ip67", "ip68", "ip69",
    ],
}


# =============================================================================
# DOMAIN TO DOCUMENT ROUTING
# =============================================================================
# Source: domain_classifier.py:49-83, standards_deep_agent.py:255-268

DOMAIN_TO_DOCUMENTS: Dict[StandardsDomain, List[str]] = {
    StandardsDomain.PRESSURE: [
        "instrumentation_pressure_standards.docx"
    ],
    StandardsDomain.TEMPERATURE: [
        "instrumentation_temperature_standards.docx"
    ],
    StandardsDomain.FLOW: [
        "instrumentation_flow_standards.docx"
    ],
    StandardsDomain.LEVEL: [
        "instrumentation_level_standards.docx"
    ],
    StandardsDomain.SAFETY: [
        "instrumentation_safety_standards.docx"
    ],
    StandardsDomain.CONTROL: [
        "instrumentation_control_systems_standards.docx"
    ],
    StandardsDomain.ANALYTICAL: [
        "instrumentation_analytical_standards.docx"
    ],
    StandardsDomain.COMMUNICATION: [
        "instrumentation_comm_signal_standards.docx"
    ],
    StandardsDomain.CALIBRATION: [
        "instrumentation_calibration_maintenance_standards.docx"
    ],
    StandardsDomain.VALVES: [
        "instrumentation_valves_actuators_standards.docx"
    ],
    StandardsDomain.MONITORING: [
        "instrumentation_condition_monitoring_standards.docx"
    ],
    StandardsDomain.ACCESSORIES: [
        "instrumentation_accessories_calibration_standards.docx"
    ],
}

# String-keyed version for backwards compatibility with standards_deep_agent.py
STANDARD_DOCUMENT_FILES: Dict[str, str] = {
    "safety": "instrumentation_safety_standards.docx",
    "pressure": "instrumentation_pressure_standards.docx",
    "temperature": "instrumentation_temperature_standards.docx",
    "flow": "instrumentation_flow_standards.docx",
    "level": "instrumentation_level_standards.docx",
    "control": "instrumentation_control_systems_standards.docx",
    "valves": "instrumentation_valves_actuators_standards.docx",
    "calibration": "instrumentation_calibration_maintenance_standards.docx",
    "communication": "instrumentation_comm_signal_standards.docx",
    "condition_monitoring": "instrumentation_condition_monitoring_standards.docx",
    "analytical": "instrumentation_analytical_standards.docx",
    "accessories": "instrumentation_accessories_calibration_standards.docx",
}


# =============================================================================
# PRODUCT TYPE TO DOMAIN MAPPING
# =============================================================================
# Source: domain_classifier.py:163-224

PRODUCT_TYPE_TO_DOMAINS: Dict[str, List[StandardsDomain]] = {
    # Pressure instruments
    "pressure transmitter": [StandardsDomain.PRESSURE],
    "differential pressure transmitter": [StandardsDomain.PRESSURE],
    "dp transmitter": [StandardsDomain.PRESSURE],
    "pressure gauge": [StandardsDomain.PRESSURE],
    "pressure switch": [StandardsDomain.PRESSURE],
    "pressure sensor": [StandardsDomain.PRESSURE],

    # Temperature instruments
    "temperature transmitter": [StandardsDomain.TEMPERATURE],
    "temperature sensor": [StandardsDomain.TEMPERATURE],
    "rtd": [StandardsDomain.TEMPERATURE],
    "thermocouple": [StandardsDomain.TEMPERATURE],
    "thermowell": [StandardsDomain.TEMPERATURE],
    "temperature indicator": [StandardsDomain.TEMPERATURE],

    # Flow instruments
    "flow meter": [StandardsDomain.FLOW],
    "flowmeter": [StandardsDomain.FLOW],
    "coriolis meter": [StandardsDomain.FLOW],
    "magnetic flow meter": [StandardsDomain.FLOW],
    "mag meter": [StandardsDomain.FLOW],
    "vortex flow meter": [StandardsDomain.FLOW],
    "ultrasonic flow meter": [StandardsDomain.FLOW],
    "turbine flow meter": [StandardsDomain.FLOW],
    "flow transmitter": [StandardsDomain.FLOW],

    # Level instruments
    "level transmitter": [StandardsDomain.LEVEL],
    "level sensor": [StandardsDomain.LEVEL],
    "radar level": [StandardsDomain.LEVEL],
    "guided wave radar": [StandardsDomain.LEVEL],
    "ultrasonic level": [StandardsDomain.LEVEL],
    "level gauge": [StandardsDomain.LEVEL],
    "level switch": [StandardsDomain.LEVEL],

    # Control/Valves (multi-domain)
    "control valve": [StandardsDomain.CONTROL, StandardsDomain.VALVES],
    "valve": [StandardsDomain.VALVES],
    "actuator": [StandardsDomain.CONTROL, StandardsDomain.VALVES],
    "positioner": [StandardsDomain.CONTROL],
    "valve positioner": [StandardsDomain.CONTROL, StandardsDomain.VALVES],
    "isolation valve": [StandardsDomain.VALVES],
    "ball valve": [StandardsDomain.VALVES],
    "globe valve": [StandardsDomain.VALVES],
    "butterfly valve": [StandardsDomain.VALVES],
    "gate valve": [StandardsDomain.VALVES],
    "check valve": [StandardsDomain.VALVES],
    "safety valve": [StandardsDomain.VALVES, StandardsDomain.SAFETY],
    "relief valve": [StandardsDomain.VALVES, StandardsDomain.PRESSURE],

    # Analytical instruments
    "ph sensor": [StandardsDomain.ANALYTICAL],
    "ph analyzer": [StandardsDomain.ANALYTICAL],
    "conductivity sensor": [StandardsDomain.ANALYTICAL],
    "conductivity meter": [StandardsDomain.ANALYTICAL],
    "dissolved oxygen sensor": [StandardsDomain.ANALYTICAL],
    "turbidity meter": [StandardsDomain.ANALYTICAL],
    "gas analyzer": [StandardsDomain.ANALYTICAL],
    "analyzer": [StandardsDomain.ANALYTICAL],
}


# =============================================================================
# STANDARD DOMAIN DEFINITIONS (Extended Info)
# =============================================================================
# Source: standards_deep_agent.py:174-234

STANDARD_DOMAINS: Dict[str, Dict[str, any]] = {
    "safety": {
        "name": "Safety & Protection Standards",
        "keywords": DOMAIN_KEYWORDS[StandardsDomain.SAFETY],
        "description": "SIL ratings, ATEX zones, hazardous area classifications, safety instrumented systems"
    },
    "pressure": {
        "name": "Pressure Measurement Standards",
        "keywords": DOMAIN_KEYWORDS[StandardsDomain.PRESSURE],
        "description": "Pressure measurement devices, transmitters, gauges, calibration requirements"
    },
    "temperature": {
        "name": "Temperature Measurement Standards",
        "keywords": DOMAIN_KEYWORDS[StandardsDomain.TEMPERATURE],
        "description": "Temperature sensors, thermocouples, RTDs, calibration standards"
    },
    "flow": {
        "name": "Flow Measurement Standards",
        "keywords": DOMAIN_KEYWORDS[StandardsDomain.FLOW],
        "description": "Flow measurement devices, mass flow, volumetric flow standards"
    },
    "level": {
        "name": "Level Measurement Standards",
        "keywords": DOMAIN_KEYWORDS[StandardsDomain.LEVEL],
        "description": "Level measurement for tanks, vessels, silos"
    },
    "control": {
        "name": "Control Systems Standards",
        "keywords": DOMAIN_KEYWORDS[StandardsDomain.CONTROL],
        "description": "Process control systems, DCS, PLC, control loops"
    },
    "valves": {
        "name": "Valves & Actuators Standards",
        "keywords": DOMAIN_KEYWORDS[StandardsDomain.VALVES],
        "description": "Control valves, actuators, positioners, valve specifications"
    },
    "calibration": {
        "name": "Calibration & Maintenance Standards",
        "keywords": DOMAIN_KEYWORDS[StandardsDomain.CALIBRATION],
        "description": "Calibration procedures, maintenance requirements, traceability"
    },
    "communication": {
        "name": "Communication & Signals Standards",
        "keywords": DOMAIN_KEYWORDS[StandardsDomain.COMMUNICATION],
        "description": "Industrial communication protocols, signal types"
    },
    "condition_monitoring": {
        "name": "Condition Monitoring Standards",
        "keywords": DOMAIN_KEYWORDS[StandardsDomain.MONITORING],
        "description": "Equipment condition monitoring, predictive maintenance"
    },
    "analytical": {
        "name": "Analytical Instrumentation Standards",
        "keywords": DOMAIN_KEYWORDS[StandardsDomain.ANALYTICAL],
        "description": "Process analyzers, gas detection, chemical analysis"
    },
    "accessories": {
        "name": "Accessories & Installation Standards",
        "keywords": DOMAIN_KEYWORDS[StandardsDomain.ACCESSORIES],
        "description": "Installation requirements, accessories, enclosures"
    },
}


# =============================================================================
# SAFETY STANDARDS (From standards_detector.py)
# =============================================================================

SAFETY_STANDARD_KEYWORDS: Dict[str, List[str]] = {
    "sil": ["sil", "sil1", "sil2", "sil3", "sil4", "iec 61508", "iec 61511", "isa 84"],
    "atex": ["atex", "iecex", "hazloc", "iec 60079"],
    "certification": ["ul listed", "fm approved", "csa", "ce marking", "rohs", "reach"],
    "other_standards": ["iso 9001", "iso 14001", "iso 17025", "nace", "nema", "ansi", "iec"],
}

PROCESS_STANDARD_KEYWORDS: Dict[str, List[str]] = {
    "api": ["api", "api 520", "api 521", "api 526", "api 2000"],
    "asme": ["asme", "asme b31", "asme viii"],
    "pressure": ["psi", "bar", "mpa", "pressure rating"],
}

DOMAIN_INDICATOR_KEYWORDS: Dict[str, List[str]] = {
    "hazardous": [
        "hazardous", "hazardous area", "hazardous zone", "explosive atmosphere",
        "explosion-proof", "explosionproof", "atex zone 0", "atex zone 1", "atex zone 2",
        "combustible dust", "flammable gas", "flammable liquid",
    ],
    "oil_gas": [
        "oil", "gas", "crude oil", "petroleum", "refinery", "distillation",
        "pipeline", "upstream", "downstream", "midstream", "wellhead",
    ],
    "pharma": [
        "pharma", "pharmaceutical", "gmp", "cleanroom", "sterile", "biopharm",
        "vaccine", "drug manufacturing",
    ],
    "food": [
        "food", "beverage", "dairy", "meat", "sanitary", "washdown",
    ],
    "chemical": [
        "chemical", "petrochemical", "corrosive", "caustic", "acid",
    ],
}

REQUIREMENT_KEYWORDS: Dict[str, List[str]] = {
    "sil_level": ["sil level", "sil rating", "sil certification"],
    "hazard": ["hazardous", "explosive", "flammable"],
    "pressure": ["pressure", "psi", "bar", "mpa"],
    "temperature": ["temperature", "cryogenic", "high temp"],
}

IP_RATING_PATTERNS: List[str] = [
    "ip rating", "ip65", "ip66", "ip67", "ip68", "ip69",
]

CRITICAL_SPEC_KEYWORDS: List[str] = [
    "cryogenic", "deep cryogenic",
    "high temperature", "extreme temperature",
    "high pressure", "extreme pressure",
    "vacuum", "low pressure",
]


# =============================================================================
# FIELD GROUPS FOR PARALLEL ENRICHMENT
# =============================================================================
# Source: parallel_standards_enrichment.py:38-59

FIELD_GROUPS: Dict[str, List[str]] = {
    "process_parameters": [
        "process_temperature", "ambient_temperature", "process_pressure",
        "flow_rate", "media", "viscosity", "density",
    ],
    "performance": [
        "accuracy", "repeatability", "stability", "turndown_ratio",
        "response_time", "rangeability",
    ],
    "electrical": [
        "power_supply", "output_signal", "communication_protocol",
        "cable_entry", "hazardous_area_certification",
    ],
    "mechanical": [
        "process_connection", "housing_material", "wetted_parts",
        "ip_rating", "weight", "dimensions",
    ],
    "compliance": [
        "certifications", "standards", "sil_rating", "atex_zone",
        "material_compliance",
    ],
}


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enum
    "StandardsDomain",
    # Main keyword mappings
    "DOMAIN_KEYWORDS",
    "DOMAIN_TO_DOCUMENTS",
    "STANDARD_DOCUMENT_FILES",
    "PRODUCT_TYPE_TO_DOMAINS",
    "STANDARD_DOMAINS",
    # Detection keywords
    "SAFETY_STANDARD_KEYWORDS",
    "PROCESS_STANDARD_KEYWORDS",
    "DOMAIN_INDICATOR_KEYWORDS",
    "REQUIREMENT_KEYWORDS",
    "IP_RATING_PATTERNS",
    "CRITICAL_SPEC_KEYWORDS",
    # Field groups
    "FIELD_GROUPS",
]
