"""
Domain Validator for Intent Classification

Validates that queries are within instrumentation scope and that target
workflows can handle the classified intent.

This prevents:
1. Out-of-domain queries from being misclassified as CHAT
2. Requests outside workflow capabilities from wasting processing time
3. Adjacent industrial domains (PLC, electrical, mechanical) from slipping through
"""

import logging
from typing import Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class WorkflowTarget(Enum):
    """Target workflow types (import from intent_classifier to avoid circular import)."""
    ENGENIE_CHAT = "engenie_chat"
    INSTRUMENT_IDENTIFIER = "instrument_identifier"
    SOLUTION_WORKFLOW = "solution"
    OUT_OF_DOMAIN = "out_of_domain"


class DomainValidator:
    """
    Validates domain scope and workflow capabilities for intent classification.

    Usage:
        validator = DomainValidator()
        is_valid, reason = validator.is_in_domain("What is a pressure transmitter?")
        can_handle, reason = validator.validate_workflow_capability(
            query="I need a transmitter",
            target_workflow=WorkflowTarget.INSTRUMENT_IDENTIFIER
        )
    """

    # In-scope keywords (instrumentation domain)
    IN_SCOPE_KEYWORDS = {
        # === INSTRUMENTS (Core Product Types) ===
        "transmitter", "sensor", "valve", "actuator", "controller",
        "flowmeter", "flow meter", "analyzer", "gauge", "switch",
        "positioner", "indicator", "recorder", "thermocouple", "rtd",
        "level transmitter", "level sensor", "level gauge",
        "pressure transmitter", "pressure sensor", "pressure gauge",
        "temperature sensor", "temperature transmitter",
        "control valve", "safety valve", "relief valve",
        "differential pressure", "dp transmitter",

        # === ACCESSORIES ===
        "thermowell", "manifold", "cable", "junction box",
        "mounting bracket", "process connection", "isolation valve",
        "instrument cable", "gland", "seal",

        # === STANDARDS & CERTIFICATIONS ===
        "iec", "iso", "api", "atex", "sil", "iecex", "nace", "asme",
        "iec 61508", "iec 61511", "api 526", "api 520",
        "sil 2", "sil 3", "sil rated", "safety integrity level",
        "atex zone", "hazardous area", "explosion proof",

        # === VENDORS (Major) ===
        "rosemount", "yokogawa", "emerson", "honeywell", "endress",
        "fisher", "abb", "siemens", "foxboro", "krohne",
        "vega", "magnetrol", "wika", "omega",

        # === INSTRUMENTATION CONCEPTS ===
        "measurement", "calibration", "certification", "datasheet",
        "specification", "accuracy", "range", "output signal",
        "4-20ma", "hart", "modbus", "profibus", "fieldbus",
        "process variable", "primary element", "impulse line",

        # === SYSTEM DESIGN (Valid for SOLUTION) ===
        "instrumentation package", "measurement system",
        "control system", "monitoring system", "safety system",
    }

    # Out-of-scope keywords (adjacent domains that are NOT instrumentation)
    OUT_OF_SCOPE_KEYWORDS = {
        # === PLC/SCADA PROGRAMMING ===
        "plc programming", "ladder logic", "function block", "structured text",
        "hmi design", "scada programming", "control logic",
        "tag database", "opc server", "historian",

        # === ELECTRICAL ENGINEERING (Non-instrument) ===
        "motor starter", "vfd", "variable frequency drive", "soft starter",
        "generator", "transformer", "switchgear", "mcc",
        "power distribution", "electrical panel", "breaker sizing",

        # === MECHANICAL SYSTEMS (Non-instrument) ===
        "pump sizing", "compressor selection", "turbine design",
        "heat exchanger design", "piping design", "pipe stress",
        "hydraulic calculation", "npsh calculation",

        # === PROCESS ENGINEERING ===
        "reaction kinetics", "heat transfer coefficient",
        "mass transfer", "fluid dynamics", "cfd",
        "distillation column design", "reactor design",

        # === NON-INDUSTRIAL ===
        "weather", "sports", "entertainment", "politics", "food",
        "cooking", "recipe", "travel", "tourism",
        "medical", "legal", "financial", "investment",

        # === COMMERCIAL (Not technical) ===
        "price", "pricing", "cost", "quote", "quotation",
        "purchase order", "availability", "lead time", "shipping",
        "sales rep", "distributor", "warranty", "return policy",

        # === OPERATIONS (Not selection/design) ===
        "troubleshoot", "troubleshooting", "error code",
        "maintenance", "repair", "installation", "commissioning",
        "calibration procedure", "loop check",
    }

    # Strong reject keywords (immediate OUT_OF_DOMAIN)
    STRONG_REJECT_KEYWORDS = {
        "weather", "joke", "funny", "story", "poem",
        "recipe", "cooking", "food", "restaurant",
        "movie", "music", "celebrity", "sports", "game",
        "politics", "election", "president", "government",
    }

    @classmethod
    def is_in_domain(cls, query: str) -> Tuple[bool, str]:
        """
        Check if query is within instrumentation domain.

        Returns:
            (is_valid, reason)

        Examples:
            >>> is_in_domain("What is a pressure transmitter?")
            (True, "In-domain keyword detected: 'pressure transmitter'")

            >>> is_in_domain("How to program a PLC?")
            (False, "Out-of-domain keyword detected: 'plc programming'")

            >>> is_in_domain("What's the weather?")
            (False, "Strong reject keyword detected: 'weather'")
        """
        query_lower = query.lower()

        # Check for strong reject keywords first (immediate rejection)
        for keyword in cls.STRONG_REJECT_KEYWORDS:
            if keyword in query_lower:
                logger.info(f"[DomainValidator] Strong reject: '{keyword}' in query")
                return False, f"Non-industrial query detected: '{keyword}'"

        # Check for out-of-scope keywords (adjacent domains)
        for keyword in cls.OUT_OF_SCOPE_KEYWORDS:
            if keyword in query_lower:
                logger.info(f"[DomainValidator] Out-of-scope: '{keyword}' in query")
                return False, f"Out-of-scope keyword detected: '{keyword}'"

        # Check for in-scope keywords (instrumentation)
        matched_keywords = []
        for keyword in cls.IN_SCOPE_KEYWORDS:
            if keyword in query_lower:
                matched_keywords.append(keyword)

        if matched_keywords:
            logger.debug(f"[DomainValidator] In-scope keywords: {matched_keywords[:3]}")
            return True, f"In-domain keyword detected: '{matched_keywords[0]}'"

        # Ambiguous - no clear indicators
        # Default to accepting (will be classified by semantic/LLM)
        # This handles variations and novel phrasings
        logger.debug(f"[DomainValidator] No clear indicators, defaulting to accept")
        return True, "No clear domain indicators (defaulting to accept for further classification)"

    @classmethod
    def validate_workflow_capability(
        cls,
        query: str,
        target_workflow: str  # Use string to avoid circular import
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that target workflow can handle the query.

        Args:
            query: User query string
            target_workflow: Target workflow name (e.g., "engenie_chat", "instrument_identifier", "solution")

        Returns:
            (can_handle, reason_if_not)

        Examples:
            >>> validate_workflow_capability("What is a transmitter?", "engenie_chat")
            (True, None)

            >>> validate_workflow_capability("I need a transmitter", "engenie_chat")
            (False, "Purchase intent detected, should route to SEARCH")

            >>> validate_workflow_capability("What is a transmitter?", "solution")
            (False, "Knowledge question detected, should route to CHAT")
        """
        query_lower = query.lower()

        # Convert string to enum-like comparison
        if target_workflow in ["engenie_chat", "ENGENIE_CHAT"]:
            # CHAT cannot handle purchase/design intents
            purchase_keywords = ["i need", "looking for", "find me", "recommend", "suggest"]
            design_keywords = ["design a", "design an", "implement a", "build a", "create a"]

            # Check for purchase intent
            has_purchase = any(kw in query_lower for kw in purchase_keywords)
            has_instrument = any(inst in query_lower for inst in [
                "transmitter", "sensor", "valve", "actuator", "flowmeter", "analyzer"
            ])

            if has_purchase and has_instrument:
                logger.info(f"[DomainValidator] CHAT cannot handle purchase intent")
                return False, "Purchase intent detected, should route to SEARCH"

            # Check for design intent
            if any(kw in query_lower for kw in design_keywords):
                system_keywords = ["system", "package", "instrumentation"]
                if any(sk in query_lower for sk in system_keywords):
                    logger.info(f"[DomainValidator] CHAT cannot handle design intent")
                    return False, "Design intent detected, should route to SOLUTION"

        elif target_workflow in ["instrument_identifier", "INSTRUMENT_IDENTIFIER"]:
            # SEARCH cannot handle knowledge questions
            knowledge_keywords = ["what is", "how does", "explain", "tell me about", "describe"]

            if any(kw in query_lower for kw in knowledge_keywords):
                logger.info(f"[DomainValidator] SEARCH cannot handle knowledge question")
                return False, "Knowledge question detected, should route to CHAT"

            # SEARCH cannot handle multi-instrument systems
            system_keywords = ["complete system", "monitoring system", "control system",
                             "instrumentation package", "measurement system"]

            if any(kw in query_lower for kw in system_keywords):
                logger.info(f"[DomainValidator] SEARCH cannot handle system design")
                return False, "System design detected, should route to SOLUTION"

        elif target_workflow in ["solution", "SOLUTION_WORKFLOW"]:
            # SOLUTION cannot handle knowledge questions (unless "what is needed...")
            knowledge_keywords = ["what is", "how does", "explain", "tell me about", "describe"]

            # Exception: "what is needed" or "what do i need" are design questions
            design_exceptions = ["what is needed", "what do i need", "what instruments"]

            has_knowledge = any(kw in query_lower for kw in knowledge_keywords)
            has_exception = any(ex in query_lower for ex in design_exceptions)

            if has_knowledge and not has_exception:
                logger.info(f"[DomainValidator] SOLUTION cannot handle knowledge question")
                return False, "Knowledge question detected, should route to CHAT"

        # No capability issues detected
        logger.debug(f"[DomainValidator] Workflow '{target_workflow}' can handle query")
        return True, None

    @classmethod
    def get_reject_message(cls, reason: str) -> str:
        """
        Get user-friendly reject message based on validation reason.

        Args:
            reason: Rejection reason from is_in_domain()

        Returns:
            User-friendly message explaining why query was rejected
        """
        # Map reasons to user-friendly messages
        if "plc" in reason.lower() or "scada" in reason.lower():
            return (
                "I specialize in instrumentation selection and system design, not PLC/SCADA programming. "
                "For control logic and programming, please consult your automation engineering team."
            )

        if "price" in reason.lower() or "cost" in reason.lower() or "pricing" in reason.lower():
            return (
                "I don't have access to pricing information. "
                "For pricing and quotes, please contact your procurement team or vendor sales representative."
            )

        if "troubleshoot" in reason.lower() or "error" in reason.lower():
            return (
                "I specialize in instrument selection and system design, not troubleshooting. "
                "For operational issues, please consult your maintenance team or vendor technical support."
            )

        if any(word in reason.lower() for word in ["motor", "pump", "compressor", "turbine"]):
            return (
                "I specialize in instrumentation and control devices. "
                "For mechanical equipment selection, please consult your mechanical engineering team."
            )

        # Default reject message
        return (
            "I specialize in industrial instrumentation and process measurement. I can help with:\n"
            "• Instrument specifications and selection (transmitters, sensors, valves, actuators)\n"
            "• Standards and compliance (IEC, ISO, API, ATEX, SIL)\n"
            "• Vendor information and approved suppliers\n"
            "• System design for instrumentation packages\n\n"
            f"Your query appears to be outside this scope. {reason}"
        )


# Singleton instance for easy access
_domain_validator = DomainValidator()

def get_domain_validator() -> DomainValidator:
    """Get the global domain validator instance."""
    return _domain_validator


# Convenience functions for backward compatibility
def is_in_domain(query: str) -> Tuple[bool, str]:
    """Check if query is in instrumentation domain."""
    return _domain_validator.is_in_domain(query)

def validate_workflow_capability(query: str, target_workflow: str) -> Tuple[bool, Optional[str]]:
    """Validate workflow can handle query."""
    return _domain_validator.validate_workflow_capability(query, target_workflow)

def get_reject_message(reason: str) -> str:
    """Get user-friendly reject message."""
    return _domain_validator.get_reject_message(reason)
