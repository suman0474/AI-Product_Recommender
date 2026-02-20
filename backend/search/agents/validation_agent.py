# search/agents/validation_agent.py
# =============================================================================
# VALIDATION AGENT
# =============================================================================
#
# Handles product type extraction, schema loading, and validation.
# Encapsulates all validation, schema loading, and enrichment logic.
#
# Flow:
# 1. Extract product type using intent tools
# 2. Load schema from database or trigger PPI generation
# 3. Apply standards population and enrichment
# 4. Validate requirements against schema
#
# =============================================================================

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from the validation agent."""

    # Product Type
    product_type: str
    original_product_type: str
    product_type_refined: bool

    # Schema
    schema: Dict[str, Any]
    schema_source: str  # "database" | "ppi" | "default" | "error"

    # Validation
    is_valid: bool
    missing_fields: List[str]
    optional_fields: List[str]
    provided_requirements: Dict[str, Any]

    # Standards
    standards_applied: bool = False
    standards_info: Optional[Dict[str, Any]] = None
    enrichment_result: Optional[Dict[str, Any]] = None
    rag_invocations: Dict[str, Any] = field(default_factory=dict)

    # Errors
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "product_type": self.product_type,
            "original_product_type": self.original_product_type,
            "product_type_refined": self.product_type_refined,
            "schema": self.schema,
            "schema_source": self.schema_source,
            "is_valid": self.is_valid,
            "missing_fields": self.missing_fields,
            "optional_fields": self.optional_fields,
            "provided_requirements": self.provided_requirements,
            "standards_applied": self.standards_applied,
            "standards_info": self.standards_info,
            "enrichment_result": self.enrichment_result,
            "rag_invocations": self.rag_invocations,
            "error": self.error,
        }

    def to_cache_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for caching (minimal)."""
        return {
            "product_type": self.product_type,
            "schema": self.schema,
            "schema_source": self.schema_source,
            "standards_info": self.standards_info,
            "standards_applied": self.standards_applied,
        }


class ValidationAgent:
    """
    Handles product type extraction, schema loading, and validation.

    This agent encapsulates the validation phase logic and can be
    tested independently of the LangGraph workflow.
    """

    def __init__(self):
        """Initialize the ValidationAgent."""
        self._schema_service = None
        self._intent_tools = None

    def validate(
        self,
        user_input: str,
        expected_product_type: Optional[str] = None,
        enable_ppi: bool = True,
        standards_depth: str = "shallow",
    ) -> ValidationResult:
        """
        Run the full validation pipeline.

        Args:
            user_input: Raw user requirement description
            expected_product_type: Hint for product type detection
            enable_ppi: Allow PPI schema generation
            standards_depth: "none" | "shallow" | "deep"

        Returns:
            ValidationResult with all validation data
        """
        logger.info("[ValidationAgent] Starting validation pipeline")

        try:
            # 1. Extract product type
            product_type, original_type, refined = self._extract_product_type(
                user_input, expected_product_type
            )

            if not product_type:
                return ValidationResult(
                    product_type="",
                    original_product_type="",
                    product_type_refined=False,
                    schema={},
                    schema_source="error",
                    is_valid=False,
                    missing_fields=[],
                    optional_fields=[],
                    provided_requirements={},
                    error="Could not extract product type from input",
                )

            # 2. Load schema
            schema, source = self._load_schema(product_type, enable_ppi)

            # 3. Apply standards enrichment
            standards_info = None
            enrichment_result = None
            rag_invocations = {}

            if standards_depth != "none" and schema:
                schema, standards_info, enrichment_result, rag_invocations = (
                    self._apply_standards_enrichment(product_type, schema, standards_depth)
                )

            # 4. Validate requirements
            is_valid, missing, optional, provided = self._validate_requirements(
                user_input, product_type, schema
            )

            return ValidationResult(
                product_type=product_type,
                original_product_type=original_type,
                product_type_refined=refined,
                schema=schema,
                schema_source=source,
                is_valid=is_valid,
                missing_fields=missing,
                optional_fields=optional,
                provided_requirements=provided,
                standards_applied=standards_info is not None,
                standards_info=standards_info,
                enrichment_result=enrichment_result,
                rag_invocations=rag_invocations,
            )

        except Exception as exc:
            logger.error("[ValidationAgent] Validation failed: %s", exc, exc_info=True)
            return ValidationResult(
                product_type=expected_product_type or "",
                original_product_type=expected_product_type or "",
                product_type_refined=False,
                schema={},
                schema_source="error",
                is_valid=False,
                missing_fields=[],
                optional_fields=[],
                provided_requirements={},
                error=str(exc),
            )

    def _extract_product_type(
        self,
        user_input: str,
        hint: Optional[str],
    ) -> Tuple[str, str, bool]:
        """
        Extract product type from user input.

        Args:
            user_input: Raw user input
            hint: Expected product type hint

        Returns:
            Tuple of (product_type, original_product_type, was_refined)
        """
        logger.debug("[ValidationAgent] Extracting product type")

        # Use hint if provided
        if hint:
            logger.debug("[ValidationAgent] Using product type hint: %s", hint)
            return (hint, hint, False)

        try:
            # Try to import and use the intent tools
            from common.tools.intent_tools import extract_requirements_tool

            result = extract_requirements_tool(user_input)

            if result and result.get("success"):
                product_type = result.get("product_type", "")
                original = product_type
                refined = result.get("refined", False)

                logger.info(
                    "[ValidationAgent] Extracted product type: %s (refined=%s)",
                    product_type,
                    refined,
                )
                return (product_type, original, refined)

        except ImportError:
            logger.warning("[ValidationAgent] Intent tools not available, using fallback")
        except Exception as exc:
            logger.warning("[ValidationAgent] Intent extraction failed: %s", exc)

        # Fallback: Simple keyword extraction
        product_type = self._fallback_extract_product_type(user_input)
        return (product_type, product_type, False)

    def _fallback_extract_product_type(self, user_input: str) -> str:
        """Fallback product type extraction using keywords."""
        input_lower = user_input.lower()

        # Common product type keywords
        keywords = [
            "pressure transmitter", "temperature transmitter", "level transmitter",
            "flow transmitter", "dp transmitter", "differential pressure transmitter",
            "flowmeter", "flow meter", "coriolis", "magnetic flowmeter",
            "control valve", "globe valve", "ball valve",
            "analyzer", "gas analyzer", "ph analyzer",
            "sensor", "thermocouple", "rtd",
            "transmitter",  # Generic fallback
        ]

        for keyword in keywords:
            if keyword in input_lower:
                return keyword.title()

        return ""

    def _load_schema(
        self,
        product_type: str,
        enable_ppi: bool,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Load schema from database or generate via PPI.

        Args:
            product_type: The product type to load schema for
            enable_ppi: Whether to allow PPI generation

        Returns:
            Tuple of (schema, source)
        """
        logger.debug("[ValidationAgent] Loading schema for: %s", product_type)

        try:
            from common.services.schema_service import schema_service

            # Try to get schema from database
            schema = schema_service.get_schema(product_type)

            if schema:
                logger.info("[ValidationAgent] Schema loaded from database")
                return (schema, "database")

            # If no schema and PPI enabled, generate
            if enable_ppi:
                logger.info("[ValidationAgent] No schema found, triggering PPI generation")
                # Note: PPI generation would be handled here
                # For now, return empty schema
                return ({}, "ppi_pending")

            return ({}, "not_found")

        except ImportError:
            logger.warning("[ValidationAgent] Schema service not available")
            return ({}, "error")
        except Exception as exc:
            logger.error("[ValidationAgent] Schema loading failed: %s", exc)
            return ({}, "error")

    def _apply_standards_enrichment(
        self,
        product_type: str,
        schema: Dict[str, Any],
        depth: str,
    ) -> Tuple[Dict[str, Any], Optional[Dict], Optional[Dict], Dict]:
        """
        Apply standards population and enrichment.

        Args:
            product_type: The product type
            schema: Current schema
            depth: "shallow" or "deep"

        Returns:
            Tuple of (enriched_schema, standards_info, enrichment_result, rag_invocations)
        """
        logger.debug("[ValidationAgent] Applying standards enrichment (depth=%s)", depth)

        standards_info = None
        enrichment_result = None
        rag_invocations = {}

        try:
            # Get applicable standards
            from common.tools.standards_enrichment_tool import get_applicable_standards

            standards_result = get_applicable_standards(product_type, top_k=5)

            if standards_result and standards_result.get("success"):
                standards_info = standards_result
                rag_invocations["standards_rag"] = {
                    "invoked": True,
                    "success": True,
                }

                # Apply standards to schema if deep enrichment
                if depth == "deep":
                    from common.tools.standards_enrichment_tool import (
                        populate_schema_fields_from_standards,
                    )

                    enrichment = populate_schema_fields_from_standards(product_type, schema)
                    if enrichment:
                        schema = enrichment.get("schema", schema)
                        enrichment_result = enrichment

                logger.info("[ValidationAgent] Standards enrichment applied")

        except ImportError:
            logger.warning("[ValidationAgent] Standards tools not available")
            rag_invocations["standards_rag"] = {"invoked": False, "reason": "import_error"}
        except Exception as exc:
            logger.warning("[ValidationAgent] Standards enrichment failed: %s", exc)
            rag_invocations["standards_rag"] = {"invoked": True, "success": False, "error": str(exc)}

        return (schema, standards_info, enrichment_result, rag_invocations)

    def _validate_requirements(
        self,
        user_input: str,
        product_type: str,
        schema: Dict[str, Any],
    ) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
        """
        Validate user requirements against schema.

        Args:
            user_input: Raw user input
            product_type: The product type
            schema: The schema to validate against

        Returns:
            Tuple of (is_valid, missing_fields, optional_fields, provided_requirements)
        """
        logger.debug("[ValidationAgent] Validating requirements against schema")

        missing_fields = []
        optional_fields = []
        provided_requirements = {}

        try:
            from common.tools.schema_tools import validate_requirements_tool

            result = validate_requirements_tool(user_input, product_type, schema)

            if result:
                is_valid = result.get("is_valid", False)
                missing_fields = result.get("missing_fields", [])
                optional_fields = result.get("optional_fields", [])
                provided_requirements = result.get("provided_requirements", {})

                logger.info(
                    "[ValidationAgent] Validation result: valid=%s, missing=%d, optional=%d",
                    is_valid,
                    len(missing_fields),
                    len(optional_fields),
                )

                return (is_valid, missing_fields, optional_fields, provided_requirements)

        except ImportError:
            logger.warning("[ValidationAgent] Schema tools not available")
        except Exception as exc:
            logger.warning("[ValidationAgent] Validation failed: %s", exc)

        # Fallback: Extract requirements from input
        provided_requirements = self._fallback_extract_requirements(user_input, schema)

        # Determine missing fields from schema
        if schema:
            mandatory = schema.get("mandatory_requirements", {})
            optional_schema = schema.get("optional_requirements", {})

            missing_fields = [
                field for field in mandatory.keys()
                if field not in provided_requirements
            ]
            optional_fields = list(optional_schema.keys())

        is_valid = len(missing_fields) == 0

        return (is_valid, missing_fields, optional_fields, provided_requirements)

    def _fallback_extract_requirements(
        self,
        user_input: str,
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fallback requirement extraction from user input."""
        # Simple extraction - look for patterns in input
        requirements = {}

        # This is a basic fallback - the actual tool does more sophisticated extraction
        import re

        # Pressure range pattern
        pressure_match = re.search(r'(\d+)\s*-\s*(\d+)\s*(psi|bar|mbar|kpa)', user_input, re.I)
        if pressure_match:
            requirements["measurementRange"] = f"{pressure_match.group(1)}-{pressure_match.group(2)} {pressure_match.group(3)}"

        # Temperature range pattern
        temp_match = re.search(r'(\d+)\s*-\s*(\d+)\s*(°?[CF]|celsius|fahrenheit)', user_input, re.I)
        if temp_match:
            requirements["temperatureRange"] = f"{temp_match.group(1)}-{temp_match.group(2)} {temp_match.group(3)}"

        # Output signal pattern
        signal_match = re.search(r'(4-20|0-10|1-5)\s*ma', user_input, re.I)
        if signal_match:
            requirements["outputSignal"] = f"{signal_match.group(1)} mA"

        # Protocol pattern
        protocol_match = re.search(r'(hart|profibus|modbus|foundation fieldbus)', user_input, re.I)
        if protocol_match:
            requirements["communicationProtocol"] = protocol_match.group(1).upper()

        return requirements
