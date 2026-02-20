# search/agents/requirements_agent.py
# =============================================================================
# REQUIREMENTS COLLECTION AGENT (HITL)
# =============================================================================
#
# Handles immediate post-validation HITL interaction.
#
# Two-phase HITL:
# 1. awaitMissingInfo: Prompt user for missing mandatory fields
#    - User can: provide data, skip fields, or "proceed anyway"
# 2. awaitAdditionalSpecs: Ask for extra requirements
#    - LLM parses free-text into structured_requirements
#
# =============================================================================

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RequirementsCollectionResult:
    """Result from the requirements collection agent."""

    hitl_phase: str  # "awaitMissingInfo" | "awaitAdditionalSpecs" | "complete"
    missing_mandatory_fields: List[Dict[str, Any]]  # Detailed field info
    structured_requirements: Dict[str, Any]
    needs_user_input: bool
    sales_agent_response: str  # Formatted prompt for user
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "hitl_phase": self.hitl_phase,
            "missing_mandatory_fields": self.missing_mandatory_fields,
            "structured_requirements": self.structured_requirements,
            "needs_user_input": self.needs_user_input,
            "sales_agent_response": self.sales_agent_response,
            "error": self.error,
        }


class RequirementsCollectionAgent:
    """
    Handles immediate post-validation HITL interaction.

    This agent manages the two-phase HITL flow:
    1. awaitMissingInfo: Collect missing mandatory fields
    2. awaitAdditionalSpecs: Collect extra requirements beyond schema
    """

    def __init__(self):
        """Initialize the RequirementsCollectionAgent."""
        self._llm = None

    def collect_missing_info(
        self,
        schema: Dict[str, Any],
        missing_fields: List[str],
        provided_requirements: Dict[str, Any],
    ) -> RequirementsCollectionResult:
        """
        Phase 1: Identify and prompt for missing mandatory fields.

        Args:
            schema: The product schema with field definitions
            missing_fields: List of missing mandatory field names
            provided_requirements: Requirements already provided

        Returns:
            RequirementsCollectionResult with prompt for user
        """
        logger.info(
            "[RequirementsAgent] Collecting missing info for %d fields",
            len(missing_fields),
        )

        # Build detailed field information for user
        missing_field_details = []
        mandatory_schema = schema.get("mandatory_requirements", {})

        for field_name in missing_fields:
            field_def = mandatory_schema.get(field_name, {})
            missing_field_details.append({
                "field_name": field_name,
                "display_name": self._format_display_name(field_name),
                "field_type": field_def.get("type", "string"),
                "description": field_def.get("description", ""),
                "example": field_def.get("example", ""),
                "options": field_def.get("options", []),
                "required": True,
            })

        # Build user-friendly prompt
        sales_agent_response = self._build_missing_info_prompt(
            missing_field_details,
            provided_requirements,
        )

        return RequirementsCollectionResult(
            hitl_phase="awaitMissingInfo",
            missing_mandatory_fields=missing_field_details,
            structured_requirements=provided_requirements,
            needs_user_input=len(missing_fields) > 0,
            sales_agent_response=sales_agent_response,
        )

    def process_user_response(
        self,
        response_type: str,  # "provide" | "skip" | "proceed_anyway"
        user_values: Optional[Dict[str, Any]] = None,
        skipped_fields: Optional[List[str]] = None,
        current_requirements: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process user's response to missing info prompt.

        Args:
            response_type: Type of response ("provide", "skip", "proceed_anyway")
            user_values: Values provided by user for missing fields
            skipped_fields: Fields user chose to skip
            current_requirements: Current requirements to merge with

        Returns:
            Updated requirements dictionary
        """
        logger.info("[RequirementsAgent] Processing user response: %s", response_type)

        updated_requirements = dict(current_requirements or {})

        if response_type == "provide" and user_values:
            # Merge user-provided values
            for key, value in user_values.items():
                if value is not None and value != "":
                    updated_requirements[key] = value
                    logger.debug("[RequirementsAgent] Added user value: %s", key)

        elif response_type == "skip" and skipped_fields:
            # Mark fields as intentionally skipped
            updated_requirements["_skipped_fields"] = skipped_fields
            logger.info("[RequirementsAgent] User skipped fields: %s", skipped_fields)

        elif response_type == "proceed_anyway":
            # User wants to proceed with incomplete requirements
            updated_requirements["_proceed_incomplete"] = True
            logger.info("[RequirementsAgent] User chose to proceed anyway")

        return updated_requirements

    def collect_additional_specs(
        self,
        current_requirements: Dict[str, Any],
    ) -> RequirementsCollectionResult:
        """
        Phase 2: Ask for extra requirements beyond schema.

        Args:
            current_requirements: Requirements collected so far

        Returns:
            RequirementsCollectionResult with prompt for additional specs
        """
        logger.info("[RequirementsAgent] Collecting additional specifications")

        sales_agent_response = self._build_additional_specs_prompt(current_requirements)

        return RequirementsCollectionResult(
            hitl_phase="awaitAdditionalSpecs",
            missing_mandatory_fields=[],
            structured_requirements=current_requirements,
            needs_user_input=True,
            sales_agent_response=sales_agent_response,
        )

    def parse_additional_specs(
        self,
        raw_text: str,
        current_requirements: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Use LLM to parse free-text requirements into structured format.

        Args:
            raw_text: Free-text requirements from user
            current_requirements: Current requirements

        Returns:
            Parsed requirements dictionary
        """
        logger.info("[RequirementsAgent] Parsing additional specs from free-text")

        if not raw_text or not raw_text.strip():
            return {}

        try:
            # Try to use LLM for parsing
            parsed = self._llm_parse_specs(raw_text, current_requirements)
            if parsed:
                return parsed
        except Exception as exc:
            logger.warning("[RequirementsAgent] LLM parsing failed: %s", exc)

        # Fallback: Simple rule-based parsing
        return self._fallback_parse_specs(raw_text)

    def merge_requirements(
        self,
        provided: Dict[str, Any],
        user_values: Dict[str, Any],
        additional_parsed: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge all requirement sources into final structured_requirements.

        Priority (highest to lowest):
        1. Additional specs (user's latest input)
        2. User values (missing field responses)
        3. Provided (from initial validation)

        Args:
            provided: Requirements from initial validation
            user_values: Values user provided for missing fields
            additional_parsed: Parsed additional specifications

        Returns:
            Merged requirements dictionary
        """
        logger.info("[RequirementsAgent] Merging all requirements sources")

        # Start with provided requirements
        merged = dict(provided)

        # Add user values (overwrite if exists)
        for key, value in user_values.items():
            if not key.startswith("_"):  # Skip internal flags
                if value is not None and value != "":
                    merged[key] = value

        # Add additional parsed specs (overwrite if exists)
        for key, value in additional_parsed.items():
            if not key.startswith("_"):
                if value is not None and value != "":
                    merged[key] = value

        logger.info(
            "[RequirementsAgent] Merged requirements: %d total fields",
            len([k for k in merged.keys() if not k.startswith("_")]),
        )

        return merged

    # =========================================================================
    # PROMPT BUILDING
    # =========================================================================

    def _build_missing_info_prompt(
        self,
        missing_fields: List[Dict[str, Any]],
        provided: Dict[str, Any],
    ) -> str:
        """Build user-friendly prompt for missing information."""
        lines = [
            "I need a few more details to find the best products for you.",
            "",
            "**Missing Information:**",
        ]

        for field in missing_fields:
            line = f"- **{field['display_name']}**"
            if field.get("description"):
                line += f": {field['description']}"
            if field.get("example"):
                line += f" (e.g., {field['example']})"
            lines.append(line)

        lines.extend([
            "",
            "You can:",
            "1. **Provide** the values for these fields",
            "2. **Skip** specific fields if not applicable",
            "3. **Proceed anyway** with the information already provided",
            "",
            "What would you like to do?",
        ])

        return "\n".join(lines)

    def _build_additional_specs_prompt(
        self,
        current_requirements: Dict[str, Any],
    ) -> str:
        """Build prompt for additional specifications."""
        lines = [
            "Great! I have the basic requirements.",
            "",
            "Do you have any **additional or special requirements** that I should consider?",
            "",
            "For example:",
            "- Specific certifications (ATEX, SIL, etc.)",
            "- Environmental conditions",
            "- Vendor preferences",
            "- Budget constraints",
            "- Delivery requirements",
            "",
            "You can type your requirements in plain English, or say **'No, proceed'** to continue.",
        ]

        return "\n".join(lines)

    # =========================================================================
    # PARSING HELPERS
    # =========================================================================

    def _llm_parse_specs(
        self,
        raw_text: str,
        current_requirements: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to parse free-text into structured specs."""
        try:
            from common.services.llm.fallback import create_llm_with_fallback
            from common.config import AgenticConfig
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import JsonOutputParser

            llm = create_llm_with_fallback(AgenticConfig.FLASH_MODEL)

            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a technical requirements parser. Extract structured specifications from user's free-text input.

Current requirements already collected:
{current_requirements}

Output ONLY a JSON object with extracted specifications. Use camelCase keys.
Common fields: measurementRange, accuracy, outputSignal, processConnection, material, certification, temperatureRange, pressureRange, communicationProtocol, ingressProtection, vendorPreference, budgetConstraint.

If no new specifications found, return empty object {{}}.
"""),
                ("human", "{user_input}"),
            ])

            chain = prompt | llm | JsonOutputParser()

            result = chain.invoke({
                "current_requirements": str(current_requirements),
                "user_input": raw_text,
            })

            if isinstance(result, dict):
                logger.info("[RequirementsAgent] LLM parsed %d specs", len(result))
                return result

        except Exception as exc:
            logger.warning("[RequirementsAgent] LLM parsing error: %s", exc)

        return None

    def _fallback_parse_specs(self, raw_text: str) -> Dict[str, Any]:
        """Fallback rule-based parsing of additional specs."""
        import re

        specs = {}
        text_lower = raw_text.lower()

        # Certification patterns
        cert_patterns = [
            (r'\batex\b', "ATEX"),
            (r'\bsil\s*[123]\b', lambda m: m.group(0).upper()),
            (r'\biecex\b', "IECEx"),
            (r'\bfm\b', "FM"),
            (r'\bcsa\b', "CSA"),
        ]

        certifications = []
        for pattern, value in cert_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if callable(value):
                    certifications.append(value(match))
                else:
                    certifications.append(value)

        if certifications:
            specs["certification"] = ", ".join(certifications)

        # Vendor preference
        vendor_match = re.search(r'prefer\s+(\w+)|(\w+)\s+preferred', text_lower)
        if vendor_match:
            vendor = vendor_match.group(1) or vendor_match.group(2)
            specs["vendorPreference"] = vendor.title()

        # Budget pattern
        budget_match = re.search(r'\$?\s*(\d+[,\d]*)\s*(budget|max|limit)', text_lower)
        if budget_match:
            specs["budgetConstraint"] = budget_match.group(1).replace(",", "")

        # IP rating
        ip_match = re.search(r'ip\s*(\d{2})', text_lower)
        if ip_match:
            specs["ingressProtection"] = f"IP{ip_match.group(1)}"

        return specs

    def _format_display_name(self, field_name: str) -> str:
        """Convert camelCase field name to display name."""
        import re

        # Insert spaces before capitals
        name = re.sub(r'([A-Z])', r' \1', field_name)
        # Capitalize first letter
        return name.strip().title()
