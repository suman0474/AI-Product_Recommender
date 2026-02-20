# solution_N/context_manager.py
# =============================================================================
# CONVERSATION MEMORY & PERSONAL CONTEXT MANAGER
# =============================================================================
#
# Manages:
# 1. Rolling Window Conversation Memory - Last N messages with entity extraction
# 2. Personal Context Integration - User preferences, project history
# 3. Active Thread Analysis - Context continuity across turns
# 4. Semantic Entity Extraction - Specs, vendors, domains from conversation
#
# =============================================================================

import logging
import re
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ConversationMessage:
    """A single message in the conversation history."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedEntities:
    """Entities extracted from conversation history."""
    specifications: Dict[str, str]  # spec_name -> value
    vendors: List[str]  # Mentioned vendor names
    domains: List[str]  # Mentioned industries/domains
    instrument_types: List[str]  # Mentioned instrument categories
    process_parameters: Dict[str, str]  # temperature, pressure, etc.
    safety_requirements: Dict[str, str]  # SIL, ATEX, etc.
    constraints: List[str]  # Budget, timeline, etc.


# =============================================================================
# ENTITY EXTRACTION PATTERNS
# =============================================================================

# Specification patterns to extract from text
SPEC_PATTERNS = {
    "temperature_range": r'(-?\d+)\s*(?:to|[-–])\s*(-?\d+)\s*°?[CF]',
    "pressure_range": r'(\d+(?:\.\d+)?)\s*(?:to|[-–])\s*(\d+(?:\.\d+)?)\s*(?:bar|psi|kPa|MPa)',
    "flow_rate": r'(\d+(?:\.\d+)?)\s*(?:to|[-–])?\s*(\d+(?:\.\d+)?)?\s*(?:m3/h|l/min|GPM|gpm)',
    "accuracy": r'[±]\s*(\d+(?:\.\d+)?)\s*%',
    "sil_level": r'SIL\s*([1-4])',
    "atex_zone": r'(?:ATEX|Zone)\s*([012])',
    "ip_rating": r'IP\s*(\d{2})',
    "output_signal": r'(4-20\s*mA|0-10\s*V|HART|Modbus|Profibus|Foundation\s*Fieldbus)',
    "pipe_size": r'DN\s*(\d+)|(\d+)\s*(?:inch|")',
    "material": r'(316L?\s*SS|Hastelloy\s*[A-Z]?\d*|Inconel\s*\d*|Monel|Titanium|PTFE|PFA)',
}

# Vendor name patterns
KNOWN_VENDORS = [
    "Endress+Hauser", "E+H", "Siemens", "ABB", "Emerson", "Rosemount",
    "Yokogawa", "Honeywell", "Krohne", "Vega", "WIKA", "Fluke",
    "Pepperl+Fuchs", "Phoenix Contact", "Turck", "IFM", "Danfoss",
    "Fisher", "Masoneilan", "Samson", "Metso", "Flowserve",
]

# Instrument type keywords
INSTRUMENT_TYPES = [
    "pressure transmitter", "temperature transmitter", "flow meter",
    "level transmitter", "control valve", "thermocouple", "RTD",
    "pressure gauge", "temperature sensor", "flow sensor",
    "level sensor", "safety valve", "positioner", "analyzer",
    "pH meter", "conductivity meter", "turbidity meter",
    "Coriolis", "magnetic flow", "vortex", "ultrasonic",
    "radar level", "guided wave radar", "differential pressure",
]


# =============================================================================
# CONTEXT MANAGER
# =============================================================================

class SolutionContextManager:
    """
    Manages conversation memory and personal context for the Solution Deep Agent.

    Features:
    - Rolling window of last N messages
    - Entity extraction from conversation history
    - Personal context loading (user preferences, project history)
    - Active thread analysis for context continuity
    - Semantic search using embeddings for relevant context retrieval
    """

    def __init__(self, max_history: int = 20):
        """
        Initialize the context manager.

        Args:
            max_history: Maximum number of messages to keep in rolling window
        """
        self.max_history = max_history
        self._history: deque = deque(maxlen=max_history)
        self._extracted_entities = ExtractedEntities(
            specifications={},
            vendors=[],
            domains=[],
            instrument_types=[],
            process_parameters={},
            safety_requirements={},
            constraints=[],
        )
        self._personal_context: Dict[str, Any] = {}
        self._active_thread: Dict[str, Any] = {}

    # ---- Message Management ----

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a message to the conversation history."""
        msg = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self._history.append(msg)

        # Extract entities from new message
        if role == "user":
            self._extract_from_text(content)

    def load_history(self, messages: List[Dict[str, str]]) -> None:
        """Load conversation history from a list of message dicts."""
        for msg in messages:
            self.add_message(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                metadata=msg.get("metadata"),
            )

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history as list of dicts."""
        return [
            {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp}
            for msg in self._history
        ]

    def get_recent_messages(self, n: int = 5) -> List[Dict[str, str]]:
        """Get the N most recent messages."""
        recent = list(self._history)[-n:]
        return [
            {"role": msg.role, "content": msg.content}
            for msg in recent
        ]

    # ---- Entity Extraction ----

    def _extract_from_text(self, text: str) -> None:
        """Extract entities from a text message and accumulate."""
        text_lower = text.lower()

        # Extract specifications
        for spec_name, pattern in SPEC_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the last match (most recent mention)
                match = matches[-1]
                if isinstance(match, tuple):
                    value = " to ".join(m for m in match if m)
                else:
                    value = match
                self._extracted_entities.specifications[spec_name] = value

        # Extract vendors
        for vendor in KNOWN_VENDORS:
            if vendor.lower() in text_lower:
                if vendor not in self._extracted_entities.vendors:
                    self._extracted_entities.vendors.append(vendor)

        # Extract instrument types
        for inst_type in INSTRUMENT_TYPES:
            if inst_type.lower() in text_lower:
                if inst_type not in self._extracted_entities.instrument_types:
                    self._extracted_entities.instrument_types.append(inst_type)

        # Extract safety requirements
        sil_match = re.search(r'SIL\s*([1-4])', text, re.IGNORECASE)
        if sil_match:
            self._extracted_entities.safety_requirements["sil_level"] = f"SIL {sil_match.group(1)}"

        atex_match = re.search(r'(?:ATEX|Zone)\s*([012])', text, re.IGNORECASE)
        if atex_match:
            self._extracted_entities.safety_requirements["atex_zone"] = f"Zone {atex_match.group(1)}"

        # Extract domain keywords
        from .intent_analyzer import DOMAIN_KEYWORDS
        for domain, keywords in DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower and domain not in self._extracted_entities.domains:
                    self._extracted_entities.domains.append(domain)

    def get_extracted_entities(self) -> Dict[str, Any]:
        """Get all extracted entities as a dictionary."""
        return {
            "specifications": self._extracted_entities.specifications,
            "vendors": self._extracted_entities.vendors,
            "domains": self._extracted_entities.domains,
            "instrument_types": self._extracted_entities.instrument_types,
            "process_parameters": self._extracted_entities.process_parameters,
            "safety_requirements": self._extracted_entities.safety_requirements,
            "constraints": self._extracted_entities.constraints,
        }

    # ---- Personal Context ----

    def load_personal_context(self, user_id: str) -> Dict[str, Any]:
        """
        Load personal context for a user.

        Tries to load from:
        1. CosmosDB user profile
        2. Redis session cache
        3. Default empty context
        """
        if not user_id:
            return {}

        # Try CosmosDB
        try:
            from common.infrastructure.state.session.cosmos_manager import CosmosSessionManager
            cosmos = CosmosSessionManager.get_instance()
            user_profile = cosmos.get_user_profile(user_id)
            if user_profile:
                self._personal_context = {
                    "preferred_units": user_profile.get("preferred_units", "metric"),
                    "industry": user_profile.get("industry", ""),
                    "preferred_vendors": user_profile.get("preferred_vendors", []),
                    "saved_projects": user_profile.get("saved_projects", []),
                    "communication_protocols": user_profile.get("communication_protocols", []),
                    "default_safety_level": user_profile.get("default_safety_level", ""),
                }
                logger.info(f"[ContextManager] Loaded personal context for user {user_id}")
                return self._personal_context
        except Exception as e:
            logger.debug(f"[ContextManager] CosmosDB personal context unavailable: {e}")

        # Try Redis session
        try:
            from common.infrastructure.state.session.orchestrator import SessionOrchestrator
            session_orch = SessionOrchestrator.get_instance()
            sessions = session_orch.get_all_active_sessions()
            for session in sessions:
                if session.user_id == user_id:
                    self._personal_context = session.metadata.get("personal_context", {})
                    if self._personal_context:
                        logger.info(f"[ContextManager] Loaded context from active session for user {user_id}")
                        return self._personal_context
        except Exception as e:
            logger.debug(f"[ContextManager] Session context unavailable: {e}")

        return {}

    def get_personal_context(self) -> Dict[str, Any]:
        """Get loaded personal context."""
        return self._personal_context

    # ---- Active Thread Analysis ----

    def analyze_active_thread(self) -> Dict[str, Any]:
        """
        Analyze the current conversation thread for context continuity.

        Returns accumulated context from the conversation:
        - Mentioned specifications
        - Active domain/industry
        - Vendor preferences
        - Ongoing requirements
        """
        entities = self.get_extracted_entities()

        # Build active thread context
        self._active_thread = {
            "has_context": bool(self._history),
            "message_count": len(self._history),
            "accumulated_specs": entities["specifications"],
            "active_domain": entities["domains"][-1] if entities["domains"] else "",
            "mentioned_vendors": entities["vendors"],
            "mentioned_instruments": entities["instrument_types"],
            "safety_requirements": entities["safety_requirements"],
            "context_summary": self._build_context_summary(),
        }

        return self._active_thread

    def _build_context_summary(self) -> str:
        """Build a concise summary of conversation context."""
        parts = []
        entities = self._extracted_entities

        if entities.domains:
            parts.append(f"Industry: {', '.join(entities.domains)}")
        if entities.specifications:
            spec_summary = ", ".join(f"{k}: {v}" for k, v in list(entities.specifications.items())[:5])
            parts.append(f"Specs: {spec_summary}")
        if entities.vendors:
            parts.append(f"Vendors: {', '.join(entities.vendors)}")
        if entities.safety_requirements:
            safety = ", ".join(f"{k}: {v}" for k, v in entities.safety_requirements.items())
            parts.append(f"Safety: {safety}")

        return " | ".join(parts) if parts else "No prior context"

    # ---- Enriched Context for Prompts ----

    def get_enriched_context(self) -> str:
        """
        Build enriched context string combining conversation history,
        personal preferences, and extracted entities.

        This is used to augment LLM prompts with conversation awareness.
        """
        sections = []

        # Conversation context
        if self._history:
            recent = self.get_recent_messages(3)
            if recent:
                conv_text = "\n".join(f"[{m['role']}]: {m['content'][:200]}" for m in recent)
                sections.append(f"RECENT CONVERSATION:\n{conv_text}")

        # Extracted entities
        entities = self.get_extracted_entities()
        if entities["specifications"]:
            spec_text = ", ".join(f"{k}: {v}" for k, v in entities["specifications"].items())
            sections.append(f"PREVIOUSLY MENTIONED SPECIFICATIONS: {spec_text}")

        if entities["safety_requirements"]:
            safety_text = ", ".join(f"{k}: {v}" for k, v in entities["safety_requirements"].items())
            sections.append(f"SAFETY REQUIREMENTS: {safety_text}")

        if entities["vendors"]:
            sections.append(f"PREFERRED VENDORS: {', '.join(entities['vendors'])}")

        # Personal context
        if self._personal_context:
            if self._personal_context.get("industry"):
                sections.append(f"USER INDUSTRY: {self._personal_context['industry']}")
            if self._personal_context.get("preferred_vendors"):
                sections.append(f"USER PREFERRED VENDORS: {', '.join(self._personal_context['preferred_vendors'])}")
            if self._personal_context.get("default_safety_level"):
                sections.append(f"USER DEFAULT SAFETY: {self._personal_context['default_safety_level']}")

        return "\n\n".join(sections) if sections else ""

    # ---- Vectorization Support ----

    def get_context_for_embedding(self) -> str:
        """
        Build a text representation suitable for embedding/vectorization.
        Used for semantic search of relevant context.
        """
        parts = []

        # Current user input context
        if self._history:
            last_user_msg = None
            for msg in reversed(list(self._history)):
                if msg.role == "user":
                    last_user_msg = msg.content
                    break
            if last_user_msg:
                parts.append(last_user_msg)

        # Accumulated entities
        entities = self.get_extracted_entities()
        if entities["domains"]:
            parts.append(" ".join(entities["domains"]))
        if entities["instrument_types"]:
            parts.append(" ".join(entities["instrument_types"]))

        return " ".join(parts)
