"""
Node: collect_requirements_node
=================================

Step 3 of the Product Search Deep Agent — Human-in-the-Loop (HITL).

Two modes
---------
auto  (sales_agent_mode == "auto"):
    Skips all human interaction. Builds structured_requirements directly from
    provided_requirements + available_advanced_params and sets user_confirmed=True.

interactive (default):
    First pass  → builds a requirements summary, sets awaiting_user_input=True,
                  and sets the sales_agent_response the caller can return to UI.
                  The LangGraph interrupt is raised so the graph waits.
    Resume pass → user has confirmed/provided requirements via the resume call.
                  Sets user_confirmed=True, awaiting_user_input=False, and
                  merges any user-supplied updates into structured_requirements.

Reads from state:
  sales_agent_mode, user_confirmed, provided_requirements,
  available_advanced_params, schema, product_type, session_id

Writes to state:
  structured_requirements, user_confirmed, awaiting_user_input,
  sales_agent_response, current_step
"""

import json
import logging
from typing import TYPE_CHECKING, Dict, Any, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from common.agentic.models import ProductSearchDeepAgentState


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_requirements_summary(state: "ProductSearchDeepAgentState") -> str:
    """Build a human-readable requirements summary for the HITL response."""
    product_type = state.get("product_type", "product")
    provided = state.get("provided_requirements") or {}
    advanced = state.get("available_advanced_params") or []
    schema = state.get("schema") or {}

    lines = [f"## Requirements Summary for: {product_type}\n"]

    # Provided requirements
    mandatory = provided.get("mandatoryRequirements") or provided.get("mandatory") or {}
    optional = provided.get("optionalRequirements") or provided.get("optional") or {}

    if mandatory:
        lines.append("### Mandatory Requirements")
        for k, v in mandatory.items():
            if v:
                lines.append(f"- **{_title(k)}**: {v}")

    if optional:
        lines.append("\n### Optional Requirements")
        for k, v in optional.items():
            if v:
                lines.append(f"- {_title(k)}: {v}")

    # Missing fields
    missing = state.get("missing_fields") or []
    if missing:
        lines.append("\n### Missing Specifications (please provide if known)")
        for field in missing:
            field_def = (
                schema.get("mandatory", {}).get(field)
                or schema.get("optional", {}).get(field)
                or {}
            )
            desc = field_def.get("description", "") if isinstance(field_def, dict) else ""
            lines.append(f"- **{_title(field)}**{f': {desc}' if desc else ''}")

    # Available advanced params
    if advanced:
        lines.append("\n### Available Advanced Specifications")
        lines.append("*(Select any that apply to your application)*")
        for spec in advanced[:10]:
            name = spec.get("name") or spec.get("key", "")
            desc = spec.get("description", "")
            lines.append(f"- {name}{f': {desc}' if desc else ''}")
        if len(advanced) > 10:
            lines.append(f"  *(and {len(advanced) - 10} more)*")

    lines.append("\n---")
    lines.append("Please confirm these requirements or provide any corrections/additions.")

    return "\n".join(lines)


def _build_structured_requirements(state: "ProductSearchDeepAgentState") -> Dict[str, Any]:
    """
    Build the structured_requirements dict that vendor analysis expects.
    Merges provided_requirements with selected advanced params.
    """
    provided = state.get("provided_requirements") or {}

    structured: Dict[str, Any] = {
        "mandatoryRequirements": provided.get("mandatoryRequirements")
                                 or provided.get("mandatory")
                                 or {},
        "optionalRequirements": provided.get("optionalRequirements")
                                or provided.get("optional")
                                or {},
        "selectedAdvancedParams": {},
        "product_type": state.get("product_type", ""),
        "schema_source": state.get("schema_source", ""),
    }

    # Include any advanced params the user may have selected
    # (In auto mode we include none; in interactive mode they come from user_input on resume)
    advanced_selected = provided.get("selectedAdvancedParams") or {}
    if advanced_selected:
        structured["selectedAdvancedParams"] = advanced_selected

    return structured


def _title(field: str) -> str:
    """Convert camelCase / snake_case to Title Case."""
    import re
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", field).replace("_", " ")
    return words.title()


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def collect_requirements_node(state: "ProductSearchDeepAgentState") -> "ProductSearchDeepAgentState":
    """
    LangGraph node — Step 3: Collect / confirm requirements (HITL or auto).
    """
    logger.info("[collect_requirements_node] ===== STEP 3: COLLECT REQUIREMENTS =====")
    state["current_step"] = "collect_requirements"

    # -------------------------------------------------------------------------
    # AUTO MODE — no human interaction
    # -------------------------------------------------------------------------
    if state.get("sales_agent_mode") == "auto":
        logger.info("[collect_requirements_node] Auto mode — building structured requirements and confirming")
        state["structured_requirements"] = _build_structured_requirements(state)
        state["user_confirmed"] = True
        state["awaiting_user_input"] = False
        state["messages"] = state.get("messages", []) + [
            {"role": "system", "content": "[Step 3] Auto mode: requirements confirmed"}
        ]
        return state

    # -------------------------------------------------------------------------
    # INTERACTIVE MODE
    # -------------------------------------------------------------------------

    # Resume path — user has already confirmed
    if state.get("user_confirmed"):
        logger.info("[collect_requirements_node] Resume path — user_confirmed=True")
        # Ensure structured_requirements is populated (may have been set by resume call)
        if not state.get("structured_requirements"):
            state["structured_requirements"] = _build_structured_requirements(state)
        state["awaiting_user_input"] = False
        state["messages"] = state.get("messages", []) + [
            {"role": "system", "content": "[Step 3] Requirements confirmed by user"}
        ]
        return state

    # First pass — build summary and interrupt
    logger.info("[collect_requirements_node] First pass — building requirements summary and interrupting")
    summary = _build_requirements_summary(state)
    state["sales_agent_response"] = summary
    state["awaiting_user_input"] = True

    # Signal to the graph that we need user input (interrupt will be raised in workflow)
    state["messages"] = state.get("messages", []) + [
        {"role": "assistant", "content": summary}
    ]

    logger.info("[collect_requirements_node] HITL interrupt — awaiting user confirmation")

    return state
