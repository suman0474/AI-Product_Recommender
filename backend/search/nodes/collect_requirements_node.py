# search/nodes/collect_requirements_node.py
# =============================================================================
# COLLECT REQUIREMENTS NODE (Node 3) - HITL
# =============================================================================
#
# Handles immediate post-validation HITL interaction.
#
# Two-phase HITL:
# 1. awaitMissingInfo: Prompt user for missing mandatory fields
# 2. awaitAdditionalSpecs: Ask for extra requirements beyond schema
#
# =============================================================================

import logging
from typing import TYPE_CHECKING

from langgraph.types import interrupt

from ..agents import RequirementsCollectionAgent
from ..state import add_system_message, mark_step_complete

if TYPE_CHECKING:
    from ..state import SearchDeepAgentState

logger = logging.getLogger(__name__)


def collect_requirements_node(state: "SearchDeepAgentState") -> "SearchDeepAgentState":
    """
    LangGraph node - Phase 3: Collect Requirements (HITL).

    Handles two-phase HITL interaction:
    1. awaitMissingInfo - Collect missing mandatory fields
    2. awaitAdditionalSpecs - Collect extra requirements

    Reads: missing_fields, schema, provided_requirements, hitl_phase
    Writes: structured_requirements, awaiting_user_input, sales_agent_response
    """
    logger.info("[collect_requirements_node] ===== PHASE 3: COLLECT REQUIREMENTS =====")
    state["current_step"] = "collect_requirements"

    try:
        agent = RequirementsCollectionAgent()
        hitl_phase = state.get("hitl_phase", "awaitMissingInfo")
        auto_mode = state.get("auto_mode", False)

        # In auto mode, skip HITL and merge what we have
        if auto_mode:
            logger.info("[collect_requirements_node] Auto mode - skipping HITL")
            state["structured_requirements"] = state.get("provided_requirements", {})
            state["hitl_phase"] = "complete"
            state["user_confirmed"] = True
            add_system_message(state, "Auto mode - proceeding with provided requirements", "collect_requirements")
            mark_step_complete(state, "collect_requirements")
            return state

        # =====================================================================
        # PHASE 1: awaitMissingInfo
        # =====================================================================
        if hitl_phase == "awaitMissingInfo":
            missing = state.get("missing_fields", [])

            # Check if there are missing fields and user hasn't confirmed
            if missing and not state.get("user_confirmed"):
                logger.info("[collect_requirements_node] Phase 1: Collecting missing info for %d fields", len(missing))

                result = agent.collect_missing_info(
                    schema=state.get("schema", {}),
                    missing_fields=missing,
                    provided_requirements=state.get("provided_requirements", {}),
                )

                # Set HITL interrupt state
                state["hitl_phase"] = "awaitMissingInfo"
                state["missing_mandatory_fields"] = result.missing_mandatory_fields
                state["awaiting_user_input"] = True
                state["sales_agent_response"] = result.sales_agent_response
                state["workflow_interrupted"] = True
                state["interrupt_reason"] = "awaitMissingInfo"

                add_system_message(
                    state,
                    f"Awaiting user input for {len(missing)} missing fields",
                    "collect_requirements",
                )

                # Interrupt workflow for user input
                logger.info("[collect_requirements_node] Interrupting for awaitMissingInfo")
                interrupt({
                    "reason": "awaitMissingInfo",
                    "phase": "collect_requirements",
                    "missing_fields": result.missing_mandatory_fields,
                    "sales_agent_response": result.sales_agent_response,
                })

                return state

            # User has responded - process their input
            if state.get("user_provided_values"):
                logger.info("[collect_requirements_node] Processing user response for missing info")

                merged = agent.process_user_response(
                    response_type=state.get("hitl_response_type", "provide"),
                    user_values=state.get("user_provided_values"),
                    skipped_fields=state.get("skipped_fields"),
                    current_requirements=state.get("provided_requirements", {}),
                )

                # Update provided requirements
                state["provided_requirements"] = merged

            # Move to Phase 2
            state["hitl_phase"] = "awaitAdditionalSpecs"
            hitl_phase = "awaitAdditionalSpecs"

        # =====================================================================
        # PHASE 2: awaitAdditionalSpecs
        # =====================================================================
        if hitl_phase == "awaitAdditionalSpecs":
            if not state.get("additional_specs_collected"):
                logger.info("[collect_requirements_node] Phase 2: Collecting additional specs")

                result = agent.collect_additional_specs(
                    current_requirements=state.get("provided_requirements", {}),
                )

                # Set HITL interrupt state
                state["awaiting_user_input"] = True
                state["sales_agent_response"] = result.sales_agent_response
                state["workflow_interrupted"] = True
                state["interrupt_reason"] = "awaitAdditionalSpecs"

                add_system_message(
                    state,
                    "Awaiting user input for additional specifications",
                    "collect_requirements",
                )

                # Interrupt workflow for user input
                logger.info("[collect_requirements_node] Interrupting for awaitAdditionalSpecs")
                interrupt({
                    "reason": "awaitAdditionalSpecs",
                    "phase": "collect_requirements",
                    "sales_agent_response": result.sales_agent_response,
                })

                return state

            # User has responded - parse their free-text
            raw_specs = state.get("additional_specs_raw", "")
            if raw_specs and raw_specs.strip():
                logger.info("[collect_requirements_node] Parsing additional specs from user")

                parsed = agent.parse_additional_specs(
                    raw_text=raw_specs,
                    current_requirements=state.get("provided_requirements", {}),
                )
                state["additional_specs_parsed"] = parsed
            else:
                state["additional_specs_parsed"] = {}

            # Move to complete
            state["hitl_phase"] = "complete"

        # =====================================================================
        # FINAL: Merge all requirements
        # =====================================================================
        if state.get("hitl_phase") == "complete" or not missing:
            logger.info("[collect_requirements_node] Merging all requirements")

            state["structured_requirements"] = agent.merge_requirements(
                provided=state.get("provided_requirements", {}),
                user_values=state.get("user_provided_values", {}),
                additional_parsed=state.get("additional_specs_parsed", {}),
            )

            # Clear interrupt state
            state["awaiting_user_input"] = False
            state["user_confirmed"] = True
            state["workflow_interrupted"] = False
            state["interrupt_reason"] = ""

            req_count = len([k for k in state["structured_requirements"].keys() if not k.startswith("_")])
            add_system_message(
                state,
                f"Requirements collected: {req_count} fields",
                "collect_requirements",
            )

            mark_step_complete(state, "collect_requirements")

            logger.info(
                "[collect_requirements_node] Requirements collection complete: %d fields",
                req_count,
            )

    except Exception as exc:
        logger.error("[collect_requirements_node] Failed: %s", exc, exc_info=True)

        # On error, proceed with what we have
        state["structured_requirements"] = state.get("provided_requirements", {})
        state["user_confirmed"] = True
        state["workflow_interrupted"] = False
        state["error"] = f"Requirements collection failed: {str(exc)}"

        add_system_message(state, f"Error, proceeding with defaults: {str(exc)}", "collect_requirements")
        mark_step_complete(state, "collect_requirements")

    return state
