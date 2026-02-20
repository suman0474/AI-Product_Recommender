"""
Node: discover_advanced_params_node
=====================================

Step 2 of the Product Search Deep Agent.

Discovers advanced/extended parameters (vendor-specific, real-world specs)
that go beyond the baseline schema. The discovery is powered by the
AdvancedParametersTool which wraps the core advanced_parameters.py logic.

Reads from state:
  product_type, schema, session_id, skip_advanced_params

Writes to state:
  advanced_params_result, available_advanced_params,
  discovered_specifications, messages, current_step
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from common.agentic.models import ProductSearchDeepAgentState


# =============================================================================
# ADVANCED PARAMETERS TOOL  (self-contained — no external file dependency)
# =============================================================================

class _AdvancedParametersTool:
    """
    Wraps the core `discover_advanced_parameters` function (from
    advanced_parameters.py) and exposes a structured result dict.

    All imports are lazy so the node still loads even if
    advanced_parameters.py is not present (graceful degradation).
    """

    def discover(
        self,
        product_type: str,
        session_id: Optional[str] = None,
        existing_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Discover advanced parameters for a product type.

        Returns:
            {
                "success":                     bool,
                "product_type":                str,
                "session_id":                  str | None,
                "unique_specifications":       list[dict],
                "total_unique_specifications": int,
                "existing_specifications_filtered": int,
                "vendors_searched":            list[str],
                "discovery_successful":        bool,
            }
        """
        logger.info("[AdvancedParametersTool] Starting discovery for '%s'", product_type)

        result: Dict[str, Any] = {
            "success": False,
            "product_type": product_type,
            "session_id": session_id,
            "unique_specifications": [],
            "total_unique_specifications": 0,
            "existing_specifications_filtered": 0,
            "vendors_searched": [],
            "discovery_successful": False,
        }

        try:
            from .advanced_parameters import discover_advanced_parameters  # type: ignore[import]
        except ImportError:
            logger.warning(
                "[AdvancedParametersTool] advanced_parameters module not found — "
                "returning empty result"
            )
            return result

        try:
            discovery_result = discover_advanced_parameters(
                product_type=product_type.strip()
            )

            unique_specs: List[Dict[str, Any]] = discovery_result.get(
                "unique_specifications", []
            )
            existing_filtered: int = discovery_result.get(
                "existing_specifications_filtered", 0
            )
            vendors_searched: List[str] = discovery_result.get("vendors_searched", [])
            specs_count = len(unique_specs)

            # Optionally filter against an existing schema to remove duplicates
            if existing_schema and unique_specs:
                schema_keys: set = set()
                for section in (
                    "mandatory",
                    "optional",
                    "mandatory_requirements",
                    "optional_requirements",
                ):
                    schema_keys.update(existing_schema.get(section, {}).keys())

                filtered: List[Dict[str, Any]] = [
                    s for s in unique_specs
                    if s.get("key", "").lower() not in schema_keys
                ]
                extra_filtered = len(unique_specs) - len(filtered)
                if extra_filtered:
                    logger.info(
                        "[AdvancedParametersTool] Additional %d specs removed "
                        "(already in schema)",
                        extra_filtered,
                    )
                    existing_filtered += extra_filtered
                    unique_specs = filtered
                    specs_count = len(filtered)

            if specs_count > 0:
                logger.info(
                    "[AdvancedParametersTool] ✓ %d unique specs, %d filtered, %d vendors",
                    specs_count, existing_filtered, len(vendors_searched),
                )
                for i, spec in enumerate(unique_specs[:3], 1):
                    logger.info(
                        "[AdvancedParametersTool]   %d. %s",
                        i, spec.get("name", spec.get("key", "?")),
                    )
                if specs_count > 3:
                    logger.info(
                        "[AdvancedParametersTool]   … and %d more", specs_count - 3
                    )
            else:
                logger.info("[AdvancedParametersTool] No new specifications discovered")

            result.update(
                {
                    "success": True,
                    "unique_specifications": unique_specs,
                    "total_unique_specifications": specs_count,
                    "existing_specifications_filtered": existing_filtered,
                    "vendors_searched": vendors_searched,
                    "discovery_successful": specs_count > 0,
                }
            )

        except Exception as exc:
            logger.error(
                "[AdvancedParametersTool] ✗ Discovery failed: %s", exc, exc_info=True
            )
            result.update(
                {
                    "success": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )

        return result


# Module-level singleton (avoids re-instantiation on every node invocation)
_tool = _AdvancedParametersTool()


# =============================================================================
# NODE FUNCTION
# =============================================================================

def discover_advanced_params_node(
    state: "ProductSearchDeepAgentState",
) -> "ProductSearchDeepAgentState":
    """
    LangGraph node — Step 2: Discover advanced/extended parameters from vendor PDFs.

    If state["skip_advanced_params"] is True the step is skipped and an empty
    (but success=True) result is stored so downstream nodes continue normally.
    """
    logger.info(
        "[discover_advanced_params_node] ===== STEP 2: DISCOVER ADVANCED PARAMS ====="
    )
    state["current_step"] = "discover_advanced_params"

    # ------------------------------------------------------------------
    # Fast-exit: skip flag
    # ------------------------------------------------------------------
    if state.get("skip_advanced_params"):
        logger.info(
            "[discover_advanced_params_node] skip_advanced_params=True — skipping"
        )
        state["advanced_params_result"] = {
            "success": True,
            "unique_specifications": [],
            "total_unique_specifications": 0,
            "skipped": True,
        }
        state["available_advanced_params"] = []
        state["discovered_specifications"] = []
        state["messages"] = state.get("messages", []) + [
            {"role": "system", "content": "[Step 2] Advanced params discovery skipped"}
        ]
        return state

    product_type: str = state.get("product_type", "")
    session_id: Optional[str] = state.get("session_id")
    existing_schema: Optional[Dict[str, Any]] = state.get("schema")

    # Default result shape
    result: Dict[str, Any] = {
        "success": False,
        "product_type": product_type,
        "session_id": session_id,
        "unique_specifications": [],
        "total_unique_specifications": 0,
        "vendors_searched": [],
    }

    if not product_type:
        logger.warning(
            "[discover_advanced_params_node] No product_type in state — skipping"
        )
        result["skipped"] = True
        result["success"] = True
    else:
        # ------------------------------------------------------------------
        # Core discovery
        # ------------------------------------------------------------------
        discovery = _tool.discover(
            product_type=product_type,
            session_id=session_id,
            existing_schema=existing_schema,
        )
        result.update(discovery)

        unique_specs: List[Dict[str, Any]] = discovery.get("unique_specifications", [])
        vendors_searched: List[str] = discovery.get("vendors_searched", [])

        # ------------------------------------------------------------------
        # Update state
        # ------------------------------------------------------------------
        # available_advanced_params is what collect_requirements shows the user
        state["available_advanced_params"] = unique_specs
        # discovered_specifications is what format_response exposes in the output
        state["discovered_specifications"] = unique_specs

        logger.info(
            "[discover_advanced_params_node] ✓ %d unique specs discovered, "
            "%d vendors searched",
            len(unique_specs),
            len(vendors_searched),
        )

    state["advanced_params_result"] = result

    # Append a system message for context
    spec_count = result.get("total_unique_specifications", 0)
    state["messages"] = state.get("messages", []) + [
        {
            "role": "system",
            "content": (
                f"[Step 2] Advanced params: {spec_count} unique specifications discovered"
                if result.get("success")
                else f"[Step 2] Advanced params discovery failed: {result.get('error', 'unknown error')}"
            ),
        }
    ]

    return state
