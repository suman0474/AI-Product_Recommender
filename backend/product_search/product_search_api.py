"""
Product Search API Blueprint — Phase C
========================================

Registers REST endpoints for the Product Search Deep Agent workflow.

Endpoints
---------
POST /api/product-search/run
    Start a full product search workflow (auto-mode or interactive).

POST /api/product-search/resume
    Resume a HITL-interrupted workflow after the user has confirmed requirements.

POST /api/product-search/validate
    Run only Step 1 (validate product type, load/generate schema, enrich with standards).

POST /api/product-search/advanced-params
    Run only Step 2 (discover advanced parameters for a given product type).

GET  /api/product-search/status/<session_id>
    Return current workflow status for a session.

POST /api/product-search/schema
    Load or generate a schema for a product type (no validation against user input).

All endpoints expect JSON bodies and return JSON responses.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

product_search_bp = Blueprint("product_search", __name__)


# =============================================================================
# Helpers
# =============================================================================

def _get_json() -> Dict[str, Any]:
    """Parse request JSON body, returning empty dict on failure."""
    try:
        data = request.get_json(force=True, silent=True)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _error(message: str, status: int = 400) -> Any:
    return jsonify({"success": False, "error": message}), status


def _ok(payload: Dict[str, Any]) -> Any:
    payload.setdefault("success", True)
    return jsonify(payload), 200


# =============================================================================
# POST /api/product-search/run
# =============================================================================

@product_search_bp.route("/api/product-search/run", methods=["POST"])
def product_search_run():
    """
    Run the full Product Search Deep Agent pipeline.

    Request body (JSON):
        user_input          (str, required)  — raw user requirements text
        session_id          (str, optional)  — caller session; generated if absent
        auto_mode           (bool, optional) — default True
        enable_ppi          (bool, optional) — default True
        skip_advanced_params(bool, optional) — default False
        max_vendor_workers  (int, optional)  — default 10
        expected_product_type (str, optional)

    Response:
        Standard analysis_result + response_data from the final graph state.
    """
    data = _get_json()

    user_input: str = data.get("user_input", "").strip()
    if not user_input:
        return _error("user_input is required")

    session_id: str = data.get("session_id") or str(uuid.uuid4())
    auto_mode: bool = bool(data.get("auto_mode", True))
    enable_ppi: bool = bool(data.get("enable_ppi", True))
    skip_advanced_params: bool = bool(data.get("skip_advanced_params", False))
    max_vendor_workers: int = int(data.get("max_vendor_workers", 10))
    expected_product_type: Optional[str] = data.get("expected_product_type")

    logger.info(
        "[product_search_run] session=%s auto=%s ppi=%s skip_adv=%s",
        session_id, auto_mode, enable_ppi, skip_advanced_params,
    )

    try:
        from product_search.product_search_workflow import run_product_search_workflow

        result: Dict[str, Any] = run_product_search_workflow(
            user_input=user_input,
            session_id=session_id,
            auto_mode=auto_mode,
            enable_ppi=enable_ppi,
            skip_advanced_params=skip_advanced_params,
            max_vendor_workers=max_vendor_workers,
            expected_product_type=expected_product_type,
        )

        return _ok({
            "session_id": session_id,
            "analysis_result": result.get("analysis_result", {}),
            "response_data": result.get("response_data", {}),
            "product_type": result.get("product_type", ""),
            "awaiting_user_input": result.get("awaiting_user_input", False),
            "sales_agent_response": result.get("sales_agent_response", ""),
            "steps_completed": result.get("steps_completed", []),
            "current_step": result.get("current_step", ""),
        })

    except Exception as exc:
        logger.error("[product_search_run] Failed: %s", exc, exc_info=True)
        return _error(f"Workflow error: {exc}", 500)


# =============================================================================
# POST /api/product-search/resume
# =============================================================================

@product_search_bp.route("/api/product-search/resume", methods=["POST"])
def product_search_resume():
    """
    Resume a HITL-interrupted Product Search workflow.

    Request body (JSON):
        session_id          (str, required)  — must match a previously interrupted session
        thread_id           (str, optional)  — LangGraph thread ID (same as session_id by default)
        user_response       (str, required)  — "YES" | "NO" | free-text additional specs
        selected_advanced_params (dict, optional) — {key: value} for chosen advanced specs

    Response:
        Updated analysis_result after the workflow continues past Step 3.
    """
    data = _get_json()

    session_id: str = data.get("session_id", "").strip()
    if not session_id:
        return _error("session_id is required")

    user_response: str = data.get("user_response", "").strip()
    if not user_response:
        return _error("user_response is required")

    thread_id: str = data.get("thread_id") or session_id
    selected_advanced_params: Dict[str, Any] = data.get("selected_advanced_params", {})

    logger.info(
        "[product_search_resume] session=%s thread=%s response='%s'",
        session_id, thread_id, user_response[:80],
    )

    try:
        from product_search.product_search_workflow import resume_product_search_workflow

        result: Dict[str, Any] = resume_product_search_workflow(
            workflow_thread_id=thread_id,
            user_response=user_response,
            selected_advanced_params=selected_advanced_params,
        )

        return _ok({
            "session_id": session_id,
            "analysis_result": result.get("analysis_result", {}),
            "response_data": result.get("response_data", {}),
            "product_type": result.get("product_type", ""),
            "awaiting_user_input": result.get("awaiting_user_input", False),
            "current_step": result.get("current_step", ""),
        })

    except Exception as exc:
        logger.error("[product_search_resume] Failed: %s", exc, exc_info=True)
        return _error(f"Resume error: {exc}", 500)


# =============================================================================
# POST /api/product-search/validate
# =============================================================================

@product_search_bp.route("/api/product-search/validate", methods=["POST"])
def product_search_validate():
    """
    Run only Step 1 of the workflow: validate product type + load/enrich schema.

    Request body (JSON):
        user_input            (str, required)
        session_id            (str, optional)
        enable_ppi            (bool, optional) — default True
        expected_product_type (str, optional)

    Response:
        product_type, schema, validation_result, missing_fields, is_valid
    """
    data = _get_json()

    user_input: str = data.get("user_input", "").strip()
    if not user_input:
        return _error("user_input is required")

    session_id: str = data.get("session_id") or str(uuid.uuid4())
    enable_ppi: bool = bool(data.get("enable_ppi", True))
    expected_product_type: Optional[str] = data.get("expected_product_type")

    logger.info(
        "[product_search_validate] session=%s ppi=%s", session_id, enable_ppi
    )

    try:
        from product_search.product_search_workflow import run_validation_only

        result: Dict[str, Any] = run_validation_only(
            user_input=user_input,
            session_id=session_id,
            enable_ppi=enable_ppi,
            expected_product_type=expected_product_type,
        )

        return _ok({
            "session_id": session_id,
            "product_type": result.get("product_type", ""),
            "schema": result.get("schema", {}),
            "validation_result": result.get("validation_result", {}),
            "is_valid": result.get("is_valid", False),
            "missing_fields": result.get("missing_fields", []),
            "optional_fields": result.get("optional_fields", []),
            "provided_requirements": result.get("provided_requirements", {}),
        })

    except Exception as exc:
        logger.error("[product_search_validate] Failed: %s", exc, exc_info=True)
        return _error(f"Validation error: {exc}", 500)


# =============================================================================
# POST /api/product-search/advanced-params
# =============================================================================

@product_search_bp.route("/api/product-search/advanced-params", methods=["POST"])
def product_search_advanced_params():
    """
    Run only Step 2: discover advanced/vendor-specific parameters.

    Request body (JSON):
        product_type  (str, required)
        session_id    (str, optional)
        schema        (dict, optional)  — existing schema to filter against

    Response:
        unique_specifications, total_unique_specifications, vendors_searched
    """
    data = _get_json()

    product_type: str = data.get("product_type", "").strip()
    if not product_type:
        return _error("product_type is required")

    session_id: str = data.get("session_id") or str(uuid.uuid4())
    existing_schema: Optional[Dict[str, Any]] = data.get("schema")

    logger.info(
        "[product_search_advanced_params] product_type='%s' session=%s",
        product_type, session_id,
    )

    try:
        from product_search.nodes.discover_advanced_params_node import (
            _tool as advanced_tool,
        )

        result = advanced_tool.discover(
            product_type=product_type,
            session_id=session_id,
            existing_schema=existing_schema,
        )

        return _ok({
            "session_id": session_id,
            "product_type": product_type,
            "unique_specifications": result.get("unique_specifications", []),
            "total_unique_specifications": result.get("total_unique_specifications", 0),
            "existing_specifications_filtered": result.get(
                "existing_specifications_filtered", 0
            ),
            "vendors_searched": result.get("vendors_searched", []),
            "discovery_successful": result.get("discovery_successful", False),
        })

    except Exception as exc:
        logger.error(
            "[product_search_advanced_params] Failed: %s", exc, exc_info=True
        )
        return _error(f"Advanced params error: {exc}", 500)


# =============================================================================
# GET /api/product-search/status/<session_id>
# =============================================================================

@product_search_bp.route(
    "/api/product-search/status/<string:session_id>", methods=["GET"]
)
def product_search_status(session_id: str):
    """
    Return the current workflow status for a session.

    Uses WorkflowInstanceManager to look up the registered instance.
    Falls back to {"status": "unknown"} gracefully.
    """
    if not session_id:
        return _error("session_id path parameter is required")

    try:
        from common.infrastructure.state.execution.instance_manager import (
            WorkflowInstanceManager,
        )

        manager = WorkflowInstanceManager.get_instance()
        instance = manager.get_instance_by_session(session_id)

        if instance is None:
            return _ok({"session_id": session_id, "status": "not_found"})

        return _ok({
            "session_id": session_id,
            "instance_id": instance.get("instance_id"),
            "status": instance.get("status", "unknown"),
            "current_step": instance.get("current_step"),
            "created_at": instance.get("created_at"),
            "updated_at": instance.get("updated_at"),
        })

    except Exception as exc:
        logger.warning(
            "[product_search_status] Could not retrieve status: %s", exc
        )
        return _ok({"session_id": session_id, "status": "unknown"})


# =============================================================================
# POST /api/product-search/schema
# =============================================================================

@product_search_bp.route("/api/product-search/schema", methods=["POST"])
def product_search_schema():
    """
    Load or generate a product schema (no user-input validation).

    Request body (JSON):
        product_type  (str, required)
        session_id    (str, optional)
        enable_ppi    (bool, optional) — default True

    Response:
        schema, product_type, schema_source
    """
    data = _get_json()

    product_type: str = data.get("product_type", "").strip()
    if not product_type:
        return _error("product_type is required")

    session_id: str = data.get("session_id") or str(uuid.uuid4())
    enable_ppi: bool = bool(data.get("enable_ppi", True))

    logger.info(
        "[product_search_schema] product_type='%s' session=%s ppi=%s",
        product_type, session_id, enable_ppi,
    )

    try:
        from product_search.product_search_workflow import get_schema_only

        result = get_schema_only(
            product_type=product_type,
            session_id=session_id,
            enable_ppi=enable_ppi,
        )

        return _ok({
            "session_id": session_id,
            "product_type": product_type,
            "schema": result.get("schema", {}),
            "schema_source": result.get("schema_source", ""),
        })

    except Exception as exc:
        logger.error("[product_search_schema] Failed: %s", exc, exc_info=True)
        return _error(f"Schema error: {exc}", 500)
