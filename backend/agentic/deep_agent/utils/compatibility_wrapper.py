# agentic/deep_agent/compatibility_wrapper.py
"""
Compatibility Wrapper for Gradual Migration

Provides backward-compatible interface that can use either:
- Legacy in-memory system (existing code)
- New distributed queue-based system

Controlled by feature flag: ENABLE_DISTRIBUTED_ENRICHMENT
"""

import logging
import os
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE FLAG
# =============================================================================

def is_distributed_enabled() -> bool:
    """
    Check if distributed enrichment is enabled.

    Returns:
        True if ENABLE_DISTRIBUTED_ENRICHMENT=true, False otherwise
    """
    return os.getenv("ENABLE_DISTRIBUTED_ENRICHMENT", "false").lower() == "true"


# =============================================================================
# COMPATIBILITY FUNCTIONS
# =============================================================================

def run_standards_deep_agent_compat(
    user_requirement: str,
    session_id: Optional[str] = None,
    inferred_specs: Optional[Dict[str, Any]] = None,
    min_specs: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compatibility wrapper for run_standards_deep_agent.

    If distributed mode enabled, uses new queue-based system.
    Otherwise, uses legacy in-memory system.

    Args:
        user_requirement: User's requirement string
        session_id: Optional session ID
        inferred_specs: Optional inferred specifications
        min_specs: Minimum specs required

    Returns:
        Enrichment result dict (same format as legacy)
    """
    if is_distributed_enabled():
        logger.info("[COMPAT] Using DISTRIBUTED enrichment system")
        return _run_distributed_single(
            user_requirement, session_id, inferred_specs, min_specs
        )
    else:
        logger.info("[COMPAT] Using LEGACY enrichment system")
        return _run_legacy_single(
            user_requirement, session_id, inferred_specs, min_specs
        )


def run_standards_deep_agent_batch_compat(
    items: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    domain_context: Optional[str] = None,
    safety_requirements: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compatibility wrapper for run_standards_deep_agent_batch.

    If distributed mode enabled, uses new queue-based system.
    Otherwise, uses legacy in-memory system.

    Args:
        items: List of items to enrich
        session_id: Optional session ID
        domain_context: Optional domain context
        safety_requirements: Optional safety requirements

    Returns:
        Batch enrichment result dict (same format as legacy)
    """
    if is_distributed_enabled():
        logger.info(f"[COMPAT] Using DISTRIBUTED batch enrichment for {len(items)} items")
        return _run_distributed_batch(
            items, session_id, domain_context, safety_requirements
        )
    else:
        logger.info(f"[COMPAT] Using LEGACY batch enrichment for {len(items)} items")
        return _run_legacy_batch(
            items, session_id, domain_context, safety_requirements
        )


# =============================================================================
# DISTRIBUTED IMPLEMENTATIONS
# =============================================================================

def _run_distributed_single(
    user_requirement: str,
    session_id: Optional[str],
    inferred_specs: Optional[Dict[str, Any]],
    min_specs: Optional[int]
) -> Dict[str, Any]:
    """Run single-item enrichment using distributed system."""
    try:
        from ..orchestration.stateless_orchestrator import get_orchestrator

        orchestrator = get_orchestrator(use_redis=True)

        # Create item from requirement
        item = {
            "name": user_requirement,
            "specifications": inferred_specs or {}
        }

        # Start session
        session_id = orchestrator.start_enrichment_session(
            items=[item],
            session_id=session_id
        )

        # Wait for completion (blocking)
        result = orchestrator.wait_for_session_completion(
            session_id,
            timeout=300  # 5 minutes
        )

        # Convert to legacy format
        if result["success"] and result["items"]:
            item_result = result["items"][0]

            return {
                "success": True,
                "status": "completed",
                "final_specifications": {
                    "specifications": item_result["specifications"],
                    "confidence": 0.85
                },
                "specs_count": item_result["spec_count"],
                "iterations": item_result["iterations"],
                "processing_time_ms": result["batch_metadata"]["processing_time_ms"],
                "session_id": session_id,
                "enrichment_mode": "distributed"
            }
        else:
            return {
                "success": False,
                "status": "failed",
                "error": "Enrichment failed or timed out",
                "session_id": session_id,
                "enrichment_mode": "distributed"
            }

    except Exception as e:
        logger.error(f"[COMPAT] Distributed enrichment failed: {e}", exc_info=True)
        return {
            "success": False,
            "status": "error",
            "error": str(e),
            "enrichment_mode": "distributed"
        }


def _run_distributed_batch(
    items: List[Dict[str, Any]],
    session_id: Optional[str],
    domain_context: Optional[str],
    safety_requirements: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Run batch enrichment using distributed system."""
    try:
        from ..orchestration.stateless_orchestrator import get_orchestrator

        orchestrator = get_orchestrator(use_redis=True)

        # Start session
        session_id = orchestrator.start_enrichment_session(
            items=items,
            session_id=session_id
        )

        # Wait for completion (blocking)
        result = orchestrator.wait_for_session_completion(
            session_id,
            timeout=600  # 10 minutes for batch
        )

        # Convert to legacy format
        if result["success"]:
            # Map distributed results to legacy format
            enriched_items = []

            for item_result in result["items"]:
                enriched_item = {
                    "standards_info": {
                        "enrichment_status": "success",
                        "enrichment_method": "distributed_deep_agent",
                        "total_specs_count": item_result["spec_count"],
                        "standards_analyzed": item_result.get("domains_analyzed", [])
                    },
                    "standards_specifications": item_result["specifications"],
                    "combined_specifications": item_result["specifications"],
                    "specification_source": {
                        "standards_pct": 100,
                        "database_pct": 0,
                        "standards_count": item_result["spec_count"],
                        "total_count": item_result["spec_count"],
                        "target_reached": item_result["spec_count"] >= 30
                    }
                }

                enriched_items.append(enriched_item)

            return {
                "success": True,
                "items": enriched_items,
                "batch_metadata": {
                    **result["batch_metadata"],
                    "enrichment_mode": "distributed"
                }
            }
        else:
            return {
                "success": False,
                "error": "Batch enrichment failed or timed out",
                "items": items,  # Return original items
                "batch_metadata": {
                    "enrichment_mode": "distributed",
                    "error": result.get("error")
                }
            }

    except Exception as e:
        logger.error(f"[COMPAT] Distributed batch enrichment failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "items": items,
            "batch_metadata": {
                "enrichment_mode": "distributed",
                "error": str(e)
            }
        }


# =============================================================================
# LEGACY IMPLEMENTATIONS
# =============================================================================

def _run_legacy_single(
    user_requirement: str,
    session_id: Optional[str],
    inferred_specs: Optional[Dict[str, Any]],
    min_specs: Optional[int]
) -> Dict[str, Any]:
    """Run single-item enrichment using legacy system."""
    try:
        from ..standards_deep_agent import run_standards_deep_agent

        return run_standards_deep_agent(
            user_requirement=user_requirement,
            session_id=session_id,
            inferred_specs=inferred_specs,
            min_specs=min_specs
        )

    except Exception as e:
        logger.error(f"[COMPAT] Legacy enrichment failed: {e}", exc_info=True)
        return {
            "success": False,
            "status": "error",
            "error": str(e),
            "enrichment_mode": "legacy"
        }


def _run_legacy_batch(
    items: List[Dict[str, Any]],
    session_id: Optional[str],
    domain_context: Optional[str],
    safety_requirements: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Run batch enrichment using legacy system."""
    try:
        from ..standards_deep_agent import run_standards_deep_agent_batch

        return run_standards_deep_agent_batch(
            items=items,
            session_id=session_id,
            domain_context=domain_context,
            safety_requirements=safety_requirements
        )

    except Exception as e:
        logger.error(f"[COMPAT] Legacy batch enrichment failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "items": items,
            "batch_metadata": {
                "enrichment_mode": "legacy",
                "error": str(e)
            }
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "is_distributed_enabled",
    "run_standards_deep_agent_compat",
    "run_standards_deep_agent_batch_compat"
]
