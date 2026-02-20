"""
Indexing Agent — Quality Assurance Node
=====================================
Final quality assessment, readiness scoring, and deployment decision.
Replaces QASpecialistAgent.
"""

import logging
from typing import Dict, Any, List

from ..utils.llm_helpers import get_llm
from ..state import IndexingState
from .. import config

logger = logging.getLogger(__name__)


# ── Helper functions ────────────────────────────────────────────────────────

def _assess_completeness(schema: Dict[str, Any]) -> float:
    score = checks = 0

    for field in ("product_type", "parameters", "model_families", "metadata"):
        checks += 1
        if field in schema and schema[field]:
            score += 1.0

    param_count = schema.get("metadata", {}).get("parameter_count", 0)
    checks += 1
    if param_count >= 20:
        score += 1.0
    elif param_count >= 15:
        score += 0.8
    elif param_count >= 10:
        score += 0.6
    elif param_count >= 5:
        score += 0.4

    model_count = len(schema.get("model_families", []))
    checks += 1
    if model_count >= 5:
        score += 1.0
    elif model_count >= 3:
        score += 0.7
    elif model_count >= 1:
        score += 0.4

    return score / checks if checks > 0 else 0.0


def _assess_accuracy(
    schema: Dict[str, Any],
    validation_results: Dict[str, Any],
) -> float:
    base = validation_results.get("overall_confidence", 0.5)
    consistency = validation_results.get("validations", {}).get("consistency", 0.5)
    return base * 0.7 + consistency * 0.3


def _assess_consistency(validation_results: Dict[str, Any]) -> float:
    return validation_results.get("validations", {}).get("coherence", 0.7)


def _assess_usability(schema: Dict[str, Any]) -> float:
    score = checks = 0
    parameters = schema.get("parameters", {})

    if parameters:
        with_units = total = 0
        for cat_params in parameters.values():
            for param in cat_params:
                total += 1
                data = param.get("data", {})
                if isinstance(data, dict) and "unit" in data:
                    with_units += 1
        checks += 1
        if total > 0:
            score += with_units / total

    checks += 1
    if len(parameters) >= 3:
        score += 1.0
    elif len(parameters) >= 2:
        score += 0.6

    checks += 1
    metadata = schema.get("metadata", {})
    if metadata and len(metadata) >= 4:
        score += 1.0
    elif metadata:
        score += 0.5

    return score / checks if checks > 0 else 0.5


def _assess_documentation(schema: Dict[str, Any]) -> float:
    score = checks = 0
    for key in ("specifications", "features", "notes"):
        checks += 1
        if schema.get(key):
            score += 1.0
    checks += 1
    if schema.get("sources", {}).get("pdf_count", 0) > 0:
        score += 1.0
    return score / checks if checks > 0 else 0.0


def _calculate_overall(dimensions: Dict[str, float]) -> float:
    weights = {
        "completeness": 0.25,
        "accuracy": 0.30,
        "consistency": 0.20,
        "usability": 0.15,
        "documentation": 0.10,
    }
    return sum(dimensions.get(d, 0) * w for d, w in weights.items())


def _determine_readiness(quality: float) -> str:
    if quality >= config.QUALITY_THRESHOLD_PRODUCTION:
        return "production_ready"
    if quality >= config.QUALITY_THRESHOLD_STAGING:
        return "staging_ready"
    if quality >= 0.50:
        return "needs_improvement"
    return "not_ready"


def _generate_recommendations(
    dimensions: Dict[str, float],
    schema: Dict[str, Any],
) -> List[str]:
    recs: List[str] = []
    if dimensions.get("completeness", 1.0) < 0.7:
        recs.append("Add more parameters to improve completeness")
        if len(schema.get("model_families", [])) < 3:
            recs.append("Include more model family information")
    if dimensions.get("accuracy", 1.0) < 0.7:
        recs.append("Verify specifications with additional sources")
        recs.append("Cross-check parameter values for accuracy")
    if dimensions.get("usability", 1.0) < 0.7:
        recs.append("Add units to all parameters")
        recs.append("Improve parameter categorisation")
    if dimensions.get("documentation", 1.0) < 0.6:
        recs.append("Add detailed feature descriptions")
        recs.append("Include application notes and usage guidelines")
    return recs


def _recommended_action(quality: float) -> str:
    if quality >= config.QUALITY_THRESHOLD_PRODUCTION:
        return "Deploy to production"
    if quality >= config.QUALITY_THRESHOLD_STAGING:
        return "Deploy to staging for testing"
    if quality >= 0.50:
        return "Refine and re-validate"
    return "Gather additional data and restart workflow"


# ── Node function ───────────────────────────────────────────────────────────

def quality_assurance_node(state: IndexingState) -> dict:
    """
    LangGraph node — comprehensive quality assessment of the generated schema.

    Reads:
        ``generated_schema``, ``validation_results``, ``agent_outputs``

    Writes:
        ``qa_assessment``, ``final_quality_score``, ``deployment_ready``,
        ``current_stage``, ``agent_outputs``
    """
    schema = state.get("generated_schema", {})
    validation = state.get("validation_results", {})

    dimensions = {
        "completeness": _assess_completeness(schema),
        "accuracy": _assess_accuracy(schema, validation),
        "consistency": _assess_consistency(validation),
        "usability": _assess_usability(schema),
        "documentation": _assess_documentation(schema),
    }

    quality = _calculate_overall(dimensions)
    readiness = _determine_readiness(quality)
    recs = _generate_recommendations(dimensions, schema)

    strengths = [
        f"Excellent {d.replace('_', ' ')}"
        for d, s in dimensions.items() if s >= 0.8
    ]
    weaknesses = [
        f"Low {d.replace('_', ' ')}"
        for d, s in dimensions.items() if s < 0.6
    ]

    assessment = {
        "overall_quality_score": quality,
        "readiness": readiness,
        "quality_dimensions": dimensions,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "improvement_recommendations": recs,
        "deployment_readiness": {
            "can_deploy": quality >= config.QUALITY_THRESHOLD_STAGING,
            "requires_review": quality < config.QUALITY_THRESHOLD_PRODUCTION,
            "confidence_threshold_met": validation.get("overall_confidence", 0) >= 0.6,
            "recommended_action": _recommended_action(quality),
        },
    }

    logger.info(f"QA complete — score={quality:.2f}, readiness={readiness}")

    return {
        "qa_assessment": assessment,
        "final_quality_score": quality,
        "deployment_ready": assessment["deployment_readiness"]["can_deploy"],
        "current_stage": "quality_assurance",
        "agent_outputs": {
            "quality_assurance": {
                "quality_score": quality,
                "readiness": readiness,
                "recommendations_count": len(recs),
                "status": "completed",
            }
        },
    }
