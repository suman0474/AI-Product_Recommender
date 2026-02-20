"""
Indexing Agent — Planning Node
===========================
Complexity analysis, execution plan creation, and resource allocation.
Replaces MetaOrchestratorAgent (dead code removed).
"""

import logging
from typing import Dict, Any, List, Optional

from ..utils.llm_helpers import get_llm, parse_json_response
from ..utils.prompt_loader import load_prompt
from ..state import IndexingState
from .. import config

logger = logging.getLogger(__name__)


# ── Helper functions ────────────────────────────────────────────────────────

def _analyze_complexity(
    product_type: str,
    context: Optional[Dict[str, Any]],
    llm,
) -> Dict[str, Any]:
    """Analyse task complexity using heuristics + LLM."""
    analysis: Dict[str, Any] = {
        "level": "moderate",
        "factors": [],
        "confidence": 0.7,
        "reasoning": "",
    }

    # Heuristic: product-type specificity
    if len(product_type.split()) > 3:
        analysis["factors"].append("Highly specific product type")
        analysis["level"] = "complex"

    # Heuristic: existing data availability
    if context and context.get("existing_data"):
        analysis["factors"].append("Existing data available")
    else:
        analysis["factors"].append("No existing data — full discovery needed")
        analysis["level"] = "complex"

    # Heuristic: industrial domain
    industrial_keywords = ["industrial", "process", "control", "instrumentation"]
    if any(kw in product_type.lower() for kw in industrial_keywords):
        analysis["factors"].append("Industrial domain — requires specialised knowledge")

    # LLM refinement
    try:
        prompt = (
            f"Analyze the complexity of discovering and indexing specifications for:\n"
            f"Product Type: {product_type}\n"
            f"Context: {context or 'None'}\n\n"
            f"Consider:\n"
            f"- Availability of public specifications\n"
            f"- Number of major vendors\n"
            f"- Standardization level\n"
            f"- Technical complexity\n\n"
            f"Return JSON with:\n"
            f"- complexity_level: (simple/moderate/complex/very_complex)\n"
            f"- reasoning: detailed explanation\n"
            f"- confidence: 0.0-1.0\n"
            f"- key_challenges: list of challenges"
        )
        response = llm.invoke(prompt)
        llm_analysis = parse_json_response(response.content)
        if llm_analysis:
            analysis["reasoning"] = llm_analysis.get("reasoning", "")
            analysis["confidence"] = llm_analysis.get("confidence", 0.7)
    except Exception as e:
        logger.warning(f"LLM complexity analysis failed: {e}")

    return analysis


def _workflow_steps(complexity_level: str) -> List[Dict[str, Any]]:
    """Return ordered workflow steps tuned for *complexity_level*."""
    if complexity_level == "simple":
        return [
            {"agent": "discovery", "action": "discover_vendors", "priority": 1},
            {"agent": "search", "action": "search_pdfs", "priority": 2, "parallel_per_vendor": False},
            {"agent": "extraction", "action": "extract_specs", "priority": 3},
            {"agent": "schema_retrieval", "action": "retrieve_schema", "priority": 4},
            {"agent": "quality_assurance", "action": "quick_review", "priority": 5},
        ]
    if complexity_level == "moderate":
        return [
            {"agent": "discovery", "action": "discover_vendors", "priority": 1, "num_vendors": 5},
            {"agent": "search", "action": "multi_tier_search", "priority": 2, "parallel_per_vendor": True},
            {"agent": "extraction", "action": "parallel_extraction", "priority": 3},
            {"agent": "validation", "action": "cross_validate", "priority": 4},
            {"agent": "schema_retrieval", "action": "retrieve_schema", "priority": 5},
            {"agent": "quality_assurance", "action": "full_assessment", "priority": 6},
        ]
    # complex / very_complex
    return [
        {"agent": "discovery", "action": "comprehensive_vendor_discovery", "priority": 1, "num_vendors": 7},
        {"agent": "search", "action": "exhaustive_multi_tier_search", "priority": 2, "parallel_per_vendor": True},
        {"agent": "extraction", "action": "parallel_extraction_with_synthesis", "priority": 3},
        {"agent": "validation", "action": "multi_source_validation", "priority": 4},
        {"agent": "schema_retrieval", "action": "detailed_schema_design", "priority": 5},
        {"agent": "validation", "action": "schema_validation", "priority": 6},
        {"agent": "quality_assurance", "action": "comprehensive_review", "priority": 7},
        {"agent": "schema_retrieval", "action": "refinement_if_needed", "priority": 8, "conditional": True},
    ]


def _create_execution_plan(
    product_type: str,
    complexity: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a full execution plan dict."""
    level = complexity["level"]
    is_heavy = level in ("complex", "very_complex")

    return {
        "strategy": config.COMPLEXITY_STRATEGIES.get(level, "standard_discovery"),
        "steps": _workflow_steps(level),
        "parallelization": {
            "vendor_processing": True,
            "pdf_download": True,
            "pdf_extraction": True,
            "max_workers": 3 if is_heavy else 2,
        },
        "fallback_options": [
            "Use broader search terms if no PDFs found",
            "Expand to alternative vendors",
            "Use LLM knowledge for basic schema if no sources available",
        ],
        "resource_allocation": {
            "llm_model": config.REASONING_MODEL if is_heavy else config.DEFAULT_MODEL,
            "max_parallel_workers": 3 if is_heavy else 2,
            "timeout_multiplier": 1.5 if is_heavy else 1.0,
            "retry_attempts": 3 if is_heavy else 2,
            "cache_strategy": "aggressive" if level == "simple" else "moderate",
            "num_vendors": 7 if is_heavy else 5,
        },
        "estimated_duration": config.DURATION_ESTIMATES.get(
            level, config.DURATION_ESTIMATES["moderate"]
        ),
    }


# ── Node function ───────────────────────────────────────────────────────────

def planning_node(state: IndexingState) -> dict:
    """
    LangGraph node — analyse complexity and build an execution plan.

    Reads:
        ``product_type``, ``context``

    Writes:
        ``execution_plan``, ``complexity_level``, ``current_stage``,
        ``agent_outputs``
    """
    product_type = state["product_type"]
    context = state.get("context", {})

    llm = get_llm(model=config.REASONING_MODEL, temperature=0.3)

    complexity = _analyze_complexity(product_type, context, llm)
    plan = _create_execution_plan(product_type, complexity)

    logger.info(
        f"Planning complete — complexity={complexity['level']}, "
        f"steps={len(plan['steps'])}"
    )

    return {
        "execution_plan": {**plan, "complexity": complexity},
        "complexity_level": complexity["level"],
        "current_stage": "planning",
        "agent_outputs": {
            "planning": {
                "plan_created": True,
                "complexity": complexity["level"],
                "estimated_duration": plan["estimated_duration"]["typical"],
                "status": "completed",
            }
        },
    }
