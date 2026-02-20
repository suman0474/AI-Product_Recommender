"""
Indexing Agent — State Definition
==============================
LangGraph-compatible state with Annotated reducers for automatic merging.
"""

from typing import Annotated, TypedDict, Dict, Any, List
import operator


# ── Reducer helpers ─────────────────────────────────────────────────────────

def _merge_dicts(existing: dict, update: dict) -> dict:
    """Shallow-merge two dicts (used as a LangGraph reducer)."""
    return {**existing, **update}


def _replace(existing, update):
    """Simple replace reducer — latest value wins."""
    return update


# ── State ───────────────────────────────────────────────────────────────────

class IndexingState(TypedDict, total=False):
    """
    Main workflow state for the Indexing Agent.

    Fields annotated with ``operator.add`` are *appendable*: node return
    values extend the existing list.  Fields annotated with
    ``_merge_dicts`` are *mergeable*: node return values are shallow-merged
    into the existing dict.
    """

    # ── Input (set once at invocation) ──────────────────────────────────
    product_type: str
    context: Dict[str, Any]
    session_id: str

    # ── Planning stage ──────────────────────────────────────────────────
    execution_plan: Dict[str, Any]
    complexity_level: str

    # ── Discovery stage ─────────────────────────────────────────────────
    vendors: List[Dict[str, Any]]

    # ── Search stage ────────────────────────────────────────────────────
    pdf_results: Annotated[list, operator.add]

    # ── Extraction stage ────────────────────────────────────────────────
    extracted_specs: Annotated[list, operator.add]
    synthesized_specs: Dict[str, Any]

    # ── Validation stage ────────────────────────────────────────────────
    validation_results: Dict[str, Any]

    # ── Schema generation stage ─────────────────────────────────────────
    generated_schema: Dict[str, Any]
    schema_completeness: Dict[str, Any]
    schema_source: str  # "azure" | "generated" | "refined" | "default"

    # ── Quality-assurance stage ─────────────────────────────────────────
    qa_assessment: Dict[str, Any]
    final_quality_score: float
    deployment_ready: bool

    # ── Workflow tracking ───────────────────────────────────────────────
    agent_outputs: Annotated[dict, _merge_dicts]
    errors: Annotated[list, operator.add]
    current_stage: str
    refinement_count: int
