"""
Indexing Agent — Extraction Node
==============================
Extracts technical specifications from PDFs using LLM with parallel processing.
Replaces ExtractionAgent.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.llm_helpers import (
    get_llm,
    invoke_llm_with_prompt,
    parse_json_response,
    truncate_to_token_limit,
)
from ..utils.prompt_loader import load_prompt
from ..utils.pdf_utils import extract_text_from_pdf
from ..state import IndexingState
from .. import config

logger = logging.getLogger(__name__)


# ── Helper functions ────────────────────────────────────────────────────────

def _assess_extraction_quality(specs: Dict[str, Any]) -> float:
    """Rule-based quality score for extracted specifications."""
    score = 0.0
    total = 0

    for field in ("parameters", "specifications", "features"):
        total += 1
        if field in specs and specs[field]:
            score += 1.0

    if "parameters" in specs:
        total += 1
        count = len(specs["parameters"])
        if count >= 10:
            score += 1.0
        elif count >= 5:
            score += 0.5

    total += 1
    if "models" in specs or "model_families" in specs:
        score += 1.0

    return score / total if total > 0 else 0.0


def _extract_from_single_pdf(
    pdf: Dict[str, Any],
    product_type: str,
    vendor: Optional[str],
    llm,
    system_prompt: str,
    user_prompt_template: str,
) -> Optional[Dict[str, Any]]:
    """Extract specifications from one PDF."""
    try:
        pdf_path = Path(pdf["download_path"])
        text = extract_text_from_pdf(pdf_path, max_pages=config.MAX_PDF_PAGES)

        if not text or len(text) < 100:
            logger.warning(f"Insufficient text from {pdf_path.name}")
            return None

        text = truncate_to_token_limit(text, max_tokens=config.MAX_EXTRACTION_TOKENS)

        user_prompt = user_prompt_template.format(
            product_type=product_type,
            vendor=vendor or pdf.get("vendor", "Unknown"),
            pdf_text=text,
        )

        response = invoke_llm_with_prompt(
            llm=llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        specs = parse_json_response(response)
        if specs:
            specs["source_pdf"] = pdf_path.name
            specs["source_url"] = pdf.get("url", "")
            specs["vendor"] = pdf.get("vendor", vendor)
            specs["extraction_confidence"] = _assess_extraction_quality(specs)
            return specs

        logger.warning(f"Failed to parse specs from {pdf_path.name}")
        return None

    except Exception as e:
        logger.error(f"Extraction error for {pdf.get('title', 'Unknown')}: {e}")
        return None


def _extract_parallel(
    pdfs: List[Dict[str, Any]],
    product_type: str,
    vendor: Optional[str],
    llm,
    system_prompt: str,
    user_prompt_template: str,
) -> List[Dict[str, Any]]:
    """Extract specifications from multiple PDFs in parallel."""
    results: List[Dict[str, Any]] = []

    eligible = [p for p in pdfs if p.get("download_path")]
    if not eligible:
        return results

    with ThreadPoolExecutor(max_workers=config.MAX_EXTRACTION_WORKERS) as executor:
        future_to_pdf = {
            executor.submit(
                _extract_from_single_pdf,
                pdf, product_type, vendor,
                llm, system_prompt, user_prompt_template,
            ): pdf
            for pdf in eligible
        }
        for future in as_completed(future_to_pdf):
            pdf_meta = future_to_pdf[future]
            try:
                specs = future.result()
                if specs:
                    results.append(specs)
                    logger.info(f"Extracted specs from: {pdf_meta.get('title', '')[:50]}")
            except Exception as e:
                logger.error(f"Extraction failed for {pdf_meta.get('title', '')}: {e}")

    logger.info(f"Extracted {len(results)} specification sets")
    return results


def _synthesize_specifications(
    all_specs: List[Dict[str, Any]],
    product_type: str,
    llm,
) -> Dict[str, Any]:
    """Merge multiple specification sets into one via LLM."""
    if not all_specs:
        return {}
    if len(all_specs) == 1:
        return all_specs[0]

    try:
        formatted = []
        for i, specs in enumerate(all_specs, 1):
            formatted.append(f"\n--- Source {i} ({specs.get('source_pdf', 'Unknown')}) ---")
            formatted.append(str(specs)[:2000])

        prompt = (
            f"Synthesize the following {len(all_specs)} specification sets into a "
            f"comprehensive, unified specification for {product_type}.\n\n"
            f"Combine overlapping information, resolve conflicts by preferring more "
            f"specific/recent data, and organise into a clear structure.\n\n"
            f"Specification Sets:\n{''.join(formatted)}\n\n"
            f"Return a JSON object with:\n"
            f"- parameters: {{name: {{value, unit, range}}, ...}}\n"
            f"- specifications: comprehensive list\n"
            f"- model_families: list of models\n"
            f"- features: key features\n"
            f"- notes: any important synthesis notes"
        )

        response = llm.invoke(prompt)
        synthesized = parse_json_response(response.content)

        if synthesized:
            synthesized["synthesis_metadata"] = {
                "source_count": len(all_specs),
                "sources": [s.get("source_pdf", "unknown") for s in all_specs],
            }
            return synthesized

        return max(all_specs, key=lambda s: s.get("extraction_confidence", 0))

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return all_specs[0]


# ── Node function ───────────────────────────────────────────────────────────

def extraction_node(state: IndexingState) -> dict:
    """
    LangGraph node — extract and synthesise specifications from downloaded PDFs.

    Reads:
        ``product_type``, ``pdf_results``, ``execution_plan``

    Writes:
        ``extracted_specs``, ``synthesized_specs``, ``current_stage``,
        ``agent_outputs``
    """
    product_type = state["product_type"]
    pdfs = state.get("pdf_results", [])
    plan = state.get("execution_plan", {})

    llm_model = plan.get("resource_allocation", {}).get("llm_model", config.DEFAULT_MODEL)
    llm = get_llm(model=llm_model, temperature=0.1)

    system_prompt = load_prompt("extraction_agent_system_prompt")
    user_prompt_template = load_prompt("extraction_agent_user_prompt")

    all_specs = _extract_parallel(
        pdfs, product_type, None,
        llm, system_prompt, user_prompt_template,
    )

    synthesized = _synthesize_specifications(all_specs, product_type, llm)

    logger.info(
        f"Extraction complete — {len(all_specs)} sets extracted, "
        f"synthesis confidence: {synthesized.get('extraction_confidence', 0):.2f}"
    )

    return {
        "extracted_specs": all_specs,
        "synthesized_specs": synthesized,
        "current_stage": "extraction",
        "agent_outputs": {
            "extraction": {
                "specs_extracted": len(all_specs),
                "synthesis_quality": synthesized.get("extraction_confidence", 0),
                "status": "completed",
            }
        },
    }
