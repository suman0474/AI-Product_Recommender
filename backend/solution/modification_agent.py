# solution_N/modification_agent.py
# =============================================================================
# SOLUTION MODIFICATION AGENT
# =============================================================================
#
# Two distinct modification operations:
#
#   1. update_requirements()
#      CRUD on user input context — not on output items.
#      User changes/adds/removes requirements. Extracts the delta,
#      merges it into the original requirements, and signals the
#      identification pipeline to re-run with the updated context.
#
#   2. concise_bom()
#      Score and filter an existing BOM without re-running identification.
#      Assigns a relevance_score (0-100) to every instrument and accessory,
#      returns keep=True/False per item, lets the user trim the list.
#
# Design decision: Two plain functions — no class wrapper.
# Each operation is a stateless transformation; nothing is learned across
# calls that would justify OOP overhead.
#
# =============================================================================

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from common.services.llm.fallback import create_llm_with_fallback
from common.config import AgenticConfig
from common.prompts import SOLUTION_DEEP_AGENT_PROMPTS

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPTS
# =============================================================================

_PROMPTS = SOLUTION_DEEP_AGENT_PROMPTS
_MODIFICATION_PROMPT = _PROMPTS.get("MODIFICATION_PROMPT", "")
_BOM_CONCISENESS_PROMPT = _PROMPTS.get("BOM_CONCISENESS_PROMPT", "")


# =============================================================================
# SHARED: FETCH USER DOCUMENTS FROM AZURE BLOB
# =============================================================================

def _fetch_user_documents(session_id: str) -> str:
    """
    Fetch user-uploaded documents from Azure Blob for a given session.
    Returns a formatted context string. Never raises.
    """
    try:
        from common.config.azure_blob_config import get_azure_blob_connection
        blob_conn = get_azure_blob_connection()
        documents_collection = blob_conn["collections"]["documents"]
        user_docs = documents_collection.find({"session_id": session_id})

        if user_docs:
            docs_summary = []
            for doc in user_docs[:3]:
                name = doc.get("filename", doc.get("name", "Unknown Document"))
                summary = (
                    f"Document: {name}\n"
                    f"Type: {doc.get('document_type', 'General')}\n"
                    f"Description: {doc.get('description', '')}"
                )
                docs_summary.append(summary)

            if docs_summary:
                logger.info(
                    f"[ModifyItems] Fetched {len(docs_summary)} user documents "
                    f"for session '{session_id}'"
                )
                return "\n---\n".join(docs_summary)

    except Exception as e:
        logger.warning(f"[ModifyItems] Azure Blob document fetch failed: {e}")

    return "No user specific documents found."


# =============================================================================
# SHARED: APPLY STANDARDS RAG PER ITEM (PARALLEL)
# =============================================================================

def _apply_standards_to_single_item(
    item: Dict[str, Any],
    is_accessory: bool = False,
) -> Dict[str, Any]:
    """
    Enrich a single instrument or accessory with standards RAG data.
    Skips items that already have standards_specs populated. Never raises.
    """
    category = item.get("category", "")
    if not category:
        return item

    if item.get("standards_specs") and len(item.get("standards_specs", {})) > 0:
        return item

    try:
        from common.tools.standards_enrichment_tool import get_applicable_standards
        standards_result = get_applicable_standards(product_type=category)

        if standards_result and standards_result.get("success"):
            item["standards_specs"] = {
                "applicable_standards": standards_result.get("applicable_standards", []),
                "certifications": standards_result.get("certifications", []),
                "safety_requirements": standards_result.get("safety_requirements", {}),
            }
            item["applicable_standards"] = standards_result.get("applicable_standards", [])
            item["standards_summary"] = (
                "Applicable Standards: "
                + ", ".join(standards_result.get("applicable_standards", [])[:3])
            )

            certs = standards_result.get("certifications", [])
            if certs:
                specs = item.get("specifications", {})
                if "Certifications" not in specs:
                    specs["Certifications"] = ", ".join(certs[:3])
                    item["specifications"] = specs

            original_sample = item.get("sample_input", "")
            if "Standards:" not in original_sample and item.get("applicable_standards"):
                item["sample_input"] = (
                    f"{original_sample}. "
                    f"Standards: {', '.join(item['applicable_standards'][:2])}"
                )

            item_kind = "accessory" if is_accessory else "instrument"
            logger.debug(f"[ModifyItems] Standards applied to {item_kind}: {category}")

    except Exception as e:
        logger.warning(f"[ModifyItems] Standards RAG failed for '{category}': {e}")

    return item


def _apply_standards_parallel(
    instruments: List[Dict[str, Any]],
    accessories: List[Dict[str, Any]],
    max_workers: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Apply standards RAG to all instruments and accessories concurrently.
    Returns (updated_instruments, updated_accessories) preserving order.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        inst_futures = {
            executor.submit(_apply_standards_to_single_item, inst, False): idx
            for idx, inst in enumerate(instruments)
        }
        updated_instruments: List[Optional[Dict]] = [None] * len(instruments)
        for future in as_completed(inst_futures):
            updated_instruments[inst_futures[future]] = future.result()

        acc_futures = {
            executor.submit(_apply_standards_to_single_item, acc, True): idx
            for idx, acc in enumerate(accessories)
        }
        updated_accessories: List[Optional[Dict]] = [None] * len(accessories)
        for future in as_completed(acc_futures):
            updated_accessories[acc_futures[future]] = future.result()

    return updated_instruments, updated_accessories


# =============================================================================
# OPERATION 1: UPDATE REQUIREMENTS (CRUD on input context)
# =============================================================================

def update_requirements(
    modification_request: str,
    original_requirements: str,
    session_id: str = "default",
    memory=None,
) -> Dict[str, Any]:
    """
    CRUD operation on solution requirements — not on output items.

    The user is changing, adding, or removing requirements from their
    original solution description. This function:

      Step 1 — Fetch user documents for additional context
      Step 2 — Extract the delta: what changed vs original requirements
      Step 3 — Return updated requirements context + re-identification signal

    The caller is responsible for re-running the identification pipeline
    (INSTRUMENT_IDENTIFICATION + ACCESSORIES_IDENTIFICATION) with the
    updated requirements returned here.

    Args:
        modification_request:   Natural language change request from the user.
                                e.g. "change pressure to 100 psi and add SIL 2"
        original_requirements:  The full original solution description from the
                                first turn (stored in conversation state).
        session_id:             Used to look up session documents in Azure Blob.
        memory:                 Optional DeepAgentMemory for logging.

    Returns:
        Dict containing:
          success                bool
          operation_type         UPDATE | ADD | REMOVE | REDESIGN | MIXED
          changes_detected       list of change descriptions
          updated_requirements   merged requirements string for re-identification
          re_identification      bool — always True for requirement changes
          elapsed_ms             int
    """
    start_time = time.time()

    if not modification_request:
        return {"success": False, "error": "Modification request is required"}

    if not original_requirements:
        return {"success": False, "error": "Original requirements not provided — cannot compute delta"}

    if not _MODIFICATION_PROMPT:
        logger.error("[UpdateReqs] MODIFICATION_PROMPT not loaded")
        return {"success": False, "error": "Modification prompt not available"}

    # Step 1: Fetch user documents for additional context
    logger.info(f"[UpdateReqs] Step 1 — Fetching documents for session '{session_id}'")
    user_documents_context = _fetch_user_documents(session_id)

    # Step 2: Extract delta via LLM (text output — not JSON)
    llm = create_llm_with_fallback(
        model=AgenticConfig.FLASH_MODEL,
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    logger.info(f"[UpdateReqs] Step 2 — Extracting requirement delta...")
    try:
        prompt = ChatPromptTemplate.from_template(_MODIFICATION_PROMPT)
        chain = prompt | llm | StrOutputParser()

        raw_analysis = chain.invoke({
            "original_requirements": original_requirements,
            "modification_request": modification_request,
            "user_documents_context": user_documents_context,
        })
    except Exception as e:
        logger.error(f"[UpdateReqs] LLM delta extraction failed: {e}")
        return {
            "success": False,
            "error": f"Requirement delta extraction failed: {e}",
        }

    # Parse structured fields from the plain-text LLM output
    operation_type = _extract_field(raw_analysis, "OPERATION_TYPE")
    changes_raw = _extract_block(raw_analysis, "CHANGES_DETECTED")
    updated_requirements = _extract_block(raw_analysis, "UPDATED_REQUIREMENTS")
    re_id_needed = "YES" in _extract_field(raw_analysis, "RE_IDENTIFICATION_NEEDED").upper()

    changes_detected = [
        line.lstrip("- ").strip()
        for line in changes_raw.splitlines()
        if line.strip().startswith("-")
    ]

    # Step 3: Log to memory if provided
    if memory is not None:
        try:
            if hasattr(memory, "add_system_message"):
                memory.add_system_message(
                    f"[UpdateReqs] Requirements updated: {operation_type}. "
                    f"Changes: {'; '.join(changes_detected) or 'see delta'}. "
                    f"Re-identification required: {re_id_needed}"
                )
        except Exception as e:
            logger.debug(f"[UpdateReqs] Memory logging skipped: {e}")

    elapsed_ms = int((time.time() - start_time) * 1000)

    logger.info(
        f"[UpdateReqs] Completed in {elapsed_ms}ms — "
        f"type={operation_type}, changes={len(changes_detected)}, "
        f"re_identification={re_id_needed}"
    )

    return {
        "success": True,
        "operation_type": operation_type or "MIXED",
        "changes_detected": changes_detected,
        "updated_requirements": updated_requirements or modification_request,
        "re_identification": re_id_needed,
        "raw_analysis": raw_analysis,
        "elapsed_ms": elapsed_ms,
    }


# =============================================================================
# OPERATION 2: CONCISE BOM (Score and filter existing items)
# =============================================================================

def concise_bom(
    conciseness_request: str,
    current_instruments: List[Dict[str, Any]],
    current_accessories: List[Dict[str, Any]],
    original_requirements: str = "",
    session_id: str = "default",
    memory=None,
) -> Dict[str, Any]:
    """
    Score and filter an existing BOM — does NOT re-run identification.

    The user wants to trim or prioritize the identified instrument list.
    Each existing item receives a relevance_score (0-100) and a keep flag.
    Items with keep=False are candidates for removal.

    This operation runs on the EXISTING BOM with full conversation context —
    the original requirements and all previous turns inform the scoring.

    Args:
        conciseness_request:    What the user wants to concise/trim.
                                e.g. "remove optional items, keep only mandatory"
                                e.g. "I only have budget for 5 instruments, which are most critical?"
        current_instruments:    Existing identified instrument list.
        current_accessories:    Existing identified accessory list.
        original_requirements:  Full original requirements for context.
        session_id:             Session identifier.
        memory:                 Optional DeepAgentMemory for logging.

    Returns:
        Dict containing:
          success               bool
          scored_instruments    list with relevance_score and keep fields
          scored_accessories    list with relevance_score and keep fields
          retained_count        int — items with keep=True
          removed_count         int — items with keep=False
          conciseness_summary   str — what was kept and why
          elapsed_ms            int
    """
    start_time = time.time()

    if not conciseness_request:
        return {"success": False, "error": "Conciseness request is required"}

    if not current_instruments and not current_accessories:
        return {
            "success": False,
            "error": "No BOM to concise. Please identify instruments first.",
        }

    if not _BOM_CONCISENESS_PROMPT:
        logger.error("[ConciseBOM] BOM_CONCISENESS_PROMPT not loaded")
        return {"success": False, "error": "Conciseness prompt not available"}

    # Step 1: Fetch user documents for additional scoring context
    logger.info(f"[ConciseBOM] Step 1 — Fetching documents for session '{session_id}'")
    user_documents_context = _fetch_user_documents(session_id)

    # Build current BOM context
    current_bom = json.dumps(
        {"instruments": current_instruments, "accessories": current_accessories},
        indent=2,
    )

    llm = create_llm_with_fallback(
        model=AgenticConfig.FLASH_MODEL,
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    logger.info(
        f"[ConciseBOM] Scoring {len(current_instruments)} instruments "
        f"and {len(current_accessories)} accessories..."
    )

    try:
        prompt = ChatPromptTemplate.from_template(_BOM_CONCISENESS_PROMPT)
        parser = JsonOutputParser()
        chain = prompt | llm | parser

        result = chain.invoke({
            "current_bom": current_bom,
            "conciseness_request": conciseness_request,
            "original_requirements": original_requirements or "Not provided",
            "user_documents_context": user_documents_context,
        })
    except Exception as e:
        logger.error(f"[ConciseBOM] LLM scoring failed: {e}")
        return {
            "success": False,
            "scored_instruments": current_instruments,
            "scored_accessories": current_accessories,
            "error": f"BOM scoring failed: {e}",
        }

    # Apply standards enrichment only to items that will be KEPT
    kept_instruments = [
        i for i in result.get("scored_instruments", []) if i.get("keep", True)
    ]
    kept_accessories = [
        a for a in result.get("scored_accessories", []) if a.get("keep", True)
    ]

    logger.info("[ConciseBOM] Applying standards to retained items...")
    enriched_instruments, enriched_accessories = _apply_standards_parallel(
        kept_instruments, kept_accessories
    )

    retained_count = result.get("retained_count", len(kept_instruments))
    removed_count = result.get("removed_count", 0)

    # Log to memory if provided
    if memory is not None:
        try:
            if hasattr(memory, "add_system_message"):
                memory.add_system_message(
                    f"[ConciseBOM] BOM concised: {retained_count} kept, "
                    f"{removed_count} removed. "
                    f"Request: {conciseness_request[:80]}"
                )
        except Exception as e:
            logger.debug(f"[ConciseBOM] Memory logging skipped: {e}")

    elapsed_ms = int((time.time() - start_time) * 1000)

    # Merge enriched kept items back with scored-but-removed items
    all_scored_instruments = result.get("scored_instruments", [])
    all_scored_accessories = result.get("scored_accessories", [])

    # Replace kept items with enriched versions (preserve scores on removed items)
    kept_idx = 0
    for item in all_scored_instruments:
        if item.get("keep", True) and kept_idx < len(enriched_instruments):
            item.update(enriched_instruments[kept_idx])
            kept_idx += 1

    kept_idx = 0
    for item in all_scored_accessories:
        if item.get("keep", True) and kept_idx < len(enriched_accessories):
            item.update(enriched_accessories[kept_idx])
            kept_idx += 1

    logger.info(
        f"[ConciseBOM] Completed in {elapsed_ms}ms — "
        f"{retained_count} kept, {removed_count} removed"
    )

    return {
        "success": True,
        "scored_instruments": all_scored_instruments,
        "scored_accessories": all_scored_accessories,
        "retained_count": retained_count,
        "removed_count": removed_count,
        "conciseness_summary": result.get("conciseness_summary", ""),
        "elapsed_ms": elapsed_ms,
    }


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _extract_field(text: str, label: str) -> str:
    """Extract a single-line field value from structured plain-text LLM output."""
    for line in text.splitlines():
        if line.strip().upper().startswith(label.upper() + ":"):
            return line.split(":", 1)[-1].strip()
    return ""


def _extract_block(text: str, label: str) -> str:
    """
    Extract a multi-line block between two section labels from plain-text output.
    Returns content between `label:` and the next all-caps label.
    """
    lines = text.splitlines()
    collecting = False
    block = []
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith(label.upper() + ":"):
            collecting = True
            remainder = line.split(":", 1)[-1].strip()
            if remainder:
                block.append(remainder)
            continue
        if collecting:
            # Stop at next uppercase label (e.g. CHANGES_DETECTED:)
            if stripped and stripped.endswith(":") and stripped[:-1].isupper():
                break
            block.append(line)
    return "\n".join(block).strip()
