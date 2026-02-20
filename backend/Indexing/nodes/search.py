"""
Indexing Agent — Search Node
=========================
Multi-tier PDF search with parallel download.
Replaces SearchAgent.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..tools.web_search import serper_search_results
from ..tools.pdf_tools import download_pdfs_parallel
from ..utils.pdf_utils import rank_pdfs_by_relevance
from ..state import IndexingState
from .. import config

logger = logging.getLogger(__name__)

_DOWNLOAD_DIR = Path(__file__).parent.parent / "temp_pdfs"


# ── Helper functions ────────────────────────────────────────────────────────

def _get_vendor_domain(vendor: str) -> str:
    """Map vendor name to its primary domain."""
    vendor_lower = vendor.lower()
    for key, domain in config.VENDOR_DOMAIN_MAP.items():
        if key in vendor_lower:
            return domain
    return f"{vendor.split()[0].lower()}.com"


def _extract_pdfs_from_results(
    results: Dict[str, Any],
    tier: int,
) -> List[Dict[str, Any]]:
    """Filter PDF URLs from search results."""
    pdfs: List[Dict[str, Any]] = []
    for item in results.get("organic", []):
        url = item.get("link", "")
        if url.lower().endswith(".pdf"):
            pdfs.append({
                "url": url,
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "tier": tier,
                "source": "serper",
            })
    return pdfs


def _deduplicate_pdfs(pdfs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicates by URL."""
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for pdf in pdfs:
        if pdf["url"] not in seen:
            seen.add(pdf["url"])
            unique.append(pdf)
    return unique


def _tier1_vendor_site_search(
    product_type: str,
    vendor: str,
    model_families: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Tier 1: vendor-site-specific PDF search."""
    domain = _get_vendor_domain(vendor)
    queries = [
        f"site:{domain} {product_type} datasheet filetype:pdf",
        f"site:{domain} {product_type} specification filetype:pdf",
    ]
    if model_families:
        for family in model_families[:2]:
            queries.append(f"site:{domain} {family} datasheet filetype:pdf")

    pdfs: List[Dict[str, Any]] = []
    for query in queries[:3]:
        try:
            results = serper_search_results(query)
            pdfs.extend(_extract_pdfs_from_results(results, tier=1))
        except Exception as e:
            logger.warning(f"Tier-1 search failed for '{query}': {e}")
    return _deduplicate_pdfs(pdfs)


def _tier2_web_search(
    product_type: str,
    vendor: str,
    model_families: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Tier 2: general web search."""
    queries = [
        f"{vendor} {product_type} datasheet pdf",
        f"{vendor} {product_type} technical specification pdf",
    ]
    pdfs: List[Dict[str, Any]] = []
    for query in queries:
        try:
            results = serper_search_results(query)
            pdfs.extend(_extract_pdfs_from_results(results, tier=2))
        except Exception as e:
            logger.warning(f"Tier-2 search failed for '{query}': {e}")
    return _deduplicate_pdfs(pdfs)


def _tier3_fallback_search(
    product_type: str,
    vendor: str,
) -> List[Dict[str, Any]]:
    """Tier 3: broad fallback search."""
    pdfs: List[Dict[str, Any]] = []
    try:
        results = serper_search_results(f"{product_type} specification pdf")
        pdfs.extend(_extract_pdfs_from_results(results, tier=3))
    except Exception as e:
        logger.error(f"Tier-3 search error: {e}")
    return _deduplicate_pdfs(pdfs)


def _search_pdfs_for_vendor(
    product_type: str,
    vendor: str,
    model_families: Optional[List[str]],
    max_pdfs: int = config.MAX_PDFS_PER_VENDOR,
) -> List[Dict[str, Any]]:
    """Run multi-tier search for a single vendor."""
    all_pdfs: List[Dict[str, Any]] = []

    tier1 = _tier1_vendor_site_search(product_type, vendor, model_families)
    all_pdfs.extend(tier1)
    logger.info(f"Tier 1 found {len(tier1)} PDFs for {vendor}")

    if len(all_pdfs) < max_pdfs:
        tier2 = _tier2_web_search(product_type, vendor, model_families)
        all_pdfs.extend(tier2)
        logger.info(f"Tier 2 found {len(tier2)} PDFs for {vendor}")

    if len(all_pdfs) < 2:
        tier3 = _tier3_fallback_search(product_type, vendor)
        all_pdfs.extend(tier3)
        logger.info(f"Tier 3 found {len(tier3)} PDFs for {vendor}")

    ranked = rank_pdfs_by_relevance(all_pdfs, product_type, vendor)
    return ranked[:max_pdfs]


# ── Node function ───────────────────────────────────────────────────────────

def search_node(state: IndexingState) -> dict:
    """
    LangGraph node — search and download PDFs for all discovered vendors.

    Reads:
        ``product_type``, ``vendors``, ``execution_plan``

    Writes:
        ``pdf_results``, ``current_stage``, ``agent_outputs``
    """
    product_type = state["product_type"]
    vendors = state.get("vendors", [])
    plan = state.get("execution_plan", {})
    max_workers = plan.get("parallelization", {}).get("max_workers", config.MAX_DOWNLOAD_WORKERS)

    all_pdfs: List[Dict[str, Any]] = []
    for vendor_info in vendors:
        vendor = vendor_info.get("vendor", "")
        model_families = vendor_info.get("model_families", [])
        pdfs = _search_pdfs_for_vendor(product_type, vendor, model_families)
        for pdf in pdfs:
            pdf["vendor"] = vendor
        all_pdfs.extend(pdfs)

    # Download in parallel
    downloaded = download_pdfs_parallel(all_pdfs, _DOWNLOAD_DIR, max_workers=max_workers)

    logger.info(f"Search complete — {len(downloaded)}/{len(all_pdfs)} PDFs downloaded")

    return {
        "pdf_results": downloaded,
        "current_stage": "search",
        "agent_outputs": {
            "search": {
                "pdfs_found": len(all_pdfs),
                "pdfs_downloaded": len(downloaded),
                "status": "completed",
            }
        },
    }
