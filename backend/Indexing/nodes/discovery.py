"""
Indexing Agent — Discovery Node
============================
Discovers top vendors and product families via web search + LLM analysis.
Replaces DiscoveryAgent (dead code removed).
"""

import logging
from typing import Dict, Any, List, Optional

from ..utils.llm_helpers import get_llm, invoke_llm_with_prompt, parse_json_response
from ..utils.prompt_loader import load_prompt
from ..tools.web_search import serpapi_search, serpapi_batch_search
from ..state import IndexingState
from .. import config

logger = logging.getLogger(__name__)


# ── Helper functions ────────────────────────────────────────────────────────

def _generate_search_queries(product_type: str) -> List[str]:
    """Generate effective search queries for vendor discovery."""
    return [
        f"top manufacturers of {product_type}",
        f"leading {product_type} suppliers industrial",
        f"best {product_type} vendors industrial automation",
        f"{product_type} manufacturers list",
        f"industrial {product_type} brands",
    ]


def _validate_vendor(vendor: Dict[str, Any]) -> bool:
    """Check vendor dict has required fields."""
    required = ["vendor", "model_families"]
    return all(field in vendor and vendor[field] for field in required)


def _enrich_vendor(vendor: Dict[str, Any]) -> Dict[str, Any]:
    """Add default metadata to vendor dict."""
    vendor.setdefault("confidence", 0.8)
    vendor.setdefault("source", "web_search")
    vendor.setdefault("timestamp", None)
    return vendor


def _fallback_vendor_extraction(
    search_results: str,
    num_vendors: int,
) -> List[Dict[str, Any]]:
    """Keyword-based fallback extraction from a known vendor list."""
    found: List[Dict[str, Any]] = []
    for vendor in config.FALLBACK_VENDORS:
        if vendor.lower() in search_results.lower():
            found.append({
                "vendor": vendor,
                "model_families": [],
                "confidence": 0.5,
                "source": "fallback",
            })
        if len(found) >= num_vendors:
            break
    return found


def _discover_vendors(
    product_type: str,
    num_vendors: int,
    llm,
    system_prompt: str,
    user_prompt_template: str,
) -> List[Dict[str, Any]]:
    """Run full vendor discovery pipeline."""
    queries = _generate_search_queries(product_type)
    search_results = serpapi_batch_search(queries, max_queries=config.MAX_SEARCH_QUERIES)

    try:
        user_prompt = user_prompt_template.format(
            product_type=product_type,
            num_vendors=num_vendors,
            search_results=search_results[:8000],
        )
        response = invoke_llm_with_prompt(
            llm=llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        parsed = parse_json_response(response)

        if parsed and "vendors" in parsed:
            vendors = []
            for v in parsed["vendors"][:num_vendors]:
                if _validate_vendor(v):
                    vendors.append(_enrich_vendor(v))
            return vendors

        logger.warning("Failed to parse vendor list from LLM — using fallback")
        return _fallback_vendor_extraction(search_results, num_vendors)

    except Exception as e:
        logger.error(f"Vendor analysis failed: {e}")
        return _fallback_vendor_extraction(search_results, num_vendors)


# ── Node function ───────────────────────────────────────────────────────────

def discovery_node(state: IndexingState) -> dict:
    """
    LangGraph node — discover top vendors for the product type.

    Reads:
        ``product_type``, ``execution_plan``

    Writes:
        ``vendors``, ``current_stage``, ``agent_outputs``
    """
    product_type = state["product_type"]
    plan = state.get("execution_plan", {})
    num_vendors = (
        plan.get("resource_allocation", {}).get("num_vendors", config.DEFAULT_VENDOR_COUNT)
    )

    llm_model = plan.get("resource_allocation", {}).get("llm_model", config.DEFAULT_MODEL)
    llm = get_llm(model=llm_model, temperature=0.3)

    system_prompt = load_prompt("discovery_agent_system_prompt")
    user_prompt_template = load_prompt("discovery_agent_user_prompt")

    vendors = _discover_vendors(
        product_type=product_type,
        num_vendors=num_vendors,
        llm=llm,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
    )

    logger.info(f"Discovery complete — {len(vendors)} vendors found")

    return {
        "vendors": vendors,
        "current_stage": "discovery",
        "agent_outputs": {
            "discovery": {
                "vendors_found": len(vendors),
                "vendors": vendors,
                "status": "completed",
            }
        },
    }
