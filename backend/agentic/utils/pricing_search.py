# agentic/utils/pricing_search.py
# ==============================================================================
# PRICING LINK UTILITIES FOR AGENTIC WORKFLOWS
# ==============================================================================
#
# Internal utilities for fetching pricing or "where to buy" links from the web
# during vendor analysis.
#
# ==============================================================================

import logging
import os
import re
from typing import Dict, Any, Optional, List
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Get API keys from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")  # Custom Search Engine ID
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")


def fetch_pricing_links_google_cse(
    vendor_name: str,
    product_name: Optional[str] = None,
    model_family: Optional[str] = None,
    timeout: int = 10
) -> List[Dict[str, Any]]:
    """
    Fetch pricing/buying links using Google Custom Search API.
    """
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        logger.warning("[PRICING_SEARCH] Google CSE credentials not configured")
        return []

    try:
        from googleapiclient.discovery import build

        # Build search query
        query_parts = [vendor_name]
        if model_family:
            query_parts.append(model_family)
        if product_name and product_name != model_family:
            query_parts.append(product_name)
        
        base_query = " ".join(query_parts)
        # Search for pricing, quote, or buy
        search_query = f"{base_query} price buy quote"

        logger.debug(f"[PRICING_SEARCH] Google CSE search for '{base_query}': {search_query}")

        # Execute search
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        result = service.cse().list(
            q=search_query,
            cx=GOOGLE_CX,
            num=3,  # Top 3 results
            safe="active" 
        ).execute()

        links = []
        for item in result.get("items", []):
            url = item.get("link", "")
            title = item.get("title", "")
            snippet = item.get("snippet", "")

            # Filter out obvious non-pricing pages if possible (e.g., PDF datasheets)
            if url.lower().endswith(".pdf"):
                continue

            links.append({
                "url": url,
                "title": title,
                "snippet": snippet,
                "source": "google_cse",
                "domain": item.get("displayLink", "")
            })

        return links

    except Exception as e:
        logger.warning(f"[PRICING_SEARCH] Google CSE search failed for {vendor_name}: {e}")
        return []


def fetch_pricing_links_serpapi(
    vendor_name: str,
    product_name: Optional[str] = None,
    model_family: Optional[str] = None,
    timeout: int = 10
) -> List[Dict[str, Any]]:
    """
    Fetch pricing links using SerpAPI (fallback).
    """
    if not SERPAPI_KEY:
        logger.warning("[PRICING_SEARCH] SerpAPI key not configured")
        return []

    try:
        # Build query
        query_parts = [vendor_name]
        if model_family:
            query_parts.append(model_family)
        if product_name and product_name != model_family:
            query_parts.append(product_name)
            
        search_query = f"{' '.join(query_parts)} price buy"

        logger.debug(f"[PRICING_SEARCH] SerpAPI search: {search_query}")

        params = {
            "q": search_query,
            "api_key": SERPAPI_KEY,
            "num": 3
        }

        response = requests.get(
            "https://serpapi.com/search",
            params=params,
            timeout=timeout
        )

        if response.status_code != 200:
            logger.warning(f"[PRICING_SEARCH] SerpAPI returned {response.status_code}")
            return []

        data = response.json()
        links = []

        # Check organic results
        for item in data.get("organic_results", []):
            url = item.get("link", "")
            if url.lower().endswith(".pdf"):
                continue
                
            links.append({
                "url": url,
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "source": "serpapi",
                "domain": item.get("displayed_link", "")
            })

        # Check for shopping results if available
        if "shopping_results" in data:
            for item in data.get("shopping_results", [])[:2]:
                 links.append({
                    "url": item.get("link", ""),
                    "title": item.get("title", ""),
                    "snippet": f"Price: {item.get('price', '')}",
                    "source": "serpapi_shopping",
                    "domain": item.get("source", "")
                })

        return links

    except Exception as e:
        logger.warning(f"[PRICING_SEARCH] SerpAPI search failed: {e}")
        return []


def fetch_best_pricing_link(
    vendor_name: str,
    product_name: Optional[str] = None,
    model_family: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch the single best pricing link for a product.
    Tries Google CSE first, then SerpAPI.
    """
    # 1. Try Google CSE
    links = fetch_pricing_links_google_cse(vendor_name, product_name, model_family)
    
    # 2. Fallback to SerpAPI
    if not links:
        links = fetch_pricing_links_serpapi(vendor_name, product_name, model_family)

    if not links:
        return {}

    # Simple heuristic: return the first result
    # Future enhancement: Use LLM to verify like images
    best = links[0]
    return {
        "pricing_url": best.get("url"),
        "pricing_source": best.get("source"),
        "pricing_title": best.get("title")
    }


def enrich_matches_with_pricing(
    matches: List[Dict[str, Any]],
    max_workers: int = 3
) -> List[Dict[str, Any]]:
    """
    Enrich a list of product matches with pricing links.
    """
    logger.info(f"[PRICING_SEARCH] Enriching {len(matches)} matches with pricing links")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_match = {}
        
        for i, match in enumerate(matches):
            vendor = match.get('vendor', '')
            model_family = match.get('modelFamily') or match.get('model_family')
            product_name = match.get('productName') or match.get('product_name')

            if not vendor:
                continue

            future = executor.submit(
                fetch_best_pricing_link,
                vendor,
                product_name,
                model_family
            )
            future_to_match[future] = i

        for future in as_completed(future_to_match):
            i = future_to_match[future]
            try:
                result = future.result()
                if result:
                    matches[i]['pricing_url'] = result.get('pricing_url', '')
                    matches[i]['pricing_source'] = result.get('pricing_source', '')
                    logger.debug(f"[PRICING_SEARCH] Found link for match #{i}: {result.get('pricing_url')}")
            except Exception as e:
                logger.warning(f"[PRICING_SEARCH] Error for match #{i}: {e}")

    return matches
