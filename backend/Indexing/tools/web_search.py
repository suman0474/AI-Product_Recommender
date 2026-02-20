"""
Indexing Agent — Web Search Tools
==============================
Reusable search functions wrapping SerpAPI and GoogleSerper.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def serpapi_search(query: str) -> str:
    """
    Run a single query via SerpAPIWrapper.

    Args:
        query: Search query string.

    Returns:
        Search results as a string, or empty string on failure.
    """
    try:
        from langchain_community.utilities import SerpAPIWrapper
        search = SerpAPIWrapper()
        return search.run(query)
    except Exception as e:
        logger.warning(f"SerpAPI search failed for '{query}': {e}")
        return ""


def serpapi_batch_search(queries: List[str], max_queries: int = 3) -> str:
    """
    Execute multiple SerpAPI queries and aggregate results.

    Args:
        queries: List of search queries.
        max_queries: Maximum queries to execute (rate-limit guard).

    Returns:
        Aggregated results as a single string.
    """
    all_results = []
    for query in queries[:max_queries]:
        result = serpapi_search(query)
        if result:
            all_results.append(f"Query: {query}\n{result}\n")
    return "\n---\n".join(all_results)


def serper_search(query: str) -> str:
    """
    Run a single query via GoogleSerperAPIWrapper (text results).

    Args:
        query: Search query string.

    Returns:
        Search results as a string, or empty string on failure.
    """
    try:
        from langchain_community.utilities import GoogleSerperAPIWrapper
        serper = GoogleSerperAPIWrapper()
        return serper.run(query)
    except Exception as e:
        logger.warning(f"Serper search failed for '{query}': {e}")
        return ""


def serper_search_results(query: str) -> Dict[str, Any]:
    """
    Run a single query via GoogleSerperAPIWrapper, returning structured results.

    Args:
        query: Search query string.

    Returns:
        Dict with 'organic' results list, or empty dict on failure.
    """
    try:
        from langchain_community.utilities import GoogleSerperAPIWrapper
        serper = GoogleSerperAPIWrapper()
        return serper.results(query)
    except Exception as e:
        logger.warning(f"Serper results failed for '{query}': {e}")
        return {}
