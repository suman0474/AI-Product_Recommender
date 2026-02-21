# agentic/standards_rag_enrichment.py
# =============================================================================
# STANDARDS RAG ENRICHMENT UTILITIES
# =============================================================================
#
# Shared utility functions for enriching identified instruments/accessories
# with Standards RAG data. Used by both instrument_identifier_workflow and
# Solution Deep Agent (solution).
#
# PERFORMANCE OPTIMIZATIONS:
# - Parallel processing with ThreadPoolExecutor (items processed simultaneously)
# - Product type caching (avoid duplicate queries for same product type)
# - Singleton agent pattern (avoid re-initialization overhead)
#
# NOTE: This module now uses consolidated functions from the standards submodule.
# See agentic/workflows/standards_rag/standards/ for the canonical implementations.
#
# =============================================================================

import logging
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# CONSOLIDATED IMPORTS (Single Source of Truth)
# =============================================================================
# Import from the consolidated standards module to avoid duplication.
# The standards/ submodule contains the canonical implementations.

from ..shared.constants import MIN_STANDARDS_SPECS_COUNT, MAX_SPECS_PER_ITEM
from ..shared.detector import classify_domain, get_routed_documents, is_standards_related_question
from ..shared.cache import (
    get_cached_standards,
    cache_standards,
    clear_standards_cache,
    get_cache_key,
)
from ..shared.enrichment import normalize_category as _normalize_category

logger = logging.getLogger(__name__)

# =============================================================================
# BACKWARDS COMPATIBILITY ALIASES
# =============================================================================
# These aliases maintain backwards compatibility for existing code that imports
# the old function names from this module.

def _get_cached_standards(product_type: str) -> Optional[Dict[str, Any]]:
    """Backwards-compatible alias for get_cached_standards."""
    return get_cached_standards(product_type)


def _cache_standards(product_type: str, result: Dict[str, Any]):
    """Backwards-compatible alias for cache_standards."""
    cache_standards(product_type, result)


# =============================================================================
# PARALLEL ENRICHMENT FUNCTIONS
# =============================================================================

def _enrich_single_item(
    item: Dict[str, Any],
    product_type_context: Optional[str],
    top_k: int
) -> Dict[str, Any]:
    """
    Enrich a single item with standards. Used by parallel executor.

    OPTIMIZATION: Uses domain-based document routing to query only relevant
    standards documents instead of all 12+ documents.

    Args:
        item: Item to enrich
        product_type_context: Context product type
        top_k: Number of standards documents to retrieve

    Returns:
        Enriched item with standards_info
    """
    try:
        from common.tools.standards_enrichment_tool import get_applicable_standards

        # Determine the product type for this item
        item_product_type = (
            item.get("category") or
            item.get("product_name") or
            item.get("name") or
            product_type_context or
            "industrial instrument"
        )

        # ╔═══════════════════════════════════════════════════════════════════════╗
        # ║     DOMAIN-BASED ROUTING (Performance Optimization)                   ║
        # ║     Classify product into domain(s) and route to relevant docs only   ║
        # ╚═══════════════════════════════════════════════════════════════════════╝
        user_specs = str(item.get("specifications", {}))
        domains = classify_domain(user_specs, item_product_type, max_domains=3)
        routed_documents = get_routed_documents(domains)

        logger.info(
            f"[StandardsEnrichment] Domain routing for '{item_product_type}': "
            f"domains={[d.value for d in domains]}, "
            f"docs={routed_documents} (vs 12+ without routing)"
        )

        # Check cache first (MAJOR PERF OPTIMIZATION)
        # Include routed_documents in cache key for source-specific caching
        cache_key = f"{item_product_type}|{','.join(sorted(routed_documents)) if routed_documents else 'all'}"
        cached_result = _get_cached_standards(cache_key)
        if cached_result is not None:
            logger.info(f"[StandardsEnrichment] Cache HIT for: {cache_key}")
            item["standards_info"] = cached_result.copy()
            item["standards_info"]["from_cache"] = True
            if cached_result.get("applicable_standards"):
                item["normalized_category"] = _normalize_category(
                    item_product_type,
                    cached_result.get("applicable_standards", [])
                )
            return item

        logger.info(f"[StandardsEnrichment] Querying standards for: {item_product_type} (filtered to {len(routed_documents)} docs)")

        # ╔═══════════════════════════════════════════════════════════════════════╗
        # ║     PASS ROUTED DOCUMENTS TO RAG (Critical Filter Integration)        ║
        # ║     This filters Pinecone queries to only relevant domain documents   ║
        # ╚═══════════════════════════════════════════════════════════════════════╝
        standards_result = get_applicable_standards(
            product_type=item_product_type,
            top_k=top_k,
            source_filter=routed_documents if routed_documents else None
        )
        
        if standards_result.get("success"):
            # Build standards_info
            standards_info = {
                "applicable_standards": standards_result.get("applicable_standards", []),
                "certifications": standards_result.get("certifications", []),
                "safety_requirements": standards_result.get("safety_requirements", {}),
                "communication_protocols": standards_result.get("communication_protocols", []),
                "environmental_requirements": standards_result.get("environmental_requirements", {}),
                "confidence": standards_result.get("confidence", 0.0),
                "sources": standards_result.get("sources", []),
                "enrichment_status": "success",
                "from_cache": False
            }
            
            item["standards_info"] = standards_info

            # Cache for future use (use same key as lookup)
            _cache_standards(cache_key, standards_info)
            
            # Add normalized category if available from standards
            if standards_result.get("applicable_standards"):
                item["normalized_category"] = _normalize_category(
                    item_product_type, 
                    standards_result.get("applicable_standards", [])
                )
                
            logger.info(
                f"[StandardsEnrichment] Enriched {item_product_type}: "
                f"{len(standards_result.get('applicable_standards', []))} standards, "
                f"{len(standards_result.get('certifications', []))} certifications"
            )
        else:
            # Mark as failed but continue
            item["standards_info"] = {
                "enrichment_status": "failed",
                "error": standards_result.get("error", "Unknown error"),
                "from_cache": False
            }
            logger.warning(
                f"[StandardsEnrichment] Failed to enrich {item_product_type}: "
                f"{standards_result.get('error')}"
            )
                
    except Exception as e:
        logger.error(f"[StandardsEnrichment] Error enriching item: {e}")
        item["standards_info"] = {
            "enrichment_status": "error",
            "error": str(e),
            "from_cache": False
        }
    
    return item


def enrich_identified_items_with_standards(
    items: List[Dict[str, Any]],
    product_type: Optional[str] = None,
    domain: Optional[str] = None,
    top_k: int = 3,
    max_workers: int = 5
) -> List[Dict[str, Any]]:
    """
    Enriches identified instruments/accessories with Standards RAG data.
    
    PERFORMANCE OPTIMIZATIONS:
    - Uses parallel processing (ThreadPoolExecutor) for multiple items
    - Caches results by product type to avoid duplicate queries
    - Uses singleton agent for LLM/vector store connections
    
    For each item:
    1. Check cache for previously queried product types
    2. Query Standards RAG for applicable standards (in parallel)
    3. Add normalized category, certifications, and compliance info
    4. Return enriched item list
    
    Args:
        items: List of identified instruments/accessories
        product_type: Optional product type context (e.g., "pressure transmitter")
        domain: Optional industry domain (e.g., "Oil & Gas", "Chemical")
        top_k: Number of standards documents to retrieve per query
        max_workers: Maximum parallel threads for enrichment
        
    Returns:
        List of enriched items with standards_info field added
    """
    if not items:
        logger.info("[StandardsEnrichment] No items to enrich")
        return items
    
    import time
    start_time = time.time()
    
    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║     🔵 PARALLEL STANDARDS ENRICHMENT (Performance Optimized)  🔵     ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    logger.info("="*70)
    logger.info("PARALLEL STANDARDS ENRICHMENT STARTING")
    logger.info(f"   Items to enrich: {len(items)}")
    logger.info(f"   Max workers: {max_workers}")
    try:
        from ..shared.cache import get_standards_cache_stats
        _cache_size = get_standards_cache_stats().get("size", 0)
    except Exception:
        _cache_size = "n/a"
    logger.info(f"   Cache size: {_cache_size}")
    logger.info("="*70)
    
    try:
        # Determine optimal worker count
        actual_workers = min(len(items), max_workers)
        
        if len(items) == 1:
            # Single item - no parallelization needed
            enriched_items = [_enrich_single_item(items[0], product_type, top_k)]
        else:
            # Multiple items - use parallel processing
            enriched_items = []
            
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                # Submit all items for parallel processing
                future_to_item = {
                    executor.submit(_enrich_single_item, item.copy(), product_type, top_k): idx 
                    for idx, item in enumerate(items)
                }
                
                # Collect results as they complete
                results = [None] * len(items)
                for future in as_completed(future_to_item):
                    idx = future_to_item[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error(f"[StandardsEnrichment] Future error for item {idx}: {e}")
                        # Return original item with error status
                        original_item = items[idx].copy()
                        original_item["standards_info"] = {
                            "enrichment_status": "error",
                            "error": str(e)
                        }
                        results[idx] = original_item
                
                enriched_items = results
        
        elapsed_time = time.time() - start_time
        cache_hits = sum(1 for item in enriched_items if item.get("standards_info", {}).get("from_cache", False))
        
        logger.info("="*70)
        logger.info("PARALLEL STANDARDS ENRICHMENT COMPLETED")
        logger.info(f"   Total time: {elapsed_time:.2f}s")
        logger.info(f"   Items enriched: {len(enriched_items)}")
        logger.info(f"   Cache hits: {cache_hits}")
        logger.info(f"   Avg time per item: {elapsed_time/len(items):.2f}s")
        logger.info("="*70)
        
        return enriched_items
        
    except ImportError as e:
        logger.error(f"[StandardsEnrichment] Import error: {e}")
        # Return items unchanged if import fails
        for item in items:
            item["standards_info"] = {
                "enrichment_status": "unavailable",
                "error": "Standards enrichment tool not available"
            }
        return items
    except Exception as e:
        logger.error(f"[StandardsEnrichment] Unexpected error: {e}", exc_info=True)
        for item in items:
            item["standards_info"] = {
                "enrichment_status": "error",
                "error": str(e)
            }
        return items


def validate_items_against_domain_standards(
    items: List[Dict[str, Any]],
    domain: str,
    safety_requirements: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Validates identified items against domain-specific standards.

    🔥 FIX #4: VALIDATION CACHE OPTIMIZATION 🔥
    Uses cached standards_specifications from Phase 3 enrichment instead of
    re-running RAG. This saves 2000+ seconds by avoiding duplicate RAG queries.

    Used by Solution Deep Agent to validate compatibility with industry standards.

    Args:
        items: List of identified instruments/accessories
        domain: Industry domain (e.g., "Oil & Gas", "Chemical", "Pharma")
        safety_requirements: Safety requirements from solution analysis

    Returns:
        Validation result with compliance status and issues
    """
    logger.info(f"[StandardsValidation] Validating {len(items)} items for domain: {domain}")

    validation_result = {
        "is_compliant": True,
        "compliance_issues": [],
        "recommendations": [],
        "domain": domain,
        "cache_hits": 0,
        "cache_misses": 0
    }

    try:
        from common.tools.standards_enrichment_tool import validate_requirements_against_standards

        for item in items:
            item_product_type = item.get("category") or item.get("name") or "instrument"
            item_name = item.get("name") or item.get("product_name") or "Unknown"

            # 🔥 FIX #2 CRITICAL: Verify Phase3 enrichment actually reached minimum specs 🔥
            # Previously: SKIPPED RAG for ANY item with enrichment_source == "phase3_optimized"
            # Problem: Items marked phase3_optimized but only had 10-15 specs (< 30 minimum)
            # Fix: Check total_specs_count >= MIN_STANDARDS_SPECS_COUNT before skipping
            if item.get("enrichment_source") == "phase3_optimized":
                total_specs_count = item.get("standards_info", {}).get("total_specs_count", 0)

                # FIX #3: DEFENSIVE: Fallback counting if total_specs_count is not set
                if total_specs_count == 0:
                    combined_specs = item.get("combined_specifications", {})
                    total_specs_count = len([v for v in combined_specs.values()
                                            if v and str(v).lower() not in ["null", "none", ""]])
                    logger.warning(
                        f"[StandardsValidation] Fallback counting for '{item_name}': "
                        f"standards_info.total_specs_count was 0, counted {total_specs_count} from combined_specifications"
                    )

                # CRITICAL: Verify specs meet minimum before skipping RAG
                if total_specs_count >= MIN_STANDARDS_SPECS_COUNT:
                    logger.info(
                        f"[StandardsValidation] SKIPPING RAG for {item_name} - enriched via phase3 "
                        f"with {total_specs_count} specs (saves 40-50s!)"
                    )
                    validation_result["cache_hits"] += 1

                    # Just do lightweight validation, no RAG needed
                    cached_specs = item.get("standards_specifications", {})
                    if cached_specs.get("applicable_standards"):
                        logger.debug(
                            f"[StandardsValidation] {item_name} is phase3 optimized with "
                            f"{len(cached_specs.get('applicable_standards', []))} standards"
                        )
                    continue
                else:
                    # CRITICAL FIX: Phase3 marked but insufficient specs - need RAG
                    logger.warning(
                        f"[StandardsValidation] Phase3 enrichment incomplete for {item_name}: "
                        f"only {total_specs_count} specs (minimum: {MIN_STANDARDS_SPECS_COUNT}). "
                        f"Will run RAG for completion."
                    )
                    # Fall through to RAG path below

            # 🔥 FIX #3 CRITICAL: Verify cached standards are sufficient before using them 🔥
            # Previously: Used ANY cached standards_specifications without checking spec count
            # Problem: Items with 10 standards specs treated as complete when they need 30+
            # Fix: Check total_specs_count >= MIN_STANDARDS_SPECS_COUNT before accepting cache
            if item.get("standards_specifications"):
                total_specs_count = item.get("standards_info", {}).get("total_specs_count", 0)
                cached_specs = item.get("standards_specifications", {})

                # FIX #3: DEFENSIVE: Fallback counting if total_specs_count is not set
                if total_specs_count == 0:
                    # Try combined_specifications first (most reliable)
                    combined_specs = item.get("combined_specifications", {})
                    if combined_specs:
                        total_specs_count = len([v for v in combined_specs.values()
                                                if v and str(v).lower() not in ["null", "none", ""]])
                    else:
                        # Fallback to standards_specifications length
                        total_specs_count = len([v for v in cached_specs.values()
                                                if v and str(v).lower() not in ["null", "none", ""]])
                    logger.warning(
                        f"[StandardsValidation] Fallback counting for cached '{item_name}': "
                        f"standards_info.total_specs_count was 0, counted {total_specs_count} from specs"
                    )

                # CRITICAL: Verify cached specs are sufficient
                if total_specs_count >= MIN_STANDARDS_SPECS_COUNT:
                    logger.info(
                        f"[StandardsValidation] Using CACHED standards for {item_name} - "
                        f"{total_specs_count} specs (lightweight validation only)"
                    )
                    validation_result["cache_hits"] += 1

                    requirements = {
                        "product_type": item_product_type,
                        "specifications": cached_specs,
                    }

                    # Add safety requirements if provided
                    # In Solution Deep Agent (enrichment node):
                    if safety_requirements:
                        if safety_requirements.get("sil_level"):
                            requirements["sil_level"] = safety_requirements["sil_level"]
                        if safety_requirements.get("hazardous_area"):
                            requirements["hazardous_area"] = True
                        if safety_requirements.get("atex_zone"):
                            requirements["atex_zone"] = safety_requirements["atex_zone"]

                    # Simple validation logic (no RAG needed!)
                    try:
                        # Check if certifications are specified
                        user_certs = cached_specs.get("certifications", [])
                        if not user_certs:
                            validation_result["recommendations"].append({
                                "item": item_name,
                                "category": "certifications",
                                "recommendation": "Verify required certifications are specified"
                            })

                        # Mark as compliant if we have standards info
                        if cached_specs.get("applicable_standards"):
                            logger.debug(f"[StandardsValidation] {item_name} has {len(cached_specs.get('applicable_standards', []))} applicable standards")

                    except Exception as e:
                        logger.warning(f"[StandardsValidation] Simple validation failed for {item_product_type}: {e}")

                else:
                    # CRITICAL FIX: Cached specs insufficient - treat as cache miss and run RAG
                    logger.info(
                        f"[StandardsValidation] Cached standards insufficient for {item_name}: "
                        f"only {total_specs_count} specs (minimum: {MIN_STANDARDS_SPECS_COUNT}). "
                        f"Running RAG for completion."
                    )
                    validation_result["cache_misses"] += 1

                    # Build requirements dict from item
                    requirements = {
                        "product_type": item_product_type,
                        "specifications": item.get("specifications", {}),
                    }

                    # Add safety requirements if provided
                    if safety_requirements:
                        if safety_requirements.get("sil_level"):
                            requirements["sil_level"] = safety_requirements["sil_level"]
                        if safety_requirements.get("hazardous_area"):
                            requirements["hazardous_area"] = True
                        if safety_requirements.get("atex_zone"):
                            requirements["atex_zone"] = safety_requirements["atex_zone"]

                    # Validate against standards (this will run RAG)
                    try:
                        val_result = validate_requirements_against_standards(
                            product_type=item_product_type,
                            requirements=requirements
                        )

                        if not val_result.get("is_compliant"):
                            validation_result["is_compliant"] = False
                            validation_result["compliance_issues"].extend([
                                {
                                    "item": item_name,
                                    **issue
                                }
                                for issue in val_result.get("compliance_issues", [])
                            ])

                        validation_result["recommendations"].extend([
                            {
                                "item": item_name,
                                **rec
                            }
                            for rec in val_result.get("recommendations", [])
                        ])

                    except Exception as e:
                        logger.warning(f"[StandardsValidation] Validation failed for {item_product_type}: {e}")

            else:
                # 🔥 FIX #4 STEP 2: Cache miss - only then run RAG 🔥
                logger.info(f"[StandardsValidation] Cache MISS for {item_name} - running RAG (slower path)")
                validation_result["cache_misses"] += 1

                # Build requirements dict from item
                requirements = {
                    "product_type": item_product_type,
                    "specifications": item.get("specifications", {}),
                }

                # Add safety requirements if provided
                if safety_requirements:
                    if safety_requirements.get("sil_level"):
                        requirements["sil_level"] = safety_requirements["sil_level"]
                    if safety_requirements.get("hazardous_area"):
                        requirements["hazardous_area"] = True
                    if safety_requirements.get("atex_zone"):
                        requirements["atex_zone"] = safety_requirements["atex_zone"]

                # Validate against standards (this will run RAG)
                try:
                    val_result = validate_requirements_against_standards(
                        product_type=item_product_type,
                        requirements=requirements
                    )

                    if not val_result.get("is_compliant"):
                        validation_result["is_compliant"] = False
                        validation_result["compliance_issues"].extend([
                            {
                                "item": item_name,
                                **issue
                            }
                            for issue in val_result.get("compliance_issues", [])
                        ])

                    validation_result["recommendations"].extend([
                        {
                            "item": item_name,
                            **rec
                        }
                        for rec in val_result.get("recommendations", [])
                    ])

                except Exception as e:
                    logger.warning(f"[StandardsValidation] Validation failed for {item_product_type}: {e}")

    except ImportError:
        logger.warning("[StandardsValidation] Standards validation tool not available")
        validation_result["validation_status"] = "unavailable"
    except Exception as e:
        logger.error(f"[StandardsValidation] Validation error: {e}")
        validation_result["validation_status"] = "error"
        validation_result["error"] = str(e)

    logger.info(f"[StandardsValidation] Validation complete - Cache hits: {validation_result['cache_hits']}, Misses: {validation_result['cache_misses']}")

    return validation_result


# =============================================================================
# NOTE: _normalize_category, STANDARDS_KEYWORDS, and is_standards_related_question
# are now imported from .standards.detector and .standards.enrichment modules.
# =============================================================================


def route_standards_question(question: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Route a standards-related question to Standards RAG and return the response.
    
    Args:
        question: User's standards-related question
        top_k: Number of documents to retrieve
        
    Returns:
        Standards RAG response with answer and citations
    """
    logger.info(f"[StandardsRouting] Routing standards question: {question[:50]}...")
    
    try:
        from .workflow import run_standards_rag_workflow
        
        result = run_standards_rag_workflow(
            question=question,
            top_k=top_k
        )
        
        if result.get("status") == "success":
            final_response = result.get("final_response", {})
            return {
                "success": True,
                "answer": final_response.get("answer", ""),
                "citations": final_response.get("citations", []),
                "sources_used": final_response.get("sources_used", []),
                "confidence": final_response.get("confidence", 0.0),
                "is_standards_response": True
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Standards RAG query failed"),
                "is_standards_response": True
            }
            
    except Exception as e:
        logger.error(f"[StandardsRouting] Error routing question: {e}")
        return {
            "success": False,
            "error": str(e),
            "is_standards_response": True
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "enrich_identified_items_with_standards",
    "validate_items_against_domain_standards",
    "is_standards_related_question",
    "route_standards_question",
]
