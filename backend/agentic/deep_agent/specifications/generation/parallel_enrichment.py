# agentic/deep_agent/parallel_specs_enrichment.py
# =============================================================================
# PARALLEL SPECIFICATION ENRICHMENT (3-SOURCE)
# =============================================================================
#
# Provides parallel execution for the 3 main sources of specifications:
# 1. User Input (highest priority)
# 2. Standards RAG
# 3. LLM/Vendor Generation
#
# =============================================================================

import logging
import concurrent.futures
from typing import Dict, Any, List, Optional
import time

from .llm_generator import extract_user_specified_specs, generate_llm_specs
try:
    # Try absolute import (if backend is in path)
    from tools.standards_enrichment_tool import get_applicable_standards
except ImportError:
    # Try relative import (if running as package)
    from ...tools.standards_enrichment_tool import get_applicable_standards

logger = logging.getLogger(__name__)

def run_parallel_3_source_enrichment(
    product_type: str,
    user_input: Optional[str] = None,
    user_specs: Optional[Dict[str, Any]] = None,
    standards_context: Optional[str] = None,
    vendor_data: Optional[List[Dict[str, Any]]] = None, # Future use
    max_workers: int = 3
) -> Dict[str, Any]:
    """
    Run enrichment from User, Standards, and LLM sources in parallel.
    
    Args:
        product_type: Type of product
        user_input: Raw user text
        user_specs: Structured user specs
        standards_context: Context for standards search
        vendor_data: Vendor data (not yet fully parallelized here, passed for context)
        max_workers: Thread pool size
        
    Returns:
        Merged dictionary of specifications
    """
    start_time = time.time()
    logger.info(f"[ParallelEnrichment] Starting 3-source enrichment for {product_type}")
    
    results = {
        "user": {},
        "standards": {},
        "llm": {}
    }
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        # 1. User Specs Task
        if user_input or user_specs:
            futures["user"] = executor.submit(
                _extract_user_specs_task, 
                product_type, 
                user_input, 
                user_specs
            )
            
        # 2. Standards Task
        if standards_context:
            futures["standards"] = executor.submit(
                _extract_standards_specs_task,
                product_type,
                standards_context
            )
            
        # 3. LLM Generation Task (always run to fill gaps)
        # We assume a base context is helpful
        futures["llm"] = executor.submit(
            _generate_llm_specs_task,
            product_type,
            user_input or ""
        )
        
        # Wait for all
        for source, future in futures.items():
            try:
                res = future.result(timeout=60) # 60s timeout
                results[source] = res
                logger.info(f"[ParallelEnrichment] {source} task completed with {len(res)} specs")
            except Exception as e:
                logger.error(f"[ParallelEnrichment] {source} task failed: {e}")
                
    # Merge results
    merged_specs = deduplicate_and_merge_specifications([
        results["user"],
        results["standards"],
        results["llm"]
    ])
    
    elapsed = time.time() - start_time
    logger.info(f"[ParallelEnrichment] Completed in {elapsed:.2f}s. Total specs: {len(merged_specs)}")
    return merged_specs

def deduplicate_and_merge_specifications(specs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge list of spec dictionaries with priority strategies.
    Earlier dictionaries in the list have higher priority.
    """
    merged = {}
    
    # Iterate in reverse order so that higher priority (earlier items) overwrite later ones
    # We want User (index 0) to overwrite Standards (index 1), etc.
    # List is [User, Standards, LLM].
    # So we want LLM first, then Standards, then User.
    
    for specs in reversed(specs_list):
        if not specs:
            continue
            
        for key, val in specs.items():
            merged[key] = val
            
    return merged

# =============================================================================
# WORKER TASKS
# =============================================================================

def _extract_user_specs_task(product_type, user_input, user_specs):
    specs = {}
    if user_specs:
        specs.update(user_specs)
    if user_input:
        try:
            extracted = extract_user_specified_specs(user_input, product_type)
            # User specs passed explicitly take precedence over text extraction
            for k, v in extracted.items():
                if k not in specs:
                    specs[k] = v
        except Exception as e:
            logger.warning(f"User spec task error: {e}")
    return specs

def _extract_standards_specs_task(product_type, context):
    try:
        # Use tool function directly
        result = get_applicable_standards(product_type)
        
        # Transform structural result into specs dict
        specs = {}
        if result.get("applicable_standards"):
             specs["Applicable Standards"] = ", ".join(result["applicable_standards"])
        if result.get("certifications"):
             specs["Certifications"] = ", ".join(result["certifications"])
        return specs
    except Exception as e:
        logger.warning(f"Standards task error: {e}")
        return {}

def _generate_llm_specs_task(product_type, context):
    try:
        # Generate some baseline specs if we have little info
        result = generate_llm_specs(product_type, product_type, f"Context: {context}")
        return result.get("specifications", {})
    except Exception as e:
        logger.warning(f"LLM task error: {e}")
        return {}
