"""
Vendor Analysis Tool for Product Search Workflow
=================================================

Step 4 of Product Search Workflow:
- Loads vendors matching product type
- **APPLIES STRATEGY RAG** to filter/prioritize vendors before analysis
- Retrieves PDF datasheets and JSON product catalogs for approved vendors
- Runs parallel vendor analysis using tools.analysis_tools.analyze_vendor_match_tool
- Returns matched products with detailed analysis + strategy context
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import hashlib

# Import the core tool
from common.tools.analysis_tools import analyze_vendor_match_tool
from common.services.azure.blob_utils import (
    get_vendors_for_product_type,
    get_products_for_vendors
)

# Configure logging
logger = logging.getLogger(__name__)

# Debug flags
try:
    from debug_flags import debug_log, timed_execution, is_debug_enabled, issue_debug
    DEBUG_AVAILABLE = True
except ImportError:
    DEBUG_AVAILABLE = False
    def debug_log(module, **kwargs):
        def decorator(func): return func
        return decorator
    def timed_execution(module, **kwargs):
        def decorator(func): return func
        return decorator
    def is_debug_enabled(module): return False
    issue_debug = None


class VendorAnalysisTool:
    """
    Vendor Analysis Tool - Step 4 of Product Search Workflow
    
    Responsibilities:
    1. Orchestrate vendor analysis (loading, strategy filtering, parallelism)
    2. Delegate actual analysis to tools.analysis_tools.analyze_vendor_match_tool
    """

    def __init__(self, max_workers: int = 10, max_retries: int = 3):
        self.max_workers = max_workers
        self.max_retries = max_retries
        self._response_cache = {}
        logger.info("[VendorAnalysisTool] Initialized with max_workers=%d", max_workers)

    @timed_execution("TOOLS", threshold_ms=45000)
    @debug_log("TOOLS", log_args=True, log_result=False)
    def analyze(
        self,
        structured_requirements: Dict[str, Any],
        product_type: str,
        session_id: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze vendors for matching products with Strategy RAG filtering.
        """
        logger.info("[VendorAnalysisTool] Starting vendor analysis")
        logger.info("[VendorAnalysisTool] Product type: %s", product_type)
        logger.info("[VendorAnalysisTool] Session: %s", session_id or "N/A")

        # Cache check
        try:
            cache_key = hashlib.md5(
                f"{product_type}:{json.dumps(structured_requirements, sort_keys=True, default=str)}".encode()
            ).hexdigest()

            if cache_key in self._response_cache:
                logger.info("[VendorAnalysisTool] ✓ Cache hit - returning cached results")
                if issue_debug:
                    issue_debug.cache_hit("vendor_analysis", cache_key[:16])
                return self._response_cache[cache_key]
        except Exception as cache_check_error:
            logger.warning("[VendorAnalysisTool] Cache check failed: %s", cache_check_error)

        result = {
            "success": False,
            "product_type": product_type,
            "session_id": session_id
        }

        try:
            # Step 1: Load vendors
            logger.info("[VendorAnalysisTool] Step 1: Loading vendors for product type: '%s'", product_type)
            vendors = get_vendors_for_product_type(product_type) if product_type else []
            
            if not vendors:
                logger.warning("[VendorAnalysisTool] NO VENDORS FOUND")
                result['success'] = True
                result['vendor_matches'] = []
                result['vendor_run_details'] = []
                result['total_matches'] = 0
                result['vendors_analyzed'] = 0
                result['analysis_summary'] = "No vendors available for analysis"
                return result

            # Step 2: Strategy RAG Filtering
            logger.info("[VendorAnalysisTool] Step 2: Applying Strategy RAG")
            strategy_context = None
            filtered_vendors = vendors.copy()
            excluded_vendors = []
            vendor_priorities = {}
            strategy_rag_invoked = False
            
            try:
                strategy_rag_invoked = True
                from common.rag.strategy.enrichment import get_strategy_with_auto_fallback
                from common.rag.strategy.mongodb_loader import filter_vendors_by_strategy
                
                strategy_context = get_strategy_with_auto_fallback(
                    product_type=product_type,
                    requirements=structured_requirements,
                    top_k=7
                )
                
                if strategy_context.get('success'):
                    filter_result = filter_vendors_by_strategy(vendors, strategy_context)
                    filtered_vendors = [v['vendor'] for v in filter_result.get('accepted_vendors', [])]
                    vendor_priorities = {v['vendor']: v.get('priority_score', 0) for v in filter_result.get('accepted_vendors', [])}
                    excluded_vendors = filter_result.get('excluded_vendors', [])
                    logger.info(f"[VendorAnalysisTool] Strategy filtered: {len(vendors)} -> {len(filtered_vendors)}")
                
            except Exception as strategy_error:
                logger.warning(f"[VendorAnalysisTool] Strategy RAG failed: {strategy_error}")
                filtered_vendors = vendors

            result['strategy_context'] = strategy_context or {}
            result['rag_invocations'] = {
                'strategy_rag': {
                    'invoked': strategy_rag_invoked,
                    'success': strategy_context.get('success', False) if strategy_context else False
                }
            }

            # Step 3: Load Product Data
            logger.info("[VendorAnalysisTool] Step 3: Loading product data from Azure Blob")
            products_data = get_products_for_vendors(filtered_vendors, product_type)
            
            if not products_data:
                logger.warning("[VendorAnalysisTool] NO PRODUCT DATA LOADED")
                result['success'] = True
                result['vendor_matches'] = []
                # ... empty result ...
                return result

            # Step 4: Prepare Payloads
            vendor_payloads = {}
            for vendor_name, products in products_data.items():
                if products:
                    vendor_payloads[vendor_name] = {
                        "products": products,
                        "pdf_text": json.dumps(products, indent=2, ensure_ascii=False) # Simplified as per previous logic
                    }

            # Step 5: Load Standards (Simplified)
            applicable_standards = []
            standards_specs = "No specific standards requirements provided."
            if structured_requirements and isinstance(structured_requirements, dict):
                 applicable_standards = structured_requirements.get('applicable_standards', [])
                 standards_specs = structured_requirements.get('standards_specifications', standards_specs)

            # Step 6: Parallel Analysis
            logger.info("[VendorAnalysisTool] Step 6: Running parallel vendor analysis")
            vendor_matches = []
            run_details = []
            actual_workers = min(len(vendor_payloads), self.max_workers)
            
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                futures = {}
                for vendor, data in vendor_payloads.items():
                    future = executor.submit(
                        self._analyze_vendor,
                        structured_requirements, # Use DICT here
                        vendor,
                        data,
                        applicable_standards,
                        standards_specs
                    )
                    futures[future] = vendor

                for future in as_completed(futures):
                    vendor = futures[future]
                    try:
                        vendor_result, error = future.result()
                        
                        if vendor_result and isinstance(vendor_result.get("vendor_matches"), list):
                            for match in vendor_result["vendor_matches"]:
                                # Match is already normalized by the tool
                                # But we need to enrich with strategy priority (from step 2)
                                match['strategy_priority'] = vendor_priorities.get(vendor, 0)
                                match['is_preferred_vendor'] = vendor in (strategy_context.get('preferred_vendors', []) if strategy_context else [])
                                vendor_matches.append(match)
                            
                            run_details.append({"vendor": vendor, "status": "success"})
                        else:
                            run_details.append({"vendor": vendor, "status": "failed", "error": error})
                            
                    except Exception as e:
                        logger.error(f"[VendorAnalysisTool] Vendor {vendor} failed: {e}")
                        run_details.append({"vendor": vendor, "status": "error", "error": str(e)})

            # Build result
            result['success'] = True
            result['vendor_matches'] = vendor_matches
            result['vendor_run_details'] = run_details
            result['total_matches'] = len(vendor_matches)
            result['vendors_analyzed'] = len(vendor_payloads)
            result['original_vendor_count'] = len(vendors)
            result['filtered_vendor_count'] = len(filtered_vendors)
            result['excluded_by_strategy'] = len(excluded_vendors)

            # Enrichments (Images, Logos, Pricing)
            
            try:
                from common.utils.vendor_images import enrich_matches_with_logos
                vendor_matches = enrich_matches_with_logos(vendor_matches, max_workers=2)
            except ImportError: pass
            
            try:
                from common.utils.pricing_search import enrich_matches_with_pricing
                vendor_matches = enrich_matches_with_pricing(vendor_matches, max_workers=5)
            except ImportError: pass

            result['analysis_summary'] = f"Analyzed {len(vendor_payloads)} vendors, Found {len(vendor_matches)} matching products"
            
            # Cache
            try:
                if result.get('success'):
                    self._response_cache[cache_key] = result
            except Exception: pass

            return result

        except Exception as e:
            logger.error("[VendorAnalysisTool] Analysis failed: %s", str(e), exc_info=True)
            result['success'] = False
            result['error'] = str(e)
            return result

    def _format_requirements(self, requirements: Dict[str, Any]) -> str:
        # Kept for backward compatibility if needed, but not used in new flow?
        # The tool handles formatting.
        return "" 

    def _analyze_vendor(
        self,
        requirements: Dict[str, Any],
        vendor: str,
        vendor_data: Dict[str, Any],
        applicable_standards: Optional[List[str]] = None,
        standards_specs: Optional[str] = None
    ) -> tuple:
        """
        Analyze a single vendor using delegation to tool.
        """
        error = None
        result = None
        
        for attempt in range(self.max_retries):
            try:
                pdf_text = vendor_data.get("pdf_text", "")
                products = vendor_data.get("products", [])
                
                # Delegate to tool
                result = analyze_vendor_match_tool.invoke({
                    "vendor": vendor,
                    "requirements": requirements,
                    "pdf_content": pdf_text,
                    "product_data": products,
                    "applicable_standards": applicable_standards,
                    "standards_specs": standards_specs
                })
                
                # Check success in tool result
                if result.get("success"):
                     return result, None
                else:
                     error = result.get("error")
                     
            except Exception as e:
                error = str(e)
                logger.warning(f"[VendorAnalysisTool] Attempt {attempt+1} failed for {vendor}: {e}")
                time.sleep(2) # Backoff
        
        return None, error

def analyze_vendors(
    structured_requirements: Dict[str, Any],
    product_type: str,
    session_id: Optional[str] = None,
    schema: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience function for analyzing vendors."""
    tool = VendorAnalysisTool()
    return tool.analyze(structured_requirements, product_type, session_id, schema)
