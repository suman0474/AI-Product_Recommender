"""
Vendor Analysis Tool for Product Search Workflow
=================================================

Step 4 of Product Search Workflow:
- Loads vendors matching product type
- **APPLIES STRATEGY RAG** to filter/prioritize vendors before analysis
- Retrieves PDF datasheets and JSON product catalogs for approved vendors
- Runs parallel vendor analysis using get_vendor_prompt
- Returns matched products with detailed analysis + strategy context

STRATEGY RAG INTEGRATION (Step 2.5):
- Retrieves strategic context from Pinecone vector store (TRUE RAG)
- Falls back to LLM inference if vector store is empty
- Filters out FORBIDDEN vendors defined in strategy documents
- Prioritizes PREFERRED vendors for analysis
- Enriches final matches with strategy_priority scores

Strategic context includes:
- Cost optimization priorities
- Sustainability requirements
- Compliance alignment
- Supplier reliability metrics
- Long-term partnership alignment

This tool integrates:
- Vendor loading from MongoDB/Azure
- Strategy RAG for vendor filtering (NEW)
- PDF content extraction
- JSON product catalog loading
- LLM-powered vendor analysis
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configure logging
logger = logging.getLogger(__name__)

# Debug flags for vendor analysis debugging
try:
    from debug_flags import debug_log, timed_execution, is_debug_enabled
    DEBUG_AVAILABLE = True
except ImportError:
    DEBUG_AVAILABLE = False
    def debug_log(module, **kwargs):
        def decorator(func):
            return func
        return decorator
    def timed_execution(module, **kwargs):
        def decorator(func):
            return func
        return decorator
    def is_debug_enabled(module):
        return False

# Issue-specific debug logging
try:
    from debug_flags import issue_debug
except ImportError:
    issue_debug = None


class VendorAnalysisTool:
    """
    Vendor Analysis Tool - Step 4 of Product Search Workflow

    Responsibilities:
    1. Load vendors matching the detected product type
    2. Retrieve vendor documentation (PDFs and JSON)
    3. Run parallel vendor analysis
    4. Return matched products with detailed analysis
    """

    def __init__(self, max_workers: int = 10, max_retries: int = 3):
        """
        Initialize the vendor analysis tool.

        Args:
            max_workers: Maximum parallel workers for vendor analysis (PHASE 1 FIX: increased from 5 to 10)
            max_retries: Maximum retries for rate-limited requests
        """
        self.max_workers = max_workers
        self.max_retries = max_retries
        # PHASE 1 FIX: Add response caching for repeated product type + requirements combinations
        self._response_cache = {}  # Cache for vendor analysis responses
        logger.info("[VendorAnalysisTool] Initialized with max_workers=%d", max_workers)

    @timed_execution("VENDOR_ANALYSIS", threshold_ms=45000)
    @debug_log("VENDOR_ANALYSIS", log_args=True, log_result=False)
    def analyze(
        self,
        structured_requirements: Dict[str, Any],
        product_type: str,
        session_id: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze vendors for matching products with Strategy RAG filtering.
        
        WORKFLOW:
        1. Load vendors for product type
        2. Apply Strategy RAG to filter/prioritize vendors based on:
           - Cost optimization priorities
           - Sustainability requirements
           - Compliance alignment
           - Supplier reliability
           - Long-term partnership alignment
        3. Load product data for strategically-approved vendors only
        4. Run parallel vendor analysis
        5. Return matches enriched with strategy context

        Args:
            structured_requirements: Collected user requirements
            product_type: Detected product type
            session_id: Session tracking ID
            schema: Optional product schema

        Returns:
            {
                'success': bool,
                'product_type': str,
                'vendor_matches': list,  # Enriched with strategy_priority and is_preferred_vendor
                'vendor_run_details': list,
                'total_matches': int,
                'vendors_analyzed': int,
                'original_vendor_count': int,  # Before strategy filtering
                'filtered_vendor_count': int,  # After strategy filtering
                'excluded_by_strategy': int,   # Count of excluded vendors
                'strategy_context': {          # Strategy RAG results
                    'applied': bool,
                    'rag_type': str,           # 'true_rag' or 'llm_inference'
                    'preferred_vendors': list,
                    'forbidden_vendors': list,
                    'excluded_vendors': list,
                    'vendor_priorities': dict,
                    'confidence': float,
                    'strategy_notes': str,
                    'sources_used': list
                },
                'analysis_summary': str
            }
        """
        logger.info("[VendorAnalysisTool] Starting vendor analysis")
        logger.info("[VendorAnalysisTool] Product type: %s", product_type)
        logger.info("[VendorAnalysisTool] Session: %s", session_id or "N/A")

        # PHASE 1 FIX: Check response cache for identical requests (saves 200-600 seconds)
        import hashlib
        try:
            cache_key = hashlib.md5(
                f"{product_type}:{json.dumps(structured_requirements, sort_keys=True, default=str)}".encode()
            ).hexdigest()

            if cache_key in self._response_cache:
                logger.info("[VendorAnalysisTool] ✓ Cache hit - returning cached results (saves ~200-600 seconds)")
                if issue_debug:
                    issue_debug.cache_hit("vendor_analysis", cache_key[:16])
                return self._response_cache[cache_key]
        except Exception as cache_check_error:
            logger.warning("[VendorAnalysisTool] Cache check failed: %s (continuing without cache)", cache_check_error)
            if issue_debug:
                issue_debug.cache_miss("vendor_analysis", str(cache_check_error)[:50])

        result = {
            "success": False,
            "product_type": product_type,
            "session_id": session_id
        }

        try:
            # Import required modules
            from core.chaining import (
                setup_langchain_components,
                invoke_vendor_chain,
                to_dict_if_pydantic
            )
            from services.azure.blob_utils import (
                get_vendors_for_product_type,
                get_products_for_vendors
            )
            # Step 1: Setup LLM components
            logger.info("[VendorAnalysisTool] Step 1: Setting up LLM components")
            components = setup_langchain_components()

            # Step 2: Load vendors directly from Azure Blob Storage
            logger.info("[VendorAnalysisTool] Step 2: Loading vendors for product type: '%s'", product_type)
            
            vendors = get_vendors_for_product_type(product_type) if product_type else []
            
            logger.info("[VendorAnalysisTool] ===== DIAGNOSTIC: VENDOR LOADING =====")
            logger.info("[VendorAnalysisTool] Product type: '%s'", product_type)
            logger.info("[VendorAnalysisTool] Found vendors: %d - %s", len(vendors), vendors[:5] if vendors else [])

            if not vendors:
                logger.warning("[VendorAnalysisTool] NO VENDORS FOUND - Returning empty result")
                logger.warning("[VendorAnalysisTool] This could mean: 1) No data in Azure Blob, 2) Product type doesn't match metadata")
                result['success'] = True
                result['vendor_matches'] = []
                result['vendor_run_details'] = []
                result['total_matches'] = 0
                result['vendors_analyzed'] = 0
                result['analysis_summary'] = "No vendors available for analysis"
                return result

            # =================================================================
            # STEP 2.5: STRATEGY RAG NODE - Filter/Prioritize Vendors
            # Apply strategic context BEFORE analysis:
            # - Cost optimization priorities
            # - Sustainability requirements
            # - Compliance alignment
            # - Supplier reliability
            # - Long-term partnership alignment
            # =================================================================
            
            # ╔══════════════════════════════════════════════════════════════╗
            # ║           STRATEGY RAG INVOCATION STARTING             ║
            # ╚══════════════════════════════════════════════════════════════╝
            import datetime
            strategy_rag_invocation_time = datetime.datetime.now().isoformat()
            strategy_rag_invoked = False
            
            logger.info("="*70)
            logger.info("STRATEGY RAG INVOKED")
            logger.info(f"   Timestamp: {strategy_rag_invocation_time}")
            logger.info(f"   Product Type: {product_type}")
            logger.info(f"   Vendors to Filter: {len(vendors)}")
            logger.info(f"   Session: {session_id}")
            logger.info("="*70)
            logger.info("="*70)
            logger.info("[STRATEGY RAG] INVOCATION STARTED")
            logger.info(f"   Time: {strategy_rag_invocation_time}")
            logger.info(f"   Product: {product_type}")
            logger.info(f"   Vendors: {len(vendors)}")
            logger.info("="*70)
            
            logger.info("[VendorAnalysisTool] Step 2.5: Applying Strategy RAG to filter/prioritize vendors")
            
            strategy_context = None
            filtered_vendors = vendors.copy()
            excluded_vendors = []
            vendor_priorities = {}
            
            try:
                strategy_rag_invoked = True
                # FIX: Import from correct path (strategy_rag subdirectory, not stub)
                from agentic.workflows.strategy_rag.strategy_rag_enrichment import (
                    get_strategy_with_auto_fallback,
                    filter_vendors_by_strategy
                )
                
                # Get strategy data from TRUE RAG (or fallback to LLM)
                strategy_context = get_strategy_with_auto_fallback(
                    product_type=product_type,
                    requirements=structured_requirements,
                    top_k=7
                )
                
                if strategy_context.get('success'):
                    rag_type = strategy_context.get('rag_type', 'unknown')
                    preferred = strategy_context.get('preferred_vendors', [])
                    forbidden = strategy_context.get('forbidden_vendors', [])
                    priorities = strategy_context.get('procurement_priorities', {})
                    strategy_notes = strategy_context.get('strategy_notes', '')
                    confidence = strategy_context.get('confidence', 0.0)
                    
                    logger.info("="*70)
                    logger.info(f"STRATEGY RAG APPLIED SUCCESSFULLY ({rag_type})")
                    logger.info(f"   Preferred vendors: {preferred}")
                    logger.info(f"   Forbidden vendors: {forbidden}")
                    logger.info(f"   Confidence: {confidence:.2f}")
                    logger.info("="*70)
                    logger.info("="*70)
                    logger.info(f"[STRATEGY RAG] COMPLETED - Type: {rag_type}")
                    logger.info(f"   Preferred: {preferred}")
                    logger.info(f"   Forbidden: {forbidden}")
                    logger.info(f"   Confidence: {confidence:.2f}")
                    logger.info("="*70)
                    
                    logger.info(f"[VendorAnalysisTool] Priorities: {priorities}")
                    logger.info(f"[VendorAnalysisTool] Strategy notes: {strategy_notes[:200] if strategy_notes else 'None'}...")
                    
                    # Apply vendor filtering
                    filter_result = filter_vendors_by_strategy(vendors, strategy_context)
                    
                    # Extract filtered vendors (prioritized order)
                    accepted = filter_result.get('accepted_vendors', [])
                    excluded_vendors = filter_result.get('excluded_vendors', [])
                    
                    if accepted:
                        # Use filtered, prioritized vendor list
                        filtered_vendors = [v['vendor'] for v in accepted]
                        vendor_priorities = {v['vendor']: v.get('priority_score', 0) for v in accepted}
                        
                        logger.info(f"[VendorAnalysisTool] Strategy filtering: {len(filtered_vendors)} accepted, {len(excluded_vendors)} excluded")
                        logger.info(f"[VendorAnalysisTool] Prioritized vendor order: {filtered_vendors[:5]}...")
                        
                        # Log excluded vendors
                        for ex in excluded_vendors:
                            logger.info(f"[VendorAnalysisTool] Excluded: {ex['vendor']} - {ex['reason']}")
                    else:
                        logger.warning("[VendorAnalysisTool] No vendors passed strategy filter - using original list")
                        logger.warning("="*70)
                        logger.warning("[STRATEGY RAG] No vendors passed filter")
                        logger.warning("="*70)
                        filtered_vendors = vendors
                else:
                    logger.warning(f"[VendorAnalysisTool] Strategy RAG returned no results: {strategy_context.get('error', 'Unknown')}")
                    logger.warning("="*70)
                    logger.warning(f"[STRATEGY RAG] NO RESULTS: {strategy_context.get('error', 'Unknown')}")
                    logger.warning("="*70)
                    
            except Exception as strategy_error:
                logger.warning(f"[VendorAnalysisTool] ⚠ Strategy RAG failed (proceeding without): {strategy_error}")
                logger.error("="*70)
                logger.error(f"[STRATEGY RAG] ERROR: {strategy_error}")
                logger.error("="*70)
                filtered_vendors = vendors
            
            # Store strategy context in result for downstream use
            # ══════════════════════════════════════════════════════════════
            # RAG INVOCATION TRACKING - Visible in browser Network tab
            # ══════════════════════════════════════════════════════════════
            result['rag_invocations'] = {
                'strategy_rag': {
                    'invoked': strategy_rag_invoked,
                    'invocation_time': strategy_rag_invocation_time,
                    'success': strategy_context is not None and strategy_context.get('success', False),
                    'rag_type': strategy_context.get('rag_type') if strategy_context else None,
                    'product_type': product_type,
                    'vendors_before_filter': len(vendors),
                    'vendors_after_filter': len(filtered_vendors),
                    'excluded_count': len(excluded_vendors)
                },
                'standards_rag': {
                    'invoked': False,
                    'note': 'Standards RAG is applied in validation_tool.py, not during vendor analysis'
                }
            }
            
            result['strategy_context'] = {
                'applied': strategy_context is not None and strategy_context.get('success', False),
                'rag_type': strategy_context.get('rag_type') if strategy_context else None,
                'preferred_vendors': strategy_context.get('preferred_vendors', []) if strategy_context else [],
                'forbidden_vendors': strategy_context.get('forbidden_vendors', []) if strategy_context else [],
                'excluded_vendors': excluded_vendors,
                'vendor_priorities': vendor_priorities,
                'confidence': strategy_context.get('confidence', 0.0) if strategy_context else 0.0,
                'strategy_notes': strategy_context.get('strategy_notes', '') if strategy_context else '',
                'sources_used': strategy_context.get('sources_used', []) if strategy_context else []
            }
            
            # Use filtered vendors for product loading
            vendors_for_analysis = filtered_vendors
            logger.info(f"[VendorAnalysisTool] Proceeding with {len(vendors_for_analysis)} strategically-filtered vendors")


            # Step 3: Load product data for STRATEGICALLY-FILTERED vendors
            logger.info("[VendorAnalysisTool] Step 3: Loading product data from Azure Blob (filtered vendors only)")
            
            # Use vendors_for_analysis which has been filtered by Strategy RAG
            products_data = get_products_for_vendors(vendors_for_analysis, product_type)
            
            logger.info("[VendorAnalysisTool] ===== DIAGNOSTIC: PRODUCT DATA (STRATEGY-FILTERED) =====")
            logger.info("[VendorAnalysisTool] Products loaded for %d strategically-approved vendors", len(products_data))
            for v, products in list(products_data.items())[:3]:
                product_count = len(products) if products else 0
                priority = vendor_priorities.get(v, 0)
                logger.info("[VendorAnalysisTool]   - %s: %d product entries (priority: %d)", v, product_count, priority)

            if not products_data:
                logger.warning("[VendorAnalysisTool] NO PRODUCT DATA LOADED")
                result['success'] = True
                result['vendor_matches'] = []
                result['vendor_run_details'] = []
                result['total_matches'] = 0
                result['vendors_analyzed'] = 0
                result['analysis_summary'] = "No product data available for analysis"
                return result

            # Step 4: Prepare structured requirements string
            logger.info("[VendorAnalysisTool] Step 4: Preparing requirements")
            requirements_str = self._format_requirements(structured_requirements)
            logger.info("[VendorAnalysisTool] Requirements: %s...", requirements_str[:200] if requirements_str else "EMPTY")

            # Step 5: Prepare vendor payloads (product JSON as content, no PDF required)
            logger.info("[VendorAnalysisTool] Step 5: Preparing vendor payloads")
            vendor_payloads = {}
            
            for vendor_name, products in products_data.items():
                if products:
                    # Convert products to JSON string for the prompt
                    products_json = json.dumps(products, indent=2, ensure_ascii=False)
                    vendor_payloads[vendor_name] = {
                        "products": products,
                        "pdf_text": products_json  # Use product JSON as the analysis content
                    }
                    
            logger.info("[VendorAnalysisTool] ===== DIAGNOSTIC: VENDOR PAYLOADS =====")
            logger.info("[VendorAnalysisTool] Total payloads prepared: %d", len(vendor_payloads))
            for v, data in list(vendor_payloads.items())[:5]:
                content_len = len(data.get('pdf_text', '')) if data.get('pdf_text') else 0
                products_list = data.get('products', [])
                products_count = len(products_list)
                
                # Detailed structure analysis
                models_count = 0
                submodels_count = 0
                specs_count = 0
                
                if products_list and isinstance(products_list[0], dict):
                    first_product = products_list[0]
                    if 'models' in first_product:
                        models = first_product['models']
                        models_count = len(models) if models else 0
                        for model in (models or []):
                            if 'sub_models' in model:
                                submodels_count += len(model['sub_models'])
                                for sub in model['sub_models']:
                                    if 'specifications' in sub:
                                        specs_count += len(sub.get('specifications', {}))
                
                logger.info("[VendorAnalysisTool]   - %s: Content=%d chars, Products=%d, Models=%d, Submodels=%d, Specs=%d", 
                           v, content_len, products_count, models_count, submodels_count, specs_count)

            if not vendor_payloads:
                logger.warning("[VendorAnalysisTool] NO VALID VENDOR PAYLOADS")
                result['success'] = True
                result['vendor_matches'] = []
                result['vendor_run_details'] = []
                result['total_matches'] = 0
                result['vendors_analyzed'] = 0
                result['analysis_summary'] = "No vendor data available for analysis"
                return result

            # Step 5.5: Load applicable standards (if available)
            logger.info("[VendorAnalysisTool] Step 5.5: Loading applicable engineering standards")
            applicable_standards = []
            standards_specs = "No specific standards requirements provided."
            
            try:
                # TODO: Implement standards loading from user context/session
                # For now, check if standards are provided in requirements
                if structured_requirements and isinstance(structured_requirements, dict):
                    if 'applicable_standards' in structured_requirements:
                        applicable_standards = structured_requirements.get('applicable_standards', [])
                    if 'standards_specifications' in structured_requirements:
                        standards_specs = structured_requirements.get('standards_specifications', standards_specs)
                
                logger.info("[VendorAnalysisTool] Loaded %d applicable standards", len(applicable_standards))
                if applicable_standards:
                    logger.debug("[VendorAnalysisTool] Standards: %s", applicable_standards)
            except Exception as standards_error:
                logger.warning("[VendorAnalysisTool] Failed to load standards: %s (continuing without standards)", standards_error)

            # Step 6: Run parallel vendor analysis
            logger.info("[VendorAnalysisTool] Step 6: Running parallel vendor analysis")
            vendor_matches = []
            run_details = []

            # Run parallel analysis
            actual_workers = min(len(vendor_payloads), self.max_workers)
            logger.info("[VendorAnalysisTool] Using %d workers for %d vendors",
                       actual_workers, len(vendor_payloads))

            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                futures = {}
                for i, (vendor, data) in enumerate(vendor_payloads.items()):
                    # PHASE 1 FIX: Removed time.sleep(5) - ThreadPoolExecutor handles concurrency
                    # This alone saves 5 vendors × 5s = 25 seconds per request
                    future = executor.submit(
                        self._analyze_vendor,
                        components,
                        requirements_str,
                        vendor,
                        data,
                        applicable_standards,  # Pass standards
                        standards_specs        # Pass standards specs
                    )
                    futures[future] = vendor

                for future in as_completed(futures):
                    vendor = futures[future]
                    try:
                        vendor_result, error = future.result()

                        if vendor_result and isinstance(vendor_result.get("vendor_matches"), list):
                            for match in vendor_result["vendor_matches"]:
                                match_dict = to_dict_if_pydantic(match)
                                # Normalize to camelCase for frontend compatibility
                                normalized_match = {
                                    'productName': match_dict.get('product_name', match_dict.get('productName', '')),
                                    'vendor': vendor,
                                    'modelFamily': match_dict.get('model_family', match_dict.get('modelFamily', '')),
                                    'productType': match_dict.get('product_type', match_dict.get('productType', '')),
                                    'matchScore': match_dict.get('match_score', match_dict.get('matchScore', 0)),
                                    'requirementsMatch': match_dict.get('requirements_match', match_dict.get('requirementsMatch', False)),
                                    'reasoning': match_dict.get('reasoning', ''),
                                    'limitations': match_dict.get('limitations', ''),
                                    'productDescription': match_dict.get('product_description', match_dict.get('productDescription', '')),
                                    # Standards compliance data
                                    'standardsCompliance': match_dict.get('standards_compliance', match_dict.get('standardsCompliance', {})),
                                    # Structured requirements matching
                                    'matchedRequirements': match_dict.get('matched_requirements', match_dict.get('matchedRequirements', {})),
                                    'unmatchedRequirements': match_dict.get('unmatched_requirements', match_dict.get('unmatchedRequirements', [])),
                                    # Key strengths and recommendation
                                    'keyStrengths': match_dict.get('key_strengths', match_dict.get('keyStrengths', [])),
                                    'recommendation': match_dict.get('recommendation', '')
                                }
                                vendor_matches.append(normalized_match)
                                logger.info(f"[VendorAnalysisTool] DEBUG: Normalized match for {vendor}: {normalized_match}")
                            run_details.append({"vendor": vendor, "status": "success"})
                            logger.info("[VendorAnalysisTool] Vendor '%s' returned %d matches",
                                       vendor, len(vendor_result["vendor_matches"]))
                        else:
                            run_details.append({
                                "vendor": vendor,
                                "status": "failed" if error else "empty",
                                "error": error
                            })
                            logger.warning("[VendorAnalysisTool] Vendor '%s' failed: %s", vendor, error)

                    except Exception as e:
                        logger.error("[VendorAnalysisTool] Vendor '%s' exception: %s", vendor, str(e))
                        run_details.append({
                            "vendor": vendor,
                            "status": "error",
                            "error": str(e)
                        })

            # Build result with strategy context
            result['success'] = True
            result['vendor_matches'] = vendor_matches
            result['vendor_run_details'] = run_details
            result['total_matches'] = len(vendor_matches)
            result['vendors_analyzed'] = len(vendor_payloads)
            result['original_vendor_count'] = len(vendors)  # Before filtering
            result['filtered_vendor_count'] = len(vendors_for_analysis)  # After strategy filtering
            result['excluded_by_strategy'] = len(excluded_vendors)  # Vendors excluded by strategy

            # Enrich matches with strategy priority scores
            for match in vendor_matches:
                vendor_name = match.get('vendor', '')
                match['strategy_priority'] = vendor_priorities.get(vendor_name, 0)
                match['is_preferred_vendor'] = vendor_name.lower() in [
                    p.lower() for p in (strategy_context.get('preferred_vendors', []) if strategy_context else [])
                ]

                # Ensure requirementsMatch is set (Frontend relies on this)
                if 'requirementsMatch' not in match:
                    score = match.get('matchScore', 0)
                    match['requirementsMatch'] = score >= 80

            # ══════════════════════════════════════════════════════════════════════
            # ENRICH MATCHES WITH VENDOR PRODUCT IMAGES (NEW)
            # ══════════════════════════════════════════════════════════════════════
            # Optionally fetch real product images from the web for each vendor
            # This enriches vendor analysis results with authentic product imagery
            logger.info("[VendorAnalysisTool] Step 7: Enriching matches with vendor product images")

            try:
                from agentic.utils.vendor_images import fetch_images_for_vendor_matches

                # Group matches by vendor and enrich with images
                vendor_groups = {}
                for match in vendor_matches:
                    vendor = match.get('vendor', '')
                    if vendor not in vendor_groups:
                        vendor_groups[vendor] = []
                    vendor_groups[vendor].append(match)

                # Fetch images for each vendor and enrich matches
                for vendor_name, vendor_match_list in vendor_groups.items():
                    try:
                        logger.info(f"[VendorAnalysisTool] Fetching images for vendor: {vendor_name}")
                        enriched = fetch_images_for_vendor_matches(
                            vendor_name=vendor_name,
                            matches=vendor_match_list,
                            max_workers=2
                        )
                        # Update matches in vendor_matches list with enriched versions
                        for enriched_match in enriched:
                            for i, match in enumerate(vendor_matches):
                                if (match.get('vendor') == enriched_match.get('vendor') and
                                    match.get('productName') == enriched_match.get('productName')):
                                    vendor_matches[i] = enriched_match
                    except Exception as vendor_image_error:
                        logger.warning(f"[VendorAnalysisTool] Failed to fetch images for {vendor_name}: {vendor_image_error}")
                        # Continue without images - not critical to analysis

                logger.info("[VendorAnalysisTool] Image enrichment complete")

            except ImportError:
                logger.debug("[VendorAnalysisTool] Image utilities not available - skipping image enrichment")
            except Exception as image_enrichment_error:
                logger.warning(f"[VendorAnalysisTool] Image enrichment failed: {image_enrichment_error}")
                # Continue without images - analysis results are still valid

            # ══════════════════════════════════════════════════════════════════════
            # ENRICH MATCHES WITH VENDOR LOGOS (NEW)
            # ══════════════════════════════════════════════════════════════════════
            # Enrich vendor matches with explicit vendor logos (for UI display)
            logger.info("[VendorAnalysisTool] Step 8: Enriching matches with vendor logos")
            try:
                from agentic.utils.vendor_images import enrich_matches_with_logos
                
                vendor_matches = enrich_matches_with_logos(
                    matches=vendor_matches,
                    max_workers=2
                )
                logger.info("[VendorAnalysisTool] Logo enrichment complete")
            except ImportError:
                logger.debug("[VendorAnalysisTool] Logo utilities not available - skipping logo enrichment")
            except Exception as logo_enrichment_error:
                 logger.warning(f"[VendorAnalysisTool] Logo enrichment failed: {logo_enrichment_error}")

            # ══════════════════════════════════════════════════════════════════════
            # ENRICH MATCHES WITH PRICING LINKS (NEW)
            # ══════════════════════════════════════════════════════════════════════
            # Search for pricing/buying links for each product
            logger.info("[VendorAnalysisTool] Step 9: Enriching matches with pricing links")
            try:
                from agentic.utils.pricing_search import enrich_matches_with_pricing
                
                vendor_matches = enrich_matches_with_pricing(
                    matches=vendor_matches,
                    max_workers=5
                )
                logger.info("[VendorAnalysisTool] Pricing link enrichment complete")
            except ImportError:
                logger.debug("[VendorAnalysisTool] Pricing search utilities not available - skipping enrichment")
            except Exception as pricing_error:
                logger.warning(f"[VendorAnalysisTool] Pricing enrichment failed: {pricing_error}")


            # Generate enhanced summary with strategy info
            successful = sum(1 for d in run_details if d['status'] == 'success')
            strategy_source = result.get('strategy_context', {}).get('rag_type', 'none')
            
            result['analysis_summary'] = (
                f"Strategy RAG ({strategy_source}): Filtered {len(vendors)} → {len(vendors_for_analysis)} vendors | "
                f"Analyzed {len(vendor_payloads)} vendors, {successful} successful | "
                f"Found {len(vendor_matches)} matching products"
            )

            logger.info("[VendorAnalysisTool] ===== ANALYSIS COMPLETE =====")
            logger.info("[VendorAnalysisTool] %s", result['analysis_summary'])
            if excluded_vendors:
                logger.info("[VendorAnalysisTool] Excluded by strategy: %s", [e['vendor'] for e in excluded_vendors])

            # PHASE 1 FIX: Cache successful results for future identical requests
            try:
                if result.get('success'):
                    self._response_cache[cache_key] = result
                    logger.info("[VendorAnalysisTool] ✓ Cached results for future requests (key: %s)", cache_key[:8])
            except Exception as cache_write_error:
                logger.warning("[VendorAnalysisTool] Failed to cache results: %s", cache_write_error)

            return result

        except Exception as e:
            logger.error("[VendorAnalysisTool] Analysis failed: %s", str(e), exc_info=True)
            result['success'] = False
            result['error'] = str(e)
            result['error_type'] = type(e).__name__
            return result

    def _format_requirements(self, requirements: Dict[str, Any]) -> str:
        """Format requirements dictionary into structured string."""
        lines = []

        # Handle nested structure
        if 'mandatoryRequirements' in requirements or 'mandatory' in requirements:
            mandatory = requirements.get('mandatoryRequirements') or requirements.get('mandatory', {})
            if mandatory:
                lines.append("## Mandatory Requirements")
                for key, value in mandatory.items():
                    if value:
                        lines.append(f"- {self._format_field_name(key)}: {value}")

        if 'optionalRequirements' in requirements or 'optional' in requirements:
            optional = requirements.get('optionalRequirements') or requirements.get('optional', {})
            if optional:
                lines.append("\n## Optional Requirements")
                for key, value in optional.items():
                    if value:
                        lines.append(f"- {self._format_field_name(key)}: {value}")

        if 'selectedAdvancedParams' in requirements or 'advancedSpecs' in requirements:
            advanced = requirements.get('selectedAdvancedParams') or requirements.get('advancedSpecs', {})
            if advanced:
                lines.append("\n## Advanced Specifications")
                for key, value in advanced.items():
                    if value:
                        lines.append(f"- {key}: {value}")

        # FIX: Return structured guidance that instructs LLM to still return JSON
        if not lines:
            return """## Requirements Summary
No specific mandatory or optional requirements have been provided for this product search.

## Analysis Instruction
Analyze available products and return JSON with general recommendations based on:
- Standard industrial specifications and certifications
- Product feature completeness and quality
- Typical use case suitability for this product type
- Provide match_score based on product quality (use 85-95 range for well-documented, certified products)"""
        
        return "\n".join(lines)

    def _format_field_name(self, field: str) -> str:
        """Convert camelCase or snake_case to Title Case."""
        # Replace underscores and split on capital letters
        import re
        words = re.sub(r'([a-z])([A-Z])', r'\1 \2', field)
        words = words.replace('_', ' ')
        return words.title()

    def _prepare_payloads(
        self,
        vendors: List[str],
        pdf_content: Dict[str, str],
        products_json: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Prepare vendor payloads for analysis."""
        payloads = {}

        for vendor in vendors:
            pdf_text = pdf_content.get(vendor, "")
            products = products_json.get(vendor, [])

            # Only include vendors with some data
            if pdf_text or products:
                payloads[vendor] = {
                    "pdf_text": pdf_text,
                    "products": products
                }
                logger.debug("[VendorAnalysisTool] Prepared payload for '%s': PDF=%s, Products=%d",
                           vendor, bool(pdf_text), len(products) if products else 0)

        return payloads

    def _analyze_vendor(
        self,
        components: Dict[str, Any],
        requirements_str: str,
        vendor: str,
        vendor_data: Dict[str, Any],
        applicable_standards: Optional[List[str]] = None,
        standards_specs: Optional[str] = None
    ) -> tuple:
        """
        Analyze a single vendor.
        
        Args:
            components: LLM and other component references
            requirements_str: Formatted requirements string
            vendor: Vendor name being analyzed
            vendor_data: Vendor PDF and product data
            applicable_standards: List of applicable engineering standards
            standards_specs: Standards specifications from documents
            
        Returns:
            Tuple of (result dict, error string or None)
        """
        error = None
        result = None
        base_retry_delay = 15

        logger.info("[VendorAnalysisTool] START analysis for vendor: %s", vendor)

        for attempt in range(self.max_retries):
            try:
                from core.chaining import invoke_vendor_chain

                pdf_text = vendor_data.get("pdf_text", "")
                products = vendor_data.get("products", [])

                pdf_payload = json.dumps({vendor: pdf_text}, ensure_ascii=False) if pdf_text else "{}"
                products_payload = json.dumps(products, ensure_ascii=False)

                result = invoke_vendor_chain(
                    components,
                    vendor,                      # Pass vendor name
                    requirements_str,
                    products_payload,
                    pdf_payload,
                    components['vendor_format_instructions'],
                    applicable_standards=applicable_standards or [],
                    standards_specs=standards_specs or "No specific standards requirements provided."
                )

                # Convert to dict if needed and parse for robust handling
                from core.chaining import to_dict_if_pydantic, parse_vendor_analysis_response
                result = to_dict_if_pydantic(result)
                
                # FIX: Handle malformed responses (missing vendor_matches wrapper)
                result = parse_vendor_analysis_response(result, vendor)

                logger.info("[VendorAnalysisTool] END analysis for vendor: %s (success)", vendor)
                return result, None

            except Exception as e:
                error_msg = str(e)
                is_rate_limit = any(x in error_msg.lower() for x in ['429', 'resource has been exhausted', 'quota', '503', 'overloaded'])

                if is_rate_limit and attempt < self.max_retries - 1:
                    wait_time = base_retry_delay * (2 ** attempt)
                    logger.warning("[VendorAnalysisTool] Rate limit for %s, retry %d/%d after %ds",
                                  vendor, attempt + 1, self.max_retries, wait_time)
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("[VendorAnalysisTool] Analysis failed for %s: %s", vendor, error_msg)
                    error = error_msg
                    break

        logger.info("[VendorAnalysisTool] END analysis for vendor: %s (error: %s)", vendor, error)
        return None, error


# Convenience function
def analyze_vendors(
    structured_requirements: Dict[str, Any],
    product_type: str,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run vendor analysis.

    Args:
        structured_requirements: Collected user requirements
        product_type: Detected product type
        session_id: Session tracking ID

    Returns:
        Vendor analysis result
    """
    tool = VendorAnalysisTool()
    return tool.analyze(
        structured_requirements=structured_requirements,
        product_type=product_type,
        session_id=session_id
    )
