# chaining.py
# Contains LangChain components setup and analysis chain creation
import json
import logging
import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
# Memory imports removed - not used in current implementation
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from models import VendorAnalysis, OverallRanking, RequirementValidation
from prompts import (
    validation_prompt,
    requirements_prompt,
    vendor_prompt,
    ranking_prompt,
    additional_requirements_prompt,
)
from loading import load_requirements_schema, load_products_runnable
from mongodb_utils import get_available_vendors_from_mongodb, get_vendors_for_product_type
from dotenv import load_dotenv

# OutputFixingParser is no longer available in current LangChain versions
# Using base parsers with error handling instead
from langchain_google_genai import ChatGoogleGenerativeAI

# Import standardization utilities
from standardization_utils import standardize_ranking_result


# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_langchain_components():
    # Use the updated import from langchain_community
    from langchain_community.callbacks import OpenAICallbackHandler
    callback_handler = OpenAICallbackHandler()
    
    # Create different models for different purposes
    # Enable thinking mode using the correct parameter: thinking_budget
    # -1 = dynamic thinking (model adjusts based on complexity)
    # 0 = disable thinking (only for Flash models)
    # Max: 24k for Flash/Flash Lite, 32k for Pro
    
    # Gemini 2.5 Flash with dynamic thinking
    llm_flash = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.1, 
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        thinking_budget=-1  # Dynamic thinking enabled
    )
    
    # Gemini 2.5 Flash Lite with dynamic thinking
    llm_flash_lite = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", 
        temperature=0.1, 
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    
    # Gemini 2.5 Pro (thinking always enabled, cannot be disabled)
    llm_pro = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", 
        temperature=0.1, 
        google_api_key=os.getenv("GOOGLE_API_KEY1")
        # Note: Pro has thinking enabled by default, no parameter needed
    )
    # Memory object removed - not used in current implementation
    
    # 1. Instantiate the base parsers
    validation_parser = JsonOutputParser(pydantic_object=RequirementValidation)
    vendor_parser = JsonOutputParser(pydantic_object=VendorAnalysis)
    ranking_parser = JsonOutputParser(pydantic_object=OverallRanking)
    str_parser = StrOutputParser()
    
    # 2. Use direct parsers (OutputFixingParser is no longer available)
    # The base JsonOutputParser should handle most parsing needs
    
    # 3. Use the appropriate models for different tasks
    # Fast tasks use Flash model
    validation_chain = validation_prompt | llm_pro | validation_parser
    requirements_chain = requirements_prompt | llm_flash | str_parser
    
    # Final analysis tasks use Pro model (vendor analysis and ranking)
    vendor_chain = vendor_prompt | llm_pro | vendor_parser
    ranking_chain = ranking_prompt | llm_pro | ranking_parser
    
    # --- NEW CHAIN FOR ADDITIONAL REQUIREMENTS ---
    additional_requirements_parser = JsonOutputParser(pydantic_object=RequirementValidation)
    additional_requirements_chain = additional_requirements_prompt | llm_flash | additional_requirements_parser
    
    # --- NEW CHAIN FOR SCHEMA KEY DESCRIPTIONS ---
    # Using Flash Lite model for generating user-friendly descriptions of schema keys
    from prompts import schema_description_prompt
    schema_description_chain = schema_description_prompt | llm_flash_lite | str_parser

    # Get format instructions from the original parsers (the fixing parser wraps them)
    validation_format_instructions = validation_parser.get_format_instructions()
    vendor_format_instructions = vendor_parser.get_format_instructions()
    ranking_format_instructions = ranking_parser.get_format_instructions()
    additional_requirements_format_instructions = additional_requirements_parser.get_format_instructions()
    
    return {
        'llm': llm_flash,  # General LLM for conversations and simple text generation
        'llm_flash': llm_flash,  # For conversations and simple text generation
        'llm_flash_lite': llm_flash_lite,  # For generating schema key descriptions
        'llm_pro': llm_pro,  # For complex analysis tasks
        # 'memory': removed - not used
        'validation_chain': validation_chain,
        'requirements_chain': requirements_chain,
        'vendor_chain': vendor_chain,
        'ranking_chain': ranking_chain,
        # Add the new chain and its format instructions
        'additional_requirements_chain': additional_requirements_chain,
        'schema_description_chain': schema_description_chain,
        'additional_requirements_format_instructions': additional_requirements_format_instructions,
        'validation_format_instructions': validation_format_instructions,
        'vendor_format_instructions': vendor_format_instructions,
        'ranking_format_instructions': ranking_format_instructions,
        'callback_handler': callback_handler
    }

def get_final_ranking(vendor_analysis_dict):
    """
    Processes a dictionary of vendor analysis to create a final ranked list.
    Takes a dictionary, not a Pydantic object.
    """
    products = []
    # Use .get() for safe access in case vendor_matches is missing
    if not vendor_analysis_dict or not vendor_analysis_dict.get('vendor_matches'):
        return {'ranked_products': []}
        
    for product in vendor_analysis_dict['vendor_matches']:
        product_score = product.get('match_score', 0)
        product_name = product.get('product_name', '')
        
        # Extract model_family from the vendor analysis
        # model_family is the broader series (e.g., 'STD800', '3051C')
        # product_name is the specific submodel (e.g., 'STD850', '3051CD')
        model_family = product.get('model_family', '')
        
        # Fallback: if model_family is empty, use product_name
        # This handles cases where LLM doesn't provide model_family
        if not model_family:
            model_family = product_name
        
        # Ensure requirements_match is a boolean. Default to False if missing.
        products.append({
            'product_name': product_name,
            'vendor': product.get('vendor', ''),
            'model_family': model_family,
            'match_score': product_score,
            'requirements_match': product.get('requirements_match', False),
            'reasoning': product.get('reasoning', ''),
            'limitations': product.get('limitations', '')
        })
    
    products_sorted = sorted(products, key=lambda x: x['match_score'], reverse=True)
    
    final_ranking = []
    rank = 1
    for product in products_sorted:
        final_ranking.append({
            'rank': rank,
            'product_name': product['product_name'],
            'vendor': product['vendor'],
            'model_family': product['model_family'],
            'overall_score': product['match_score'],
            'requirements_match': product['requirements_match'],
            'key_strengths': product['reasoning'],  # Use the same reasoning from vendor match
            'concerns': product['limitations']
        })
        rank += 1
        
    # Apply standardization to the ranking result
    ranking_result = {'ranked_products': final_ranking}
    standardized_ranking = standardize_ranking_result(ranking_result)
    return standardized_ranking

def to_dict_if_pydantic(obj):
    """Helper function to safely convert Pydantic object to dict."""
    if hasattr(obj, 'dict'):
        return obj.dict()
    return obj


def _prepare_vendor_payloads(products_json_str, pdf_content_json_str):
    """Split combined vendor data into per-vendor payloads."""
    try:
        products = json.loads(products_json_str) if products_json_str else []
    except (json.JSONDecodeError, TypeError):
        logging.warning("Failed to parse products_json; defaulting to empty list.")
        products = []

    try:
        pdf_content = json.loads(pdf_content_json_str) if pdf_content_json_str else {}
    except (json.JSONDecodeError, TypeError):
        logging.warning("Failed to parse pdf_content_json; defaulting to empty dict.")
        pdf_content = {}

    # CSV filtering is now done earlier in the pipeline before PDF loading
    logging.info(f"[VENDOR_PAYLOADS] Processing {len(pdf_content)} vendors from filtered PDF content")

    vendor_payloads = {}

    # First, seed payloads from PDF content so only vendors with PDFs are considered
    for vendor_name, text in pdf_content.items():
        vendor_payloads[vendor_name] = {"products": [], "pdf_text": text}

    # Then, attach products to the corresponding PDF vendors (matching by exact vendor key)
    for product in products:
        if not isinstance(product, dict):
            continue
        vendor_name = product.get("vendor") or product.get("vendor_name") or "Unknown Vendor"
        if vendor_name in vendor_payloads:
            vendor_payloads[vendor_name]["products"].append(product)

    # Final filtering: keep only vendors that actually have non-empty PDF text
    vendor_payloads = {
        vendor: data
        for vendor, data in vendor_payloads.items()
        if data.get("pdf_text") and str(data["pdf_text"]).strip()
    }

    return vendor_payloads


def _invoke_vendor_chain_for_payload(
    vendor_chain,
    format_instructions,
    structured_requirements,
    vendor_name,
    vendor_data,
    applicable_standards=None,
    standards_specs=None,
):
    """Invoke vendor chain for a single vendor."""
    pdf_text = vendor_data.get("pdf_text")
    pdf_payload = json.dumps({vendor_name: pdf_text}, ensure_ascii=False) if pdf_text else json.dumps({})
    products_payload = json.dumps(vendor_data.get("products", []), ensure_ascii=False)
    
    # Format applicable standards for the prompt
    if applicable_standards and isinstance(applicable_standards, list) and len(applicable_standards) > 0:
        standards_list = "\n".join([f"- {std}" for std in applicable_standards])
    else:
        standards_list = "No applicable engineering standards specified."
    
    # Format standards specifications for the prompt
    if standards_specs and isinstance(standards_specs, dict) and len(standards_specs) > 0:
        specs_list = "\n".join([f"- {key}: {value}" for key, value in standards_specs.items()])
    else:
        specs_list = "No specific standards specifications provided."

    return vendor_chain.invoke(
        {
            "structured_requirements": structured_requirements,
            "products_json": products_payload,
            "pdf_content_json": pdf_payload,
            "format_instructions": format_instructions,
            "applicable_standards": standards_list,
            "standards_specs": specs_list,
        }
    )


def _run_parallel_vendor_analysis(
    structured_requirements,
    products_json_str,
    pdf_content_json_str,
    vendor_chain,
    format_instructions,
    applicable_standards=None,
    standards_specs=None,
):
    """Fan out vendor analysis so each vendor hits the LLM independently."""
    payloads = _prepare_vendor_payloads(products_json_str, pdf_content_json_str)

    # Debug logging to inspect how many vendors and which vendors are analyzed
    if not payloads:
        logging.warning("[VENDOR_ANALYSIS] No vendor payloads available for analysis.")
        return {"vendor_matches": [], "vendor_run_details": []}

    logging.info("[VENDOR_ANALYSIS] Preparing analysis for vendors: %s", list(payloads.keys()))
    
    if applicable_standards:
        logging.info("[VENDOR_ANALYSIS] Using applicable standards: %s", applicable_standards)

    max_workers_env = os.getenv("VENDOR_ANALYSIS_MAX_WORKERS")
    try:
        max_workers = int(max_workers_env) if max_workers_env else 5
    except ValueError:
        max_workers = 5

    max_workers = max(1, min(len(payloads), max_workers))

    logging.info(
        "[VENDOR_ANALYSIS] Using max_workers=%s for %s vendors",
        max_workers,
        len(payloads),
    )

    vendor_matches = []
    run_details = []

    def _worker(vendor, data):
        result_dict = None
        error = None
        max_retries = 3
        base_retry_delay = 15  # Start with 15 seconds
        
        logging.info("[VENDOR_ANALYSIS] START vendor=%s", vendor)
        
        for attempt in range(max_retries):
            try:
                result = _invoke_vendor_chain_for_payload(
                    vendor_chain,
                    format_instructions,
                    structured_requirements,
                    vendor,
                    data,
                    applicable_standards=applicable_standards,
                    standards_specs=standards_specs,
                )
                result_dict = to_dict_if_pydantic(result)
                break  # Success, exit retry loop
                
            except Exception as exc:
                error_msg = str(exc)
                is_rate_limit = "429" in error_msg or "Resource has been exhausted" in error_msg or "quota" in error_msg.lower()
                
                if is_rate_limit and attempt < max_retries - 1:
                    wait_time = base_retry_delay * (2 ** attempt)  # Exponential backoff: 15s, 30s, 60s
                    logging.warning(f"[VENDOR_ANALYSIS] Rate limit hit for {vendor}, retry {attempt + 1}/{max_retries} after {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    logging.error(f"Vendor analysis failed for {vendor}: {error_msg}")
                    error = error_msg
                    break
                    
        logging.info("[VENDOR_ANALYSIS] END   vendor=%s error=%s", vendor, error)
        return vendor, result_dict, error

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        # Submit threads with 2-second gaps between each submission
        for i, (vendor, data) in enumerate(payloads.items()):
            if i > 0:  # Add 2-second delay before submitting each subsequent thread
                time.sleep(2)
                logging.info("[VENDOR_ANALYSIS] Submitting thread %d for vendor: %s (after 2s delay)", i+1, vendor)
            else:
                logging.info("[VENDOR_ANALYSIS] Submitting thread %d for vendor: %s (no delay)", i+1, vendor)
            future = executor.submit(_worker, vendor, data)
            future_map[future] = vendor
        
        for future in as_completed(future_map):
            vendor = future_map[future]
            vendor_result = None
            error = None
            try:
                vendor, vendor_result, error = future.result()
            except Exception as exc:  # pragma: no cover
                error = str(exc)
                logging.error(f"Unexpected failure joining future for {vendor}: {exc}")

            if vendor_result and isinstance(vendor_result.get("vendor_matches"), list):
                for match in vendor_result["vendor_matches"]:
                    vendor_matches.append(to_dict_if_pydantic(match))
                run_details.append({"vendor": vendor, "status": "success"})
            else:
                entry = {
                    "vendor": vendor,
                    "status": "failed" if error else "empty",
                }
                if error:
                    entry["error"] = error
                run_details.append(entry)

    return {"vendor_matches": vendor_matches, "vendor_run_details": run_details}

def load_vendors_and_filter():
    """Load vendors from database and apply filtering with priority:
    1. User-specified vendors from input (highest priority)
    2. Strategy-based vendors from CSV/PDF
    3. Top 5 vendors fallback (lowest priority)
    """
    def _get_filtered_vendors(input_dict):
        # Get detected product type from previous step
        detected_product_type = input_dict.get("detected_product_type", None)
        logging.info(f"[VENDOR_LOADING] Using detected product type: {detected_product_type}")
        
        # Get vendors that match the detected product type from database
        if detected_product_type:
            all_db_vendors = get_vendors_for_product_type(detected_product_type)
            logging.info(f"[VENDOR_LOADING] Found {len(all_db_vendors)} vendors for product type '{detected_product_type}': {all_db_vendors}")
        else:
            # Fallback to all vendors if no product type detected
            all_db_vendors = get_available_vendors_from_mongodb()
            logging.info(f"[VENDOR_LOADING] No product type detected, found {len(all_db_vendors)} vendors in database: {all_db_vendors}")
        
        # ===========================================================================
        # PRIORITY 1: Check for user-specified vendors from input
        # These vendors were extracted by identify_instruments() or validation
        # ===========================================================================
        user_specified_vendors = input_dict.get("specified_vendors", [])
        
        # Also check session for specified vendors (from Project page workflow)
        from flask import session
        session_specified_vendors = session.get('specified_vendors', [])
        if session_specified_vendors and not user_specified_vendors:
            user_specified_vendors = session_specified_vendors
            logging.info(f"[VENDOR_PRIORITY] Retrieved user-specified vendors from session: {user_specified_vendors}")
        
        # ===========================================================================
        # PRIORITY 2: Check for strategy-based CSV vendor filter
        # ===========================================================================
        csv_vendor_filter = session.get('csv_vendor_filter', {})
        strategy_vendors = csv_vendor_filter.get('vendor_names', [])
        
        # If no session filter, check background task results from vendor_search_tasks
        if not strategy_vendors:
            try:
                from main import vendor_search_tasks, vendor_search_lock
                user_id = session.get('user_id')
                if user_id:
                    with vendor_search_lock:
                        task = vendor_search_tasks.get(user_id)
                    if task and task.get('status') == 'completed' and task.get('csv_vendor_filter'):
                        csv_vendor_filter = task['csv_vendor_filter']
                        strategy_vendors = csv_vendor_filter.get('vendor_names', [])
                        # Also store in session for future use
                        session['csv_vendor_filter'] = csv_vendor_filter
                        logging.info(f"[VENDOR_PRIORITY] Retrieved strategy vendors from background task: {len(strategy_vendors)} vendors")
            except ImportError:
                logging.warning("[VENDOR_PRIORITY] Could not import vendor_search_tasks from main")
            except Exception as e:
                logging.warning(f"[VENDOR_PRIORITY] Error checking background task: {e}")
        
        # ===========================================================================
        # APPLY PRIORITY LOGIC
        # ===========================================================================
        final_vendors = []
        priority_used = "none"
        
        if user_specified_vendors:
            # PRIORITY 1: User-specified vendors take highest precedence
            # User vendors OVERRIDE strategy (per user requirement)
            logging.info(f"[VENDOR_PRIORITY] User specified vendors: {user_specified_vendors} - these take priority!")
            priority_used = "user_specified"
            
            # Match user-specified vendors against database vendors using LLM for variations
            final_vendors = _match_vendors_with_llm(user_specified_vendors, all_db_vendors)
            
            if not final_vendors:
                # If no matches found, use user vendors as-is (LLM will handle during analysis)
                logging.warning(f"[VENDOR_PRIORITY] No DB vendors matched user-specified vendors, using as-is: {user_specified_vendors}")
                final_vendors = user_specified_vendors
                
        elif strategy_vendors:
            # PRIORITY 2: Strategy-based filtering (no user vendors specified)
            logging.info(f"[VENDOR_PRIORITY] Using strategy vendors: {strategy_vendors}")
            priority_used = "strategy"
            
            # Match strategy vendors against database vendors using LLM
            final_vendors = _match_vendors_with_llm(strategy_vendors, all_db_vendors)
            
            if not final_vendors:
                logging.warning("[VENDOR_PRIORITY] No DB vendors matched strategy, falling back to all DB vendors")
                final_vendors = all_db_vendors
                
        else:
            # PRIORITY 3: No user vendors and no strategy - use top 5 fallback
            logging.info("[VENDOR_PRIORITY] No user vendors or strategy - using top 5 fallback")
            priority_used = "top_5_fallback"
            
            if len(all_db_vendors) <= 5:
                final_vendors = all_db_vendors
            else:
                # Use LLM to discover top 5 vendors for this product type
                from loading import discover_top_vendors
                try:
                    top_vendors_result = discover_top_vendors(detected_product_type)
                    # Extract just vendor names from result
                    final_vendors = [v.get('vendor') if isinstance(v, dict) else v for v in top_vendors_result[:5]]
                    logging.info(f"[VENDOR_PRIORITY] Discovered top 5 vendors: {final_vendors}")
                except Exception as e:
                    logging.warning(f"[VENDOR_PRIORITY] Failed to discover top vendors: {e}, using first 5 from DB")
                    final_vendors = all_db_vendors[:5]
        
        # SAFETY CHECK: Ensure we have at least some vendors
        if not final_vendors:
            logging.warning("[VENDOR_PRIORITY] No vendors after all priority checks, using all DB vendors")
            final_vendors = all_db_vendors if all_db_vendors else ["Emerson", "ABB", "Yokogawa", "Endress+Hauser", "Honeywell"]
        
        logging.info(f"[VENDOR_PRIORITY] Final vendors (priority={priority_used}): {final_vendors}")
        
        # ===========================================================================
        # RETRIEVE MODEL FAMILIES from session (if specified by user)
        # ===========================================================================
        user_specified_model_families = input_dict.get("specified_model_families", [])
        session_model_families = session.get('specified_model_families', [])
        if session_model_families and not user_specified_model_families:
            user_specified_model_families = session_model_families
            logging.info(f"[MODEL_FAMILY] Retrieved user-specified model families from session: {user_specified_model_families}")
        
        if user_specified_model_families:
            logging.info(f"[MODEL_FAMILY] Will filter products to model families: {user_specified_model_families}")
        
        enriched = dict(input_dict)
        enriched["available_vendors"] = all_db_vendors
        enriched["filtered_vendors"] = final_vendors
        enriched["vendor_priority_used"] = priority_used
        enriched["specified_model_families"] = user_specified_model_families  # Pass to product loading
        return enriched
    
    return RunnableLambda(_get_filtered_vendors)


def _match_vendors_with_llm(target_vendors: list, db_vendors: list) -> list:
    """Match target vendor names against database vendors using LLM for fuzzy matching.
    
    Args:
        target_vendors: List of vendor names to match (from user input or strategy)
        db_vendors: List of vendor names from the database
        
    Returns:
        List of matched database vendor names
    """
    if not target_vendors or not db_vendors:
        return []
    
    matched_vendors = []
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        import concurrent.futures
        
        llm_flash_lite = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            temperature=0.0,  # Deterministic for matching
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        
        vendor_match_prompt = ChatPromptTemplate.from_template("""
You are an expert at matching vendor/company names in industrial automation.

Target Vendor Name: "{target_vendor}"

Available Database Vendors:
{db_vendors}

Instructions:
1. Find the BEST matching vendor from the database for the target vendor
2. Consider these as MATCHING:
   - Same company with different formatting (e.g., "Endress+Hauser" vs "Endress + Hauser")
   - Abbreviations (e.g., "ABB" vs "ABB Ltd", "E+H" vs "Endress+Hauser")
   - Parent/subsidiary relationships (e.g., "Siemens AG" vs "Siemens")
   - Division names (e.g., "Honeywell" matches "Honeywell Process Solutions")
   - Common variations (e.g., "Rosemount" matches "Emerson" as Rosemount is Emerson brand)
3. Only report a match if they clearly refer to the SAME company

Respond with ONLY a JSON object:
{{"matched_vendor": "<exact name from database vendors or null if no match>", "confidence": "high/medium/low/none"}}
""")
        
        chain = vendor_match_prompt | llm_flash_lite | StrOutputParser()
        
        def match_single_vendor(target_vendor):
            try:
                db_vendors_str = "\n".join([f"- {v}" for v in db_vendors])
                result = chain.invoke({
                    "target_vendor": target_vendor,
                    "db_vendors": db_vendors_str
                })
                
                # Parse JSON response
                import json
                result = result.strip()
                if result.startswith("```json"):
                    result = result[7:]
                elif result.startswith("```"):
                    result = result[3:]
                if result.endswith("```"):
                    result = result[:-3]
                result = result.strip()
                
                match_data = json.loads(result)
                matched = match_data.get("matched_vendor")
                confidence = match_data.get("confidence", "none")
                
                if matched and matched != "null" and confidence in ["high", "medium"]:
                    return target_vendor, matched, confidence
                return target_vendor, None, confidence
                
            except Exception as e:
                logging.warning(f"[VENDOR_MATCH] LLM error for vendor '{target_vendor}': {e}")
                return target_vendor, None, "error"
        
        # Run matching in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(match_single_vendor, v): v for v in target_vendors}
            
            for future in concurrent.futures.as_completed(futures):
                target_vendor, matched_vendor, confidence = future.result()
                if matched_vendor:
                    matched_vendors.append(matched_vendor)
                    logging.info(f"[VENDOR_MATCH] ✓ '{target_vendor}' → '{matched_vendor}' (confidence: {confidence})")
                else:
                    logging.info(f"[VENDOR_MATCH] ✗ No match for '{target_vendor}' (confidence: {confidence})")
                    
    except Exception as e:
        logging.error(f"[VENDOR_MATCH] Error in LLM vendor matching: {e}")
        # Fallback: try simple case-insensitive matching
        for target in target_vendors:
            target_lower = target.lower()
            for db_vendor in db_vendors:
                if target_lower in db_vendor.lower() or db_vendor.lower() in target_lower:
                    matched_vendors.append(db_vendor)
                    break
    
    # Remove duplicates while preserving order
    seen = set()
    unique_vendors = []
    for v in matched_vendors:
        if v not in seen:
            seen.add(v)
            unique_vendors.append(v)
    
    return unique_vendors

def load_filtered_pdf_content():
    """Load PDF content only for filtered vendors"""
    def _load_pdfs_for_filtered_vendors(input_dict):
        filtered_vendors = input_dict.get("filtered_vendors", [])
        
        if not filtered_vendors:
            logging.warning("[PDF_LOADING] No filtered vendors available for PDF loading")
            enriched = dict(input_dict)
            enriched["pdf_content_json"] = json.dumps({})
            return enriched
        
        logging.info(f"[PDF_LOADING] Loading PDFs for {len(filtered_vendors)} vendors: {filtered_vendors}")
        
        # Import the PDF loading function
        from loading import load_pdf_content_runnable
        pdf_loader = load_pdf_content_runnable()
        
        # Load all PDFs first
        pdf_result = pdf_loader.invoke(input_dict)
        all_pdf_content = json.loads(pdf_result.get("pdf_content_json", "{}"))
        
        # Filter PDF content to only include filtered vendors (with fuzzy vendor name matching)
        filtered_pdf_content = {}
        for vendor in filtered_vendors:
            # First try exact match
            if vendor in all_pdf_content:
                filtered_pdf_content[vendor] = all_pdf_content[vendor]
                logging.info(f"[PDF_LOADING] Loaded PDF for vendor: {vendor}")
            else:
                # Try fuzzy matching for case-insensitive and slight variations
                matched_key = None
                vendor_lower = vendor.lower()
                
                for pdf_vendor_key in all_pdf_content.keys():
                    pdf_vendor_lower = pdf_vendor_key.lower()
                    
                    # Check for exact case-insensitive match
                    if vendor_lower == pdf_vendor_lower:
                        matched_key = pdf_vendor_key
                        break
                    
                    # Check for fuzzy match (handles Endress+Hauser vs Endress+hauser)
                    from fuzzywuzzy import fuzz
                    similarity = fuzz.ratio(vendor_lower, pdf_vendor_lower)
                    if similarity >= 85:  # 85% similarity threshold
                        matched_key = pdf_vendor_key
                        break
                
                if matched_key:
                    filtered_pdf_content[vendor] = all_pdf_content[matched_key]
                    logging.info(f"[PDF_LOADING] Loaded PDF for vendor: {vendor} (matched with: {matched_key})")
                else:
                    logging.warning(f"[PDF_LOADING] No PDF found for vendor: {vendor}")
        
        logging.info(f"[PDF_LOADING] Successfully loaded PDFs for {len(filtered_pdf_content)} vendors")
        
        enriched = dict(pdf_result)
        enriched["pdf_content_json"] = json.dumps(filtered_pdf_content)
        enriched["filtered_vendors"] = filtered_vendors
        return enriched
    
    return RunnableLambda(_load_pdfs_for_filtered_vendors)

def create_analysis_chain(components, vendors_base_path=None):
    """Create analysis chain using MongoDB for all data loading with vendor pre-filtering"""
    product_loader = load_products_runnable()  # No path needed - uses MongoDB
    # --- Load vendors first, then filter, then load PDFs ---
    vendor_loader = load_vendors_and_filter()
    filtered_pdf_loader = load_filtered_pdf_content()
    
    def attach_parallel_vendor_analysis(input_dict):
        enriched = dict(input_dict)
        
        # Extract applicable standards if provided
        applicable_standards = enriched.get("applicable_standards", [])
        standards_specs = enriched.get("standards_specs", {})
        
        if applicable_standards:
            logging.info(f"[ANALYSIS_CHAIN] Passing applicable standards to vendor analysis: {applicable_standards}")
        
        enriched["vendor_analysis"] = _run_parallel_vendor_analysis(
            structured_requirements=enriched.get("structured_requirements"),
            products_json_str=enriched.get("products_json"),
            pdf_content_json_str=enriched.get("pdf_content_json"),
            vendor_chain=components['vendor_chain'],
            format_instructions=components['vendor_format_instructions'],
            applicable_standards=applicable_standards,
            standards_specs=standards_specs,
        )
        return enriched
    
    analysis_chain = (
        RunnablePassthrough.assign(
            structured_requirements=lambda x: components['requirements_chain'].invoke({"user_input": x["user_input"]})
        ).with_config(run_name="StructuredRequirementsGeneration")
        | RunnablePassthrough.assign(
            detected_product_type=lambda x: components['validation_chain'].invoke({
                "user_input": x["user_input"],
                "schema": json.dumps(load_requirements_schema(), indent=2),
                "format_instructions": components['validation_format_instructions']
            }).get('product_type', None)
        ).with_config(run_name="ProductTypeDetection")
        | product_loader.with_config(run_name="ProductDataLoading")
        | vendor_loader.with_config(run_name="VendorListLoading")
        # --- Load PDFs only for filtered vendors ---
        | filtered_pdf_loader.with_config(run_name="FilteredPDFLoading")
        | RunnableLambda(attach_parallel_vendor_analysis).with_config(run_name="ParallelVendorAnalysis")
        | RunnablePassthrough.assign(
            overall_ranking=lambda x: get_final_ranking(to_dict_if_pydantic(x.get("vendor_analysis", {})))
        ).with_config(run_name="FinalRanking")
    )
    
    return analysis_chain