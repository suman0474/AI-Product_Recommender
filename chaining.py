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
    validation_chain = validation_prompt | llm_flash | validation_parser
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
        'llm': llm_pro,  # General LLM for conversations and simple text generation
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
):
    """Invoke vendor chain for a single vendor."""
    pdf_text = vendor_data.get("pdf_text")
    pdf_payload = json.dumps({vendor_name: pdf_text}, ensure_ascii=False) if pdf_text else json.dumps({})
    products_payload = json.dumps(vendor_data.get("products", []), ensure_ascii=False)

    return vendor_chain.invoke(
        {
            "structured_requirements": structured_requirements,
            "products_json": products_payload,
            "pdf_content_json": pdf_payload,
            "format_instructions": format_instructions,
        }
    )


def _run_parallel_vendor_analysis(
    structured_requirements,
    products_json_str,
    pdf_content_json_str,
    vendor_chain,
    format_instructions,
):
    """Fan out vendor analysis so each vendor hits the LLM independently."""
    payloads = _prepare_vendor_payloads(products_json_str, pdf_content_json_str)

    # Debug logging to inspect how many vendors and which vendors are analyzed
    if not payloads:
        logging.warning("[VENDOR_ANALYSIS] No vendor payloads available for analysis.")
        return {"vendor_matches": [], "vendor_run_details": []}

    logging.info("[VENDOR_ANALYSIS] Preparing analysis for vendors: %s", list(payloads.keys()))

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
        # Submit threads with 10-second gaps between each submission
        for i, (vendor, data) in enumerate(payloads.items()):
            if i > 0:  # Add 10-second delay before submitting each subsequent thread
                time.sleep(10)
                logging.info("[VENDOR_ANALYSIS] Submitting thread %d for vendor: %s (after 10s delay)", i+1, vendor)
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
    """Load vendors from database and apply CSV filtering after product type detection"""
    def _get_filtered_vendors(input_dict):
        # Get detected product type from previous step
        detected_product_type = input_dict.get("detected_product_type", None)
        logging.info(f"[VENDOR_LOADING] Using detected product type: {detected_product_type}")
        
        # Get vendors that match the detected product type
        if detected_product_type:
            all_vendors = get_vendors_for_product_type(detected_product_type)
            logging.info(f"[VENDOR_LOADING] Found {len(all_vendors)} vendors for product type '{detected_product_type}': {all_vendors}")
        else:
            # Fallback to all vendors if no product type detected
            all_vendors = get_available_vendors_from_mongodb()
            logging.info(f"[VENDOR_LOADING] No product type detected, found {len(all_vendors)} vendors in database: {all_vendors}")
        
        # Check for CSV vendor filter in session
        from flask import session
        csv_vendor_filter = session.get('csv_vendor_filter', {})
        allowed_vendors = csv_vendor_filter.get('vendor_names', [])
        
        if allowed_vendors:
            logging.info(f"[CSV_FILTER] Restricting analysis to CSV vendors: {allowed_vendors}")
            
            # Import fuzzy matching
            from fuzzywuzzy import process
            from standardization_utils import standardize_vendor_name
            
            # Filter vendors using fuzzy matching (no LLM standardization needed!)
            filtered_vendors = []
            logging.info(f"[CSV_FILTER] Starting fuzzy matching with {len(all_vendors)} DB vendors against {len(allowed_vendors)} CSV vendors")
            
            try:
                # Process each DB vendor against all CSV vendors using fuzzy matching
                for i, db_vendor in enumerate(all_vendors):
                    logging.info(f"[CSV_FILTER] Processing DB vendor {i+1}/{len(all_vendors)}: '{db_vendor}'")
                    
                    # Use fuzzy matching to find best match among all CSV vendors (no standardization needed)
                    fuzzy_result = process.extractOne(db_vendor, allowed_vendors)
                    
                    if fuzzy_result and fuzzy_result[1] >= 70:  # 70% similarity threshold to avoid false matches
                        best_match = fuzzy_result[0]
                        best_score = fuzzy_result[1]
                        filtered_vendors.append(db_vendor)
                        logging.info(f"[CSV_FILTER] ✓ Matched DB vendor '{db_vendor}' with CSV vendor '{best_match}' (Score: {best_score}%)")
                    else:
                        logging.info(f"[CSV_FILTER] ✗ No match found for DB vendor '{db_vendor}' (best score: {fuzzy_result[1] if fuzzy_result else 0}%)")
                        
            except Exception as e:
                logging.error(f"[CSV_FILTER] Error in fuzzy matching: {e}")
                # Fallback to all vendors if fuzzy matching fails
                filtered_vendors = all_vendors
                logging.warning(f"[CSV_FILTER] Falling back to all {len(filtered_vendors)} vendors due to error")
            
            logging.info(f"[CSV_FILTER] After fuzzy filtering, will analyze vendors: {filtered_vendors}")
            
            # FALLBACK: If CSV filtering resulted in 0 vendors, use all database vendors
            if not filtered_vendors:
                logging.warning("[FALLBACK] CSV filtering resulted in 0 vendors, falling back to all database vendors")
                filtered_vendors = all_vendors
                logging.info(f"[FALLBACK] Using all {len(filtered_vendors)} database vendors for analysis: {filtered_vendors}")
        else:
            filtered_vendors = all_vendors
            logging.info("[CSV_FILTER] No CSV vendor filter found, analyzing all vendors")
        
        # FINAL SAFETY CHECK: Ensure we have at least some vendors for analysis
        if not filtered_vendors:
            logging.warning("[SAFETY_FALLBACK] No vendors available after all filtering")
            if all_vendors:
                filtered_vendors = all_vendors
                logging.info(f"[SAFETY_FALLBACK] Using {len(filtered_vendors)} product-specific vendors: {filtered_vendors}")
            else:
                logging.error("[SAFETY_FALLBACK] No vendors found for this product type - analysis cannot proceed")
        
        enriched = dict(input_dict)
        enriched["available_vendors"] = all_vendors
        enriched["filtered_vendors"] = filtered_vendors
        return enriched
    
    return RunnableLambda(_get_filtered_vendors)

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
        enriched["vendor_analysis"] = _run_parallel_vendor_analysis(
            structured_requirements=enriched.get("structured_requirements"),
            products_json_str=enriched.get("products_json"),
            pdf_content_json_str=enriched.get("pdf_content_json"),
            vendor_chain=components['vendor_chain'],
            format_instructions=components['vendor_format_instructions']
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