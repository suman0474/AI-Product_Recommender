# chaining.py
# Contains LangChain components setup and analysis chain creation
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
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
from dotenv import load_dotenv

# Import the OutputFixingParser
from langchain.output_parsers import OutputFixingParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Import standardization utilities
from standardization_utils import standardize_ranking_result


# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_langchain_components():
    # Use the old import since it's still functional (just shows deprecation warning)
    from langchain.callbacks import OpenAICallbackHandler
    callback_handler = OpenAICallbackHandler()
    
    # Create different models for different purposes
    # Gemini 2.5 Flash for simple conversations and text generation
    llm_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Gemini 2.5 Flash Lite for generating descriptions of schema keys (lightweight tasks)
    llm_flash_lite = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1, google_api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Gemini 2.5 Pro for complex analysis tasks - uses second API key
    llm_pro = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1, google_api_key=os.getenv("GOOGLE_API_KEY1"))
    
    memory = ConversationBufferWindowMemory(k=1, memory_key="chat_history", return_messages=True)
    
    # 1. Instantiate the base parsers
    validation_parser = JsonOutputParser(pydantic_object=RequirementValidation)
    vendor_parser = JsonOutputParser(pydantic_object=VendorAnalysis)
    ranking_parser = JsonOutputParser(pydantic_object=OverallRanking)
    str_parser = StrOutputParser()
    
    # 2. Wrap the JSON parsers with OutputFixingParser for robustness
    # Use Flash model for validation and additional requirements (faster)
    validation_fixing_parser = OutputFixingParser.from_llm(parser=validation_parser, llm=llm_flash)
    # Use Pro model only for complex vendor analysis and ranking
    vendor_fixing_parser = OutputFixingParser.from_llm(parser=vendor_parser, llm=llm_pro)
    ranking_fixing_parser = OutputFixingParser.from_llm(parser=ranking_parser, llm=llm_pro)
    
    # 3. Use the appropriate models for different tasks
    # Fast tasks use Flash model
    validation_chain = validation_prompt | llm_flash | validation_fixing_parser
    requirements_chain = requirements_prompt | llm_flash | str_parser
    
    # Final analysis tasks use Pro model (vendor analysis and ranking)
    vendor_chain = vendor_prompt | llm_pro | vendor_fixing_parser
    ranking_chain = ranking_prompt | llm_pro | ranking_fixing_parser
    
    # --- NEW CHAIN FOR ADDITIONAL REQUIREMENTS ---
    additional_requirements_parser = JsonOutputParser(pydantic_object=RequirementValidation)
    additional_requirements_fixing_parser = OutputFixingParser.from_llm(parser=additional_requirements_parser, llm=llm_flash)
    additional_requirements_chain = additional_requirements_prompt | llm_flash | additional_requirements_fixing_parser
    
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
        'llm': llm_flash,  # Default LLM for conversations (backward compatibility)
        'llm_flash': llm_flash,  # For conversations and simple text generation
        'llm_flash_lite': llm_flash_lite,  # For generating schema key descriptions
        'llm_pro': llm_pro,  # For complex analysis tasks
        'memory': memory,
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
        # Ensure requirements_match is a boolean. Default to False if missing.
        products.append({
            'product_name': product.get('product_name', ''),
            'vendor': product.get('vendor', ''),
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
        try:
            logging.info("[VENDOR_ANALYSIS] START vendor=%s", vendor)
            result = _invoke_vendor_chain_for_payload(
                vendor_chain,
                format_instructions,
                structured_requirements,
                vendor,
                data,
            )
            result_dict = to_dict_if_pydantic(result)
        except Exception as exc:
            logging.error(f"Vendor analysis failed for {vendor}: {exc}")
            error = str(exc)
        finally:
            logging.info("[VENDOR_ANALYSIS] END   vendor=%s error=%s", vendor, error)
        return vendor, result_dict, error

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_worker, vendor, data): vendor for vendor, data in payloads.items()
        }
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

# chaining.py
# ... (add this to your imports)
from loading import load_pdf_content_runnable

# ... (inside the file)

def create_analysis_chain(components, vendors_base_path=None):
    """Create analysis chain using MongoDB for all data loading"""
    product_loader = load_products_runnable()  # No path needed - uses MongoDB
    # --- Load PDFs from MongoDB ---
    pdf_loader = load_pdf_content_runnable()  # No path needed - uses MongoDB
    
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
        # --- NEW: Add the PDF loading step to the chain ---
        | pdf_loader.with_config(run_name="LocalPDFLoading")
        | RunnableLambda(attach_parallel_vendor_analysis).with_config(run_name="ParallelVendorAnalysis")
        | RunnablePassthrough.assign(
            overall_ranking=lambda x: get_final_ranking(to_dict_if_pydantic(x.get("vendor_analysis", {})))
        ).with_config(run_name="FinalRanking")
    )
    
    return analysis_chain