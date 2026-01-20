import asyncio
from datetime import datetime
import json
import logging
import re
import os
import urllib.parse
import threading
import csv
import tempfile
from io import BytesIO
from functools import wraps
from unittest import result
 
# =========================
# Flask & Extensions
# =========================
from flask import (
    Flask,
    request,
    jsonify,
    session,
    send_file
)
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
from werkzeug.utils import secure_filename
 
# =========================
# Environment & Config
# =========================
from dotenv import load_dotenv
 
# =========================
# External Services & APIs
# =========================
import requests
import redis
from serpapi import GoogleSearch
from googleapiclient.discovery import build
 
# =========================
# Fuzzy Matching
# =========================
from fuzzywuzzy import fuzz, process
 
# =========================
# Authentication & Logging
# =========================
from auth_models import db, User, Log
from auth_utils import hash_password, check_password
 
# =========================
# MongoDB & Project Management
# =========================
from mongo_project_manager import mongo_project_manager
from mongodb_config import get_mongodb_connection
from mongodb_utils import (
    get_schema_from_mongodb,
    get_json_from_mongodb,
    MongoDBFileManager,
    mongodb_file_manager
)
from mongodb_projects import (
    save_project_to_mongodb,
    get_user_projects_from_mongodb,
    get_project_details_from_mongodb,
    delete_project_from_mongodb
)
 # Import Standard RAG functions
from standard_rag import (
    extract_and_store_standards_text,
    get_standards_for_category
)

# =========================
# LangChain / LLM Pipeline
# =========================
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

from chaining import setup_langchain_components, create_analysis_chain
from loading import (
    load_requirements_schema,
    build_requirements_schema_from_web
)
 
# =========================
# Advanced Parameters
# =========================
from advanced_parameters import discover_advanced_parameters
# Load environment variables
load_dotenv()
# =========================================================================
# === FLASK APP CONFIGURATION ===
# =========================================================================
app = Flask(__name__, static_folder="static")
# ----------------------------
# 1) Base secret + session defaults (use env vars; no hard-coded secrets)
# ----------------------------
app.secret_key = os.getenv("SECRET_KEY", "fallback-secret-key-for-development")
# Detect production environment (Railway or FLASK_ENV=production)
IS_PRODUCTION = (
    os.getenv("FLASK_ENV") == "production"
    or bool(os.getenv("RAILWAY_ENVIRONMENT"))
    or os.getenv("ENV") == "production"
)
# Default sensible session config
app.config["SESSION_PERMANENT"] = False
# Session storage: use Redis in production (Railway), filesystem in development
if IS_PRODUCTION:
    app.config["SESSION_TYPE"] = "redis"
    app.config["SESSION_PERMANENT"] = True  # sessions can expire per your session settings
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            app.config["SESSION_REDIS"] = redis.from_url(redis_url)
        except Exception as e:
            logging.warning(f"Failed to parse REDIS_URL or connect to Redis: {e}")
    else:
        logging.warning("REDIS_URL not found in environment; sessions may not be persistent in production.")
else:
    # development
    app.config["SESSION_TYPE"] = "filesystem"
    app.config["SESSION_PERMANENT"] = False
# ----------------------------
# 2) Cookie flags (critical for cross-site auth)
# ----------------------------
# In production we must set SameSite=None and Secure=True so cookies are sent from en-genie -> railway
if IS_PRODUCTION:
    app.config["SESSION_COOKIE_SAMESITE"] = "None"
    app.config["SESSION_COOKIE_SECURE"] = True
    app.config["SESSION_COOKIE_HTTPONLY"] = True
else:
    # Local dev: Lax or None depending on your dev flow; Lax is safer for local debugging
    app.config["SESSION_COOKIE_SAMESITE"] = os.getenv("DEV_SESSION_COOKIE_SAMESITE", "Lax")
    app.config["SESSION_COOKIE_SECURE"] = False
    app.config["SESSION_COOKIE_HTTPONLY"] = True
# Optional: expose a readable setting for debugging
logging.info(f"IS_PRODUCTION={IS_PRODUCTION}, SESSION_COOKIE_SAMESITE={app.config.get('SESSION_COOKIE_SAMESITE')}")
# ----------------------------
# 3) Initialize server-side session support
# ----------------------------
Session(app)
# ----------------------------
# 4) Manual CORS handling (after session config)
# ----------------------------
allowed_origins = [
    "https://en-genie.vercel.app",  # Your production frontend
    "http://localhost:8080",        # Add your specific local dev port if needed
    "http://localhost:5173",
    "http://localhost:3000"
]
# Make sure supports_credentials=True so cookies are allowed in cross-origin requests
CORS(
    app,
    origins=allowed_origins,
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    expose_headers=["Content-Type", "Authorization"]
)
logging.basicConfig(level=logging.INFO)
# --- DYNAMIC DATABASE CONFIGURATION (unchanged from your code) ---
database_url = os.getenv('DATABASE_URL')
if database_url:
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url.replace("mysql://", "mysql+pymysql://")
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# initialize db
db.init_app(app)


# =========================================================================
# === HELPER FUNCTIONS AND UTILITIES ===
# =========================================================================
# Import our updated functions from test.py
from test import (
    extract_data_from_pdf,
    send_to_language_model,
    aggregate_results,
    generate_dynamic_path,
    split_product_types,
    save_json
    # Removed: identify_and_save_product_image - no longer needed with API-based images
)

# Import standardization utilities
from standardization_utils import (
    standardize_vendor_analysis_result,
    standardize_ranking_result,
    enhance_submodel_mapping,
    standardize_product_image_mapping,
    create_standardization_report,
    update_existing_vendor_files_with_standardization,
    standardize_vendor_name
)

# =========================================================================
# === BACKGROUND TASK STORAGE FOR VENDOR SEARCH ===
# =========================================================================
# Store background task results (keyed by user_id)
vendor_search_tasks = {}
vendor_search_lock = threading.Lock()


# Initialize LangChain components
try:
    components = setup_langchain_components()
    analysis_chain = create_analysis_chain(components)  # Uses MongoDB, no local paths needed
    logging.info("LangChain components initialized.")
except Exception as e:
    logging.error(f"LangChain components initialization failed: {e}")
    components = None
    analysis_chain = None

def convert_keys_to_camel_case(obj):
    """Recursively converts dictionary keys from snake_case to camelCase."""
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            camel_key = re.sub(r'_([a-z])', lambda m: m.group(1).upper(), key)
            new_dict[camel_key] = convert_keys_to_camel_case(value)
        return new_dict
    elif isinstance(obj, list):
        return [convert_keys_to_camel_case(item) for item in obj]
    return obj

def apply_standardization_to_response(data):
    """
    Apply appropriate standardization to response data based on its content.
    This function detects the type of data and applies the correct standardization.
    Only applies LLM-based standardization for critical analysis endpoints.
    """
    if not isinstance(data, dict):
        return data
    
    try:
        # Only apply LLM-based standardization for analysis results
        # Skip for basic endpoints like /vendors to prevent connection issues
        
        # For nested dictionaries, apply standardization recursively to relevant parts
        standardized = data.copy()
        for key, value in data.items():
            if key in ["vendor_analysis", "vendorAnalysis"] and isinstance(value, dict):
                try:
                    standardized[key] = standardize_vendor_analysis_result(value)
                except Exception as e:
                    logging.warning(f"Failed to standardize vendor analysis: {e}")
                    standardized[key] = value
            elif key in ["overall_ranking", "overallRanking", "ranking"] and isinstance(value, dict):
                try:
                    standardized[key] = standardize_ranking_result(value)
                except Exception as e:
                    logging.warning(f"Failed to standardize ranking: {e}")
                    standardized[key] = value
            elif key in ["vendors"] and isinstance(value, list):
                # Apply basic vendor name standardization only
                standardized_vendors = []
                for vendor in value:
                    if isinstance(vendor, dict) and "name" in vendor:
                        vendor_copy = vendor.copy()
                        try:
                            vendor_copy["name"] = standardize_vendor_name(vendor.get("name", ""))
                        except Exception as e:
                            logging.warning(f"Failed to standardize vendor name: {e}")
                            vendor_copy["name"] = vendor.get("name", "")
                        standardized_vendors.append(vendor_copy)
                    else:
                        standardized_vendors.append(vendor)
                standardized[key] = standardized_vendors
        return standardized
    
    except Exception as e:
        logging.warning(f"Standardization failed for response data: {e}")
        return data
    
    return data

def standardized_jsonify(data, status_code=200):
    """
    Enhanced jsonify function that applies standardization before converting to camelCase.
    Use this instead of jsonify() for responses containing vendor/product data.
    """
    try:
        # Apply standardization first (with timeout protection)
        standardized_data = apply_standardization_to_response(data)
        
        # Then convert to camelCase for frontend compatibility
        camel_case_data = convert_keys_to_camel_case(standardized_data)
        
        return jsonify(camel_case_data), status_code
    except Exception as e:
        logging.error(f"Failed to apply standardization in jsonify wrapper: {e}")
        # Fallback to regular conversion if standardization fails
        try:
            camel_case_data = convert_keys_to_camel_case(data)
            return jsonify(camel_case_data), status_code
        except Exception as fallback_error:
            logging.error(f"Even fallback conversion failed: {fallback_error}")
            # Last resort: return data as-is
            return jsonify(data), status_code

def prettify_req(req):
    return req.replace('_', ' ').replace('-', ' ').title()

def flatten_schema(schema_dict):
    flat = {}
    for k, v in schema_dict.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                flat[subk] = subv
        else:
            flat[k] = v
    return flat

def login_required(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Unauthorized: Please log in"}), 401
        return func(*args, **kwargs)
    return decorated_function

ALLOWED_EXTENSIONS = {"pdf"}

def allowed_file(filename: str):
    """Check if the uploaded filename has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_strategy_from_file(file_bytes: bytes, filename: str, user_id: int) -> dict:
    """
    Extract structured strategy data from a file using Gemini 2.5 Flash LLM.
    
    Sends the ENTIRE document to the LLM in ONE call for extraction.
    
    Args:
        file_bytes: Raw file content
        filename: Original filename
        user_id: User ID for MongoDB storage
        
    Returns:
        dict with success status, extracted data, and document_id if stored
    """
    from file_extraction_utils import extract_text_from_file
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    def clean_llm_response(response: str) -> str:
        """Clean markdown code blocks from LLM response"""
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()
    
    try:
        logging.info(f"[STRATEGY_EXTRACT] Processing file: {filename} ({len(file_bytes)} bytes)")
        
        # Extract text from file
        extraction_result = extract_text_from_file(file_bytes, filename)
        
        if not extraction_result['success']:
            logging.warning(f"[STRATEGY_EXTRACT] Failed to extract text from {filename}")
            return {
                "success": False,
                "error": f"Could not extract text from {extraction_result['file_type']} file",
                "file_type": extraction_result['file_type']
            }
        
        extracted_text = extraction_result['extracted_text']
        logging.info(f"[STRATEGY_EXTRACT] ✓ Extracted {extraction_result['character_count']} characters")
        
        # Initialize Gemini 2.5 Flash
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
        
        if not GOOGLE_API_KEY:
            logging.error("[STRATEGY_EXTRACT] GOOGLE_API_KEY not configured")
            return {
                "success": False,
                "error": "LLM API key not configured"
            }
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY,
        )
        
        # Strategy extraction prompt
        strategy_extraction_prompt = """
You are an expert at extracting structured procurement strategy data from documents.

Analyze the following document content and extract ALL vendor/supplier strategy information.
For each vendor entry found, extract the following fields:
- vendor_name: Name of the vendor/supplier/manufacturer
- category: Product category (e.g., "Pressure Transmitter", "Flow Meter", "Control Valve", "Temperature Transmitter")
- subcategory: Specific product type or subcategory within the category
- stratergy: Procurement strategy (e.g., "Cost optimization", "Life-cycle cost evaluation", "Sustainability and green procurement", "Dual sourcing", "Framework agreement", "Single sourcing", "Quality-focused")

Document Content:
{document_text}

Extract ALL vendor entries from the document. If a field is not explicitly mentioned, use an empty string.

Return ONLY a valid JSON array of objects with this structure:
[
  {{
    "vendor_name": "<vendor name>",
    "category": "<product category>",
    "subcategory": "<product subcategory>",
    "stratergy": "<procurement strategy>"
  }},
  ...
]

Important:
1. Include ALL vendor/strategy entries found in the ENTIRE document
2. If the document contains a table or list of vendors, extract EVERY row/item as a separate entry
3. If strategy information is not provided for a vendor, leave the "stratergy" field empty
4. Return ONLY the JSON array, no additional text or explanation
5. If no valid vendor data can be extracted, return an empty array: []
"""

        prompt = ChatPromptTemplate.from_template(strategy_extraction_prompt)
        chain = prompt | llm | StrOutputParser()
        
        # Configure limits
        MAX_SINGLE_CALL_CHARS = 2000000  # 2M chars (~500K tokens) - safe limit for single call
        CHUNK_SIZE = 100000  # 100K chars per chunk for fallback
        
        strategy_data = []
        
        # Check if document is small enough for single call
        if len(extracted_text) <= MAX_SINGLE_CALL_CHARS:
            # Try sending ENTIRE document in ONE call
            logging.info(f"[STRATEGY_EXTRACT] Sending ENTIRE document ({len(extracted_text)} chars) to LLM in single call...")
            
            try:
                llm_response = chain.invoke({"document_text": extracted_text})
                cleaned_response = clean_llm_response(llm_response)
                strategy_data = json.loads(cleaned_response)
                
                if not isinstance(strategy_data, list):
                    logging.warning(f"[STRATEGY_EXTRACT] LLM returned non-array response: {type(strategy_data)}")
                    strategy_data = []
                    
                logging.info(f"[STRATEGY_EXTRACT] ✓ Single call extracted {len(strategy_data)} strategy entries")
                
            except Exception as single_call_error:
                logging.warning(f"[STRATEGY_EXTRACT] Single call failed, falling back to chunked: {single_call_error}")
                strategy_data = []  # Will trigger chunked processing below
        
        # Fallback to chunked processing for very large files or if single call failed
        if len(strategy_data) == 0 and len(extracted_text) > 0:
            logging.info(f"[STRATEGY_EXTRACT] Using chunked processing for large document ({len(extracted_text)} chars)...")
            
            # Split text into chunks at line boundaries
            lines = extracted_text.split('\n')
            chunks = []
            current_chunk = []
            current_size = 0
            
            for line in lines:
                line_size = len(line) + 1
                if current_size + line_size > CHUNK_SIZE and current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_size = line_size
                else:
                    current_chunk.append(line)
                    current_size += line_size
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            
            logging.info(f"[STRATEGY_EXTRACT] Split into {len(chunks)} chunks for processing")
            
            # Process chunks in parallel using ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def process_chunk(chunk_data):
                chunk_num, chunk_text = chunk_data
                try:
                    response = chain.invoke({"document_text": chunk_text})
                    cleaned = clean_llm_response(response)
                    data = json.loads(cleaned)
                    if isinstance(data, list):
                        logging.info(f"[STRATEGY_EXTRACT] Chunk {chunk_num} extracted {len(data)} entries")
                        return data
                except Exception as e:
                    logging.warning(f"[STRATEGY_EXTRACT] Chunk {chunk_num} failed: {e}")
                return []
            
            all_entries = []
            max_workers = min(5, len(chunks))  # Limit parallel workers
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_chunk, (i, chunk)): i for i, chunk in enumerate(chunks, 1)}
                for future in as_completed(futures):
                    entries = future.result()
                    all_entries.extend(entries)
            
            # Deduplicate entries
            seen = set()
            for entry in all_entries:
                key = (
                    entry.get('vendor_name', '').lower().strip(),
                    entry.get('category', '').lower().strip(),
                    entry.get('subcategory', '').lower().strip()
                )
                if key not in seen:
                    seen.add(key)
                    strategy_data.append(entry)
            
            logging.info(f"[STRATEGY_EXTRACT] ✓ Chunked processing extracted {len(strategy_data)} unique entries")

        logging.info(f"[STRATEGY_EXTRACT] ✓ Total extracted: {len(strategy_data)} strategy entries")
        
        # Store in MongoDB if we have data
        if strategy_data and len(strategy_data) > 0:
            try:
                conn = get_mongodb_connection()
                stratergy_collection = conn['collections']['stratergy']
                
                strategy_document = {
                    "user_id": user_id,
                    "filename": filename,
                    "file_type": extraction_result['file_type'],
                    "data": strategy_data,
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "character_count": extraction_result['character_count'],
                    "entry_count": len(strategy_data)
                }
                
                result = stratergy_collection.insert_one(strategy_document)
                
                logging.info(f"[STRATEGY_EXTRACT] ✓ Stored {len(strategy_data)} entries in MongoDB with ID: {result.inserted_id}")
                
                return {
                    "success": True,
                    "document_id": str(result.inserted_id),
                    "entry_count": len(strategy_data),
                    "extracted_data": strategy_data,
                    "file_type": extraction_result['file_type']
                }
                
            except Exception as e:
                logging.error(f"[STRATEGY_EXTRACT] MongoDB storage failed: {e}")
                return {
                    "success": False,
                    "error": f"Failed to store strategy data: {str(e)}"
                }
        else:
            logging.warning("[STRATEGY_EXTRACT] No strategy data extracted from document")
            return {
                "success": False,
                "error": "No vendor strategy data could be extracted from the document",
                "extracted_text_preview": extracted_text[:500] if len(extracted_text) > 500 else extracted_text
            }
            
    except json.JSONDecodeError as e:
        logging.error(f"[STRATEGY_EXTRACT] Failed to parse LLM response: {e}")
        return {
            "success": False,
            "error": "Failed to parse strategy data from document"
        }
    except Exception as e:
        logging.error(f"[STRATEGY_EXTRACT] Extraction failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }



def extract_strategy_from_file_background(file_bytes: bytes, filename: str, user_id: int):
    """
    Run strategy extraction in background thread.
    
    This function spawns a new thread to process the file so it doesn't block
    the main request/response flow (signup, profile update, etc.)
    
    Args:
        file_bytes: Raw file content
        filename: Original filename
        user_id: User ID for MongoDB storage
    """
    def background_task():
        try:
            logging.info(f"[BACKGROUND] Starting strategy extraction for user {user_id}: {filename}")
            result = extract_strategy_from_file(file_bytes, filename, user_id)
            
            if result.get('success'):
                logging.info(f"[BACKGROUND] ✓ Successfully extracted {result.get('entry_count', 0)} strategy entries for user {user_id}")
            else:
                logging.warning(f"[BACKGROUND] Strategy extraction failed for user {user_id}: {result.get('error')}")
                
        except Exception as e:
            logging.error(f"[BACKGROUND] Error in background strategy extraction for user {user_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Start extraction in background thread
    extraction_thread = threading.Thread(target=background_task, daemon=True)
    extraction_thread.start()
    logging.info(f"[BACKGROUND] Started background extraction thread for user {user_id}: {filename}")



# =========================================================================
# === STANDARDS TEXT EXTRACTION BACKGROUND WRAPPER ===
# =========================================================================

def extract_and_store_standards_text_background(file_bytes: bytes, filename: str, user_id: int):
    """
    Run standards text extraction in background thread.
    Uses the extract_and_store_standards_text function from standard_rag module.
    """
    def background_task():
        try:
            logging.info(f"[BACKGROUND] Starting standards text extraction for user {user_id}: {filename}")
            result = extract_and_store_standards_text(file_bytes, filename, user_id)
            
            if result.get('success'):
                logging.info(f"[BACKGROUND] ✓ Successfully extracted and stored standards text for user {user_id}")
            else:
                logging.warning(f"[BACKGROUND] Standards text extraction failed for user {user_id}: {result.get('error')}")
                
        except Exception as e:
            logging.error(f"[BACKGROUND] Error in background standards extraction for user {user_id}: {e}")
            import traceback
            traceback.print_exc()
    
    extraction_thread = threading.Thread(target=background_task, daemon=True)
    extraction_thread.start()
    logging.info(f"[BACKGROUND] Started background standards extraction thread for user {user_id}: {filename}")



@app.route("/api/intent", methods=["POST"])
@login_required
def api_intent():
    """
    Workflow-focused intent classifier for the Solution page multi-step validation flow.
    
    NOTE: Greeting, chitchat, questions, and out-of-scope inputs are now handled by 
    /api/route-classifier. This API focuses ONLY on:
    - productRequirements: User is providing technical specs/requirements
    - workflow: User is responding to workflow prompts (yes/no, providing data)
    
    INTENT TYPES:
    -------------
    1. "productRequirements" - User is providing new industrial requirements
    2. "workflow" - User is continuing the workflow (yes/no, additional specs)
    
    This API manages the workflow step progression:
    initialInput → awaitMissingInfo → awaitAdditionalAndLatestSpecs → awaitAdvancedSpecs → showSummary → finalAnalysis
    """
    if not components or not components.get("llm"):
        return jsonify({"error": "LLM component not ready."}), 503

    data = request.get_json(force=True)
    user_input = data.get("userInput", "").strip()
    if not user_input:
        return jsonify({"error": "userInput is required"}), 400

    # Get search session ID if provided (for session isolation)
    search_session_id = data.get("search_session_id", "default")

    # Get current workflow state from session (session-isolated)
    current_step_key = f'current_step_{search_session_id}'
    current_intent_key = f'current_intent_{search_session_id}'
    current_step = session.get(current_step_key, None)
    current_intent = session.get(current_intent_key, None)
    
    # Debug logging to track session state
    logging.info(f"[INTENT] Session state: step={current_step}, intent={current_intent}, user_input='{user_input[:50]}...' if len > 50 else '{user_input}'")

    # === Workflow-Focused Classification Prompt ===
    # Greeting, chitchat, questions, and out-of-scope are handled by route-classifier
    prompt = f"""
You are Engenie - classifying user input for a step-based industrial product validation workflow.

Current workflow context:
- Current step: {current_step or "None"}
- Current intent: {current_intent or "None"}

User message: "{user_input}"

Return ONLY a JSON object with these keys:
1. "intent": one of ["productRequirements", "workflow"]
2. "nextStep": one of ["initialInput", "awaitMissingInfo", "awaitAdditionalAndLatestSpecs", "awaitAdvancedSpecs", "showSummary", "finalAnalysis", null]
3. "resumeWorkflow": true/false
4. "stayAtStep": true/false (optional)

**Classification Rules:**

**PRIORITY 1 - Product Requirements:**
- User is PROVIDING/STATING technical specs, product needs, measurements, materials, standards
- Look for: industrial equipment (transmitter, valve, pump, sensor, meter, gauge, controller, actuator)
- Look for: specifications (psi, bar, mm, inch, 4-20mA, HART, 316SS, ANSI, ASME, temperature, pressure, flow)
- Intent: "productRequirements", nextStep: "initialInput"
- Examples: "I need a pressure transmitter 0-100 psi", "valve with 316SS flanged connection"

**PRIORITY 2 - Workflow Continuation:**
- User is responding to workflow prompts with yes/no, confirmations, or providing additional data
- Determine nextStep based on current step:

  **At awaitMissingInfo or awaitMandatory:**
  - YES/proceed/continue/skip/okay → nextStep: "awaitAdditionalAndLatestSpecs"
  - NO/not yet/want to add → nextStep: "awaitMissingInfo", stayAtStep: true
  - User provides data/specs → nextStep: "awaitMissingInfo"

  **At awaitAdditionalAndLatestSpecs:**
  - YES/add specs → nextStep: "awaitAdvancedSpecs"
  - NO/skip → nextStep: "showSummary"

  **At awaitAdvancedSpecs:**
  - NO/skip/done → nextStep: "showSummary"
  - User selects specs → nextStep: "awaitAdvancedSpecs"

Workflow Steps:
initialInput → awaitMissingInfo → awaitAdditionalAndLatestSpecs → awaitAdvancedSpecs → showSummary → finalAnalysis

Respond ONLY with valid JSON, no additional text.
"""

    try:
        # Use LLM for classification
        full_prompt = ChatPromptTemplate.from_template(prompt)
        response_chain = full_prompt | components['llm'] | StrOutputParser()
        llm_response = response_chain.invoke({"user_input": user_input})

        # Clean and parse JSON response
        cleaned_response = llm_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        try:
            result_json = json.loads(cleaned_response)
        except json.JSONDecodeError:
            logging.warning(f"[INTENT] Failed to parse LLM JSON response: {llm_response[:200]}")
            result_json = {"intent": "workflow", "nextStep": current_step, "resumeWorkflow": True}

        # Validate required fields - default to workflow if not specified
        if "intent" not in result_json or result_json["intent"] not in ["productRequirements", "workflow"]:
            result_json["intent"] = "workflow"
        if "nextStep" not in result_json:
            result_json["nextStep"] = current_step
        if "resumeWorkflow" not in result_json:
            result_json["resumeWorkflow"] = True

        # === SAFETY CHECK: Prevent incorrect workflow regression ===
        mid_workflow_steps = ["awaitMissingInfo", "awaitAdditionalAndLatestSpecs", "awaitAdvancedSpecs", "showSummary", "finalAnalysis"]
        
        if current_step in mid_workflow_steps:
            if result_json.get("nextStep") == "initialInput" and result_json.get("intent") != "productRequirements":
                logging.warning(f"[INTENT] Prevented workflow regression: Keeping current step {current_step}")
                result_json["nextStep"] = current_step
                result_json["intent"] = "workflow"
                result_json["resumeWorkflow"] = True
            elif result_json.get("nextStep") is None:
                result_json["nextStep"] = current_step

        # Update session based on classification
        if result_json.get("intent") == "productRequirements":
            session[f'current_step_{search_session_id}'] = 'initialInput'
            session[f'current_intent_{search_session_id}'] = 'productRequirements'
        elif result_json.get("intent") == "workflow" and result_json.get("nextStep"):
            session[f'current_step_{search_session_id}'] = result_json.get("nextStep")
            session[f'current_intent_{search_session_id}'] = 'workflow'
        
        logging.info(f"[INTENT] Classification: {result_json}")
        return jsonify(result_json), 200

    except Exception as e:
        logging.exception("Intent classification failed.")
        return jsonify({"error": str(e), "intent": "workflow", "nextStep": current_step, "resumeWorkflow": True}), 500

@app.route("/api/update_profile", methods=["POST"])
@login_required
def api_update_profile():
    """
    Updates the user's profile (first name, last name, username).
    Also handles strategy and standards document uploads.
    """
    try:
        # Handle both JSON and Multipart data
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            data = request.form
            file = request.files.get('document')
            standards_file = request.files.get('standards_document')
        else:
            data = request.get_json(force=True)
            file = None
            standards_file = None

        user_id = session.get('user_id')
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({"error": "User not found"}), 404
            
        first_name = data.get("first_name")
        last_name = data.get("last_name")
        username = data.get("username")
        company_name = data.get("company_name")
        location = data.get("location")
        
        if first_name is not None:
            user.first_name = first_name.strip()
        if last_name is not None:
            user.last_name = last_name.strip()
            
        if username is not None:
            new_username = username.strip()
            if new_username and new_username != user.username:
                existing_user = User.query.filter_by(username=new_username).first()
                if existing_user:
                    return jsonify({"error": "Username already taken"}), 400
                user.username = new_username

        if company_name is not None:
            user.company_name = company_name.strip()
        if location is not None:
            user.location = location.strip()

        # Handle Strategy File Upload - Store in GridFS AND extract strategy data using LLM IN BACKGROUND
        strategy_file_uploaded = False
        if file:
            try:
                filename = secure_filename(file.filename)
                file_data = file.read()
                metadata = {
                    'filename': filename,
                    'content_type': file.content_type,
                    'uploaded_by_username': user.username,
                    'collection_type': 'strategy_documents'
                }
                # Upload to GridFS for raw file storage (immediate)
                document_file_id = mongodb_file_manager.upload_to_mongodb(file_data, metadata)
                logging.info(f"Uploaded strategy document for user {user.username}: {document_file_id}")
                
                # Extract strategy data using LLM IN BACKGROUND
                # This runs in a separate thread so user gets immediate response
                logging.info(f"[UPDATE_PROFILE] Queueing background strategy extraction for user {user.id}: {filename}")
                extract_strategy_from_file_background(file_data, filename, user_id)
                strategy_file_uploaded = True
                    
            except Exception as e:
                logging.error(f"Failed to upload/process strategy document: {e}")
                # We continue profile update even if file upload/extraction fails
        
        # Handle Standards File Upload - Store in GridFS AND extract text for RAG IN BACKGROUND
        standards_file_uploaded = False
        if standards_file:
            try:
                standards_filename = secure_filename(standards_file.filename)
                standards_file_data = standards_file.read()
                standards_metadata = {
                    'filename': standards_filename,
                    'content_type': standards_file.content_type,
                    'uploaded_by_username': user.username,
                    'uploaded_by_user_id': user_id,
                    'collection_type': 'standards_documents',
                    'uploaded_at': datetime.utcnow().isoformat()
                }
                # Upload to GridFS for raw file storage
                standards_doc_file_id = mongodb_file_manager.upload_to_mongodb(standards_file_data, standards_metadata)
                logging.info(f"Uploaded standards document for user {user.username}: {standards_doc_file_id}")
                
                # Extract text and store in MongoDB IN BACKGROUND for RAG
                logging.info(f"[UPDATE_PROFILE] Queueing background standards text extraction for user {user.id}: {standards_filename}")
                extract_and_store_standards_text_background(standards_file_data, standards_filename, user_id)
                standards_file_uploaded = True
                    
            except Exception as e:
                logging.error(f"Failed to upload standards document: {e}")
                # We continue profile update even if file upload fails
            
        db.session.commit()
        
        # Build response message
        message_parts = ["Profile updated successfully"]
        if strategy_file_uploaded:
            message_parts.append("Strategy file processing in background")
        if standards_file_uploaded:
            message_parts.append("Standards file processing in background")
        
        response_data = {
            "success": True, 
            "message": ". ".join(message_parts) + ".",
            "user": {
                "first_name": user.first_name,
                "last_name": user.last_name,
                "username": user.username,
                "company_name": user.company_name,
                "location": user.location
            }
        }
        
        # Indicate if strategy file is being processed in background
        if strategy_file_uploaded:
            response_data["strategy_extraction"] = {
                "status": "processing",
                "message": "Strategy file is being processed in background. Data will be available shortly."
            }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logging.exception("Profile update failed.")
        return jsonify({"error": str(e)}), 500



@app.route('/health', methods=['GET'])
@login_required

def health_check():
    return {
        "status": "healthy",
        "workflow_initialized": False,
        "langsmith_enabled": False
    }, 200

# =========================================================================
# === IMAGE SERVING ENDPOINT ===
# =========================================================================
@app.route('/api/images/<file_id>', methods=['GET'])
# @login_required
def serve_image(file_id):
    """
    Serve images from MongoDB GridFS
    
    Args:
        file_id: GridFS file ID or filename/hash
    """
    try:
        from bson import ObjectId
        from mongodb_utils import mongodb_file_manager
        
        gridfs = mongodb_file_manager.gridfs
        grid_out = None
        
        # Try 1: Treat as ObjectId
        try:
            if ObjectId.is_valid(file_id):
                grid_out = gridfs.get(ObjectId(file_id))
        except Exception:
            pass
            
        # Try 2: Treat as filename (if not found by ID)
        if grid_out is None:
            try:
                # Search by filename
                grid_out = gridfs.find_one({"filename": file_id})
                
                # If still not found, try searching by original_url or other metadata if it looks like a hash
                if grid_out is None and len(file_id) == 64:
                    # Try finding by filename with extension wildcard? No, exact match.
                    # Maybe the filename has an extension we don't know.
                    # Try finding any file starting with this hash
                    import re
                    grid_out = gridfs.find_one({"filename": {"$regex": f"^{file_id}"}})
            except Exception as e:
                logging.warning(f"Failed to find image by filename {file_id}: {e}")

        if grid_out is None:
            logging.error(f"Image not found in GridFS: {file_id}")
            return jsonify({"error": "Image not found"}), 404
        
        # Read image data
        image_data = grid_out.read()
        content_type = grid_out.content_type or 'image/jpeg'
        
        # Create response with proper headers
        response = send_file(
            BytesIO(image_data),
            mimetype=content_type,
            as_attachment=False,
            download_name=grid_out.filename or f"image.{content_type.split('/')[-1]}"
        )
        
        # Add caching headers (cache for 30 days)
        response.headers['Cache-Control'] = 'public, max-age=2592000'
        
        logging.info(f"Served image from GridFS: {file_id} ({len(image_data)} bytes)")
        return response
    except Exception as e:
        logging.exception(f"Error serving image {file_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Fallback route for images requested without /api/images/ prefix
@app.route('/<string:file_id>', methods=['GET'])
def serve_image_root(file_id):
    """
    Fallback for images requested at root (e.g. legacy or malformed URLs)
    Only handles long hashes to avoid conflict with other routes
    """
    # Only handle if it looks like a hash (64 chars) or ObjectId (24 chars)
    if len(file_id) in [24, 64] and all(c in '0123456789abcdefABCDEF' for c in file_id):
        return serve_image(file_id)
    
    # Otherwise let it 404 (or be handled by frontend router if applicable)
    return jsonify({"error": "Not found"}), 404

@app.route('/api/generic_image/<product_type>', methods=['GET'])
@login_required
def get_generic_image(product_type):
    """
    Fetch generic product type image with MongoDB caching
    
    Strategy:
    1. Check MongoDB cache first
    2. If not found, search external APIs with "generic <product_type>"
    3. Cache the result
    4. Return image URL or GridFS file ID
    
    Args:
        product_type: Product type name (e.g., "Pressure Transmitter")
    """
    try:
        from generic_image_utils import fetch_generic_product_image
        
        # Decode URL-encoded product type
        import urllib.parse
        decoded_product_type = urllib.parse.unquote(product_type)
        
        logging.info(f"[API] ===== Generic Image Request START =====")
        logging.info(f"[API] Product Type (raw): {product_type}")
        logging.info(f"[API] Product Type (decoded): {decoded_product_type}")
        
        # Fetch image (checks cache first, then external APIs)
        image_result = fetch_generic_product_image(decoded_product_type)
        
        if image_result:
            logging.info(f"[API] ✓ Image found! Source: {image_result.get('source')}, Cached: {image_result.get('cached')}")
            logging.info(f"[API] ===== Generic Image Request END (SUCCESS) =====")
            return jsonify({
                "success": True,
                "image": image_result,
                "product_type": decoded_product_type
            }), 200
        else:
            logging.warning(f"[API] ✗ No image found for: {decoded_product_type}")
            logging.info(f"[API] ===== Generic Image Request END (NOT FOUND) =====")
            return jsonify({
                "success": False,
                "error": "No image found",
                "product_type": decoded_product_type
            }), 404
            
    except Exception as e:
        logging.exception(f"[API] ✗ ERROR fetching generic image for {product_type}: {e}")
        logging.info(f"[API] ===== Generic Image Request END (ERROR) =====")
        return jsonify({
            "success": False,
            "error": str(e),
            "product_type": product_type
        }), 500


# =========================================================================
# === FILE UPLOAD AND TEXT EXTRACTION ENDPOINT ===
# =========================================================================
@app.route('/api/upload-requirements', methods=['POST'])
@login_required
def upload_requirements_file():
    """
    Upload file (PDF, DOCX, TXT, Images) and extract text as requirements
    
    Accepts: multipart/form-data with 'file' field
    Returns: Extracted text from the file
    """
    try:
        from file_extraction_utils import extract_text_from_file
        
        logging.info("[API] ===== File Upload Request START =====")
        
        # Check if file is present
        if 'file' not in request.files:
            logging.warning("[API] No file provided in request")
            return jsonify({
                "success": False,
                "error": "No file provided"
            }), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            logging.warning("[API] Empty filename")
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400
        
        # Read file bytes
        file_bytes = file.read()
        filename = file.filename
        
        logging.info(f"[API] Processing file: {filename} ({len(file_bytes)} bytes)")
        
        # Extract text from file
        extraction_result = extract_text_from_file(file_bytes, filename)
        
        if not extraction_result['success']:
            logging.warning(f"[API] Failed to extract text from {filename}")
            return jsonify({
                "success": False,
                "error": f"Could not extract text from {extraction_result['file_type']} file",
                "file_type": extraction_result['file_type']
            }), 400
        
        logging.info(f"[API] ✓ Successfully extracted {extraction_result['character_count']} characters from {filename}")
        logging.info("[API] ===== File Upload Request END (SUCCESS) =====")
        
        return jsonify({
            "success": True,
            "extracted_text": extraction_result['extracted_text'],
            "filename": filename,
            "file_type": extraction_result['file_type'],
            "character_count": extraction_result['character_count']
        }), 200
        
    except Exception as e:
        logging.exception(f"[API] ✗ ERROR processing file upload: {e}")
        logging.info("[API] ===== File Upload Request END (ERROR) =====")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================================
# === STRATEGY FILE UPLOAD AND LLM EXTRACTION ENDPOINT ===
# =========================================================================
@app.route('/api/upload-strategy-file', methods=['POST'])
@login_required
def upload_strategy_file():
    """
    Upload strategy file (PDF, DOCX, TXT, Images) and extract structured strategy data using Gemini 2.5 Flash.
    
    Accepts: multipart/form-data with 'file' field
    Returns: Extracted strategy data stored in MongoDB
    
    The LLM will extract the following fields from the document:
    - vendor_name: Name of the vendor/supplier
    - category: Product category (e.g., "Pressure Transmitter", "Flow Meter")
    - subcategory: Product subcategory or specific product type
    - stratergy: Procurement strategy (e.g., "Cost optimization", "Dual sourcing")
    """
    try:
        from file_extraction_utils import extract_text_from_file
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        logging.info("[STRATEGY_UPLOAD] ===== Strategy File Upload Request START =====")
        
        # Check if file is present
        if 'file' not in request.files:
            logging.warning("[STRATEGY_UPLOAD] No file provided in request")
            return jsonify({
                "success": False,
                "error": "No file provided"
            }), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            logging.warning("[STRATEGY_UPLOAD] Empty filename")
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400
        
        # Read file bytes
        file_bytes = file.read()
        filename = file.filename
        
        logging.info(f"[STRATEGY_UPLOAD] Processing file: {filename} ({len(file_bytes)} bytes)")
        
        # Get user ID for storage
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({
                "success": False,
                "error": "User not authenticated"
            }), 401
        
        # Extract text from file
        extraction_result = extract_text_from_file(file_bytes, filename)
        
        if not extraction_result['success']:
            logging.warning(f"[STRATEGY_UPLOAD] Failed to extract text from {filename}")
            return jsonify({
                "success": False,
                "error": f"Could not extract text from {extraction_result['file_type']} file. {extraction_result.get('error', '')}",
                "file_type": extraction_result['file_type']
            }), 400
        
        extracted_text = extraction_result['extracted_text']
        logging.info(f"[STRATEGY_UPLOAD] ✓ Extracted {extraction_result['character_count']} characters from {filename}")
        
        # Initialize Gemini 2.5 Flash for strategy extraction
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
        
        if not GOOGLE_API_KEY:
            logging.error("[STRATEGY_UPLOAD] GOOGLE_API_KEY not configured")
            return jsonify({
                "success": False,
                "error": "LLM API key not configured"
            }), 500
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY,
        )
        
        # Create the strategy extraction prompt
        strategy_extraction_prompt = """
You are an expert at extracting structured procurement strategy data from documents.

Analyze the following document content and extract all vendor/supplier strategy information.
For each vendor entry found, extract the following fields:
- vendor_name: Name of the vendor/supplier/manufacturer
- category: Product category (e.g., "Pressure Transmitter", "Flow Meter", "Control Valve", "Temperature Transmitter")
- subcategory: Specific product type or subcategory within the category
- stratergy: Procurement strategy (e.g., "Cost optimization", "Life-cycle cost evaluation", "Sustainability and green procurement", "Dual sourcing", "Framework agreement", "Single sourcing", "Quality-focused")

Document Content:
{document_text}

Extract ALL vendor entries from the document. If a field is not explicitly mentioned, use an empty string.

Return ONLY a valid JSON array of objects with this structure:
[
  {{
    "vendor_name": "<vendor name>",
    "category": "<product category>",
    "subcategory": "<product subcategory>",
    "stratergy": "<procurement strategy>"
  }},
  ...
]

Important:
1. Include ALL vendor/strategy entries found in the document
2. If the document contains a table or list of vendors, extract each row/item as a separate entry
3. If strategy information is not provided for a vendor, leave the "stratergy" field empty
4. Return ONLY the JSON array, no additional text or explanation
5. If no valid vendor data can be extracted, return an empty array: []
"""

        try:
            logging.info("[STRATEGY_UPLOAD] Sending to Gemini 2.5 Flash for extraction...")
            
            prompt = ChatPromptTemplate.from_template(strategy_extraction_prompt)
            chain = prompt | llm | StrOutputParser()
            
            llm_response = chain.invoke({"document_text": extracted_text[:50000]})  # Limit text length
            
            # Clean the LLM response
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # Parse the JSON response
            strategy_data = json.loads(cleaned_response)
            
            if not isinstance(strategy_data, list):
                logging.warning(f"[STRATEGY_UPLOAD] LLM returned non-array response: {type(strategy_data)}")
                strategy_data = []
            
            logging.info(f"[STRATEGY_UPLOAD] ✓ LLM extracted {len(strategy_data)} strategy entries")
            
        except json.JSONDecodeError as e:
            logging.error(f"[STRATEGY_UPLOAD] Failed to parse LLM response as JSON: {e}")
            logging.error(f"[STRATEGY_UPLOAD] LLM Response: {llm_response}")
            return jsonify({
                "success": False,
                "error": "Failed to parse strategy data from document. The document format may not be compatible."
            }), 400
        except Exception as e:
            logging.error(f"[STRATEGY_UPLOAD] LLM extraction failed: {e}")
            return jsonify({
                "success": False,
                "error": f"Failed to extract strategy data: {str(e)}"
            }), 500
        
        # Store in MongoDB
        if strategy_data and len(strategy_data) > 0:
            try:
                conn = get_mongodb_connection()
                stratergy_collection = conn['collections']['stratergy']
                
                # Create document with user_id and data
                strategy_document = {
                    "user_id": user_id,
                    "filename": filename,
                    "file_type": extraction_result['file_type'],
                    "data": strategy_data,
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "character_count": extraction_result['character_count'],
                    "entry_count": len(strategy_data)
                }
                
                # Insert into MongoDB
                result = stratergy_collection.insert_one(strategy_document)
                
                logging.info(f"[STRATEGY_UPLOAD] ✓ Stored {len(strategy_data)} strategy entries in MongoDB with ID: {result.inserted_id}")
                
                return jsonify({
                    "success": True,
                    "message": f"Successfully extracted and stored {len(strategy_data)} vendor strategy entries",
                    "document_id": str(result.inserted_id),
                    "filename": filename,
                    "file_type": extraction_result['file_type'],
                    "entry_count": len(strategy_data),
                    "extracted_data": strategy_data  # Return extracted data for frontend display
                }), 200
                
            except Exception as e:
                logging.error(f"[STRATEGY_UPLOAD] MongoDB storage failed: {e}")
                return jsonify({
                    "success": False,
                    "error": f"Failed to store strategy data: {str(e)}"
                }), 500
        else:
            logging.warning("[STRATEGY_UPLOAD] No strategy data extracted from document")
            return jsonify({
                "success": False,
                "error": "No vendor strategy data could be extracted from the document. Please ensure the document contains vendor/supplier information with categories and strategies.",
                "extracted_text_preview": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
            }), 400
        
    except Exception as e:
        logging.exception(f"[STRATEGY_UPLOAD] ✗ ERROR processing strategy file upload: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================================
# === CORE LLM-BASED SALES AGENT ENDPOINT ===
# =========================================================================
@app.route("/api/sales-agent", methods=["POST"])
@login_required
def api_sales_agent():
    """
    Handles step-based workflow responses with session tracking, knowledge questions,
    and workflow continuity. Maintains professional and friendly conversation flow.
    """
    if not components or not components.get('llm'):
        return jsonify({"error": "LLM component is not ready."}), 503

    try:
        # NOTE: request.get_json(force=True) is used for debugging/non-standard headers
        data = request.get_json(force=True)
        # Debug log incoming product_type information to trace saving issues (sales agent)
        incoming_pt = data.get('product_type') if isinstance(data, dict) else None
        incoming_detected = data.get('detected_product_type') if isinstance(data, dict) else None
        logging.info(f"[SALES_AGENT_INCOMING] Incoming product_type='{incoming_pt}' detected_product_type='{incoming_detected}' project_name='{data.get('project_name') if isinstance(data, dict) else None}'")
        

        
        # Continue with normal sales-agent workflow if no CSV vendors
        step = data.get("step")
        data_context = data.get("dataContext", {})
        user_message = data.get("userMessage", "")
        intent = data.get("intent", "")
        search_session_id = data.get("search_session_id", "default")

        # Log session-specific request for debugging
        logging.info(f"[SALES_AGENT] Session {search_session_id}: Step={step}, Intent={intent}")

        # Get session state for workflow continuity (session-isolated)
        current_step_key = f'current_step_{search_session_id}'
        current_intent_key = f'current_intent_{search_session_id}'
        current_step = session.get(current_step_key)
        current_intent = session.get(current_intent_key)
        
        # --- Helper function for formatting advanced parameters ---
        def format_available_parameters(params):
            """
            Return parameter keys one by one, each on a separate line.
            Each item in `params` can be:
            - A dict with 'key' field: {"key": "parameter_key_name", ...}
            - A string: "parameter_key_name"
            Replaces underscores with spaces and formats for display.
            """
            formatted = []
            for param in params:
                # Extract the key from dict or use string directly
                if isinstance(param, dict):
                    # Priority: prefer human-friendly 'name' field, then 'key',
                    # then fall back to the first dict key if present.
                    name = param.get('name') or param.get('key') or (list(param.keys())[0] if param else '')
                else:
                    # Parameter keys are typically strings like "parameter_key_name"
                    name = str(param).strip()

                # Replace underscores with spaces
                name = name.replace('_', ' ')
                # Remove any parenthetical or bracketed content by taking text
                # before the first bracket. Example: "X (Y)" -> "X"
                name = re.split(r'[\(\[\{]', name, 1)[0].strip()
                # Normalize spacing (remove extra spaces)
                name = " ".join(name.split())
                # Title case for display (first letter of each word capitalized)
                name = name.title()

                # Prefix with a bullet so the list appears as points one per line
                formatted.append(f"- {name}")
            return "\n".join(formatted)

        # Treat short affirmative/negative replies (e.g., 'yes', 'no') as
        # workflow input regardless of the classifier. The frontend's
        # intent classifier can sometimes label brief confirmations as
        # knowledge questions; we prefer to route them into the sales-agent
        # workflow so steps like awaitAdditionalAndLatestSpecs and
        # awaitAdvancedSpecs behave deterministically.
        try:
            short_yesno_re = re.compile(r"^\s*(?:yes|y|yeah|yep|sure|ok|okay|no|n|nope|skip)\b[\.\!\?\s]*$", re.IGNORECASE)
        except Exception:
            short_yesno_re = None

        if isinstance(user_message, str) and short_yesno_re and short_yesno_re.match(user_message):
            matched = short_yesno_re.match(user_message).group(0).strip()
            if intent == 'knowledgeQuestion':
                logging.info(f"[SALES_AGENT] Overriding intent 'knowledgeQuestion' for short reply: '{user_message}' (matched='{matched}')")
            intent = 'workflow'
            logging.info(f"[SALES_AGENT] Routed short reply to workflow branch: '{user_message}' (matched='{matched}', step={step})")

        # Handle knowledge questions - answer and resume workflow
        if intent == "knowledgeQuestion":
            # Determine context-aware response based on current workflow step
            if step == "greeting":
                context_hint = "What industrial product are you looking for today?"
            elif step == "initialInput":
                context_hint = "Please share your product requirements when you're ready."
            elif step == "awaitMissingInfo":
                context_hint = "Now, please provide the missing details so we can continue with your product selection, or shall we proceed with the given details for approximate results?"
            elif step == "awaitAdditionalAndLatestSpecs":
                context_hint = "Would you like to add any additional or latest specifications?"
            elif step == "awaitAdvancedSpecs":
                context_hint = "Would you like to select any of the advanced parameters?"
            elif step == "showSummary":
                context_hint = "Ready to proceed with your product analysis?"
            elif step == "finalAnalysis":
                context_hint = "Your analysis is complete. Is there anything else you'd like to know?"
            elif step == "analysisError":
                context_hint = "Would you like to retry the analysis?"
            else:
                context_hint = "How can I help you with your product selection?"
            
            # Let LLM decide if question is industrial-related and respond appropriately
            prompt_template = f"""
You are Engenie - an expert industrial sales consultant specializing in industrial automation, instrumentation, and process control.

The user asked: "{user_message}"

First, determine if this question is related to your expertise:
- Industrial products (transmitters, sensors, valves, pumps, controllers, actuators, gauges, meters)
- Process control and automation (PLC, DCS, SCADA, HMI, protocols like HART, Modbus, Profibus)
- Instrumentation (flow, pressure, temperature, level, humidity measurement)
- Industrial standards and certifications (ATEX, IECEx, SIL, NEMA, IP ratings)
- Vendors and manufacturers (Emerson, Honeywell, ABB, Siemens, Yokogawa, Endress+Hauser)
- Technical specifications (accuracy, range, output signals, materials, calibration)

**If the question IS related to industrial topics:**
- Provide a clear, professional answer (2-3 sentences max)
- After answering, smoothly transition back with: "{context_hint}"

**If the question is NOT related to industrial topics:**
- Politely say: "I appreciate your question, but I specialize in industrial products and processes."
- Offer to help with their industrial needs
- Transition back with: "{context_hint}"

Keep your response friendly, concise, and helpful.
"""
            
            # Build and execute LLM chain
            full_prompt = ChatPromptTemplate.from_template(prompt_template)
            response_chain = full_prompt | components['llm'] | StrOutputParser()
            llm_response = response_chain.invoke({"user_input": user_message, "prompt": prompt_template})
            
            # Return response without changing workflow step - use the step sent by frontend
            return jsonify({
                "content": llm_response,
                "nextStep": step,  # Resume the exact step where user was interrupted
                "maintainWorkflow": True
            }), 200

        # === Step-Based Workflow Responses (Preserving Original Prompts) ===
        
        # --- Original Prompt Selection Based on Step ---
        if step == 'initialInput':
            product_type = data_context.get('productType', 'a product')
            # If frontend indicated this request should save immediately (e.g., initial full-spec submit),
            # bypass the greeting prompt and persist the detected product type in session.
            save_flag = False
            if isinstance(data, dict):
                # Accept both `saveImmediately` boolean and `action: 'save'` patterns
                save_flag = bool(data.get('saveImmediately')) or data.get('action') == 'save'
            if save_flag:
                session[f'product_type_{search_session_id}'] = product_type
                session[f'current_step_{search_session_id}'] = 'initialInput'
                session.modified = True
                # Return quick confirmation to frontend so it can proceed without receiving the greeting prompt
                return jsonify({
                    "content": f"Saved product type: {product_type}",
                    "nextStep": "awaitAdditionalAndLatestSpecs"
                }), 200
            
            
            prompt_template = f"""
[Session: {search_session_id}] - You are Engenie - a helpful sales agent in a fresh conversation. The user shared their requirements, and you identified the product type as '{product_type}'.
Your response must:
1. Start positively (e.g., "Great choice!" or similar).
2. Confirm the identified product type in a friendly way.
3. Ask: "Additional and latest specifications are available. Would you like to add them?"

Important: This is an independent conversation session. Do not reference any previous interactions.
"""
            next_step = "awaitAdditionalAndLatestSpecs"
        
        elif step == 'initialInputWithSpecs':
            # All mandatory fields provided - ask if user wants to see additional specs
            # The specs list will be shown when user says "yes" at awaitAdditionalAndLatestSpecs
            product_type = data_context.get('productType', 'a product')
            
            prompt_template = f"""
You are Engenie - a helpful sales agent. The user provided all mandatory requirements for a {product_type}.

Your response must follow this EXACT format:

"Perfect! It seems like you are looking for a {product_type}. Additional and latest specifications are available. Would you like to add them?"

You may vary the opening word (Perfect/Great/Excellent) but keep the rest of the sentence structure exactly as shown.
"""
            next_step = "awaitAdditionalAndLatestSpecs"
        
        elif step == 'askForMissingFields':
            # User said "no" at awaitMissingInfo - they want to provide missing fields
            product_type = data_context.get('productType', 'your product')
            missing_fields = data_context.get('missingFields', '')
            
            prompt_template = f"""
You are Engenie - a helpful sales assistant. The user indicated they want to provide more details.

The following specifications are still missing: **{missing_fields}**

Your response must:
1. Acknowledge their choice to provide more details (e.g., "Sure!" or "Great!")
2. List the missing specifications clearly: **{missing_fields}**
3. Ask which one they would like to add, OR if they've changed their mind and want to proceed with approximate results

Keep it friendly and concise. Example:
"Sure! Here are the specifications we're still missing: **{missing_fields}**. Which one would you like to add? Or if you prefer, we can proceed with the given details for approximate results."
"""
            next_step = "awaitMissingInfo"  # Stay at this step until user says yes
        
        elif step == 'confirmAfterMissingInfo':
            # User confirmed to proceed (either skipped or provided all fields)
            product_type = data_context.get('productType', 'a product')
            prompt_template = f"""
You are Engenie - a helpful sales agent. The user is ready to proceed with their {product_type} search.

Your response must:
1. Acknowledge positively (e.g., "Perfect!" or "Great!")
2. Mention that additional and latest specifications are available
3. Ask: "Would you like to add any additional or latest specifications?"

Keep it short and friendly (2-3 sentences max).
"""
            next_step = "awaitAdditionalAndLatestSpecs"
            
        elif step == 'awaitAdditionalAndLatestSpecs':
            # Handle the combined "Additional and Latest Specs" step
            user_lower = user_message.lower().strip()
            
            # Get available parameters and track which are already added
            available_parameters = data_context.get('availableParameters', [])
            added_specs_key = f'added_specs_{search_session_id}'
            added_specs = session.get(added_specs_key, [])
            
            # Get product type for context
            product_type = data_context.get('productType') or session.get(f'product_type_{search_session_id}', 'your product')
            
            # Calculate remaining parameters (those not yet added)
            remaining_parameters = []
            for param in available_parameters:
                param_name = param.get('name') if isinstance(param, dict) else str(param)
                param_lower = param_name.lower().replace('_', ' ')
                if not any(added.lower() in param_lower or param_lower in added.lower() for added in added_specs):
                    remaining_parameters.append(param)
            
            # Define keywords
            affirmative_keywords = ['yes', 'y', 'yeah', 'yep', 'sure', 'ok', 'okay']
            negative_keywords = ['no', 'n', 'nope', 'skip', 'done', 'finish', 'proceed', 'continue', 'that\'s all', 'nothing more']
            
            # Track state in session
            asking_state_key = f'awaiting_additional_specs_yesno_{search_session_id}'
            is_awaiting_yesno = session.get(asking_state_key, True)
            
            # Check if user response matches any parameter
            def find_matching_parameter(user_input, params):
                """Check if user input matches any available parameter"""
                user_words = set(user_input.lower().replace('_', ' ').split())
                for param in params:
                    param_name = param.get('name') if isinstance(param, dict) else str(param)
                    param_lower = param_name.lower().replace('_', ' ')
                    param_words = set(param_lower.split())
                    # Check if there's significant overlap
                    if user_words & param_words or param_lower in user_input.lower() or user_input.lower() in param_lower:
                        return param_name
                return None
            
            # Check if user input is yes/no
            is_yes = any(keyword == user_lower or user_lower.startswith(keyword + ' ') for keyword in affirmative_keywords)
            is_no = any(keyword == user_lower or user_lower.startswith(keyword + ' ') for keyword in negative_keywords)
            
            # CASE 2: User says NO -> Go to showSummary
            if is_no:
                session[asking_state_key] = True  # Reset for next time
                session[added_specs_key] = []  # Clear added specs
                prompt_template = """
You are Engenie - a helpful sales agent.

The user said NO to adding additional specifications.

Respond with EXACTLY this single sentence and nothing else:

"It sounds like you're ready to move forward. Here's a quick summary of what you have provided:"
"""
                next_step = "showSummary"
            
            # CASE 3: User says YES -> Show list and ask which to add
            elif is_yes:
                session[asking_state_key] = False  # Now we're collecting input
                
                # If no parameters available, discover them now
                if not remaining_parameters and product_type:
                    try:
                        parameters_result = discover_advanced_parameters(product_type)
                        discovered_params = parameters_result.get('unique_parameters') or parameters_result.get('unique_specifications', [])
                        remaining_parameters = discovered_params[:15] if discovered_params else []
                        # Store in session for future use
                        session[f'available_parameters_{search_session_id}'] = remaining_parameters
                        session.modified = True
                    except Exception as e:
                        logging.warning(f"Could not discover parameters for {product_type}: {e}")
                        remaining_parameters = []
                
                if remaining_parameters:
                    params_display = format_available_parameters(remaining_parameters)
                    prompt_template = f"""
You are Engenie - a helpful sales assistant. The user said YES to adding specifications.

Here are the available specifications:

{params_display}

Your response must follow this format:
1. Start with "Great!" or "Perfect!"
2. Say "Here are the additional and latest specifications that are available:"
3. List the specifications above as bullet points
4. Ask: "Would you like to add any of these?"

Keep it friendly and concise.
"""
                else:
                    # No specs available - move to next step
                    session[asking_state_key] = True  # Reset
                    session[added_specs_key] = []  # Clear
                    prompt_template = f"""
You are Engenie - a helpful sales agent.
No additional specifications were found. Say: "Let me show you the advanced parameters available for {product_type}."
"""
                    next_step = "awaitAdvancedSpecs"
                next_step = "awaitAdditionalAndLatestSpecs" if remaining_parameters else "awaitAdvancedSpecs"
            
            # CASE 1: User adds a specification from the list
            elif not is_awaiting_yesno:
                matched_param = find_matching_parameter(user_message, remaining_parameters)
                
                if matched_param:
                    # User provided a valid spec - add to collection
                    added_specs.append(matched_param)
                    session[added_specs_key] = added_specs
                    
                    # Store for processing
                    existing_specs = session.get(f'additional_specs_input_{search_session_id}', '')
                    if existing_specs:
                        session[f'additional_specs_input_{search_session_id}'] = f"{existing_specs}, {user_message}"
                    else:
                        session[f'additional_specs_input_{search_session_id}'] = user_message
                    
                    # Recalculate remaining after adding
                    new_remaining = []
                    for param in available_parameters:
                        param_name = param.get('name') if isinstance(param, dict) else str(param)
                        param_lower = param_name.lower().replace('_', ' ')
                        if not any(added.lower() in param_lower or param_lower in added.lower() for added in added_specs):
                            new_remaining.append(param)
                    
                    if new_remaining:
                        # More specs remaining - show them and ask for more
                        params_display = format_available_parameters(new_remaining)
                        prompt_template = f"""
You are Engenie - a helpful sales assistant. The user added: "{matched_param}".

Acknowledge briefly (e.g., "Got it! Added {matched_param}.").

Here are the remaining specifications:

{params_display}

Ask: "Would you like to add any more?"

Keep it concise.
"""
                        next_step = "awaitAdditionalAndLatestSpecs"
                    else:
                        # All specs added - automatically move to next step
                        session[asking_state_key] = True  # Reset
                        session[added_specs_key] = []  # Clear
                        prompt_template = f"""
You are Engenie - a helpful sales agent.
Say: "Great! You've added all available specifications. Let me show you the advanced parameters available for {product_type}."
"""
                        next_step = "awaitAdvancedSpecs"
                else:
                    # CASE 5: Irrelevant input - not matching any parameter
                    if remaining_parameters:
                        params_display = format_available_parameters(remaining_parameters)
                        prompt_template = f"""
You are Engenie - a helpful sales assistant.

The user's input doesn't match any of the available specifications.

Respond politely:
1. Say: "I couldn't find that in the available specifications."
2. Show the list again:

{params_display}

3. Ask: "Please choose from the list above"

Keep it friendly and concise.
"""
                    else:
                        prompt_template = """
You are Engenie - a helpful sales assistant.
Say: "I'm not sure what you mean. Would you like to continue to the next step?"
"""
                    next_step = "awaitAdditionalAndLatestSpecs"
            
            # First time asking - show list and ask yes/no
            else:
                if remaining_parameters:
                    params_display = format_available_parameters(remaining_parameters)
                    prompt_template = f"""
You are Engenie - a helpful sales assistant.

Here are the available additional and latest specifications:

{params_display}

Ask: "Would you like to add any of these specifications?"

Keep it concise.
"""
                else:
                    # No parameters available - skip to next step
                    prompt_template = f"""
You are Engenie - a helpful sales agent.
Say: "No additional specifications needed. Let me show you the advanced parameters available for {product_type}."
"""
                    next_step = "awaitAdvancedSpecs"
                next_step = "awaitAdditionalAndLatestSpecs" if remaining_parameters else "awaitAdvancedSpecs"
        
        elif step == 'awaitAdvancedSpecs':
            # Handle advanced parameters step
            user_lower = user_message.lower().strip()

            # Get context data (session-isolated)
            product_type = data_context.get('productType') or session.get(f'product_type_{search_session_id}')
            # NOTE: available_parameters is expected to be a list of strings or dicts
            available_parameters = data_context.get('availableParameters', [])
            selected_parameters = data_context.get('selectedParameters', {})
            total_selected = data_context.get('totalSelected', 0)
            
            # Define trigger keywords
            display_keywords = ['show', 'display', 'list', 'see', 'view', 'what are', 'remind']
            affirmative_keywords = ['yes', 'y', 'yeah', 'yep', 'sure', 'proceed', 'continue', 'ok', 'okay', 'go ahead']
            negative_keywords = ['no', 'n', 'skip', 'none', 'not needed', 'done', 'not interested']

            # Debug logging
            logging.info(f"awaitAdvancedSpecs - product_type: {product_type}")
            logging.info(f"awaitAdvancedSpecs - available_parameters count: {len(available_parameters)}")
            logging.info(f"awaitAdvancedSpecs - user_message: {user_message}")
            
            # Check if this is first time (no parameters discovered yet)
            if not available_parameters or len(available_parameters) == 0:
                
                # --- Handling retry/skip when discovery yielded 0 results or after an error ---
                parameter_error = data_context.get('parameterError', False)
                no_params_found = data_context.get('no_params_found', False)
                
                if parameter_error and user_lower in affirmative_keywords:
                    # User confirms they want to skip after an error
                    prompt_template = "You are Engenie - a helpful sales agent. Acknowledge skipping and move to summary."
                    next_step = "showSummary"
                elif parameter_error and user_lower in negative_keywords:
                    # User wants to retry (or says 'no' to skipping) - fall through to discovery
                    prompt_template = "" # Clears the error state prompt
                    pass
                elif user_lower in affirmative_keywords or user_lower == "":
                    # User agreed to proceed with discovery (or sent empty message) - run discovery
                    prompt_template = "" # Clears any prior prompt
                    pass
                elif user_lower in negative_keywords:
                    # The LLM previously returned zero parameters and user answered 'no' to
                    # "Shall I proceed to the summary?" — user chose NOT to proceed, so retry discovery.
                    prompt_template = "" # Clears any prior prompt
                    pass
                else:
                    # Default action for unexpected input when no params are known
                    prompt_template = "I'm not sure how to proceed. Would you like me to try discovering the parameters, or shall we skip to the summary?"
                    next_step = "awaitAdvancedSpecs"
                    
                # --- Initial Discovery Block (Runs on first entry or retry) ---
                # Only run discovery if no specific prompt_template has been set above
                if not prompt_template.strip() and 'llm_response' not in locals():
                    logging.info(f"Attempting discovery for product_type: {product_type}")
                    try:
                        if product_type:
                            # Discover advanced parameters (works for both MongoDB cache and LLM discovery)
                            parameters_result = discover_advanced_parameters(product_type)
                            # Handle both 'unique_parameters' and 'unique_specifications' keys (MongoDB uses 'unique_specifications')
                            discovered_params = parameters_result.get('unique_parameters') or parameters_result.get('unique_specifications', [])
                            discovered_params = discovered_params[:15] if discovered_params else []
                            filtered_count = parameters_result.get('existing_parameters_filtered', 0) or parameters_result.get('existing_specifications_filtered', 0)

                            # Store discovered parameters in session for future use
                            data_context['availableParameters'] = discovered_params
                            # Track whether discovery returned zero parameters
                            data_context['no_params_found'] = len(discovered_params) == 0
                            session['data'] = data_context
                            session.modified = True

                            if len(discovered_params) == 0:
                                # CASE 2 — No advanced parameters found: ask if user wants to proceed to summary
                                filter_info = (
                                    f" All {filtered_count} potential advanced parameters were already covered in your mandatory/optional requirements."
                                    if filtered_count > 0
                                    else " No new advanced parameters were found for this product type."
                                )

                                # Direct deterministic response (no extra LLM prompting)
                                llm_response = (
                                    f"No advanced parameters were found.{filter_info}\n\n"
                                    "No advanced parameters were found. Do you want to proceed to summary?"
                                )
                                prompt_template = ""
                            else:
                                # CASE 1 — Advanced parameters found: show list and ask if user wants to add them
                                params_display = format_available_parameters(discovered_params)

                                # Direct deterministic response listing parameters
                                llm_response = (
                                    "These advanced parameters were identified:\n\n"
                                    f"{params_display}\n\n"
                                    "Do you want to add these advanced parameters?"
                                )
                                # Set prompt_template to empty so LLM is not called
                                prompt_template = ""
                        else:
                            # No product type found
                            data_context['parameterError'] = True
                            session['data'] = data_context
                            session.modified = True
                            prompt_template = "I'm having trouble accessing advanced parameters because the product type isn't clear. Would you like to skip this step?"

                    except Exception as e:
                        # General error case
                        logging.error(f"Error during parameter discovery: {e}", exc_info=True)
                        data_context['parameterError'] = True
                        session['data'] = data_context
                        session.modified = True
                        prompt_template = "I encountered an issue discovering advanced parameters. Would you like to skip this step?"
                
                # Now interpret user reply when no available_parameters exist (only if we didn't run discovery or discovery yielded 0)
                if data_context.get('no_params_found', False) or parameter_error:
                    # CASE 2 — Follow-up after "No advanced parameters were found. Do you want to proceed to summary?"
                    if user_lower in affirmative_keywords:
                        # User said YES -> go directly to summary
                        llm_response = "Okay, I'll proceed to the summary without advanced parameters."
                        prompt_template = ""
                        next_step = "showSummary"
                    elif user_lower in negative_keywords:
                        # User said NO -> retry discovery (loop will run discovery again)
                        data_context['parameterError'] = False
                        data_context['no_params_found'] = False  # Force a retry
                        session['data'] = data_context
                        session.modified = True
                        llm_response = "No problem, I'll try discovering advanced parameters again."
                        prompt_template = ""
                        next_step = "awaitAdvancedSpecs"
                    elif parameter_error and not user_message.strip():
                        # Empty message while in error state - stay and prompt again
                        prompt_template = "I encountered an issue discovering advanced parameters. Would you like to skip this step?"
                        next_step = "awaitAdvancedSpecs"
                    elif not user_message.strip():
                        # Empty message while in no_params_found state - repeat the question
                        llm_response = "No advanced parameters were found. Do you want to proceed to summary?"
                        prompt_template = ""
                        next_step = "awaitAdvancedSpecs"
                    else:
                        # If user gave something else, ask the clarifying question again
                        llm_response = "Please answer with yes or no. Do you want to proceed to summary without advanced parameters?"
                        prompt_template = ""
                        next_step = "awaitAdvancedSpecs"
                else:
                    # Default: stay in awaitAdvancedSpecs and attempt discovery based on affirmative/empty
                    next_step = "awaitAdvancedSpecs"
            else:
                # --- Parameters already discovered - handle user response ---
                parameter_error = data_context.get('parameterError', False)
                
                wants_display = any(keyword in user_lower for keyword in display_keywords)
                user_affirmed = any(keyword in user_lower for keyword in affirmative_keywords)
                user_denied = any(keyword in user_lower for keyword in negative_keywords)

                # CASE 1: User says 'yes' to adding parameters - show keys and ask them to provide values
                # Check this FIRST before CASE 2, so "yes" doesn't get caught by the display condition
                if user_affirmed:
                    # CASE 1 — User said YES: use LLM to ask for values, listing the parameters
                    params_display = format_available_parameters(available_parameters)
                    prompt_template = f"""
You are Engenie - a helpful sales agent.

The user said YES to adding advanced parameters. You have the following advanced parameters available:

{params_display}

Respond with a concise message that:
1. Starts with "Great!" or similar.
2. Asks the user to enter the values for the advanced parameters they'd like to add.
3. Reminds them of the available parameters.

Keep it short and friendly.
"""
                    next_step = "awaitAdvancedSpecs"  # Stay in step to collect values
                    
                # CASE 2: User says 'no' to adding parameters (normal flow)
                elif user_denied:
                    # User explicitly declined adding advanced parameters -> go directly to SUMMARY
                    # Use the same summary-intro sentence as in awaitAdditionalAndLatestSpecs
                    prompt_template = """
You are Engenie - a helpful sales agent.

The user said NO to adding advanced parameters.

Respond with EXACTLY this single sentence and nothing else:

"It sounds like you're ready to move forward. Here's a quick summary of what you have provided:"
"""
                    next_step = "showSummary"
                    
                # CASE 3: Force the list to display if the user explicitly asks to see it, or empty message
                elif wants_display or (not user_message.strip() and not total_selected > 0):
                    params_display = format_available_parameters(available_parameters)
                    
                    # Check if we should include intro message (coming from awaitAdditionalAndLatestSpecs transition)
                    show_intro = data_context.get('showIntro', False)
                    
                    if show_intro:
                        # Combined intro + parameters response (for smooth transition from yes/no)
                        llm_response = (
                            f"Perfect! Here are the advanced parameters available for {product_type}:\n\n"
                            f"{params_display}\n\n"
                            "Would you like to add any of these?"
                        )
                        prompt_template = ""  # Skip LLM call, use direct response
                    else:
                        # Use LLM to present the list and ask if they want to add them
                        prompt_template = f"""
You are Engenie - a helpful sales agent.

You need to show the user the advanced parameters that were identified and ask if they want to add them.

These additional advanced parameters were identified:

{params_display}

Respond with a short message that:
1. Shows these parameters in a readable way.
2. Asks: "Would you like to add them?" (or very close wording).

Keep it concise and friendly.
"""
                    next_step = "awaitAdvancedSpecs"
                    
                # CASE 4: User provided parameter selections/values
                elif total_selected > 0 or user_message.strip():
                    # User provided values for parameters
                    selected_names = [param.replace('_', ' ').title() for param in selected_parameters.keys()] if selected_parameters else []
                    if selected_names:
                        selected_display = ", ".join(selected_names)
                        prompt_template = f"""
You are Engenie - a helpful sales agent. The user provided advanced parameter values.
Respond with: "**Added Advanced Parameters:** {selected_display}

Proceeding to the summary now."
"""
                    else:
                        prompt_template = "Thank you for providing the advanced specifications. Proceeding to the summary now."
                    next_step = "showSummary"
                    
                # CASE 5: No parameters matched or user provided other input (Default fallback)
                else:
                    prompt_template = "Please respond with yes or no. These additional advanced parameters were identified. Would you like to add them?"
                    next_step = "awaitAdvancedSpecs"
            
        elif step == 'confirmAfterMissingInfo':
            # Discover advanced parameters to show in the response
            product_type = data_context.get('productType') or session.get(f'product_type_{search_session_id}')
            
            # Initial prompt is set as a fallback
            prompt_template = """
You are Engenie - a helpful sales assistant. The user just provided the last piece of required information.

In a single, encouraging sentence, ask if they have any other additional specs or latest advanced specs preferences to add before you finalize things.

Make it friendly and encouraging. Example: "Perfect! Before we finalize, would you like to add any additional or latest advanced specifications to enhance your selection?"
"""
            
            if product_type:
                try:
                    # Discover advanced parameters
                    parameters_result = discover_advanced_parameters(product_type)
                    discovered_params = parameters_result.get('unique_parameters', []) or parameters_result.get('unique_specifications', [])
                    discovered_params = discovered_params[:15] if discovered_params else []
                    
                    if len(discovered_params) > 0:
                        # Format parameters for display
                        params_display = format_available_parameters(discovered_params)
                        
                        # Store discovered parameters in session for later use
                        data_context['availableParameters'] = discovered_params
                        session['data'] = data_context
                        session.modified = True

                        # Use LLM with a strict prompt so it returns the exact desired message
                        prompt_template = f"""
You are Engenie - a helpful sales assistant. The user has provided all required information and you have discovered additional latest advanced specifications.

Respond with EXACTLY the following message and nothing else:

Fantastic! Before we finalize, do you have any other additional specs or latest advanced specifications preferences:

 these are the latest advanced specifications:
{params_display}

Would you like to add them?
"""
                    # Else: prompt_template remains the one set before the try block
                except Exception as e:
                    logging.error(f"Error discovering parameters in confirmAfterMissingInfo: {e}", exc_info=True)
                    # Fallback prompt is already set
            
            next_step = "awaitAdditionalAndLatestSpecs"
            
        elif step == 'showSummary':
            # Check if user is confirming to proceed with analysis
            user_lower = user_message.lower().strip()
            if user_lower in ['yes', 'y', 'proceed', 'continue', 'run', 'analyze', 'ok', 'okay']:
                prompt_template = """
You are Engenie - a helpful sales agent. The user confirmed they want to proceed with the analysis.
Respond with: "Excellent! Starting the product analysis now..."
"""
                next_step = "finalAnalysis"
            else:
                # First time showing summary - trigger summary generation
                # The frontend will call handleShowSummaryAndProceed which generates the summary
                prompt_template = """
You are Engenie - a helpful and friendly sales agent.
Your response must ONLY be:
"It sounds like you're ready to move forward. Here's a quick summary of what you have provided:"
Do NOT add any summary details, bullet points, or extra text. Just return this exact friendly sentence.
"""
                next_step = "showSummary"  # Stay in showSummary to trigger summary display
            
        elif step == 'finalAnalysis':
            ranked_products = data_context.get('analysisResult', {}).get('overallRanking', {}).get('rankedProducts', [])
            # NOTE: Logic to determine 'matching_products' based on 'requirementsMatch' is in the original code.
            # Assuming 'requirementsMatch' is a boolean key in each product dict.
            matching_products = [p for p in ranked_products if p.get('requirementsMatch') is True] 
            count = len(matching_products)
            prompt_template = f"""
You are Engenie – a helpful sales agent. I have completed the analysis and found {count} product(s). 
Inform the user about the number of products and let them know they can check the right panel for details. 
Keep the tone clear and professional.

IMPORTANT: Respond ONLY with the message to the user. Do NOT include any meta-text like "Here is the message" or "Of course".
"""
            next_step = None  # End of workflow
            
        elif step == 'analysisError':
            prompt_template = "You are Engenie - a helpful sales agent. An error happened during the analysis. Apologize and politely ask the user to try again when ready."
            next_step = "showSummary"  # Allow retry from summary
            
        elif step == 'default':
            prompt_template = "You are Engenie - a helpful sales agent. Reply to the user's message in a simple, friendly way and keep the conversation moving forward."
            next_step = current_step or None
            
        # === NEW WORKFLOW STEPS (Added for enhanced functionality) ===
        elif step == 'greeting':
            prompt_template = f"""
[Session: {search_session_id}] - You are Engenie - a friendly and professional industrial sales consultant starting a fresh conversation.
Greet the user warmly and ask them what type of industrial product they are looking for.
Mention that you can help them find the right solution from various manufacturers.
Keep it concise and welcoming.

Important: This is the start of a new, independent conversation session.
"""
            next_step = "initialInput"
            
        else:
            # Default fallback for unrecognized steps
            prompt_template = f"""
You are Engenie - an helpful sales agent. Reply to the user's message: "{user_message}" in a simple, friendly way and keep the conversation moving forward.
"""
            next_step = current_step or "greeting"

        # --- Build Chain and Generate Response ---
        # Initialize llm_response if not already set (for direct responses without LLM)
        if 'llm_response' not in locals():
            llm_response = ""
            
        if prompt_template.strip():
            full_prompt = ChatPromptTemplate.from_template(prompt_template)
            response_chain = full_prompt | components['llm'] | StrOutputParser()
            llm_response = response_chain.invoke({"user_input": user_message, "product_type": data_context.get('productType'), "count": count if 'count' in locals() else 0, "params_display": params_display if 'params_display' in locals() else 'No parameters.', "prompt": prompt_template})

        # Update session with new step (session-isolated)
        if next_step:
            session[f'current_step_{search_session_id}'] = next_step
            session[f'current_intent_{search_session_id}'] = 'workflow'

        # Prepare response
        response_data = {
            "content": llm_response,
            "nextStep": next_step
        }

        # Store the sales agent response as system response for logging

        return jsonify(response_data), 200

    except Exception as e:
        logging.exception("Sales agent response generation failed.")
        # Retrieve current step for fallback, defaulting to 'initialInput' if not found
        fallback_step = session.get(f'current_step_{search_session_id}', 'initialInput')
        return jsonify({
            "error": "Failed to generate response: " + str(e),
            "content": "I apologize, but I'm having technical difficulties. Please try again.",
            "nextStep": fallback_step
        }), 500


# =========================================================================
# === NEW FEEDBACK ENDPOINT ===
# =========================================================================
@app.route("/api/feedback", methods=["POST"])
@login_required
def handle_feedback():
    """
    Handles user feedback and saves a complete log entry to the database.
    """
    if not components or not components.get('llm'):
        return jsonify({"error": "LLM component is not ready."}), 503

    try:
        data = request.get_json(force=True)
        feedback_type = data.get("feedbackType")
        comment = data.get("comment", "")

        # --- DATABASE LOGGING LOGIC STARTS HERE ---
        
        # 1. Retrieve the stored data from the session
        user_query = session.get('log_user_query', 'No query found - user may have provided feedback without validation')
        system_response = session.get('log_system_response', {})

        # 2. Format the feedback for the database
        feedback_log_entry = feedback_type
        if comment:
            feedback_log_entry += f" ({comment})"

        # 3. Get the current user's information to log their username
        current_user = db.session.get(User, session['user_id'])
        if not current_user:
            logging.error(f"Could not find user with ID {session['user_id']} to create log entry.")
            return jsonify({"error": "Authenticated user not found for logging."}), 404
        
        username = current_user.username

        # 4. Persist feedback to MongoDB only (do not store in SQL)
        try:
            project_id_for_feedback = session.get('current_project_id') or data.get('projectId')

            feedback_entry = {
                'timestamp': datetime.utcnow(),
                'user_id': str(session.get('user_id')) if session.get('user_id') else None,
                'user_name': username,
                'feedback_type': feedback_type,
                'comment': comment,
                'user_query': user_query,
                'system_response': system_response
            }

            from mongo_project_manager import mongo_project_manager
            # If we have a project id, append to that project's feedback_entries array
            if project_id_for_feedback:
                try:
                    mongo_project_manager.append_feedback_to_project(project_id_for_feedback, str(session.get('user_id')), feedback_entry)
                    logging.info(f"Appended feedback to MongoDB project {project_id_for_feedback}")
                except Exception as me:
                    logging.warning(f"Failed to append feedback to MongoDB project {project_id_for_feedback}: {me}")
                    # On failure to append to project, fall back to inserting into a top-level feedback collection
                    try:
                        mongo_project_manager.file_manager.db['feedback'].insert_one({**feedback_entry, 'project_id': project_id_for_feedback})
                    except Exception as e:
                        logging.error(f"Failed to save feedback to feedback collection: {e}")
            else:
                # No project id: save feedback as standalone document for later linking
                try:
                    mongo_project_manager.file_manager.db['feedback'].insert_one({**feedback_entry, 'project_id': None})
                    logging.info("Saved feedback to MongoDB 'feedback' collection (no project id)")
                except Exception as e:
                    logging.error(f"Failed to save feedback to MongoDB feedback collection: {e}")

        except Exception as e:
            logging.exception(f"Failed to persist feedback to MongoDB: {e}")

        # Clean up the session logging keys
        session.pop('log_user_query', None)
        session.pop('log_system_response', None)
        
        # --- LOGGING LOGIC ENDS ---

        if not feedback_type and not comment:
            return jsonify({"error": "No feedback provided."}), 400
        
        # --- LLM RESPONSE GENERATION ---
        if feedback_type == 'positive':
            prompt_template = """
You are Engenie - an helpful assistant. The user provided positive feedback on the recent analysis.
If they left a comment, it is: '{comment}'.
Please respond warmly and thank them for their time and input. Keep the response to a single, friendly sentence.
"""
        elif feedback_type == 'negative':
            prompt_template = """
You are Engenie - an helpful assistant. The user provided negative feedback on the recent analysis.
If they left a comment, it is: '{comment}'.
Respond with empathy. Acknowledge their feedback, apologize for the inconvenience, and state that their input is valuable and will be used to make improvements. Keep it to one or two professional sentences.
"""
        else:  # This handles the case where only a comment is provided
            prompt_template = """
You are Engenie - an helpful assistant. The user provided the following feedback on the analysis: '{comment}'.
Acknowledge their comment and thank them for taking the time to provide their input.
"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        feedback_chain = prompt | components['llm'] | StrOutputParser()
        llm_response = feedback_chain.invoke({"comment": comment})

        return jsonify({"response": llm_response}), 200

    except Exception as e:
        logging.exception("Feedback handling or MongoDB storage failed.")
        return jsonify({"error": "Failed to process feedback: " + str(e)}), 500

# =========================================================================
# === UNIFIED ROUTE CLASSIFIER API ===
# =========================================================================
@app.route("/api/route-classifier", methods=["POST"])
@login_required
def route_classifier():
    """
    Unified Route Classifier API - Analyzes user input and determines the best routing page.
    This API should be called on EVERY user message from ANY page.
    
    Handles TWO types of responses:
    1. NON-ROUTING: Greetings, chitchat, questions, etc. - returns direct_response, stays on current page
    2. ROUTING: Instrument requirements - may require navigation to different page
    
    INPUT CATEGORIES:
    -----------------
    NON-ROUTING (stay on current page, return direct_response):
    - empty_gibberish: Invalid/meaningless input
    - greeting: "Hi", "Hello", etc.
    - farewell: "Bye", "Goodbye", etc.
    - gratitude: "Thanks", "Thank you", etc.
    - confirmation: "Yes", "No", "Okay" (let page handle)
    - question_help: "What can you do?", "Help"
    - chitchat: "How are you?", casual conversation
    - out_of_scope: Weather, jokes, unrelated topics
    - complaint: "That's wrong", negative feedback
    - question_general: Industrial knowledge questions
    
    ROUTING (may navigate to different page):
    - route_solution: Multi-instrument/project requirements
    - route_product_info: Database/vendor queries
    - route_search: Single item matching
    
    REQUEST:
    --------
    POST /api/route-classifier
    {
        "user_input": "<user's message>",
        "current_page": "solution" | "product_info" | "search"
    }
    
    RESPONSE:
    ---------
    {
        "category": "<input category>",
        "target_page": "solution" | "product_info" | "search",
        "current_page": "<current page>",
        "requires_confirmation": true/false,
        "requires_routing": true/false,
        "direct_response": "<message for non-routing categories>",
        "confirmation_message": "<routing confirmation prompt>",
        "opening_message": "<message when user confirms>",
        "decline_message": "<message when user declines>",
        "popup_blocked_message": "<message when popup blocked>",
        "original_query": "<user's input>",
        "reasoning": "<classification reasoning>"
    }
    """
    if not components or not components.get('llm_flash'):
        return jsonify({"error": "LLM component is not ready."}), 503

    try:
        data = request.get_json(force=True)
        user_input = (data.get("user_input") or "").strip()
        current_page = (data.get("current_page") or "solution").lower().strip()
        
        if not user_input:
            return jsonify({"error": "user_input is required"}), 400
        
        if current_page not in ["solution", "product_info", "search"]:
            return jsonify({"error": "current_page must be 'solution', 'product_info', or 'search'"}), 400
        
        logging.info(f"[ROUTE_CLASSIFIER] Input: '{user_input[:100]}...', Current Page: {current_page}")
        
        # Build the classification prompt (using LangChain template syntax)
        classification_prompt = """
You are an intelligent route classifier for an industrial automation assistant called "Engenie".

The user is currently on the **{current_page}** page and has provided input.

Your job is to:
1. Classify the INPUT CATEGORY
2. Determine if routing to a different page is needed
3. For non-routing categories, stay on current page and provide appropriate response

=============================================================================
INPUT CATEGORIES (in priority order - check from top to bottom):
=============================================================================

**CATEGORY 1: EMPTY_GIBBERISH** - Invalid or meaningless input
- Empty strings, single characters, random letters, symbols only
- Examples: "", "asdfgh", "???", "...", "test", "abc123", "!@#"
- Action: Stay on current page, ask for valid input

**CATEGORY 2: GREETING** - Simple greetings
- "Hi", "Hello", "Hey", "Good morning", "Hi there", "Hello Engenie"
- Action: Stay on current page, respond with friendly greeting

**CATEGORY 3: FAREWELL** - Goodbyes
- "Bye", "Goodbye", "Thanks, bye", "See you", "That's all"
- Action: Stay on current page, respond with farewell

**CATEGORY 4: GRATITUDE** - Thank you messages
- "Thanks", "Thank you", "Thanks a lot", "Appreciate it", "That was helpful"
- Action: Stay on current page, acknowledge and offer more help

**CATEGORY 5: CONFIRMATION** - Yes/No/OK responses (to previous prompts)
- "Yes", "No", "Okay", "Sure", "Continue", "Go ahead", "Skip", "Cancel"
- Action: Stay on current page, let page handle the response

**CATEGORY 6: QUESTION_HELP** - Questions about how to use the app
- "What can you do?", "How do I use this?", "Help", "Show features"
- Action: Stay on current page, explain app capabilities

**CATEGORY 7: CHITCHAT** - Casual conversation (still acceptable)
- "How are you?", "Are you a robot?", "Who made you?", "What's your name?"
- Action: Stay on current page, friendly response + redirect

**CATEGORY 8: OUT_OF_SCOPE** - Completely irrelevant to industrial automation
- Weather, jokes, politics, entertainment, recipes, general knowledge unrelated to industry
- Examples: "What's the weather?", "Tell me a joke", "Who is the president?", "Write a poem"
- Action: Stay on current page, politely decline + redirect to industrial automation

**CATEGORY 9: COMPLAINT** - Negative feedback about responses
- "That's wrong", "This isn't helpful", "You're not helping", "Try again"
- Action: Stay on current page, apologize and ask for clarification

**CATEGORY 10: QUESTION_GENERAL** - General/conceptual questions about industrial concepts
- Educational questions that can be answered from GENERAL KNOWLEDGE (no database needed)
- Questions about CONCEPTS, DEFINITIONS, HOW THINGS WORK, PRINCIPLES
- NO vendor names, NO "what vendors", NO "which products", NO "tell me about [specific product]"
- Examples:
  - "What is a pressure transmitter?" (concept definition)
  - "How does HART protocol work?" (how it works)
  - "Explain 4-20mA signaling" (explanation)
  - "What's the difference between DP and GP?" (concept comparison)
  - "Why use stainless steel in instruments?" (principle)
  - "What is SIL rating?" (definition)
- Action: Stay on current page, answer using LLM knowledge

**DISTINGUISH question_general vs route_product_info:**
- "What is a pressure transmitter?" → QUESTION_GENERAL (concept)
- "What vendors have pressure transmitters?" → ROUTE_PRODUCT_INFO (database query)
- "How does a flow meter work?" → QUESTION_GENERAL (how it works)
- "Which flow meters does Emerson offer?" → ROUTE_PRODUCT_INFO (vendor-specific)
- "Explain HART protocol" → QUESTION_GENERAL (education)
- "What products support HART?" → ROUTE_PRODUCT_INFO (database lookup)

**CATEGORY 11: ROUTE_SOLUTION** - Multi-instrument/project requirements
- MULTIPLE instruments (2+), complex projects, process requirements
- Examples: "I need 5 control valves", "Refinery project with instruments"
- Action: Route to SOLUTION page if not already there

**CATEGORY 12: ROUTE_PRODUCT_INFO** - Database/vendor/product queries
- Questions that require DATABASE LOOKUP (not general knowledge)
- TRIGGER WORDS: "vendors", "which vendors", "what vendors", "tell me about [vendor]", 
  "products from", "[vendor name] products", "strategy", "procurement", "available", "offer"
- Questions about SPECIFIC VENDORS or their product offerings
- Questions about what's IN YOUR DATABASE
- Examples:
  - "What vendors have flow meters?" (database query)
  - "Tell me about Rosemount products" (vendor-specific)
  - "What's the strategy for ABB?" (strategy lookup)
  - "Which companies make pressure transmitters?" (vendor list)
  - "Does Honeywell have temperature sensors?" (availability)
  - "Compare Emerson vs Yokogawa offerings" (vendor comparison from DB)
  - "List all vendors in database" (database query)
- Action: Route to PRODUCT_INFO page if not already there

**CATEGORY 13: ROUTE_SEARCH** - Single item matching
- SINGLE instrument with "no accessories", standalone accessory
- Examples: "I need a pressure transmitter without accessories", "Find a junction box"
- Action: Route to SEARCH page if not already there

=============================================================================
ROUTING RULES:
=============================================================================
1. MULTIPLE instruments (2+) → SOLUTION (even with "no accessories")
2. SINGLE instrument WITH "no accessories" → SEARCH
3. SINGLE instrument WITHOUT "no accessories" → SOLUTION (accessories auto-inferred)
4. Querying DATABASE for vendor/product INFO → PRODUCT_INFO
5. Standalone accessory (no instrument) → SEARCH

**CRITICAL - PREFER CURRENT PAGE:**
- If input relates to current page's purpose, STAY on current page
- Only route when input is COMPLETELY INCOMPATIBLE with current page

=============================================================================
RESPONSE FORMAT:
=============================================================================
Return a JSON object with these fields:

{{
  "category": "<category name from above>",
  "target_page": "<solution|product_info|search>",
  "requires_routing": <true if routing to different page, false otherwise>,
  "direct_response": "<response message for non-routing categories, empty string if routing>",
  "reasoning": "<brief explanation>"
}}

**RESPONSE GUIDELINES FOR EACH CATEGORY:**
(Generate a response based on the user's actual input, not a fixed template)

- EMPTY_GIBBERISH: Politely ask the user to provide a valid input. Mention examples of what they can ask.

- GREETING: Respond warmly to their greeting. Introduce yourself as Engenie, an industrial automation assistant. Briefly mention you can help with instruments, vendors, and product questions. Ask how you can help.

- FAREWELL: Respond warmly to their goodbye. Thank them for using Engenie. Invite them to return anytime.

- GRATITUDE: Acknowledge their thanks warmly. Ask if there's anything else you can help with regarding instruments, vendors, or their project.

- CONFIRMATION: Return an empty string - let the page workflow handle yes/no responses.

- QUESTION_HELP: Explain what Engenie can do - identify instruments (Solution page), find matching products (Search page), query vendor database (Product Info page). Keep it helpful and conversational.

- CHITCHAT: Respond in a friendly, conversational way to their casual question. Always redirect back to how you can help with industrial automation needs.

- OUT_OF_SCOPE: Politely explain that the topic is outside your expertise. You specialize in industrial automation (transmitters, valves, sensors, vendor strategies). Redirect them to ask about industrial equipment.

- COMPLAINT: Apologize sincerely. Ask them to rephrase or provide more details so you can give accurate information.

- QUESTION_GENERAL: GENERATE A COMPLETE, HELPFUL ANSWER to their industrial/technical question.
  - Be informative and educational (2-4 sentences)
  - Provide accurate technical information
  - Use a friendly, professional tone
  - Offer to help further if needed

User's Current Page: {current_page}
User Input: "{user_input}"

Respond with ONLY the JSON object, no additional text.
"""
        
        classification_chain = ChatPromptTemplate.from_template(classification_prompt) | components['llm_flash'] | StrOutputParser()
        classification_response = classification_chain.invoke({
            "current_page": current_page,
            "user_input": user_input
        })
        
        # Clean and parse the response
        cleaned_response = classification_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        try:
            result = json.loads(cleaned_response)
            category = (result.get("category") or "").lower().strip()
            target_page = (result.get("target_page") or current_page).lower().strip()
            requires_routing = result.get("requires_routing", False)
            direct_response = result.get("direct_response", "")
            reasoning = result.get("reasoning", "")
            
            # Validate target_page
            if target_page not in ["solution", "product_info", "search"]:
                target_page = current_page
            
            # List of non-routing categories (user stays on current page)
            non_routing_categories = [
                "empty_gibberish", "greeting", "farewell", "gratitude", 
                "confirmation", "question_help", "chitchat", "out_of_scope", 
                "complaint", "question_general"
            ]
            
            # If category is non-routing, always stay on current page
            if category in non_routing_categories:
                requires_routing = False
                target_page = current_page
                
        except Exception as e:
            logging.warning(f"[ROUTE_CLASSIFIER] Failed to parse response, staying on current page: {e}")
            category = "unknown"
            target_page = current_page
            requires_routing = False
            direct_response = ""
            reasoning = "Classification parsing failed, staying on current page"
        
        logging.info(f"[ROUTE_CLASSIFIER] Category: {category}, Target: {target_page}, Current: {current_page}, Requires Routing: {requires_routing}")
        
        # =========================================================================
        # NON-ROUTING CATEGORIES: Return direct response without navigation
        # =========================================================================
        non_routing_categories = [
            "empty_gibberish", "greeting", "farewell", "gratitude", 
            "confirmation", "question_help", "chitchat", "out_of_scope", 
            "complaint", "question_general"
        ]
        
        if category in non_routing_categories:
            logging.info(f"[ROUTE_CLASSIFIER] Non-routing category '{category}', returning direct response")
            return jsonify({
                "category": category,
                "target_page": current_page,  # Stay on current page
                "current_page": current_page,
                "requires_confirmation": False,
                "requires_routing": False,
                "direct_response": direct_response,
                "confirmation_message": "",
                "opening_message": "",
                "decline_message": "",
                "popup_blocked_message": "",
                "original_query": user_input,
                "reasoning": reasoning
            }), 200
        
        # =========================================================================
        # ROUTING CATEGORIES: Determine if confirmation is needed
        # =========================================================================
        # Only need confirmation if routing to a DIFFERENT page
        requires_confirmation = target_page != current_page
        confirmation_message = ""
        opening_message = ""
        decline_message = ""
        popup_blocked_message = ""
        
        # Page descriptions for LLM context
        page_descriptions = {
            "solution": "Solution page to identify instruments and accessories",
            "product_info": "Product Info page to query our vendor database",
            "search": "Search page to find matching products"
        }
        
        page_names = {
            "solution": "Solution",
            "product_info": "Product Info",
            "search": "Search"
        }
        
        if requires_confirmation:
            target_page_name = page_names.get(target_page, target_page)
            current_page_name = page_names.get(current_page, current_page)
            
            try:
                # Generate all required messages in one LLM call for efficiency
                messages_prompt = f"""You are Engenie - a friendly industrial automation assistant.

The user is on the {current_page_name} page but their input seems better suited for the {target_page_name} page.

User Input: "{user_input}"
Target Page: {target_page_name}
Target Page Purpose: {page_descriptions.get(target_page, target_page)}
Current Page: {current_page_name}
Current Page Purpose: {page_descriptions.get(current_page, current_page)}

Generate 4 different messages as a JSON object:

1. "confirmation_message": A friendly message (2-3 sentences) that:
   - Acknowledges what they're asking about
   - Explains which page would be better
   - Asks if they'd like you to open that page

2. "opening_message": A brief message (1 sentence) to show when the user confirms and you're opening the new page.
   Example: "Opening the {target_page_name} page to help you with that..."

3. "decline_message": A friendly message (2-3 sentences) when the user says no and wants to stay on the current page.
   This message MUST:
   - Acknowledge their choice to stay on the current page
   - ASK them to provide relevant input for the CURRENT page's functionality
   - Give a brief example of what they can do on the current page
   
   Examples based on current page:
   - If current page is "Solution": "No problem! Please provide requirements, and I'll help you identify what you need."
   - If current page is "Product Info": "No problem. Is there anything else you would like to know about ?"
   - If current page is "Search": "Understood. Let's continue finding the best matching products here. What else are you looking for?"

4. "popup_blocked_message": A helpful message when the popup was blocked, telling them to click the link.
   Should be brief and include "[link]" as a placeholder for the actual link.
   Example: "The popup was blocked by your browser. Please [link] to open the {target_page_name} page."

Respond ONLY with a valid JSON object, no additional text:
"""
                messages_prompt += '{\n  "confirmation_message": "...",\n  "opening_message": "...",\n  "decline_message": "...",\n  "popup_blocked_message": "..."\n}'
                
                # Use LLM directly with HumanMessage to avoid template parsing issues
                from langchain_core.messages import HumanMessage
                messages_response = components['llm_flash'].invoke([HumanMessage(content=messages_prompt)])
                messages_json = messages_response.content.strip()
                
                # Clean and parse the JSON
                if messages_json.startswith("```json"):
                    messages_json = messages_json[7:]
                elif messages_json.startswith("```"):
                    messages_json = messages_json[3:]
                if messages_json.endswith("```"):
                    messages_json = messages_json[:-3]
                messages_json = messages_json.strip()
                
                messages_data = json.loads(messages_json)
                confirmation_message = messages_data.get("confirmation_message", "")
                opening_message = messages_data.get("opening_message", "")
                decline_message = messages_data.get("decline_message", "")
                popup_blocked_message = messages_data.get("popup_blocked_message", "")
                
            except Exception as e:
                logging.warning(f"[ROUTE_CLASSIFIER] Failed to generate messages, using fallbacks: {e}")
                # Fallback messages (still LLM-style but generated here as backup)
                try:
                    from langchain_core.messages import HumanMessage
                    
                    # Try individual prompts if batch fails
                    confirm_prompt = f"""You are Engenie - a friendly industrial automation assistant.
The user is on the {current_page_name} page but asked about something better suited for {target_page_name}.
User Input: "{user_input}"
Generate a brief, friendly confirmation message asking if they'd like to open the {target_page_name} page.
Respond ONLY with the message text."""
                    confirmation_message = components['llm'].invoke([HumanMessage(content=confirm_prompt)]).content.strip()
                    
                    opening_prompt = f"You are Engenie. Generate a brief message saying you're opening the {target_page_name} page. One sentence only."
                    opening_message = components['llm'].invoke([HumanMessage(content=opening_prompt)]).content.strip()
                    
                    decline_prompt = f"""You are Engenie. The user declined to go to {target_page_name} and wants to stay on {current_page_name}. 
The {current_page_name} page is for: {page_descriptions.get(current_page, current_page)}.
Generate a friendly 2-3 sentence response that:
1. Acknowledges their choice to stay
2. Asks them to provide input relevant to the {current_page_name} page's functionality
3. Gives a brief example of what they can do here
Respond ONLY with the message text."""
                    decline_message = components['llm'].invoke([HumanMessage(content=decline_prompt)]).content.strip()
                    
                    popup_prompt = f"You are Engenie. The popup to {target_page_name} was blocked. Generate a brief message telling the user to click the link. Include '[link]' as placeholder."
                    popup_blocked_message = components['llm'].invoke([HumanMessage(content=popup_prompt)]).content.strip()
                    
                except Exception as e2:
                    logging.error(f"[ROUTE_CLASSIFIER] Individual prompts also failed: {e2}")
                    # Final fallback with page-specific messages
                    confirmation_message = f"This looks like it would be better handled on the {page_descriptions.get(target_page, target_page)}. Would you like me to open it?"
                    opening_message = f"Opening the {target_page_name} page..."
                    
                    # Page-specific decline messages
                    decline_messages = {
                        "solution": "No problem! Please provide your instrument or accessory requirements, and I'll help you identify what you need for your project.",
                        "product_info": "Understood! Feel free to ask me about vendors, product specifications, or procurement strategies from our database.",
                        "search": "Got it! Please describe what you're looking for, and I'll find the best matching products for you."
                    }
                    decline_message = decline_messages.get(current_page, f"No problem! How can I help you with {current_page_name} functionality?")
                    popup_blocked_message = f"The popup was blocked. Please [link] to open the {target_page_name} page."
        
        return jsonify({
            "category": category,
            "target_page": target_page,
            "current_page": current_page,
            "requires_confirmation": requires_confirmation,
            "requires_routing": requires_confirmation,  # True if routing to different page
            "direct_response": "",  # Empty for routing categories
            "confirmation_message": confirmation_message if requires_confirmation else "",
            "opening_message": opening_message if requires_confirmation else "",
            "decline_message": decline_message if requires_confirmation else "",
            "popup_blocked_message": popup_blocked_message if requires_confirmation else "",
            "original_query": user_input,
            "reasoning": reasoning
        }), 200
        
    except Exception as e:
        logging.exception("[ROUTE_CLASSIFIER] Classification failed")
        return jsonify({
            "error": "Classification failed: " + str(e),
            "category": "error",
            "target_page": current_page,
            "current_page": current_page,
            "requires_confirmation": False,
            "requires_routing": False,
            "direct_response": "I'm having trouble processing your request. Please try again."
        }), 500


# =========================================================================
# === ROUTE CONFIRMATION API (LLM-based yes/no classification) ===
# =========================================================================
@app.route("/api/route-confirm", methods=["POST"])
@login_required
def route_confirm():
    """
    Classifies user's response to a routing confirmation prompt.
    Uses LLM to determine if user is confirming (yes), declining (no), or something else.
    
    Request body:
    {
        "user_input": "user's response to confirmation",
        "current_page": "solution|product_info|search",
        "target_page": "solution|product_info|search",
        "original_query": "the original query that triggered the routing"
    }
    
    Returns:
    {
        "action": "confirm|decline|unclear",
        "message": "LLM-generated response message",
        "proceed_with_routing": true/false,
        "target_page": "the page to route to (if confirm)"
    }
    """
    try:
        data = request.get_json(force=True)
        user_input = data.get("user_input", "").strip()
        current_page = data.get("current_page", "solution").lower().strip()
        target_page = data.get("target_page", "").lower().strip()
        original_query = data.get("original_query", "").strip()
        
        if not user_input:
            return jsonify({"error": "user_input is required"}), 400
        
        if not target_page:
            return jsonify({"error": "target_page is required"}), 400
        
        logging.info(f"[ROUTE_CONFIRM] Input: '{user_input}', Target: {target_page}, Current: {current_page}")
        
        # Page name mappings
        page_names = {
            "solution": "Solution",
            "product_info": "Product Info",
            "search": "Search"
        }
        
        page_descriptions = {
            "solution": "Solution page to identify instruments and accessories",
            "product_info": "Product Info page to query vendor database",
            "search": "Search page to find matching products"
        }
        
        target_page_name = page_names.get(target_page, target_page)
        current_page_name = page_names.get(current_page, current_page)
        
        # Use LLM to classify the confirmation response
        classification_prompt = """
You are Engenie - a friendly industrial automation assistant.

The user was asked if they want to navigate from {current_page} to {target_page}.
They already saw a confirmation prompt asking if they'd like to open the {target_page} page.

Now analyze the user's response to determine their intent:

User's Response: "{user_input}"

CLASSIFICATION RULES:
1. If user is CONFIRMING (yes, yeah, sure, okay, proceed, go ahead, please, y, affirmative, etc.) → "confirm"
2. If user is DECLINING (no, nope, cancel, never mind, stay, n, don't, negative, etc.) → "decline"  
3. If user is asking something NEW or the response is UNCLEAR → "unclear"

Consider context and variations:
- "yes please" → confirm
- "sure, go ahead" → confirm
- "nah, I'll stay here" → decline
- "actually no" → decline
- "I need a temperature sensor" → unclear (this is a new request)
- "what can you do?" → unclear (this is a new question)

Respond with ONLY a JSON object:
{{
  "action": "<confirm|decline|unclear>",
  "reasoning": "<brief explanation>"
}}

No additional text outside the JSON.
"""
        
        classification_chain = ChatPromptTemplate.from_template(classification_prompt) | components['llm_flash'] | StrOutputParser()
        classification_response = classification_chain.invoke({
            "current_page": current_page_name,
            "target_page": target_page_name,
            "user_input": user_input
        }).strip()
        
        # Clean and parse response
        if classification_response.startswith("```json"):
            classification_response = classification_response[7:]
        elif classification_response.startswith("```"):
            classification_response = classification_response[3:]
        if classification_response.endswith("```"):
            classification_response = classification_response[:-3]
        classification_response = classification_response.strip()
        
        result = json.loads(classification_response)
        action = result.get("action", "unclear").lower().strip()
        reasoning = result.get("reasoning", "")
        
        # Validate action
        if action not in ["confirm", "decline", "unclear"]:
            action = "unclear"
        
        logging.info(f"[ROUTE_CONFIRM] Action: {action}, Reasoning: {reasoning}")
        
        # Generate appropriate response message based on action
        message = ""
        proceed_with_routing = False
        
        if action == "confirm":
            proceed_with_routing = True
            # Generate opening message
            message_prompt = f"""You are Engenie - a friendly industrial automation assistant.
The user confirmed they want to go to the {target_page_name} page.
Generate a brief, friendly message (1 sentence) saying you're opening that page for them.
Respond ONLY with the message text, no quotes or formatting."""
            from langchain_core.messages import HumanMessage
            message = components['llm_flash'].invoke([HumanMessage(content=message_prompt)]).content.strip()
            
        elif action == "decline":
            proceed_with_routing = False
            
            # Determine instruction based on current page
            if current_page == "solution":
                instruction = "Respond with a variation of: 'No problem! Please provide requirements, and I'll help you identify what you need.'"
            elif current_page == "search":
                instruction = "Respond with a variation of: 'Understood. Let's continue finding the best matching products here. What else are you looking for?'"
            elif current_page == "product_info":
                instruction = "Respond with a variation of: 'No problem. Is there anything else you would like to know about ?'"
            else:
                instruction = f"Acknowledge their choice to stay on the {current_page_name} page and ask them to provide relevant input."

            # Generate decline message that asks for relevant input
            message_prompt = f"""You are Engenie - a friendly industrial automation assistant.
The user declined to go to {target_page_name} and wants to stay on {current_page_name}.

{instruction}

Respond ONLY with the message text, no quotes or formatting."""
            from langchain_core.messages import HumanMessage
            message = components['llm_flash'].invoke([HumanMessage(content=message_prompt)]).content.strip()
            
        else:  # unclear
            proceed_with_routing = False
            # The input is unclear - return this so frontend can treat it as new input
            message = ""  # No message needed, frontend will re-process as new input
        
        return jsonify({
            "action": action,
            "message": message,
            "proceed_with_routing": proceed_with_routing,
            "target_page": target_page if proceed_with_routing else current_page,
            "original_query": original_query,
            "reasoning": reasoning
        }), 200
        
    except Exception as e:
        logging.exception("[ROUTE_CONFIRM] Classification failed")
        return jsonify({
            "error": "Confirmation classification failed: " + str(e),
            "action": "unclear",
            "proceed_with_routing": False
        }), 500


# =========================================================================
# === PROJECT INTENT CLASSIFICATION API ===
# =========================================================================
@app.route("/api/project-intent", methods=["POST"])
@login_required
def project_intent():
    """
    Solution page intent classifier for instrument identification and modification.
    
    NOTE: Greeting, questions, product_info queries, and unrelated content are now handled by 
    /api/route-classifier. This API focuses ONLY on:
    - requirements: User is providing NEW industrial requirements
    - modification: User wants to modify existing instruments/accessories
    - confirm_search: User confirms to search (single item flow)
    - add_more: User wants to add more requirements (single item flow)
    
    INTENT TYPES:
    -------------
    1. "requirements" - User is providing new industrial requirements
    2. "modification" - User wants to modify existing instruments/accessories
    3. "confirm_search" - User confirms to search (single item flow)
    4. "add_more" - User wants to add more (single item flow)
    
    REQUEST:
    --------
    POST /api/project-intent
    {
        "user_input": "<user's message>",
        "current_instruments": [...],  // Optional - existing instruments
        "current_accessories": [...],  // Optional - existing accessories
        "awaiting_single_item_confirmation": true/false  // Optional - for single item flow
    }
    
    RESPONSE:
    ---------
    {
        "intent": "<intent_type>",
        "confidence": "high|medium|low",
        "reasoning": "<explanation>",
        "should_call": "<api_to_call>"  // "identify-instruments", "modify-instruments", or null
    }
    """
    if not components or not components.get('llm_flash'):
        return jsonify({"error": "LLM component is not ready."}), 503

    try:
        data = request.get_json(force=True)
        user_input = data.get("user_input", "").strip()
        
        if not user_input:
            return jsonify({"error": "user_input is required"}), 400
        
        # Get existing data context
        current_instruments = data.get("current_instruments", [])
        current_accessories = data.get("current_accessories", [])
        has_existing_data = len(current_instruments) > 0 or len(current_accessories) > 0
        
        # Check for single item confirmation flow
        awaiting_single_item_confirmation = data.get("awaiting_single_item_confirmation", False)
        
        logging.info(f"[PROJECT_INTENT] Classifying input: '{user_input[:100]}...'")
        logging.info(f"[PROJECT_INTENT] has_existing_data: {has_existing_data}, awaiting_confirmation: {awaiting_single_item_confirmation}")
        
        # =====================================================
        # SPECIAL CASE: Single Item Confirmation Flow
        # =====================================================
        if awaiting_single_item_confirmation and has_existing_data:
            logging.info("[PROJECT_INTENT] Processing single item confirmation flow...")
            
            try:
                intent_prompt = f"""
You are an intelligent assistant analyzing user intent.
Context: The user was just shown a single identified item and asked: "Would you like to add more requirements, or should I search for the best match?".

User Input: "{user_input}"

Classify the user's response into one of these categories:
1. CONFIRM_SEARCH: User wants to proceed with the search (e.g., "Yes", "Go ahead", "Find best match", "Search", "Sure", "Okay", "Proceed").
2. ADD_MORE: User wants to add more items/requirements (e.g., "No", "Add more", "Wait", "I have another item", "Not yet").
3. MODIFY: User wants to change/update the identified item (e.g., "Change to X", "Actually I need Y", "Wrong item", "Make it Z", "I need to change...").
4. QUESTION: User is asking a question (e.g., "What is a pressure transmitter?", "How does it work?", "Explain this").
5. OTHER: Greeting, unrelated, or unclear.

Respond ONLY with a JSON object:
{"intent": "CONFIRM_SEARCH" | "ADD_MORE" | "MODIFY" | "QUESTION" | "OTHER"}
"""
                # Use HumanMessage directly to avoid template parsing issues with JSON braces
                from langchain_core.messages import HumanMessage
                intent_response = components['llm'].invoke([HumanMessage(content=intent_prompt)])
                intent_json = intent_response.content
                
                logging.info(f"[PROJECT_INTENT] Raw LLM response: '{intent_json}'")
                
                if not intent_json or not intent_json.strip():
                    logging.warning("[PROJECT_INTENT] LLM returned empty response, defaulting to MODIFY")
                    intent = "MODIFY"
                else:
                    # Clean the response
                    cleaned_intent_json = intent_json.strip()
                    if cleaned_intent_json.startswith("```json"):
                        cleaned_intent_json = cleaned_intent_json[7:]
                    elif cleaned_intent_json.startswith("```"):
                        cleaned_intent_json = cleaned_intent_json[3:]
                    if cleaned_intent_json.endswith("```"):
                        cleaned_intent_json = cleaned_intent_json[:-3]
                    cleaned_intent_json = cleaned_intent_json.strip()
                    
                    intent_data = json.loads(cleaned_intent_json)
                    intent = intent_data.get("intent", "OTHER").upper()
                    
            except Exception as e:
                logging.warning(f"[PROJECT_INTENT] LLM classification failed, using keyword fallback: {e}")
                # Keyword-based fallback
                req_lower = user_input.strip().lower()
                confirm_keywords = ['yes', 'y', 'yeah', 'yep', 'sure', 'ok', 'okay', 'go', 'go ahead', 'proceed', 'search', 'find', 'correct', 'right', 'confirm']
                reject_keywords = ['no', 'n', 'nope', 'not yet', 'wait', 'add more', 'another', 'more']
                
                if any(req_lower == kw or req_lower.startswith(kw + ' ') or req_lower.endswith(' ' + kw) for kw in confirm_keywords):
                    intent = "CONFIRM_SEARCH"
                elif any(req_lower == kw or req_lower.startswith(kw + ' ') or req_lower.endswith(' ' + kw) for kw in reject_keywords):
                    intent = "ADD_MORE"
                else:
                    intent = "MODIFY"
            
            # Handle specific intents for single item flow
            logging.info(f"[PROJECT_INTENT] Single item flow intent classified as: {intent}")
            
            if intent == "CONFIRM_SEARCH":
                logging.info("[PROJECT_INTENT] User confirmed search - preparing dashboard trigger response")
                
                # Generate confirmation message
                try:
                    confirm_prompt = """
You are Engenie - a friendly industrial automation assistant.

The user has confirmed they want to search for the best matching products.

Generate a brief, enthusiastic confirmation message (2-3 sentences) that:
1. Confirms you'll now search for the best matching products
2. Mentions a new window will open with detailed analysis
3. Asks if they would like to add more requirements

Example: "Great! I'm opening a detailed search to find the best matching products for your requirement. A new window will open with the detailed analysis. Would you like to add more requirements?"

Respond ONLY with the message text, no JSON.
"""
                    # Use HumanMessage directly to avoid template parsing issues
                    from langchain_core.messages import HumanMessage
                    confirm_response = components['llm'].invoke([HumanMessage(content=confirm_prompt)])
                    confirm_message = confirm_response.content
                except Exception as e:
                    logging.warning(f"Failed to generate confirmation message: {e}")
                    confirm_message = "Great! I'm opening a detailed search to find the best matching products for your requirement. Would you like to add more requirements?"
                
                # Get sample input from current instruments/accessories
                sample_input = ""
                if current_instruments:
                    inst = current_instruments[0]
                    sample_input = inst.get("sample_input") or inst.get("sampleInput") or ""
                elif current_accessories:
                    acc = current_accessories[0]
                    sample_input = acc.get("sample_input") or acc.get("sampleInput") or ""
                
                return jsonify({
                    "intent": "confirm_search",
                    "confidence": "high",
                    "reasoning": "Single item confirmation flow - user response classified as CONFIRM_SEARCH",
                    "should_call": None,
                    "is_single_item_flow": True,
                    "open_dashboard": True,
                    "sample_input": sample_input,
                    "message": confirm_message.strip(),
                    "instruments": current_instruments,
                    "accessories": current_accessories
                }), 200
            
            elif intent == "ADD_MORE":
                logging.info("[PROJECT_INTENT] User wants to add more - preparing add more response")
                
                # Generate add more message
                try:
                    add_more_prompt = """
You are Engenie - a friendly industrial automation assistant.

The user wants to add more requirements to their project instead of searching now.

Generate a friendly, encouraging message (1-2 sentences) that:
1. Acknowledges their choice to add more items
2. Invites them to provide additional instruments or accessories
3. Mentions they can modify existing items too

Example: "Sounds good! Please go ahead and provide more requirements. You can add new instruments, accessories, or modify the existing ones."

Respond ONLY with the message text, no JSON.
"""
                    # Use HumanMessage directly to avoid template parsing issues
                    from langchain_core.messages import HumanMessage
                    add_more_response = components['llm'].invoke([HumanMessage(content=add_more_prompt)])
                    add_more_message = add_more_response.content
                except Exception as e:
                    logging.warning(f"Failed to generate add more message: {e}")
                    add_more_message = "Sounds good! Please go ahead and provide more requirements. You can add new instruments, accessories, or modify the existing ones."
                
                return jsonify({
                    "intent": "add_more",
                    "confidence": "high",
                    "reasoning": "Single item confirmation flow - user response classified as ADD_MORE",
                    "should_call": None,
                    "is_single_item_flow": True,
                    "awaiting_more_requirements": True,
                    "message": add_more_message.strip(),
                    "instruments": current_instruments,
                    "accessories": current_accessories
                }), 200
            
            elif intent == "QUESTION":
                logging.info("[PROJECT_INTENT] User asked a question - routing to project-response with pending confirmation")
                return jsonify({
                    "intent": "question",
                    "confidence": "high",
                    "reasoning": "Single item confirmation flow - user asked a question",
                    "should_call": "project-response",
                    "is_single_item_flow": True,
                    "still_awaiting_confirmation": True
                }), 200
            
            else:  # MODIFY or OTHER - route to appropriate API
                if intent == "MODIFY":
                    return jsonify({
                        "intent": "modification",
                        "confidence": "high",
                        "reasoning": "Single item confirmation flow - user wants to modify the item",
                        "should_call": "modify-instruments",
                        "is_single_item_flow": True
                    }), 200
                else:
                    return jsonify({
                        "intent": "unrelated",
                        "confidence": "medium",
                        "reasoning": "Single item confirmation flow - unclear or unrelated response",
                        "should_call": "project-response",
                        "is_single_item_flow": True,
                        "still_awaiting_confirmation": True
                    }), 200
        
        # =====================================================
        # STANDARD INTENT CLASSIFICATION
        # =====================================================
        # NOTE: Greeting, question, product_info, and unrelated are now handled by route-classifier.
        # This API only classifies: requirements vs modification
        
        # Build dynamic classification prompt
        modification_category = ""
        modification_types = ""
        if has_existing_data:
            modification_category = """
2. "modification" - If the user wants to MODIFY, ADD TO, or REMOVE FROM the existing list of instruments/accessories.
   This includes:
   - Adding new items: "Add a control valve", "I also need a flow meter", "Include another sensor"
   - Removing items: "Remove the temperature transmitter", "Delete the flow meter", "Take out the valve"
   - Updating specs: "Change the pressure range to 0-200 psi", "Update the material to Hastelloy"
   - Quantity changes: "I need 2 more transmitters", "Reduce to 1 valve", "Add another one"
   
   **KEY INDICATORS OF MODIFICATION**:
   - Words like: add, remove, delete, change, update, modify, include, also need, more, another, additional, replace, swap, edit, increase, decrease, fewer
   - References to existing items being changed
   - Requests that imply changing the current list rather than starting fresh

   **IMPORTANT**: If the user provides completely NEW requirements (a full new project description), classify as "requirements" not "modification".
   Modification is for incremental changes to an existing list.
"""
            modification_types = "|modification"
        
        classification_prompt = f"""
You are an intelligent classifier for an industrial automation assistant named "Engenie".

{f"**CONTEXT**: The user already has an existing list of instruments and accessories. Consider if they want to modify this existing list." if has_existing_data else ""}

Analyze the user's input and classify it into ONE of these categories:

1. "requirements" - If the user is providing NEW/FRESH technical requirements, specifications, or asking to identify instruments/equipment for an industrial process. This includes:
   - Process control requirements (pressure, temperature, flow, level measurements)
   - Industrial equipment specifications (valves, transmitters, controllers, actuators)
   - Automation system requirements
   - Instrumentation needs for industrial processes
   - Technical specifications with measurements, ranges, materials, standards (ANSI, ASME, DIN, ISO, API)
   {f"**NOTE**: Only use 'requirements' if it's a COMPLETELY NEW set of requirements, not changes to existing items." if has_existing_data else ""}
{modification_category}
**CLASSIFICATION RULES**:
1. If user provides industrial equipment specifications/requirements → "requirements"
{f"2. If user wants to ADD, REMOVE, CHANGE, or UPDATE existing items → 'modification'" if has_existing_data else ""}

User Input: {{user_input}}

Respond with ONLY a JSON object in this exact format:
{{{{
  "type": "<requirements{modification_types}>",
  "confidence": "<high|medium|low>",
  "reasoning": "<brief explanation of why you chose this classification>"
}}}}

No additional text or explanation outside the JSON.
"""
    
        classification_chain = ChatPromptTemplate.from_template(classification_prompt) | components['llm_flash'] | StrOutputParser()
        classification_response = classification_chain.invoke({"user_input": user_input})
        
        # Clean and parse classification
        cleaned_classification = classification_response.strip()
        if cleaned_classification.startswith("```json"):
            cleaned_classification = cleaned_classification[7:]
        elif cleaned_classification.startswith("```"):
            cleaned_classification = cleaned_classification[3:]
        if cleaned_classification.endswith("```"):
            cleaned_classification = cleaned_classification[:-3]
        cleaned_classification = cleaned_classification.strip()
        
        try:
            classification = json.loads(cleaned_classification)
            input_type = classification.get("type", "requirements").lower()
            confidence = classification.get("confidence", "medium").lower()
            reasoning = classification.get("reasoning", "")
            
            # Validate input_type is one of the supported types
            if input_type not in ["requirements", "modification"]:
                input_type = "requirements"  # Default to requirements
            
            logging.info(f"[PROJECT_INTENT] Classified as '{input_type}' (confidence: {confidence})")
            logging.info(f"[PROJECT_INTENT] Reasoning: {reasoning}")
            
        except Exception as e:
            logging.warning(f"[PROJECT_INTENT] Failed to parse classification, defaulting to requirements: {e}")
            input_type = "requirements"
            confidence = "low"
            reasoning = "Classification parsing failed, defaulting to requirements"
        
        # Map intent to API call (simplified - only requirements and modification)
        api_mapping = {
            "requirements": "identify-instruments",
            "modification": "modify-instruments"
        }
        
        should_call = api_mapping.get(input_type, "identify-instruments")
        
        return jsonify({
            "intent": input_type,
            "confidence": confidence,
            "reasoning": reasoning,
            "should_call": should_call,
            "has_existing_data": has_existing_data
        }), 200
        
    except Exception as e:
        logging.exception("[PROJECT_INTENT] Intent classification failed")
        return jsonify({
            "error": "Failed to classify intent: " + str(e),
            "intent": "requirements",
            "confidence": "low",
            "should_call": "identify-instruments"
        }), 500


# =========================================================================
# === PROJECT RESPONSE API (DEPRECATED - Use route-classifier instead) ===
# =========================================================================
@app.route("/api/project-response", methods=["POST"])
@login_required
def project_response():
    """
    DEPRECATED: This API is deprecated. Greetings, questions, and unrelated content
    are now handled directly by /api/route-classifier with direct_response.
    
    This API is kept for backward compatibility with legacy frontend code.
    New code should use /api/route-classifier which returns direct_response for
    non-routing categories (greeting, question_general, chitchat, out_of_scope, etc.).
    
    REQUEST:
    --------
    POST /api/project-response
    {
        "user_input": "<user's message>",
        "intent": "<greeting|question|unrelated>",
        "current_instruments": [...],  // Optional - for context
        "current_accessories": [...],  // Optional - for context
        "reasoning": "<optional reasoning from intent API>"
    }
    
    RESPONSE:
    ---------
    {
        "response_type": "<greeting|question>",
        "message": "<LLM generated response>",
        "is_industrial": true/false  // For questions only
    }
    """
    if not components or not components.get('llm_flash'):
        return jsonify({"error": "LLM component is not ready."}), 503

    try:
        data = request.get_json(force=True)
        user_input = data.get("user_input", "").strip()
        intent = data.get("intent", "question").lower()
        reasoning = data.get("reasoning", "")
        
        # Context for single item flow
        current_instruments = data.get("current_instruments", [])
        current_accessories = data.get("current_accessories", [])
        still_awaiting_confirmation = data.get("still_awaiting_confirmation", False)
        
        if not user_input:
            return jsonify({"error": "user_input is required"}), 400
        
        logging.info(f"[PROJECT_RESPONSE] Processing intent: {intent}, input: '{user_input[:50]}...'")
        
        # =====================================================
        # GREETING RESPONSE
        # =====================================================
        if intent == "greeting":
            greeting_prompt = """
You are Engenie - a friendly and professional industrial automation assistant.

The user has greeted you with: "{user_input}"

Respond with a warm, professional greeting and briefly introduce yourself. 
Mention that you can help them identify instruments and accessories for their industrial projects.
Keep it concise (2-3 sentences max).

Respond ONLY with the greeting text, no JSON.
"""
            greeting_chain = ChatPromptTemplate.from_template(greeting_prompt) | components['llm_flash'] | StrOutputParser()
            greeting_response = greeting_chain.invoke({"user_input": user_input})
            
            return standardized_jsonify({
                "response_type": "greeting",
                "message": greeting_response.strip(),
                "instruments": [],
                "accessories": []
            }, 200)
        
        # =====================================================
        # UNRELATED CONTENT RESPONSE
        # =====================================================
        elif intent == "unrelated":
            unrelated_prompt = """
You are Engenie - a friendly and professional industrial automation assistant.

The user has provided content that appears to be unrelated to industrial automation or instrumentation.

Content type detected: {reasoning}

Craft a polite, helpful response that:
1. Acknowledges their input
2. Explains that you specialize in industrial automation, process control, and instrumentation
3. Provides examples of what you CAN help with (e.g., "identifying instruments for a distillation column", "specifying pressure transmitters", "selecting control valves")
4. Invites them to provide industrial requirements or ask industrial-related questions

Keep it friendly, professional, and concise (2-3 sentences).

Respond ONLY with the message text, no JSON.
"""
            unrelated_chain = ChatPromptTemplate.from_template(unrelated_prompt) | components['llm_flash'] | StrOutputParser()
            unrelated_response = unrelated_chain.invoke({"reasoning": reasoning})
            
            return standardized_jsonify({
                "response_type": "question",  # Use "question" type for frontend compatibility
                "is_industrial": False,
                "message": unrelated_response.strip(),
                "instruments": [],
                "accessories": []
            }, 200)
        
        # =====================================================
        # QUESTION RESPONSE (with optional single item reminder)
        # =====================================================
        elif intent == "question":
            # Check if we need to remind about pending single item choice
            if still_awaiting_confirmation and (current_instruments or current_accessories):
                if current_instruments:
                    item_name = current_instruments[0].get("product_name", current_instruments[0].get("category", "the instrument"))
                else:
                    item_name = current_accessories[0].get("accessory_name", current_accessories[0].get("category", "the accessory"))
                
                question_with_reminder_prompt = f"""
You are Engenie - an expert in Industrial Process Control Systems.

CONTEXT: The user has already identified a single item: **{item_name}**
They were asked if they want to find the best match or add more requirements.
Instead of answering yes/no, they asked a QUESTION.

USER'S QUESTION: "{user_input}"

Your task:
1. First, provide a helpful, accurate answer to their question (2-4 sentences)
2. Then, at the END of your response, add a separator and remind them about their pending choice

Format your response like this:
[Answer to their question]

---

*By the way, would you prefer to **add more requirements**? or would you still like me to **find the best matching products** for your {item_name}?*

Respond ONLY with the formatted message text, no JSON.
"""
                try:
                    question_chain = ChatPromptTemplate.from_template(question_with_reminder_prompt) | components['llm_flash'] | StrOutputParser()
                    question_response = question_chain.invoke({})
                except Exception as e:
                    logging.warning(f"[PROJECT_RESPONSE] Failed to generate question response with reminder: {e}")
                    question_response = f"I'd be happy to help with that. However, I noticed you haven't yet decided about the {item_name}. Would you like me to **find the best matching products** for it, or would you prefer to **add more requirements** first?"
                
                return standardized_jsonify({
                    "response_type": "question_with_pending_choice",
                    "message": question_response.strip(),
                    "instruments": current_instruments,
                    "accessories": current_accessories,
                    "still_awaiting_confirmation": True
                }, 200)
            
            # Standard question response
            question_prompt = """
You are Engenie - an expert in Industrial Process Control Systems and automation.

The user has asked: "{user_input}"

First, determine if this question is related to industrial automation, process control, instrumentation, or related topics.

If YES (related to industrial topics):
- Provide a helpful, accurate, and concise answer (2-4 sentences)
- Focus on practical information

If NO (not related to industrial topics):
- Politely redirect the user
- Mention that you specialize in industrial automation and instrumentation
- Suggest they ask about industrial topics or provide requirements for instrument identification

Respond ONLY with a JSON object in this format:
{{
  "is_industrial": <true|false>,
  "answer": "<your response text>"
}}

No additional text outside the JSON.
"""
            question_chain = ChatPromptTemplate.from_template(question_prompt) | components['llm_flash'] | StrOutputParser()
            question_response = question_chain.invoke({"user_input": user_input})
            
            # Clean and parse
            cleaned_question = question_response.strip()
            if cleaned_question.startswith("```json"):
                cleaned_question = cleaned_question[7:]
            elif cleaned_question.startswith("```"):
                cleaned_question = cleaned_question[3:]
            if cleaned_question.endswith("```"):
                cleaned_question = cleaned_question[:-3]
            cleaned_question = cleaned_question.strip()
            
            try:
                question_result = json.loads(cleaned_question)
                return standardized_jsonify({
                    "response_type": "question",
                    "is_industrial": question_result.get("is_industrial", True),
                    "message": question_result.get("answer", ""),
                    "instruments": [],
                    "accessories": []
                }, 200)
            except:
                # Fallback
                fallback_prompt = f"You are Engenie, an industrial automation assistant. The user asked: '{user_input}'. Provide a brief, helpful response about industrial automation and instrumentation (2-3 sentences)."
                fallback_chain = ChatPromptTemplate.from_template(fallback_prompt) | components['llm_flash'] | StrOutputParser()
                fallback_message = fallback_chain.invoke({})
                
                return standardized_jsonify({
                    "response_type": "question",
                    "is_industrial": False,
                    "message": fallback_message.strip(),
                    "instruments": [],
                    "accessories": []
                }, 200)
        
        # =====================================================
        # SINGLE ITEM CONFIRMATION RESPONSES
        # =====================================================
        elif intent == "confirm_search":
            # Generate confirmation message for YES
            confirm_prompt = """
You are Engenie - a friendly industrial automation assistant.

The user has confirmed they want to search for the best matching products.

Generate a brief, enthusiastic confirmation message (2-3 sentences) that:
1. Confirms you'll now search for the best matching products
2. Mentions a new window will open with detailed analysis
3. Asks if they would like to add more requirements

Example: "Great! I'm opening a detailed search to find the best matching products for your requirement. A new window will open with the detailed analysis. Would you like to add more requirements?"

Respond ONLY with the message text, no JSON.
"""
            try:
                confirm_chain = ChatPromptTemplate.from_template(confirm_prompt) | components['llm'] | StrOutputParser()
                confirm_message = confirm_chain.invoke({})
            except Exception as e:
                logging.warning(f"Failed to generate confirmation message: {e}")
                confirm_message = "Great! I'm opening a detailed search to find the best matching products for your requirement. Would you like to add more requirements?"
            
            # Get sample input
            sample_input = ""
            if current_instruments:
                inst = current_instruments[0]
                sample_input = inst.get("sample_input") or inst.get("sampleInput") or ""
            elif current_accessories:
                acc = current_accessories[0]
                sample_input = acc.get("sample_input") or acc.get("sampleInput") or ""
            
            return standardized_jsonify({
                "response_type": "single_item_confirmed_yes",
                "message": confirm_message.strip(),
                "instruments": current_instruments,
                "accessories": current_accessories,
                "open_dashboard": True,
                "sample_input": sample_input
            }, 200)
        
        elif intent == "add_more":
            # Generate message for NO (add more)
            add_more_prompt = """
You are Engenie - a friendly industrial automation assistant.

The user wants to add more requirements to their project instead of searching now.

Generate a friendly, encouraging message (1-2 sentences) that:
1. Acknowledges their choice to add more items
2. Invites them to provide additional instruments or accessories
3. Mentions they can modify existing items too

Example: "Sounds good! Please go ahead and provide more requirements. You can add new instruments, accessories, or modify the existing ones."

Respond ONLY with the message text, no JSON.
"""
            try:
                add_more_chain = ChatPromptTemplate.from_template(add_more_prompt) | components['llm'] | StrOutputParser()
                add_more_message = add_more_chain.invoke({})
            except Exception as e:
                logging.warning(f"Failed to generate add more message: {e}")
                add_more_message = "Sounds good! Please go ahead and provide more requirements. You can add new instruments, accessories, or modify the existing ones."
            
            return standardized_jsonify({
                "response_type": "single_item_confirmed_no",
                "message": add_more_message.strip(),
                "instruments": current_instruments,
                "accessories": current_accessories,
                "awaiting_more_requirements": True
            }, 200)
        
        # Default fallback
        else:
            return standardized_jsonify({
                "response_type": "question",
                "message": "I'm here to help with industrial automation and instrumentation. Please provide your requirements or ask a related question.",
                "instruments": [],
                "accessories": []
            }, 200)
        
    except Exception as e:
        logging.exception("[PROJECT_RESPONSE] Response generation failed")
        return jsonify({
            "response_type": "error",
            "error": "Failed to generate response: " + str(e),
            "message": "I apologize, but I'm having technical difficulties. Please try again."
        }), 500


# =========================================================================
# === INSTRUMENT IDENTIFICATION ENDPOINT (Focused on Requirements Only) ===
# =========================================================================
@app.route("/api/identify-instruments", methods=["POST"])
@login_required
def identify_instruments():
    """
    Focused API for identifying instruments and accessories from user requirements.
    This API should be called after project-intent returns intent="requirements".
    
    REQUEST:
    --------
    POST /api/identify-instruments
    {
        "requirements": "<user's technical requirements>",
        "search_session_id": "<optional session ID>"
    }
    
    RESPONSE:
    ---------
    {
        "response_type": "requirements" | "single_item_prompt",
        "project_name": "<extracted project name>",
        "instruments": [...],
        "accessories": [...],
        "message": "<summary message>",
        "awaiting_confirmation": true/false  // For single item flow
    }
    """
    if not components or not components.get('llm_pro'):
        return jsonify({"error": "LLM component is not ready."}), 503

    try:
        data = request.get_json(force=True)
        requirements = data.get("requirements", "").strip()
        search_session_id = data.get("search_session_id", "default")
        
        if not requirements:
            return jsonify({"error": "Requirements text is required"}), 400
        
        logging.info(f"[IDENTIFY_INSTRUMENTS] Processing requirements: '{requirements[:100]}...'")
        
        instrument_prompt = """
You are Engenie - an expert assistant in Industrial Process Control Systems. Analyze the given requirements and identify the Bill of Materials (instruments) needed.
**IMPORTANT: Think step-by-step through your identification process.**

Before providing the final JSON:
1. First, read the requirements and extract a concise project name (1-2 words) that best represents the objective.
2. Then, identify every instrument required by the problem statement. For each instrument, determine category, generic product name, quantity (explicit or inferred), and all specifications mentioned.
3. For any specification not explicitly present but required for sensible procurement (e.g., typical ranges, common materials), mark it with the tag `[INFERRED]` and explain briefly in an internal analysis (see Validation step).
4. Create a clear `sample_input` field for each instrument that contains every specification key/value exactly as listed in the `specifications` object.
5. **ACCESSORY INFERENCE RULE:**
   - If user explicitly says "no accessories", "without accessories", "only the instrument", "just the [instrument]", or similar → Do NOT add any accessories
   - If user does NOT mention anything about accessories → Auto-infer relevant accessories of that instrument(impulse lines, isolation valves, manifolds, mounting brackets, junction boxes, power supplies, calibration kits, connectors)
6. **VENDOR EXTRACTION RULE (CRITICAL):**
   - If user mentions specific vendor/manufacturer names (e.g., "Honeywell", "ABB", "Emerson", "Siemens", "Yokogawa", "Endress+Hauser", "Rosemount", etc.), extract them for EACH instrument/accessory they apply to.
   - Example: "I need a pressure transmitter from Honeywell and flow meter from Emerson" → PT gets ["Honeywell"], Flow Meter gets ["Emerson"]
   - Example: "I need Honeywell instruments" → ALL instruments get ["Honeywell"]
   - If NO vendor is mentioned for an instrument, leave specified_vendors as an empty array []
7. **STANDARDS EXTRACTION RULE (CRITICAL):**
   - **User-Mentioned Standards**: If user mentions specific engineering standards (e.g., "as per ISA-S20", "per IEC 61511", "following API 551", "ASME B31.3", "NEMA 4X", "ATEX certified", "SIL 2 rated", etc.), extract ALL mentioned standards into the `user_specified_standards` array.
   - **Skip Standards Detection**: If user explicitly says "no standards required", "skip standards", "without standards compliance", "don't need standards", "ignore standards", or similar → set `skip_standards` to true
   - **Default Behavior**: If user doesn't mention anything about standards → leave `user_specified_standards` as empty array and `skip_standards` as false (system will auto-fetch from user's uploaded standards document)
   - Examples:
     - "Pressure transmitter as per ISA-S20" → user_specified_standards: ["ISA-S20"], skip_standards: false
     - "Flow meter per IEC 61511 and API 551" → user_specified_standards: ["IEC 61511", "API 551"], skip_standards: false
     - "Temperature sensor, no standards needed" → user_specified_standards: [], skip_standards: true
     - "Level transmitter with ATEX certification" → user_specified_standards: ["ATEX"], skip_standards: false
     - "Just need a basic PT, no certifications required" → user_specified_standards: [], skip_standards: true

Requirements:
{requirements}

Instructions:
1. Extract a unique, descriptive project name (1-2 words) from the requirements that best represents the objective of the industrial system or process described. This should be concise and professional.
2. Identify all instruments required for the given Industrial Process Control System Problem Statement 
3. For each instrument, provide:
   - Category (e.g., Pressure Transmitter, Temperature Transmitter, Flow Meter, etc.)
   - Product Name (generic name based on the requirements)
   - Quantity
   - Specifications (extract from requirements or infer based on industry standards)
   - Strategy (analyze user requirements to identify procurement approach: budget constraints suggest "Cost optimization", quality emphasis suggests "Life-cycle cost evaluation", sustainability mentions suggest "Sustainability and green procurement", critical applications suggest "Dual sourcing", standard applications suggest "Framework agreement", or leave empty if none identified)
   - Specified Vendors (extract vendor/manufacturer names mentioned by user for THIS specific instrument, empty array if none)
   - Specified Model Families (extract model/series names if user mentions them, e.g., "Rosemount 3051" → ["3051"], "Honeywell STT850" → ["STT850"], empty array if none)
   - Sample Input(must include all specification details exactly as listed in the specifications field (no field should be missing)).
   - Ensure every parameter appears explicitly in the sample input text.
4. Mark inferred requirements explicitly with [INFERRED] tag
5. **IMPORTANT:** Extract user-mentioned engineering standards FOR EACH instrument/accessory individually
6. Detect if user wants to skip standards FOR EACH instrument/accessory individually
 
Additionally, identify any accessories, consumables, or ancillary items required to support the instruments (for example: impulse lines, mounting brackets, isolation valves, manifolds, cable/connector types, junction boxes, power supplies, or calibration kits). For accessories, provide:
    - Category (e.g., Impulse Line, Isolation Valve, Mounting Bracket, Junction Box)
    - Accessory Name (generic)
    - Quantity
    - Specifications (size, material, pressure rating, connector type, etc.)
    - Strategy (analyze user requirements for procurement approach or leave empty if none identified)
    - Specified Vendors: **VENDOR PRIORITY RULE**
      1. If user EXPLICITLY mentions a vendor for THIS accessory → use that vendor (e.g., "impulse lines from Parker" → specified_vendors: ["Parker"])
      2. If no explicit accessory vendor BUT parent instrument has a vendor → INHERIT from parent (e.g., "Honeywell PT" → all its accessories get ["Honeywell"])
      3. Only leave empty if neither condition applies
    - Specified Model Families: Extract model/series names if user mentions them for this accessory. Also INHERIT from parent instrument if applicable (e.g., parent has "3051" → accessories get ["3051"]). Empty array if none.
    - User Specified Standards: Standards mentioned by user for THIS accessory (or inherited from parent instrument). Empty array if none.
    - Skip Standards: true if user says no standards needed for this accessory, false otherwise
    - Parent Instrument Category (link to the parent instrument this accessory supports)
    - Sample Input(must also include every specification field listed.)


Return ONLY a valid JSON object with this structure:
{{
  "project_name": "<unique project name describing the system>",
  "instruments": [
    {{
      "category": "<category>",
      "product_name": "<product name>",
      "quantity": "quantity",
      "specifications": {{
        "<spec_field>": "<spec_value>",
        "<spec_field>": "<spec_value>",
        "<spec_field>": "<spec_value>"
      }},
      "strategy": "<procurement strategy from user requirements or empty string>",
      "specified_vendors": ["<vendor1>", "<vendor2>"],
      "specified_model_families": ["<model_family1>", "<model_family2>"],
      "user_specified_standards": ["<standard mentioned by user for THIS instrument, e.g., ISA-S20, IEC 61511, ATEX>"],
      "skip_standards": false,
      "sample_input": "<category> with <key specifications>"
    }}
  ],
    "accessories": [
        {{
            "category": "<accessory category>",
            "accessory_name": "<accessory name>",
            "quantity": "<quantity>",
            "specifications": {{
                "<spec_field>": "<spec_value>"
            }},
            "strategy": "<procurement strategy from user requirements or empty string>",
            "specified_vendors": ["<inherited from parent instrument or empty>"],
            "specified_model_families": ["<inherited from parent instrument or empty>"],
            "user_specified_standards": ["<standards inherited from parent instrument or explicitly mentioned for this accessory>"],
            "skip_standards": false,
            "parent_instrument_category": "<category of parent instrument this accessory supports>",
            "sample_input": " <accessory category> for <instrument or purpose> with <key specs>"
        }}
    ],
  "summary": "Brief summary of identified instruments"
}}
 
Respond ONLY with valid JSON, no additional text.
Validate the outputs and adherence to the output structure.
"""
        session_isolated_requirements = f"[Session: {search_session_id}] - This is an independent instrument identification request. Requirements: {requirements}"
        
        full_prompt = ChatPromptTemplate.from_template(instrument_prompt)
        response_chain = full_prompt | components['llm_pro'] | StrOutputParser()
        llm_response = response_chain.invoke({"requirements": session_isolated_requirements})

        # Clean the LLM response
        cleaned_response = llm_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]  
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]  
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]  
        cleaned_response = cleaned_response.strip()

        try:
            result = json.loads(cleaned_response)
            
            # Validate the response structure
            if "instruments" not in result or not isinstance(result["instruments"], list):
                raise ValueError("Invalid response structure from LLM")
            
            # Ensure all instruments have required fields
            for instrument in result["instruments"]:
                if not all(key in instrument for key in ["category", "product_name", "specifications", "sample_input"]):
                    raise ValueError("Missing required fields in instrument data")
            
            # Validate accessories if present
            if "accessories" in result:
                if not isinstance(result["accessories"], list):
                    raise ValueError("'accessories' must be a list if provided")
                for accessory in result["accessories"]:
                    expected_acc_keys = ["category", "accessory_name", "specifications", "sample_input"]
                    if not all(key in accessory for key in expected_acc_keys):
                        raise ValueError("Missing required fields in accessory data")
            
            # Ensure strategy field exists for all instruments and accessories
            for instrument in result.get("instruments", []):
                if "strategy" not in instrument:
                    instrument["strategy"] = ""
                # Ensure specified_vendors is always an array
                if "specified_vendors" not in instrument:
                    instrument["specified_vendors"] = []
                elif not isinstance(instrument["specified_vendors"], list):
                    instrument["specified_vendors"] = []
                # Ensure specified_model_families is always an array
                if "specified_model_families" not in instrument:
                    instrument["specified_model_families"] = []
                elif not isinstance(instrument["specified_model_families"], list):
                    instrument["specified_model_families"] = []

            for accessory in result.get("accessories", []):
                if "strategy" not in accessory:
                    accessory["strategy"] = ""
                # Ensure specified_vendors is always an array
                if "specified_vendors" not in accessory:
                    accessory["specified_vendors"] = []
                elif not isinstance(accessory["specified_vendors"], list):
                    accessory["specified_vendors"] = []
                # Ensure specified_model_families is always an array
                if "specified_model_families" not in accessory:
                    accessory["specified_model_families"] = []
                elif not isinstance(accessory["specified_model_families"], list):
                    accessory["specified_model_families"] = []
                # Ensure parent_instrument_category is present
                if "parent_instrument_category" not in accessory:
                    accessory["parent_instrument_category"] = ""
            
            # =========================================================================
            # === STANDARDS RAG INTEGRATION ===
            # Priority per instrument/accessory: 
            # 1) User-specified standards for this item
            # 2) Skip if user requested for this item
            # 3) Fetch from document
            # =========================================================================
            user_id = session.get('user_id')
            
            # Check if user has standards documents for fallback
            conn = get_mongodb_connection()
            standards_collection = conn['collections']['standards']
            has_standards_doc = standards_collection.find_one({'user_id': user_id, 'full_text': {'$exists': True}})
            
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def process_instrument_standards(instrument):
                """Process standards for a single instrument based on its own flags"""
                category = instrument.get('category', '')
                user_specified = instrument.get('user_specified_standards', [])
                skip_standards = instrument.get('skip_standards', False)
                
                # Ensure fields exist
                if 'user_specified_standards' not in instrument:
                    instrument['user_specified_standards'] = []
                if 'skip_standards' not in instrument:
                    instrument['skip_standards'] = False
                
                # Priority 1: Skip if user requested
                if skip_standards:
                    instrument['applicable_standards'] = []
                    instrument['standards_source'] = 'skipped_by_user'
                    instrument['standards_summary'] = 'User requested no standards for this instrument'
                    logging.info(f"[STANDARDS_RAG] Skipping standards for {category} (user requested)")
                    return instrument
                
                # Priority 2: Use user-specified standards
                if user_specified and len(user_specified) > 0:
                    instrument['applicable_standards'] = user_specified
                    instrument['standards_source'] = 'user_specified'
                    instrument['standards_summary'] = f"User specified: {', '.join(user_specified)}"
                    logging.info(f"[STANDARDS_RAG] Using user-specified standards for {category}: {user_specified}")
                    return instrument
                
                # Priority 3: Fetch from document
                if has_standards_doc and category:
                    try:
                        standards_result = get_standards_for_category(user_id, category)
                        if standards_result and standards_result.get('found'):
                            instrument['standards_specs'] = standards_result.get('specifications', {})
                            instrument['applicable_standards'] = standards_result.get('applicable_standards', [])
                            instrument['standards_summary'] = standards_result.get('requirements_summary', '')
                            instrument['standards_source'] = 'document'
                            
                            # Update specifications with standards annotations
                            original_specs = instrument.get('specifications', {})
                            standards_specs = standards_result.get('specifications', {})
                            
                            enhanced_specs = {}
                            for key, value in original_specs.items():
                                matching_standard = None
                                for std_key, std_value in standards_specs.items():
                                    if key.lower() in std_key.lower() or std_key.lower() in key.lower():
                                        matching_standard = std_value
                                        break
                                if matching_standard:
                                    enhanced_specs[key] = f"{value} [STANDARD: {matching_standard}]"
                                else:
                                    enhanced_specs[key] = value
                            
                            # Add standards specs not in original
                            for std_key, std_value in standards_specs.items():
                                key_found = any(std_key.lower() in k.lower() or k.lower() in std_key.lower() 
                                               for k in original_specs.keys())
                                if not key_found:
                                    enhanced_specs[std_key] = f"{std_value} [STANDARD]"
                            
                            instrument['specifications'] = enhanced_specs
                            
                            # Update sample_input
                            original_sample = instrument.get('sample_input', '')
                            additions = [f"{k}: {v}" for k, v in list(standards_specs.items())[:5]]
                            if additions:
                                instrument['sample_input'] = f"{original_sample}. Standards: {', '.join(additions)}"
                            
                            logging.info(f"[STANDARDS_RAG] Applied document standards to {category}")
                        else:
                            instrument['standards_source'] = 'none'
                    except Exception as e:
                        logging.warning(f"[STANDARDS_RAG] Failed to fetch standards for {category}: {e}")
                        instrument['standards_source'] = 'error'
                else:
                    instrument['standards_source'] = 'none'
                
                return instrument
            
            def process_accessory_standards(accessory):
                """Process standards for a single accessory based on its own flags"""
                category = accessory.get('category', '')
                user_specified = accessory.get('user_specified_standards', [])
                skip_standards = accessory.get('skip_standards', False)
                
                # Ensure fields exist
                if 'user_specified_standards' not in accessory:
                    accessory['user_specified_standards'] = []
                if 'skip_standards' not in accessory:
                    accessory['skip_standards'] = False
                
                # Priority 1: Skip if user requested
                if skip_standards:
                    accessory['applicable_standards'] = []
                    accessory['standards_source'] = 'skipped_by_user'
                    accessory['standards_summary'] = 'User requested no standards for this accessory'
                    logging.info(f"[STANDARDS_RAG] Skipping standards for accessory {category} (user requested)")
                    return accessory
                
                # Priority 2: Use user-specified standards
                if user_specified and len(user_specified) > 0:
                    accessory['applicable_standards'] = user_specified
                    accessory['standards_source'] = 'user_specified'
                    accessory['standards_summary'] = f"User specified: {', '.join(user_specified)}"
                    logging.info(f"[STANDARDS_RAG] Using user-specified standards for accessory {category}: {user_specified}")
                    return accessory
                
                # Priority 3: Fetch from document
                if has_standards_doc and category:
                    try:
                        standards_result = get_standards_for_category(user_id, category)
                        if standards_result and standards_result.get('found'):
                            accessory['standards_specs'] = standards_result.get('specifications', {})
                            accessory['applicable_standards'] = standards_result.get('applicable_standards', [])
                            accessory['standards_summary'] = standards_result.get('requirements_summary', '')
                            accessory['standards_source'] = 'document'
                            
                            # Update specifications with standards annotations
                            original_specs = accessory.get('specifications', {})
                            standards_specs = standards_result.get('specifications', {})
                            
                            enhanced_specs = {}
                            for key, value in original_specs.items():
                                matching_standard = None
                                for std_key, std_value in standards_specs.items():
                                    if key.lower() in std_key.lower() or std_key.lower() in key.lower():
                                        matching_standard = std_value
                                        break
                                if matching_standard:
                                    enhanced_specs[key] = f"{value} [STANDARD: {matching_standard}]"
                                else:
                                    enhanced_specs[key] = value
                            
                            # Add standards specs not in original
                            for std_key, std_value in standards_specs.items():
                                key_found = any(std_key.lower() in k.lower() or k.lower() in std_key.lower() 
                                               for k in original_specs.keys())
                                if not key_found:
                                    enhanced_specs[std_key] = f"{std_value} [STANDARD]"
                            
                            accessory['specifications'] = enhanced_specs
                            
                            # Update sample_input
                            original_sample = accessory.get('sample_input', '')
                            additions = [f"{k}: {v}" for k, v in list(standards_specs.items())[:5]]
                            if additions:
                                accessory['sample_input'] = f"{original_sample}. Standards: {', '.join(additions)}"
                            
                            logging.info(f"[STANDARDS_RAG] Applied document standards to accessory {category}")
                        else:
                            accessory['standards_source'] = 'none'
                    except Exception as e:
                        logging.warning(f"[STANDARDS_RAG] Failed to fetch standards for accessory {category}: {e}")
                        accessory['standards_source'] = 'error'
                else:
                    accessory['standards_source'] = 'none'
                
                return accessory
            
            # Process all instruments in parallel
            logging.info(f"[STANDARDS_RAG] Processing standards for {len(result.get('instruments', []))} instruments")
            with ThreadPoolExecutor(max_workers=5) as executor:
                instrument_futures = [executor.submit(process_instrument_standards, inst) 
                                     for inst in result.get("instruments", [])]
                result["instruments"] = [f.result() for f in instrument_futures]
            
            # Process all accessories in parallel
            logging.info(f"[STANDARDS_RAG] Processing standards for {len(result.get('accessories', []))} accessories")
            with ThreadPoolExecutor(max_workers=5) as executor:
                accessory_futures = [executor.submit(process_accessory_standards, acc) 
                                    for acc in result.get("accessories", [])]
                result["accessories"] = [f.result() for f in accessory_futures]
            
            # Set overall standards status
            has_any_standards = any(
                inst.get('applicable_standards') for inst in result.get('instruments', [])
            ) or any(
                acc.get('applicable_standards') for acc in result.get('accessories', [])
            )
            result["standards_applied"] = has_any_standards
            logging.info(f"[STANDARDS_RAG] Processing complete - standards_applied: {has_any_standards}")
            
            # Add response type
            result["response_type"] = "requirements"
            
            # Generate summary message using LLM
            instrument_count = len(result.get("instruments", []))
            accessory_count = len(result.get("accessories", []))
            total_count = instrument_count + accessory_count
            
            logging.info(f"[IDENTIFY_INSTRUMENTS] Identified - instruments: {instrument_count}, accessories: {accessory_count}, total: {total_count}")
            
            # NOTE: Single item routing is handled by route classifier
            # If user has a single item, route classifier sends to Search page
            # identify_instruments only receives requests for Project page (multiple items or explicit project workflow)
            
            # Generate summary message for all items
            instrument_names = [inst.get("product_name", inst.get("category", "")) for inst in result.get("instruments", [])]
            accessory_names = [acc.get("accessory_name", acc.get("category", "")) for acc in result.get("accessories", [])]
            
            summary_prompt = """
You are Engenie - a friendly industrial automation assistant.

You have just analyzed the user's requirements and identified the following:
- Number of instruments: {instrument_count}
- Instrument types: {instrument_names}
- Number of accessories: {accessory_count}
- Accessory types: {accessory_names}

Generate a brief, friendly summary message (2-3 sentences) that:
1. Tells the user how many instruments and accessories you've identified
2. Mentions they can view the detailed list in the right panel
3. Invites them to click on any item to run a detailed vendor analysis

Keep it natural, professional, and helpful. Use markdown for emphasis (e.g., **bold** for counts).
Respond ONLY with the message text, no JSON or additional formatting.
"""
            try:
                summary_chain = ChatPromptTemplate.from_template(summary_prompt) | components['llm'] | StrOutputParser()
                llm_summary = summary_chain.invoke({
                    "instrument_count": instrument_count,
                    "instrument_names": ", ".join(instrument_names[:5]) if instrument_names else "None",
                    "accessory_count": accessory_count,
                    "accessory_names": ", ".join(accessory_names[:5]) if accessory_names else "None"
                })
                result["message"] = llm_summary.strip()
            except Exception as summary_error:
                logging.warning(f"Failed to generate LLM summary, using fallback: {summary_error}")
                instrument_word = "instrument" if instrument_count == 1 else "instruments"
                accessory_word = "accessory" if accessory_count == 1 else "accessories"
                result["message"] = f"I've identified **{instrument_count} {instrument_word}** and **{accessory_count} {accessory_word}** based on your requirements. You can view the detailed list in the right panel. Click on any item to run a detailed analysis."
            
            return standardized_jsonify(result, 200)
            
        except json.JSONDecodeError as e:
            logging.error(f"[IDENTIFY_INSTRUMENTS] Failed to parse LLM response as JSON: {e}")
            logging.error(f"[IDENTIFY_INSTRUMENTS] LLM Response: {llm_response}")
            
            return jsonify({
                "response_type": "error",
                "error": "Failed to parse instrument identification",
                "instruments": [],
                "accessories": [],
                "summary": "Unable to identify instruments from the provided requirements"
            }), 500

    except Exception as e:
        logging.exception("[IDENTIFY_INSTRUMENTS] Instrument identification failed.")
        return jsonify({
            "response_type": "error",
            "error": "Failed to process request: " + str(e),
            "instruments": [],
            "accessories": [],
            "summary": ""
        }), 500

# =========================================================================
# === INSTRUMENT MODIFICATION ENDPOINT ===
# =========================================================================

@app.route("/api/modify-instruments", methods=["POST"])
@login_required
def modify_instruments():
    """
    Allows users to modify/refine the identified instruments and accessories list.
    Users can add new items, remove existing items, or modify specifications.
    
    REQUEST:
    --------
    POST /api/modify-instruments
    Content-Type: application/json
    
    {
        "modification_request": "<user's modification request in natural language>",
        "current_instruments": [<current list of identified instruments>],
        "current_accessories": [<current list of identified accessories>],
        "search_session_id": "<optional session ID>"
    }
    
    MODIFICATION EXAMPLES:
    ----------------------
    - "Add a control valve with 2-inch size"
    - "Remove the temperature transmitter"
    - "Change the pressure range to 0-200 psi for the first pressure transmitter"
    - "Add 2 more flow meters with magnetic sensing"
    - "Remove all impulse lines"
    - "Update the material to Hastelloy for all transmitters"
    
    RESPONSE:
    ---------
    {
        "responseType": "modification",
        "message": "<friendly summary of changes made>",
        "instruments": [<updated list of instruments>],
        "accessories": [<updated list of accessories>],
        "changes_made": [<list of changes applied>]
    }
    """
    if not components or not components.get('llm_flash'):
        return jsonify({"error": "LLM component is not ready."}), 503

    try:
        data = request.get_json(force=True)
        modification_request = data.get("modification_request", "").strip()
        current_instruments = data.get("current_instruments", [])
        current_accessories = data.get("current_accessories", [])
        search_session_id = data.get("search_session_id", "default")
        
        if not modification_request:
            return jsonify({"error": "Modification request is required"}), 400

        if not current_instruments and not current_accessories:
            return jsonify({
                "error": "No instruments or accessories to modify. Please identify instruments first.",
                "response_type": "error"
            }), 400

        # Create a JSON representation of current items for the LLM
        current_state = {
            "instruments": current_instruments,
            "accessories": current_accessories
        }
        current_state_json = json.dumps(current_state, indent=2)

        modification_prompt = """
You are Engenie - an expert assistant in Industrial Process Control Systems.
Your task is to modify the current list of identified instruments and accessories based on the user's request.

CURRENT INSTRUMENTS AND ACCESSORIES:
{current_state}

USER'S MODIFICATION REQUEST:
"{modification_request}"

INSTRUCTIONS:
1. Carefully analyze the user's request to understand what changes they want:
   - ADD: Add new instruments or accessories to the list
   - REMOVE: Remove specific items from the list
   - MODIFY: Change specifications of existing items
   - UPDATE: Update quantities, materials, or other properties

2. Apply the requested changes to the current list.

3. For new items being added, include:
   - category: Product category (e.g., "Pressure Transmitter", "Control Valve")
   - product_name (for instruments) or accessory_name (for accessories)
   - quantity: Number of items
   - specifications: Detailed specs based on user request or industry standards
   - strategy: Procurement strategy (leave empty if none specified)
   - sample_input: A description combining category and key specifications

4. Mark any inferred specifications with [INFERRED] tag.

5. Provide a clear summary of changes made.

Return ONLY a valid JSON object with this structure:
{{
    "instruments": [
        {{
            "category": "<category>",
            "product_name": "<product name>",
            "quantity": "<quantity>",
            "specifications": {{
                "<spec_field>": "<spec_value>"
            }},
            "strategy": "<strategy or empty>",
            "sample_input": "<category> with <key specifications>"
        }}
    ],
    "accessories": [
        {{
            "category": "<accessory category>",
            "accessory_name": "<accessory name>",
            "quantity": "<quantity>",
            "specifications": {{
                "<spec_field>": "<spec_value>"
            }},
            "strategy": "<strategy or empty>",
            "sample_input": "<accessory category> for <purpose> with <key specs>"
        }}
    ],
    "changes_made": [
        "<description of change 1>",
        "<description of change 2>"
    ],
    "summary": "<brief friendly summary of all modifications>"
}}

Respond ONLY with valid JSON, no additional text.
"""
        
        # Invoke LLM
        full_prompt = ChatPromptTemplate.from_template(modification_prompt)
        response_chain = full_prompt | components['llm_pro'] | StrOutputParser()
        llm_response = response_chain.invoke({
            "current_state": current_state_json,
            "modification_request": modification_request
        })

        # Clean the LLM response
        cleaned_response = llm_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        try:
            result = json.loads(cleaned_response)
            
            # Validate response structure
            if "instruments" not in result:
                result["instruments"] = current_instruments
            if "accessories" not in result:
                result["accessories"] = current_accessories
            if "changes_made" not in result:
                result["changes_made"] = []
            if "summary" not in result:
                result["summary"] = "Modifications applied successfully."

            # Ensure all required fields exist in instruments
            for instrument in result.get("instruments", []):
                if "strategy" not in instrument:
                    instrument["strategy"] = ""
                if "quantity" not in instrument:
                    instrument["quantity"] = "1"

            # Ensure all required fields exist in accessories
            for accessory in result.get("accessories", []):
                if "strategy" not in accessory:
                    accessory["strategy"] = ""
                if "quantity" not in accessory:
                    accessory["quantity"] = "1"

            # =========================================================================
            # === STANDARDS RAG INTEGRATION FOR MODIFIED ITEMS ===
            # Apply standards to any newly added or modified instruments/accessories
            # =========================================================================
            user_id = session.get('user_id')
            
            # Check if user has standards documents
            conn = get_mongodb_connection()
            standards_collection = conn['collections']['standards']
            has_standards = standards_collection.find_one({'user_id': user_id, 'full_text': {'$exists': True}})
            
            if has_standards:
                logging.info(f"[MODIFY_INSTRUMENTS] Applying standards RAG to modified items")
                
                from concurrent.futures import ThreadPoolExecutor, as_completed
                
                def apply_standards_to_item(item, is_accessory=False):
                    """Apply standards to an instrument or accessory if not already applied"""
                    category = item.get('category', '')
                    if not category:
                        return item
                    
                    # Skip if standards already applied (has standards_specs)
                    if item.get('standards_specs') and len(item.get('standards_specs', {})) > 0:
                        return item
                    
                    try:
                        standards_result = get_standards_for_category(user_id, category)
                        
                        if standards_result and standards_result.get('found'):
                            # Add standards specifications
                            item['standards_specs'] = standards_result.get('specifications', {})
                            item['applicable_standards'] = standards_result.get('applicable_standards', [])
                            item['standards_summary'] = standards_result.get('requirements_summary', '')
                            
                            # Enhance specifications with standards annotations
                            original_specs = item.get('specifications', {})
                            standards_specs = standards_result.get('specifications', {})
                            
                            enhanced_specs = {}
                            for key, value in original_specs.items():
                                # Check if there's a matching standard spec
                                matching_standard = None
                                for std_key, std_value in standards_specs.items():
                                    if key.lower() in std_key.lower() or std_key.lower() in key.lower():
                                        matching_standard = std_value
                                        break
                                
                                if matching_standard:
                                    enhanced_specs[key] = f"{value} [STANDARD: {matching_standard}]"
                                else:
                                    enhanced_specs[key] = value
                            
                            # Add standards specs not in original
                            for std_key, std_value in standards_specs.items():
                                key_found = False
                                for orig_key in original_specs.keys():
                                    if std_key.lower() in orig_key.lower() or orig_key.lower() in std_key.lower():
                                        key_found = True
                                        break
                                if not key_found:
                                    enhanced_specs[std_key] = f"{std_value} [STANDARD]"
                            
                            item['specifications'] = enhanced_specs
                            
                            # Update sample_input to include standards
                            original_sample_input = item.get('sample_input', '')
                            standards_additions = [f"{k}: {v}" for k, v in list(standards_specs.items())[:5]]
                            if standards_additions:
                                standards_text = ", ".join(standards_additions)
                                item['sample_input'] = f"{original_sample_input}. Standards: {standards_text}"
                            
                            item_type = "accessory" if is_accessory else "instrument"
                            logging.info(f"[STANDARDS_RAG] Applied standards to {item_type} {category}: {len(standards_specs)} specs")
                        
                        return item
                    except Exception as e:
                        item_type = "accessory" if is_accessory else "instrument"
                        logging.warning(f"[STANDARDS_RAG] Failed to apply standards to {item_type} {category}: {e}")
                        return item
                
                # Apply standards to all instruments and accessories in parallel
                with ThreadPoolExecutor(max_workers=5) as executor:
                    # Process instruments
                    inst_futures = {executor.submit(apply_standards_to_item, inst, False): i 
                                   for i, inst in enumerate(result.get("instruments", []))}
                    updated_instruments = [None] * len(result.get("instruments", []))
                    for future in as_completed(inst_futures):
                        idx = inst_futures[future]
                        updated_instruments[idx] = future.result()
                    result["instruments"] = updated_instruments
                    
                    # Process accessories
                    acc_futures = {executor.submit(apply_standards_to_item, acc, True): i 
                                  for i, acc in enumerate(result.get("accessories", []))}
                    updated_accessories = [None] * len(result.get("accessories", []))
                    for future in as_completed(acc_futures):
                        idx = acc_futures[future]
                        updated_accessories[idx] = future.result()
                    result["accessories"] = updated_accessories
                
                result["standards_applied"] = True
                logging.info(f"[MODIFY_INSTRUMENTS] Standards RAG complete")
            else:
                result["standards_applied"] = False


            # Generate a friendly message using LLM
            changes_count = len(result.get("changes_made", []))
            instrument_count = len(result.get("instruments", []))
            accessory_count = len(result.get("accessories", []))
            
            message_prompt = """
You are Engenie - a friendly industrial automation assistant.

You have just modified the user's instrument list based on their request.

Changes made: {changes_list}
Total instruments now: {instrument_count}
Total accessories now: {accessory_count}

Generate a brief, friendly confirmation message (2-3 sentences) that:
1. Confirms the changes were successful
2. Summarizes what was changed
3. Mentions the updated counts

Use markdown formatting for emphasis (e.g., **bold** for counts).
Respond ONLY with the message text, no JSON.
"""
            try:
                message_chain = ChatPromptTemplate.from_template(message_prompt) | components['llm'] | StrOutputParser()
                friendly_message = message_chain.invoke({
                    "changes_list": ", ".join(result.get("changes_made", ["No specific changes noted"])),
                    "instrument_count": instrument_count,
                    "accessory_count": accessory_count
                })
                result["message"] = friendly_message.strip()
            except Exception as msg_error:
                logging.warning(f"Failed to generate friendly modification message: {msg_error}")
                result["message"] = f"I've updated your list! You now have **{instrument_count} instruments** and **{accessory_count} accessories**."

            result["response_type"] = "modification"
            
            logging.info(f"[MODIFY] Successfully modified instruments. Changes: {result.get('changes_made', [])}")
            
            return standardized_jsonify(result, 200)

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse modification LLM response as JSON: {e}")
            logging.error(f"LLM Response: {llm_response}")
            
            return jsonify({
                "response_type": "error",
                "error": "Failed to process modification request",
                "instruments": current_instruments,
                "accessories": current_accessories,
                "message": "I couldn't process your modification request. Please try rephrasing it."
            }), 500

    except Exception as e:
        logging.exception("Instrument modification failed.")
        return jsonify({
            "response_type": "error",
            "error": "Failed to modify instruments: " + str(e),
            "instruments": [],
            "accessories": []
        }), 500

@app.route("/api/search-vendors", methods=["POST"])
@login_required
def search_vendors():
    """
    Search for vendors based on selected instrument/accessory details.
    Runs LLM matching in BACKGROUND and returns immediately.
    Results are stored in session for use by analysis chain.
    """
    try:
        data = request.get_json(force=True)
        category = data.get("category", "").strip()
        product_name = data.get("product_name", "").strip() or data.get("accessory_name", "").strip()
        strategy = data.get("strategy", "").strip()
        
        # NEW: Handle user-specified vendors from instrument identification
        specified_vendors = data.get("specified_vendors", [])
        if specified_vendors and isinstance(specified_vendors, list) and len(specified_vendors) > 0:
            # Store in session for analysis chain to use with priority logic
            session['specified_vendors'] = specified_vendors
            logging.info(f"[VENDOR_SEARCH] User-specified vendors stored in session: {specified_vendors}")
        
        # NEW: Handle user-specified model families from instrument identification
        specified_model_families = data.get("specified_model_families", [])
        if specified_model_families and isinstance(specified_model_families, list) and len(specified_model_families) > 0:
            # Store in session for analysis chain to use
            session['specified_model_families'] = specified_model_families
            logging.info(f"[VENDOR_SEARCH] User-specified model families stored in session: {specified_model_families}")
        
        # NEW: Handle applicable engineering standards from Standards RAG
        applicable_standards = data.get("applicable_standards", [])
        if applicable_standards and isinstance(applicable_standards, list) and len(applicable_standards) > 0:
            # Store in session for analysis chain to use
            session['applicable_standards'] = applicable_standards
            logging.info(f"[VENDOR_SEARCH] Applicable standards stored in session: {applicable_standards}")
        
        # NEW: Handle standards specifications from user's standards document
        standards_specs = data.get("standards_specs", {})
        if standards_specs and isinstance(standards_specs, dict) and len(standards_specs) > 0:
            # Store in session for analysis chain to use
            session['standards_specs'] = standards_specs
            logging.info(f"[VENDOR_SEARCH] Standards specifications stored in session: {len(standards_specs)} specs")
        
        print(f"[VENDOR_SEARCH] Received request: category='{category}', product='{product_name}', strategy='{strategy}', vendors={specified_vendors}, model_families={specified_model_families}, applicable_standards={applicable_standards}")
        
        if not category or not product_name:
            return jsonify({"error": "Category and product_name/accessory_name are required"}), 400
        
        user_id = session.get('user_id')
        if not user_id:
             return jsonify({"error": "User not authenticated"}), 401
        
        # Start background task for LLM matching
        def run_vendor_search_background(user_id, category, product_name, strategy):
            """Background worker for LLM-based vendor matching"""
            try:
                print(f"[VENDOR_SEARCH_BG] Starting background search for user {user_id}")
                
                conn = get_mongodb_connection()
                stratergy_collection = conn['collections']['stratergy']
                
                cursor = stratergy_collection.find({'user_id': user_id})
                
                vendors = []
                for doc in cursor:
                    if 'data' in doc and isinstance(doc['data'], list):
                        for item in doc['data']:
                            vendors.append({
                                "vendor name": item.get("vendor_name", ""),
                                "category": item.get("category", ""),
                                "subcategory": item.get("subcategory", ""),
                                "strategy": item.get("stratergy", ""),
                            })
                
                if not vendors:
                    print(f"[VENDOR_SEARCH_BG] No strategy data found for user {user_id}")
                    with vendor_search_lock:
                        vendor_search_tasks[user_id] = {
                            'status': 'completed',
                            'vendors': [],
                            'csv_vendor_filter': None,
                            'completed_at': datetime.now().isoformat()
                        }
                    return
                
                # Get unique categories and subcategories
                csv_categories = list(set([v.get('category', '').strip() for v in vendors if v.get('category')]))
                csv_subcategories = list(set([v.get('subcategory', '').strip() for v in vendors if v.get('subcategory')]))
                csv_strategies = list(set([v.get('strategy', '').strip() for v in vendors if v.get('strategy')]))
                
                # LLM matching
                from langchain_google_genai import ChatGoogleGenerativeAI
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_core.output_parsers import StrOutputParser
                import concurrent.futures
                
                llm_flash_lite = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite", 
                    temperature=0.1,
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                )
                
                match_prompt = ChatPromptTemplate.from_template("""
You are an expert at matching product categories and names in industrial automation systems.

Match the INPUT value to the BEST matching option from the available list.

INPUT: "{input_value}"
INPUT TYPE: {input_type}

AVAILABLE OPTIONS:
{options}

Instructions:
1. Find the BEST matching option from the available list
2. Consider these as MATCHING:
   - Same meaning with different wording (e.g., "Pressure Transmitter" vs "Pressure Sensors")
   - Abbreviations (e.g., "PT" vs "Pressure Transmitter")
   - Partial matches (e.g., "Flow" matching "Flow Meter")
   - Industry-standard equivalents (e.g., "DP Transmitter" vs "Differential Pressure Transmitter")
3. Only match if they are clearly referring to the SAME category/product/strategy

Respond with ONLY a JSON object in this format:
{{"matched": "<exact text from available options or null if no match>", "confidence": "<high/medium/low/none>"}}

If no good match exists, respond with: {{"matched": null, "confidence": "none"}}
""")
                
                chain = match_prompt | llm_flash_lite | StrOutputParser()
                
                def llm_match(input_value, input_type, options_list):
                    if not input_value or not options_list:
                        return None, "none"
                    try:
                        options_str = "\n".join([f"- {opt}" for opt in options_list])
                        result = chain.invoke({
                            "input_value": input_value,
                            "input_type": input_type,
                            "options": options_str
                        })
                        result = result.strip()
                        if result.startswith("```json"):
                            result = result[7:]
                        elif result.startswith("```"):
                            result = result[3:]
                        if result.endswith("```"):
                            result = result[:-3]
                        result = result.strip()
                        
                        parsed = json.loads(result)
                        return parsed.get("matched"), parsed.get("confidence", "none")
                    except Exception as e:
                        logging.error(f"[VENDOR_SEARCH_BG] LLM error: {e}")
                        return None, "error"
                
                # Run matches in parallel
                matched_category = None
                matched_subcategory = None
                matched_strategy = None
                category_match_type = None
                subcategory_match_type = None
                strategy_match_type = None
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    future_category = executor.submit(llm_match, category, "category", csv_categories)
                    future_subcategory = executor.submit(llm_match, product_name, "subcategory/product name", csv_subcategories)
                    future_strategy = executor.submit(llm_match, strategy, "procurement strategy", csv_strategies) if strategy else None
                    
                    matched_category, cat_conf = future_category.result()
                    if matched_category and cat_conf in ["high", "medium"]:
                        category_match_type = f"llm_{cat_conf}"
                        print(f"[VENDOR_SEARCH_BG] Matched category: '{category}' -> '{matched_category}'")
                    
                    matched_subcategory, sub_conf = future_subcategory.result()
                    if matched_subcategory and sub_conf in ["high", "medium"]:
                        subcategory_match_type = f"llm_{sub_conf}"
                        print(f"[VENDOR_SEARCH_BG] Matched subcategory: '{product_name}' -> '{matched_subcategory}'")
                    
                    if future_strategy:
                        matched_strategy, strat_conf = future_strategy.result()
                        if matched_strategy and strat_conf in ["high", "medium"]:
                            strategy_match_type = f"llm_{strat_conf}"
                
                # Filter vendors
                filtered_vendors = []
                for vendor in vendors:
                    vendor_category = vendor.get('category', '').strip()
                    vendor_subcategory = vendor.get('subcategory', '').strip()
                    vendor_strategy = vendor.get('strategy', '').strip()
                    
                    if vendor_category != matched_category:
                        continue
                    if matched_subcategory and vendor_subcategory != matched_subcategory:
                        continue
                    if strategy and matched_strategy and vendor_strategy != matched_strategy:
                        continue
                    
                    filtered_vendors.append({
                        "vendor_name": vendor.get('vendor name', ''),
                        "category": vendor_category,
                        "subcategory": vendor_subcategory,
                        "strategy": vendor_strategy,
                    })
                
                matching_criteria = {
                    "category_match": {"input": category, "matched": matched_category, "match_type": category_match_type},
                    "subcategory_match": {"input": product_name, "matched": matched_subcategory, "match_type": subcategory_match_type} if matched_subcategory else None,
                    "strategy_match": {"input": strategy, "matched": matched_strategy, "match_type": strategy_match_type} if strategy and matched_strategy else None
                }
                
                vendor_names_only = [v.get('vendor_name', '').strip() for v in filtered_vendors]
                
                # Store results for later retrieval by analysis chain
                with vendor_search_lock:
                    vendor_search_tasks[user_id] = {
                        'status': 'completed',
                        'vendors': vendor_names_only,
                        'total_count': len(filtered_vendors),
                        'matching_criteria': matching_criteria,
                        'csv_vendor_filter': {
                            'vendor_names': vendor_names_only,
                            'csv_data': filtered_vendors,
                            'product_type': category,
                            'detected_product': product_name,
                            'matching_criteria': matching_criteria
                        } if filtered_vendors else None,
                        'completed_at': datetime.now().isoformat()
                    }
                
                print(f"[VENDOR_SEARCH_BG] Completed for user {user_id}: {len(filtered_vendors)} vendors found")
                
            except Exception as e:
                logging.exception(f"[VENDOR_SEARCH_BG] Error for user {user_id}")
                with vendor_search_lock:
                    vendor_search_tasks[user_id] = {
                        'status': 'error',
                        'error': str(e),
                        'completed_at': datetime.now().isoformat()
                    }
        
        # Start background thread (fire and forget)
        thread = threading.Thread(
            target=run_vendor_search_background,
            args=(user_id, category, product_name, strategy),
            daemon=True
        )
        thread.start()
        
        # Return immediately - don't wait for LLM matching
        return jsonify({
            "status": "processing",
            "message": "Vendor search started. Results will be used by analysis.",
            "category": category,
            "product_name": product_name
        }), 202
        
    except Exception as e:
        logging.exception("Vendor search failed.")
        return jsonify({
            "error": "Failed to start vendor search: " + str(e),
            "status": "error"
        }), 500


@app.route("/api/search-vendors/status", methods=["GET"])
@login_required
def search_vendors_status():
    """
    Check status of background vendor search.
    Returns results when LLM matching is complete.
    """
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "User not authenticated"}), 401
    
    with vendor_search_lock:
        task = vendor_search_tasks.get(user_id)
    
    if not task:
        return jsonify({
            "status": "not_found",
            "message": "No vendor search in progress"
        }), 404
    
    if task['status'] == 'processing':
        return jsonify({
            "status": "processing",
            "message": "LLM matching in progress..."
        }), 200
    
    if task['status'] == 'error':
        return jsonify({
            "status": "error",
            "error": task.get('error', 'Unknown error')
        }), 500
    
    # Completed - return results
    return jsonify({
        "status": "completed",
        "vendors": task.get('vendors', []),
        "total_count": task.get('total_count', 0),
        "matching_criteria": task.get('matching_criteria', {})
    }), 200

# =========================================================================
# === API ENDPOINTS ===
# =========================================================================
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GOOGLE_API_KEY1 = os.getenv("GOOGLE_API_KEY1")
GOOGLE_CSE_ID = "066b7345f94f64897"

# Image search API configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY1")  # Use GOOGLE_API_KEY1 for custom search
GOOGLE_CX = os.getenv("GOOGLE_CX", GOOGLE_CSE_ID)  # Custom search engine ID
SERPER_API_KEY_IMAGES = os.getenv("SERPER_API_KEY", SERPER_API_KEY)  # Reuse existing key

if not SERPER_API_KEY:
    logging.warning("SERPER_API_KEY environment variable not set! Will fallback to SERP API.")

if not SERPAPI_KEY:
    logging.warning("SERPAPI_KEY environment variable not set! Will use Google Custom Search as fallback.")

if not GOOGLE_API_KEY1:
    logging.warning("GOOGLE_API_KEY1 environment variable not set! Custom search fallback not available.")

if not GOOGLE_API_KEY:
    logging.warning("GOOGLE_API_KEY environment variable not set! Image search functionality limited.")

if not GOOGLE_CX:
    logging.warning("GOOGLE_CX environment variable not set! Custom image search not available.")


def serper_search_pdfs(query):
    """Perform a PDF search with Serper API."""
    if not SERPER_API_KEY:
        return []
    
    try:
        url = "https://google.serper.dev/search"
        payload = {
            "q": f"{query} filetype:pdf",
            "num": 10
        }
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("organic", [])
        
        pdf_results = []
        for item in results:
            pdf_results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet", ""),
                "source": "serper"
            })
        return pdf_results
    except Exception as e:
        print(f"[WARNING] Serper API search failed: {e}")
        return []


def serpapi_search_pdfs(query):
    """Perform a PDF search with SerpApi."""
    if not SERPAPI_KEY:
        return []
    
    try:
        search = GoogleSearch({
            "q": f"{query} filetype:pdf",
            "api_key": SERPAPI_KEY,
            "num": 10
        })
        results = search.get_dict()
        items = results.get("organic_results", [])
        pdf_results = []
        for item in items:
            pdf_results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet", ""),
                "source": "serpapi"
            })
        return pdf_results
    except Exception as e:
        print(f"[WARNING] SerpAPI search failed: {e}")
        return []


def google_custom_search_pdfs(query):
    """Perform a PDF search with Google Custom Search API as fallback."""
    if not GOOGLE_API_KEY1 or not GOOGLE_CSE_ID:
        return []
    
    try:
        import threading
        from googleapiclient.discovery import build
        
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY1)
        
        result_container = [None]
        exception_container = [None]
        
        def google_request():
            try:
                result = service.cse().list(
                    q=f"{query} filetype:pdf",
                    cx=GOOGLE_CSE_ID,
                    num=10,
                    fileType='pdf'
                ).execute()
                result_container[0] = result.get('items', [])
            except Exception as e:
                exception_container[0] = e
        
        thread = threading.Thread(target=google_request)
        thread.daemon = True
        thread.start()
        thread.join(timeout=60)
        
        if thread.is_alive() or exception_container[0]:
            return []
        
        items = result_container[0] or []
        
        pdf_results = []
        for item in items:
            title = item.get('title', '')
            link = item.get('link', '')
            snippet = item.get('snippet', '')
            
            if link:
                pdf_results.append({
                    "title": title.strip(),
                    "url": link,
                    "snippet": snippet,
                    "source": "google_custom"
                })
        
        return pdf_results
        
    except Exception as e:
        print(f"[WARNING] Google Custom Search failed: {e}")
        return []


def search_pdfs_with_fallback(query):
    """
    Search for PDFs using Serper API first, then SERP API, then Google Custom Search as final fallback.
    Returns combined results with source information.
    """
    # First, try Serper API
    serper_results = serper_search_pdfs(query)
    
    # If Serper API returns results, use them
    if serper_results:
        return serper_results
    
    # If Serper API fails or returns no results, try SERP API
    serpapi_results = serpapi_search_pdfs(query)
    
    # If SERP API returns results, use them
    if serpapi_results:
        return serpapi_results
    
    # If both Serper and SERP API fail or return no results, try Google Custom Search
    google_results = google_custom_search_pdfs(query)
    
    if google_results:
        return google_results
    
    # If all three fail, return empty list
    return []

@app.route("/api/search_pdfs", methods=["GET"])
@login_required
def search_pdfs():
    """
    Performs a PDF search using Serper API first, then SERP API, then Google Custom Search as final fallback.
    """
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Missing search query"}), 400

    try:
        # Use the new fallback search function
        results = search_pdfs_with_fallback(query)
        
        # Add metadata about which search engine was used
        response_data = {
            "results": results,
            "total_results": len(results),
            "query": query
        }
        
        # Add source information if results exist
        if results:
            sources_used = list(set(result.get("source", "unknown") for result in results))
            response_data["sources_used"] = sources_used
            
            # Add fallback indicator based on the new three-tier system
            if "serper" in sources_used:
                response_data["fallback_used"] = False
                response_data["message"] = "Results from Serper API"
            elif "serpapi" in sources_used:
                response_data["fallback_used"] = True
                response_data["message"] = "Results from SERP API (Serper fallback)"
            elif "google_custom" in sources_used:
                response_data["fallback_used"] = True
                response_data["message"] = "Results from Google Custom Search (final fallback)"
            else:
                response_data["fallback_used"] = False
        else:
            response_data["sources_used"] = []
            response_data["fallback_used"] = True
            response_data["message"] = "No results found from any search engine"
        
        return jsonify(response_data), 200

    except Exception as e:
        logging.exception("PDF search failed.")
        return jsonify({"error": "Failed to perform PDF search: " + str(e)}), 500


@app.route("/api/view_pdf", methods=["GET"])
@login_required
def view_pdf():
    """
    Fetches a PDF from a URL and serves it for viewing.
    """
    pdf_url = request.args.get("url")
    if not pdf_url:
        return jsonify({"error": "Missing PDF URL"}), 400
    
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        
        pdf_stream = BytesIO(response.content)
        
        return send_file(
            pdf_stream,
            mimetype='application/pdf',
            as_attachment=False,
            download_name=os.path.basename(pdf_url)
        )
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to fetch PDF from URL: " + str(e)}), 500
    except Exception as e:
        return jsonify({"error": "An error occurred while viewing the PDF: " + str(e)}), 500


def fetch_price_and_reviews_serpapi(product_name: str, vendor_name: str = None, model_family: str = None, sub_model: str = None, product_type: str = None, user_location: str = None):
    """Use SerpApi to fetch price and review info for a product."""
    if not SERPAPI_KEY:
        return []

    # Construct specific query
    query_parts = []
    if vendor_name: query_parts.append(vendor_name)
    if model_family: query_parts.append(model_family)
    if sub_model: query_parts.append(sub_model)
    elif product_name: query_parts.append(product_name) # Fallback to product_name if sub_model not explicit
    
    if product_type: query_parts.append(product_type)
    
    query_parts.append("price review")
    
    base_query = " ".join(query_parts)
    
    if user_location:
         base_query += f" in {user_location}"
        
    try:
        search = GoogleSearch({
            "q": base_query,
            "api_key": SERPAPI_KEY,
            "num": 10
        })
        results = search.get_dict()
        
        organic_results = results.get("organic_results", [])
        return [{
            "price": r.get("price") or (re.search(r"([$₹€£¥])\s?\d+(?:[.,]\d+)?", r.get("snippet", "")).group(0) if re.search(r"([$₹€£¥])\s?\d+(?:[.,]\d+)?", r.get("snippet", "")) else None),
            "reviews": r.get("reviews"),
            "source": r.get("source"),
            "link": r.get("link")
        } for r in organic_results if r.get("price") or "price" in r.get("title", "").lower() or (r.get("snippet") and re.search(r"([$₹€£¥])\s?\d+", r.get("snippet", "")))]
        
    except Exception as e:
        print(f"[WARNING] SerpAPI price/review search failed: {e}")
        return []

def fetch_price_and_reviews_serper(product_name: str, vendor_name: str = None, model_family: str = None, sub_model: str = None, product_type: str = None, user_location: str = None):
    """Use Serper API to fetch price and review info for a product."""
    if not SERPER_API_KEY:
        return []

    # Construct specific query
    query_parts = []
    if vendor_name: query_parts.append(vendor_name)
    if model_family: query_parts.append(model_family)
    if sub_model: query_parts.append(sub_model)
    elif product_name: query_parts.append(product_name)
    
    if product_type: query_parts.append(product_type)
    
    query_parts.append("price review")
    
    base_query = " ".join(query_parts)

    if user_location:
        base_query += f" in {user_location}"
        
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": base_query,
        "num": 10
    })
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        response_data = response.json()
        organic = response_data.get("organic", [])
        
        return [{
            "price": item.get("price") or (re.search(r"([$₹€£¥])\s?\d+(?:[.,]\d+)?", item.get("snippet", "")).group(0) if re.search(r"([$₹€£¥])\s?\d+(?:[.,]\d+)?", item.get("snippet", "")) else None),
            "reviews": item.get("rating"), 
            "source": item.get("title"), 
            "link": item.get("link")
        } for item in organic if "price" in item.get("snippet", "").lower() or item.get("price")]
        
    except Exception as e:
        print(f"[WARNING] Serper API price/review search failed: {e}")
        return []

def fetch_price_and_reviews_google_custom(product_name: str, vendor_name: str = None, model_family: str = None, sub_model: str = None, product_type: str = None, user_location: str = None):
    """Use Google Custom Search to fetch price and review info for a product as fallback."""
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        return []
        
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        
        # Construct specific query
        query_parts = []
        if vendor_name: query_parts.append(vendor_name)
        if model_family: query_parts.append(model_family)
        if sub_model: query_parts.append(sub_model)
        elif product_name: query_parts.append(product_name)
        
        if product_type: query_parts.append(product_type)
        
        query_parts.append("price review")
        
        base_query = " ".join(query_parts)
    
        if user_location:
            base_query += f" in {user_location}"
            
        res = service.cse().list(
            q=base_query,
            cx=GOOGLE_CX,
            num=5
        ).execute()

        items = res.get('items', [])
        return [{
            "price": None, 
            "reviews": None,
            "source": item.get("displayLink"),
            "link": item.get("link") 
        } for item in items]
        
    except Exception as e:
        print(f"[WARNING] Google Custom Search price/review search failed: {e}")
        return []

def fetch_price_and_reviews(product_name: str, vendor_name: str = None, model_family: str = None, sub_model: str = None, product_type: str = None, user_location: str = None):
    """
    Fetch price and review info using SERP API first, then Serper API, then Google Custom Search as final fallback.
    Returns a list of dicts: [ {price, reviews, source, link}, ... ]
    """
    
    # 1. Try SerpApi
    serpapi_results = fetch_price_and_reviews_serpapi(product_name, vendor_name, model_family, sub_model, product_type, user_location)
    if serpapi_results:
        return serpapi_results

    # 2. Try Serper
    serper_results = fetch_price_and_reviews_serper(product_name, vendor_name, model_family, sub_model, product_type, user_location)
    if serper_results:
        return serper_results

    # 3. Try Google Custom Search
    google_results = fetch_price_and_reviews_google_custom(product_name, vendor_name, model_family, sub_model, product_type, user_location)
    if google_results:
        return google_results
    
    # If all three fail, return empty list
    return []


# =========================================================================
# === IMAGE SEARCH FUNCTIONS ===
# =========================================================================

def get_manufacturer_domains_from_llm(vendor_name: str) -> list:
    """
    Use LLM to dynamically generate manufacturer domain names based on vendor name
    """
    if not components or not components.get('llm'):
        # Fallback to common domains if LLM is not available
        return [
            "emerson.com", "yokogawa.com", "siemens.com", "abb.com", "honeywell.com",
            "schneider-electric.com", "ge.com", "rockwellautomation.com", "endress.com",
            "fluke.com", "krohne.com", "rosemount.com", "fisher.com", "metso.com"
        ]
    
    try:
        prompt_template = """
Based on the vendor/manufacturer name "{vendor_name}", generate a list of possible official domain names for this company and related manufacturers in the industrial/instrumentation sector.

Instructions:
1. Include the most likely official domain for "{vendor_name}"
2. Include common domain variations (.com, .de, .co.uk, etc.)
3. Include subsidiary or parent company domains if applicable
4. Include related industrial automation/instrumentation companies
5. Maximum 10-15 domains
6. Return only domain names, one per line, no explanations

Vendor: {vendor_name}

Domains:
"""
        
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        full_prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = full_prompt | components['llm'] | StrOutputParser()
        
        response = chain.invoke({"vendor_name": vendor_name})
        
        # Parse the response to extract domain names
        domains = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and '.' in line:
                # Clean up the line - remove any prefixes, bullets, numbers
                domain = line.split()[-1] if ' ' in line else line
                domain = domain.replace('www.', '').replace('http://', '').replace('https://', '')
                domain = domain.strip('.,()[]{}')
                
                if '.' in domain and len(domain) > 3:
                    domains.append(domain)
        
        # Ensure we have at least some domains
        if not domains:
            # Fallback: generate based on vendor name
            vendor_clean = vendor_name.lower().replace(' ', '').replace('&', '').replace('+', '')
            domains = [f"{vendor_clean}.com", f"{vendor_clean}.de", f"{vendor_clean}group.com"]
        
        logging.info(f"LLM generated {len(domains)} domains for {vendor_name}: {domains[:5]}...")
        return domains[:15]  # Limit to 15 domains
        
    except Exception as e:
        logging.warning(f"Failed to generate domains via LLM for {vendor_name}: {e}")
        # Fallback: generate based on vendor name
        vendor_clean = vendor_name.lower().replace(' ', '').replace('&', '').replace('+', '')
        return [f"{vendor_clean}.com", f"{vendor_clean}.de", f"{vendor_clean}group.com"]

def fetch_images_google_cse_sync(vendor_name: str, product_name: str = None, manufacturer_domains: list = None, model_family: str = None, product_type: str = None):
    """
    Synchronous version: Google Custom Search API for images from manufacturer domains
    Uses progressive query fallback:
      Priority 1: vendor + model_family (most specific)
      Priority 2: vendor + model_family + product_type
      Priority 3: vendor + product_type (least specific)
    
    Args:
        vendor_name: Name of the vendor/manufacturer
        product_name: (Optional) Specific product name/model
        manufacturer_domains: (Optional) List of manufacturer domains to search within
        model_family: (Optional) Model family/series to include in search
        product_type: (Optional) Type of product to help refine search
    """
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        logging.warning("Google CSE credentials not available for image search")
        return []
    
    # Build manufacturer domains filter
    if manufacturer_domains is None:
        manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
    domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains])
    
    # Build progressive queries (from most specific to least specific)
    queries_to_try = []
    
    # Priority 1: vendor + model_family (most specific)
    if model_family:
        queries_to_try.append(f"{vendor_name} {model_family} product image")
    
    # Priority 2: vendor + model_family + product_type
    if model_family and product_type:
        queries_to_try.append(f"{vendor_name} {model_family} {product_type} product image")
    
    # Priority 3: vendor + product_type (fallback)
    if product_type:
        queries_to_try.append(f"{vendor_name} {product_type} product image")
    
    # Fallback: just vendor name
    if not queries_to_try:
        queries_to_try.append(f"{vendor_name} product image")
    
    unsupported_schemes = ['x-raw-image://', 'data:', 'blob:', 'chrome://', 'about:']
    
    for priority_idx, base_query in enumerate(queries_to_try, 1):
        try:
            search_query = f"{base_query} ({domain_filter}) filetype:jpg OR filetype:png"
            logging.info(f"Google CSE Priority {priority_idx} query: {search_query[:100]}...")
            
            service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
            result = service.cse().list(
                q=search_query,
                cx=GOOGLE_CX,
                searchType="image",
                num=8,
                safe="medium",
                imgSize="MEDIUM"
            ).execute()
            
            images = []
            for item in result.get("items", []):
                url = item.get("link")
                
                # Skip images with unsupported URL schemes
                if not url or any(url.startswith(scheme) for scheme in unsupported_schemes):
                    continue
                
                # Only include http/https URLs
                if not url.startswith(('http://', 'https://')):
                    continue
                    
                images.append({
                    "url": url,
                    "title": item.get("title", ""),
                    "source": "google_cse",
                    "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                    "domain": item.get("displayLink", "")
                })
            
            if images:
                logging.info(f"Google CSE found {len(images)} images at Priority {priority_idx} for {vendor_name}")
                return images
            else:
                logging.info(f"Google CSE Priority {priority_idx} returned no images, trying next priority...")
                
        except Exception as e:
            logging.warning(f"Google CSE Priority {priority_idx} failed for {vendor_name}: {e}")
            continue
    
    logging.warning(f"Google CSE all priorities exhausted for {vendor_name}")
    return []

def fetch_images_serpapi_sync(vendor_name: str, product_name: str = None, model_family: str = None, product_type: str = None):
    """
    Synchronous version: SerpAPI fallback for Google Images
    Uses progressive query fallback:
      Priority 1: vendor + model_family (most specific)
      Priority 2: vendor + model_family + product_type
      Priority 3: vendor + product_type (least specific)
    
    Args:
        vendor_name: Name of the vendor/manufacturer
        product_name: (Optional) Specific product name/model
        model_family: (Optional) Model family/series to include in search
        product_type: (Optional) Type of product to help refine search
    """
    if not SERPAPI_KEY:
        logging.warning("SerpAPI key not available for image search")
        return []
    
    # Build manufacturer domain filter
    try:
        manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
    except Exception:
        manufacturer_domains = []
    
    domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains]) if manufacturer_domains else ""
    
    # Anti-spam keywords
    anti_spam = " product image OR product OR datasheet OR specification -used -refurbished -ebay -amazon -alibaba -walmart -etsy -pinterest -youtube -video -pdf -doc -xls -ppt -docx -xlsx -pptx"
    
    # Build progressive queries (from most specific to least specific)
    queries_to_try = []
    
    # Priority 1: vendor + model_family (most specific)
    if model_family:
        queries_to_try.append(f"{vendor_name} {model_family}{anti_spam}")
    
    # Priority 2: vendor + model_family + product_type
    if model_family and product_type:
        queries_to_try.append(f"{vendor_name} {model_family} {product_type}{anti_spam}")
    
    # Priority 3: vendor + product_type (fallback)
    if product_type:
        queries_to_try.append(f"{vendor_name} {product_type}{anti_spam}")
    
    # Fallback: just vendor name
    if not queries_to_try:
        queries_to_try.append(f"{vendor_name}{anti_spam}")
    
    for priority_idx, base_query in enumerate(queries_to_try, 1):
        try:
            if domain_filter:
                search_query = f"{base_query} ({domain_filter}) filetype:jpg OR filetype:png"
            else:
                search_query = f"{base_query} filetype:jpg OR filetype:png"
            
            logging.info(f"SerpAPI Priority {priority_idx} query: {search_query[:100]}...")
            
            search = GoogleSearch({
                "q": search_query,
                "engine": "google_images",
                "api_key": SERPAPI_KEY,
                "num": 8,
                "safe": "active",
                "ijn": 0
            })
            
            results = search.get_dict()
            images = []
            
            for item in results.get("images_results", []):
                images.append({
                    "url": item.get("original"),
                    "title": item.get("title", ""),
                    "source": "serpapi",
                    "thumbnail": item.get("thumbnail", ""),
                    "domain": item.get("source", "")
                })
            
            if images:
                logging.info(f"SerpAPI found {len(images)} images at Priority {priority_idx} for {vendor_name}")
                return images
            else:
                logging.info(f"SerpAPI Priority {priority_idx} returned no images, trying next priority...")
                
        except Exception as e:
            logging.warning(f"SerpAPI Priority {priority_idx} failed for {vendor_name}: {e}")
            continue
    
    logging.warning(f"SerpAPI all priorities exhausted for {vendor_name}")
    return []

def fetch_images_serper_sync(vendor_name: str, product_name: str = None, model_family: str = None, product_type: str = None):
    """
    Synchronous version: Serper.dev fallback for images
    Uses progressive query fallback:
      Priority 1: vendor + model_family (most specific)
      Priority 2: vendor + model_family + product_type
      Priority 3: vendor + product_type (least specific)
    
    Args:
        vendor_name: Name of the vendor/manufacturer
        product_name: (Optional) Specific product name/model
        model_family: (Optional) Model family/series to include in search
        product_type: (Optional) Type of product to help refine search
    """
    if not SERPER_API_KEY_IMAGES:
        logging.warning("Serper API key not available for image search")
        return []
    
    # Build manufacturer domain filter
    try:
        manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
    except Exception:
        manufacturer_domains = []
    
    domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains]) if manufacturer_domains else ""
    
    # Anti-spam keywords
    anti_spam = " product image OR product OR datasheet OR specification -used -refurbished -ebay -amazon -alibaba -walmart -etsy -pinterest -youtube -video -pdf -doc -xls -ppt -docx -xlsx -pptx"
    
    # Build progressive queries (from most specific to least specific)
    queries_to_try = []
    
    # Priority 1: vendor + model_family (most specific)
    if model_family:
        queries_to_try.append(f"{vendor_name} {model_family}{anti_spam}")
    
    # Priority 2: vendor + model_family + product_type
    if model_family and product_type:
        queries_to_try.append(f"{vendor_name} {model_family} {product_type}{anti_spam}")
    
    # Priority 3: vendor + product_type (fallback)
    if product_type:
        queries_to_try.append(f"{vendor_name} {product_type}{anti_spam}")
    
    # Fallback: just vendor name
    if not queries_to_try:
        queries_to_try.append(f"{vendor_name}{anti_spam}")
    
    for priority_idx, base_query in enumerate(queries_to_try, 1):
        try:
            if domain_filter:
                search_query = f"{base_query} ({domain_filter}) filetype:jpg OR filetype:png"
            else:
                search_query = f"{base_query} filetype:jpg OR filetype:png"
            
            logging.info(f"Serper Priority {priority_idx} query: {search_query[:100]}...")
            
            url = "https://google.serper.dev/images"
            payload = {
                "q": search_query,
                "num": 8
            }
            headers = {
                "X-API-KEY": SERPER_API_KEY_IMAGES,
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            images = []
            
            for item in data.get("images", []):
                images.append({
                    "url": item.get("imageUrl"),
                    "title": item.get("title", ""),
                    "source": "serper",
                    "thumbnail": item.get("imageUrl"),  # Serper doesn't provide separate thumbnail
                    "domain": item.get("link", "")
                })
            
            if images:
                logging.info(f"Serper found {len(images)} images at Priority {priority_idx} for {vendor_name}")
                return images
            else:
                logging.info(f"Serper Priority {priority_idx} returned no images, trying next priority...")
                
        except Exception as e:
            logging.warning(f"Serper Priority {priority_idx} failed for {vendor_name}: {e}")
            continue
    
    logging.warning(f"Serper all priorities exhausted for {vendor_name}")
    return []

def select_best_image_from_batch(image_urls: list, context: str) -> int:
    """
    Selects the best image from a batch using LLM vision.
    Args:
        image_urls: List of image URLs to evaluate
        context: Description of what the images should be
    Returns:
        int: Index of the best image (0-based), or -1 if no good image found.
    """
    if not image_urls:
        return -1

    if not components or 'llm_flash' not in components:
        logging.warning("LLM components not ready for image selection")
        # Fail open: pick the first one if LLM not ready
        return 0

    try:
        logging.info(f"Selecting best image from {len(image_urls)} candidates for context: {context}")
        
        # Prepare the prompt
        prompt = f"""
        You are an expert product image curator.
        Context: "{context}"
        
        Task: Analyze the provided images and select the SINGLE BEST image that accurately represents this product.
        
        Criteria:
        1. Relevance: Must clearly show the product mentioned in the context (correct Vendor, Model).
        2. Quality: Prefer high-resolution, clear lighting.
        3. Style: Prefer isolated product shots (white background) over messy lifestyle shots, but accuracy is most important.
        4. Exclusions: Do NOT select company logos, generic icons, diagrams, people, or unrelated items.
        
        Return JSON ONLY: {{"best_image_index": <integer index 0-{len(image_urls)-1}>, "reason": "brief reason"}}
        If NONE of the images are good, return "best_image_index": -1.
        """
        
        # Construct message content with multiple images
        content_parts = [{"type": "text", "text": prompt}]
        
        # Add images (Limit to top 5 to avoid payload issues and timeouts)
        # Check calling function, but enforce limit here too
        images_to_check = image_urls[:5]
        
        for i, url in enumerate(images_to_check):
            content_parts.append({
                "type": "text", 
                "text": f"Image {i}:"
            })
            content_parts.append({
                "type": "image_url", 
                "image_url": {"url": url}
            })
            
        message = HumanMessage(content=content_parts)
        
        # Call LLM with explicit timeout (10 seconds)
        try:
            # Using config timeout for LangChain
            response = components['llm_flash'].invoke([message], config={"timeout": 10})
            content = response.content
        except Exception as timeout_error:
            # Specific error message matching user report style
            logging.warning(f"LLM image selection timed out. Context: {context}")
            logging.warning(f"Detailed error: {timeout_error}")
            return 0 # Fallback to first image
        
        # Parse JSON
        import json
        try:
            # Clean possible markdown code blocks
            clean_content = content.replace("```json", "").replace("```", "").strip()
            # Find JSON object start/end
            start = clean_content.find('{')
            end = clean_content.rfind('}') + 1
            if start != -1 and end != -1:
                clean_content = clean_content[start:end]
                
            result = json.loads(clean_content)
            best_index = result.get("best_image_index", -1)
            reason = result.get("reason", "No reason provided")
            
            logging.info(f"LLM Selected Image Index: {best_index}. Reason: {reason}")
            
            # Validate index
            if isinstance(best_index, int) and 0 <= best_index < len(images_to_check):
                return best_index
            else:
                return -1
            
        except Exception as e:
            logging.warning(f"Failed to parse LLM selection response: {e}. Content: {content}")
            # Fallback to -1 if we can't parse, so logic outside knows LLM failed
            return -1
            
    except Exception as e:
        logging.warning(f"Batch image selection failed: {e}")
        return 0 # Fallback to first image on catastrophic failure

def fetch_product_images_with_fallback_sync(vendor_name: str, product_name: str = None, manufacturer_domains: list = None, model_family: str = None, product_type: str = None):
    """
    Synchronous 3-level image search fallback system with MongoDB caching and LLM verification
    0. Check MongoDB cache first
    1. Google Custom Search API (manufacturer domains)
    2. SerpAPI Google Images
    3. Serper.dev images
    
    Args:
        vendor_name: Name of the vendor/manufacturer
        product_name: (Optional) Specific product name/model
        manufacturer_domains: (Optional) List of manufacturer domains to search within
        model_family: (Optional) Model family/series to include in search
        product_type: (Optional) Type of product to help refine search
    """
    logging.info(f"Starting image search for vendor: {vendor_name}, product: {product_name}, model family: {model_family}, type: {product_type}")
    
    # Step 0: Check MongoDB cache first (if model_family is provided)
    if vendor_name:
        from mongodb_utils import get_cached_image, cache_image
        
        # Use model_family for key if available, otherwise just vendor (less specific but better than nothing)
        cache_key_model = model_family if model_family else ""
        
        cached_image = get_cached_image(vendor_name, cache_key_model)
        if cached_image:
            logging.info(f"Using cached image from GridFS for {vendor_name} - {cache_key_model}")
            # Convert GridFS file_id to backend URL
            gridfs_file_id = cached_image.get('gridfs_file_id')
            backend_url = f"/api/images/{gridfs_file_id}"
            
            # Return image with backend URL
            image_response = {
                'url': backend_url,
                'title': cached_image.get('title', ''),
                'source': 'mongodb_gridfs',
                'thumbnail': backend_url,  # Same URL for thumbnail
                'domain': 'local',
                'cached': True,
                'gridfs_file_id': gridfs_file_id
            }
            return [image_response], "mongodb_gridfs"
    
    # Helper to process and select best image
    def process_found_images(found_images, source_name):
        if not found_images:
            return None
            
        logging.info(f"Source {source_name} found {len(found_images)} images. Selecting best...")
        
        # Construct context for verification
        context_parts = [vendor_name]
        if model_family: context_parts.append(model_family)
        if product_type: context_parts.append(product_type)
        if product_name: context_parts.append(product_name)
        verification_context = " ".join(context_parts)
        
        # Select candidates (User requested Top 8)
        candidates = found_images[:8]
        # Filter out None/invalid URLs before passing to LLM
        candidate_urls = [img.get('url') for img in candidates if img.get('url') and img.get('url').startswith(('http://', 'https://'))]
        
        # Select best image
        best_idx = select_best_image_from_batch(candidate_urls, verification_context)
        
        if best_idx != -1 and best_idx < len(candidates):
            best_image = candidates[best_idx]
            logging.info(f"Selected best image from {source_name} (index {best_idx}): {best_image.get('url')}")
            
            # Cache the BEST image
            if vendor_name: 
                from mongodb_utils import cache_image
                # Use model_family if available, else empty string
                cache_key_model = model_family if model_family else ""
                
                cache_image(vendor_name, cache_key_model, best_image)
            
            # Return ONLY the best image as a single-item list
            return [best_image]
        else:
            logging.warning(f"No good image selected from {source_name} batch.")
            return None

    # Step 1: Try SerpAPI (Primary - Best for industrial products)
    images_serp = fetch_images_serpapi_sync(
        vendor_name=vendor_name,
        product_name=product_name,
        model_family=model_family,
        product_type=product_type
    )
    if images_serp:
        best_serp = process_found_images(images_serp, "serpapi")
        if best_serp:
            return best_serp, "serpapi"
    
    # Step 2: Try Serper.dev (Secondary - Cost-effective fallback)
    images_serper = fetch_images_serper_sync(
        vendor_name=vendor_name,
        product_name=product_name,
        model_family=model_family,
        product_type=product_type
    )
    if images_serper:
        best_serper = process_found_images(images_serper, "serper")
        if best_serper:
            return best_serper, "serper"
    
    # Step 3: Try Google Custom Search API (Tertiary - Only if GOOGLE_CX is configured)
    images_cse = fetch_images_google_cse_sync(
        vendor_name=vendor_name,
        product_name=product_name,
        manufacturer_domains=manufacturer_domains,
        model_family=model_family,
        product_type=product_type
    )
    if images_cse:
        best_cse = process_found_images(images_cse, "google_cse")
        if best_cse:
            return best_cse, "google_cse"
    
    # Fallback: If verification failed for all sources but we had images, return the first one unverified
    # Priority order matches above: SerpAPI → Serper → CSE
    
    logging.info("Selection failed or no images found in all sources. Returning unverified fallback (top 1) if available.")
    
    # Cache the unverified fallback to prevent repeated slow searches
    fallback_image = None
    fallback_source = "none"
    
    if images_serp:
        fallback_image = images_serp[0]
        fallback_source = "serpapi (unverified)"
    elif images_serper:
        fallback_image = images_serper[0]
        fallback_source = "serper (unverified)"
    elif images_cse:
        fallback_image = images_cse[0]
        fallback_source = "google_cse (unverified)"
    
    # Cache the fallback image to prevent repeated slow searches
    if fallback_image and vendor_name:
        try:
            from mongodb_utils import cache_image
            cache_key_model = model_family if model_family else ""
            cache_image(vendor_name, cache_key_model, fallback_image)
            logging.info(f"Cached unverified fallback image for {vendor_name} - {cache_key_model}")
        except Exception as e:
            logging.warning(f"Failed to cache unverified fallback: {e}")
        return [fallback_image], fallback_source
    
    # All failed - no images at all
    logging.warning(f"All image search APIs failed or returned no results for {vendor_name}")
    return [], "none"

def fetch_vendor_logo_sync(vendor_name: str, manufacturer_domains: list = None):
    """
    Specialized function to fetch vendor logo with MongoDB caching
    """
    logging.info(f"Fetching logo for vendor: {vendor_name}")
    
    # Step 0: Check MongoDB cache first
    try:
        from mongodb_utils import mongodb_file_manager, download_image_from_url
        
        logos_collection = mongodb_file_manager.collections.get('vendor_logos')
        if logos_collection is not None:
            normalized_vendor = vendor_name.strip().lower()
            
            cached_logo = logos_collection.find_one({
                'vendor_name_normalized': normalized_vendor
            })
            
            if cached_logo and cached_logo.get('gridfs_file_id'):
                logging.info(f"Using cached logo from GridFS for {vendor_name}")
                gridfs_file_id = cached_logo.get('gridfs_file_id')
                backend_url = f"/api/images/{gridfs_file_id}"
                
                return {
                    'url': backend_url,
                    'thumbnail': backend_url,
                    'source': 'mongodb_gridfs',
                    'title': cached_logo.get('title', f"{vendor_name} Logo"),
                    'domain': 'local',
                    'cached': True,
                    'gridfs_file_id': str(gridfs_file_id)
                }
    except Exception as e:
        logging.warning(f"Failed to check logo cache for {vendor_name}: {e}")
    
    # Step 1: Cache miss - fetch from web
    logo_result = None
    
    # Try different logo-specific searches
    logo_queries = [
        f"{vendor_name} logo transparent",
        f"{vendor_name} company logo transparent", 
        f"{vendor_name} brand transparent",
        f"{vendor_name} transparent"
    ]
    
    for query in logo_queries:
        try:
            # Use Google CSE first for official logos
            if GOOGLE_API_KEY and GOOGLE_CX:
                # Build site restriction for manufacturer domains using LLM (or reuse if provided)
                if manufacturer_domains is None:
                    manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
                domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains])
                search_query = f"{query} ({domain_filter}) filetype:jpg OR filetype:png OR filetype:svg"
                
                service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
                result = service.cse().list(
                    q=search_query,
                    cx=GOOGLE_CX,
                    searchType="image",
                    num=3,  # Only need a few logo options
                    safe="medium",
                    imgSize="MEDIUM"
                ).execute()
                
                for item in result.get("items", []):
                    logo_url = item.get("link")
                    title = item.get("title", "").lower()
                    
                    # Check if this looks like a logo
                    if any(keyword in title for keyword in ["logo", "brand", "company"]):
                        logo_result = {
                            "url": logo_url,
                            "thumbnail": item.get("image", {}).get("thumbnailLink", logo_url),
                            "source": "google_cse_logo",
                            "title": item.get("title", ""),
                            "domain": item.get("displayLink", "")
                        }
                        break
                
                # If no specific logo found, use first result from official domain
                if not logo_result and result.get("items"):
                    item = result["items"][0]
                    logo_result = {
                        "url": item.get("link"),
                        "thumbnail": item.get("image", {}).get("thumbnailLink", item.get("link")),
                        "source": "google_cse_general",
                        "title": item.get("title", ""),
                        "domain": item.get("displayLink", "")
                    }
                
                if logo_result:
                    break
                    
        except Exception as e:
            logging.warning(f"Logo search failed for query '{query}': {e}")
            continue
    
    # Fallback: use general vendor search
    if not logo_result:
        try:
            images, source = fetch_product_images_with_fallback_sync(vendor_name, "")
            if images:
                # Return first image as logo
                logo_result = images[0].copy()
                logo_result["source"] = f"{source}_fallback"
        except Exception as e:
            logging.warning(f"Fallback logo search failed for {vendor_name}: {e}")
    
    # Step 2: Cache the logo in MongoDB if found
    if logo_result:
        try:
            from mongodb_utils import mongodb_file_manager, download_image_from_url
            
            logo_url = logo_result.get('url')
            if logo_url and not logo_url.startswith('/api/images/'):  # Don't re-cache GridFS URLs
                # Download the logo
                download_result = download_image_from_url(logo_url)
                if download_result:
                    image_bytes, content_type, file_size = download_result
                    
                    gridfs = mongodb_file_manager.gridfs
                    logos_collection = mongodb_file_manager.collections.get('vendor_logos')
                    
                    if logos_collection is not None:
                        normalized_vendor = vendor_name.strip().lower()
                        file_extension = content_type.split('/')[-1] if '/' in content_type else 'png'
                        filename = f"logo_{normalized_vendor}.{file_extension}"
                        
                        # Store in GridFS
                        gridfs_file_id = gridfs.put(
                            image_bytes,
                            filename=filename,
                            content_type=content_type,
                            vendor_name=vendor_name,
                            original_url=logo_url,
                            logo_type='vendor_logo'
                        )
                        
                        logging.info(f"Stored vendor logo in GridFS: {filename} (ID: {gridfs_file_id})")
                        
                        # Store metadata
                        logo_doc = {
                            'vendor_name': vendor_name,
                            'vendor_name_normalized': normalized_vendor,
                            'gridfs_file_id': gridfs_file_id,
                            'original_url': logo_url,
                            'title': logo_result.get('title', f"{vendor_name} Logo"),
                            'source': logo_result.get('source', ''),
                            'domain': logo_result.get('domain', ''),
                            'content_type': content_type,
                            'file_size': file_size,
                            'filename': filename,
                            'created_at': datetime.utcnow()
                        }
                        
                        logos_collection.update_one(
                            {'vendor_name_normalized': normalized_vendor},
                            {'$set': logo_doc},
                            upsert=True
                        )
                        
                        logging.info(f"Successfully cached vendor logo for {vendor_name}")
                        
                        # Return cached version
                        backend_url = f"/api/images/{gridfs_file_id}"
                        return {
                            'url': backend_url,
                            'thumbnail': backend_url,
                            'source': 'mongodb_gridfs',
                            'title': logo_doc['title'],
                            'domain': 'local',
                            'cached': True,
                            'gridfs_file_id': str(gridfs_file_id)
                        }
        except Exception as e:
            logging.warning(f"Failed to cache vendor logo for {vendor_name}: {e}")
    
    return logo_result

async def fetch_images_google_cse(vendor_name: str, model_family: str = None, product_type: str = None):
    """
    Step 1: Google Custom Search API for images from manufacturer domains
    """
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        logging.warning("Google CSE credentials not available for image search")
        return []
    
    try:
        query = f"{vendor_name}"
        if model_family:
            query += f" {model_family}"
        if product_type:
            query += f" {product_type}"
        
        # Build site restriction for manufacturer domains using LLM
        manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
        domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains])
        search_query = f"{query} ({domain_filter}) filetype:jpg OR filetype:png"
        
        # Use Google Custom Search API
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        result = service.cse().list(
            q=search_query,
            cx=GOOGLE_CX,
            searchType="image",
            num=8,
            safe="medium",
            imgSize="MEDIUM"
        ).execute()
        
        images = []
        unsupported_schemes = ['x-raw-image://', 'data:', 'blob:', 'chrome://', 'about:']
        
        for item in result.get("items", []):
            url = item.get("link")
            
            # Skip images with unsupported URL schemes
            if not url or any(url.startswith(scheme) for scheme in unsupported_schemes):
                logging.debug(f"Skipping image with unsupported URL scheme: {url}")
                continue
            
            # Only include http/https URLs
            if not url.startswith(('http://', 'https://')):
                logging.debug(f"Skipping non-HTTP URL: {url}")
                continue
                
            images.append({
                "url": url,
                "title": item.get("title", ""),
                "source": "google_cse",
                "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                "domain": item.get("displayLink", "")
            })
        
        if images:
            logging.info(f"Google CSE found {len(images)} valid images for {vendor_name}")
        return images
        
    except Exception as e:
        logging.warning(f"Google CSE image search failed for {vendor_name}: {e}")
        return []

async def fetch_images_serpapi(vendor_name: str, model_family: str = None, product_type: str = None):
    """
    Step 2: SerpAPI fallback for Google Images
    """
    if not SERPAPI_KEY:
        logging.warning("SerpAPI key not available for image search")
        return []
    
    try:
        base_query = vendor_name
        if model_family:
            base_query += f" {model_family}"
        if product_type:
            base_query += f" {product_type}"
        base_query += " product OR datasheet OR specification -used -refurbished -ebay -amazon -alibaba -walmart -etsy -pinterest -youtube -video -pdf -doc -xls -ppt -docx -xlsx -pptx"

        try:
            manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
        except Exception:
            manufacturer_domains = []

        domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains]) if manufacturer_domains else ""
        if domain_filter:
            search_query = f"{base_query} ({domain_filter}) filetype:jpg OR filetype:png"
        else:
            search_query = f"{base_query} filetype:jpg OR filetype:png"

        search = GoogleSearch({
            "q": search_query,
            "engine": "google_images",
            "api_key": SERPAPI_KEY,
            "num": 8,
            "safe": "medium",
            "ijn": 0
        })
        
        results = search.get_dict()
        images = []
        
        for item in results.get("images_results", []):
            images.append({
                "url": item.get("original"),
                "title": item.get("title", ""),
                "source": "serpapi",
                "thumbnail": item.get("thumbnail", ""),
                "domain": item.get("source", "")
            })
        
        if images:
            logging.info(f"SerpAPI found {len(images)} images for {vendor_name}")
        return images
        
    except Exception as e:
        logging.warning(f"SerpAPI image search failed for {vendor_name}: {e}")
        return []

async def fetch_images_serper(vendor_name: str, model_family: str = None, product_type: str = None):
    """
    Step 3: Serper.dev fallback for images
    """
    if not SERPER_API_KEY_IMAGES:
        logging.warning("Serper API key not available for image search")
        return []
    
    try:
        base_query = vendor_name
        if model_family:
            base_query += f" {model_family}"
        if product_type:
            base_query += f" {product_type}"
        base_query += " product OR datasheet OR specification -used -refurbished -ebay -amazon -alibaba -walmart -etsy -pinterest -youtube -video -pdf -doc -xls -ppt -docx -xlsx -pptx"

        try:
            manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
        except Exception:
            manufacturer_domains = []

        domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains]) if manufacturer_domains else ""
        if domain_filter:
            search_query = f"{base_query} ({domain_filter}) filetype:jpg OR filetype:png"
        else:
            search_query = f"{base_query} filetype:jpg OR filetype:png"

        url = "https://google.serper.dev/images"
        payload = {
            "q": search_query,
            "num": 8
        }
        headers = {
            "X-API-KEY": SERPER_API_KEY_IMAGES,
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        images = []
        
        for item in data.get("images", []):
            images.append({
                "url": item.get("imageUrl"),
                "title": item.get("title", ""),
                "source": "serper",
                "thumbnail": item.get("imageUrl"),  # Serper doesn't provide separate thumbnail
                "domain": item.get("link", "")
            })
        
        if images:
            logging.info(f"Serper found {len(images)} images for {vendor_name}")
        return images
        
    except Exception as e:
        logging.warning(f"Serper image search failed for {vendor_name}: {e}")
        return []

async def fetch_product_images_with_fallback(vendor_name: str, product_name: str = None, model_family: str = None, product_type: str = None):
    """
    3-level image search fallback system
    1. Google Custom Search API (manufacturer domains)
    2. SerpAPI Google Images
    3. Serper.dev images
    """
    logging.info(f"Starting image search for vendor: {vendor_name}, product: {product_name}")
    
    # Step 1: Try Google Custom Search API (pass model_family/product_type, avoid raw product_name)
    images = await fetch_images_google_cse(vendor_name, model_family if model_family else None)
    if images:
        logging.info(f"Using Google CSE images for {vendor_name}")
        return images, "google_cse"
    
    # Step 2: Try SerpAPI
    images = await fetch_images_serpapi(vendor_name, model_family if model_family else None)
    if images:
        logging.info(f"Using SerpAPI images for {vendor_name}")
        return images, "serpapi"
    
    # Step 3: Try Serper.dev
    images = await fetch_images_serper(vendor_name, model_family if model_family else None)
    if images:
        logging.info(f"Using Serper images for {vendor_name}")
        return images, "serper"
    
    # All failed
    logging.warning(f"All image search APIs failed for {vendor_name}")
    return [], "none"


@app.route("/api/test_image_search", methods=["GET"])
@login_required
def test_image_search():
    """
    Test endpoint for the image search functionality
    """
    vendor_name = request.args.get("vendor", "Emerson")
    product_name = request.args.get("product", "")
    
    try:
        # Use synchronous version for reliability; pass model_family instead of product_name
        model_family = None
        # If user provided product as a family list or string, prefer it
        if product_name and ',' in product_name:
            # accept '3051C,3051S' style input from quick tests
            model_family = product_name.split(',')[0].strip()
        elif product_name:
            model_family = product_name.strip()

        images, source_used = fetch_product_images_with_fallback_sync(
            vendor_name,
            product_name=None,
            manufacturer_domains=None,
            model_family=model_family,
            product_type=None
        )
        
        # Also test domain generation
        generated_domains = get_manufacturer_domains_from_llm(vendor_name)
        
        return jsonify({
            "vendor": vendor_name,
            "product": product_name,
            "images": images,
            "source_used": source_used,
            "count": len(images),
            "generated_domains": generated_domains
        })
        
    except Exception as e:
        logging.error(f"Image search test failed: {e}")
        return jsonify({
            "error": str(e),
            "vendor": vendor_name,
            "product": product_name,
            "images": [],
            "source_used": "error",
            "count": 0,
            "generated_domains": []
        }), 500


@app.route("/api/get_analysis_product_images", methods=["POST"])
@login_required
def get_analysis_product_images():
    """
    Get images for specific products from analysis results.
    Expected input:
    {
        "vendor": "Emerson",
        "product_type": "Flow Transmitter", 
        "product_name": "Rosemount 3051",
        "model_families": ["3051C", "3051S", "3051T"]
    }
    """
    try:
        data = request.get_json()

        vendor = data.get("vendor", "")
        product_type = data.get("product_type", "")
        product_name = data.get("product_name", "")
        model_families = data.get("model_families", [])

        if not vendor:
            return jsonify({"error": "Vendor name is required"}), 400

        # Removed requirements_match check - fetch images for ALL products (exact and approximate matches)
        # This supports the fallback display of approximate matches when no exact matches are found
        logging.info(f"Fetching images for analysis result: {vendor} {product_type} {product_name}")

        # Generate manufacturer domains once per request for this vendor
        manufacturer_domains = get_manufacturer_domains_from_llm(vendor)

        # Search for images with different combinations
        all_images = []
        search_combinations = []

        # Prefer model family for search if available (e.g., STD800 instead of submodel STD830)
        primary_family = None
        if isinstance(model_families, list) and model_families:
            primary_family = str(model_families[0]).strip()

        # Build a base name for search: model family if present, otherwise product_name
        # Example: "STD800" instead of "STD830 Pressure Transmitter"
        base_name_for_search = primary_family or product_name

        # 1. Most specific: vendor + base_name_for_search + product_type
        if base_name_for_search and product_type:
            search_query = f"{vendor} {base_name_for_search} {product_type}"
            search_combinations.append({
                "query": search_query,
                "type": "family_with_type",
                "priority": 1
            })

        # 2. Medium specific: vendor + base_name_for_search
        if base_name_for_search:
            search_query = f"{vendor} {base_name_for_search}"
            search_combinations.append({
                "query": search_query,
                "type": "family_or_name",
                "priority": 2
            })

        # 3. General: vendor + product_type
        if product_type:
            search_query = f"{vendor} {product_type}"
            search_combinations.append({
                "query": search_query,
                "type": "type_general",
                "priority": 3
            })
        
        # Execute searches and collect results
        for search_info in search_combinations:
            try:
                # Pass model_family and product_type to the fetcher and avoid using raw product_name
                images, source_used = fetch_product_images_with_fallback_sync(
                    vendor_name=vendor,
                    product_name=None,
                    manufacturer_domains=manufacturer_domains,
                    model_family=base_name_for_search if base_name_for_search else None,
                    product_type=product_type if product_type else None,
                )
                
                # Add metadata to images
                for img in images:
                    img["search_type"] = search_info["type"]
                    img["search_priority"] = search_info["priority"]
                    img["search_query"] = search_info["query"]
                
                all_images.extend(images)
                
                # If we get good results from high-priority search, we can stop early
                if len(images) >= 5 and search_info["priority"] <= 2:
                    logging.info(f"Got {len(images)} images from high-priority search: {search_info['type']}")
                    break
                    
            except Exception as e:
                logging.warning(f"Search failed for query '{search_info['query']}': {e}")
                continue
        
        # Remove duplicates based on URL
        unique_images = []
        seen_urls = set()
        for img in all_images:
            url = img.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_images.append(img)
        
        # Sort by priority and quality
        def image_quality_score(img):
            score = 0
            
            # Priority weight (lower priority number = higher score)
            score += (5 - img.get("search_priority", 5)) * 10
            
            # Domain quality (official domains get higher score)
            domain = img.get("domain", "").lower()
            if any(mfg_domain in domain for mfg_domain in manufacturer_domains):
                score += 15
            
            # Source quality
            source = img.get("source", "")
            if source == "google_cse":
                score += 10
            elif source == "serpapi":
                score += 5
            
            # Title relevance (contains product name or model family)
            title = img.get("title", "").lower()
            if product_name.lower() in title:
                score += 8
            for model in model_families:
                if model.lower() in title:
                    score += 6
                    break
            
            return score
        
        # Sort by quality score (highest first)
        unique_images.sort(key=image_quality_score, reverse=True)
        
        # Select best images - top 1 for main display, top 10 for "view more"
        top_image = unique_images[0] if unique_images else None
        best_images = unique_images[:10]
        
        # Get vendor logo using specialized logo search
        vendor_logo = None
        try:
            vendor_logo = fetch_vendor_logo_sync(vendor, manufacturer_domains=manufacturer_domains)
        except Exception as e:
            logging.warning(f"Failed to fetch vendor logo for {vendor}: {e}")
        
        # Prepare response
        response_data = {
            "vendor": vendor,
            "product_type": product_type,
            "product_name": product_name,
            "model_families": model_families,
            "top_image": top_image,  # Single best image for main display
            "vendor_logo": vendor_logo,  # Vendor logo
            "all_images": best_images,  # All images for "view more"
            # Compatibility fields: many frontends expect `images` or `image`
            "images": best_images,
            "image": top_image,
            "total_found": len(all_images),
            "unique_count": len(unique_images),
            "best_count": len(best_images),
            "search_summary": {
                "searches_performed": len(search_combinations),
                "search_types": list(set(img.get("search_type") for img in best_images)),
                "sources_used": list(set(img.get("source") for img in best_images))
            }
        }
        
        logging.info(f"Analysis image search completed: {len(best_images)} best images selected from {len(all_images)} total")
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"Analysis product image search failed: {e}")
        return jsonify({
            "error": f"Failed to fetch analysis product images: {str(e)}",
            "vendor": data.get("vendor", ""),
            "product_type": data.get("product_type", ""),
            "product_name": data.get("product_name", ""),
            "model_families": data.get("model_families", []),
            "top_image": None,
            "vendor_logo": None,
            "all_images": [],
            "total_found": 0,
            "unique_count": 0,
            "best_count": 0
        }), 500


@app.route("/api/upload_pdf_from_url", methods=["POST"])
@login_required
def upload_pdf_from_url():
    data = request.get_json(force=True)
    pdf_url = data.get("url")
    if not pdf_url:
        return jsonify({"error": "Missing 'url' parameter"}), 400

    try:
        # --- 1. Download PDF ---
        print(f"[DOWNLOAD] Fetching PDF: {pdf_url}")
        response = requests.get(pdf_url, stream=True, timeout=30)
        response.raise_for_status()

        filename = os.path.basename(urllib.parse.urlparse(pdf_url).path) or "document.pdf"
        pdf_bytes = response.content  # keep PDF in memory

        # --- 2. Extract data from PDF ---
        text_chunks = extract_data_from_pdf(BytesIO(pdf_bytes))
        raw_results = send_to_language_model(text_chunks)

        # Flatten results
        def flatten_results(results):
            flat = []
            for r in results:
                if isinstance(r, list):
                    flat.extend(r)
                else:
                    flat.append(r)
            return flat

        all_results = flatten_results(raw_results)
        final_result = aggregate_results(all_results, filename)
        
        # Apply standardization to the final result before splitting
        try:
            standardized_final_result = standardize_vendor_analysis_result(final_result)
            logging.info("Applied standardization to PDF from URL analysis")
        except Exception as e:
            logging.warning(f"Failed to standardize PDF from URL result: {e}")
            standardized_final_result = final_result

        # --- 3. Split by product types ---
        split_results = split_product_types([standardized_final_result])

        saved_json_paths = []
        saved_pdf_paths = []

        for result in split_results:
            # --- 4. Save JSON result to MongoDB ---
            vendor = (result.get("vendor") or "UnknownVendor").replace(" ", "_")
            product_type = (result.get("product_type") or "UnknownProduct").replace(" ", "_")
            model_series = (
                result.get("models", [{}])[0].get("model_series") or "UnknownModel"
            ).replace(" ", " ")
            
            try:
                # Upload product JSON to MongoDB
                # Structure: vendors/{vendor}/{product_type}/{model}.json
                product_metadata = {
                    'vendor_name': vendor,
                    'product_type': product_type,
                    'model_series': model_series,
                    'file_type': 'json',
                    'collection_type': 'products',
                    'path': f'vendors/{vendor}/{product_type}/{model_series}.json'
                }
                mongodb_file_manager.upload_json_data(result, product_metadata)
                saved_json_paths.append(f"MongoDB:products:{vendor}:{product_type}:{model_series}")
                print(f"[INFO] Stored product JSON to MongoDB: {vendor} - {product_type}")
            except Exception as e:
                logging.error(f"Failed to save product JSON to MongoDB: {e}")

            # --- 5. Save PDF to MongoDB GridFS ---
            try:
                pdf_metadata = {
                    'vendor_name': vendor,
                    'product_type': product_type,
                    'model_series': model_series,
                    'file_type': 'pdf',
                    'collection_type': 'documents',
                    'filename': filename,
                    'path': f'documents/{vendor}/{product_type}/{filename}'
                }
                file_id = mongodb_file_manager.upload_to_mongodb(pdf_bytes, pdf_metadata)
                saved_pdf_paths.append(f"MongoDB:GridFS:{file_id}")
                print(f"[INFO] Stored PDF to MongoDB GridFS: {filename} (ID: {file_id})")
            except Exception as e:
                logging.error(f"Failed to save PDF to MongoDB: {e}")

            # --- Note: Product image extraction removed - now using API-based image search ---

        return jsonify({
            "data": split_results,
            "pdfFiles": saved_pdf_paths,
            "jsonFiles": saved_json_paths
        }), 200

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch PDF from URL: {str(e)}"}), 500
    except Exception as e:
        logging.exception("PDF analysis from URL failed.")
        return jsonify({"error": f"Failed to analyze PDF from URL: {str(e)}"}), 500

    
@app.route("/register", methods=["POST"])
def register():
    # Handle both JSON and Multipart data
    if request.content_type.startswith('multipart/form-data'):
        data = request.form
        file = request.files.get('document')
    else:
        data = request.get_json() or {}
        file = None

    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    first_name = data.get("first_name")
    last_name = data.get("last_name")
    
    # New Fields
    company_name = data.get("company_name")
    location = data.get("location")

    if not username or not email or not password:
        return jsonify({"error": "Missing username, email, or password"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 409
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 409

    # Handle File Upload - Store in GridFS AND extract strategy data using LLM
    document_file_id = None
    strategy_extraction_result = None
    pending_strategy_file_data = None
    pending_strategy_filename = None
    
    if file:
        try:
            filename = secure_filename(file.filename)
            file_data = file.read()
            
            # Store the file data for later LLM processing (after user is created)
            pending_strategy_file_data = file_data
            pending_strategy_filename = filename
            
            metadata = {
                'filename': filename,
                'content_type': file.content_type,
                'uploaded_by_username': username,
                'collection_type': 'strategy_documents'
            }
            # Upload to GridFS for raw file storage
            document_file_id = mongodb_file_manager.upload_to_mongodb(file_data, metadata)
            logging.info(f"Uploaded signup document for user {username}: {document_file_id}")
        except Exception as e:
            logging.error(f"Failed to upload signup document: {e}")
            # We continue registration even if file upload fails

    hashed_pw = hash_password(password)
    new_user = User(
        username=username,
        email=email,
        password_hash=hashed_pw,
        first_name=first_name,
        last_name=last_name,
        company_name=company_name,
        location=location,
        status='pending',
        role='user'
    )
    db.session.add(new_user)
    db.session.commit()
    
    # After user is created, process the strategy file with LLM extraction IN BACKGROUND
    # This ensures registration completes quickly without waiting for LLM processing
    if pending_strategy_file_data and pending_strategy_filename:
        logging.info(f"[REGISTER] Queueing background strategy extraction for new user {new_user.id}: {pending_strategy_filename}")
        extract_strategy_from_file_background(
            pending_strategy_file_data, 
            pending_strategy_filename, 
            new_user.id
        )

    return jsonify({"message": "User registration submitted. Awaiting admin approval."}), 201


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    user = User.query.filter_by(username=username).first()
    if user and check_password(user.password_hash, password):
        if user.status != 'active':
            return jsonify({"error": f"Account not active. Current status: {user.status}."}), 403
        
        # CLEAR EXISTING SESSION DATA TO START FRESH
        # This ensures no state from a previous session (even if unexpired) persists after re-login
        session.clear()
        
        session['user_id'] = user.id
        # Construct full name from first_name and last_name
        full_name = ""
        if user.first_name and user.last_name:
            full_name = f"{user.first_name} {user.last_name}"
        elif user.first_name:
            full_name = user.first_name
        elif user.last_name:
            full_name = user.last_name
        else:
            full_name = user.username
        
        return jsonify({
            "message": "Login successful",
            "user": {
                "username": user.username,
                "name": full_name,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "email": user.email,
                "role": user.role,
                "companyName": user.company_name,
                "location": user.location
            }
        }), 200

    return jsonify({"error": "Invalid username or password"}), 401

@app.route("/logout", methods=["POST"])
def logout():
    session.pop('user_id', None)
    return jsonify({"message": "Logout successful"}), 200

@app.route("/user", methods=["GET"])
@login_required
def get_current_user():
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({"error": "User not found"}), 404
    # Construct full name from first_name and last_name
    full_name = ""
    if user.first_name and user.last_name:
        full_name = f"{user.first_name} {user.last_name}"
    elif user.first_name:
        full_name = user.first_name
    elif user.last_name:
        full_name = user.last_name
    else:
        full_name = user.username
    
    return jsonify({
        "user": {
            "username": user.username,
            "name": full_name,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": user.email,
            "role": user.role,
            "companyName": user.company_name,
            "location": user.location
        }
    }), 200

def clean_empty_values(data):
    """Recursively replaces 'Not specified', 'Not requested', etc., with empty strings."""
    if isinstance(data, dict):
        return {k: clean_empty_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_empty_values(item) for item in data]
    elif isinstance(data, str) and data.lower().strip() in ["not specified", "not requested", "none specified", "n/a", "na"]:
        return ""
    return data
import copy

def map_provided_to_schema(detected_schema: dict, provided: dict) -> dict:
    """
    Always maps providedRequirements (flat or nested) into the schema structure.
    Works dynamically for any product type (Humidity, Pressure, etc.).
    """
    mapped = copy.deepcopy(detected_schema)

    # Case 1: Provided already structured (mandatory/optional) → overlay values
    if "mandatoryRequirements" in provided or "optionalRequirements" in provided:
        for section in ["mandatoryRequirements", "optionalRequirements"]:
            if section in provided and section in mapped:
                for key, value in provided[section].items():
                    if key in mapped[section]:
                        mapped[section][key] = value
        return mapped

    # Case 2: Provided is flat dict → distribute into schema
    for key, value in provided.items():
        if key in mapped.get("mandatoryRequirements", {}):
            mapped["mandatoryRequirements"][key] = value
        elif key in mapped.get("optionalRequirements", {}):
            mapped["optionalRequirements"][key] = value

    return mapped

# =========================================================================
# === PROGRESS TRACKING ENDPOINT ===
# =========================================================================

# Global progress tracker for long-running operations
current_operation_progress = None

@app.route("/api/progress", methods=["GET"])
@login_required
def get_operation_progress():
    """Get progress of current long-running operation"""
    global current_operation_progress
    
    if current_operation_progress is None:
        return jsonify({
            "status": "no_active_operation",
            "message": "No active operation in progress"
        }), 200
    
    try:
        progress_data = current_operation_progress.get_progress()
        return jsonify({
            "status": "in_progress",
            "progress": progress_data
        }), 200
    except Exception as e:
        logging.error(f"Failed to get progress: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to retrieve progress information"
        }), 500

# =========================================================================
# === VALIDATION ENDPOINT ===
# =========================================================================

@app.route("/debug-session/<session_id>", methods=["GET"])
@login_required
def debug_session_state(session_id):
    """Debug endpoint to check session state for a specific search session"""
    current_step_key = f'current_step_{session_id}'
    current_intent_key = f'current_intent_{session_id}'
    product_type_key = f'product_type_{session_id}'
    
    session_data = {
        'session_id': session_id,
        'current_step': session.get(current_step_key, 'None'),
        'current_intent': session.get(current_intent_key, 'None'),
        'product_type': session.get(product_type_key, 'None'),
        'all_session_keys': [k for k in session.keys() if session_id in k],
        'all_keys': list(session.keys()),  # Show all keys for debugging
        'session_size': len(session.keys())
    }
    
    return jsonify(session_data), 200

@app.route("/debug-session-clear/<session_id>", methods=["POST"])
@login_required  
def clear_session_state(session_id):
    """Debug endpoint to manually clear session state for testing"""
    keys_to_remove = [k for k in session.keys() if session_id in k]
    
    for key in keys_to_remove:
        del session[key]
    
    return jsonify({
        'session_id': session_id,
        'cleared_keys': keys_to_remove,
        'status': 'cleared'
    }), 200

@app.route("/validate", methods=["POST"])
@login_required
def api_validate():
    if not components:
        return jsonify({"error": "Backend is not ready. LangChain failed."}), 503
    try:
        data = request.get_json(force=True)
        user_input = data.get("user_input", "").strip()
        if not user_input:
            return jsonify({"error": "Missing user_input"}), 400

        # Get search session ID if provided (for multiple search tabs)
        search_session_id = data.get("search_session_id", "default")
        
        # By default preserve any previously-detected product type and workflow state for this
        # search session. Only clear them when the client explicitly requests a reset
        # (for example when initializing a brand-new independent search tab).
        session_key = f'product_type_{search_session_id}'
        step_key = f'current_step_{search_session_id}'
        intent_key = f'current_intent_{search_session_id}'

        if data.get('reset', False):
            if session_key in session:
                logging.info(f"[VALIDATE] Session {search_session_id}: Clearing previous product type due to reset request: {session[session_key]}")
                del session[session_key]
            if step_key in session:
                logging.info(f"[VALIDATE] Session {search_session_id}: Clearing step state due to reset request: {session[step_key]}")
                del session[step_key]
            if intent_key in session:
                logging.info(f"[VALIDATE] Session {search_session_id}: Clearing intent state due to reset request: {session[intent_key]}")
                del session[intent_key]
        else:
            logging.info(f"[VALIDATE] Session {search_session_id}: Preserving existing product_type and workflow state if present.")
        
        # Store original user input for logging (session-isolated)
        session[f'log_user_query_{search_session_id}'] = user_input

        initial_schema = load_requirements_schema()
        
        # Add session context to LLM validation to prevent cross-contamination
        session_isolated_input = f"[Session: {search_session_id}] - This is a fresh, independent validation request. User input: {user_input}"
        
        temp_validation_result = components['validation_chain'].invoke({
            "user_input": session_isolated_input,
            "schema": json.dumps(initial_schema, indent=2),
            "format_instructions": components['validation_format_instructions']
        })
        
        # Guard against LLM returning None/null
        if temp_validation_result is None:
            logging.warning("[VALIDATE] Wrapper: temp_validation_result is None from LLM. Defaulting product_type.")
            temp_validation_result = {}

        detected_type = temp_validation_result.get('product_type', 'UnknownProduct')
        
        specific_schema = load_requirements_schema(detected_type)
        if not specific_schema:
            global current_operation_progress
            try:
                # Set up progress tracking for web schema building
                from loading import ProgressTracker
                current_operation_progress = ProgressTracker(4, f"Building Schema for {detected_type}")
                specific_schema = build_requirements_schema_from_web(detected_type)
            finally:
                # Clear progress tracker when done
                current_operation_progress = None

        # Add session context to detailed validation as well
        session_isolated_input = f"[Session: {search_session_id}] - This is a fresh, independent validation request. User input: {user_input}"
        
        validation_result = components['validation_chain'].invoke({
            "user_input": session_isolated_input,
            "schema": json.dumps(specific_schema, indent=2),
            "format_instructions": components['validation_format_instructions']
        })

        if validation_result is None:
            logging.warning("[VALIDATE] Wrapper: validation_result is None from LLM. Using empty result.")
            validation_result = {}

        cleaned_provided_reqs = clean_empty_values(validation_result.get("provided_requirements", {}))

        mapped_provided_reqs = map_provided_to_schema(
            convert_keys_to_camel_case(specific_schema),
            convert_keys_to_camel_case(cleaned_provided_reqs)
        )

        response_data = {
            "productType": validation_result.get("product_type", detected_type),
            "detectedSchema": convert_keys_to_camel_case(specific_schema),
            "providedRequirements": mapped_provided_reqs
        }

        # ---------------- Helpers for missing mandatory fields ----------------
        def get_missing_mandatory_fields(provided: dict, schema: dict) -> list:
            missing = []
            mandatory_schema = schema.get("mandatoryRequirements", {})
            provided_mandatory = provided.get("mandatoryRequirements", {})

            def traverse_and_check(schema_node, provided_node):
                for key, schema_value in schema_node.items():
                    if isinstance(schema_value, dict):
                        traverse_and_check(schema_value, provided_node.get(key, {}) if isinstance(provided_node, dict) else {})
                    else:
                        provided_value = provided_node.get(key) if isinstance(provided_node, dict) else None
                        if provided_value is None or str(provided_value).strip() in ["", ","]:
                            missing.append(key)

            traverse_and_check(mandatory_schema, provided_mandatory)
            return missing

        missing_mandatory_fields = get_missing_mandatory_fields(
            mapped_provided_reqs, response_data["detectedSchema"]
        )

        # ---------------- Helper: Convert camelCase to friendly label ----------------
        def friendly_field_name(field):
            import re
            s1 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', field)
            return s1.replace("_", " ").title()

        # ---------------- Prompt user if any mandatory fields are missing ----------------
        if missing_mandatory_fields:
            # Convert missing fields to friendly labels
            missing_fields_friendly = [friendly_field_name(f) for f in missing_mandatory_fields]
            missing_fields_str = ", ".join(missing_fields_friendly)
            is_repeat = data.get("is_repeat", False)

            if not is_repeat:
                alert_prompt = ChatPromptTemplate.from_template(
                    """You are Engenie - an helpful sales agent. The user shared their requirements, and you identified the product type as '{product_type}'.
Your response must:
1. Start positively (e.g., "Great choice!" or similar).
2. Confirm the identified product type in a friendly way.
3. Tell the user some important details are still missing.
4. List the missing fields as simple key names only: **{missing_fields}**, separated by commas.
5. Explain that results may only be approximate without them.
6. Ask if they like to continue anyway."""
                )
            else:
                alert_prompt = ChatPromptTemplate.from_template(
                    """You are Engenie - a helpful sales assistant.
Write a short, clear response (1–2 sentences):
1. Tell the user there are still some missing specifications.
2. List the missing fields as simple key names only: **{missing_fields}**, separated by commas.
3. Explain that the search can continue, but results may only be approximate.
4. Ask if they like to proceed."""
                )

            alert_chain = alert_prompt | components['llm'] | StrOutputParser()
            agent_message = alert_chain.invoke({
                "product_type": response_data["productType"],
                "missing_fields": missing_fields_str
            })

            response_data["validationAlert"] = {
                "message": agent_message,
                "canContinue": True,
                "missingFields": missing_mandatory_fields
            }

        # NOTE: Vendor extraction is handled by identify_instruments endpoint
        # The Project page workflow uses identify_instruments which already extracts
        # specified_vendors and specified_model_families from user input.
        # No duplicate extraction needed here.

        # Store product_type in session for later use in advanced parameters (session-isolated)
        session[f'product_type_{search_session_id}'] = response_data["productType"]
        
        # CRITICAL: Set session step to 'awaitMissingInfo' if there are missing mandatory fields
        # This is required for the /api/intent endpoint to correctly handle "yes" responses
        # The intent API checks current_step to determine if user is responding to missing info prompt
        if missing_mandatory_fields:
            session[f'current_step_{search_session_id}'] = 'awaitMissingInfo'
            logging.info(f"[VALIDATE] Session {search_session_id}: Set step to 'awaitMissingInfo' - missing fields: {missing_mandatory_fields}")
        else:
            # If all mandatory fields are provided, stay at initialInput or move forward
            session[f'current_step_{search_session_id}'] = 'initialInput'
        session.modified = True

        return jsonify(response_data), 200

    except Exception as e:
        logging.exception("Validation failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/new-search", methods=["POST"])
@login_required
def api_new_search():
    """Initialize a new search session, clearing any previous state"""
    try:
        data = request.get_json(force=True) if request.is_json else {}
        search_session_id = data.get("search_session_id", "default")
        
        # Clear all session data related to this search session
        keys_to_clear = [k for k in session.keys() if search_session_id in k or k.startswith('product_type')]
        for key in keys_to_clear:
            del session[key]
        
        # Clear general workflow state for new search
        workflow_keys = ['current_step', 'current_intent', 'data']
        for key in workflow_keys:
            if key in session:
                del session[key]
        
        logging.info(f"[NEW_SEARCH] Initialized new search session: {search_session_id}")
        logging.info(f"[NEW_SEARCH] Cleared session keys: {keys_to_clear}")
        
        return jsonify({
            "success": True,
            "search_session_id": search_session_id,
            "message": "New search session initialized"
        }), 200
        
    except Exception as e:
        logging.exception("Failed to initialize new search session.")
        return jsonify({"error": str(e)}), 500


@app.route("/schema", methods=["GET"])
@login_required
def api_schema():
    if not components:
        return jsonify({"error": "Backend is not ready. LangChain failed."}), 503
    try:
        product_type = request.args.get("product_type", "").strip()
        
        if product_type:
            try:
                # Try to load from MongoDB with timeout protection
                schema_data = load_requirements_schema(product_type)
                
                # Check if schema is valid (not empty)
                if schema_data and (schema_data.get("mandatory_requirements") or schema_data.get("optional_requirements")):
                    logging.info(f"[SCHEMA] Successfully loaded schema for '{product_type}'")
                    return jsonify(convert_keys_to_camel_case(schema_data)), 200
                else:
                    logging.warning(f"[SCHEMA] Empty schema returned for '{product_type}', building from web")
                    # Fallback to web discovery if schema is empty
                    schema_data = build_requirements_schema_from_web(product_type)
                    return jsonify(convert_keys_to_camel_case(schema_data)), 200
                    
            except Exception as mongo_error:
                # MongoDB timeout or connection error - fallback to web-based schema
                logging.error(f"[SCHEMA] MongoDB error for '{product_type}': {str(mongo_error)}")
                logging.info(f"[SCHEMA] Falling back to web-based schema generation for '{product_type}'")
                
                try:
                    schema_data = build_requirements_schema_from_web(product_type)
                    return jsonify(convert_keys_to_camel_case(schema_data)), 200
                except Exception as web_error:
                    logging.error(f"[SCHEMA] Web-based schema generation also failed: {str(web_error)}")
                    # Return minimal schema to prevent complete failure
                    return jsonify({
                        "productType": product_type,
                        "mandatoryRequirements": {},
                        "optionalRequirements": {},
                        "error": f"Failed to load schema: {str(mongo_error)}"
                    }), 200  # Return 200 with error message instead of 500
        else:
            # No product type - return generic schema
            schema_data = load_requirements_schema()
            return jsonify(convert_keys_to_camel_case(schema_data)), 200
            
    except Exception as e:
        logging.exception("Schema fetch failed.")
        # Return minimal schema with error instead of failing completely
        return jsonify({
            "productType": product_type if 'product_type' in locals() else "",
            "mandatoryRequirements": {},
            "optionalRequirements": {},
            "error": str(e)
        }), 200  # Return 200 to prevent frontend from breaking

@app.route("/additional_requirements", methods=["POST"])
@login_required
def api_additional_requirements():
    if not components:
        return jsonify({"error": "Backend is not ready. LangChain failed."}), 503
    try:
        data = request.get_json(force=True)
        product_type = data.get("product_type", "").strip()
        user_input = data.get("user_input", "").strip()
        search_session_id = data.get("search_session_id", "default")

        
        if not product_type:
            return jsonify({"error": "Missing product_type"}), 400

        specific_schema = load_requirements_schema(product_type)
        
        # Add session isolation to prevent cross-contamination
        session_isolated_input = f"[Session: {search_session_id}] - This is an independent additional requirements request. User input: {user_input}"
        
        validation_result = components['additional_requirements_chain'].invoke({
            "user_input": session_isolated_input,
            "product_type": product_type,
            "schema": json.dumps(specific_schema, indent=2),
            "format_instructions": components['additional_requirements_format_instructions']
        })

        new_requirements = validation_result.get("provided_requirements", {})
        combined_reqs = new_requirements

        if combined_reqs:
            reqs_for_llm = '\n'.join([
                f"- {prettify_req(key)}: {value}" for key, value in combined_reqs.items()
            ])
            prompt = ChatPromptTemplate.from_template(
                """You are an expert assistant. For the following requirements for a {product_type}, provide a clear, user-friendly explanation of what they mean and why they are important. Use a conversational and encouraging tone.

                Requirements:
                {requirements}
                
                Provide your explanation in plain text format (not JSON).
                """
            )
            llm_chain = prompt | components['llm'] | StrOutputParser()
            explanation = llm_chain.invoke({
                "product_type": prettify_req(product_type),
                "requirements": reqs_for_llm
            })
        else:
            explanation = "I could not identify any specific requirements from your input."

        provided_requirements = new_requirements
        if new_requirements.get('mandatoryRequirements'):
            provided_requirements = {
                **new_requirements.get('mandatoryRequirements', {}),
                **new_requirements.get('optionalRequirements', {})
            }

        response_data = {
            "explanation": explanation,
            "providedRequirements": convert_keys_to_camel_case(provided_requirements),
        }


        return jsonify(response_data), 200

    except Exception as e:
        logging.exception("Additional requirements handling failed.")
        return jsonify({"error": str(e)}), 500

@app.route("/structure_requirements", methods=["POST"])
@login_required
def api_structure_requirements():
    if not components:
        return jsonify({"error": "Backend is not ready. LangChain failed."}), 503
    try:
        data = request.get_json(force=True)
        full_input = data.get("full_input", "")
        if not full_input:
            return jsonify({"error": "Missing full_input"}), 400

        structured_req = components['requirements_chain'].invoke({"user_input": full_input})
        return jsonify({"structured_requirements": structured_req}), 200
    except Exception as e:
        logging.exception("Requirement structuring failed.")
        return jsonify({"error": str(e)}), 500

@app.route("/api/advanced_parameters", methods=["POST"])
@login_required
def api_advanced_parameters():
    """
    Discovers latest advanced specifications with series numbers from top vendors for a product type
    """
    try:
        data = request.get_json(force=True)
        product_type = data.get("product_type", "").strip()
        search_session_id = data.get("search_session_id", "default")
        
        if not product_type:
            return jsonify({"error": "Missing 'product_type' parameter"}), 400

        # Store for logging (session-isolated)
        session[f'log_user_query_{search_session_id}'] = f"Latest advanced specifications for {product_type}"
        
        # Discover advanced specifications with series numbers
        logging.info(f"Starting latest advanced specifications discovery for: {product_type}")
        result = discover_advanced_parameters(product_type)
        
        # Log detailed information about filtering
        unique_count = len(result.get('unique_specifications', result.get('unique_parameters', [])))
        filtered_count = result.get('existing_specifications_filtered', result.get('existing_parameters_filtered', 0))
        total_found = unique_count + filtered_count
        
        logging.info(f"Advanced specifications discovery complete: {total_found} total specifications found, {filtered_count} filtered out (already in schema), {unique_count} new specifications returned")
        
        # Store result for logging
        session['log_system_response'] = result
        
        # Convert to camelCase for frontend
        camel_case_result = convert_keys_to_camel_case(result)
        
        logging.info(f"Advanced specifications discovery complete: {len(result.get('unique_specifications', result.get('unique_parameters', [])))} new specifications found (filtered out {result.get('existing_specifications_filtered', result.get('existing_parameters_filtered', 0))} existing specifications)")
        
        return jsonify(camel_case_result), 200

    except Exception as e:
        logging.exception("Advanced specifications discovery failed.")
        return jsonify({"error": str(e)}), 500

@app.route("/api/add_advanced_parameters", methods=["POST"])
@login_required
def api_add_advanced_parameters():
    """
    Processes user input for latest advanced specifications selection with series numbers
    """
    if not components:
        return jsonify({"error": "Backend is not ready. LangChain failed."}), 503
    try:
        data = request.get_json(force=True)
        product_type = data.get("product_type", "").strip()
        user_input = data.get("user_input", "").strip()
        available_parameters = data.get("available_parameters", [])

        if not product_type:
            return jsonify({"error": "Missing product_type"}), 400

        if not user_input:
            return jsonify({"error": "Missing user_input"}), 400

        # Use LLM to extract selected specifications from user input
        prompt = ChatPromptTemplate.from_template("""
You are Engenie - an expert assistant helping users select latest advanced specifications with series numbers for industrial equipment.

Product Type: {product_type}
Available Specifications: {available_parameters}
User Input: "{user_input}"

Extract the specifications the user wants to add from their input. The user might:
1. Select specific specifications by name
2. Say "all" or "everything" to select all specifications
3. Select categories or groups of specifications
4. Provide specific values for specifications

CRITICAL: Return ONLY valid JSON, no markdown, no explanations, no code blocks.

Return a JSON object with EXACTLY this structure:
{{
    "selected_parameters": {{"specification_name": "user_specified_value_or_empty_string"}},
    "explanation": "Brief explanation of what was selected"
}}

If the user didn't specify values, use empty strings for the specification values.
Only include specifications that are in the available specifications list.

Examples:
- "I want response_time and accuracy" → {{"selected_parameters": {{"response_time": "", "accuracy": ""}}, "explanation": "Selected response_time and accuracy"}}
- "Add all specifications" → Include all available specifications with empty values
- "Set accuracy to 0.1% and response_time to 1ms" → {{"selected_parameters": {{"accuracy": "0.1%", "response_time": "1ms"}}, "explanation": "Set accuracy to 0.1% and response_time to 1ms"}}

Remember: Return ONLY the JSON object, nothing else.
""")

        try:
            chain = prompt | components['llm'] | StrOutputParser()
            llm_response = chain.invoke({
                "product_type": product_type,
                "available_parameters": json.dumps(available_parameters),
                "user_input": user_input
            })

            # Parse the LLM response
            result = json.loads(llm_response)
            selected_parameters = result.get("selected_parameters", {})
            explanation = result.get("explanation", "Latest specifications selected successfully.")

        except (json.JSONDecodeError, Exception) as e:
            logging.warning(f"LLM parsing failed, using fallback: {e}")
            # Fallback: simple keyword matching
            selected_parameters = {}
            user_lower = user_input.lower()
            
            if "all" in user_lower or "everything" in user_lower:
                # Handle both dict format (new) and string format (old)
                for param in available_parameters:
                    param_key = param.get('key', param) if isinstance(param, dict) else param
                    selected_parameters[param_key] = ""
                explanation = "All available latest specifications have been selected."
            else:
                # Look for specification names in user input
                for param in available_parameters:
                    # Handle both dict format (new) and string format (old)
                    if isinstance(param, dict):
                        param_key = param.get('key', '')
                        param_name = param.get('name', '').lower()
                        if param_key.lower() in user_lower or param_name in user_lower:
                            selected_parameters[param_key] = ""
                    else:
                        if param.lower() in user_lower or param.replace('_', ' ').lower() in user_lower:
                            selected_parameters[param] = ""
                
                explanation = f"Selected {len(selected_parameters)} latest specifications based on your input."

        def wants_parameter_display(text: str) -> bool:
            lowered = text.lower()
            display_keywords = ["show", "display", "list", "see", "view", "what are"]
            spec_keywords = ["spec", "parameter", "latest"]
            return any(keyword in lowered for keyword in display_keywords) and any(
                key in lowered for key in spec_keywords
            )

        normalized_input = user_input.strip().lower()
        
        # Generate friendly response
        if selected_parameters:
            param_list = ", ".join([param.replace('_', ' ').title() for param in selected_parameters.keys()])
            friendly_response = f"Great! I've added these latest advanced specifications: {param_list}. Would you like to add any more advanced specifications?"
        else:
            if wants_parameter_display(normalized_input) and available_parameters:
                formatted_available = ", ".join(
                    [
                        (param.get("name") or param.get("key", "")).strip()
                        if isinstance(param, dict)
                        else str(param)
                        for param in available_parameters
                    ]
                )
                friendly_response = (
                    "Here are the latest advanced specifications you can add: "
                    f"{formatted_available}. Let me know the names you want to include "
                )
            else:
                friendly_response = "I didn't find any matching specifications in your input. Could you please specify which latest specifications you'd like to add?"

        response_data = {
            "selectedParameters": convert_keys_to_camel_case(selected_parameters),
            "explanation": explanation,
            "friendlyResponse": friendly_response,
            "totalSelected": len(selected_parameters)
        }

        return jsonify(response_data), 200

    except Exception as e:
        logging.exception("Latest advanced specifications addition failed.")
        return jsonify({"error": str(e)}), 500

@app.route("/analyze", methods=["POST"])
@login_required
def api_analyze():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Check if CSV vendors are provided for targeted analysis
        csv_vendors = data.get("csv_vendors", [])
        requirements = data.get("requirements", "")
        product_type = data.get("product_type", "")
        detected_product = data.get("detected_product", "")
        user_input = data.get("user_input")
        
        # Handle CSV vendor filtering for existing analysis process
        if csv_vendors and len(csv_vendors) > 0:
            logging.info(f"[CSV_VENDOR_FILTER] Applying CSV vendor filter with {len(csv_vendors)} vendors")
            
            # Standardize CSV vendor names for filtering
            csv_vendor_names = []
            for csv_vendor in csv_vendors:
                try:
                    original_name = csv_vendor.get("vendor_name", "")
                    standardized_name = standardize_vendor_name(original_name)
                    csv_vendor_names.append(standardized_name.lower())
                except Exception as e:
                    logging.warning(f"Failed to standardize CSV vendor {csv_vendor.get('vendor_name', '')}: {e}")
                    csv_vendor_names.append(csv_vendor.get("vendor_name", "").lower())
            
            # Store CSV filter in session for analysis chain to use
            session[f'csv_vendor_filter_{session.get("user_id", "default")}'] = {
                'vendor_names': csv_vendor_names,
                'csv_vendors': csv_vendors,
                'product_type': product_type,
                'detected_product': detected_product
            }
            
            logging.info(f"[CSV_VENDOR_FILTER] Applied filter for vendors: {csv_vendor_names}")
        
        # Handle regular analysis (with or without CSV vendor filtering)
        elif user_input is not None:
            # Ensure user_input is a dict
            if isinstance(user_input, str):
                try:
                    user_input = json.loads(user_input)
                except json.JSONDecodeError:
                    # fallback: wrap string into dict
                    user_input = {"raw_input": user_input}

            if not isinstance(user_input, dict):
                return jsonify({"error": "user_input must be a dict or JSON string representing a dict"}), 400

            # ===========================================================================
            # VENDOR PRIORITY: Extract specified_vendors from input or session
            # Priority: 1) From request data, 2) From session
            # ===========================================================================
            specified_vendors = data.get("specified_vendors", [])
            
            # If not in request, check session (set by identify-instruments or validation)
            if not specified_vendors:
                specified_vendors = session.get('specified_vendors', [])
                if specified_vendors:
                    logging.info(f"[VENDOR_PRIORITY] Retrieved specified_vendors from session: {specified_vendors}")
            
            # Also store in session for analysis chain to use (if provided in request)
            if specified_vendors:
                session['specified_vendors'] = specified_vendors
                logging.info(f"[VENDOR_PRIORITY] User-specified vendors for analysis: {specified_vendors}")
            
            # ===========================================================================
            # APPLICABLE STANDARDS: Extract standards from input for analysis
            # These come from the Standards RAG applied during instrument identification
            # ===========================================================================
            applicable_standards = data.get("applicable_standards", [])
            standards_specs = data.get("standards_specs", {})
            
            # If not in request, check session (set by search-vendors from instrument)
            if not applicable_standards:
                applicable_standards = session.get('applicable_standards', [])
                if applicable_standards:
                    logging.info(f"[STANDARDS_PRIORITY] Retrieved applicable_standards from session: {applicable_standards}")
            
            if not standards_specs:
                standards_specs = session.get('standards_specs', {})
                if standards_specs:
                    logging.info(f"[STANDARDS_PRIORITY] Retrieved standards_specs from session: {len(standards_specs)} specs")
            
            if applicable_standards:
                logging.info(f"[STANDARDS_PRIORITY] Applicable standards for analysis: {applicable_standards}")
            
            # Pass user_input, specified_vendors, AND applicable_standards to the analysis chain
            chain_input = {"user_input": user_input}
            if specified_vendors:
                chain_input["specified_vendors"] = specified_vendors
            if applicable_standards:
                chain_input["applicable_standards"] = applicable_standards
            if standards_specs:
                chain_input["standards_specs"] = standards_specs
                
            analysis_result = analysis_chain.invoke(chain_input)
        
        else:
            return jsonify({"error": "Missing 'user_input' parameter or CSV vendor data"}), 400
        
        # Apply standardization to the analysis result
        try:
            # Standardize vendor analysis if it exists
            if "vendor_analysis" in analysis_result:
                analysis_result["vendor_analysis"] = standardize_vendor_analysis_result(analysis_result["vendor_analysis"])
            
            # Standardize overall ranking if it exists
            if "overall_ranking" in analysis_result:
                analysis_result["overall_ranking"] = standardize_ranking_result(analysis_result["overall_ranking"])
                
            logging.info("Applied standardization to analysis result")
        except Exception as e:
            logging.warning(f"Standardization failed, proceeding with original result: {e}")

        camel_case_result = convert_keys_to_camel_case(analysis_result)

        # Store the analysis result as system response for logging
        session['log_system_response'] = analysis_result

        return jsonify(camel_case_result)

    except Exception as e:
        logging.error("Analysis failed.", exc_info=True)
        return jsonify({"error": str(e)}), 500



def match_user_with_pdf(user_input, pdf_data):
    """
    Matches user input fields with PDF data.
    Accepts user_input as a dict or JSON string.
    """
    # Ensure user_input is a dict
    if isinstance(user_input, str):
        try:
            user_input = json.loads(user_input)
        except json.JSONDecodeError:
            logging.warning("user_input is a string that cannot be parsed; wrapping in dict.")
            user_input = {"raw_input": user_input}

    if not isinstance(user_input, dict):
        raise ValueError("user_input must be a dict after parsing.")

    matched_results = {}
    for field, requirement in user_input.items():
        # Example matching logic; replace with your actual logic
        matched_results[field] = pdf_data.get(field, None)

    return matched_results

@app.route("/get_field_description", methods=["POST"])
@login_required
def api_get_field_description():
    if not components:
        return jsonify({"error": "Backend is not ready. LangChain failed."}), 503
    try:
        data = request.get_json(force=True)
        field = data.get("field", "").strip()
        product_type = data.get("product_type", "general").strip()

        if not field:
            return jsonify({"error": "Missing 'field' parameter."}), 400

        # Use the new schema description chain with Gemini 2.5 Flash Lite model
        description_response = components['schema_description_chain'].invoke({
            "field": field, 
            "product_type": product_type
        })

        return jsonify({"description": description_response}), 200

    except Exception as e:
        logging.exception("Failed to get field description from LLM.")
        return jsonify({"error": "Failed to get field description: " + str(e)}), 500

def get_submodel_to_model_series_mapping():
    """
    Creates a mapping from submodel names to their parent model series
    by scanning all vendor JSON files.
    """
    """Load submodel mapping from MongoDB instead of local files"""
    submodel_to_series = {}
    
    try:
        # Query MongoDB for all product data
        products_collection = mongodb_file_manager.collections.get('products')
        
        if not products_collection:
            logging.warning("Products collection not found in MongoDB")
            return submodel_to_series
        
        # Get all products from MongoDB
        cursor = products_collection.find({})
        
        for doc in cursor:
            try:
                # Extract product data
                if 'data' in doc:
                    data = doc['data']
                else:
                    data = {k: v for k, v in doc.items() if k not in ['_id', 'metadata']}
                
                # Process models and submodels
                models = data.get('models', [])
                for model in models:
                    model_series = model.get('model_series', '')
                    submodels = model.get('sub_models', [])
                    
                    for submodel in submodels:
                        submodel_name = submodel.get('name', '')
                        if submodel_name and model_series:
                            submodel_to_series[submodel_name] = model_series
                            
            except Exception as e:
                logging.warning(f"Failed to process MongoDB document: {e}")
                continue
                
    except Exception as e:
        logging.error(f"Failed to load submodel mapping from MongoDB: {e}")
        return submodel_to_series
                        
    logging.info(f"Generated submodel mapping with {len(submodel_to_series)} entries")
    return submodel_to_series

@app.route("/vendors", methods=["GET"])
@login_required
def get_vendors():
    """
    Get vendors with product images - ONLY for vendors in analysis results.
    Optimized to avoid unnecessary API calls.
    """
    try:
        # Get vendor list from query parameter (sent by frontend with analysis results)
        vendors_param = request.args.get('vendors', '')
        
        if vendors_param:
            # Use vendors from analysis results
            vendor_list = [v.strip() for v in vendors_param.split(',') if v.strip()]
            logging.info(f"Fetching images for {len(vendor_list)} vendors from analysis results: {vendor_list}")
        else:
            # Fallback: return empty list if no vendors specified
            logging.warning("No vendors specified in request, returning empty list")
            return jsonify({
                "vendors": [],
                "summary": {
                    "total_vendors": 0,
                    "total_images": 0,
                    "sources_used": {}
                }
            }), 200
        
        # Prepare vendor details map from session for better image search context
        vendor_details_map = {}
        try:
            analysis_result = session.get('log_system_response', {})
            # extract from ranked products if available
            ranking = analysis_result.get('overall_ranking') or analysis_result.get('overallRanking')
            if ranking and isinstance(ranking, dict):
                ranked_products = ranking.get('ranked_products') or ranking.get('rankedProducts', [])
                for prod in ranked_products:
                    v_name = prod.get('vendor')
                    if v_name:
                        key = v_name.strip().lower()
                        vendor_details_map[key] = {
                            'model_family': prod.get('model_family') or prod.get('modelFamily'),
                            'product_name': prod.get('product_name') or prod.get('productName')
                        }
        except Exception as e:
            logging.warning(f"Failed to build vendor details from session: {e}")

        vendors = []
        
        def process_vendor(vendor_name):
            """Process a single vendor synchronously for better reliability"""
            try:
                # Resolve context details
                details = vendor_details_map.get(vendor_name.strip().lower(), {})
                model_family = details.get('model_family')
                product_name = details.get('product_name')
                
                # Fetch product images using the 3-level fallback system (sync version)
                images, source_used = fetch_product_images_with_fallback_sync(
                    vendor_name=vendor_name,
                    model_family=model_family,
                    product_name=product_name
                )
                
                # Convert to expected format
                formatted_images = []
                for img in images:
                    # Create a normalized product key for frontend matching
                    title = img.get("title", "")
                    norm_key = re.sub(r"[\s_]+", "", title).replace("+", "").lower()
                    
                    formatted_images.append({
                        "fileName": title,
                        "url": img.get("url", ""),
                        "productKey": norm_key,
                        "thumbnail": img.get("thumbnail", ""),
                        "source": img.get("source", source_used),
                        "domain": img.get("domain", "")
                    })
                
                # Try to get logo from the first image or a specific logo search
                logo_url = None
                if formatted_images:
                    # Use first image as logo or search specifically for logo
                    logo_url = formatted_images[0].get("thumbnail") or formatted_images[0].get("url")
                
                vendor_data = {
                    "name": vendor_name,
                    "logoUrl": logo_url,
                    "images": formatted_images,
                    "source_used": source_used,
                    "image_count": len(formatted_images)
                }
                
                # Apply basic standardization to vendor data
                try:
                    vendor_data["name"] = standardize_vendor_name(vendor_data["name"])
                except Exception as e:
                    logging.warning(f"Failed to standardize vendor name {vendor_name}: {e}")
                    # Keep original name if standardization fails
                    vendor_data["name"] = vendor_name
                
                logging.info(f"Processed vendor {vendor_name}: {len(formatted_images)} images from {source_used}")
                return vendor_data
                
            except Exception as e:
                logging.warning(f"Failed to process vendor {vendor_name}: {e}")
                # Return minimal vendor data on failure
                return {
                    "name": vendor_name,
                    "logoUrl": None,
                    "images": [],
                    "source_used": "error",
                    "image_count": 0,
                    "error": str(e)
                }
        
        # Process only the vendors from analysis results
        for vendor_name in vendor_list:
            vendor_data = process_vendor(vendor_name)
            if vendor_data:
                vendors.append(vendor_data)
        
        # Filter out any None results and add summary info
        vendors = [v for v in vendors if v is not None]
        
        # Add summary statistics
        total_images = sum(v.get("image_count", 0) for v in vendors)
        sources_used = {}
        for v in vendors:
            source = v.get("source_used", "unknown")
            sources_used[source] = sources_used.get(source, 0) + 1
        
        response_data = {
            "vendors": vendors,
            "summary": {
                "total_vendors": len(vendors),
                "total_images": total_images,
                "sources_used": sources_used
            }
        }
        
        logging.info(f"Successfully processed {len(vendors)} vendors with {total_images} total images")
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"Critical error in get_vendors: {e}")
        return jsonify({
            "error": "Failed to fetch vendors",
            "vendors": [],
            "summary": {
                "total_vendors": 0,
                "total_images": 0,
                "sources_used": {}
            }
        }), 500

@app.route("/submodel-mapping", methods=["GET"])
@login_required
def get_submodel_mapping():
    """
    Returns the mapping from submodel names to model series names.
    This helps the frontend map analysis results (submodel names) to images (model series names).
    """
    try:
        mapping = get_submodel_to_model_series_mapping()
        
        # Skip LLM-based standardization for this endpoint to prevent connection issues
        # Basic mapping is sufficient for frontend functionality
        logging.info(f"Retrieved {len(mapping)} submodel mappings")
        
        return jsonify({"mapping": mapping})
    except Exception as e:
        logging.error(f"Error getting submodel mapping: {e}")
        return jsonify({"error": "Failed to get submodel mapping", "mapping": {}}), 500

@app.route("/admin/approve_user", methods=["POST"])
@login_required
def approve_user():
    admin_user = User.query.get(session['user_id'])
    if admin_user.role != "admin":
        return jsonify({"error": "Forbidden: Admins only"}), 403

    data = request.get_json()
    user_id = data.get("user_id")
    action = data.get("action", "approve")
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    if action == "approve":
        user.status = "active"
    elif action == "reject":
        user.status = "rejected"
    else:
        return jsonify({"error": "Invalid action"}), 400

    db.session.commit()
    return jsonify({"message": f"User {user.username} status updated to {user.status}."}), 200

@app.route("/admin/pending_users", methods=["GET"])
@login_required
def pending_users():
    admin_user = User.query.get(session['user_id'])
    if admin_user.role != "admin":
        return jsonify({"error": "Forbidden: Admins only"}), 403

    pending = User.query.filter_by(status="pending").all()
    result = [{
        "id": u.id,
        "username": u.username,
        "email": u.email,
        "first_name": u.first_name,
        "last_name": u.last_name
    } for u in pending]
    return jsonify({"pending_users": result}), 200

# Duplicated ALLOWED_EXTENSIONS and allowed_file, can be removed.
# ALLOWED_EXTENSIONS = {"pdf"}
# def allowed_file(filename: str):
#    ...

@app.route("/api/get-price-review", methods=["GET"])
@login_required
def api_get_price_review():
    """
    API endpoint to fetch price and reviews for a product.
    Takes 'productName', 'vendorName', 'modelFamily', 'subModel', 'productType' 
    and optionally 'userLocation' as query parameters.
    Returns JSON { productName: str, results: [...] }
    """
    product_name = request.args.get('productName')
    vendor_name = request.args.get('vendorName')
    model_family = request.args.get('modelFamily')
    sub_model = request.args.get('subModel')
    product_type = request.args.get('productType')
    user_location = request.args.get('userLocation')
    
    if not product_name:
        return jsonify({"error": "productName is required"}), 400

    results = fetch_price_and_reviews(
        product_name=product_name, 
        vendor_name=vendor_name,
        model_family=model_family,
        sub_model=sub_model,
        product_type=product_type,
        user_location=user_location
    )
    return jsonify({
        "productName": product_name,
        "results": results
    })

@app.route("/upload", methods=["POST"])
@login_required
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request. Expected field name 'file'."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed."}), 400

    filename = secure_filename(file.filename)

    try:
        # Read file into a BytesIO stream so it can be reused
        file_stream = BytesIO(file.read())

        # Extract text chunks from PDF
        text_chunks = extract_data_from_pdf(file_stream)
        raw_results = send_to_language_model(text_chunks)
        
        def flatten_results(results):
            flat = []
            for r in results:
                if isinstance(r, list): flat.extend(r)
                else: flat.append(r)
            return flat

        all_results = flatten_results(raw_results)
        final_result = aggregate_results(all_results, filename)
        
        # Apply standardization to the final result before splitting
        try:
            standardized_final_result = standardize_vendor_analysis_result(final_result)
            logging.info("Applied standardization to uploaded file analysis")
        except Exception as e:
            logging.warning(f"Failed to standardize uploaded file result: {e}")
            standardized_final_result = final_result
        
        split_results = split_product_types([standardized_final_result])

        saved_paths = []
        for result in split_results:
            # Save to MongoDB instead of local files
            vendor = (result.get("vendor") or "UnknownVendor").replace(" ", "_")
            product_type = (result.get("product_type") or "UnknownProduct").replace(" ", "_")
            model_series = (
                result.get("models", [{}])[0].get("model_series") or "UnknownModel"
            ).replace(" ", " ")
            
            try:
                product_metadata = {
                    'vendor_name': vendor,
                    'product_type': product_type,
                    'model_series': model_series,
                    'file_type': 'json',
                    'collection_type': 'products',
                    'path': f'vendors/{vendor}/{product_type}/{model_series}.json'
                }
                mongodb_file_manager.upload_json_data(result, product_metadata)
                saved_paths.append(f"MongoDB:vendors/{vendor}/{product_type}/{model_series}.json")
                print(f"[INFO] Stored product JSON to MongoDB: {vendor} - {product_type}")
            except Exception as e:
                logging.error(f"Failed to save product JSON to MongoDB: {e}")
            
            # Note: Product image extraction removed - now using API-based image search

        return jsonify({
            "data": split_results,
            "savedFiles": saved_paths
        })
    except Exception as e:
        logging.exception("File upload processing failed.")
        return jsonify({"error": str(e)}), 500

# =========================================================================
# === STANDARDIZATION ENDPOINTS ===
# === 
# === Integrated standardization functionality:
# === - /analyze endpoint: Standardizes vendor analysis and ranking results
# === - /vendors endpoint: Standardizes vendor names and product image mappings 
# === - /submodel-mapping endpoint: Enhances submodel mappings with standardization
# === - /upload endpoint: Standardizes analysis results from PDF uploads
# === - /api/upload_pdf_from_url endpoint: Standardizes analysis results from URL uploads
# === 
# === New standardization endpoints:
# === - GET /standardization/report: Generate comprehensive standardization report
# === - POST /standardization/update-files: Update existing files with standardization (admin only)
# === - POST /standardization/vendor-analysis: Standardize vendor analysis data
# === - POST /standardization/ranking: Standardize ranking data  
# === - POST /standardization/submodel-mapping: Enhance submodel mapping data
# =========================================================================

@app.route("/standardization/report", methods=["GET"])
@login_required
def get_standardization_report():
    """
    Generate and return a comprehensive standardization report
    """
    try:
        report = create_standardization_report()
        return jsonify(report), 200
    except Exception as e:
        logging.error(f"Failed to generate standardization report: {e}")
        return jsonify({"error": "Failed to generate standardization report"}), 500

@app.route("/standardization/update-files", methods=["POST"])
@login_required
def update_files_with_standardization():
    """
    Update existing vendor files with standardized naming
    """
    try:
        admin_user = User.query.get(session['user_id'])
        if admin_user.role != "admin":
            return jsonify({"error": "Forbidden: Admins only"}), 403
            
        updated_files = update_existing_vendor_files_with_standardization()
        return jsonify({
            "message": f"Successfully updated {len(updated_files)} files with standardization",
            "updated_files": updated_files
        }), 200
    except Exception as e:
        logging.error(f"Failed to update files with standardization: {e}")
        return jsonify({"error": "Failed to update files with standardization"}), 500

@app.route("/standardization/vendor-analysis", methods=["POST"])
@login_required
def standardize_vendor_analysis():
    """
    Standardize a vendor analysis result
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        analysis_result = data.get("analysis_result")
        if not analysis_result:
            return jsonify({"error": "Missing analysis_result parameter"}), 400
            
        standardized_result = standardize_vendor_analysis_result(analysis_result)
        return jsonify(standardized_result), 200
    except Exception as e:
        logging.error(f"Failed to standardize vendor analysis: {e}")
        return jsonify({"error": "Failed to standardize vendor analysis"}), 500

@app.route("/standardization/ranking", methods=["POST"])
@login_required
def standardize_ranking():
    """
    Standardize a ranking result
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        ranking_result = data.get("ranking_result")
        if not ranking_result:
            return jsonify({"error": "Missing ranking_result parameter"}), 400
            
        standardized_result = standardize_ranking_result(ranking_result)
        return jsonify(standardized_result), 200
    except Exception as e:
        logging.error(f"Failed to standardize ranking: {e}")
        return jsonify({"error": "Failed to standardize ranking"}), 500

@app.route("/standardization/submodel-mapping", methods=["POST"])
@login_required
def enhance_submodel_mapping_endpoint():
    """
    Enhance submodel to model series mapping with standardization
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        submodel_data = data.get("submodel_data")
        if not submodel_data:
            return jsonify({"error": "Missing submodel_data parameter"}), 400
            
        enhanced_result = enhance_submodel_mapping(submodel_data)
        return jsonify(enhanced_result), 200
    except Exception as e:
        logging.error(f"Failed to enhance submodel mapping: {e}")
        return jsonify({"error": "Failed to enhance submodel mapping"}), 500
    

# =========================================================================
# === PROJECT MANAGEMENT ENDPOINTS ===
# =========================================================================

@app.route("/api/projects/save", methods=["POST"])
@login_required
def save_project():
    """
    Save or update a project with all current state data using MongoDB
    """
    try:
        data = request.get_json(force=True)
        # Debug log incoming product_type information to trace saving issues (project save)
        try:
            incoming_pt = data.get('product_type') if isinstance(data, dict) else None
            incoming_detected = data.get('detected_product_type') if isinstance(data, dict) else None
            logging.info(f"[SAVE_PROJECT] Incoming product_type='{incoming_pt}' detected_product_type='{incoming_detected}' project_name='{data.get('project_name') if isinstance(data, dict) else None}' user_id={session.get('user_id')}")
        except Exception:
            logging.exception("Failed to log incoming project save payload")
        
        # Get current user ID
        user_id = str(session['user_id'])
        
        # Extract project data
        project_id = data.get("project_id")  # If updating existing project
        project_name = data.get("project_name", "").strip()
        
        if not project_name:
            return jsonify({"error": "Project name is required"}), 400
        
        # Check if initial_requirements is provided
        # Allow saving if project has instruments/accessories (already analyzed) even without requirements text
        has_requirements = bool(data.get("initial_requirements", "").strip())
        has_instruments = bool(data.get("identified_instruments") and len(data.get("identified_instruments", [])) > 0)
        has_accessories = bool(data.get("identified_accessories") and len(data.get("identified_accessories", [])) > 0)
        
        if not has_requirements and not has_instruments and not has_accessories:
            return jsonify({"error": "Initial requirements are required"}), 400
        
        # Save project to MongoDB using project manager
        # If the frontend provided a displayed_media_map, persist those images into GridFS
        try:
            displayed_media = data.get('displayed_media_map', {}) if isinstance(data, dict) else {}
            if displayed_media:
                from mongodb_utils import upload_to_mongodb
                # For each displayed media entry, fetch the URL and store bytes in GridFS
                for key, entry in displayed_media.items():
                    try:
                        top = entry.get('top_image') if isinstance(entry, dict) else None
                        vlogo = entry.get('vendor_logo') if isinstance(entry, dict) else None

                        def process_media(obj, subtype):
                            if not obj:
                                return None
                            url = obj.get('url') if isinstance(obj, dict) else (obj if isinstance(obj, str) else None)
                            if not url:
                                return None
                            # If url already references our API, skip re-upload
                            if url.startswith('/api/projects/file/'):
                                return url
                            # If it's a data URL, decode
                            if url.startswith('data:'):
                                import base64, re
                                m = re.match(r'data:(.*?);base64,(.*)', url)
                                if m:
                                    content_type = m.group(1)
                                    b = base64.b64decode(m.group(2))
                                    metadata = {'collection_type': 'documents', 'original_url': '', 'content_type': content_type}
                                    fid = upload_to_mongodb(b, metadata)
                                    return f"/api/projects/file/{fid}"
                                return None
                            # Otherwise attempt to download the URL
                            try:
                                resp = requests.get(url, timeout=8)
                                resp.raise_for_status()
                                content_type = resp.headers.get('Content-Type', 'application/octet-stream')
                                b = resp.content
                                metadata = {'collection_type': 'documents', 'original_url': url, 'content_type': content_type}
                                fid = upload_to_mongodb(b, metadata)
                                return f"/api/projects/file/{fid}"
                            except Exception as e:
                                logging.warning(f"Failed to fetch/displayed media URL {url}: {e}")
                                return None

                        new_top = process_media(top, 'top_image')
                        new_logo = process_media(vlogo, 'vendor_logo')

                        # Inject back into data so that stored project contains references to GridFS-served URLs
                        if new_top or new_logo:
                            # attempt to find product entries in data and replace matching keys
                            # The frontend sends a map keyed by `${vendor}-${productName}`; we'll store this map as `embedded_media`
                            if 'embedded_media' not in data:
                                data['embedded_media'] = {}
                            data['embedded_media'][key] = {}
                            if new_top:
                                data['embedded_media'][key]['top_image'] = {'url': new_top}
                            if new_logo:
                                data['embedded_media'][key]['vendor_logo'] = {'url': new_logo}
                    except Exception as e:
                        logging.warning(f"Error processing displayed_media_map entry {key}: {e}")
        except Exception as e:
            logging.warning(f"Failed to persist displayed_media_map: {e}")

        # Ensure pricing and feedback are passed through from frontend payload
        # If frontend uses `pricing` or `feedback_entries` include them in the saved document
        try:
            # If frontend supplied feedback, normalize to `feedback_entries`
            if 'feedback' in data and 'feedback_entries' not in data:
                data['feedback_entries'] = data.get('feedback')
        except Exception:
            logging.warning('Failed to normalize incoming feedback payload')

        saved_project = mongo_project_manager.save_project(user_id, data)

        # Store the saved project id in the session so future feedback posts can attach to it
        try:
            session['current_project_id'] = saved_project.get('project_id')
        except Exception:
            logging.warning('Failed to set current_project_id in session')
        
        # Return the saved project data
        return jsonify({
            "message": "Project saved successfully",
            "project": saved_project
        }), 200
        
    except ValueError as e:
        logging.warning(f"Project save validation error: {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.exception("Project save failed.")
        return jsonify({"error": "Failed to save project: " + str(e)}), 500


@app.route("/api/projects/preview-save", methods=["POST"])
@login_required
def preview_save_project():
    """
    Debug helper: compute resolved product_type (prefers detected_product_type)
    and return it without saving. Useful for quick verification.
    """
    try:
        data = request.get_json(force=True)
        project_name = (data.get('project_name') or '').strip()
        detected = data.get('detected_product_type')
        incoming = (data.get('product_type') or '').strip()

        if detected:
            resolved = detected.strip()
        else:
            if incoming and project_name and incoming.lower() == project_name.lower():
                resolved = ''
            else:
                resolved = incoming

        return jsonify({
            'resolved_product_type': resolved,
            'detected_product_type': detected,
            'incoming_product_type': incoming,
            'project_name': project_name
        }), 200
    except Exception as e:
        logging.exception('Preview save failed')
        return jsonify({'error': str(e)}), 500

@app.route("/api/projects", methods=["GET"])
@login_required
def get_user_projects():
    """
    Get all projects for the current user from MongoDB
    """
    try:
        user_id = str(session['user_id'])
        
        # Get all active projects for the user from MongoDB
        projects = mongo_project_manager.get_user_projects(user_id)
        
        return standardized_jsonify({
            "projects": projects,
            "total_count": len(projects)
        }, 200)
        
    except Exception as e:
        logging.exception("Failed to retrieve user projects.")
        return jsonify({"error": "Failed to retrieve projects: " + str(e)}), 500

@app.route("/api/projects/<project_id>", methods=["GET"])
@login_required
def get_project_details(project_id):
    """
    Get full project details for loading from MongoDB
    """
    try:
        user_id = str(session['user_id'])
        
        # Get project details from MongoDB
        project_details = mongo_project_manager.get_project_details(project_id, user_id)
        
        return standardized_jsonify({"project": project_details}, 200)
        
    except ValueError as e:
        logging.warning(f"Project access denied: {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.exception(f"Failed to retrieve project {project_id}.")
        return jsonify({"error": "Failed to retrieve project: " + str(e)}), 500


@app.route('/api/projects/file/<file_id>', methods=['GET'])
@login_required
def serve_project_file(file_id):
    """
    Serve a file stored in GridFS by its file ID.
    Returns raw bytes with appropriate content-type if available.
    """
    try:
        from bson import ObjectId
        from mongodb_utils import mongodb_file_manager

        try:
            oid = ObjectId(file_id)
        except Exception:
            return jsonify({'error': 'Invalid file id'}), 400

        grid_file = mongodb_file_manager.gridfs.get(oid)
        data = grid_file.read()
        content_type = grid_file.content_type if hasattr(grid_file, 'content_type') else grid_file.metadata.get('content_type') if getattr(grid_file, 'metadata', None) else 'application/octet-stream'

        return (data, 200, {
            'Content-Type': content_type,
            'Content-Length': str(len(data)),
            'Cache-Control': 'public, max-age=31536000'
        })
    except Exception as e:
        logging.exception('Failed to serve project file')
        return jsonify({'error': str(e)}), 500

# =========================================================================
# === PRODUCT INFO RAG ENDPOINTS ===
# =========================================================================

@app.route("/api/product-info/query", methods=["POST"])
@login_required
def api_product_info_query():
    """
    RAG-powered product information query endpoint.
    
    Flow:
    1. User asks a question
    2. System searches MongoDB for relevant data
    3. If found -> Returns answer based on database context
    4. If NOT found -> Asks user "Would you like me to get this from my knowledge?"
    5. User confirms with "yes" -> Returns LLM-generated answer
    
    Request Body:
    {
        "query": "What is the pressure range for Rosemount 3051?",
        "session_id": "optional_session_identifier"
    }
    
    Response:
    {
        "success": true,
        "answer": "Based on our database...",
        "source": "database" | "llm" | "pending_confirmation",
        "found_in_database": true | false,
        "awaiting_confirmation": true | false,
        "sources_used": ["vendors", "specs"]
    }
    """
    try:
        from product_info_rag import query_product_info
        
        data = request.get_json(force=True)
        query = data.get("query", "").strip()
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        # Use user_id as part of session identifier for isolated state
        user_id = session.get('user_id')  # Get numeric user_id
        user_id_str = str(user_id) if user_id else 'anonymous'
        session_id = data.get("session_id", f"user_{user_id_str}")
        
        logging.info(f"[PRODUCT_INFO_RAG] Query from user {user_id_str}: {query[:100]}...")
        
        # Execute RAG query with user_id for user-specific context (standards, strategy)
        result = query_product_info(query, session_id, user_id)
        
        logging.info(f"[PRODUCT_INFO_RAG] Source: {result.get('source')}, Found: {result.get('found_in_database')}, User context: {result.get('user_context_used', False)}")
        
        return jsonify({
            "success": True,
            "answer": result.get("answer", ""),
            "source": result.get("source", "unknown"),
            "found_in_database": result.get("found_in_database", False),
            "awaiting_confirmation": result.get("awaiting_confirmation", False),
            "sources_used": result.get("sources_used", []),
            "results_count": result.get("results_count", 0),
            "note": result.get("note", ""),
            "user_context_used": result.get("user_context_used", False)
        }), 200
        
    except Exception as e:
        logging.exception(f"[PRODUCT_INFO_RAG] Error processing query: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "answer": "I'm sorry, I encountered an error processing your question. Please try again."
        }), 500


@app.route("/api/product-info/clear-session", methods=["POST"])
@login_required
def api_product_info_clear_session():
    """
    Clear any pending confirmation state for a session.
    Call this when user navigates away or starts a new conversation.
    """
    try:
        from product_info_rag import clear_pending_query
        
        data = request.get_json(force=True) if request.data else {}
        user_id = str(session.get('user_id', 'anonymous'))
        session_id = data.get("session_id", f"user_{user_id}")
        
        clear_pending_query(session_id)
        
        return jsonify({"success": True, "message": "Session cleared"}), 200
        
    except Exception as e:
        logging.exception(f"[PRODUCT_INFO_RAG] Error clearing session: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/product-info/init", methods=["GET"])
@login_required
def api_product_info_init():
    """
    Initialize product info page with UI labels.
    All text is generated by LLM - no hardcoded messages in frontend.
    
    Response:
    {
        "success": true,
        "ui_labels": {
            "loading_text": "Searching database...",
            "confirmation_hint": "Type 'Yes' to get an answer from AI, or 'No' to skip",
            "input_placeholder": "Ask about products, vendors, or specifications...",
            "source_database": "From Database",
            "source_llm": "From AI Knowledge",
            "source_pending": "Awaiting Your Response",
            "error_message": "Sorry, something went wrong."
        }
    }
    """
    try:
        if not components or not components.get('llm'):
            # Fallback if LLM not available
            return jsonify({
                "success": True,
                "ui_labels": {
                    "loading_text": "Searching database...",
                    "confirmation_hint": "Type 'Yes' to get an answer from AI, or 'No' to skip",
                    "input_placeholder": "Ask about products, vendors, or specifications...",
                    "source_database": "From Database",
                    "source_llm": "From AI Knowledge",
                    "source_pending": "Awaiting Your Response",
                    "error_message": "Sorry, I encountered an error. Please try again."
                }
            }), 200
        
        # Generate UI labels using LLM
        # NOTE: Double curly braces escape the braces so LangChain doesn't treat them as template variables
        labels_prompt = """Generate short, user-friendly UI labels for a Product Info chat interface.

Return a JSON object with these keys (keep values short - max 5-6 words each):
{{
    "loading_text": "<text shown while searching database>",
    "confirmation_hint": "<hint when awaiting yes/no confirmation>",
    "input_placeholder": "<placeholder text for input field>",
    "source_database": "<label when answer is from database>",
    "source_llm": "<label when answer is from AI knowledge>",
    "source_pending": "<label when awaiting user confirmation>",
    "error_message": "<brief error message for failures>"
}}

Return ONLY the JSON object, no additional text."""

        labels_chain = ChatPromptTemplate.from_template(labels_prompt) | components['llm'] | StrOutputParser()
        labels_response = labels_chain.invoke({})
        
        # Parse labels
        try:
            cleaned_labels = labels_response.strip()
            if cleaned_labels.startswith("```json"):
                cleaned_labels = cleaned_labels[7:]
            elif cleaned_labels.startswith("```"):
                cleaned_labels = cleaned_labels[3:]
            if cleaned_labels.endswith("```"):
                cleaned_labels = cleaned_labels[:-3]
            cleaned_labels = cleaned_labels.strip()
            
            ui_labels = json.loads(cleaned_labels)
        except Exception as e:
            logging.warning(f"[PRODUCT_INFO] Failed to parse UI labels, using defaults: {e}")
            ui_labels = {
                "loading_text": "Searching database...",
                "confirmation_hint": "Type 'Yes' for AI answer, or 'No' to skip",
                "input_placeholder": "Ask about products, vendors, or specifications...",
                "source_database": "From Database",
                "source_llm": "From AI Knowledge",
                "source_pending": "Awaiting Your Response",
                "error_message": "Sorry, something went wrong. Please try again."
            }
        
        return jsonify({
            "success": True,
            "ui_labels": ui_labels
        }), 200
        
    except Exception as e:
        logging.exception(f"[PRODUCT_INFO] Error initializing: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "ui_labels": {
                "loading_text": "Searching...",
                "confirmation_hint": "Type Yes or No",
                "input_placeholder": "Ask a question...",
                "source_database": "From Database",
                "source_llm": "From AI",
                "source_pending": "Awaiting Response",
                "error_message": "Error occurred. Please try again."
            }
        }), 200

@app.route("/api/projects/<project_id>", methods=["DELETE"])
@login_required
def delete_project(project_id):
    """
    Permanently delete a project from MongoDB
    """
    try:
        user_id = str(session['user_id'])
        
        # Delete project from MongoDB
        mongo_project_manager.delete_project(project_id, user_id)
        
        return standardized_jsonify({"message": "Project deleted successfully"}, 200)
        
    except ValueError as e:
        logging.warning(f"Project delete access denied: {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.exception(f"Failed to delete project {project_id}.")
        return jsonify({"error": "Failed to delete project: " + str(e)}), 500


# =========================================================================
# === STANDARDS DOCUMENT API ENDPOINTS ===
# =========================================================================

@app.route("/api/standards-documents", methods=["GET"])
@login_required
def get_standards_documents():
    """
    Get all standards documents (files) for the current user from GridFS.
    Returns list of files with their metadata.
    """
    try:
        user_id = session.get('user_id')
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Get files from GridFS that belong to this user
        conn = get_mongodb_connection()
        db = conn['database']
        fs_files = db['fs.files']
        
        # Find all standards documents for this user
        cursor = fs_files.find({
            'metadata.collection_type': 'standards_documents',
            '$or': [
                {'metadata.uploaded_by_user_id': user_id},
                {'metadata.uploaded_by_username': user.username}
            ]
        }).sort('uploadDate', -1)
        
        documents = []
        for doc in cursor:
            documents.append({
                'file_id': str(doc['_id']),
                'filename': doc.get('filename', doc.get('metadata', {}).get('filename', 'Unknown')),
                'content_type': doc.get('metadata', {}).get('content_type', 'application/octet-stream'),
                'uploaded_at': doc.get('metadata', {}).get('uploaded_at', doc.get('uploadDate', '').isoformat() if doc.get('uploadDate') else ''),
                'size': doc.get('length', 0)
            })
        
        return jsonify({
            "success": True,
            "documents": documents,
            "total_count": len(documents)
        }), 200
        
    except Exception as e:
        logging.exception(f"Failed to get standards documents: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/standards-documents", methods=["DELETE"])
@login_required
def delete_all_standards_documents():
    """
    Delete all standards documents for the current user from MongoDB.
    """
    try:
        user_id = session.get('user_id')
        
        # Delete from MongoDB standards collection
        conn = get_mongodb_connection()
        standards_collection = conn['collections']['standards']
        result = standards_collection.delete_many({'user_id': user_id})
        
        return jsonify({
            "success": True,
            "message": f"Deleted {result.deleted_count} standards documents",
            "deleted_count": result.deleted_count
        }), 200
        
    except Exception as e:
        logging.exception(f"Failed to delete standards document reference: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload-standards-file", methods=["POST"])
@login_required
def upload_standards_file():
    """
    Upload a standards file to GridFS and extract text for RAG.
    Text is extracted and stored in MongoDB for category-based lookup.
    """
    try:
        user_id = session.get('user_id')
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        filename = secure_filename(file.filename)
        file_data = file.read()
        
        # Upload to GridFS for raw file storage
        metadata = {
            'filename': filename,
            'content_type': file.content_type,
            'uploaded_by_user_id': user_id,
            'uploaded_by_username': user.username,
            'collection_type': 'standards_documents',
            'uploaded_at': datetime.utcnow().isoformat()
        }
        file_id = mongodb_file_manager.upload_to_mongodb(file_data, metadata)
        
        logging.info(f"[STANDARDS_UPLOAD] Uploaded file to GridFS: {file_id}")
        
        # Extract text and store in background for RAG
        extract_and_store_standards_text_background(file_data, filename, user_id)
        
        return jsonify({
            "success": True,
            "message": "Standards file uploaded successfully. Text extraction in progress.",
            "file_id": file_id,
            "filename": filename,
            "size": len(file_data)
        }), 200
            
    except Exception as e:
        logging.exception(f"Standards file upload failed: {e}")
        return jsonify({"error": str(e)}), 500
        
@app.cli.command("init-db")
def init_db_command():
    """Creates the database tables and the default admin user."""
    db.create_all()
    if not User.query.filter_by(role='admin').first():
        hashed_pw = hash_password("Daman@123")  # Use an environment variable for this password in a real app
        admin = User(
                username="Daman", 
                email="reddydaman04@gmail.com", 
                password_hash=hashed_pw, 
                first_name="Daman",
                last_name="Reddy",
                status='active', 
                role='admin'
            )
        db.session.add(admin)
        db.session.commit()
        print("Admin user created with username 'Daman'.")
    else:
        print("Admin user already exists.")
    print("Database initialized.")
if __name__ == "__main__":
    create_db()
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port, threaded=True, use_reloader=False)
 
