import asyncio
from datetime import datetime
from unittest import result
from flask import Flask, request, jsonify, session, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import json
import logging
import re
import os
import urllib.parse
from werkzeug.utils import secure_filename
import tempfile
import requests
from io import BytesIO
from serpapi import GoogleSearch
import threading
import uuid

from functools import wraps
from dotenv import load_dotenv

# --- NEW IMPORTS FOR SEARCH FUNCTIONALITY ---
from googleapiclient.discovery import build

# --- NEW IMPORTS FOR AUTHENTICATION ---
from auth_models import db, User, Log
from auth_utils import hash_password, check_password

# --- LLM CHAINING IMPORTS ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from chaining import setup_langchain_components, create_analysis_chain
from loading import load_requirements_schema, build_requirements_schema_from_web
from flask import Flask, session
from flask_session import Session

# Import advanced parameters functionality
from advanced_parameters import discover_advanced_parameters

# Import MongoDB utilities
from mongodb_utils import get_schema_from_mongodb, get_json_from_mongodb
from mongodb_config import get_mongodb_connection
# Load environment variables
load_dotenv()

# =========================================================================
# === FLASK APP CONFIGURATION ===
# =========================================================================
app = Flask(__name__, static_folder="static")

# Manual CORS handling

# A list of allowed origins for CORS
allowed_origins = [
    "https://ai-product-recommender-ui.vercel.app",  # Your production frontend
    "http://localhost:8080",                         # Add your specific local dev port
    "http://localhost:5173",
    "http://localhost:3000"
]


# Replace your old CORS line with this one
CORS(app, origins=allowed_origins, supports_credentials=True)
logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO)

if os.getenv('FLASK_ENV') == 'production' or os.getenv('RAILWAY_ENVIRONMENT'):
    # Production session settings
    app.config["SESSION_PERMANENT"] = True
    app.config["SESSION_TYPE"] = "filesystem"
    app.config["SESSION_FILE_DIR"] = "/tmp/flask_session"
    app.config["SESSION_COOKIE_SECURE"] = True
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "None"
else:
    # Development session settings
    app.config["SESSION_PERMANENT"] = False
    app.config["SESSION_TYPE"] = "filesystem" 

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.getenv('SECRET_KEY', 'fallback-secret-key-for-development')
Session(app)
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


@app.route("/api/intent", methods=["POST"])
@login_required
def api_intent():
    """
    Classifies user intent and determines next workflow step based on input and current session state.
    Supports step-based workflow with session tracking for continuity.
    """
    if not components or not components.get("llm"):
        return jsonify({"error": "LLM component not ready."}), 503

    data = request.get_json(force=True)
    user_input = data.get("userInput", "").strip()
    if not user_input:
        return jsonify({"error": "userInput is required"}), 400

    # Get current workflow state from session
    current_step = session.get('current_step', None)
    current_intent = session.get('current_intent', None)
    
    # --- Handle skip for missing mandatory fields ---
    # Accept both legacy and frontend step names when user wants to skip missing mandatory fields
    if current_step in ("awaitMandatory", "awaitMissingInfo") and user_input.lower() in ["yes", "skip", "y"]:
        session['current_step'] = "awaitOptional"
        response = {
            "intent": "workflow",
            "nextStep": "awaitOptional",
            "resumeWorkflow": True,
            "message": "Skipping missing mandatory fields. Please provide optional requirements if any."
        }
        return jsonify(response), 200
    
    # --- Handle transition from advanced to summary ---
    if current_step == "awaitAdvanced" and user_input.lower() in ["no", "n", "skip", "done"]:
        session['current_step'] = "showSummary"
        response = {
            "intent": "workflow",
            "nextStep": "showSummary",
            "resumeWorkflow": True
        }
        return jsonify(response), 200
    
    # Check for greeting patterns (only standalone greetings)
    greeting_patterns = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
    is_greeting = any(user_input.lower().strip() == pattern or user_input.lower().strip() == pattern + '!' for pattern in greeting_patterns)
    
    # Check for knowledge questions patterns
    knowledge_indicators = ['what is', 'what are', 'how does', 'explain', 'tell me about', 'define']
    is_knowledge_question = any(indicator in user_input.lower() for indicator in knowledge_indicators)
    
    # Check for product requirements patterns
    product_indicators = [
        'pressure transmitter', 'temperature transmitter', 'temperature transmitter', 'flow meter',
        'need', 'looking for', 'want', 'require', 'specification', 'specs', 'mbar', 'bar', 'psi',
        'celsius', 'fahrenheit', 'voltage', 'current', 'output', 'range', 'accuracy', 'sensor'
    ]
    is_product_requirement = any(indicator in user_input.lower() for indicator in product_indicators) and len(user_input.split()) > 2

    # === Enhanced Prompt for Workflow-Aware Classification ===
    prompt = f"""
You are a smart assistant that classifies user input for a step-based product recommendation workflow.

Current workflow context:
- Current step: {current_step or "None"}
- Current intent: {current_intent or "None"}

User message: "{user_input}"

Return ONLY a JSON object with these keys:
1. "intent": one of ["greeting", "knowledgeQuestion", "productRequirements", "workflow", "chitchat", "other"]
2. "nextStep": one of ["greeting", "initialInput", "awaitOptional", "awaitAdvanced", "showSummary", "finalAnalysis", null]
3. "resumeWorkflow": true/false (whether to resume current workflow after handling this input)

Classification Rules:
- If user says ONLY greeting words (hi, hello, hey) with no technical content → intent="greeting", nextStep="greeting"
- If user asks knowledge questions (what is X, explain Y, how does Z work) → intent="knowledgeQuestion", nextStep=null, resumeWorkflow=true
- If user provides product requirements or specifications (mentions products, technical specs, measurements, needs) → intent="productRequirements", nextStep="initialInput"
- If currently in workflow and user provides relevant response → intent="workflow", nextStep=<appropriate next step>
- Casual conversation → intent="chitchat", nextStep=null
- Everything else → intent="other", nextStep=null

Priority: Product requirements should take precedence over greetings if technical content is present.

Next Step Logic:
- After greeting: "initialInput" (ask for product type)
- After product requirements: "initialInput" (validate requirements)  
- After optional requirements: "awaitAdvanced" (ask for advanced parameters)
- After advanced parameters: "showSummary" (show final summary)
- During workflow: determine based on current step progression
- Knowledge questions: maintain current step for resume

Workflow Steps:
greeting → initialInput → awaitOptional → awaitAdvanced → showSummary → finalAnalysis

Respond ONLY with valid JSON.
"""

    try:
        # Use LLM component for classification
        full_prompt = ChatPromptTemplate.from_template(prompt)
        response_chain = full_prompt | components['llm'] | StrOutputParser()
        llm_response = response_chain.invoke({"user_input": user_input})

        # Parse JSON safely
        try:
            result_json = json.loads(llm_response)
        except json.JSONDecodeError:
            # Improved fallback logic for classification
            if is_product_requirement:
                result_json = {"intent": "productRequirements", "nextStep": "initialInput", "resumeWorkflow": False}
            elif is_knowledge_question:
                result_json = {"intent": "knowledgeQuestion", "nextStep": None, "resumeWorkflow": True}
            elif is_greeting and not current_step:
                result_json = {"intent": "greeting", "nextStep": "greeting", "resumeWorkflow": False}
            else:
                result_json = {"intent": "other", "nextStep": None, "resumeWorkflow": False}

        # Update session based on classification
        if result_json.get("intent") == "greeting":
            session['current_step'] = 'greeting'
            session['current_intent'] = 'greeting'
        elif result_json.get("intent") == "productRequirements":
            session['current_step'] = 'initialInput'
            session['current_intent'] = 'productRequirements'
        elif result_json.get("intent") == "workflow" and result_json.get("nextStep"):
            session['current_step'] = result_json.get("nextStep")
            session['current_intent'] = 'workflow'
        # For knowledge questions, maintain current step for resumption
        # Don't update session for chitchat or other intents
        
        # Debug logging
        logging.info(f"Intent classification result: {result_json}")
        logging.info(f"Session updated - current_step: {session.get('current_step')}, current_intent: {session.get('current_intent')}")

        return jsonify(result_json), 200

    except Exception as e:
        logging.exception("Intent classification failed.")
        return jsonify({"error": str(e), "intent": "other", "nextStep": None, "resumeWorkflow": False}), 500

@app.route('/health', methods=['GET'])
@login_required

def health_check():
    return {
        "status": "healthy",
        "workflow_initialized": False,
        "langsmith_enabled": False
    }, 200
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
        data = request.get_json(force=True)
        step = data.get("step")
        data_context = data.get("dataContext", {})
        user_message = data.get("userMessage", "")
        intent = data.get("intent", "")

        # Get session state for workflow continuity
        current_step = session.get('current_step')
        current_intent = session.get('current_intent')
        
        # Handle knowledge questions - answer and resume workflow
        if intent == "knowledgeQuestion":
            # Determine context-aware response based on current workflow step
            if step == "awaitMissingInfo":
                context_hint = "Once you have the information you need, please provide the missing details so we can continue with your product selection or Would you like to continue anyway?"
            elif step == "awaitOptional":
                context_hint = "Now, let's continue - do you have any additional requirements to add?"
            elif step == "showSummary":
                context_hint = "Now, let's proceed with your product analysis."
            else:
                context_hint = "Now, let's continue with your product selection."
            
            prompt_template = f"""
You are an expert industrial sales consultant. The user asked a knowledge question: "{user_message}"

Provide a clear, professional answer about industrial products, processes, or terminology.
Keep your response informative but concise (2-3 sentences max).
After answering, smoothly transition back with: "{context_hint}"

Focus on being helpful while maintaining the conversation flow.
"""
            
            # Build and execute LLM chain
            full_prompt = ChatPromptTemplate.from_template(prompt_template)
            response_chain = full_prompt | components['llm'] | StrOutputParser()
            llm_response = response_chain.invoke({"user_input": user_message})
            
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
            prompt_template = f"""
You are a helpful sales agent. The user shared their requirements, and you identified the product type as '{product_type}'.
Your response must:
1. Start positively (e.g., "Great choice!" or similar).
2. Confirm the identified product type in a friendly way.
3. Ask if they have any other requirements.
"""
            next_step = "awaitOptional"
            
        elif step == 'awaitOptional':
            # Check if user wants to proceed to advanced parameters
            user_lower = user_message.lower().strip()
            
            # If user says no to optional requirements, automatically move to advanced parameters
            if user_lower in ['no', 'n', 'none', 'skip', 'done', 'proceed', 'ready']:
                product_type = data_context.get('productType', session.get('product_type', 'this product'))
                prompt_template = f"""
You are a helpful sales agent. The user indicated they're done with optional requirements.
Respond with: "Perfect! Would you like to add advanced parameters to enhance your selection for {product_type}?. I will get the parameters for you."
"""
                next_step = "awaitAdvanced"
            else:
                # User provided additional requirements - check if they seem ready for advanced parameters
                # Use LLM to decide if enough requirements have been gathered
                context_prompt = f"""
Analyze this user input in the context of product requirements gathering: "{user_message}"

The user has been adding optional requirements. Based on their input, should we:
1. Continue collecting more optional requirements (respond with "CONTINUE")
2. Move to advanced parameters discovery (respond with "ADVANCED")

Consider factors like:
- If the user seems satisfied with their current requirements
- If they're asking to proceed or finish
- If they've provided substantial additional details

Respond with only "CONTINUE" or "ADVANCED".
"""
                
                # Get LLM decision
                components_dict = setup_langchain_components()
                llm = components_dict['llm']
                output_parser = StrOutputParser()
                prompt = ChatPromptTemplate.from_template(context_prompt)
                chain = prompt | llm | output_parser
                decision = chain.invoke({}).strip().upper()
                
                if decision == "ADVANCED":
                    product_type = data_context.get('productType', session.get('product_type', 'this product'))
                    prompt_template = f"""
You are a helpful sales agent. The user provided additional requirements and seems ready to proceed.
Acknowledge their input and say: "Great! Now would you like to add advanced parameters to enhance your selection for {product_type}?. I will get the parameters for you."
"""
                    next_step = "awaitAdvanced"
                else:
                    prompt_template = """
You are a helpful sales agent. The user has provided additional information or requirements.
Acknowledge what they've shared and ask if there's anything else they'd like to add before moving forward.
Keep it friendly and concise.
"""
                    next_step = "awaitOptional"  # Stay in same step to allow more additions
        
        elif step == 'awaitAdvanced':
            # Handle advanced parameters step
            user_lower = user_message.lower().strip()
            
            # Get context data
            product_type = data_context.get('productType') or session.get('product_type')
            available_parameters = data_context.get('availableParameters', [])
            selected_parameters = data_context.get('selectedParameters', {})
            total_selected = data_context.get('totalSelected', 0)
            
            # Debug logging
            logging.info(f"awaitAdvanced - product_type: {product_type}")
            logging.info(f"awaitAdvanced - available_parameters: {available_parameters}")
            logging.info(f"awaitAdvanced - user_message: {user_message}")
            
            # Check if this is first time (no parameters discovered yet)
            if not available_parameters or len(available_parameters) == 0:
                # First time in awaitAdvanced - discover and show parameters
                logging.info(f"Discovering parameters for product_type: {product_type}")
                try:
                    if product_type:
                        # Discover advanced parameters
                        parameters_result = discover_advanced_parameters(product_type)
                        discovered_params = parameters_result.get('unique_parameters', [])[:15]
                        filtered_count = parameters_result.get('existing_parameters_filtered', 0)
                        
                        logging.info(f"Discovered {len(discovered_params)} new parameters: {discovered_params}")
                        if filtered_count > 0:
                            logging.info(f"Filtered out {filtered_count} parameters that were already in mandatory/optional requirements")
                        
                        # Store discovered parameters in session for future use
                        data_context['availableParameters'] = discovered_params
                        session['data'] = data_context
                        session.modified = True
                        
                        # Check if no new parameters were found
                        if len(discovered_params) == 0:
                            # When count is 0, ask if they want to proceed to summary only
                            if filtered_count > 0:
                                filter_info = f" All {filtered_count} potential advanced parameters were already covered in your mandatory/optional requirements."
                            else:
                                filter_info = " No new advanced parameters were found for this product type."
                            
                            prompt_template = f"""
You are a helpful sales agent. No new advanced parameters were discovered.
Respond with: "I couldn't find any new advanced parameters for your {product_type}.{filter_info}

Shall I proceed to the summary?"
"""
                        else:
                            # Format parameters for display when count > 0
                            params_list = []
                            for i, param in enumerate(discovered_params, 1):
                                # Handle both dict format (new) and string format (old)
                                if isinstance(param, dict):
                                    formatted_param = param.get('name', param.get('key', '').replace('_', ' ').title())
                                else:
                                    formatted_param = param.replace('_', ' ').title()
                                params_list.append(f"{i}. **{formatted_param}**")
                            
                            params_display = "\n".join(params_list)
                            
                            # Create message with filtering information
                            if filtered_count > 0:
                                filter_info = f" (found {len(discovered_params) + filtered_count} total, filtered out {filtered_count} already specified)"
                            else:
                                filter_info = ""
                            
                            prompt_template = f"""
You are a helpful sales agent. The user is ready for advanced parameters.
Respond with: "Great! I discovered **{len(discovered_params)} new advanced parameters**:

{params_display}

**Would you like to add advanced parameters?**
"
"""
                    else:
                        # Mark as error case for later handling
                        data_context['parameterError'] = True
                        session['data'] = data_context
                        session.modified = True
                        
                        prompt_template = """
You are a helpful sales agent. There was an issue discovering advanced parameters.
Respond with: "I'm having trouble accessing advanced parameters. Would you like to skip this step?"
"""
                except Exception as e:
                    # Mark as error case for later handling
                    data_context['parameterError'] = True
                    session['data'] = data_context
                    session.modified = True
                    
                    prompt_template = """
You are a helpful sales agent. There was an error discovering advanced parameters.
Respond with: "I encountered an issue discovering advanced parameters. Would you like to skip this step?"
"""
                next_step = "awaitAdvanced"
            else:
                # Parameters already discovered - handle user response
                parameter_error = data_context.get('parameterError', False)
                
                # If there was an error and user says yes to skip
                if parameter_error and user_lower in ['yes', 'y', 'yeah', 'yep', 'sure', 'ok', 'okay']:
                    prompt_template = """
You are a helpful sales agent. The user chose to skip advanced parameters.
Respond with: "No problem! Let me proceed to the summary of your requirements."
"""
                    next_step = "showSummary"
                # If user says no to adding parameters (normal flow)
                elif user_lower in ['no', 'n', 'skip', 'none', 'not needed', 'done']:
                    prompt_template = """
You are a helpful sales agent. The user chose to skip advanced parameters.
Respond with: "No problem! Let me proceed to the summary of your requirements."
"""
                    next_step = "showSummary"
                # If user says yes to adding parameters (normal flow)
                elif user_lower in ['yes', 'y', 'yeah', 'yep', 'sure'] and not parameter_error:
                    prompt_template = """
You are a helpful sales agent. The user wants to add advanced parameters.
Respond with: "Great! Please tell me which parameters you'd like to add. You can specify by name or say 'all' to add all of them."
"""
                    next_step = "awaitAdvanced"
                # User provided parameter selections
                elif total_selected > 0:
                    # Parameters were selected
                    selected_names = [param.replace('_', ' ').title() for param in selected_parameters.keys()]
                    selected_display = ", ".join(selected_names)
                    
                    prompt_template = f"""
You are a helpful sales agent. The user selected advanced parameters.
Respond with: "**Added Advanced Parameters:** {selected_display}

Would you like to add more parameters, or shall I proceed to the summary?"
"""
                    next_step = "awaitAdvanced"
                else:
                    # No parameters matched or user provided other input
                    prompt_template = """
You are a helpful sales agent. The user's parameter selection didn't match any available options.
Respond with: "I didn't find any matching parameters. Could you please specify which parameters you'd like to add, or say 'no' to skip?"
"""
                    next_step = "awaitAdvanced"
            
        elif step == 'confirmAfterMissingInfo':
            prompt_template = """
You are a helpful sales assistant. The user just provided the last piece of required information.

In a single, encouraging sentence, ask if they have any other optional details or preferences to add before you finalize things.
"""
            next_step = "awaitOptional"
            
        elif step == 'showSummary':
            # Check if user is confirming to proceed with analysis
            user_lower = user_message.lower().strip()
            if user_lower in ['yes', 'y', 'proceed', 'continue', 'run', 'analyze', 'ok', 'okay']:
                prompt_template = """
You are a helpful sales agent. The user confirmed they want to proceed with the analysis.
Respond with: "Excellent! Starting the product analysis now..."
"""
                next_step = "finalAnalysis"
            else:
                # First time showing summary or user asking for clarification
                prompt_template = """
You are a helpful and friendly sales agent.
Your response must ONLY be:
"It sounds like you're ready to move forward. Here's a quick summary of what you have provided:"
Do NOT add any summary details, bullet points, or extra text. Just return this exact friendly sentence.
"""
                next_step = "finalAnalysis"
            
        elif step == 'finalAnalysis':
            ranked_products = data_context.get('analysisResult', {}).get('overallRanking', {}).get('rankedProducts', [])
            matching_products = [p for p in ranked_products if p.get('requirementsMatch') is True]
            count = len(matching_products)
            prompt_template = f"""
You are a helpful sales agent. The analysis is done and {count} products were found.
Tell the user the number of products and let them know they can check the right panel for details.
Keep it clear and professional.
"""
            next_step = None  # End of workflow
            
        elif step == 'analysisError':
            prompt_template = "You are a helpful sales agent. An error happened during the analysis. Apologize and politely ask the user to type 'rerun' to try again."
            next_step = "showSummary"  # Allow retry from summary
            
        elif step == 'default':
            prompt_template = "You are a helpful sales agent. Reply to the user's message in a simple, friendly way and keep the conversation moving forward."
            next_step = current_step or None
            
        # === NEW WORKFLOW STEPS (Added for enhanced functionality) ===
        elif step == 'greeting':
            prompt_template = """
You are a friendly and professional industrial sales consultant.
Greet the user warmly and ask them what type of industrial product they are looking for.
Mention that you can help them find the right solution from various manufacturers.
Keep it concise and welcoming.
"""
            next_step = "initialInput"
            
        else:
            # Default fallback for unrecognized steps
            prompt_template = f"""
You are a helpful sales agent. Reply to the user's message: "{user_message}" in a simple, friendly way and keep the conversation moving forward.
"""
            next_step = current_step or "greeting"

        # --- Build Chain and Generate Response ---
        if prompt_template.strip():
            full_prompt = ChatPromptTemplate.from_template(prompt_template)
            response_chain = full_prompt | components['llm'] | StrOutputParser()
            llm_response = response_chain.invoke({"user_input": user_message})
        else:
            llm_response = ""

        # Update session with new step
        if next_step:
            session['current_step'] = next_step
            session['current_intent'] = 'workflow'

        # Prepare response
        response_data = {
            "content": llm_response,
            "nextStep": next_step
        }

        # Store the sales agent response as system response for logging

        return jsonify(response_data), 200

    except Exception as e:
        logging.exception("Sales agent response generation failed.")
        return jsonify({
            "error": "Failed to generate response: " + str(e),
            "content": "I apologize, but I'm having technical difficulties. Please try again.",
            "nextStep": session.get('current_step')
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

        # 4. Create the new Log object based on the updated schema
        new_log = Log(
            user_name=username,
            user_query=user_query,
            system_response=json.dumps(system_response),
            feedback=feedback_log_entry
        )

        # 5. Add to the session and commit to the database
        db.session.add(new_log)
        db.session.commit()

        # 6. Log to console *after* successful commit to confirm it's saved
        logging.info(f"Successfully saved log entry with ID: {new_log.id} for user: '{username}' to the database.")
        
        # 7. Clean up the session
        session.pop('log_user_query', None)
        session.pop('log_system_response', None)
        
        # --- LOGGING LOGIC ENDS ---

        if not feedback_type and not comment:
            return jsonify({"error": "No feedback provided."}), 400
        
        # --- LLM RESPONSE GENERATION ---
        if feedback_type == 'positive':
            prompt_template = """
You are a helpful assistant. The user provided positive feedback on the recent analysis.
If they left a comment, it is: '{comment}'.
Please respond warmly and thank them for their time and input. Keep the response to a single, friendly sentence.
"""
        elif feedback_type == 'negative':
            prompt_template = """
You are a helpful assistant. The user provided negative feedback on the recent analysis.
If they left a comment, it is: '{comment}'.
Respond with empathy. Acknowledge their feedback, apologize for the inconvenience, and state that their input is valuable and will be used to make improvements. Keep it to one or two professional sentences.
"""
        else:  # This handles the case where only a comment is provided
            prompt_template = """
You are a helpful assistant. The user provided the following feedback on the analysis: '{comment}'.
Acknowledge their comment and thank them for taking the time to provide their input.
"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        feedback_chain = prompt | components['llm'] | StrOutputParser()
        llm_response = feedback_chain.invoke({"comment": comment})

        return jsonify({"response": llm_response}), 200

    except Exception as e:
        logging.exception("Feedback handling or database logging failed.")
        db.session.rollback() # Important: Roll back the transaction if an error occurs
        return jsonify({"error": "Failed to process feedback: " + str(e)}), 500

# =========================================================================
# === INSTRUMENT IDENTIFICATION ENDPOINT ===
# =========================================================================
@app.route("/api/identify-instruments", methods=["POST"])
@login_required
def identify_instruments():
    """
    Identifies instruments from user requirements using LLM.
    Returns a list of identified instruments with their specifications and sample inputs.
    """
    if not components or not components.get('llm'):
        return jsonify({"error": "LLM component is not ready."}), 503

    try:
        data = request.get_json(force=True)
        requirements = data.get("requirements", "").strip()
        
        if not requirements:
            return jsonify({"error": "Requirements text is required"}), 400

        # Create the prompt for instrument identification
        prompt_template = """
You are an expert in Industrial Process Control Systems. Analyze the following requirements and identify the Bill of Materials (instruments) needed.

Requirements:
{requirements}

Instructions:
1. Identify all instruments required for the given Industrial Process Control System Problem Statement
2. For each instrument, provide:
   - Category (e.g., Pressure Transmitter, Temperature Transmitter, Flow Meter, etc.)
   - Product Name (generic name based on the requirements)
   - Specifications (extract from requirements or infer based on industry standards)
   - Sample Input (create a detailed sample input that can be used for product recommendation)

3. Mark inferred requirements explicitly with [INFERRED] tag

Additionally, identify any accessories, consumables, or ancillary items required to support the instruments (for example: impulse lines, mounting brackets, isolation valves, manifolds, cable/connector types, junction boxes, power supplies, or calibration kits). For accessories, provide:
    - Category (e.g., Impulse Line, Isolation Valve, Mounting Bracket, Junction Box)
    - Accessory Name (generic)
    - Specifications (size, material, pressure rating, connector type, etc.)
    - Sample Input (a short sentence describing the accessory need)

Return ONLY a valid JSON object with this structure:
{{
  "instruments": [
    {{
      "category": "<category>",
      "product_name": "<product name>",
      "specifications": {{
        "<spec_field>": "<spec_value>",
        "<spec_field>": "<spec_value>",
        "<spec_field>": "<spec_value>"
      }},
      "sample_input": "I need a <category> with <key specifications>"
    }}
  ],
    "accessories": [
        {{
            "category": "<accessory category>",
            "accessory_name": "<accessory name>",
            "specifications": {{
                "<spec_field>": "<spec_value>"
            }},
            "sample_input": "I need a <accessory category> for <instrument or purpose> with <key specs>"
        }}
    ],
  "summary": "Brief summary of identified instruments"
}}

Respond ONLY with valid JSON, no additional text.
"""

        # Build and execute LLM chain
        full_prompt = ChatPromptTemplate.from_template(prompt_template)
        response_chain = full_prompt | components['llm'] | StrOutputParser()
        llm_response = response_chain.invoke({"requirements": requirements})

        # Clean the LLM response - remove markdown code blocks if present
        cleaned_response = llm_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]  
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]  
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]  
        cleaned_response = cleaned_response.strip()

        # Parse JSON response
        try:
            result = json.loads(cleaned_response)
            
            # Validate the response structure
            if "instruments" not in result or not isinstance(result["instruments"], list):
                raise ValueError("Invalid response structure from LLM")
            
            # Ensure all instruments have required fields
            for instrument in result["instruments"]:
                if not all(key in instrument for key in ["category", "product_name", "specifications", "sample_input"]):
                    raise ValueError("Missing required fields in instrument data")

            
            # Optional: validate accessories if present
            if "accessories" in result:
                if not isinstance(result["accessories"], list):
                    raise ValueError("'accessories' must be a list if provided")
                for accessory in result["accessories"]:
                    # expected keys for accessories
                    expected_acc_keys = ["category", "accessory_name", "specifications", "sample_input"]
                    if not all(key in accessory for key in expected_acc_keys):
                        raise ValueError("Missing required fields in accessory data")
            
            return standardized_jsonify(result, 200)
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM response as JSON: {e}")
            logging.error(f"LLM Response: {llm_response}")
            
            # Fallback: Try to extract instruments from text response
            return jsonify({
                "error": "Failed to parse instrument identification",
                "instruments": [],
                "summary": "Unable to identify instruments from the provided requirements"
            }), 500

    except Exception as e:
        logging.exception("Instrument identification failed.")
        return jsonify({
            "error": "Failed to identify instruments: " + str(e),
            "instruments": [],
            "summary": ""
        }), 500

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


def fetch_price_and_reviews_serpapi(product_name: str):
    """Use SerpApi to fetch price and review info for a product."""
    if not SERPAPI_KEY:
        return []
    
    try:
        search = GoogleSearch({
            "q": f"{product_name} price review",
            "api_key": SERPAPI_KEY,
            "num": 10
        })
        res = search.get_dict()
        results = []

        for item in res.get("organic_results", []):
            snippet = item.get("snippet", "")
            price = None
            reviews = None
            source = item.get("source")
            link = item.get("link")

            # Try to pull price from structured extensions
            ext = (
                item.get("rich_snippet", {})
                    .get("bottom", {})
                    .get("detected_extensions", {})
            )
            if "price" in ext:
                price = f"${ext['price']}"
            elif "price_from" in ext and "price_to" in ext:
                price = f"${ext['price_from']} to ${ext['price_to']}"
            else:
                # Fallback: regex on snippet
                price_match = re.search(r"([$₹€£¥])\s?\d+(?:[.,]\d+)?", snippet)
                if price_match:
                    price = price_match.group(0)

            # Extract reviews (look in snippet)
            review_match = re.search(r"(\d(?:\.\d)?)\s?out of 5", snippet)
            if review_match:
                reviews = float(review_match.group(1))

            if price or reviews or source or link:
                results.append({
                    "price": price,
                    "reviews": reviews,
                    "source": source,
                    "link": link
                })

        return results

    except Exception as e:
        print(f"[WARNING] SerpAPI price/review search failed for {product_name}: {e}")
        return []


def fetch_price_and_reviews_serper(product_name: str):
    """Use Serper API to fetch price and review info for a product."""
    if not SERPER_API_KEY:
        return []
    
    try:
        url = "https://google.serper.dev/search"
        payload = {
            "q": f"{product_name} price review",
            "num": 10
        }
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = []

        for item in data.get("organic", []):
            snippet = item.get("snippet", "")
            price = None
            reviews = None
            source = item.get("displayLink")
            link = item.get("link")

            # Extract price from snippet using regex
            price_match = re.search(r"([$₹€£¥])\s?\d+(?:[.,]\d+)?", snippet)
            if price_match:
                price = price_match.group(0)

            # Extract reviews (look in snippet)
            review_match = re.search(r"(\d(?:\.\d)?)\s?out of 5", snippet)
            if review_match:
                reviews = float(review_match.group(1))

            if price or reviews or source or link:
                results.append({
                    "price": price,
                    "reviews": reviews,
                    "source": source,
                    "link": link
                })

        return results

    except Exception as e:
        print(f"[WARNING] Serper API price/review search failed for {product_name}: {e}")
        return []


def fetch_price_and_reviews_google_custom(product_name: str):
    """Use Google Custom Search to fetch price and review info for a product as fallback."""
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
                    q=f"{product_name} price review",
                    cx=GOOGLE_CSE_ID,
                    num=10
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
        results = []

        for item in items:
            snippet = item.get("snippet", "")
            price = None
            reviews = None
            source = item.get("displayLink")
            link = item.get("link")

            # Extract price using regex
            price_match = re.search(r"([$₹€£¥])\s?\d+(?:[.,]\d+)?", snippet)
            if price_match:
                price = price_match.group(0)

            # Extract reviews
            review_match = re.search(r"(\d(?:\.\d)?)\s?out of 5", snippet)
            if review_match:
                reviews = float(review_match.group(1))

            if price or reviews or source or link:
                results.append({
                    "price": price,
                    "reviews": reviews,
                    "source": source,
                    "link": link
                })

        return results

    except Exception as e:
        print(f"[WARNING] Google Custom Search price/review search failed for {product_name}: {e}")
        return []


def fetch_price_and_reviews(product_name: str):
    """
    Fetch price and review info using SERP API first, then Serper API, then Google Custom Search as final fallback.
    Special order for pricing: SERP API → Serper → Google Custom Search
    Returns a structured response with results and metadata.
    """
    # First, try SERP API (special order for pricing)
    serpapi_results = fetch_price_and_reviews_serpapi(product_name)
    
    # If SERP API returns results, use them
    if serpapi_results:
        return {
            "productName": product_name, 
            "results": serpapi_results,
            "source_used": "serpapi",
            "fallback_used": False
        }
    
    # If SERP API fails or returns no results, try Serper API
    serper_results = fetch_price_and_reviews_serper(product_name)
    
    if serper_results:
        return {
            "productName": product_name, 
            "results": serper_results,
            "source_used": "serper",
            "fallback_used": True
        }
    
    # If both SERP API and Serper fail or return no results, try Google Custom Search
    google_results = fetch_price_and_reviews_google_custom(product_name)
    
    if google_results:
        return {
            "productName": product_name, 
            "results": google_results,
            "source_used": "google_custom",
            "fallback_used": True
        }
    
    # If all three fail, return empty results
    return {
        "productName": product_name, 
        "results": [],
        "source_used": "none",
        "fallback_used": True
    }


# =========================================================================
# === IMAGE SEARCH FUNCTIONS ===
# =========================================================================

# Common manufacturer domains for image search
MANUFACTURER_DOMAINS = [
    "emerson.com", "yokogawa.com", "siemens.com", "abb.com", "honeywell.com",
    "schneider-electric.com", "ge.com", "rockwellautomation.com", "endress.com",
    "fluke.com", "krohne.com", "rosemount.com", "fisher.com", "metso.com",
    "valmet.com", "foxboro.com", "invensys.com", "triconex.com", "deltaV.com",
    "dcs.com", "hima.com", "sis.com", "panalytical.com", "thermo.com",
    "agilent.com", "waters.com", "perkinelmer.com", "shimadzu.com", "bruker.com",
    "varian.com", "beckman.com", "bio-rad.com", "teledyne.com", "omega.com",
    "ni.com", "keysight.com", "tektronix.com", "rohde-schwarz.com", "anritsu.com"
]

def fetch_images_google_cse_sync(vendor_name: str, product_name: str = None):
    """
    Synchronous version: Google Custom Search API for images from manufacturer domains
    """
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        logging.warning("Google CSE credentials not available for image search")
        return []
    
    try:
        query = f"{vendor_name}"
        if product_name:
            query += f" {product_name}"
        
        # Build site restriction for manufacturer domains
        domain_filter = " OR ".join([f"site:{domain}" for domain in MANUFACTURER_DOMAINS])
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
        for item in result.get("items", []):
            images.append({
                "url": item.get("link"),
                "title": item.get("title", ""),
                "source": "google_cse",
                "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                "domain": item.get("displayLink", "")
            })
        
        if images:
            logging.info(f"Google CSE found {len(images)} images for {vendor_name}")
        return images
        
    except Exception as e:
        logging.warning(f"Google CSE image search failed for {vendor_name}: {e}")
        return []

def fetch_images_serpapi_sync(vendor_name: str, product_name: str = None):
    """
    Synchronous version: SerpAPI fallback for Google Images
    """
    if not SERPAPI_KEY:
        logging.warning("SerpAPI key not available for image search")
        return []
    
    try:
        query = f"{vendor_name}"
        if product_name:
            query += f" {product_name}"
        query += " industrial product"
        
        search = GoogleSearch({
            "q": query,
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

def fetch_images_serper_sync(vendor_name: str, product_name: str = None):
    """
    Synchronous version: Serper.dev fallback for images
    """
    if not SERPER_API_KEY_IMAGES:
        logging.warning("Serper API key not available for image search")
        return []
    
    try:
        query = f"{vendor_name}"
        if product_name:
            query += f" {product_name}"
        query += " industrial equipment"
        
        url = "https://google.serper.dev/images"
        payload = {
            "q": query,
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

def fetch_product_images_with_fallback_sync(vendor_name: str, product_name: str = None):
    """
    Synchronous 3-level image search fallback system
    1. Google Custom Search API (manufacturer domains)
    2. SerpAPI Google Images
    3. Serper.dev images
    """
    logging.info(f"Starting image search for vendor: {vendor_name}, product: {product_name}")
    
    # Step 1: Try Google Custom Search API
    images = fetch_images_google_cse_sync(vendor_name, product_name)
    if images:
        logging.info(f"Using Google CSE images for {vendor_name}")
        return images, "google_cse"
    
    # Step 2: Try SerpAPI
    images = fetch_images_serpapi_sync(vendor_name, product_name)
    if images:
        logging.info(f"Using SerpAPI images for {vendor_name}")
        return images, "serpapi"
    
    # Step 3: Try Serper.dev
    images = fetch_images_serper_sync(vendor_name, product_name)
    if images:
        logging.info(f"Using Serper images for {vendor_name}")
        return images, "serper"
    
    # All failed
    logging.warning(f"All image search APIs failed for {vendor_name}")
    return [], "none"

def fetch_vendor_logo_sync(vendor_name: str):
    """
    Specialized function to fetch vendor logo
    """
    logging.info(f"Fetching logo for vendor: {vendor_name}")
    
    # Try different logo-specific searches
    logo_queries = [
        f"{vendor_name} logo",
        f"{vendor_name} company logo", 
        f"{vendor_name} brand",
        f"{vendor_name}"
    ]
    
    for query in logo_queries:
        try:
            # Use Google CSE first for official logos
            if GOOGLE_API_KEY and GOOGLE_CX:
                # Build site restriction for manufacturer domains
                domain_filter = " OR ".join([f"site:{domain}" for domain in MANUFACTURER_DOMAINS])
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
                        return {
                            "url": logo_url,
                            "thumbnail": item.get("image", {}).get("thumbnailLink", logo_url),
                            "source": "google_cse_logo",
                            "title": item.get("title", ""),
                            "domain": item.get("displayLink", "")
                        }
                
                # If no specific logo found, use first result from official domain
                if result.get("items"):
                    item = result["items"][0]
                    return {
                        "url": item.get("link"),
                        "thumbnail": item.get("image", {}).get("thumbnailLink", item.get("link")),
                        "source": "google_cse_general",
                        "title": item.get("title", ""),
                        "domain": item.get("displayLink", "")
                    }
                    
        except Exception as e:
            logging.warning(f"Logo search failed for query '{query}': {e}")
            continue
    
    # Fallback: use general vendor search
    try:
        images, source = fetch_product_images_with_fallback_sync(vendor_name, "")
        if images:
            # Return first image as logo
            logo = images[0].copy()
            logo["source"] = f"{source}_fallback"
            return logo
    except Exception as e:
        logging.warning(f"Fallback logo search failed for {vendor_name}: {e}")
    
    return None

async def fetch_images_google_cse(vendor_name: str, product_name: str = None):
    """
    Step 1: Google Custom Search API for images from manufacturer domains
    """
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        logging.warning("Google CSE credentials not available for image search")
        return []
    
    try:
        query = f"{vendor_name}"
        if product_name:
            query += f" {product_name}"
        
        # Build site restriction for manufacturer domains
        domain_filter = " OR ".join([f"site:{domain}" for domain in MANUFACTURER_DOMAINS])
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
        for item in result.get("items", []):
            images.append({
                "url": item.get("link"),
                "title": item.get("title", ""),
                "source": "google_cse",
                "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                "domain": item.get("displayLink", "")
            })
        
        if images:
            logging.info(f"Google CSE found {len(images)} images for {vendor_name}")
        return images
        
    except Exception as e:
        logging.warning(f"Google CSE image search failed for {vendor_name}: {e}")
        return []

async def fetch_images_serpapi(vendor_name: str, product_name: str = None):
    """
    Step 2: SerpAPI fallback for Google Images
    """
    if not SERPAPI_KEY:
        logging.warning("SerpAPI key not available for image search")
        return []
    
    try:
        query = f"{vendor_name}"
        if product_name:
            query += f" {product_name}"
        query += " industrial product"
        
        search = GoogleSearch({
            "q": query,
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

async def fetch_images_serper(vendor_name: str, product_name: str = None):
    """
    Step 3: Serper.dev fallback for images
    """
    if not SERPER_API_KEY_IMAGES:
        logging.warning("Serper API key not available for image search")
        return []
    
    try:
        query = f"{vendor_name}"
        if product_name:
            query += f" {product_name}"
        query += " industrial equipment"
        
        url = "https://google.serper.dev/images"
        payload = {
            "q": query,
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

async def fetch_product_images_with_fallback(vendor_name: str, product_name: str = None):
    """
    3-level image search fallback system
    1. Google Custom Search API (manufacturer domains)
    2. SerpAPI Google Images
    3. Serper.dev images
    """
    logging.info(f"Starting image search for vendor: {vendor_name}, product: {product_name}")
    
    # Step 1: Try Google Custom Search API
    images = await fetch_images_google_cse(vendor_name, product_name)
    if images:
        logging.info(f"Using Google CSE images for {vendor_name}")
        return images, "google_cse"
    
    # Step 2: Try SerpAPI
    images = await fetch_images_serpapi(vendor_name, product_name)
    if images:
        logging.info(f"Using SerpAPI images for {vendor_name}")
        return images, "serpapi"
    
    # Step 3: Try Serper.dev
    images = await fetch_images_serper(vendor_name, product_name)
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
        # Use synchronous version for reliability
        images, source_used = fetch_product_images_with_fallback_sync(vendor_name, product_name)
        
        return jsonify({
            "vendor": vendor_name,
            "product": product_name,
            "images": images,
            "source_used": source_used,
            "count": len(images)
        })
        
    except Exception as e:
        logging.error(f"Image search test failed: {e}")
        return jsonify({
            "error": str(e),
            "vendor": vendor_name,
            "product": product_name,
            "images": [],
            "source_used": "error",
            "count": 0
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
            
        logging.info(f"Fetching images for analysis result: {vendor} {product_type} {product_name}")
        
        # Search for images with different combinations
        all_images = []
        search_combinations = []
        
        # 1. Most specific: vendor + product_name + model_family
        for model_family in model_families[:3]:  # Limit to first 3 model families
            search_query = f"{vendor} {product_name} {model_family}"
            search_combinations.append({
                "query": search_query,
                "type": "specific_model",
                "priority": 1
            })
        
        # 2. Medium specific: vendor + product_name + product_type
        if product_name and product_type:
            search_query = f"{vendor} {product_name} {product_type}"
            search_combinations.append({
                "query": search_query,
                "type": "product_with_type",
                "priority": 2
            })
        
        # 3. General: vendor + product_name
        if product_name:
            search_query = f"{vendor} {product_name}"
            search_combinations.append({
                "query": search_query,
                "type": "product_general",
                "priority": 3
            })
        
        # 4. Fallback: vendor + product_type
        if product_type:
            search_query = f"{vendor} {product_type}"
            search_combinations.append({
                "query": search_query,
                "type": "type_general",
                "priority": 4
            })
        
        # Execute searches and collect results
        for search_info in search_combinations:
            try:
                # Use the search query as product_name parameter
                images, source_used = fetch_product_images_with_fallback_sync(
                    vendor_name=vendor,
                    product_name=search_info["query"].replace(vendor, "").strip()
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
            if any(mfg_domain in domain for mfg_domain in MANUFACTURER_DOMAINS):
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
            vendor_logo = fetch_vendor_logo_sync(vendor)
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
    data = request.get_json()
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return jsonify({"error": "Missing username, email, or password"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 409
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 409

    hashed_pw = hash_password(password)
    new_user = User(
        username=username,
        email=email,
        password_hash=hashed_pw,
        status='pending',
        role='user'
    )
    db.session.add(new_user)
    db.session.commit()

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
        session['user_id'] = user.id
        return jsonify({
            "message": "Login successful",
            "user": {
                "username": user.username,
                "name": getattr(user, "name", user.username),
                "email": user.email,
                "role": user.role
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
    return jsonify({
        "user": {
            "username": user.username,
            "name": getattr(user, "name", user.username),
            "email": user.email,
            "role": user.role
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
        
        # Clear any previous product type for this search session to ensure independent searches
        session_key = f'product_type_{search_session_id}'
        if session_key in session:
            del session[session_key]
        
        # Store original user input for logging
        session['log_user_query'] = user_input

        initial_schema = load_requirements_schema()
        temp_validation_result = components['validation_chain'].invoke({
            "user_input": user_input,
            "schema": json.dumps(initial_schema, indent=2),
            "format_instructions": components['validation_format_instructions']
        })
        detected_type = temp_validation_result.get('product_type', 'UnknownProduct')
        
        # Debug logging for search independence
        logging.info(f"[VALIDATION] Search Session ID: {search_session_id}")
        logging.info(f"[VALIDATION] User Input: {user_input}")
        logging.info(f"[VALIDATION] Detected Product Type: {detected_type}")
        logging.info(f"[VALIDATION] Previous session keys: {[k for k in session.keys() if 'product_type' in k]}")

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

        validation_result = components['validation_chain'].invoke({
            "user_input": user_input,
            "schema": json.dumps(specific_schema, indent=2),
            "format_instructions": components['validation_format_instructions']
        })

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
                    """You are a helpful sales agent. The user shared their requirements, and you identified the product type as '{product_type}'.
Your response must:
1. Start positively (e.g., "Great choice!" or similar).
2. Confirm the identified product type in a friendly way.
3. Tell the user some important details are still missing.
4. List the missing fields as simple key names only: **{missing_fields}**, separated by commas.
5. Explain that results may only be approximate without them.
6. Ask if they’d like to continue anyway."""
                )
            else:
                alert_prompt = ChatPromptTemplate.from_template(
                    """You are a helpful sales assistant.
Write a short, clear response (1–2 sentences):
1. Tell the user there are still some missing specifications.
2. List the missing fields as simple key names only: **{missing_fields}**, separated by commas.
3. Explain that the search can continue, but results may only be approximate.
4. Ask if they’d like to proceed."""
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

        # Store product_type in session for later use in advanced parameters (with search session ID)
        session[f'product_type_{search_session_id}'] = response_data["productType"]
        # Also store in default location for backward compatibility
        session['product_type'] = response_data["productType"]

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
            schema_data = load_requirements_schema(product_type)
            if not schema_data:
                # Fallback to web discovery if local schema is missing
                schema_data = build_requirements_schema_from_web(product_type)
        else:
            schema_data = load_requirements_schema()
        return jsonify(convert_keys_to_camel_case(schema_data)), 200
    except Exception as e:
        logging.exception("Schema fetch failed.")
        return jsonify({"error": str(e)}), 500

@app.route("/additional_requirements", methods=["POST"])
@login_required
def api_additional_requirements():
    if not components:
        return jsonify({"error": "Backend is not ready. LangChain failed."}), 503
    try:
        data = request.get_json(force=True)
        product_type = data.get("product_type", "").strip()
        user_input = data.get("user_input", "").strip()

        
        if not product_type:
            return jsonify({"error": "Missing product_type"}), 400

        specific_schema = load_requirements_schema(product_type)
        validation_result = components['additional_requirements_chain'].invoke({
            "user_input": user_input,
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
    Discovers advanced parameters from top vendors for a product type
    """
    try:
        data = request.get_json(force=True)
        product_type = data.get("product_type", "").strip()
        
        if not product_type:
            return jsonify({"error": "Missing 'product_type' parameter"}), 400

        # Store for logging
        session['log_user_query'] = f"Advanced parameters for {product_type}"
        
        # Discover advanced parameters
        logging.info(f"Starting advanced parameters discovery for: {product_type}")
        result = discover_advanced_parameters(product_type)
        
        # Log detailed information about filtering
        unique_count = len(result.get('unique_parameters', []))
        filtered_count = result.get('existing_parameters_filtered', 0)
        total_found = unique_count + filtered_count
        
        logging.info(f"Advanced parameters discovery complete: {total_found} total parameters found, {filtered_count} filtered out (already in schema), {unique_count} new parameters returned")
        
        # Store result for logging
        session['log_system_response'] = result
        
        # Convert to camelCase for frontend
        camel_case_result = convert_keys_to_camel_case(result)
        
        logging.info(f"Advanced parameters discovery complete: {len(result.get('unique_parameters', []))} new parameters found (filtered out {result.get('existing_parameters_filtered', 0)} existing parameters)")
        
        return jsonify(camel_case_result), 200

    except Exception as e:
        logging.exception("Advanced parameters discovery failed.")
        return jsonify({"error": str(e)}), 500

@app.route("/api/add_advanced_parameters", methods=["POST"])
@login_required
def api_add_advanced_parameters():
    """
    Processes user input for advanced parameters selection
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

        # Use LLM to extract selected parameters from user input
        prompt = ChatPromptTemplate.from_template("""
You are an expert assistant helping users select advanced parameters for industrial equipment.

Product Type: {product_type}
Available Parameters: {available_parameters}
User Input: "{user_input}"

Extract the parameters the user wants to add from their input. The user might:
1. Select specific parameters by name
2. Say "all" or "everything" to select all parameters
3. Select categories or groups of parameters
4. Provide specific values for parameters

Return a JSON object with:
{{
    "selected_parameters": {{"parameter_name": "user_specified_value_or_empty_string"}},
    "explanation": "Brief explanation of what was selected"
}}

If the user didn't specify values, use empty strings for the parameter values.
Only include parameters that are in the available_parameters list.

Examples:
- "I want response_time and accuracy" → {{"response_time": "", "accuracy": ""}}
- "Add all parameters" → Include all available parameters with empty values
- "Set accuracy to 0.1% and response_time to 1ms" → {{"accuracy": "0.1%", "response_time": "1ms"}}
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
            explanation = result.get("explanation", "Parameters selected successfully.")

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
                explanation = "All available parameters have been selected."
            else:
                # Look for parameter names in user input
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
                
                explanation = f"Selected {len(selected_parameters)} parameters based on your input."

        # Generate friendly response
        if selected_parameters:
            param_list = ", ".join([param.replace('_', ' ').title() for param in selected_parameters.keys()])
            friendly_response = f"Great! I've added these advanced parameters: {param_list}. Would you like to add any more advanced parameters?"
        else:
            friendly_response = "I didn't find any matching parameters in your input. Could you please specify which parameters you'd like to add?"

        response_data = {
            "selectedParameters": convert_keys_to_camel_case(selected_parameters),
            "explanation": explanation,
            "friendlyResponse": friendly_response,
            "totalSelected": len(selected_parameters)
        }

        return jsonify(response_data), 200

    except Exception as e:
        logging.exception("Advanced parameters addition failed.")
        return jsonify({"error": str(e)}), 500

@app.route("/analyze", methods=["POST"])
@login_required
def api_analyze():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        user_input = data.get("user_input")
        if user_input is None:
            return jsonify({"error": "Missing 'user_input' parameter"}), 400

  
        # Ensure user_input is a dict
        if isinstance(user_input, str):
            try:
                user_input = json.loads(user_input)
            except json.JSONDecodeError:
                # fallback: wrap string into dict
                user_input = {"raw_input": user_input}

        if not isinstance(user_input, dict):
            return jsonify({"error": "user_input must be a dict or JSON string representing a dict"}), 400

        # Pass user_input to the analysis chain
        analysis_result = analysis_chain.invoke({"user_input": user_input})
        
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
        
        vendors = []
        
        def process_vendor(vendor_name):
            """Process a single vendor synchronously for better reliability"""
            try:
                # Fetch product images using the 3-level fallback system (sync version)
                images, source_used = fetch_product_images_with_fallback_sync(vendor_name)
                
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
        "email": u.email
    } for u in pending]
    return jsonify({"pending_users": result}), 200

# Duplicated ALLOWED_EXTENSIONS and allowed_file, can be removed.
# ALLOWED_EXTENSIONS = {"pdf"}
# def allowed_file(filename: str):
#    ...

@app.route("/api/get-price-review", methods=["GET"])
@login_required
def api_get_price_review():
    product_name = request.args.get("productName")
    if not product_name:
        return jsonify({"error": "Missing productName parameter"}), 400

    results = fetch_price_and_reviews(product_name)

    return jsonify(results), 200

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
    

def create_db():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(role='admin').first():
            hashed_pw = hash_password("Daman@123")
            admin = User(username="Daman", email="reddydaman04@gmail.com", password_hash=hashed_pw, status='active', role='admin')
            db.session.add(admin)
            db.session.commit()
            print("Admin user created with username 'Daman' and password 'Daman@123'.")
if __name__ == "__main__":
    create_db()
    import os

    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)