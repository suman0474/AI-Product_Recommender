# tools/instrument_tools.py
# Instrument Identification and Accessory Tools

import json
import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os
from services.llm.fallback import create_llm_with_fallback

from config import AgenticConfig
from prompts_library import load_prompt, load_prompt_sections
from tools.standards_enrichment_tool import get_applicable_standards
import concurrent.futures
from azure_blob_config import get_azure_blob_connection, Collections
logger = logging.getLogger(__name__)


# ============================================================================
# TOOL INPUT SCHEMAS
# ============================================================================

class IdentifyInstrumentsInput(BaseModel):
    """Input for instrument identification"""
    requirements: str = Field(description="Process requirements or problem statement")


class IdentifyAccessoriesInput(BaseModel):
    """Input for accessory identification"""
    instruments: List[Dict[str, Any]] = Field(description="List of identified instruments")
    process_context: Optional[str] = Field(default=None, description="Process context")


class ModifyInstrumentsInput(BaseModel):
    """Input for modifying instruments and accessories"""
    modification_request: str = Field(description="User's request to modify the list")
    current_instruments: List[Dict[str, Any]] = Field(description="Current list of instruments")
    current_accessories: List[Dict[str, Any]] = Field(description="Current list of accessories")
    search_session_id: Optional[str] = Field(default="default", description="Session ID for user context")



# ============================================================================
# PROMPTS - Loaded from prompts_library
# ============================================================================

# Unified product identification prompts (instruments + accessories)
# default_section captures content before first section marker as "INSTRUMENT_IDENTIFICATION"
_PRODUCT_IDENTIFICATION_PROMPTS = load_prompt_sections("product_identification_prompt", default_section="INSTRUMENT_IDENTIFICATION")
INSTRUMENT_IDENTIFICATION_PROMPT = _PRODUCT_IDENTIFICATION_PROMPTS.get("INSTRUMENT_IDENTIFICATION", "")
ACCESSORIES_IDENTIFICATION_PROMPT = _PRODUCT_IDENTIFICATION_PROMPTS.get("ACCESSORIES_IDENTIFICATION", "")

# Single-prompt file - default_section captures the entire content
INSTRUMENT_MODIFICATION_PROMPTS = load_prompt_sections("instrument_modification_prompt", default_section="INSTRUMENT_MODIFICATION")


# ============================================================================
# TOOLS
# ============================================================================

@tool("identify_instruments", args_schema=IdentifyInstrumentsInput)
def identify_instruments_tool(requirements: str) -> Dict[str, Any]:
    """
    Identify instruments needed from process requirements.
    Extracts product types, specifications, and creates a Bill of Materials.

    Features:
    - 3-attempt retry mechanism for LLM-based identification
    - Extended timeout (120s) to handle complex requirements
    - Returns partial results on timeout rather than empty list
    """
    last_error = None

    # 3 retry attempts for robustness
    for attempt in range(1, 4):
        try:
            logger.info(f"[Instrument ID] LLM attempt {attempt}/3...")

            llm = create_llm_with_fallback(
                model=AgenticConfig.FLASH_MODEL,
                temperature=0.1,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                timeout=120  # Increased from default 60s to handle complex requirements
            )

            prompt = ChatPromptTemplate.from_template(INSTRUMENT_IDENTIFICATION_PROMPT)
            parser = JsonOutputParser()

            chain = prompt | llm | parser

            result = chain.invoke({"requirements": requirements})

            # Validate result structure
            if not isinstance(result, dict):
                logger.warning(f"[Instrument ID] Attempt {attempt}: Invalid response type {type(result)}")
                last_error = f"Invalid response type: {type(result)}"
                continue

            instruments = result.get("instruments", [])

            if instruments and isinstance(instruments, list):
                logger.info(f"[Instrument ID] Attempt {attempt} succeeded: {len(instruments)} instruments identified")
                return {
                    "success": True,
                    "project_name": result.get("project_name"),
                    "instruments": instruments,
                    "instrument_count": len(instruments),
                    "summary": result.get("summary")
                }
            else:
                logger.warning(f"[Instrument ID] Attempt {attempt}: Empty or invalid instruments list")
                last_error = "Empty instruments list from LLM"
                continue

        except Exception as e:
            logger.warning(f"[Instrument ID] Attempt {attempt}: {type(e).__name__} - {str(e)}")
            last_error = str(e)
            continue

    # All attempts failed
    logger.error(f"[Instrument ID] All 3 attempts failed. Last error: {last_error}")
    return {
        "success": False,
        "instruments": [],
        "error": f"Instrument identification failed after 3 attempts: {last_error}"
    }


# ============================================================================
# KEYWORD-BASED FALLBACK FOR ACCESSORY EXTRACTION
# ============================================================================

# Universal accessory keywords applicable to multiple system types
ACCESSORY_KEYWORDS = {
    # Temperature systems
    "thermowell": "Thermowell / Protective Sleeve",
    "extension wire": "Thermocouple Extension Wire",
    "shield": "Thermocouple Protection Shield",
    "head": "Connection Head / Terminal Block",

    # Pressure systems
    "impulse line": "Impulse Line / Tubing",
    "isolation valve": "Isolation Valve / Manifold",
    "snubber": "Snubber / Pulsation Dampener",
    "equalizing": "Equalizing Connection",

    # Flow systems
    "straightener": "Flow Straightener",
    "orifice": "Orifice Plate / Flow Restriction",
    "tap": "Pressure Tap / Sensing Point",

    # General accessories
    "cable gland": "Cable Gland / Connector",
    "junction box": "Junction Box / Terminal Enclosure",
    "mounting bracket": "Mounting Bracket / Support",
    "gasket": "Gasket / Sealing Material",
    "fastener": "Fastener / Bolt Set",
    "power supply": "Power Supply Unit",
    "signal cable": "Signal Cable / Wiring",
    "connector": "Connector / Adapter",
    "conduit": "Conduit / Cable Protection",
    "strain relief": "Strain Relief / Cable Management",
    "terminal strip": "Terminal Strip / Connection Board",
    "calibration": "Calibration Kit / Standard",
    "protective cap": "Protective Cap / Cover",
    "spare parts": "Spare Parts / Replacement Kit",
}


def extract_accessories_by_keywords(
    instruments: List[Dict[str, Any]],
    context: str
) -> List[Dict[str, Any]]:
    """
    Extract accessories using keyword matching when LLM fails.
    Works universally for temperature, pressure, flow, and other systems.

    Args:
        instruments: List of identified instruments
        context: Process context for accessory determination

    Returns:
        List of accessories inferred from instrument types and context
    """
    accessories = []

    # Combine instrument info with context for analysis
    combined_text = f"{context} {json.dumps(instruments)}".lower()

    # Extract matching accessory keywords
    matched_accessories = set()
    for keyword, accessory_type in ACCESSORY_KEYWORDS.items():
        if keyword in combined_text:
            matched_accessories.add(accessory_type)

    # Create accessory items with inferred specifications
    for idx, accessory_type in enumerate(sorted(matched_accessories), 1):
        accessory = {
            "number": idx,
            "name": accessory_type,
            "category": accessory_type.split("/")[0].strip(),
            "quantity": 1,
            "specifications": {
                "extraction_method": "keyword_fallback",
                "inferred_from": "system_context_and_instrument_types",
            },
            "source_method": "keyword_extraction",
            "confidence": "low"  # Flag that this came from fallback
        }
        accessories.append(accessory)

    logger.info(f"[Keyword Extraction] Extracted {len(accessories)} accessories: {matched_accessories}")
    return accessories


@tool("identify_accessories", args_schema=IdentifyAccessoriesInput)
def identify_accessories_tool(
    instruments: List[Dict[str, Any]],
    process_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Robustly identify accessories needed for the identified instruments.

    Features:
    - 3-attempt retry mechanism for LLM-based identification
    - Automatic fallback to keyword-based extraction
    - Never returns empty if instruments exist (ensures complete solutions)
    - Universal logic works for all system types

    Returns list of accessories with specifications.
    """

    # Validate inputs
    if not instruments:
        logger.warning("[Accessory ID] No instruments provided - cannot identify accessories")
        return {
            "success": False,
            "accessories": [],
            "error": "No instruments provided",
            "method": "validation_failure"
        }

    context = process_context or "General industrial process"
    logger.info(f"[Accessory ID] Starting accessory identification for {len(instruments)} instrument(s)")

    # =========================================================================
    # ATTEMPT 1-3: LLM-based identification with retry
    # =========================================================================

    llm_accessories = None
    last_error = None

    for attempt in range(1, 4):  # 3 attempts
        try:
            logger.info(f"[Accessory ID] LLM attempt {attempt}/3...")

            llm = create_llm_with_fallback(
                model=AgenticConfig.FLASH_MODEL,
                temperature=0.1,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )

            prompt = ChatPromptTemplate.from_template(ACCESSORIES_IDENTIFICATION_PROMPT)
            parser = JsonOutputParser()
            chain = prompt | llm | parser

            result = chain.invoke({
                "instruments": json.dumps(instruments, indent=2),
                "process_context": context
            })

            # Validate LLM response structure
            if not isinstance(result, dict):
                logger.warning(f"[Accessory ID] Attempt {attempt}: Invalid response type {type(result)}")
                last_error = f"Invalid response type: {type(result)}"
                continue

            accessories = result.get("accessories", [])

            # Check if LLM returned valid accessories
            if accessories and isinstance(accessories, list):
                logger.info(f"[Accessory ID] ✅ LLM attempt {attempt} succeeded: {len(accessories)} accessories identified")
                llm_accessories = accessories
                break
            else:
                logger.warning(f"[Accessory ID] Attempt {attempt}: Empty or invalid accessories list")
                last_error = "Empty accessories list from LLM"
                continue

        except json.JSONDecodeError as e:
            logger.warning(f"[Accessory ID] Attempt {attempt}: JSON parse error - {str(e)}")
            last_error = f"JSON parse error: {str(e)}"
            continue

        except Exception as e:
            logger.warning(f"[Accessory ID] Attempt {attempt}: {type(e).__name__} - {str(e)}")
            last_error = str(e)
            continue

    # =========================================================================
    # FALLBACK: Keyword-based extraction if LLM failed
    # =========================================================================

    if llm_accessories is not None:
        # LLM succeeded
        return {
            "success": True,
            "accessories": llm_accessories,
            "accessory_count": len(llm_accessories),
            "summary": f"Identified {len(llm_accessories)} accessories using LLM",
            "method": "llm_identification"
        }
    else:
        # LLM failed all 3 attempts - use fallback
        logger.warning(f"[Accessory ID] ⚠️ LLM failed after 3 attempts. Last error: {last_error}")
        logger.info("[Accessory ID] 🔄 Activating keyword-based fallback extraction...")

        fallback_accessories = extract_accessories_by_keywords(instruments, context)

        if fallback_accessories:
            logger.info(f"[Accessory ID] ✅ Fallback succeeded: {len(fallback_accessories)} accessories extracted")
            return {
                "success": True,
                "accessories": fallback_accessories,
                "accessory_count": len(fallback_accessories),
                "summary": f"Identified {len(fallback_accessories)} accessories using keyword extraction",
                "method": "keyword_fallback",
                "llm_failure_reason": last_error
            }
        else:
            # Even fallback returned nothing
            return {
                "success": False,
                "accessories": [],
                "accessory_count": 0,
                "summary": "Could not identify accessories (LLM and fallback both failed)",
                "method": "none",
                "llm_failure_reason": last_error,
                "error": "Accessory identification failed - LLM returned invalid response and keyword fallback found no matches"
            }


@tool("modify_instruments", args_schema=ModifyInstrumentsInput)
def modify_instruments_tool(
    modification_request: str,
    current_instruments: List[Dict[str, Any]],
    current_accessories: List[Dict[str, Any]],
    search_session_id: str = "default"
) -> Dict[str, Any]:
    """
    Modify/refine the identified instruments and accessories list based on user request.
    Integrating Standards RAG from Azure Blob Storage where applicable.
    """
    try:
        if not modification_request:
            return {"error": "Modification request is required", "success": False}

        if not current_instruments and not current_accessories:
            return {
                "error": "No instruments or accessories to modify. Please identify instruments first.",
                "success": False
            }

        # Create a JSON representation of current items for the LLM
        current_state = {
            "instruments": current_instruments,
            "accessories": current_accessories
        }
        current_state_json = json.dumps(current_state, indent=2)
        
        # 0. FETCH USER DOCUMENTS FROM AZURE BLOB
        user_documents_context = "No user specific documents found."
        try:
            logger.info(f"[Modify Instruments] Fetching user documents for session: {search_session_id}")
            blob_conn = get_azure_blob_connection()
            
            # Use 'documents' collection or 'user_projects' depending on where docs are stored.
            # Assuming 'documents' for general user uploads/standards.
            documents_collection = blob_conn['collections']['documents']
            
            # Simple metadata search (assuming user_id or session_id is metadata)
            # Since we only have search_session_id, we look for docs tagged with it.
            # In a real app, we'd look up user_id from session first.
            query = {"session_id": search_session_id}
            user_docs = documents_collection.find(query)
            
            if user_docs:
                docs_summary = []
                for doc in user_docs[:3]: # Limit to 3 docs
                    name = doc.get("filename", doc.get("name", "Unknown Document"))
                    # If we have extracted text (e.g. from a previous ingestion), use it.
                    # Assuming doc has 'extracted_text' or similar.
                    # For now, we list metadata as context.
                    summary = f"Document: {name}\nType: {doc.get('document_type', 'General')}\nDescription: {doc.get('description', '')}"
                    docs_summary.append(summary)
                
                if docs_summary:
                    user_documents_context = "\n---\n".join(docs_summary)
                    logger.info(f"[Modify Instruments] Found {len(user_docs)} user documents.")
            
        except Exception as e:
            logger.warning(f"[Modify Instruments] Failed to fetch user documents: {e}")
            user_documents_context = "Error fetching user documents (ignored)."


        # Initialize LLM
        llm = create_llm_with_fallback(
            model=AgenticConfig.FLASH_MODEL,
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # 1. GENERATE MODIFICATIONS
        modification_prompt = INSTRUMENT_MODIFICATION_PROMPTS.get("MODIFICATION_PROMPT", "")
        # Fallback if section loading fails
        if not modification_prompt:
             modification_prompt = load_prompt("instrument_modification_prompt")

        prompt = ChatPromptTemplate.from_template(modification_prompt)
        parser = JsonOutputParser()
        chain = prompt | llm | parser

        logger.info(f"[Modify Instruments] Processing request: {modification_request}")
        
        try:
            result = chain.invoke({
                "current_state": current_state_json,
                "modification_request": modification_request,
                "user_documents_context": user_documents_context
            })
        except Exception as e:
            logger.error(f"[Modify Instruments] LLM processing failed: {e}")
            return {"error": f"Failed to process modification: {str(e)}", "success": False}

        # 2. VALIDATE AND CLEAN RESULT
        if "instruments" not in result:
            result["instruments"] = current_instruments
        if "accessories" not in result:
            result["accessories"] = current_accessories
        if "changes_made" not in result:
            result["changes_made"] = []
        if "summary" not in result:
            result["summary"] = "Modifications applied successfully."

        # Ensure required fields
        for instrument in result.get("instruments", []):
            if "strategy" not in instrument: instrument["strategy"] = ""
            if "quantity" not in instrument: instrument["quantity"] = "1"
        
        for accessory in result.get("accessories", []):
            if "strategy" not in accessory: accessory["strategy"] = ""
            if "quantity" not in accessory: accessory["quantity"] = "1"

        # 3. APPLY STANDARDS (RAG)
        # We will use get_applicable_standards which uses Azure Blob Storage / Vector Store internally
        logger.info(f"[Modify Instruments] Applying standards RAG to modified items...")
        
        def apply_standards_to_item(item, is_accessory=False):
            """Apply standards to an instrument or accessory if not already applied"""
            category = item.get('category', '')
            if not category:
                return item
            
            # Skip if standards already applied (has standards_specs)
            if item.get('standards_specs') and len(item.get('standards_specs', {})) > 0:
                return item
            
            try:
                # Use the existing tool which integrates with Vector Store (backed by Azure Blob docs)
                standards_result = get_applicable_standards(product_type=category)
                
                if standards_result and standards_result.get('success'):
                    # Add standards specifications
                    item['standards_specs'] = {
                        'applicable_standards': standards_result.get('applicable_standards', []),
                        'certifications': standards_result.get('certifications', []),
                        'safety_requirements': standards_result.get('safety_requirements', {})
                    }
                    item['applicable_standards'] = standards_result.get('applicable_standards', [])
                    item['standards_summary'] = f"Applicable Standards: {', '.join(standards_result.get('applicable_standards', [])[:3])}"
                    
                    # Enhance specifications with standards annotations
                    original_specs = item.get('specifications', {})
                    # In a real scenario, we might merge specific standard values, 
                    # for now we append the applicable standards list to specs for visibility
                    enhanced_specs = original_specs.copy()
                    
                    # Simple enhancement: Add certification requirements if found
                    certs = standards_result.get('certifications', [])
                    if certs and 'Certifications' not in enhanced_specs:
                         enhanced_specs['Certifications'] = ', '.join(certs[:3])

                    item['specifications'] = enhanced_specs
                    
                    # Update sample_input
                    original_sample_input = item.get('sample_input', '')
                    if 'Standards:' not in original_sample_input and item.get('applicable_standards'):
                         item['sample_input'] = f"{original_sample_input}. Standards: {', '.join(item['applicable_standards'][:2])}"
                    
                    item_type = "accessory" if is_accessory else "instrument"
                    # logger.info(f"[STANDARDS_RAG] Applied standards to {item_type} {category}")
                
                return item
            except Exception as e:
                # logger.warning(f"[STANDARDS_RAG] Failed to apply standards to {category}: {e}")
                return item

        # Apply in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Instruments
            inst_futures = {executor.submit(apply_standards_to_item, inst, False): i 
                           for i, inst in enumerate(result.get("instruments", []))}
            updated_instruments = [None] * len(result.get("instruments", []))
            for future in concurrent.futures.as_completed(inst_futures):
                updated_instruments[inst_futures[future]] = future.result()
            result["instruments"] = updated_instruments

            # Accessories
            acc_futures = {executor.submit(apply_standards_to_item, acc, True): i 
                          for i, acc in enumerate(result.get("accessories", []))}
            updated_accessories = [None] * len(result.get("accessories", []))
            for future in concurrent.futures.as_completed(acc_futures):
                updated_accessories[acc_futures[future]] = future.result()
            result["accessories"] = updated_accessories

        result["standards_applied"] = True

        # 4. GENERATE FRIENDLY MESSAGE
        changes_count = len(result.get("changes_made", []))
        instrument_count = len(result.get("instruments", []))
        accessory_count = len(result.get("accessories", []))
        
        message_prompt_template = INSTRUMENT_MODIFICATION_PROMPTS.get("MESSAGE_GENERATION", "")
        # Fallback
        if not message_prompt_template:
             message_prompt_template = """
You are Engenie - a friendly industrial automation assistant.
Changes made: {changes_list}
Total instruments now: {instrument_count}
Total accessories now: {accessory_count}
Generate a brief, friendly confirmation message.
"""

        try:
             msg_prompt = ChatPromptTemplate.from_template(message_prompt_template)
             from langchain_core.output_parsers import StrOutputParser
             msg_chain = msg_prompt | llm | StrOutputParser()
             
             friendly_message = msg_chain.invoke({
                "changes_list": ", ".join(result.get("changes_made", ["No specific changes noted"])),
                "instrument_count": instrument_count,
                "accessory_count": accessory_count
             })
             result["message"] = friendly_message.strip()
        except Exception as msg_error:
             logger.warning(f"Failed to generate friendly message: {msg_error}")
             result["message"] = f"I've updated your list! You now have **{instrument_count} instruments** and **{accessory_count} accessories**."

        result["success"] = True
        return result

    except Exception as e:
        logger.error(f"Instrument modification tool failed: {e}", exc_info=True)
        return {
            "success": False,
            "instruments": current_instruments,
            "accessories": current_accessories,
            "error": str(e)
        }
