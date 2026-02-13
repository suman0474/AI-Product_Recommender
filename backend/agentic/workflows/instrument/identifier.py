# agentic/instrument_identifier_workflow.py
# Instrument Identifier Workflow - List Generator Only
# This workflow identifies instruments/accessories and generates a selection list.
# It does NOT perform product search - that's handled by the SOLUTION workflow.

import json
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from agentic.models import (
    InstrumentIdentifierState,
    create_instrument_identifier_state
)
from agentic.infrastructure.state.checkpointing.local import compile_with_checkpointing

from tools.intent_tools import classify_intent_tool
from tools.instrument_tools import identify_instruments_tool, identify_accessories_tool

# Standards RAG enrichment for grounded identification
from ..standards_rag.standards_rag_enrichment import enrich_identified_items_with_standards

# Standards detection for conditional enrichment
from ...agents.standards_detector import detect_standards_indicators

import os
from dotenv import load_dotenv
from services.llm.fallback import create_llm_with_fallback
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# PROMPTS
# ============================================================================

# Import prompt loader directly from library
# Use absolute import assuming backend is in path
try:
    from prompts_library.prompt_loader import load_prompt
except ImportError:
    # Fallback to ensure we catch path issues
    import sys
    backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if backend_root not in sys.path:
        sys.path.append(backend_root)
    from prompts_library.prompt_loader import load_prompt

def get_instrument_list_prompt() -> str:
    return load_prompt("instrument_list_prompt")


# ============================================================================
# WORKFLOW NODES
# ============================================================================

def classify_initial_intent_node(state: InstrumentIdentifierState) -> InstrumentIdentifierState:
    """
    Node 1: Initial Intent Classification.
    Determines if this is an instrument/accessory identification request.
    """
    logger.info("[IDENTIFIER] Node 1: Initial intent classification...")

    try:
        result = classify_intent_tool.invoke({
            "user_input": state["user_input"],
            "context": None
        })

        if result.get("success"):
            state["initial_intent"] = result.get("intent", "requirements")
        else:
            state["initial_intent"] = "requirements"

        state["current_step"] = "identify_instruments"

        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Initial intent: {state['initial_intent']}"
        }]

        logger.info(f"[IDENTIFIER] Initial intent: {state['initial_intent']}")

    except Exception as e:
        logger.error(f"[IDENTIFIER] Initial intent classification failed: {e}")
        state["error"] = str(e)

    return state


def identify_instruments_and_accessories_node(state: InstrumentIdentifierState) -> InstrumentIdentifierState:
    """
    Node 2: Instrument/Accessory Identifier.
    Identifies ALL instruments and accessories from user requirements.
    Generates sample_input for each item.
    
    OPTIMIZATION: Runs instrument and accessory identification in PARALLEL
    to reduce overall latency.
    """
    logger.info("[IDENTIFIER] Node 2: Identifying instruments and accessories (PARALLEL)...")

    # Initialize empty lists
    state["identified_instruments"] = []
    state["identified_accessories"] = []
    state["project_name"] = "Untitled Project"

    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def identify_instruments_task():
        """Task to identify instruments."""
        try:
            return identify_instruments_tool.invoke({
                "requirements": state["user_input"]
            })
        except Exception as e:
            logger.error(f"[IDENTIFIER] Instrument identification exception: {e}")
            return {"success": False, "error": str(e), "instruments": []}
    
    def identify_accessories_task(instruments_list):
        """Task to identify accessories based on instruments."""
        if not instruments_list:
            return {"success": False, "accessories": [], "error": "No instruments to base accessories on"}
        try:
            return identify_accessories_tool.invoke({
                "instruments": instruments_list,
                "process_context": state["user_input"]
            })
        except Exception as e:
            logger.error(f"[IDENTIFIER] Accessory identification exception: {e}")
            return {"success": False, "error": str(e), "accessories": []}

    # First, identify instruments (can't parallelize this with accessories since accessories depend on instruments)
    inst_result = identify_instruments_task()
    
    if inst_result.get("success"):
        state["identified_instruments"] = inst_result.get("instruments", [])
        state["project_name"] = inst_result.get("project_name", "Untitled Project")
        logger.info(f"[IDENTIFIER] Successfully identified {len(state['identified_instruments'])} instruments")
    else:
        logger.warning(f"[IDENTIFIER] Instrument identification failed: {inst_result.get('error', 'Unknown error')}")

    # Now identify accessories based on identified instruments
    if state["identified_instruments"]:
        acc_result = identify_accessories_task(state["identified_instruments"])
        
        if acc_result.get("success"):
            state["identified_accessories"] = acc_result.get("accessories", [])
            logger.info(f"[IDENTIFIER] Successfully identified {len(state['identified_accessories'])} accessories")
        else:
            logger.warning(f"[IDENTIFIER] Accessory identification failed: {acc_result.get('error', 'Unknown error')}")

    # ENHANCED FALLBACK: If still no items found, use rule-based extraction
    if not state["identified_instruments"] and not state["identified_accessories"]:
        logger.warning("[IDENTIFIER] LLM-based identification failed, using rule-based fallback")
        
        # Simple keyword-based extraction
        user_input_lower = state["user_input"].lower()
        
        # Common instrument keywords
        instrument_keywords = {
            "temperature": ("Temperature Sensor/Transmitter", "Temperature Measurement"),
            "pressure": ("Pressure Transmitter", "Pressure Measurement"),
            "flow": ("Flow Meter", "Flow Measurement"),
            "level": ("Level Transmitter", "Level Measurement"),
            "valve": ("Control Valve", "Valves & Actuators"),
            "analyzer": ("Analyzer", "Analytical Instruments"),
            "transmitter": ("Transmitter", "General Instruments"),
            "sensor": ("Sensor", "General Instruments"),
            "thermocouple": ("Thermocouple", "Temperature Measurement"),
            "rtd": ("RTD Sensor", "Temperature Measurement"),
        }
        
        # Extract instruments based on keywords
        detected_instruments = []
        for keyword, (product_name, category) in instrument_keywords.items():
            if keyword in user_input_lower:
                detected_instruments.append({
                    "category": category,
                    "product_name": product_name,
                    "quantity": 1,
                    "specifications": {},
                    "sample_input": f"{product_name} for {state['user_input'][:100]}",
                    "strategy": "keyword_extraction"
                })
        
        # If still nothing, create a generic item
        if not detected_instruments:
            detected_instruments = [{
                "category": "Industrial Instrument",
                "product_name": "General Instrument",
                "quantity": 1,
                "specifications": {},
                "sample_input": state["user_input"][:200] if len(state["user_input"]) > 200 else state["user_input"],
                "strategy": "generic_fallback"
            }]
            logger.warning("[IDENTIFIER] Created generic fallback instrument")
        else:
            logger.info(f"[IDENTIFIER] Keyword extraction found {len(detected_instruments)} instruments")
        
        state["identified_instruments"] = detected_instruments
        state["project_name"] = "Keyword-Based Identification"

    # Build unified item list with sample_inputs (runs for ALL cases)
    all_items = []
    item_number = 1

    # Add instruments
    for instrument in state["identified_instruments"]:
        # Flatten nested spec dicts to clean values (removes {'value': 'x', 'confidence': 0.9} structures)
        clean_specs = _flatten_nested_specs(instrument.get("specifications", {}))
        
        # Ensure sample_input includes ALL technical specifications
        raw_sample = instrument.get("sample_input", "")
        final_sample = _ensure_full_spec_coverage(raw_sample, instrument, state.get("project_name", "Project"))
        
        all_items.append({
            "number": item_number,
            "type": "instrument",
            "name": instrument.get("product_name", "Unknown Instrument"),
            "category": instrument.get("category", "Instrument"),
            "quantity": instrument.get("quantity", 1),
            "specifications": clean_specs,
            "sample_input": final_sample,
            "strategy": instrument.get("strategy", "")
        })
        item_number += 1

    # Add accessories
    for accessory in state["identified_accessories"]:
        # Get sample_input from LLM or construct basic one
        llm_sample_input = accessory.get("sample_input", "")
        basic_sample_input = f"{accessory.get('category', 'Accessory')} for {accessory.get('related_instrument', 'instruments')}"
        
        # Use LLM one if decent length, otherwise start with basic (or LLM one if exists)
        base_input = llm_sample_input if len(llm_sample_input) > 20 else basic_sample_input
        
        # Ensure sample_input includes ALL technical specifications
        final_sample = _ensure_full_spec_coverage(base_input, accessory, state.get("project_name", "Project"))

        # Smart category extraction: If category is generic "Accessories" or "Accessory",
        # extract the product type from accessory_name (e.g., "Thermowell for X" -> "Thermowell")
        raw_category = accessory.get("category", "Accessory")
        accessory_name = accessory.get("accessory_name", "Unknown Accessory")
        
        if raw_category.lower() in ["accessories", "accessory", ""]:
            # Extract product type from accessory name (before " for ")
            if " for " in accessory_name:
                extracted_type = accessory_name.split(" for ")[0].strip()
                smart_category = extracted_type if extracted_type else raw_category
            else:
                smart_category = accessory_name  # Use full name if no " for " pattern
        else:
            smart_category = raw_category

        # Flatten nested spec dicts to clean values
        clean_specs = _flatten_nested_specs(accessory.get("specifications", {}))
        
        all_items.append({
            "number": item_number,
            "type": "accessory",
            "name": accessory_name,
            "category": smart_category,  # Use smart category instead of raw
            "quantity": accessory.get("quantity", 1),
            "specifications": clean_specs,
            "sample_input": final_sample,
            "related_instrument": accessory.get("related_instrument", "")
        })
        item_number += 1

    state["all_items"] = all_items
    state["total_items"] = len(all_items)

    state["current_step"] = "format_list"

    total_items = len(state["identified_instruments"]) + len(state["identified_accessories"])

    state["messages"] = state.get("messages", []) + [{
        "role": "system",
        "content": f"Identified {len(state['identified_instruments'])} instruments and {len(state['identified_accessories'])} accessories"
    }]

    logger.info(f"[IDENTIFIER] Found {total_items} items total")

    return state



def _ensure_full_spec_coverage(text: str, item: dict, project_name: str) -> str:
    """
    Ensure the text includes ALL identified technical specifications.
    Augments the description with any missing specifications.
    """
    if not text:
        text = ""
        
    name = item.get("name") or item.get("accessory_name") or item.get("product_name") or "Component"
    category = item.get("category", "Industrial Item")
    
    # Start with what we have
    augmented_text = text
    
    # Add intro if completely empty or name missing
    if not augmented_text:
        augmented_text = f"Technical specifications for {name} ({category}) required for {project_name}."
    elif name not in augmented_text:
        # Prepend context if missing
        augmented_text = f"For {project_name}: {augmented_text}"

    # Get all specifications
    specs = item.get("specifications", {})
    if not specs:
        return augmented_text

    # Identify which specs are missing from the text
    missing_specs = []
    text_lower = augmented_text.lower()
    
    for k, v in specs.items():
        # Extract clean value (handles nested dicts, lists, etc.)
        val = _extract_spec_value(v) if isinstance(v, (dict, list)) else str(v)
            
        # Skip empty/null values
        if not val or val.lower() in ["null", "none", "n/a", ""]:
            continue
            
        # Check if value is already in text (simple heuristic)
        # We check if the value string is present
        if str(val).lower() not in text_lower:
            # Clean up key name for readability
            clean_k = k.replace("_", " ").title()
            missing_specs.append(f"{clean_k}: {val}")
    
    # Append missing specs
    if missing_specs:
        # If text doesn't end with punctuation, add period
        if augmented_text and augmented_text[-1] not in [".", "!", "?"]:
            augmented_text += "."
            
        augmented_text += " Specifications: " + "; ".join(missing_specs) + "."
            
    return augmented_text


def _extract_spec_value(spec_value) -> str:
    """
    Extract ONLY the value from any specification format.
    
    Handles:
    - dict with 'value' key: {'value': 'x', 'confidence': 0.9} → 'x'
    - nested category dict: {'temp_range': {'value': 'y'}} → 'y' (for single key)
    - plain string: 'plain value' → 'plain value'
    - list: [{'value': 'a'}, {'value': 'b'}] → 'a, b'
    """
    if spec_value is None:
        return ""
    
    if isinstance(spec_value, dict):
        # Case 1: Direct value dict {'value': 'x', 'confidence': ...}
        if 'value' in spec_value:
            val = spec_value['value']
            # Handle double-nested case
            if isinstance(val, dict) and 'value' in val:
                val = val['value']
            return str(val) if val and str(val).lower() not in ['null', 'none', 'n/a'] else ""
        
        # Case 2: Category wrapper dict - will be handled by _flatten_nested_specs
        return ""
    
    if isinstance(spec_value, list):
        values = []
        for v in spec_value:
            extracted = _extract_spec_value(v)
            if extracted:
                values.append(extracted)
        return ", ".join(values) if values else ""
    
    # Plain string
    return str(spec_value) if spec_value else ""


def _flatten_nested_specs(spec_dict: dict) -> dict:
    """
    Flatten nested specification dictionaries into simple key-value pairs.
    
    Input:  {'environmental_specs': {'temperature_range': {'value': '200°C', 'confidence': 0.9, 'note': '...'}}}
    Output: {'temperature_range': '200°C'}
    
    Input:  {'accuracy': {'value': '±0.1%', 'confidence': 0.8}}
    Output: {'accuracy': '±0.1%'}
    """
    flattened = {}
    
    for key, value in spec_dict.items():
        if isinstance(value, dict):
            # Check if this is a value dict
            if 'value' in value:
                extracted = _extract_spec_value(value)
                if extracted:
                    flattened[key] = extracted
            else:
                # This is a category wrapper - flatten its contents
                for inner_key, inner_value in value.items():
                    if isinstance(inner_value, dict) and 'value' in inner_value:
                        extracted = _extract_spec_value(inner_value)
                        if extracted:
                            flattened[inner_key] = extracted
                    elif inner_value and str(inner_value).lower() not in ['null', 'none', '']:
                        flattened[inner_key] = str(inner_value)
        elif value and str(value).lower() not in ['null', 'none', '']:
            flattened[key] = str(value)
    
    return flattened


def _merge_deep_agent_specs_for_display(items: list) -> list:
    """
    Merge Deep Agent specifications into the specifications field for UI display.
    Labels values based on source:
    - Standards -> [STANDARDS]
    - LLM/Inferred -> [INFERRED]
    - User -> (No label)
    """
    import re
    
    # Patterns that indicate raw JSON structure - EXACT matches only for these
    RAW_JSON_PATTERNS = [
        "{'value':",
        "'confidence':",
        "'note':",
        "[{",
        "{'",
    ]
    
    # Patterns that indicate the ENTIRE value is invalid (use startswith/full match)
    INVALID_START_PATTERNS = [
        "i don't have",
        "i do not have",
        "no information",
        "not found in",
        "cannot determine",
        "not available in",
    ]
    
    # Placeholder values to skip (exact matches only)
    PLACEHOLDER_VALUES = [
        "null", "none", "n/a", "na", "not applicable",
        "extracted value or null", "consolidated value or null",
        "value if applicable", "not specified", ""
    ]
    
    for item in items:
        # Get Deep Agent specs if available
        combined_specs = item.get("combined_specifications", {})
        standards_specs = item.get("standards_specifications", {})
        llm_specs = item.get("llm_generated_specs", {})
        
        # Create clean specifications for display
        display_specs = {}
        
        # Strategy:
        # 1. If combined_specs exists (Unified path), use its 'source' field
        # 2. If separate specs exist, use implicit labeling
        
        if combined_specs:
            flattened = _flatten_nested_specs(combined_specs)
            
            for key, val in flattened.items():
                if not val: continue
                
                # Check source in original dict/object so we can show (Standards) or (Inferred)
                source_label = ""
                raw = combined_specs.get(key)
                if raw is not None:
                    src = None
                    if isinstance(raw, dict):
                        src = raw.get("source", "")
                    else:
                        src = getattr(raw, "source", None) or ""
                    
                    # USER REQUEST: 
                    # 1. LLM -> "Inferred"
                    # 2. Standards -> "Standards"
                    # 3. User -> No label
                    if src == "standards":
                        source_label = " (Standards)"
                    elif src in ("llm_generated", "database"):
                        # "database" = non-standards from batch (user/LLM); show as (Inferred)
                        source_label = " (Inferred)"
                    elif src == "user_specified":
                        source_label = "" # No label
                
                # Helper to check validity
                val_str = str(val).strip()
                # Clean up any pre-existing tags
                val_str = val_str.replace("[INFERRED]", "").replace("[STANDARDS]", "").replace("(Inferred)", "").replace("(Standards)", "").replace("(LLM)", "").replace("(USER)", "").strip()
                val_lower = val_str.lower()
                
                if any(p in val_lower for p in RAW_JSON_PATTERNS): continue
                if any(val_lower.startswith(p) for p in INVALID_START_PATTERNS): continue
                if val_lower in PLACEHOLDER_VALUES: continue
                
                display_specs[key] = f"{val_str}{source_label}"

        elif standards_specs:
            # Fallback: All are standards
            flattened = _flatten_nested_specs(standards_specs)
            for key, val in flattened.items():
                if not val: continue
                if str(val).lower() in PLACEHOLDER_VALUES: continue
                display_specs[key] = f"{str(val)} (Standards)"
                
        elif llm_specs:
            # Fallback: All are inferred
            flattened = _flatten_nested_specs(llm_specs)
            for key, val in flattened.items():
                if not val: continue
                if str(val).lower() in PLACEHOLDER_VALUES: continue
                display_specs[key] = f"{str(val)} (Inferred)"
        
        # Merge with any existing specs (User specs usually here)
        original_specs = item.get("specifications", {})
        if isinstance(original_specs, dict):
             # Original specs typically come from "Identify Instrument" prompt which are "User/Inferred"
             # But if they clash with display_specs (which came from enrichment), ENRICHMENT wins for display usually?
             # Actually, user specs should ALREADY be in combined_specs if enrichment ran.
             # So we only add original specs if they are NOT in display_specs.
             
             for key, val in original_specs.items():
                 # Key normalization for comparison
                 key_clean = key.lower().replace("_", "").replace(" ", "")
                 existing_keys = [k.lower().replace("_", "").replace(" ", "") for k in display_specs.keys()]
                 
                 if key_clean not in existing_keys:
                     # FIX: Extract clean value from nested dicts instead of str()
                     # This prevents raw dict text like "{'value': 'x', 'confidence': 0.9}" in UI
                     if isinstance(val, (dict, list)):
                         val_str = _extract_spec_value(val)
                     else:
                         val_str = str(val).strip()
                     # Clean up any pre-existing tags to avoid double labeling
                     val_str = val_str.replace("[INFERRED]", "").replace("[STANDARDS]", "").replace("(Inferred)", "").replace("(Standards)", "").replace("(LLM)", "").replace("(USER)", "").strip()
                     
                     if val_str and val_str.lower() not in PLACEHOLDER_VALUES:
                         display_specs[key] = val_str # No label for user/original specs
        
        # Update item
        if display_specs:
            item["specifications"] = display_specs
    
    return items




def format_selection_list_node(state: InstrumentIdentifierState) -> InstrumentIdentifierState:
    """
    Node 3: Format Selection List.
    Formats identified items into user-friendly numbered list with sample_inputs.
    This is the FINAL node - workflow ends here and waits for user selection.
    """
    logger.info("[IDENTIFIER] Node 3: Formatting selection list...")

    # Defensive check: ensure all_items exists
    if "all_items" not in state or not state["all_items"]:
        logger.warning("[IDENTIFIER] No items found, creating fallback response")
        state["response"] = "I couldn't identify any instruments or accessories. Please provide more details about your requirements."
        state["response_data"] = {
            "workflow": "instrument_identifier",
            "items": [],
            "total_items": 0,
            "awaiting_selection": False,
            "error": state.get("error", "No items identified")
        }
        state["current_step"] = "complete"
        return state

    # === DEBUG: Print specification source analysis ===
    try:
        from debug_spec_tracker import print_specification_report
        if state.get("all_items"):
            print_specification_report(state["all_items"])
    except Exception as debug_err:
        logger.debug(f"[IDENTIFIER] Debug tracker not available: {debug_err}")
    # === END DEBUG ===

    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )


        prompt = ChatPromptTemplate.from_template(get_instrument_list_prompt())
        parser = JsonOutputParser()
        chain = prompt | llm | parser

        result = chain.invoke({
            "instruments": json.dumps(state["identified_instruments"], indent=2),
            "accessories": json.dumps(state["identified_accessories"], indent=2),
            "project_name": state.get("project_name", "Project")
        })

        # Format response for user
        response_lines = [
            f"**📋 {state.get('project_name', 'Project')} - Identified Items**\n",
            f"I've identified **{state['total_items']} items** for your project:\n"
        ]

        formatted_list = result.get("formatted_list", [])

        for item in formatted_list:
            emoji = "🔧" if item.get("type") == "instrument" else "🔩"
            response_lines.append(
                f"{item['number']}. {emoji} **{item.get('name', 'Unknown')}** ({item.get('category', '')})"
            )
            response_lines.append(
                f"   Quantity: {item.get('quantity', 1)}"
            )

            # Show sample_input preview (first 80 chars)
            # Find the actual item from all_items to get the full sample_input
            actual_item = next((i for i in state["all_items"] if i["number"] == item["number"]), None)
            if actual_item and actual_item.get("sample_input"):
                sample_preview = actual_item["sample_input"][:80] + "..." if len(actual_item["sample_input"]) > 80 else actual_item["sample_input"]
                response_lines.append(
                    f"   🔍 Search query: {sample_preview}"
                )

            response_lines.append("")

        response_lines.append(
            f"\n**📌 Next Steps:**\n"
            f"Reply with an item number (1-{state['total_items']}) to search for vendor products.\n"
            f"I'll then find specific product recommendations for your selected item."
        )

        state["response"] = "\n".join(response_lines)
        
        # === MERGE DEEP AGENT SPECS FOR UI DISPLAY ===
        # This ensures Deep Agent specifications are shown in UI with [STANDARDS] labels
        _merge_deep_agent_specs_for_display(state["all_items"])
        logger.info(f"[IDENTIFIER] Merged Deep Agent specs for {len(state['all_items'])} items")
        # === END MERGE ===

        # === REGENERATE SAMPLE_INPUT WITH ENRICHED SPECS ===
        # After enrichment merges 50-80 specs, the sample_input (created in Node 2
        # with only ~5 initial specs) is stale. Regenerate it so the product search
        # query includes ALL enriched specification values.
        for item in state["all_items"]:
            item["sample_input"] = _ensure_full_spec_coverage(
                item.get("sample_input", ""),
                item,
                state.get("project_name", "Project")
            )
        logger.info(f"[IDENTIFIER] Regenerated sample_input with enriched specs for {len(state['all_items'])} items")
        # === END REGENERATE SAMPLE_INPUT ===

        # === FETCH GENERIC IMAGES FOR ALL ITEMS ===
        # This ensures each identified item has a generic product image for UI display
        try:
            from services.azure.image_utils import fetch_generic_images_batch

            # Collect unique product types for batch fetch (use name or category)
            product_types_map = {}  # Maps product_type -> list of item indices
            for idx, item in enumerate(state["all_items"]):
                name = item.get("name", "").strip()
                category = item.get("category", "").strip()
                
                # Combine category and name for context-rich generation (e.g., "Electrical Accessories - Terminal Head")
                # This helps disambiguate generic names like "Head" or "Box"
                if name and category and category.lower() not in name.lower() and name.lower() not in category.lower():
                    product_type = f"{category} - {name}"
                else:
                    product_type = name or category

                if product_type:
                    product_type = product_type.strip()
                    if product_type not in product_types_map:
                        product_types_map[product_type] = []
                    product_types_map[product_type].append(idx)

            if product_types_map:
                product_types = list(product_types_map.keys())
                logger.info(f"[IDENTIFIER] Fetching generic images for {len(product_types)} unique product types...")

                # Batch fetch images (uses parallel Azure cache checks + sequential LLM generation)
                image_results = fetch_generic_images_batch(product_types, max_parallel_cache_checks=5)

                # Attach image URLs to items
                images_attached = 0
                for product_type, item_indices in product_types_map.items():
                    image_data = image_results.get(product_type)
                    if image_data and image_data.get("url"):
                        image_url = image_data["url"]
                        for idx in item_indices:
                            state["all_items"][idx]["imageUrl"] = image_url
                            state["all_items"][idx]["image_url"] = image_url  # Snake_case for compatibility
                            images_attached += 1

                logger.info(f"[IDENTIFIER] Attached generic images to {images_attached}/{len(state['all_items'])} items")
            else:
                logger.warning("[IDENTIFIER] No product types found for image fetching")

        except ImportError as ie:
            logger.warning(f"[IDENTIFIER] Image fetching skipped - module not available: {ie}")
        except Exception as img_err:
            logger.warning(f"[IDENTIFIER] Image fetching failed (non-critical): {img_err}")
            # Continue without images - this is non-critical
        # === END FETCH GENERIC IMAGES ===

        state["response_data"] = {
            "workflow": "instrument_identifier",
            "project_name": state["project_name"],
            "items": state["all_items"],  # Full items with merged specs and images
            "total_items": state["total_items"],
            "awaiting_selection": True,
            "instructions": f"Reply with item number (1-{state['total_items']}) to get product recommendations"
        }

        state["current_step"] = "complete"

        logger.info(f"[IDENTIFIER] Selection list formatted with {state['total_items']} items")

    except Exception as e:
        logger.error(f"[IDENTIFIER] List formatting failed: {e}")
        state["error"] = str(e)

        # Fallback: Create simple text list
        response_lines = [
            f"**📋 {state.get('project_name', 'Project')} - Identified Items**\n"
        ]

        for item in state["all_items"]:
            emoji = "🔧" if item.get("type") == "instrument" else "🔩"
            response_lines.append(
                f"{item['number']}. {emoji} {item.get('name', 'Unknown')} ({item.get('category', '')})"
            )

        response_lines.append(
            f"\nReply with an item number (1-{state['total_items']}) to search for products."
        )

        state["response"] = "\n".join(response_lines)

        # === FALLBACK: FETCH GENERIC IMAGES ===
        # Try to attach images even in fallback path
        try:
            from services.azure.image_utils import fetch_generic_images_batch

            all_items = state.get("all_items", [])
            product_types_map = {}
            for idx, item in enumerate(all_items):
                product_type = item.get("name") or item.get("category")
                if product_type:
                    product_type = product_type.strip()
                    if product_type not in product_types_map:
                        product_types_map[product_type] = []
                    product_types_map[product_type].append(idx)

            if product_types_map:
                image_results = fetch_generic_images_batch(list(product_types_map.keys()))
                for product_type, item_indices in product_types_map.items():
                    image_data = image_results.get(product_type)
                    if image_data and image_data.get("url"):
                        for idx in item_indices:
                            all_items[idx]["imageUrl"] = image_data["url"]
                            all_items[idx]["image_url"] = image_data["url"]
                logger.info(f"[IDENTIFIER] Fallback: Attached images to items")
        except Exception as fallback_img_err:
            logger.warning(f"[IDENTIFIER] Fallback image fetching failed: {fallback_img_err}")
        # === END FALLBACK IMAGE FETCH ===

        state["response_data"] = {
            "workflow": "instrument_identifier",
            "items": state["all_items"],
            "total_items": state["total_items"],
            "awaiting_selection": True
        }

    return state


# ============================================================================
# STANDARDS DETECTION NODE (NEW - CONDITIONAL ENRICHMENT)
# ============================================================================

def detect_standards_requirements_node(state: InstrumentIdentifierState) -> InstrumentIdentifierState:
    """
    Node 3.5: Standards Detection for Conditional Enrichment.

    Scans user_input + provided_requirements for standards indicators.
    Sets standards_detected flag to skip expensive Phase 3 enrichment if not needed.

    Detection sources:
    - Text keywords: SIL, ATEX, IEC, ISO, API, hazardous, explosion-proof
    - Domain keywords: Oil & Gas, Pharma, Chemical
    - Provided requirements: sil_level, hazardous_area, domain, industry
    - Critical specs: Temperature ranges, pressure ranges

    This saves 5-8 seconds per request when standards detection is not needed.
    """
    logger.info("[IDENTIFIER] Node 3.5: Detecting standards requirements...")

    user_input = state.get("user_input", "")
    provided_requirements = state.get("provided_requirements", {})

    # Run detection
    detection_result = detect_standards_indicators(
        user_input=user_input,
        provided_requirements=provided_requirements
    )

    # Store detection results in state
    state["standards_detected"] = detection_result["detected"]
    state["standards_confidence"] = detection_result["confidence"]
    state["standards_indicators"] = detection_result["indicators"]

    if detection_result["detected"]:
        logger.info(
            f"[IDENTIFIER] Standards DETECTED (confidence={detection_result['confidence']:.2f}, "
            f"indicators={len(detection_result['indicators'])})"
        )
    else:
        logger.info("[IDENTIFIER] No standards detected - will skip Phase 3 deep enrichment")

    state["current_step"] = "enrich_with_standards"
    return state


# ============================================================================
# STANDARDS RAG ENRICHMENT NODE (BATCH OPTIMIZED)
# ============================================================================

def enrich_with_standards_node(state: InstrumentIdentifierState) -> InstrumentIdentifierState:
    """
    Node 4: Parallel 3-Source Specification Enrichment.
    
    Runs 3 specification sources in PARALLEL:
    1. USER_SPECIFIED: Extract explicit specs from user input (MANDATORY)
    2. LLM_GENERATED: Generate specs for each product type
    3. STANDARDS_BASED: Extract from standards documents
    
    All results stored in Deep Agent Memory, deduplicated, and merged.
    
    SPECIFICATION PRIORITY:
    - User-specified specs are MANDATORY (never overwritten)
    - Standards specs take precedence over LLM for conflicts
    - Non-conflicting specs from all sources are included
    
    Adds to each item:
    - user_specified_specs: Explicit specs from user (MANDATORY)
    - llm_generated_specs: LLM-generated specs for product type
    - standards_specifications: Specs from standards documents
    - combined_specifications: Final merged specs with source metadata
    - specifications: Flattened values for backward compatibility
    """
    logger.info("[IDENTIFIER] Node 4: Running PARALLEL 3-Source Specification Enrichment...")
    
    all_items = state.get("all_items", [])
    user_input = state.get("user_input", "")
    
    if not all_items:
        logger.warning("[IDENTIFIER] No items to enrich")
        return state
    
    try:
        # Import OPTIMIZED Parallel Enrichment (uses shared LLM + true parallel products)
        from agentic.deep_agent.processing.parallel.optimized_agent import run_optimized_parallel_enrichment
        
        logger.info(f"[IDENTIFIER] Starting OPTIMIZED parallel enrichment for {len(all_items)} items...")
        
        # =====================================================
        # OPTIMIZED PARALLEL ENRICHMENT
        # - Single shared LLM instance (no repeated test calls)
        # - All products processed in parallel
        # - Significant time savings vs sequential processing
        # =====================================================
        standards_detected = state.get("standards_detected", True)  # Default True for safety
        result = run_optimized_parallel_enrichment(
            items=all_items,
            user_input=user_input,
            session_id=state.get("session_id", "identifier-opt-parallel"),
            domain_context=None,
            safety_requirements=None,
            standards_detected=standards_detected,  # NEW: Pass detection flag
            max_parallel_products=5  # Process up to 5 products simultaneously
        )
        
        if result.get("success"):
            enriched_items = result.get("items", all_items)
            metadata = result.get("metadata", {})
            
            processing_time = metadata.get("processing_time_ms", 0)
            thread_results = metadata.get("thread_results", {})
            
            logger.info(f"[IDENTIFIER] Parallel enrichment complete in {processing_time}ms")
            logger.info(f"[IDENTIFIER] Thread results: {thread_results}")
            
            # Extract applicable standards and certifications for each item
            for item in enriched_items:
                if item.get("standards_info", {}).get("enrichment_status") == "success":
                    # Add extracted standards and certifications
                    standards_specs = item.get("standards_specifications", {})
                    item["standards_info"]["applicable_standards"] = _extract_standards_from_specs(
                        {"specifications": standards_specs}
                    )
                    item["standards_info"]["certifications"] = _extract_certifications_from_specs(
                        {"specifications": standards_specs}
                    )
            
            state["all_items"] = enriched_items
            
            # Count spec sources
            total_user_specs = sum(len(item.get("user_specified_specs", {})) for item in enriched_items)
            total_llm_specs = sum(len(item.get("llm_generated_specs", {})) for item in enriched_items)
            total_std_specs = sum(len(item.get("standards_specifications", {})) for item in enriched_items)
            
            state["messages"] = state.get("messages", []) + [{
                "role": "system",
                "content": f"Parallel 3-source enrichment complete: {total_user_specs} user specs (MANDATORY), {total_llm_specs} LLM specs, {total_std_specs} standards specs in {processing_time}ms"
            }]
            
            logger.info(f"[IDENTIFIER] Spec breakdown: {total_user_specs} user, {total_llm_specs} LLM, {total_std_specs} standards")
            
        else:
            error = result.get("error", "Unknown error")
            logger.error(f"[IDENTIFIER] Parallel enrichment failed: {error}")
            state["all_items"] = result.get("items", all_items)
            
            state["messages"] = state.get("messages", []) + [{
                "role": "system",
                "content": f"Parallel enrichment failed (non-critical): {error}"
            }]
        
    except ImportError as ie:
        logger.error(f"[IDENTIFIER] Parallel enrichment import failed: {ie}")
        # Fallback to existing Standards RAG enrichment
        try:
            enriched_items = enrich_identified_items_with_standards(
                items=all_items,
                product_type=None,
                domain=None,
                top_k=3
            )
            state["all_items"] = enriched_items
            logger.info("[IDENTIFIER] Fallback to existing Standards RAG enrichment")
        except Exception as e2:
            logger.error(f"[IDENTIFIER] Fallback enrichment also failed: {e2}")
            
    except Exception as e:
        logger.error(f"[IDENTIFIER] Specification enrichment failed: {e}")
        import traceback
        traceback.print_exc()
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Specification enrichment failed (non-critical): {str(e)}"
        }]
    
    state["current_step"] = "format_list"
    return state


def _extract_standards_from_specs(final_specs: dict) -> list:
    """Extract applicable standards codes from Deep Agent results."""
    standards = []
    constraints = final_specs.get("constraints_applied", [])
    for c in constraints:
        if isinstance(c, dict):
            ref = c.get("standard_reference") or c.get("constraint", "")
            if ref:
                # Extract standard codes like IEC, ISO, API
                import re
                matches = re.findall(r'\b(IEC|ISO|API|ANSI|ISA|EN|NFPA|ASME|IEEE)\s*[\d.-]+', str(ref), re.IGNORECASE)
                standards.extend(matches)
    return list(set(standards))


def _extract_certifications_from_specs(final_specs: dict) -> list:
    """Extract certifications from Deep Agent results."""
    certs = []
    specs = final_specs.get("specifications", {})
    for key, value in specs.items():
        key_lower = key.lower()
        value_str = str(value).upper()
        if any(x in key_lower for x in ["cert", "sil", "atex", "rating", "approval"]):
            certs.append(f"{key}: {value}")
        # Also check for certification patterns in values
        import re
        cert_patterns = re.findall(r'\b(SIL\s*[1-4]|ATEX|IECEx|CE|UL|CSA|FM|IP\d{2})', value_str)
        certs.extend(cert_patterns)
    return list(set(certs))


# ============================================================================
# SPECIFICATION VERIFICATION & ITERATIVE ENRICHMENT
# ============================================================================

# Target: every item must have at least this many valid KV-pair specs
_MIN_SPEC_TARGET = 60
# Maximum enrichment iterations in the verification gate
_MAX_ENRICHMENT_ITERATIONS = 3
# Hard limit: never exceed this many specs per item (prevent explosion)
_MAX_ENRICHMENT_SPECS = 100


def _count_valid_specs(item: dict) -> int:
    """Count valid KV-pair specifications for an item across all spec sources."""
    from agentic.deep_agent.processing.parallel.optimized_agent import is_valid_spec_value

    combined = item.get("combined_specifications", {})
    flat = _flatten_nested_specs(combined)

    count = 0
    for key, value in flat.items():
        if key.startswith('_'):
            continue
        if is_valid_spec_value(value):
            count += 1
    return count


def _enrich_single_deficient_item(item: dict, llm) -> dict:
    """
    Run up to _MAX_ENRICHMENT_ITERATIONS LLM calls to push a single item
    to _MIN_SPEC_TARGET valid specs.  Returns the (possibly mutated) item.
    """
    import json as _json
    import re as _re
    from langchain_core.messages import HumanMessage
    from agentic.deep_agent.processing.parallel.optimized_agent import (
        _clean_and_flatten_specs,
        is_valid_spec_value,
    )

    item_name = item.get("name") or item.get("product_name", "Unknown")
    category = item.get("category", "Industrial Instrument")
    context = item.get("sample_input", "")

    for iteration in range(1, _MAX_ENRICHMENT_ITERATIONS + 1):
        current_count = _count_valid_specs(item)
        deficit = _MIN_SPEC_TARGET - current_count

        # Stop condition 1: Reached target (60+ specs)
        if deficit <= 0:
            logger.info(f"[VERIFY_SPECS] {item_name}: reached {current_count} specs (target ≥{_MIN_SPEC_TARGET}), stopping enrichment")
            break

        # Stop condition 2: Hit hard limit (100 specs)
        if current_count >= _MAX_ENRICHMENT_SPECS:
            logger.info(f"[VERIFY_SPECS] {item_name}: reached hard limit of {_MAX_ENRICHMENT_SPECS} specs, stopping enrichment")
            break

        specs_needed = deficit + 5  # overshoot buffer

        # Collect current keys from combined_specifications
        combined = item.get("combined_specifications", {})
        flat = _flatten_nested_specs(combined)
        existing_keys = [k for k in flat.keys() if not k.startswith('_')]

        prompt_text = (
            f"You are generating additional technical specifications for a {item_name}.\n\n"
            f"Category: {category}\n"
            f"Context: {context}\n\n"
            f"Existing specification keys already present:\n{existing_keys}\n\n"
            f"Generate {specs_needed} additional REALISTIC technical specifications "
            f"that are NOT in the list above.\n\n"
            f"IMPORTANT:\n"
            f"1. Only include specifications with concrete, measurable values (not \"Not specified\" or \"N/A\")\n"
            f"2. Specifications must be technically relevant to {item_name}\n"
            f"3. Use standard industry naming conventions for keys\n"
            f"4. DO NOT repeat any key already in the existing list\n"
            f"5. Return ONLY clean technical values — no descriptions or explanations\n\n"
            f"Focus on:\n"
            f"- Performance specifications (accuracy, repeatability, response time, stability)\n"
            f"- Environmental specifications (temperature, humidity, vibration, shock)\n"
            f"- Compliance and certifications (SIL, ATEX, CE, UL, FM, CSA)\n"
            f"- Physical specifications (weight, dimensions, materials, mounting)\n"
            f"- Electrical specifications (supply voltage, power consumption, output signal, isolation)\n"
            f"- Process specifications (connection type, wetted materials, pressure rating, flow)\n"
            f"- Maintenance specifications (MTBF, calibration interval, warranty)\n\n"
            f"Return ONLY valid JSON with this exact format:\n"
            f'{{\n'
            f'    "specifications": {{\n'
            f'        "specification_key": {{"value": "concrete_value", "confidence": 0.7}}\n'
            f'    }}\n'
            f'}}'
        )

        try:
            response = llm.invoke([HumanMessage(content=prompt_text)])
            response_text = response.content if hasattr(response, 'content') else str(response)

            json_match = _re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                logger.warning(f"[VERIFY_SPECS] {item_name} iter {iteration}: no JSON in response")
                continue

            iter_result = _json.loads(json_match.group())
            raw_specs = iter_result.get("specifications", {})
            clean_specs = _clean_and_flatten_specs(raw_specs)

            # Merge into combined_specifications (top-level, no nesting)
            combined = item.setdefault("combined_specifications", {})
            new_count = 0
            for key, value in clean_specs.items():
                if key not in combined and not key.startswith('_'):
                    combined[key] = value
                    new_count += 1

            # Also mirror into specifications for downstream consumers
            specs = item.setdefault("specifications", {})
            for key, value in clean_specs.items():
                if key not in specs and not key.startswith('_'):
                    specs[key] = value

            logger.info(
                f"[VERIFY_SPECS] {item_name} iter {iteration}: "
                f"+{new_count} new specs, total now {_count_valid_specs(item)}"
            )

            if new_count == 0:
                logger.warning(f"[VERIFY_SPECS] {item_name} iter {iteration}: zero new specs, stopping early")
                break

        except Exception as e:
            logger.error(f"[VERIFY_SPECS] {item_name} iter {iteration} failed: {e}")
            break

    return item


def verify_and_enrich_specs_node(state: InstrumentIdentifierState) -> InstrumentIdentifierState:
    """
    Post-enrichment verification gate.

    Checks every item in state["all_items"] for ≥ 60 valid KV-pair specs.
    Items below the threshold are enriched iteratively (up to 3 rounds of
    parallel LLM calls) before the workflow proceeds to format_list.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from agentic.deep_agent.processing.parallel.optimized_agent import get_shared_llm

    logger.info("[VERIFY_SPECS] Starting post-enrichment specification verification...")

    all_items = state.get("all_items", [])
    if not all_items:
        logger.info("[VERIFY_SPECS] No items to verify")
        return state

    # --- Pass 1: audit counts, identify deficient items ---
    item_counts = {}
    deficient_items = []
    for item in all_items:
        name = item.get("name") or item.get("product_name", "Unknown")
        count = _count_valid_specs(item)
        item_counts[name] = count
        if count < _MIN_SPEC_TARGET:
            deficient_items.append(item)

    logger.info(f"[VERIFY_SPECS] Audit: {item_counts}")

    if not deficient_items:
        logger.info("[VERIFY_SPECS] All items verified ≥60 specs — no enrichment needed")
        return state

    logger.info(
        f"[VERIFY_SPECS] {len(deficient_items)}/{len(all_items)} items below "
        f"{_MIN_SPEC_TARGET} specs — starting iterative enrichment"
    )

    # --- Pass 2: parallel iterative enrichment ---
    try:
        llm = get_shared_llm(temperature=0.3)

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_item = {
                executor.submit(_enrich_single_deficient_item, item, llm): item
                for item in deficient_items
            }
            for future in as_completed(future_to_item):
                try:
                    future.result()  # item is mutated in-place; result is the same ref
                except Exception as e:
                    item = future_to_item[future]
                    name = item.get("name") or item.get("product_name", "Unknown")
                    logger.error(f"[VERIFY_SPECS] Enrichment failed for {name}: {e}")

    except Exception as e:
        logger.error(f"[VERIFY_SPECS] Failed to obtain shared LLM: {e}")

    # --- Pass 3: final audit & mark still-deficient items ---
    final_counts = {}
    for item in all_items:
        name = item.get("name") or item.get("product_name", "Unknown")
        count = _count_valid_specs(item)
        final_counts[name] = count
        if count < _MIN_SPEC_TARGET:
            item["enrichment_source"] = "spec_verified_partial"

    logger.info(f"[VERIFY_SPECS] Final counts: {final_counts}")

    state["current_step"] = "verify_specs"
    return state


# ============================================================================
# WORKFLOW CREATION
# ============================================================================

def create_instrument_identifier_workflow() -> StateGraph:
    """
    Create the Instrument Identifier Workflow.

    This is a 6-node workflow that identifies instruments/accessories,
    enriches them with specifications, verifies every item hits 60+ valid
    KV-pair specs, and then presents them for user selection.
    It does NOT perform product search — that is handled by the SOLUTION workflow.

    Flow:
    1. Initial Intent Classification
    2. Instrument/Accessory Identification (with sample_input generation)
    3. Standards Detection (conditional enrichment flag)
    4. Standards RAG Enrichment (3-source parallel: user → standards → LLM)
    5. Spec Verification & Iterative Enrichment (≥60 KV-pair gate)
    6. Format Selection List

    After this workflow completes, user selects an item, and the sample_input
    is routed to the SOLUTION workflow for product search.
    """

    workflow = StateGraph(InstrumentIdentifierState)

    # Add 6 nodes
    workflow.add_node("classify_intent", classify_initial_intent_node)
    workflow.add_node("identify_items", identify_instruments_and_accessories_node)
    workflow.add_node("detect_standards", detect_standards_requirements_node)
    workflow.add_node("enrich_with_standards", enrich_with_standards_node)
    workflow.add_node("verify_specs", verify_and_enrich_specs_node)  # 60+ KV verification gate
    workflow.add_node("format_list", format_selection_list_node)

    # Set entry point
    workflow.set_entry_point("classify_intent")

    # Linear flow: classify → identify → detect → enrich → verify → format
    workflow.add_edge("classify_intent", "identify_items")
    workflow.add_edge("identify_items", "detect_standards")
    workflow.add_edge("detect_standards", "enrich_with_standards")
    workflow.add_edge("enrich_with_standards", "verify_specs")  # verification gate before formatting
    workflow.add_edge("verify_specs", "format_list")
    workflow.add_edge("format_list", END)  # WORKFLOW ENDS HERE - waits for user selection

    return workflow


# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================

def run_instrument_identifier_workflow(
    user_input: str,
    session_id: str = "default",
    checkpointing_backend: str = "memory"
) -> Dict[str, Any]:
    """
    Run the instrument identifier workflow.

    This workflow identifies instruments/accessories and returns a selection list.
    It does NOT perform product search.

    Args:
        user_input: User's project requirements (e.g., "I need instruments for crude oil refinery")
        session_id: Session identifier
        checkpointing_backend: Backend for state persistence

    Returns:
        {
            "response": "Formatted selection list",
            "response_data": {
                "workflow": "instrument_identifier",
                "project_name": "...",
                "items": [
                    {
                        "number": 1,
                        "type": "instrument",
                        "name": "...",
                        "sample_input": "..."  # To be used for product search
                    },
                    ...
                ],
                "total_items": N,
                "awaiting_selection": True
            }
        }
    """
    try:
        logger.info(f"[IDENTIFIER] Starting workflow for session: {session_id}")
        logger.info(f"[IDENTIFIER] User input: {user_input[:100]}...")

        # Create initial state
        initial_state = create_instrument_identifier_state(user_input, session_id)

        # Create and compile workflow
        workflow = create_instrument_identifier_workflow()
        compiled = compile_with_checkpointing(workflow, checkpointing_backend)

        # Execute workflow
        result = compiled.invoke(
            initial_state,
            config={"configurable": {"thread_id": session_id}}
        )

        logger.info(f"[IDENTIFIER] Workflow completed successfully")
        logger.info(f"[IDENTIFIER] Generated {result.get('total_items', 0)} items for selection")

        return {
            "response": result.get("response", ""),
            "response_data": result.get("response_data", {})
        }

    except Exception as e:
        logger.error(f"[IDENTIFIER] Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()

        return {
            "response": f"I encountered an error while identifying instruments: {str(e)}",
            "response_data": {
                "workflow": "instrument_identifier",
                "error": str(e),
                "awaiting_selection": False
            }
        }


# ============================================================================
# WORKFLOW REGISTRATION (Level 4.5)
# ============================================================================

def _register_workflow():
    """Register this workflow with the central registry."""
    try:
        from .workflow_registry import get_workflow_registry, WorkflowMetadata, RetryPolicy, RetryStrategy
        
        get_workflow_registry().register(WorkflowMetadata(
            name="instrument_identifier",
            display_name="Instrument Identifier Workflow",
            description="Identifies single product requirements with clear specifications. Generates selection list of instruments and accessories for user to choose from. Does not perform product search - only identification.",
            keywords=[
                "transmitter", "sensor", "valve", "flowmeter", "thermocouple", "rtd",
                "pressure", "temperature", "level", "flow", "analyzer", "positioner",
                "meter", "gauge", "controller", "actuator", "detector"
            ],
            intents=["requirements", "additional_specs", "instrument", "accessories", "spec_request"],
            capabilities=[
                "single_product",
                "standards_enrichment",
                "accessory_matching",
                "selection_list",
                "keyword_fallback"
            ],
            entry_function=run_instrument_identifier_workflow,
            priority=80,  # High priority for direct product requests
            tags=["core", "identification", "products"],
            retry_policy=RetryPolicy(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_retries=3,
                base_delay_ms=1000
            ),
            min_confidence_threshold=0.5
        ))
        logger.info("[InstrumentIdentifierWorkflow] Registered with WorkflowRegistry")
    except ImportError as e:
        logger.debug(f"[InstrumentIdentifierWorkflow] Registry not available: {e}")
    except Exception as e:
        logger.warning(f"[InstrumentIdentifierWorkflow] Failed to register: {e}")

# Register on module load
_register_workflow()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'InstrumentIdentifierState',
    'create_instrument_identifier_workflow',
    'run_instrument_identifier_workflow'
]
