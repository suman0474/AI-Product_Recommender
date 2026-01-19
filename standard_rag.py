"""
Standard RAG Module
Implements Retrieval-Augmented Generation for engineering standards queries.
Extracts and stores text from standards documents, and provides category-based lookups.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from mongodb_config import get_mongodb_connection

logger = logging.getLogger(__name__)

# Initialize environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


def extract_and_store_standards_text(file_bytes: bytes, filename: str, user_id: int) -> dict:
    """
    Extract text from a standards document and store it in MongoDB.
    This extracts the ENTIRE text content for RAG-based search later.
    
    Args:
        file_bytes: Raw file content
        filename: Original filename
        user_id: User ID for MongoDB storage
        
    Returns:
        dict with success status and document_id if stored
    """
    from file_extraction_utils import extract_text_from_file
    
    try:
        logger.info(f"[STANDARDS_TEXT] Processing file: {filename} ({len(file_bytes)} bytes)")
        
        # Extract text from file
        extraction_result = extract_text_from_file(file_bytes, filename)
        
        if not extraction_result['success']:
            logger.warning(f"[STANDARDS_TEXT] Failed to extract text from {filename}")
            return {
                "success": False,
                "error": f"Could not extract text from {extraction_result['file_type']} file",
                "file_type": extraction_result['file_type']
            }
        
        extracted_text = extraction_result['extracted_text']
        logger.info(f"[STANDARDS_TEXT] ✓ Extracted {extraction_result['character_count']} characters")
        
        # Store in MongoDB standards collection
        conn = get_mongodb_connection()
        standards_collection = conn['collections']['standards']
        
        # Check if THIS specific file already exists for this user (by filename)
        # This allows multiple different files to be stored for the same user
        existing = standards_collection.find_one({'user_id': user_id, 'filename': filename})
        
        standards_document = {
            "user_id": user_id,
            "filename": filename,
            "file_type": extraction_result['file_type'],
            "full_text": extracted_text,
            "character_count": extraction_result['character_count'],
            "uploaded_at": datetime.utcnow().isoformat()
        }
        
        if existing:
            # Update existing document
            result = standards_collection.update_one(
                {'_id': existing['_id']},
                {'$set': standards_document}
            )
            doc_id = str(existing['_id'])
            logger.info(f"[STANDARDS_TEXT] ✓ Updated existing standards document: {filename} ({doc_id})")
        else:
            # Insert new document
            result = standards_collection.insert_one(standards_document)
            doc_id = str(result.inserted_id)
            logger.info(f"[STANDARDS_TEXT] ✓ Stored new standards document: {filename} ({doc_id})")
        
        return {
            "success": True,
            "document_id": doc_id,
            "character_count": extraction_result['character_count'],
            "file_type": extraction_result['file_type']
        }
        
    except Exception as e:
        logger.error(f"[STANDARDS_TEXT] Extraction failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def get_standards_for_category(user_id: int, category: str) -> dict:
    """
    RAG function to search standards documents for specifications related to a given category.
    
    Uses LLM to extract relevant specifications from the user's standards document(s)
    based on the instrument category.
    
    Args:
        user_id: User ID to look up their standards
        category: Instrument category (e.g., "Pressure Transmitter", "Flow Meter")
        
    Returns:
        dict with specifications found in standards, or empty dict if none found
    """
    try:
        logger.info(f"[STANDARDS_RAG] Looking up standards for category '{category}' for user {user_id}")
        
        # Get user's standards document(s)
        conn = get_mongodb_connection()
        standards_collection = conn['collections']['standards']
        
        # Find ALL standards documents for this user
        cursor = standards_collection.find({'user_id': user_id})
        standards_docs = list(cursor)
        
        if not standards_docs:
            logger.info(f"[STANDARDS_RAG] No standards documents found for user {user_id}")
            return {"found": False, "specifications": {}, "raw_text": ""}
        
        # Combine text from all documents
        full_text_parts = []
        total_chars = 0
        
        for doc in standards_docs:
            text = doc.get('full_text', '')
            if text:
                filename = doc.get('filename', 'Unknown File')
                header = f"\n\n--- DOCUMENT: {filename} ---\n\n"
                full_text_parts.append(header + text)
                total_chars += len(text)
        
        full_text = "".join(full_text_parts)
            
        if not full_text:
             logger.info(f"[STANDARDS_RAG] Standards documents found but empty text for user {user_id}")
             return {"found": False, "specifications": {}, "raw_text": ""}

        logger.info(f"[STANDARDS_RAG] Found {len(standards_docs)} standards documents with {total_chars} total characters")
        
        # Use LLM to extract relevant specifications for this category
        if not GOOGLE_API_KEY:
            logger.error("[STANDARDS_RAG] GOOGLE_API_KEY not configured")
            return {"found": False, "specifications": {}, "raw_text": ""}
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY,
        )
        
        standards_rag_prompt = """
You are extracting technical specifications from an engineering standards document for: "{category}"

Standards Document:
{standards_text}

=== YOUR TASK ===
Extract ONLY actual technical specifications as key-value pairs.

**WHAT TO EXTRACT:**
- Technical parameters with specific values (e.g., "Accuracy: ±0.25%", "Body Material: 316 SS", "Output: 4-20 mA")

**WHAT NOT TO EXTRACT:**
- Data sheet requirements or field lists
- Documentation requirements
- Procedural requirements without specific values
- Lists of what should be included in documents

=== KEY NAMING RULES ===

**RULE 1: CLEAN PARAMETER NAMES ONLY**
The key must be the pure parameter name. Remove all:
- Equipment type names (the document sections like "Magnetic Flowmeter", "Differential Pressure", etc.)
- Abbreviations of equipment types (DP, RF, ESD, SIS, Cv, etc.)
- Section prefixes (Installation, Calibration, Safety, etc. when used as prefixes)

*Principle: The category is already known, so don't repeat equipment type in the key.*

**RULE 2: NO BRACKETS OR PARENTHESES IN KEYS**
Keys must never contain () or []. This includes:
- Equipment type clarifiers: (Magnetic), (DP), etc.
- Service type clarifiers: (Gas), (Liquid), etc.
- Qualifier words: (Typical), (Recommended), (Optional), (Minimum), etc.

If there are variants or qualifiers, put them in the VALUE, not the key.

*Principle: Keep keys simple and clean. Put all qualifying information in the value.*

**RULE 3: REMOVE WORD PREFIXES THAT IDENTIFY EQUIPMENT TYPE**
If a key starts with words that identify what TYPE of equipment (not what the parameter IS), remove those words.

*Principle: Ask yourself - is this word describing the parameter, or describing which equipment? If it's describing equipment, remove it.*

=== EXAMPLES ===

Extracting from document text:
- "Magnetic Flowmeter Body Material: 304/316 SS" → Key: "Body Material", Value: "304/316 SS"
- "DP Device Upstream Straight Pipe: 10 × diameter" → Key: "Upstream Straight Pipe", Value: "10 × diameter"
- "Accuracy for gas: X, for liquid: Y" → Key: "Accuracy", Value: "X for gas, Y for liquid"

=== OUTPUT FORMAT ===

Return JSON:
{{
    "found": true,
    "category_match": "<category from document>",
    "applicable_standards": ["<standards references>"],
    "specifications": {{
        "<PARAMETER_NAME>": "<value>",
        "<PARAMETER_NAME>": "<value>"
    }},
    "requirements_summary": "<1-2 sentences>"
}}

If nothing found:
{{
    "found": false,
    "specifications": {{}},
    "requirements_summary": "No standards found"
}}

JSON only, no other text.
"""
        
        # Send full document to LLM (Gemini 2.5 Flash has large context window)
        text_to_analyze = full_text
        
        prompt = ChatPromptTemplate.from_template(standards_rag_prompt)
        chain = prompt | llm | StrOutputParser()
        
        llm_response = chain.invoke({
            "category": category,
            "standards_text": text_to_analyze
        })
        
        # Clean the response
        cleaned_response = llm_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        result = json.loads(cleaned_response)
        logger.info(f"[STANDARDS_RAG] Found standards for '{category}': {result.get('found', False)}")
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"[STANDARDS_RAG] Failed to parse LLM response: {e}")
        return {"found": False, "specifications": {}, "error": "Failed to parse standards data"}
    except Exception as e:
        logger.error(f"[STANDARDS_RAG] Standards RAG failed: {e}")
        return {"found": False, "specifications": {}, "error": str(e)}


def get_standards_context_for_query(user_id: int, query: str) -> str:
    """
    Search standards document(s) for content relevant to a general query.
    Used for Product Info RAG integration.
    
    Args:
        user_id: User ID to look up their standards
        query: User's question
        
    Returns:
        Relevant context string from standards document(s), or empty string if not found
    """
    try:
        logger.info(f"[STANDARDS_RAG] Searching standards for query: '{query[:100]}...'")
        
        conn = get_mongodb_connection()
        standards_collection = conn['collections']['standards']
        
        cursor = standards_collection.find({'user_id': user_id})
        standards_docs = list(cursor)
        
        if not standards_docs:
            logger.info(f"[STANDARDS_RAG] No standards documents found for user {user_id}")
            return ""
        
        # Combine text
        full_text_parts = []
        for doc in standards_docs:
            text = doc.get('full_text', '')
            if text:
                 filename = doc.get('filename', 'Unknown')
                 full_text_parts.append(f"\n--- SOURCE: {filename} ---\n{text}")
        
        full_text = "".join(full_text_parts)

        if not full_text:
            return ""
        
        # Use LLM to find relevant sections
        if not GOOGLE_API_KEY:
            return ""
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY,
        )
        
        search_prompt = """
Analyze the following standards document text and find sections relevant to this question:
Question: "{query}"

Standards Document:
{standards_text}

Instructions:
1. Find the most relevant paragraphs or sections that could help answer the question
2. Extract up to 1000 characters of the most relevant content
3. If the document doesn't contain relevant information, return an empty string

Return ONLY the relevant text excerpt, no explanation or JSON formatting.
If nothing relevant is found, return exactly: NO_RELEVANT_CONTENT
"""
        
        # Send full document to LLM (Gemini 2.5 Flash has large context window)
        text_to_analyze = full_text
        
        prompt = ChatPromptTemplate.from_template(search_prompt)
        chain = prompt | llm | StrOutputParser()
        
        result = chain.invoke({
            "query": query,
            "standards_text": text_to_analyze
        })
        
        if result.strip() == "NO_RELEVANT_CONTENT":
            return ""
        
        logger.info(f"[STANDARDS_RAG] Found relevant standards content: {len(result)} chars")
        return result.strip()
        
    except Exception as e:
        logger.error(f"[STANDARDS_RAG] Query search failed: {e}")
        return ""


def get_strategy_context_for_query(user_id: int, query: str) -> str:
    """
    Search strategy document for content relevant to a query.
    Used for Product Info RAG integration.
    
    Args:
        user_id: User ID to look up their strategy
        query: User's question
        
    Returns:
        Relevant context string from strategy data, or empty string if not found
    """
    try:
        logger.info(f"[STRATEGY_RAG] Searching strategy for query: '{query[:100]}...'")
        
        conn = get_mongodb_connection()
        strategy_collection = conn['collections']['stratergy']
        
        # Find all strategy documents for this user
        cursor = strategy_collection.find({'user_id': user_id})
        
        strategy_entries = []
        for doc in cursor:
            if 'data' in doc and isinstance(doc['data'], list):
                for item in doc['data']:
                    strategy_entries.append({
                        "vendor_name": item.get("vendor_name", ""),
                        "category": item.get("category", ""),
                        "subcategory": item.get("subcategory", ""),
                        "strategy": item.get("stratergy", "")  # Note: typo in DB field name
                    })
        
        if not strategy_entries:
            logger.info(f"[STRATEGY_RAG] No strategy data found for user {user_id}")
            return ""
        
        # Format strategy data as context
        strategy_text = "STRATEGY DATA:\n"
        for entry in strategy_entries:
            strategy_text += f"- Vendor: {entry['vendor_name']}, Category: {entry['category']}, "
            strategy_text += f"Subcategory: {entry['subcategory']}, Strategy: {entry['strategy']}\n"
        
        # Check if query is related to strategy/vendor content
        query_lower = query.lower()
        keywords = ['vendor', 'strategy', 'supplier', 'procurement', 'sourcing', 'manufacturer']
        
        # Also check for specific vendor names in query
        vendor_names = [e['vendor_name'].lower() for e in strategy_entries if e['vendor_name']]
        category_names = [e['category'].lower() for e in strategy_entries if e['category']]
        
        is_relevant = any(kw in query_lower for kw in keywords)
        is_relevant = is_relevant or any(vn in query_lower for vn in vendor_names if vn)
        is_relevant = is_relevant or any(cn in query_lower for cn in category_names if cn)
        
        if is_relevant:
            logger.info(f"[STRATEGY_RAG] Found relevant strategy context: {len(strategy_entries)} entries")
            return strategy_text
        
        return ""
        
    except Exception as e:
        logger.error(f"[STRATEGY_RAG] Query search failed: {e}")
        return ""


def get_combined_user_context(user_id: int, query: str) -> str:
    """
    Get combined context from both standards and strategy documents for a query.
    Used for Product Info RAG integration.
    
    Args:
        user_id: User ID
        query: User's question
        
    Returns:
        Combined context string from both sources
    """
    context_parts = []
    
    # Get standards context
    standards_context = get_standards_context_for_query(user_id, query)
    if standards_context:
        context_parts.append(f"=== FROM USER'S STANDARDS DOCUMENT ===\n{standards_context}")
    
    # Get strategy context
    strategy_context = get_strategy_context_for_query(user_id, query)
    if strategy_context:
        context_parts.append(f"=== FROM USER'S STRATEGY DATA ===\n{strategy_context}")
    
    if context_parts:
        return "\n\n".join(context_parts)
    
    return ""
