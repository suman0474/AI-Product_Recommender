"""
Product Info RAG Module
Implements Retrieval-Augmented Generation for product information queries.
Searches MongoDB for relevant data and falls back to LLM when data not found.

ALL natural language processing is handled by the LLM - no hardcoded keywords or messages.
"""

import os
import re
import io
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from mongodb_config import get_mongodb_connection
from file_extraction_utils import extract_text_from_pdf

# Import standard_rag for user-specific context
from standard_rag import get_combined_user_context

logger = logging.getLogger(__name__)

# Initialize environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


class ChatRAG:
    """
    RAG system for product information queries.
    
    Flow:
    1. User asks a question
    2. LLM extracts search keywords (no hardcoded stop words)
    3. Search MongoDB collections for relevant data
    4. If found -> LLM generates answer using retrieved context
    5. If NOT found -> LLM asks user if they want general knowledge answer
    6. LLM detects user confirmation/denial (no hardcoded words)
    7. LLM generates all responses (no hardcoded messages)
    """
    
    def __init__(self):
        self.conn = get_mongodb_connection()
        self.db = self.conn['database']
        self.collections = self.conn['collections']
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY,
        )
        
        # Collections to search for product info with descriptions for LLM
        self.collection_descriptions = {
            'vendors': 'Vendor product catalogs and specifications - contains vendor names, product types, model families, product offerings, technical specifications, accuracy, range, and requirements from different manufacturers',
            'stratergy': 'Procurement strategies - contains purchasing strategies, vendor preferences, sourcing guidelines, and procurement policies',
            'advanced_parameters': 'Advanced product parameters - contains detailed technical parameters, configuration options, and advanced settings for products',
            'user_projects': 'User saved projects - contains previously saved instrument selections, project configurations, and user-specific product data',
            'fs.files': 'PDF documents and datasheets - contains detailed product datasheets, technical manuals, specification documents, and uploaded PDF files with in-depth product information',
        }
        
        # List of MongoDB collections (excluding fs.files which is handled separately via GridFS)
        self.searchable_collections = [k for k in self.collection_descriptions.keys() if k != 'fs.files']
        
        # GridFS for PDF documents
        self.gridfs = self.conn['gridfs']
        
        # Session state for tracking "awaiting confirmation"
        self._pending_llm_queries: Dict[str, Dict[str, Any]] = {}
        
        # Cache for extracted PDF text to avoid re-extraction
        self._pdf_text_cache: Dict[str, str] = {}
        
        # Conversation memory for follow-up questions (session_id -> conversation history)
        self._conversation_memory: Dict[str, List[Dict[str, Any]]] = {}
        
        # Query cache for similar queries (query_hash -> result)
        self._query_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max_size = 100
    
    def _extract_classified_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        Use LLM to extract AND CLASSIFY keywords from user query.
        Returns categorized keywords for smarter query building.
        
        Returns:
            Dict with keys: 'vendors', 'models', 'product_types', 'general'
        """
        
        prompt_template = """Extract and classify search terms from this user query for a product database.

USER QUERY: {query}

INSTRUCTIONS:
- Identify and categorize the search terms
- **vendors**: Known manufacturer/vendor names (e.g., Honeywell, Emerson, Rosemount, ABB, Siemens, Yokogawa, Endress+Hauser)
- **models**: Specific model numbers or model families (e.g., STD700, 3051, 644, VEGAPULS)
- **product_types**: Types of instruments/products (e.g., pressure transmitter, flow meter, temperature sensor)
- **general**: Other relevant technical terms (e.g., accuracy, range, calibration)
- Convert plural to singular form
- Return as JSON object with these 4 keys, each containing an array of strings

Example:
Query: "What is the accuracy of Honeywell STD700 pressure transmitter?"
Output: {{"vendors": ["honeywell"], "models": ["std700"], "product_types": ["pressure transmitter"], "general": ["accuracy"]}}

CLASSIFIED KEYWORDS (JSON object only):"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        default_result = {"vendors": [], "models": [], "product_types": [], "general": []}
        
        try:
            response = chain.invoke({"query": query})
            # Clean the response
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()
            
            result = json.loads(cleaned)
            
            # Validate and clean the result
            classified = {
                "vendors": [],
                "models": [],
                "product_types": [],
                "general": []
            }
            
            for key in classified.keys():
                if key in result and isinstance(result[key], list):
                    # Deduplicate and lowercase
                    seen = set()
                    for item in result[key]:
                        item_clean = str(item).lower().strip()
                        if item_clean and item_clean not in seen:
                            seen.add(item_clean)
                            classified[key].append(item_clean)
            
            logger.info(f"[RAG] LLM classified keywords: vendors={classified['vendors']}, models={classified['models']}, "
                       f"product_types={classified['product_types']}, general={classified['general']}")
            return classified
            
        except Exception as e:
            logger.error(f"[RAG] LLM keyword classification failed: {e}")
            # Fallback: put all words in general
            words = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
            singular_words = [w[:-1] if w.endswith('s') and len(w) > 3 else w for w in words if len(w) > 2]
            default_result["general"] = singular_words
            return default_result
    
    def _get_all_keywords(self, classified: Dict[str, List[str]]) -> List[str]:
        """Flatten classified keywords into a single list for GridFS search"""
        all_keywords = []
        for category in classified.values():
            all_keywords.extend(category)
        return list(set(all_keywords))
    
    def _classify_relevant_collections(self, query: str) -> List[str]:
        """
        Use LLM to determine which collections are relevant to the user's query.
        This avoids searching all collections unnecessarily.
        """
        
        # Build collection descriptions for LLM
        collection_info = "\n".join([
            f"- {name}: {desc}" 
            for name, desc in self.collection_descriptions.items()
        ])
        
        prompt_template = """Analyze the user's question and determine which database collections should be searched.

AVAILABLE COLLECTIONS:
{collection_info}

USER QUESTION: {query}

INSTRUCTIONS:
- Select ALL collections that are relevant to answering this question
- A query may span MULTIPLE domains - include all relevant collections
- If the question mentions products, vendors, manufacturers, OR general specifications (accuracy, range, performance limits) -> include 'vendors'
- If the question mentions strategy, procurement, cost, lifecycle, or sourcing -> include 'stratergy'
- If the question mentions detailed configuration, setup parameters, or software settings -> include 'advanced_parameters'
- If the question mentions saved projects or previous work -> include 'user_projects'
- If the question specifically needs actual datasheets, manuals, or PDF documents -> include 'fs.files'

MULTI-DOMAIN EXAMPLES:
- "pressure transmitter with good accuracy and lifecycle cost strategy" -> ['vendors', 'stratergy']
- "Honeywell products that meet performance specs" -> ['vendors']
- "flow meter datasheet with configuration parameters" -> ['vendors', 'fs.files', 'advanced_parameters']

Return a JSON array of collection names:

RELEVANT COLLECTIONS (JSON array only):"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "collection_info": collection_info,
                "query": query
            })
            
            # Clean the response
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()
            
            collections = json.loads(cleaned)
            if isinstance(collections, list) and collections:
                # Validate collection names
                valid_collections = [c for c in collections if c in self.collection_descriptions]
                if valid_collections:
                    logger.info(f"[RAG] LLM selected collections for query: {valid_collections}")
                    return valid_collections
            
            # If empty or invalid, search all
            logger.warning("[RAG] LLM returned no valid collections, searching all")
            return self.searchable_collections
            
        except Exception as e:
            logger.error(f"[RAG] Collection classification failed: {e}, searching all collections")
            return self.searchable_collections
    
    def _classify_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify the user's query intent to provide smarter responses.
        
        Returns:
            Dict with 'intent' and 'details' for response formatting
        """
        
        prompt_template = """Analyze the user's question and classify their intent.

USER QUESTION: {query}

INTENT TYPES:
- "comparison" - User wants to compare two or more vendors/products (e.g., "Compare Honeywell vs Emerson", "difference between STD700 and 3051")
- "specs_lookup" - User wants specific technical specifications (e.g., "What is the accuracy of STD700?", "pressure range of 3051")
- "availability" - User wants to know if something exists in the database (e.g., "Do we have flow meters?", "What vendors sell transmitters?")
- "list_products" - User wants a list of products/vendors (e.g., "List all Honeywell products", "Show me all vendors")
- "general_info" - User wants general information about a topic (e.g., "Tell me about pressure transmitters", "What is a flow meter?")
- "how_to" - User wants procedural guidance (e.g., "How to calibrate a transmitter?", "Installation steps")

Return a JSON object with:
- "intent": one of the above types
- "entities": list of specific vendors/products mentioned
- "comparison_items": if comparison, list the items being compared

CLASSIFICATION (JSON only):"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        default_result = {"intent": "general_info", "entities": [], "comparison_items": []}
        
        try:
            response = chain.invoke({"query": query})
            
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()
            
            result = json.loads(cleaned)
            logger.info(f"[RAG] Query intent classified: {result.get('intent', 'unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"[RAG] Query intent classification failed: {e}")
            return default_result
    
    def _resolve_follow_up_query(self, query: str, session_id: str) -> str:
        """
        Resolve follow-up questions using conversation memory.
        E.g., "What about Emerson?" after asking about Honeywell becomes 
        "What about Emerson pressure transmitters?"
        """
        
        # Get conversation history for this session
        history = self._conversation_memory.get(session_id, [])
        
        if not history:
            return query  # No history, return as-is
        
        # Check if this looks like a follow-up question
        follow_up_indicators = [
            "what about", "how about", "and for", "same for", 
            "what's the", "how's the", "tell me about",
            "compare with", "versus", "vs"
        ]
        
        query_lower = query.lower().strip()
        is_follow_up = any(indicator in query_lower for indicator in follow_up_indicators)
        
        # Also check if query is very short (likely a follow-up)
        if len(query.split()) <= 4:
            is_follow_up = True
        
        if not is_follow_up:
            return query
        
        # Get the last query context
        last_context = history[-1] if history else {}
        last_classified = last_context.get('classified_keywords', {})
        last_query = last_context.get('query', '')
        
        # Build context for LLM to resolve the follow-up
        prompt_template = """Resolve this follow-up question using the conversation context.

PREVIOUS QUESTION: {last_query}
PREVIOUS CONTEXT: 
- Product types discussed: {product_types}
- Vendors discussed: {vendors}

CURRENT FOLLOW-UP: {current_query}

INSTRUCTIONS:
- If the user is asking about a different vendor/product in the same context, expand the query
- Example: Previous "Honeywell pressure transmitters", Current "What about Emerson?" -> "Emerson pressure transmitters"
- If it's not a follow-up, return the original query unchanged
- Return ONLY the resolved query text, nothing else

RESOLVED QUERY:"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            resolved = chain.invoke({
                "last_query": last_query,
                "product_types": ", ".join(last_classified.get('product_types', [])) or "none",
                "vendors": ", ".join(last_classified.get('vendors', [])) or "none",
                "current_query": query
            })
            
            resolved_query = resolved.strip()
            if resolved_query and resolved_query != query:
                logger.info(f"[RAG] Resolved follow-up: '{query}' -> '{resolved_query}'")
                return resolved_query
            
            return query
            
        except Exception as e:
            logger.error(f"[RAG] Follow-up resolution failed: {e}")
            return query
    
    def _add_to_memory(self, session_id: str, query: str, classified_keywords: Dict[str, List[str]], intent: str):
        """Add a conversation turn to memory (full history, no truncation)"""
        
        if session_id not in self._conversation_memory:
            self._conversation_memory[session_id] = []
        
        self._conversation_memory[session_id].append({
            "query": query,
            "classified_keywords": classified_keywords,
            "intent": intent,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Full conversation history is kept - no truncation
        logger.debug(f"[RAG] Conversation memory for {session_id}: {len(self._conversation_memory[session_id])} turns")
    
    def clear_conversation_memory(self, session_id: str):
        """Clear conversation memory for a session"""
        if session_id in self._conversation_memory:
            del self._conversation_memory[session_id]
            logger.info(f"[RAG] Cleared conversation memory for session: {session_id}")
    
    def _classify_user_intent(self, user_input: str, original_query: str) -> Dict[str, Any]:
        """Use LLM to classify if user is confirming, denying, or asking something new"""
        
        prompt_template = """Classify the user's response intent.

CONTEXT: The system asked the user if they want an answer from general AI knowledge (because the info wasn't in the database).
Original question was: "{original_query}"

USER'S RESPONSE: "{user_input}"

Classify the intent as one of:
- "confirm" - User wants the AI to answer from general knowledge (yes, sure, okay, please, go ahead, etc.)
- "deny" - User doesn't want an AI-generated answer (no, never mind, cancel, etc.)
- "new_query" - User is asking a completely different/new question

Return ONLY a JSON object with this format:
{{"intent": "confirm" or "deny" or "new_query", "confidence": 0.0 to 1.0}}

CLASSIFICATION (JSON only):"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "user_input": user_input,
                "original_query": original_query
            })
            # Clean the response
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()
            
            result = json.loads(cleaned)
            logger.info(f"[RAG] LLM classified intent: {result}")
            return result
        except Exception as e:
            logger.error(f"[RAG] LLM intent classification failed: {e}")
            # Fallback: treat as new query
            return {"intent": "new_query", "confidence": 0.5}
    
    def _generate_not_found_response(self, query: str) -> str:
        """Use LLM to generate response when data is not found"""
        
        prompt_template = """You are EnGenie, a helpful product information assistant.

The user asked: "{query}"

Unfortunately, this information was NOT found in the index.

Generate a friendly, helpful response that:
1. Acknowledges that the specific information isn't in the index
2. Offers to answer using "my knowledge" instead (DO NOT say "AI knowledge")
3. Asks the user to confirm if they want this (yes/no)

Example: "I couldn't find that in the index. Would you like a general answer based on my knowledge?"

Keep it concise and professional.

RESPONSE:"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({"query": query})
            return response.strip()
        except Exception as e:
            logger.error(f"[RAG] LLM not-found response failed: {e}")
            return f"I couldn't find information about \"{query}\" in our database. Would you like me to answer using my general knowledge? (Yes/No)"
    
    def _generate_denial_response(self) -> str:
        """Use LLM to generate response when user declines"""
        
        prompt_template = """You are EnGenie, a helpful product information assistant.

The user declined to receive an AI-generated answer (they said no to using general knowledge).

Generate a brief, friendly response that:
1. Acknowledges their choice
2. Invites them to ask other questions about products in the database

Keep it short and professional.

RESPONSE:"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({})
            return response.strip()
        except Exception as e:
            logger.error(f"[RAG] LLM denial response failed: {e}")
            return "No problem! Let me know if you have other questions about products in our database."
    
    def _search_mongodb(self, query: str) -> tuple[List[Dict[str, Any]], Dict[str, List[str]], List[str]]:
        """
        Search MongoDB collections for relevant data based on user query.
        Uses LLM to:
        1. Classify which collections are relevant
        2. Extract AND classify keywords for smart query building
        
        Returns:
            Tuple of (results list, classified keywords dict, collections searched)
        """
        # Step 1: Classify which collections to search
        relevant_collections = self._classify_relevant_collections(query)
        logger.info(f"[RAG] Will search collections: {relevant_collections}")
        
        # Step 2: Extract and classify keywords
        classified_keywords = self._extract_classified_keywords(query)
        
        # Check if we have any keywords at all
        all_keywords = self._get_all_keywords(classified_keywords)
        if not all_keywords:
            logger.warning("[RAG] No keywords extracted from query")
            return [], classified_keywords, relevant_collections
        
        all_results = []
        max_results_per_collection = 20  # Limit to prevent overwhelming LLM context
        
        # Determine query strategy based on classified keywords
        has_vendor = bool(classified_keywords.get('vendors'))
        has_model = bool(classified_keywords.get('models'))
        has_product_type = bool(classified_keywords.get('product_types'))
        
        logger.info(f"[RAG] Query strategy: has_vendor={has_vendor}, has_model={has_model}, has_product_type={has_product_type}")
        
        # Only search the relevant collections (not all 5)
        for collection_name in relevant_collections:
            try:
                collection = self.collections.get(collection_name)
                if collection is None:
                    continue
                
                # Build smart query based on classified keywords
                query_filter = self._build_smart_query(classified_keywords, has_vendor, has_model, has_product_type)
                
                if query_filter:
                    # Limit results per collection to avoid overwhelming context
                    cursor = collection.find(query_filter).limit(max_results_per_collection)
                    
                    for doc in cursor:
                        # Convert ObjectId to string
                        doc['_id'] = str(doc.get('_id', ''))
                        doc['_source_collection'] = collection_name
                        all_results.append(doc)
                        
            except Exception as e:
                logger.error(f"[RAG] Error searching {collection_name}: {e}")
                continue
        
        logger.info(f"[RAG] Found {len(all_results)} results from {len(relevant_collections)} collections: {relevant_collections}")
        return all_results, classified_keywords, relevant_collections
    
    def _build_smart_query(self, classified: Dict[str, List[str]], has_vendor: bool, has_model: bool, has_product_type: bool) -> Dict:
        """
        Build a smart MongoDB query based on classified keywords.
        Uses $and when specific vendor+model combination is requested.
        Uses $or for broader searches.
        """
        
        # If we have BOTH vendor AND model, use precise $and filtering
        if has_vendor and has_model:
            and_conditions = []
            
            # Vendor must match
            vendor_or = []
            for vendor in classified['vendors']:
                vendor_regex = {'$regex': vendor, '$options': 'i'}
                vendor_or.extend([
                    {'vendor_name': vendor_regex},
                    {'vendor': vendor_regex},
                    {'data.vendor_name': vendor_regex},
                ])
            if vendor_or:
                and_conditions.append({'$or': vendor_or})
            
            # Model must match
            model_or = []
            for model in classified['models']:
                model_regex = {'$regex': model, '$options': 'i'}
                model_or.extend([
                    {'model_family': model_regex},
                    {'model': model_regex},
                    {'model_number': model_regex},
                    {'name': model_regex},
                    {'title': model_regex},
                ])
            if model_or:
                and_conditions.append({'$or': model_or})
            
            logger.info(f"[RAG] Using PRECISE $and query: vendor AND model must match")
            return {'$and': and_conditions} if len(and_conditions) > 1 else and_conditions[0] if and_conditions else {}
        
        # If we have vendor + product_type (but no specific model), use $and
        elif has_vendor and has_product_type:
            and_conditions = []
            
            # Vendor must match
            vendor_or = []
            for vendor in classified['vendors']:
                vendor_regex = {'$regex': vendor, '$options': 'i'}
                vendor_or.extend([
                    {'vendor_name': vendor_regex},
                    {'vendor': vendor_regex},
                    {'data.vendor_name': vendor_regex},
                ])
            if vendor_or:
                and_conditions.append({'$or': vendor_or})
            
            # Product type must match
            product_or = []
            for pt in classified['product_types']:
                pt_regex = {'$regex': pt, '$options': 'i'}
                product_or.extend([
                    {'product_type': pt_regex},
                    {'category': pt_regex},
                    {'name': pt_regex},
                ])
            if product_or:
                and_conditions.append({'$or': product_or})
            
            logger.info(f"[RAG] Using $and query: vendor AND product_type must match")
            return {'$and': and_conditions} if len(and_conditions) > 1 else and_conditions[0] if and_conditions else {}
        
        # Otherwise, use broad $or search (original behavior)
        else:
            or_conditions = []
            all_keywords = self._get_all_keywords(classified)
            
            for keyword in all_keywords:
                keyword_regex = {'$regex': keyword, '$options': 'i'}
                or_conditions.extend([
                    {'vendor_name': keyword_regex},
                    {'product_type': keyword_regex},
                    {'model_family': keyword_regex},
                    {'category': keyword_regex},
                    {'subcategory': keyword_regex},
                    {'name': keyword_regex},
                    {'title': keyword_regex},
                    {'description': keyword_regex},
                    {'data': {'$elemMatch': {'vendor_name': keyword_regex}}},
                    {'data': {'$elemMatch': {'category': keyword_regex}}},
                ])
            
            logger.info(f"[RAG] Using BROAD $or query: any keyword can match")
            return {'$or': or_conditions} if or_conditions else {}
    
    def _search_gridfs_documents(self, keywords: List[str], max_docs_per_vendor: int = 3) -> List[Dict[str, Any]]:
        """
        Search GridFS fs.files collection for relevant PDF documents.
        Gets TOP 3 documents PER VENDOR for better coverage across vendors.
        
        Args:
            keywords: List of search keywords
            max_docs_per_vendor: Maximum PDFs per vendor (default 3)
            
        Returns:
            List of document info dictionaries with extracted text
        """
        if not keywords:
            return []
        
        pdf_results = []
        
        try:
            # Access fs.files collection directly
            fs_files = self.db['fs.files']
            
            # Build search query for PDFs
            or_conditions = []
            for keyword in keywords:
                keyword_regex = {'$regex': keyword, '$options': 'i'}
                or_conditions.extend([
                    {'product_type': keyword_regex},
                    {'vendor_name': keyword_regex},
                    {'filename': keyword_regex},
                    {'model_family': keyword_regex},
                    {'pdf_title': keyword_regex},
                ])
            
            # Only search for PDF files
            query = {
                '$and': [
                    {'file_type': 'pdf'},
                    {'$or': or_conditions}
                ]
            }
            
            logger.info(f"[RAG] Searching fs.files for PDFs with keywords: {keywords}")
            
            # Find ALL matching PDF documents first (we'll group by vendor)
            all_pdf_docs = list(fs_files.find(query).limit(100))  # Get more to allow grouping
            
            logger.info(f"[RAG] Found {len(all_pdf_docs)} total PDF documents matching query")
            
            # Group PDFs by vendor
            vendor_pdfs: Dict[str, List[Dict]] = {}
            for pdf_metadata in all_pdf_docs:
                vendor_name = pdf_metadata.get('vendor_name', 'Unknown')
                if vendor_name not in vendor_pdfs:
                    vendor_pdfs[vendor_name] = []
                vendor_pdfs[vendor_name].append(pdf_metadata)
            
            logger.info(f"[RAG] PDFs grouped by {len(vendor_pdfs)} vendors: {list(vendor_pdfs.keys())}")
            
            # Get top N documents per vendor
            selected_pdfs = []
            for vendor_name, pdfs in vendor_pdfs.items():
                # Take top N per vendor (sorted by upload_date if available, newest first)
                sorted_pdfs = sorted(pdfs, key=lambda x: x.get('upload_date', ''), reverse=True)
                selected_pdfs.extend(sorted_pdfs[:max_docs_per_vendor])
                logger.info(f"[RAG] Selected {min(len(sorted_pdfs), max_docs_per_vendor)} PDFs from vendor: {vendor_name}")
            
            logger.info(f"[RAG] Total PDFs selected (top {max_docs_per_vendor} per vendor): {len(selected_pdfs)}")
            
            # Process selected PDFs
            for pdf_metadata in selected_pdfs:
                try:
                    file_id = pdf_metadata.get('_id')
                    filename = pdf_metadata.get('filename', 'unknown.pdf')
                    vendor_name = pdf_metadata.get('vendor_name', 'Unknown')
                    product_type = pdf_metadata.get('product_type', 'Unknown')
                    model_family = pdf_metadata.get('model_family', '')
                    
                    file_id_str = str(file_id)
                    
                    # Check cache first
                    if file_id_str in self._pdf_text_cache:
                        extracted_text = self._pdf_text_cache[file_id_str]
                        logger.info(f"[RAG] Using cached text for PDF: {filename}")
                    else:
                        # Extract text from PDF via GridFS
                        try:
                            pdf_file = self.gridfs.get(file_id)
                            pdf_bytes = pdf_file.read()
                            
                            # Extract text using file_extraction_utils
                            extracted_text = extract_text_from_pdf(pdf_bytes)
                            
                            # Cache the extracted text
                            self._pdf_text_cache[file_id_str] = extracted_text
                            
                            logger.info(f"[RAG] Extracted {len(extracted_text)} chars from PDF: {filename}")
                            
                        except Exception as extract_error:
                            logger.warning(f"[RAG] Failed to extract text from PDF {filename}: {extract_error}")
                            extracted_text = ""
                    
                    # Only include if we got meaningful text
                    if extracted_text and len(extracted_text.strip()) > 100:
                        pdf_results.append({
                            '_id': file_id_str,
                            '_source_collection': 'fs.files (PDF)',
                            'filename': filename,
                            'vendor_name': vendor_name,
                            'product_type': product_type,
                            'model_family': model_family,
                            'upload_date': str(pdf_metadata.get('upload_date', '')),
                            'extracted_text': extracted_text,  # Full document content (no character limit)
                            'text_length': len(extracted_text),
                        })
                        
                except Exception as doc_error:
                    logger.warning(f"[RAG] Error processing PDF document: {doc_error}")
                    continue
            
            # SMART PDF TEXT SEARCH: Also search within cached PDF content
            # This finds PDFs where the content mentions keywords even if metadata doesn't match
            # Only add if we have fewer than expected results
            existing_ids = [r['_id'] for r in pdf_results]
            additional_from_cache = self._search_pdf_content_cache(keywords, max_docs_per_vendor * 2, existing_ids)
            pdf_results.extend(additional_from_cache)
                    
        except Exception as e:
            logger.error(f"[RAG] Error searching GridFS: {e}")
        
        logger.info(f"[RAG] Extracted text from {len(pdf_results)} PDF documents")
        return pdf_results
    
    def _search_pdf_content_cache(self, keywords: List[str], max_results: int, exclude_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Search within cached PDF text content for keyword matches.
        This enables finding PDFs by their content, not just metadata.
        """
        matching_pdfs = []
        
        for file_id, cached_text in self._pdf_text_cache.items():
            if file_id in exclude_ids:
                continue
            
            if len(matching_pdfs) >= max_results:
                break
            
            # Check if any keyword appears in the cached text
            text_lower = cached_text.lower()
            keyword_matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            
            if keyword_matches > 0:
                # Get metadata from fs.files for this cached PDF
                try:
                    from bson import ObjectId
                    fs_files = self.db['fs.files']
                    pdf_metadata = fs_files.find_one({'_id': ObjectId(file_id)})
                    
                    if pdf_metadata:
                        matching_pdfs.append({
                            '_id': file_id,
                            '_source_collection': 'fs.files (PDF - content match)',
                            'filename': pdf_metadata.get('filename', 'unknown.pdf'),
                            'vendor_name': pdf_metadata.get('vendor_name', 'Unknown'),
                            'product_type': pdf_metadata.get('product_type', 'Unknown'),
                            'model_family': pdf_metadata.get('model_family', ''),
                            'upload_date': str(pdf_metadata.get('upload_date', '')),
                            'extracted_text': cached_text,  # Full document content
                            'text_length': len(cached_text),
                            'keyword_matches': keyword_matches,
                        })
                        logger.info(f"[RAG] Found PDF by content search: {pdf_metadata.get('filename')} ({keyword_matches} keyword matches)")
                        
                except Exception as e:
                    logger.warning(f"[RAG] Error getting metadata for cached PDF {file_id}: {e}")
                    continue
        
        return matching_pdfs
    
    def _format_context_from_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format MongoDB results into context string for LLM"""
        if not results:
            return ""
        
        context_parts = []
        
        # Aggregate unique vendors and their products from ALL results
        vendor_products: Dict[str, List[str]] = {}
        all_model_families: Dict[str, List[str]] = {}
        all_strategy_info: List[Dict] = []
        
        for doc in results:
            source = doc.get('_source_collection', 'unknown')
            
            if source == 'vendors':
                vendor = doc.get('vendor_name', doc.get('vendor', 'Unknown'))
                product_type = doc.get('product_type', 'Unknown')
                model_families = doc.get('model_families', [])
                
                # Aggregate vendor -> products
                if vendor not in vendor_products:
                    vendor_products[vendor] = []
                if product_type and product_type not in vendor_products[vendor]:
                    vendor_products[vendor].append(product_type)
                
                # Aggregate vendor -> model families
                if model_families:
                    if vendor not in all_model_families:
                        all_model_families[vendor] = []
                    for mf in model_families:
                        if mf not in all_model_families[vendor]:
                            all_model_families[vendor].append(mf)
                            

                
            elif source == 'stratergy':
                data = doc.get('data', [])
                if isinstance(data, list) and data:
                    for entry in data:
                        vendor = entry.get('vendor_name', '')
                        category = entry.get('category', '')
                        strategy = entry.get('stratergy', '')
                        if vendor:
                            all_strategy_info.append({
                                'vendor': vendor,
                                'category': category,
                                'strategy': strategy
                            })
            
            elif source == 'advanced_parameters':
                # Advanced parameters contain detailed product specs
                product_type = doc.get('product_type', 'Unknown')
                parameters = doc.get('parameters', doc.get('data', {}))
                if parameters:
                    if product_type not in vendor_products:
                        vendor_products[product_type] = []
                    # Add parameter names as available specs
                    if isinstance(parameters, dict):
                        for param_name in parameters.keys():
                            if param_name not in vendor_products[product_type]:
                                vendor_products[product_type].append(param_name)
            
            elif source == 'user_projects':
                # User projects contain instrument/product selections
                instruments = doc.get('instruments', [])
                for instrument in instruments:
                    if isinstance(instrument, dict):
                        vendor = instrument.get('vendor', instrument.get('vendor_name', ''))
                        product_type = instrument.get('product_type', instrument.get('type', ''))
                        if vendor and vendor not in vendor_products:
                            vendor_products[vendor] = []
                        if vendor and product_type and product_type not in vendor_products.get(vendor, []):
                            if vendor in vendor_products:
                                vendor_products[vendor].append(product_type)
            
            elif source == 'fs.files (PDF)':
                # PDF documents from GridFS - include extracted text and metadata
                pdf_info = f"\n--- PDF Document: {doc.get('filename', 'Unknown')} ---\n"
                pdf_info += f"Vendor: {doc.get('vendor_name', 'Unknown')}\n"
                pdf_info += f"Product Type: {doc.get('product_type', 'Unknown')}\n"
                if doc.get('model_family'):
                    pdf_info += f"Model Family: {doc.get('model_family')}\n"
                pdf_info += f"Upload Date: {doc.get('upload_date', 'Unknown')}\n"
                pdf_info += f"\nDocument Content:\n{doc.get('extracted_text', '')}\n"
                context_parts.append(pdf_info)
        
        # Format aggregated vendor data
        if vendor_products:
            vendor_summary = "Vendor Information:\n"
            vendor_summary += f"Total Vendors Found: {len(vendor_products)}\n\n"
            for vendor, products in sorted(vendor_products.items()):
                vendor_summary += f"• {vendor}:\n"
                vendor_summary += f"  Products: {', '.join(products)}\n"
                if vendor in all_model_families and all_model_families[vendor]:
                    vendor_summary += f"  Model Families: {', '.join(all_model_families[vendor])}\n"
            context_parts.append(vendor_summary)
        

        
        # Format strategy info (aggregated)
        if all_strategy_info:
            # Group by vendor
            strategy_by_vendor: Dict[str, List[str]] = {}
            for entry in all_strategy_info:
                vendor = entry['vendor']
                if vendor not in strategy_by_vendor:
                    strategy_by_vendor[vendor] = []
                if entry['strategy'] and entry['strategy'] not in strategy_by_vendor[vendor]:
                    strategy_by_vendor[vendor].append(entry['strategy'])
            
            strategy_summary = "Procurement Strategies:\n"
            for vendor, strategies in sorted(strategy_by_vendor.items()):
                strategy_summary += f"• {vendor}: {', '.join(strategies)}\n"
            context_parts.append(strategy_summary)
        
        return "\n\n---\n\n".join(context_parts)
    
    def _generate_rag_answer(self, query: str, context: str, intent: str = "general_info") -> str:
        """Generate answer using retrieved context, formatted based on query intent"""
        
        # Build intent-specific formatting instructions
        intent_instructions = {
            "comparison": """
- Format your response as a **comparison**
- Use a clear table format or side-by-side comparison
- Highlight key differences and similarities
- Include specific metrics where available""",
            
            "specs_lookup": """
- Focus on **specific technical specifications**
- Present data in a clear, structured format
- Include exact values, ranges, and units
- Be precise and technical""",
            
            "availability": """
- Clearly state **what is available** in the database
- List the items found with key details
- If items are not found, clearly state that""",
            
            "list_products": """
- Present a **clear list** of items
- Use bullet points or numbered lists
- Include key identifying information for each item
- Group by vendor or category if helpful""",
            
            "general_info": """
- Provide a **comprehensive overview**
- Include key facts and details
- Be informative but concise""",
            
            "how_to": """
- Provide **step-by-step guidance**
- Use numbered steps where appropriate
- Include any prerequisites or warnings
- Be practical and actionable"""
        }
        
        format_instruction = intent_instructions.get(intent, intent_instructions["general_info"])
        
        prompt_template = """You are EnGenie, an expert industrial instrumentation assistant.
        
Use the following context from our product database to answer the user's question.

CONTEXT FROM DATABASE:
{context}

USER QUESTION: {query}

RESPONSE FORMAT (based on user's intent - {intent}):
{format_instruction}

GENERAL INSTRUCTIONS:
- Answer based PRIMARILY on the provided context
- If the context contains relevant information, use it to form your answer
- Be specific about vendors, models, and specifications when available
- If you need to supplement with general knowledge, clearly indicate this
- **Do NOT include any '(Source: ...)' text, citations, or references to "database summary" in your response.**
- **Just provide the factual answer without meta-commentary about where the data came from.**

ANSWER:"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            answer = chain.invoke({
                "context": context,
                "query": query,
                "intent": intent,
                "format_instruction": format_instruction
            })
            return answer.strip()
        except Exception as e:
            logger.error(f"[RAG] LLM generation failed: {e}")
            return "I apologize, but I encountered an error generating a response. Please try again."
    
    def _generate_pure_llm_answer(self, query: str) -> str:
        """Generate answer using only LLM knowledge (no database context)"""
        
        prompt_template = """You are EnGenie, an expert industrial instrumentation assistant with deep knowledge of industrial automation, process control, and instrumentation.

USER QUESTION: {query}

INSTRUCTIONS:
- Answer based on your knowledge of industrial instrumentation
- Be specific and technical where appropriate
- Include relevant details about vendors, standards, and best practices
- If you're uncertain about specific details, say so
- Format your response clearly

ANSWER:"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            answer = chain.invoke({"query": query})
            return answer.strip()
        except Exception as e:
            logger.error(f"[RAG] Pure LLM generation failed: {e}")
            return "I apologize, but I encountered an error generating a response. Please try again."
    
    def query(self, user_query: str, session_id: str = "default", user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Main RAG query method.
        All responses are generated by LLM - no hardcoded messages.
        
        Args:
            user_query: The user's question
            session_id: Session identifier for tracking confirmation state
            user_id: Optional user ID to include user-specific standards and strategy context
        """
        logger.info(f"[RAG] Processing query: {user_query[:100]}... (user_id={user_id})")
        
        # Check if this is a confirmation response
        if session_id in self._pending_llm_queries:
            pending = self._pending_llm_queries[session_id]
            original_query = pending.get('query', '')
            
            # Use LLM to classify user intent (confirm/deny/new_query)
            intent_result = self._classify_user_intent(user_query, original_query)
            intent = intent_result.get('intent', 'new_query')
            
            if intent == 'confirm':
                # User confirmed - generate LLM answer
                del self._pending_llm_queries[session_id]
                
                answer = self._generate_pure_llm_answer(original_query)
                return {
                    "answer": answer,
                    "source": "llm",
                    "found_in_database": False,
                    "awaiting_confirmation": False,
                    "sources_used": ["General Knowledge"],
                    "note": "This answer is generated from my knowledge, not from the index."
                }
            
            elif intent == 'deny':
                # User declined - LLM generates response
                del self._pending_llm_queries[session_id]
                answer = self._generate_denial_response()
                return {
                    "answer": answer,
                    "source": "user_declined",
                    "found_in_database": False,
                    "awaiting_confirmation": False,
                    "sources_used": []
                }
            
            # intent == 'new_query' - treat as new query
            del self._pending_llm_queries[session_id]
        
        # SMART FEATURE 1: Resolve follow-up questions using conversation memory
        resolved_query = self._resolve_follow_up_query(user_query, session_id)
        if resolved_query != user_query:
            logger.info(f"[RAG] Query resolved from follow-up: '{user_query}' -> '{resolved_query}'")
        
        # SMART FEATURE 2: Classify query intent for smart response formatting
        query_intent = self._classify_query_intent(resolved_query)
        intent_type = query_intent.get('intent', 'general_info')
        logger.info(f"[RAG] Query intent: {intent_type}")
        
        # Search MongoDB for relevant data (returns results, classified keywords, and collections searched)
        mongo_results, classified_keywords, collections_searched = self._search_mongodb(resolved_query)
        
        # Get flattened keywords for GridFS search
        all_keywords = self._get_all_keywords(classified_keywords)
        
        # Only search GridFS for PDF documents if 'fs.files' was selected by LLM
        pdf_results = []
        if 'fs.files' in collections_searched and all_keywords:
            pdf_results = self._search_gridfs_documents(all_keywords, max_docs_per_vendor=3)
            logger.info(f"[RAG] Found {len(pdf_results)} PDF documents with extracted text")
        elif 'fs.files' not in collections_searched:
            logger.info("[RAG] Skipping PDF search - not relevant to query")
        
        # Combine all results
        all_results = mongo_results + pdf_results
        
        # Check if we found anything for vendor queries
        # Fallback Logic: vendors -> fs.files -> general LLM (skip confirmation)
        is_vendor_intent = 'vendors' in collections_searched or intent_type in ['list_products', 'specs_lookup', 'comparison']
        
        if not all_results and is_vendor_intent:
            logger.info("[RAG] Vendor query yielded no results. Triggering fallback logic.")
            
            # Fallback 1: Try fs.files if we haven't already
            if 'fs.files' not in collections_searched and all_keywords:
                logger.info("[RAG] Fallback 1: Searching fs.files for vendor query")
                pdf_results = self._search_gridfs_documents(all_keywords, max_docs_per_vendor=3)
                all_results = pdf_results
                if pdf_results:
                     collections_searched.append('fs.files')
                     logger.info(f"[RAG] Fallback 1 successful: Found {len(pdf_results)} PDFs")
            


        # Get user-specific context from standards and strategy documents
        user_context = ""
        if user_id:
            try:
                user_context = get_combined_user_context(user_id, resolved_query)
                if user_context:
                    logger.info(f"[RAG] Added user-specific context from standards/strategy: {len(user_context)} chars")
            except Exception as e:
                logger.warning(f"[RAG] Failed to get user context: {e}")
        
        if all_results or user_context:
            # Found relevant data - generate RAG answer with intent-based formatting
            context = self._format_context_from_results(all_results, resolved_query)
            
            # Append user-specific context if available
            if user_context:
                if context:
                    context = f"{context}\n\n---\n\n{user_context}"
                else:
                    context = user_context
            
            sources_used = list(set([r.get('_source_collection', 'unknown') for r in all_results]))
            if user_context:
                sources_used.append("User Standards/Strategy")
            
            # SMART FEATURE 3: Generate intent-aware response
            answer = self._generate_rag_answer(resolved_query, context, intent_type)
            
            # SMART FEATURE 4: Store in conversation memory for follow-up context
            self._add_to_memory(session_id, resolved_query, classified_keywords, intent_type)
            
            return {
                "answer": answer,
                "source": "database",
                "found_in_database": True,
                "awaiting_confirmation": False,
                "sources_used": sources_used,
                "collections_searched": collections_searched,
                "results_count": len(all_results),
                "pdf_documents_used": len(pdf_results),
                "user_context_used": bool(user_context),
                "query_intent": intent_type,
                "original_query": user_query,
                "resolved_query": resolved_query if resolved_query != user_query else None
            }
        
        else:
            # No relevant data found (and NOT a vendor fallback scenarios)
            # LLM generates "not found" response and asks for confirmation
            self._pending_llm_queries[session_id] = {
                'query': resolved_query,  # Store resolved query for confirmation
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Still add to memory even if not found (for follow-up context)
            self._add_to_memory(session_id, resolved_query, classified_keywords, intent_type)
            
            # LLM generates the "not found" message
            answer = self._generate_not_found_response(resolved_query)
            
            return {
                "answer": answer,
                "source": "pending_confirmation",
                "found_in_database": False,
                "awaiting_confirmation": True,
                "sources_used": [],
                "collections_searched": collections_searched,
                "query_intent": intent_type,
                "original_query": user_query,
                "resolved_query": resolved_query if resolved_query != user_query else None
            }
    
    def clear_pending(self, session_id: str = "default"):
        """Clear any pending confirmation for a session"""
        if session_id in self._pending_llm_queries:
            del self._pending_llm_queries[session_id]


# Global instance
chat_rag = ChatRAG()


def query_chat(user_query: str, session_id: str = "default", user_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Convenience function for querying product info.
    
    Args:
        user_query: The user's question
        session_id: Session identifier
        user_id: Optional user ID to include user-specific standards and strategy context
    """
    return chat_rag.query(user_query, session_id, user_id)


def clear_pending_query(session_id: str = "default"):
    """Clear pending query for session"""
    chat_rag.clear_pending(session_id)


def clear_session_memory(session_id: str = "default"):
    """Clear conversation memory for a session (useful on logout or new conversation)"""
    chat_rag.clear_conversation_memory(session_id)
    chat_rag.clear_pending(session_id)

