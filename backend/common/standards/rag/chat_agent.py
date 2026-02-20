# agentic/standards_chat_agent.py
# Specialized Chat Agent for Standards Documentation Q&A

import json
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, field_validator

import os
# Import existing infrastructure
from common.rag.vector_store import get_vector_store
from common.services.llm.fallback import create_llm_with_fallback
from common.prompts import RAG_PROMPTS

# Import Azure Blob retriever for standards documents
try:
    from .blob_retriever import get_standards_blob_retriever
    AZURE_BLOB_AVAILABLE = True
except ImportError:
    AZURE_BLOB_AVAILABLE = False

# Import Deep Agent for parallel processing (2026-02-11)
try:
    from common.standards.generation.deep_agent import load_standard_text
    DEEP_AGENT_AVAILABLE = True
except ImportError:
    DEEP_AGENT_AVAILABLE = False
    load_standard_text = None

# Import consolidated JSON utilities
from common.utils.json_utils import (
    extract_json_from_response,
    sanitize_json_string,
    ensure_string,
    ensure_float
)
logger = logging.getLogger(__name__)


# ============================================================================
# RESPONSE VALIDATION MODELS (Pydantic)
# ============================================================================

class CitationModel(BaseModel):
    """Citation from a standards document."""
    source: str
    content: str
    relevance: float

    @field_validator('relevance')
    @classmethod
    def validate_relevance(cls, v):
        """Ensure relevance is between 0.0 and 1.0."""
        if not isinstance(v, (int, float)):
            raise ValueError('relevance must be numeric')
        if not (0.0 <= v <= 1.0):
            raise ValueError(f'relevance must be between 0.0 and 1.0, got {v}')
        return float(v)


class StandardsRAGResponse(BaseModel):
    """Standard format for Standards RAG responses."""
    answer: str
    citations: List[CitationModel] = []
    confidence: float
    sources_used: List[str] = []

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        """Ensure confidence is between 0.0 and 1.0."""
        if not isinstance(v, (int, float)):
            raise ValueError('confidence must be numeric')
        if not (0.0 <= v <= 1.0):
            raise ValueError(f'confidence must be between 0.0 and 1.0, got {v}')
        return float(v)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
# Note: extract_json_from_response and sanitize_json_string are now imported
# from utils.json_utils module for consistency across all RAG agents.

# Local helper kept for backward compatibility (uses imported function internally)
def _sanitize_json_string(text: str) -> str:
    """Wrapper for backward compatibility - delegates to utils.json_utils."""
    return sanitize_json_string(text)


def normalize_response_dict(response_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIX #2: Normalize LLM response dictionary to match StandardsRAGResponse schema.

    Handles:
    - answer field being a dict instead of string → converts to string
    - Missing required fields → adds defaults
    - Invalid data types → coerces to correct types
    - Malformed citations → reconstructs valid list

    Args:
        response_dict: Raw parsed response from LLM

    Returns:
        Normalized dictionary matching StandardsRAGResponse schema
    """
    logger.info("[FIX2] Normalizing response dictionary for schema compliance")

    # Use consolidated utilities for type coercion
    answer = ensure_string(
        response_dict.get('answer', ''),
        default="Unable to generate answer."
    )
    if not answer:
        answer = "Unable to generate answer."

    # Ensure confidence is a float between 0-1
    confidence = ensure_float(
        response_dict.get('confidence', 0.5),
        default=0.5,
        min_val=0.0,
        max_val=1.0
    )

    # Handle citations
    citations = response_dict.get('citations', [])
    if not isinstance(citations, list):
        citations = []

    # Validate and reconstruct citations
    valid_citations = []
    for cite in citations:
        if not isinstance(cite, dict):
            continue
        try:
            valid_cite = {
                'source': ensure_string(cite.get('source', 'unknown'), default='unknown'),
                'content': ensure_string(cite.get('content', ''), default=''),
                'relevance': ensure_float(cite.get('relevance', 0.5), default=0.5, min_val=0.0, max_val=1.0)
            }
            valid_citations.append(valid_cite)
        except (ValueError, TypeError):
            continue

    # Handle sources_used
    sources_used = response_dict.get('sources_used', [])
    if not isinstance(sources_used, list):
        sources_used = []
    sources_used = [str(s) for s in sources_used if s]

    normalized = {
        'answer': answer,
        'citations': valid_citations,
        'confidence': confidence,
        'sources_used': sources_used
    }

    logger.info("[FIX2] Response normalized successfully")
    return normalized


# ============================================================================
# PROMPT TEMPLATE - Loaded from consolidated prompts_library file
# ============================================================================

# default_section captures content before first section marker as "RAG_INTRO"
# Using consolidated prompts directly
STANDARDS_CHAT_PROMPT = RAG_PROMPTS["STANDARDS_CHAT"]
STANDARDS_SCHEMA_ENRICHMENT_PROMPT = RAG_PROMPTS.get("STANDARDS_SCHEMA_ENRICHMENT", STANDARDS_CHAT_PROMPT)


# ============================================================================
# STANDARDS CHAT AGENT
# ============================================================================

class StandardsChatAgent:
    """
    Specialized agent for answering questions using standards documents from Pinecone.

    This agent retrieves relevant standards documents from the vector store,
    constructs context, and generates grounded answers using Google Generative AI.
    Includes production-ready retry logic and error handling.
    """

    def __init__(self, llm=None, temperature: float = 0.1):
        """
        Initialize the Standards Chat Agent.

        Args:
            llm: Language model instance (if None, creates default Gemini)
            temperature: Temperature for generation (default 0.1 for factual responses)
        """
        # Initialize LLM
        if llm is None:
            self.llm = create_llm_with_fallback(
                model="gemini-2.5-flash",
                temperature=temperature,
                max_tokens=2000,
                timeout=120  # Add timeout protection to prevent hanging requests
            )
        else:
            self.llm = llm

        # Use Azure Blob retriever if available, otherwise fall back to Pinecone
        if AZURE_BLOB_AVAILABLE:
            self.retriever = get_standards_blob_retriever()
            self.retrieval_mode = "azure_blob"
            self.vector_store = None
            logger.info("StandardsChatAgent initialized with Azure Blob retriever")
        else:
            # Get vector store
            self.vector_store = get_vector_store()
            self.retrieval_mode = "pinecone"
            self.retriever = None
            logger.info("StandardsChatAgent initialized with Pinecone vector store")

        # Initialize Deep Agent loader for parallel processing (lazy)
        self._deep_agent_loader = None  # Will be initialized on first use

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template(STANDARDS_CHAT_PROMPT)

        # Create output parser
        self.parser = JsonOutputParser()

        # Create chain
        self.chain = self.prompt | self.llm | self.parser

    def _get_deep_agent_loader(self):
        """
        Lazy load Deep Agent document loader for parallel processing.

        Returns:
            Function to load standards documents by type, or None if unavailable
        """
        if self._deep_agent_loader is None and DEEP_AGENT_AVAILABLE:
            self._deep_agent_loader = load_standard_text
            logger.info("[StandardsChatAgent] Deep Agent loader initialized for parallel processing")
        return self._deep_agent_loader

    def retrieve_documents(
        self,
        question: str,
        top_k: int = 5,
        source_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant standards documents from Azure Blob or Pinecone.

        OPTIMIZATION: When source_filter is provided, queries only the specified
        documents instead of all 12+ standards documents (90% reduction in scope).

        Args:
            question: User's question
            top_k: Number of top documents to retrieve
            source_filter: Optional list of document types to filter to
                          (e.g., ["safety", "pressure", "flow"])

        Returns:
            Dictionary with retrieval results
        """
        try:
            # Use Azure Blob retriever if available
            if self.retrieval_mode == "azure_blob" and self.retriever:
                logger.info(f"[StandardsChatAgent] Using Azure Blob retriever")
                search_results = self.retriever.retrieve_documents(
                    question=question,
                    top_k=top_k,
                    source_filter=source_filter
                )
            else:
                # Fall back to Pinecone
                logger.info(f"[StandardsChatAgent] Using Pinecone vector store")
                # Build filter metadata for Pinecone
                filter_metadata = None
                if source_filter:
                    # Pinecone filter syntax: {"field": {"$in": [values]}}
                    filter_metadata = {"filename": {"$in": source_filter}}
                    logger.info(
                        f"[StandardsChatAgent] Filtering to {len(source_filter)} docs: {source_filter}"
                    )

                # Search vector store with optional filter
                search_results = self.vector_store.search(
                    collection_type="standards",
                    query=question,
                    top_k=top_k,
                    filter_metadata=filter_metadata
                )

            logger.info(
                f"Retrieved {search_results.get('result_count', 0)} documents for question"
                f"{' (filtered)' if source_filter else ''}"
            )

            return search_results

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return {
                "results": [],
                "result_count": 0,
                "error": str(e)
            }

    def retrieve_documents_parallel(
        self,
        question: str,
        product_type: Optional[str] = None,
        standards_mentioned: Optional[List[str]] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve standards documents using Deep Agent parallel processing.

        This method uses a 3-worker parallel architecture to extract specifications
        from multiple standards domains simultaneously, then synthesizes the results.

        Architecture:
        - Worker 1: Extract safety/certification requirements
        - Worker 2: Extract product-specific standards
        - Worker 3: Extract domain-specific requirements

        Args:
            question: User's question
            product_type: Product type for domain detection (e.g., "pressure_transmitter")
            standards_mentioned: List of standards mentioned in question (e.g., ["SIL-2", "IEC 61508"])
            top_k: Maximum results to return (not used in parallel mode, kept for API compatibility)

        Returns:
            Dictionary with retrieval results formatted for compatibility with existing workflow
        """
        logger.info(f"[StandardsChatAgent] Using PARALLEL processing for: {question[:100]}...")

        try:
            loader = self._get_deep_agent_loader()
            if not loader:
                logger.warning("[StandardsChatAgent] Deep Agent not available, falling back to sequential")
                return self.retrieve_documents(question, top_k)

            # Detect relevant standards domains
            domains_to_search = self._detect_standards_domains(
                question, product_type, standards_mentioned
            )

            if not domains_to_search:
                logger.warning("[StandardsChatAgent] No relevant domains detected")
                return {
                    "success": False,
                    "results": [],
                    "result_count": 0,
                    "error": "No relevant standards domains detected"
                }

            logger.info(f"[StandardsChatAgent] Detected {len(domains_to_search)} domains: {domains_to_search}")

            # Load documents for detected domains (parallel I/O)
            from concurrent.futures import ThreadPoolExecutor, as_completed

            all_chunks = []
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_domain = {
                    executor.submit(self._get_domain_content, domain): domain
                    for domain in domains_to_search
                }

                for future in as_completed(future_to_domain):
                    domain = future_to_domain[future]
                    try:
                        content = future.result()
                        if content:
                            # Split content into searchable chunks
                            chunks = self._extract_relevant_chunks_from_content(
                                content, question, domain
                            )
                            all_chunks.extend(chunks)
                            logger.debug(f"[StandardsChatAgent] Domain '{domain}': {len(chunks)} chunks")
                        else:
                            logger.warning(f"[StandardsChatAgent] Domain '{domain}': No content")
                    except Exception as e:
                        logger.error(f"[StandardsChatAgent] Error loading domain '{domain}': {e}")

            # Sort by relevance and limit
            all_chunks.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            top_chunks = all_chunks[:top_k * 2]  # Get more chunks since we're doing parallel

            logger.info(
                f"[StandardsChatAgent] PARALLEL: Retrieved {len(top_chunks)} chunks "
                f"from {len(domains_to_search)} domains"
            )

            return {
                "success": True,
                "results": top_chunks,
                "result_count": len(top_chunks),
                "source": "deep_agent_parallel",
                "domains_searched": domains_to_search,
                "processing_mode": "parallel"
            }

        except Exception as e:
            logger.error(f"[StandardsChatAgent] Parallel processing failed: {e}")
            # Fallback to sequential
            logger.info("[StandardsChatAgent] Falling back to sequential retrieval")
            return self.retrieve_documents(question, top_k)

    def _detect_standards_domains(
        self,
        question: str,
        product_type: Optional[str],
        standards_mentioned: Optional[List[str]]
    ) -> List[str]:
        """
        Detect which standards domains to search based on question context.

        Args:
            question: User's question
            product_type: Product type if known
            standards_mentioned: List of standards if mentioned

        Returns:
            List of domain names (e.g., ["safety", "pressure", "calibration"])
        """
        domains = set()
        question_lower = question.lower()

        # Always check safety for certification questions
        safety_keywords = ["sil", "certification", "safety", "atex", "iecex", "functional safety", "sis"]
        if any(kw in question_lower for kw in safety_keywords):
            domains.add("safety")

        # Add product-specific domains based on product_type
        if product_type:
            pt_lower = product_type.lower()
            if "pressure" in pt_lower:
                domains.add("pressure")
            if "temperature" in pt_lower:
                domains.add("temperature")
            if "flow" in pt_lower:
                domains.add("flow")
            if "level" in pt_lower:
                domains.add("level")
            if "valve" in pt_lower or "control" in pt_lower:
                domains.add("valves")
            if "analytical" in pt_lower or "analyzer" in pt_lower:
                domains.add("analytical")

        # Detect domains from question text
        domain_keywords = {
            "pressure": ["pressure", "transmitter", "gauge"],
            "temperature": ["temperature", "thermocouple", "rtd"],
            "flow": ["flow", "meter", "flowmeter", "coriolis"],
            "level": ["level", "radar", "ultrasonic"],
            "valves": ["valve", "actuator", "control valve"],
            "analytical": ["analyzer", "ph", "conductivity", "chromatograph"],
            "calibration": ["calibration", "maintenance", "testing", "verification"],
            "communication": ["hart", "modbus", "profibus", "foundation fieldbus", "protocol"],
            "condition_monitoring": ["vibration", "condition monitoring", "asset health"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in question_lower for kw in keywords):
                domains.add(domain)

        # Default domains if nothing detected
        if not domains:
            logger.info("[StandardsChatAgent] No specific domains detected, using defaults")
            domains = {"safety", "pressure", "temperature"}

        return list(domains)

    def _get_domain_content(self, domain: str) -> Optional[str]:
        """
        Get content for a specific standards domain using Deep Agent loader.

        Args:
            domain: Domain name (e.g., "safety", "pressure")

        Returns:
            Full text content of the domain's standards document, or None if unavailable
        """
        loader = self._get_deep_agent_loader()
        if not loader:
            return None

        try:
            content = loader(domain)
            if content:
                logger.debug(f"[StandardsChatAgent] Loaded {len(content)} chars from domain '{domain}'")
            return content
        except Exception as e:
            logger.error(f"[StandardsChatAgent] Error loading domain '{domain}': {e}")
            return None

    def _extract_relevant_chunks_from_content(
        self,
        content: str,
        question: str,
        domain: str
    ) -> List[Dict[str, Any]]:
        """
        Extract relevant chunks from standards document content.

        Args:
            content: Full document text
            question: User's question
            domain: Domain name

        Returns:
            List of chunk dictionaries with content and metadata
        """
        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        # Extract question keywords for relevance scoring
        question_lower = question.lower()
        question_words = set(re.findall(r'\b\w+\b', question_lower))

        chunks = []
        for i, para in enumerate(paragraphs):
            if len(para) < 50:  # Skip very short paragraphs
                continue

            para_lower = para.lower()
            para_words = set(re.findall(r'\b\w+\b', para_lower))

            # Calculate relevance score
            score = 0.0

            # Keyword overlap
            overlap = len(question_words & para_words)
            score += overlap * 0.1

            # Boost for standards keywords
            standards_keywords = ["sil", "iec", "iso", "api", "atex", "iecex", "certification", "requirement"]
            if any(kw in para_lower for kw in standards_keywords):
                score += 0.5

            # Boost for long paragraphs (more content)
            if len(para) > 200:
                score += 0.2

            # Only include chunks with some relevance
            if score > 0.3:
                chunks.append({
                    "id": f"{domain}_chunk_{i}",
                    "content": para,
                    "metadata": {
                        "domain": domain,
                        "chunk_index": i,
                        "source": "deep_agent_parallel"
                    },
                    "relevance_score": score
                })

        return chunks

    def build_context(self, search_results: Dict[str, Any]) -> str:
        """
        Build formatted context from search results.

        Args:
            search_results: Results from vector store search

        Returns:
            Formatted context string
        """
        results = search_results.get('results', [])

        if not results:
            return "No relevant standards documents found."

        context_parts = []

        for i, result in enumerate(results, 1):
            # Extract metadata
            metadata = result.get('metadata', {})
            source = metadata.get('filename', 'unknown')
            standard_type = metadata.get('standard_type', 'general')
            standards_refs = metadata.get('standards_references', [])

            # Get content and relevance
            content = result.get('content', '')
            relevance = result.get('relevance_score', 0.0)

            # Build document section
            doc_section = f"[Document {i}: {source}]\n"
            doc_section += f"Type: {standard_type}\n"
            if standards_refs:
                doc_section += f"Referenced Standards: {', '.join(standards_refs[:5])}\n"
            doc_section += f"Relevance: {relevance:.3f}\n"
            doc_section += f"\nContent:\n{content}"

            context_parts.append(doc_section)

        return "\n\n" + "="*80 + "\n\n".join(context_parts)

    def extract_citations(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract citation information from search results.

        Args:
            search_results: Results from vector store search

        Returns:
            List of citations
        """
        citations = []

        for result in search_results.get('results', []):
            metadata = result.get('metadata', {})
            # Normalize relevance score to 0.0-1.0 range (some retrievers return >1.0)
            raw_relevance = result.get('relevance_score', 0.0)
            normalized_relevance = min(1.0, max(0.0, float(raw_relevance)))
            citation = {
                'source': metadata.get('filename', 'unknown'),
                'content': result.get('content', '')[:200] + "...",  # First 200 chars
                'relevance': normalized_relevance,
                'standard_type': metadata.get('standard_type', 'general'),
                'standards_references': metadata.get('standards_references', [])
            }
            citations.append(citation)

        return citations

    def run(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer question using standards documents.

        Args:
            question: User's question
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with answer, citations, confidence, and sources
        """
        search_results = None
        context = ""
        
        try:
            logger.info(f"Processing question: {question[:100]}...")

            # Step 1: Retrieve documents
            search_results = self.retrieve_documents(question, top_k)

            if search_results.get('result_count', 0) == 0:
                # ROOT CAUSE FIX: Provide actionable error message based on mock reason
                mock_reason = search_results.get('mock_reason', None)

                if mock_reason == 'MISSING_API_KEY':
                    error_msg = (
                        "Vector store is not properly configured: PINECONE_API_KEY environment variable is missing. "
                        "Please set the PINECONE_API_KEY environment variable to enable document retrieval."
                    )
                    logger.error(f"[StandardsChatAgent] {error_msg}")
                elif mock_reason and 'INIT_ERROR' in mock_reason:
                    error_msg = (
                        f"Vector store initialization failed ({mock_reason}). "
                        "Please verify PINECONE_API_KEY is valid and the Pinecone index exists."
                    )
                    logger.error(f"[StandardsChatAgent] {error_msg}")
                else:
                    error_msg = "I don't have any relevant standards documents to answer this question."

                return {
                    'answer': error_msg,
                    'citations': [],
                    'confidence': 0.0,
                    'sources_used': [],
                    'error': 'No documents found' if not mock_reason else f'Vector store unavailable: {mock_reason}'
                }

            # Step 2: Build context
            context = self.build_context(search_results)

            # Step 3: Generate answer with strict JSON parsing
            logger.info("Generating answer with LLM...")

            response = None
            raw_text = None

            try:
                # Invoke LLM directly (without the parser in the chain)
                raw_chain = self.prompt | self.llm
                raw_response = raw_chain.invoke({
                    'user_input': question,
                    'history': '',  # Empty for single-turn queries
                    'context': context
                })

                # Extract text content from response
                if hasattr(raw_response, 'content'):
                    raw_text = raw_response.content
                elif hasattr(raw_response, 'text'):
                    raw_text = raw_response.text
                else:
                    raw_text = str(raw_response)

                logger.debug(f"Raw LLM response: {raw_text[:200]}...")

                # Parse JSON with strict validation
                try:
                    # First, try direct JSON parsing
                    response_dict = json.loads(raw_text)
                    logger.debug("JSON parsed successfully (direct)")
                except json.JSONDecodeError as json_error:
                    logger.warning(
                        f"Direct JSON parse failed: {json_error}. "
                        f"Attempting extraction from response: {raw_text[:100]}..."
                    )
                    # Try extracting JSON from markdown or other formats
                    response_dict = extract_json_from_response(raw_text)

                    if response_dict is None:
                        logger.error(
                            f"Failed to extract JSON from response. Raw: {raw_text[:300]}"
                        )
                        raise ValueError(
                            "LLM returned invalid JSON format. "
                            "Response must be valid JSON only."
                        )
                    logger.info("JSON extracted successfully (from response)")

                # FIX #2: Normalize response dictionary for schema compliance
                logger.debug("Normalizing response for schema compliance...")
                response_dict = normalize_response_dict(response_dict)

                # Validate against StandardsRAGResponse schema
                try:
                    response = StandardsRAGResponse(**response_dict)
                    logger.debug(f"Response validated: confidence={response.confidence}")
                except Exception as validation_error:
                    logger.error(f"Response validation failed: {validation_error}")
                    logger.error(f"Response dict: {response_dict}")
                    raise ValueError(
                        f"LLM response structure invalid: {validation_error}"
                    )

            except Exception as e:
                logger.error(f"Error generating answer: {e}", exc_info=True)
                # Create fallback response with retrieved sources
                sources_used = list(set(
                    r.get('metadata', {}).get('filename', 'unknown')
                    for r in search_results.get('results', [])
                ))
                # ROOT CAUSE FIX: Mark this as an error response, not a valid answer
                # This prevents error messages from being stored as schema field values
                response = StandardsRAGResponse(
                    answer="",  # Empty answer instead of error message
                    citations=self.extract_citations(search_results),
                    confidence=0.0,  # Zero confidence for failed responses
                    sources_used=sources_used
                )
                # We'll add error info in the result dict, not in the answer field

            # Convert Pydantic model to dict for consistency with rest of code
            if isinstance(response, StandardsRAGResponse):
                response = response.model_dump()
            elif response is None:
                logger.error("Response is None after all attempts")
                response = {}

            # Step 4: Add metadata
            result = {
                'answer': response.get('answer', ''),
                'citations': response.get('citations', []),
                'confidence': response.get('confidence', 0.0),
                'sources_used': response.get('sources_used', []),
                # ROOT CAUSE FIX: Track if this was a fallback/error response
                'is_error_response': response.get('confidence', 0.0) == 0.0 and not response.get('answer', '')
            }

            # Calculate average retrieval relevance as fallback confidence
            if search_results and 'results' in search_results and search_results['results']:
                avg_relevance = sum(r.get('relevance_score', 0) for r in search_results['results']) / len(search_results['results'])
                # Use the higher of LLM confidence or retrieval relevance
                result['confidence'] = max(result['confidence'], avg_relevance)
                
                # If sources_used is empty but we have results, populate from search results
                if not result['sources_used']:
                    result['sources_used'] = list(set(
                        r.get('metadata', {}).get('filename', 'unknown')
                        for r in search_results['results']
                    ))

            logger.info(f"Answer generated with confidence: {result['confidence']:.2f}")
            logger.info(f"Sources used: {result['sources_used']}")

            return result

        except Exception as e:
            logger.error(f"Error in StandardsChatAgent.run: {e}", exc_info=True)
            
            # Even on error, try to return useful info if we have search results
            sources_used = []
            if search_results and 'results' in search_results:
                sources_used = list(set(
                    r.get('metadata', {}).get('filename', 'unknown')
                    for r in search_results['results']
                ))
            
            return {
                'answer': f"An error occurred while processing your question: {str(e)}",
                'citations': [],
                'confidence': 0.0,
                'sources_used': sources_used,
                'error': str(e)
            }

    def run_schema_enrichment(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Extract schema field values using JSON-only output (for schema population).
        
        This method is optimized for schema enrichment and enforces strict JSON output
        with concise values (2-7 words max), preventing verbose prose responses.
        
        Args:
            question: Question requesting field values (e.g., "Provide specs for pressure transmitter")
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with extracted field values in JSON format, or empty dict on failure
        """
        search_results = None
        context = ""
        
        try:
            logger.info(f"[SCHEMA_ENRICHMENT] Processing question: {question[:100]}...")
            
            # Step 1: Retrieve documents
            search_results = self.retrieve_documents(question, top_k)
            
            if search_results.get('result_count', 0) == 0:
                logger.warning("[SCHEMA_ENRICHMENT] No documents found")
                return {}
            
            # Step 2: Build context
            context = self.build_context(search_results)
            
            # Step 3: Generate answer with strict JSON-only prompt
            logger.info("[SCHEMA_ENRICHMENT] Generating field values with JSON-only prompt...")
            
            # Create JSON-only prompt template
            json_prompt = ChatPromptTemplate.from_template(STANDARDS_SCHEMA_ENRICHMENT_PROMPT)
            raw_chain = json_prompt | self.llm
            
            raw_response = raw_chain.invoke({
                'user_input': question,
                'context': context
            })
            
            # Extract text content from response
            if hasattr(raw_response, 'content'):
                raw_text = raw_response.content
            elif hasattr(raw_response, 'text'):
                raw_text = raw_response.text
            else:
                raw_text = str(raw_response)
            
            logger.debug(f"[SCHEMA_ENRICHMENT] Raw LLM response: {raw_text[:200]}...")
            
            # Parse JSON with strict validation
            try:
                # First, try direct JSON parsing
                response_dict = json.loads(raw_text)
                logger.debug("[SCHEMA_ENRICHMENT] JSON parsed successfully")
            except json.JSONDecodeError as json_error:
                logger.warning(
                    f"[SCHEMA_ENRICHMENT] Direct JSON parse failed: {json_error}. "
                    f"Attempting extraction from response: {raw_text[:100]}..."
                )
                # Try extracting JSON from markdown or other formats
                response_dict = extract_json_from_response(raw_text)
                
                if response_dict is None or not isinstance(response_dict, dict):
                    logger.error(
                        f"[SCHEMA_ENRICHMENT] Failed to extract JSON from response. "
                        f"Raw: {raw_text[:300]}"
                    )
                    # Return empty dict instead of continuing with malformed data
                    return {}
                logger.info("[SCHEMA_ENRICHMENT] JSON extracted successfully")
            
            # ROOT CAUSE FIX: Validate and filter values before returning
            # - Filter out error messages
            # - Filter out empty/null values
            # - Trim verbose values
            validated_dict = {}
            error_patterns = [
                "i found relevant", "temporarily unavailable", "api quota",
                "please try again", "service is", "error:", "failed to",
                ".docx)", ".pdf)", "standards documents"
            ]

            for key, value in response_dict.items():
                if not value:
                    continue

                if isinstance(value, str):
                    val_lower = value.lower()

                    # Skip error messages
                    if any(pattern in val_lower for pattern in error_patterns):
                        logger.debug(f"[SCHEMA_ENRICHMENT] Filtered error value for '{key}': {value[:50]}...")
                        continue

                    # Skip overly long values (likely descriptions, not spec values)
                    if len(value) > 150:
                        logger.debug(f"[SCHEMA_ENRICHMENT] Filtered long value for '{key}': {len(value)} chars")
                        continue

                    # Skip null-like values
                    if val_lower in ['null', 'none', 'n/a', 'not specified', 'unknown', '']:
                        continue

                    # Trim to max 7 words
                    words = value.split()
                    if len(words) > 7:
                        trimmed_value = ' '.join(words[:7])
                        logger.debug(f"[SCHEMA_ENRICHMENT] Trimmed '{key}': '{value}' -> '{trimmed_value}'")
                        validated_dict[key] = trimmed_value
                    else:
                        validated_dict[key] = value
                else:
                    validated_dict[key] = value

            logger.info(f"[SCHEMA_ENRICHMENT] Extracted {len(validated_dict)} valid field values (filtered {len(response_dict) - len(validated_dict)})")
            return validated_dict
            
        except Exception as e:
            logger.error(f"[SCHEMA_ENRICHMENT] Error: {e}", exc_info=True)
            return {}




# ============================================================================
# SINGLETON INSTANCE (Performance Optimization)
# ============================================================================

_standards_chat_agent_instance = None
_standards_chat_agent_lock = None

def _get_agent_lock():
    """Get thread lock for singleton (lazy initialization)."""
    global _standards_chat_agent_lock
    if _standards_chat_agent_lock is None:
        import threading
        _standards_chat_agent_lock = threading.Lock()
    return _standards_chat_agent_lock


def get_standards_chat_agent(temperature: float = 0.1) -> StandardsChatAgent:
    """
    Get or create the singleton StandardsChatAgent instance.
    
    This avoids expensive re-initialization of LLM and vector store connections
    on every call, significantly improving performance for batch operations.
    
    Args:
        temperature: Temperature for generation (only used on first call)
        
    Returns:
        Cached StandardsChatAgent instance
    """
    global _standards_chat_agent_instance
    
    if _standards_chat_agent_instance is None:
        with _get_agent_lock():
            # Double-check after acquiring lock
            if _standards_chat_agent_instance is None:
                logger.info("[StandardsChatAgent] Creating singleton instance (first call)")
                _standards_chat_agent_instance = StandardsChatAgent(temperature=temperature)
    
    return _standards_chat_agent_instance


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_standards_chat_agent(temperature: float = 0.1) -> StandardsChatAgent:
    """
    Create a StandardsChatAgent instance with default settings.
    
    NOTE: For better performance, use get_standards_chat_agent() instead,
    which returns a cached singleton instance.

    Args:
        temperature: Temperature for generation

    Returns:
        StandardsChatAgent instance (uses singleton for efficiency)
    """
    # Use singleton for better performance
    return get_standards_chat_agent(temperature=temperature)


def ask_standards_question(question: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Quick function to ask a question to the standards system.

    Args:
        question: Question to ask
        top_k: Number of documents to retrieve

    Returns:
        Answer dictionary
    """
    agent = create_standards_chat_agent()
    return agent.run(question, top_k)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Load environment variables when running as script
    from dotenv import load_dotenv
    import os
    
    # Find and load .env file from backend directory
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    env_path = os.path.join(backend_dir, '.env')
    
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logger.info(f"Loaded environment from: {env_path}")
    else:
        logger.warning(f".env file not found at: {env_path}")
    
    # Test the agent
    logger.info("="*80)
    logger.info("TESTING STANDARDS CHAT AGENT")
    logger.info("="*80)

    test_questions = [
        "What are the SIL2 requirements for pressure transmitters?",
        "What calibration standards apply to temperature sensors?",
        "What are the ATEX requirements for flow measurement devices?"
    ]

    agent = create_standards_chat_agent()

    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n[Test {i}] Question: {question}")
        logger.info("-"*80)

        result = agent.run(question, top_k=3)

        logger.info(f"\nAnswer: {result['answer'][:300]}...")
        logger.info(f"\nConfidence: {result['confidence']:.2f}")
        logger.info(f"Sources: {', '.join(result['sources_used'])}")

        if result.get('citations'):
            logger.info(f"\nCitations:")
            for j, cite in enumerate(result['citations'][:2], 1):
                logger.info(f"  [{j}] {cite['source']} (relevance: {cite.get('relevance', 0):.2f})")

        logger.info("="*80)
