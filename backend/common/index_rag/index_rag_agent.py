# agentic/index_rag_agent.py
# Index RAG Agent - LLM-Powered Product Indexing and Retrieval
# Uses Flash Lite for intent classification and output structuring

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
logger = logging.getLogger(__name__)

# Import prompt loader
from common.prompts import INDEX_RAG_PROMPTS

# Import debug utilities
try:
    from debug_flags import debug_log, timed_execution, is_debug_enabled
except ImportError:
    # Graceful fallback
    debug_log = lambda *a, **kw: lambda f: f
    timed_execution = lambda *a, **kw: lambda f: f
    is_debug_enabled = lambda m: False


# ============================================================================
# PRODUCT TYPE NORMALIZATION (2026-02-11)
# ============================================================================
# Maps LLM output variants to database format
# Problem: LLM returns "level sensor" but database expects "level_transmitter"
# Solution: Normalize before vendor search

PRODUCT_TYPE_NORMALIZATION = {
    # Level instruments
    "level sensor": "level_transmitter",
    "level sensors": "level_transmitter",
    "radar level sensor": "level_transmitter",
    "radar level sensors": "level_transmitter",
    "ultrasonic level sensor": "level_transmitter",
    "ultrasonic level sensors": "level_transmitter",
    "guided wave radar": "level_transmitter",
    "level instrument": "level_transmitter",
    "level instruments": "level_transmitter",

    # Pressure instruments
    "pressure sensor": "pressure_transmitter",
    "pressure sensors": "pressure_transmitter",
    "pressure gauge": "pressure_transmitter",
    "pressure gauges": "pressure_transmitter",
    "pressure instrument": "pressure_transmitter",
    "pressure instruments": "pressure_transmitter",

    # Temperature instruments
    "temperature sensor": "temperature_transmitter",
    "temperature sensors": "temperature_transmitter",
    "temperature instrument": "temperature_transmitter",
    "temperature instruments": "temperature_transmitter",
    "thermocouple": "temperature_transmitter",
    "thermocouples": "temperature_transmitter",
    "rtd": "temperature_transmitter",
    "rtds": "temperature_transmitter",
    "resistance temperature detector": "temperature_transmitter",

    # Flow instruments
    "flow sensor": "flow_meter",
    "flow sensors": "flow_meter",
    "flow meters": "flow_meter",
    "flowmeter": "flow_meter",
    "flowmeters": "flow_meter",
    "coriolis meter": "flow_meter",
    "coriolis meters": "flow_meter",
    "magnetic flowmeter": "flow_meter",
    "magnetic flowmeters": "flow_meter",
    "ultrasonic flowmeter": "flow_meter",
    "ultrasonic flowmeters": "flow_meter",
    "flow instrument": "flow_meter",
    "flow instruments": "flow_meter",

    # Control valves
    "control valve": "control_valves",
    "valve": "control_valves",
    "valves": "control_valves",
    "ball valve": "control_valves",
    "ball valves": "control_valves",
    "globe valve": "control_valves",
    "globe valves": "control_valves",
    "butterfly valve": "control_valves",
    "butterfly valves": "control_valves",
    "actuated valve": "control_valves",
    "actuated valves": "control_valves",
    "modulating valve": "control_valves",
    "modulating valves": "control_valves",

    # Actuators
    "actuator": "actuators",
    "valve actuator": "actuators",
    "valve actuators": "actuators",
    "pneumatic actuator": "actuators",
    "pneumatic actuators": "actuators",
    "electric actuator": "actuators",
    "electric actuators": "actuators",

    # Gas detectors
    "gas detector": "gas_detectors",
    "gas detectors": "gas_detectors",
    "gas sensor": "gas_detectors",
    "gas sensors": "gas_detectors",

    # Analytical instruments
    "analyzer": "analytical_instruments",
    "analyzers": "analytical_instruments",
    "analytical instrument": "analytical_instruments",
    "gas chromatograph": "analytical_instruments",
    "gas chromatographs": "analytical_instruments",
    "ph meter": "analytical_instruments",
    "ph meters": "analytical_instruments",
    "conductivity meter": "analytical_instruments",
    "conductivity meters": "analytical_instruments",
}


def normalize_product_type(product_type: Optional[str]) -> Optional[str]:
    """
    Normalize product type from LLM output to database format.

    Args:
        product_type: Product type string from LLM (e.g., "level sensor")

    Returns:
        Normalized product type (e.g., "level_transmitter") or original if no mapping

    Examples:
        >>> normalize_product_type("level sensor")
        "level_transmitter"
        >>> normalize_product_type("radar level sensors")
        "level_transmitter"
        >>> normalize_product_type("unknown_product")
        "unknown_product"
    """
    if not product_type:
        return product_type

    # Try exact match first (case-insensitive)
    normalized = PRODUCT_TYPE_NORMALIZATION.get(product_type.lower())
    if normalized:
        logger.debug(f"[ProductTypeNorm] Exact match: '{product_type}' → '{normalized}'")
        return normalized

    # Try partial match (check if any key is substring of product_type)
    pt_lower = product_type.lower()
    for llm_variant, db_format in PRODUCT_TYPE_NORMALIZATION.items():
        if llm_variant in pt_lower:
            logger.debug(f"[ProductTypeNorm] Partial match: '{product_type}' → '{db_format}'")
            return db_format

    # No match - return as-is
    logger.debug(f"[ProductTypeNorm] No match for: '{product_type}' (keeping original)")
    return product_type


# ============================================================================
# PROMPTS - Loaded from consolidated prompts_library file
# ============================================================================

# default_section captures content before first section marker as "INTENT_CLASSIFICATION"
# Using consolidated prompts directly
INTENT_CLASSIFICATION_PROMPT = INDEX_RAG_PROMPTS["INTENT_CLASSIFICATION"]
OUTPUT_STRUCTURING_PROMPT = INDEX_RAG_PROMPTS["OUTPUT_STRUCTURING"]


# ============================================================================
# INDEX RAG AGENT CLASS
# ============================================================================

class IndexRAGAgent:
    """
    LLM-Powered Index RAG Agent for product search.
    
    Features:
    - Flash Lite intent classification
    - Hierarchical metadata filtering
    - Parallel database + web indexing
    - Standards + Strategy filtering
    - Structured output generation
    """
    
    def __init__(self, temperature: float = 0.1):
        """
        Initialize the Index RAG Agent.
        
        Args:
            temperature: LLM temperature for generation
        """
        from common.services.llm.fallback import create_llm_with_fallback
        
        # Use Flash for fast classification and structuring
        self.llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=temperature,
            max_tokens=2000
        )
        
        self.parser = JsonOutputParser()
        
        # Create chains
        self.intent_chain = (
            ChatPromptTemplate.from_template(INTENT_CLASSIFICATION_PROMPT) 
            | self.llm 
            | self.parser
        )
        
        self.output_chain = (
            ChatPromptTemplate.from_template(OUTPUT_STRUCTURING_PROMPT)
            | self.llm
            | self.parser
        )
        
        logger.info("[IndexRAGAgent] Initialized with Gemini Flash")
    
    @debug_log("RAG", log_args=False)
    @timed_execution("RAG", threshold_ms=3000)
    def classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify user intent using LLM with product type normalization.

        Args:
            query: User's search query

        Returns:
            Classified intent with extracted metadata (product_type normalized)
        """
        logger.info(f"[IndexRAGAgent] Classifying intent for: {query[:100]}...")

        try:
            result = self.intent_chain.invoke({"query": query})

            # NORMALIZE product type before returning (Fix #2)
            original_product_type = result.get('product_type')
            if original_product_type:
                normalized_product_type = normalize_product_type(original_product_type)
                if normalized_product_type != original_product_type:
                    logger.info(f"[IndexRAGAgent] Normalized product type: '{original_product_type}' → '{normalized_product_type}'")
                    result['product_type'] = normalized_product_type
                    result['original_product_type'] = original_product_type  # Keep original for reference

            logger.info(f"[IndexRAGAgent] Intent: {result.get('intent')}, "
                       f"Product: {result.get('product_type')}, "
                       f"Confidence: {result.get('confidence', 0):.2f}")

            return {
                "success": True,
                **result
            }

        except Exception as e:
            logger.error(f"[IndexRAGAgent] Intent classification failed: {e}")

            # Fallback to regex extraction
            from common.services.extraction.metadata_filter import (
                extract_product_types,
                extract_vendors,
                extract_models
            )

            extracted_product_type = extract_product_types(query)[0] if extract_product_types(query) else None
            # Also normalize in fallback case
            if extracted_product_type:
                extracted_product_type = normalize_product_type(extracted_product_type)

            return {
                "success": False,
                "intent": "search",
                "product_type": extracted_product_type,
                "vendors": extract_vendors(query),
                "models": extract_models(query),
                "specifications": {},
                "filters": {},
                "confidence": 0.5,
                "error": str(e)
            }
    
    def structure_output(
        self,
        query: str,
        search_results: List[Dict],
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Structure search results using LLM.
        
        Args:
            query: Original user query
            search_results: List of filtered product results
            metadata: Additional context (filters applied, etc.)
            
        Returns:
            Structured output for frontend
        """
        logger.info(f"[IndexRAGAgent] Structuring {len(search_results)} results...")
        
        try:
            # Format results for LLM
            results_text = json.dumps(search_results[:10], indent=2, default=str)  # Limit for context
            
            result = self.output_chain.invoke({
                "query": query,
                "results": results_text  # Fixed: Changed from 'search_results' to match prompt template
            })
            
            logger.info(f"[IndexRAGAgent] Structured output: {len(result.get('recommended_products', []))} products")
            
            return {
                "success": True,
                **result
            }
            
        except Exception as e:
            logger.error(f"[IndexRAGAgent] Output structuring failed: {e}")
            
            # Fallback to simple formatting
            return {
                "success": False,
                "summary": f"Found {len(search_results)} products",
                "recommended_products": search_results[:10],
                "total_found": len(search_results),
                "error": str(e)
            }
    
    def run(
        self,
        query: str,
        requirements: Dict[str, Any] = None,
        top_k: int = 7,
        enable_web_search: bool = True
    ) -> Dict[str, Any]:
        """
        Full Index RAG pipeline.
        
        Steps:
        1. Classify intent with LLM
        2. Apply hierarchical metadata filter
        3. Run parallel indexing (database + web)
        4. Apply specs filter (standards + strategy)
        5. Structure output with LLM
        
        Args:
            query: User's search query
            requirements: Optional structured requirements
            top_k: Max results to return
            enable_web_search: Enable web search thread
            
        Returns:
            Structured search results
        """
        logger.info(f"[IndexRAGAgent] Starting Index RAG for: {query[:100]}...")
        start_time = datetime.utcnow()
        
        result = {
            "success": False,
            "query": query,
            "stages": {}
        }
        
        try:
            # Stage 1: Intent Classification
            logger.info("[IndexRAGAgent] Stage 1: Intent Classification")
            intent_result = self.classify_intent(query)
            result["stages"]["intent"] = intent_result
            
            product_type = intent_result.get("product_type")
            explicit_vendors = intent_result.get("vendors", [])
            
            # Stage 2: Hierarchical Metadata Filter
            logger.info("[IndexRAGAgent] Stage 2: Metadata Filter")
            from common.services.extraction.metadata_filter import filter_by_hierarchy
            
            metadata_result = filter_by_hierarchy(
                query=query,
                top_k=top_k,
                explicit_product_type=product_type,
                explicit_vendors=explicit_vendors if explicit_vendors else None
            )
            result["stages"]["metadata_filter"] = {
                "product_types": metadata_result.get("product_types"),
                "vendors_count": len(metadata_result.get("vendors", [])),
                "filter_applied": metadata_result.get("filter_applied")
            }
            
            # Stage 3: Parallel Indexing
            logger.info("[IndexRAGAgent] Stage 3: Parallel Indexing")
            from common.services.extraction.parallel_indexer import run_parallel_indexing
            
            # Extract product types and vendors from metadata filter result
            detected_product_types = metadata_result.get("product_types", [])
            filtered_vendors = metadata_result.get("vendors", [])
            
            index_result = run_parallel_indexing(
                query=query,
                product_type=detected_product_types[0] if detected_product_types else product_type,
                vendors=filtered_vendors if filtered_vendors else None,
                top_k=top_k,
                enable_web_search=enable_web_search
            )
            result["stages"]["parallel_indexing"] = {
                "database_count": len(index_result.get("database", {}).get("results", [])),
                "web_count": len(index_result.get("web_search", {}).get("results", [])) if enable_web_search else 0,
                "merged_count": index_result.get("total_count", 0)
            }
            
            merged_products = index_result.get("merged_results", [])
            
            # Stage 4: Specs + Strategy Filter
            logger.info("[IndexRAGAgent] Stage 4: Specs & Strategy Filter")
            from common.tools.specs_filter import apply_specs_and_strategy_filter
            
            filter_result = apply_specs_and_strategy_filter(
                products=merged_products,
                product_type=product_type,
                requirements=requirements,
                top_k=top_k
            )
            result["stages"]["specs_filter"] = {
                "standards_applied": filter_result.get("standards", {}).get("applied", False),
                "strategy_applied": filter_result.get("strategy", {}).get("applied", False),
                "excluded_count": filter_result.get("strategy", {}).get("excluded_count", 0),
                "final_count": filter_result.get("filter_count", 0)
            }
            
            filtered_products = filter_result.get("filtered_products", merged_products)
            
            # Stage 5: Structure Output
            logger.info("[IndexRAGAgent] Stage 5: Output Structuring")
            structured_result = self.structure_output(
                query=query,
                search_results=filtered_products,
                metadata={
                    "intent": intent_result,
                    "filters": filter_result
                }
            )
            result["stages"]["output"] = {
                "structured": structured_result.get("success", False),
                "product_count": len(structured_result.get("recommended_products", []))
            }
            
            # Build final result
            result["success"] = True
            result["intent"] = intent_result.get("intent")
            result["product_type"] = product_type
            result["output"] = structured_result
            result["raw_results"] = filtered_products
            result["processing_time_ms"] = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            logger.info(f"[IndexRAGAgent] Pipeline complete in {result['processing_time_ms']}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"[IndexRAGAgent] Pipeline failed: {e}", exc_info=True)
            result["success"] = False
            result["error"] = str(e)
            return result


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_index_rag_agent(temperature: float = 0.1) -> IndexRAGAgent:
    """Create an IndexRAGAgent instance."""
    return IndexRAGAgent(temperature=temperature)


def run_index_rag(
    query: str,
    requirements: Dict[str, Any] = None,
    top_k: int = 7,
    enable_web_search: bool = True
) -> Dict[str, Any]:
    """
    Run the Index RAG pipeline.
    
    Args:
        query: User's search query
        requirements: Optional requirements
        top_k: Max results
        enable_web_search: Enable web search
        
    Returns:
        Structured search results
    """
    agent = create_index_rag_agent()
    return agent.run(
        query=query,
        requirements=requirements,
        top_k=top_k,
        enable_web_search=enable_web_search
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'IndexRAGAgent',
    'create_index_rag_agent',
    'run_index_rag'
]


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("TESTING INDEX RAG AGENT")
    print("=" * 80)
    
    test_queries = [
        "I need a pressure transmitter from Yokogawa for hazardous area",
        "Compare Emerson and Rosemount flow meters",
        "EJA110E specifications"
    ]
    
    agent = create_index_rag_agent()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}] Query: {query}")
        print("-" * 80)
        
        result = agent.run(query=query, top_k=3, enable_web_search=False)
        
        if result["success"]:
            print(f"Intent: {result.get('intent')}")
            print(f"Product Type: {result.get('product_type')}")
            print(f"Processing Time: {result.get('processing_time_ms')}ms")
            
            output = result.get("output", {})
            print(f"Summary: {output.get('summary', 'N/A')}")
            print(f"Products Found: {output.get('total_found', 0)}")
        else:
            print(f"Error: {result.get('error')}")
        
        print("=" * 80)
