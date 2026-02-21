# search/__init__.py
"""
Product Search Workflow - Tool-Based Architecture
=================================================

This module provides a tool-based product search workflow with the following components:

Tools
-----
ValidationTool          -- Step 1: Product type detection, schema generation, validation
AdvancedSpecificationAgent  -- Step 2: Discover advanced parameters from vendors
VendorAnalysisDeepAgent -- Step 3: Analyze vendors and match products
RankingTool            -- Step 4: Rank matched products with detailed analysis
SalesAgentTool         -- Conversational agent for workflow guidance

Workflow Functions
------------------
run_product_search_workflow()     -- Full end-to-end product search
run_validation_only()             -- Run only validation step
run_advanced_params_only()        -- Run only parameter discovery
run_analysis_only()               -- Run vendor analysis + ranking
process_from_solution_workflow()  -- Batch processing for solution workflow

Utility Functions
-----------------
get_schema_only()                 -- Get schema without validation
validate_with_schema()            -- Validate input against schema
"""

# =============================================================================
# TOOL CLASSES
# =============================================================================

from .validation_tool import ValidationTool, clear_session_enrichment_cache
from .advanced_specification_agent import AdvancedSpecificationAgent
from .vendor_analysis_deep_agent import VendorAnalysisDeepAgent, VendorAnalysisTool
from .ranking_tool import RankingTool
from .sales_agent_tool import SalesAgentTool

# =============================================================================
# WORKFLOW ORCHESTRATION FUNCTIONS
# =============================================================================

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def run_product_search_workflow(
    user_input: str,
    session_id: str,
    expected_product_type: Optional[str] = None,
    user_provided_fields: Optional[Dict[str, Any]] = None,
    enable_ppi: bool = True,
    auto_mode: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Run complete product search workflow (tool-based).

    Args:
        user_input: User's search query
        session_id: Session identifier
        expected_product_type: Optional product type hint
        user_provided_fields: User-provided specification fields
        enable_ppi: Enable Potential Product Index workflow if no schema exists
        auto_mode: Run automatically without HITL pauses
        **kwargs: Additional parameters (main_thread_id, parent_workflow_id, etc.)

    Returns:
        {
            "success": bool,
            "response": str,
            "response_data": {
                "product_type": str,
                "schema": dict,
                "ranked_products": list,
                "vendor_matches": dict,
                "missing_fields": list,
                "awaiting_user_input": bool
            },
            "error": str (if failed)
        }
    """
    logger.info(f"[ProductSearch] Starting workflow for session: {session_id}")

    try:
        # =====================================================================
        # FIX: Short-circuit conversational inputs ("ok", etc.)
        # =====================================================================
        normalized_input = user_input.lower().strip()
        conversational_phrases = {'ok', 'okay', 'proceed', 'continue', 'sure', 'thanks', 'thank you', 'stop', 'cancel'}
        
        words = normalized_input.split()
        # Only bypass if it's very short and contains conversational words, OR it's an exact match
        if (len(words) < 4 and any(word in conversational_phrases for word in words)) or normalized_input in conversational_phrases or normalized_input == 'show me':
            logger.info(f"[ProductSearch] Detected conversational input '{user_input}', routing to SalesAgentTool")
            sales_agent = SalesAgentTool()
            response_message = sales_agent.process_step(
                step="conversational", 
                user_message=user_input, 
                data_context={}, 
                session_id=session_id
            ).get("content", "I understand. Please let me know if you need anything else.")
            
            return {
                "success": True,
                "response": response_message,
                "response_data": {
                    "product_type": expected_product_type or "unknown",
                    "schema": {},
                    "ranked_products": [],
                    "vendor_matches": {},
                    "missing_fields": [],
                    "awaiting_user_input": True,
                    "current_phase": "conversational"
                }
            }

        # Step 1: Validation
        validation_tool = ValidationTool()
        validation_result = validation_tool.validate(
            user_input=user_input,
            expected_product_type=expected_product_type,
            session_id=session_id,
            enable_standards_enrichment=enable_ppi
        )

        # ═══════════════════════════════════════════════════════════════════════════
        # HANDLE VALIDATION BYPASSED (User said "YES" to HITL prompt)
        # When validation is bypassed, AdvancedSpecificationAgent already ran inside
        # validation_tool. Return the advanced specs and proceed to vendor analysis.
        # ═══════════════════════════════════════════════════════════════════════════
        if validation_result.get("validation_bypassed"):
            logger.info("[ProductSearch] Validation was bypassed (user said YES)")
            product_type = validation_result.get("product_type", expected_product_type)
            advanced_specs_info = validation_result.get("advanced_specs_info", {})
            advanced_params = advanced_specs_info.get("unique_specifications", [])

            # If advanced specs discovery failed, return that error
            if not validation_result.get("success"):
                return {
                    "success": False,
                    "response": validation_result.get("error", "Advanced specification discovery failed"),
                    "response_data": {
                        "product_type": product_type,
                        "validation_bypassed": True,
                        "error": validation_result.get("error")
                    },
                    "error": validation_result.get("error")
                }

            # Return advanced specs to user - they can now provide additional specs
            # or proceed to vendor analysis
            num_specs = len(advanced_params)
            logger.info(f"[ProductSearch] ✓ Advanced specs already discovered: {num_specs} parameters")

            return {
                "success": True,
                "response": validation_result.get("message", f"Discovered {num_specs} advanced specifications for {product_type}. Please provide your requirements for vendor matching."),
                "response_data": {
                    "product_type": product_type,
                    "schema": validation_result.get("schema", {}),  # BUG FIX: preserve schema from Turn 1
                    "advanced_parameters": advanced_params,
                    "advanced_specs_info": advanced_specs_info,
                    "ranked_products": [],
                    "vendor_matches": {},
                    "missing_fields": [],
                    "awaiting_user_input": True,  # User should provide specs now
                    "current_phase": "advanced_specs_discovered",
                    "validation_bypassed": True
                }
            }

        if not validation_result.get("is_valid"):
            # Missing fields - return for HITL
            return {
                "success": True,
                "response": validation_result.get("hitl_message", "Please provide missing specifications"),
                "response_data": {
                    "product_type": validation_result.get("product_type"),
                    "schema": validation_result.get("schema"),
                    "missing_fields": validation_result.get("missing_fields", []),
                    "provided_requirements": validation_result.get("provided_requirements", {}),
                    "awaiting_user_input": True,
                    "current_phase": "validation",
                    "ranked_products": [],
                    "vendor_matches": {}
                }
            }

        product_type = validation_result["product_type"]
        schema = validation_result["schema"]
        provided_requirements = validation_result.get("provided_requirements", {})

        # Merge user-provided fields if any
        if user_provided_fields:
            provided_requirements.update(user_provided_fields)

        # Step 2: Advanced Parameters Discovery (optional)
        advanced_params = []
        try:
            params_tool = AdvancedSpecificationAgent()
            params_result = params_tool.discover(
                product_type=product_type,
                session_id=session_id
            )
            advanced_params = params_result.get("unique_specifications", [])
            logger.info(f"[ProductSearch] Discovered {len(advanced_params)} advanced parameters")
        except Exception as e:
            logger.warning(f"[ProductSearch] Advanced params discovery failed: {e}")

        # Step 3: Vendor Analysis
        vendor_tool = VendorAnalysisTool()
        vendor_result = vendor_tool.analyze(
            structured_requirements=provided_requirements,
            product_type=product_type,
            session_id=session_id,
            schema=schema
        )

        vendor_matches = vendor_result.get("vendor_matches", {})
        if not vendor_matches:
            return {
                "success": True,
                "response": "No matching products found for your requirements",
                "response_data": {
                    "product_type": product_type,
                    "schema": schema,
                    "ranked_products": [],
                    "vendor_matches": {},
                    "missing_fields": [],
                    "awaiting_user_input": False
                }
            }

        # Step 4: Ranking
        ranking_tool = RankingTool(use_llm_ranking=True)
        ranking_result = ranking_tool.rank(
            vendor_analysis={"vendor_matches": vendor_matches if isinstance(vendor_matches, list) else list(vendor_matches.values()) if isinstance(vendor_matches, dict) else []},
            session_id=session_id,
            structured_requirements=provided_requirements
        )

        ranked_products = ranking_result.get("overall_ranking", ranking_result.get("ranked_products", []))

        # Step 5: Format response
        sales_agent = SalesAgentTool()
        response_message = sales_agent.process_step(
            step="finalAnalysis",
            user_message=user_input,
            data_context={
                "productType": product_type,
                "rankedProducts": ranked_products
            },
            session_id=session_id
        ).get("content", "Search completed successfully")

        return {
            "success": True,
            "response": response_message,
            "response_data": {
                "product_type": product_type,
                "schema": schema,
                "ranked_products": ranked_products,
                "vendor_matches": vendor_matches,
                "advanced_parameters": advanced_params,
                "provided_requirements": provided_requirements,
                "missing_fields": [],
                "awaiting_user_input": False,
                "completed": True,
                "current_phase": "completed"
            }
        }

    except Exception as e:
        logger.exception(f"[ProductSearch] Workflow failed: {e}")
        return {
            "success": False,
            "response": f"Product search failed: {str(e)}",
            "response_data": {},
            "error": str(e)
        }


def run_validation_only(
    user_input: str,
    expected_product_type: Optional[str] = None,
    session_id: str = "default",
    enable_ppi: bool = True
) -> Dict[str, Any]:
    """
    Run only validation step (product type detection + schema generation).

    Args:
        user_input: User's search query
        expected_product_type: Optional product type hint
        session_id: Session identifier
        enable_ppi: Enable PPI workflow if no schema exists

    Returns:
        {
            "product_type": str,
            "schema": dict,
            "provided_requirements": dict,
            "missing_fields": list,
            "is_valid": bool,
            "ppi_workflow_used": bool
        }
    """
    logger.info(f"[ValidationOnly] Session: {session_id}")

    try:
        tool = ValidationTool()
        result = tool.validate(
            user_input=user_input,
            expected_product_type=expected_product_type,
            session_id=session_id,
            enable_standards_enrichment=enable_ppi
        )
        return result

    except Exception as e:
        logger.exception(f"[ValidationOnly] Failed: {e}")
        return {
            "product_type": "",
            "schema": {},
            "provided_requirements": {},
            "missing_fields": [],
            "is_valid": False,
            "error": str(e)
        }


def run_advanced_params_only(
    product_type: str,
    user_input: str,
    session_id: str = "default"
) -> Dict[str, Any]:
    """
    Run only advanced parameters discovery.

    Args:
        product_type: Product type
        user_input: User's search query
        session_id: Session identifier

    Returns:
        {
            "parameters": list,
            "count": int
        }
    """
    logger.info(f"[AdvancedParamsOnly] Product: {product_type}, Session: {session_id}")

    try:
        tool = AdvancedSpecificationAgent()
        result = tool.discover(
            product_type=product_type,
            user_input=user_input,
            session_id=session_id
        )
        return result

    except Exception as e:
        logger.exception(f"[AdvancedParamsOnly] Failed: {e}")
        return {
            "parameters": [],
            "count": 0,
            "error": str(e)
        }


def run_analysis_only(
    product_type: str,
    structured_requirements: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
    session_id: str = "default",
    user_input: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run vendor analysis + ranking (skip validation).

    Args:
        product_type: Product type
        structured_requirements: User-provided requirements (structured format)
        schema: Product schema
        session_id: Session identifier
        user_input: User's search query (optional)

    Returns:
        {
            "success": bool,
            "ranked_products": list,
            "overall_ranking": list,
            "vendor_matches": dict,
            "response": str
        }
    """
    logger.info(f"[AnalysisOnly] Product: {product_type}, Session: {session_id}")

    try:
        requirements = structured_requirements or {}

        # Vendor Analysis
        vendor_tool = VendorAnalysisTool()
        vendor_result = vendor_tool.analyze(
            structured_requirements=requirements,
            product_type=product_type,
            session_id=session_id,
            schema=schema
        )

        vendor_matches = vendor_result.get("vendor_matches", {})

        # Ranking
        ranking_tool = RankingTool(use_llm_ranking=True)
        # Normalise vendor_matches into a flat list for RankingTool
        if isinstance(vendor_matches, dict):
            vendor_matches_list = []
            for match_list in vendor_matches.values():
                if isinstance(match_list, list):
                    vendor_matches_list.extend(match_list)
                elif isinstance(match_list, dict):
                    vendor_matches_list.append(match_list)
        elif isinstance(vendor_matches, list):
            vendor_matches_list = vendor_matches
        else:
            vendor_matches_list = []

        ranking_result = ranking_tool.rank(
            vendor_analysis={"vendor_matches": vendor_matches_list},
            session_id=session_id,
            structured_requirements=requirements
        )

        ranked_products = ranking_result.get("overall_ranking", ranking_result.get("ranked_products", []))
        top_product = ranking_result.get("top_product", ranked_products[0] if ranked_products else None)

        return {
            "success": True,
            "ranked_products": ranked_products,
            "overall_ranking": ranked_products,  # Alias for compatibility
            "top_product": top_product,
            "vendor_matches": vendor_matches,
            "totalRanked": len(ranked_products),
            "exactMatchCount": sum(1 for p in ranked_products if p.get("requirementsMatch") or p.get("overallScore", 0) >= 80),
            "approximateMatchCount": sum(1 for p in ranked_products if not (p.get("requirementsMatch") or p.get("overallScore", 0) >= 80)),
            "response": "Analysis completed successfully"
        }

    except Exception as e:
        logger.exception(f"[AnalysisOnly] Failed: {e}")
        return {
            "success": False,
            "ranked_products": [],
            "overall_ranking": [],
            "vendor_matches": {},
            "response": f"Analysis failed: {str(e)}",
            "error": str(e)
        }


def process_from_solution_workflow(
    items: List[Dict[str, Any]],
    session_id: str,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Batch processing for solution workflow.
    Process multiple items in parallel.

    Args:
        items: List of items to process (each with sample_input, category, etc.)
        session_id: Session identifier
        **kwargs: Additional parameters

    Returns:
        List of results for each item
    """
    logger.info(f"[BatchProcess] Processing {len(items)} items for session: {session_id}")

    results = []
    for idx, item in enumerate(items):
        try:
            sample_input = item.get("sample_input", "")
            product_type = item.get("category", "")
            item_session = f"{session_id}_item_{idx}"

            result = run_product_search_workflow(
                user_input=sample_input,
                session_id=item_session,
                expected_product_type=product_type,
                auto_mode=True
            )

            results.append({
                "item_number": idx + 1,
                "category": product_type,
                "search_result": result
            })

        except Exception as e:
            logger.exception(f"[BatchProcess] Item {idx} failed: {e}")
            results.append({
                "item_number": idx + 1,
                "error": str(e)
            })

    return results


def get_schema_only(
    product_type: str,
    session_id: str = "default"
) -> Dict[str, Any]:
    """
    Get product schema without validation.

    Args:
        product_type: Product type
        session_id: Session identifier

    Returns:
        Schema dictionary
    """
    try:
        tool = ValidationTool()
        schema = tool.get_schema_only(
            product_type=product_type,
            session_id=session_id
        )
        return schema

    except Exception as e:
        logger.exception(f"[GetSchemaOnly] Failed: {e}")
        return {}


def validate_with_schema(
    user_input: str,
    schema: Dict[str, Any],
    product_type: str,
    session_id: str = "default"
) -> Dict[str, Any]:
    """
    Validate user input against provided schema.

    Args:
        user_input: User's input
        schema: Product schema
        product_type: Product type
        session_id: Session identifier

    Returns:
        Validation result
    """
    try:
        tool = ValidationTool()
        result = tool.validate_with_schema(
            user_input=user_input,
            schema=schema,
            product_type=product_type,
            session_id=session_id
        )
        return result

    except Exception as e:
        logger.exception(f"[ValidateWithSchema] Failed: {e}")
        return {
            "is_valid": False,
            "error": str(e)
        }


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Alias for old function names
product_search_workflow = run_product_search_workflow
run_single_product_workflow = run_product_search_workflow


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Tool classes
    "ValidationTool",
    "AdvancedSpecificationAgent",
    "VendorAnalysisDeepAgent",
    "VendorAnalysisTool",
    "RankingTool",
    "SalesAgentTool",

    # Workflow functions
    "run_product_search_workflow",
    "run_validation_only",
    "run_advanced_params_only",
    "run_analysis_only",
    "process_from_solution_workflow",

    # Utility functions
    "get_schema_only",
    "validate_with_schema",
    "clear_session_enrichment_cache",

    # Backward compatibility
    "product_search_workflow",
    "run_single_product_workflow",
]
