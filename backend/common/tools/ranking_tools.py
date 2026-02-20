# tools/ranking_tools.py
# Product Ranking and Judging Tools

import json
import logging
from typing import Dict, Any, List, Optional, Union
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os
import re

# Import from common modules
try:
    from common.core.chaining import (
        setup_langchain_components,
        invoke_ranking_chain,
        to_dict_if_pydantic,
        get_final_ranking
    )
except ImportError:
    logging.getLogger(__name__).warning("Could not import common.core.chaining. LLM ranking may fail.")

from common.services.llm.fallback import create_llm_with_fallback
from common.prompts import RANKING_PROMPTS, ANALYSIS_TOOL_VENDOR_ANALYSIS_PROMPT, SCHEMA_VALIDATION_PROMPT

# Debug flags
try:
    from debug_flags import debug_log, timed_execution
except ImportError:
    def debug_log(module, **kwargs):
        def decorator(func): return func
        return decorator
    def timed_execution(module, **kwargs):
        def decorator(func): return func
        return decorator

logger = logging.getLogger(__name__)

# Load Prompts for Judge Tool
try:
    JUDGE_PROMPT = RANKING_PROMPTS.get("JUDGE", "")
except Exception:
    JUDGE_PROMPT = "" 
    logger.warning("Failed to load JUDGE prompt from ranking_prompts")


# ============================================================================
# TOOL INPUT SCHEMAS
# ============================================================================

class RankProductsInput(BaseModel):
    """Input for product ranking"""
    vendor_analysis: Dict[str, Any] = Field(description="Full vendor analysis results containing vendor_matches")
    requirements: Optional[Dict[str, Any]] = Field(default=None, description="Original user requirements")
    session_id: Optional[str] = Field(default=None, description="Session tracking ID")
    use_llm_ranking: bool = Field(default=True, description="Whether to use LLM for ranking")


class JudgeAnalysisInput(BaseModel):
    """Input for analysis judging"""
    original_requirements: Dict[str, Any] = Field(description="Original user requirements")
    vendor_analysis: Dict[str, Any] = Field(description="Vendor analysis results")
    strategy_rules: Optional[Dict[str, Any]] = Field(default=None, description="Procurement strategy rules")


# ============================================================================
# HELPER FUNCTIONS (Refactored from RankingTool)
# ============================================================================

def _normalize_reasoning(field_data: Union[str, List[Any]]) -> str:
    """Normalize complex LLM output into clean Markdown string."""
    if isinstance(field_data, str):
        return field_data
    
    if isinstance(field_data, list):
        items = []
        for item in field_data:
            # Handle Dictionary format (seen in logs: {'parameter':..., 'input':...})
            if isinstance(item, dict):
                # Construct readable string from dict keys
                param = item.get('parameter', '')
                spec = item.get('input_value', item.get('product_specification', ''))
                text = item.get('holistic_explanation', item.get('explanation', item.get('limitation', '')))
                # Fallback for simple "limitation" dicts
                if not param and not spec and text:
                    items.append(f"- {text}")
                elif param:
                    items.append(f"**{param}**: {spec} - {text}")
                else:
                    items.append(str(item))
            else:
                items.append(str(item))
        return "\n\n".join(items)
    
    return str(field_data)

def _format_field_name(field: str) -> str:
    """Convert camelCase or snake_case to Title Case."""
    words = re.sub(r'([a-z])([A-Z])', r'\1 \2', field)
    words = words.replace('_', ' ')
    return words.title()

def _rank_with_llm(vendor_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Use LLM for intelligent ranking."""
    try:
        logger.info("[RankingTool] Using LLM-powered ranking")

        # Setup LangChain components
        components = setup_langchain_components()
        format_instructions = components.get('ranking_format_instructions', '')

        # Prepare vendor analysis for prompt
        vendor_analysis_str = json.dumps(vendor_analysis, indent=2, default=str)

        # Invoke ranking chain
        ranking_result = invoke_ranking_chain(
            components,
            vendor_analysis_str,
            format_instructions
        )

        if not ranking_result:
            logger.warning("[RankingTool] LLM ranking returned empty result")
            return []

        # Convert to dict if Pydantic
        ranking_dict = to_dict_if_pydantic(ranking_result)

        # Extract ranked products
        ranked_products = ranking_dict.get('ranked_products', ranking_dict.get('rankedProducts', []))

        return _normalize_ranked_products(ranked_products, vendor_analysis, method="llm")

    except Exception as e:
        logger.error("[RankingTool] LLM ranking failed: %s", str(e), exc_info=True)
        return []

def _rank_by_score(vendor_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fallback: Simple score-based ranking without LLM."""
    try:
        logger.info("[RankingTool] Using score-based ranking (fallback)")

        # Use existing get_final_ranking function
        ranking_result = get_final_ranking(vendor_analysis)

        if not ranking_result:
            return []

        ranking_dict = to_dict_if_pydantic(ranking_result)

        # Handle both 'ranked_products' and 'rankedProducts' keys
        ranked_products = ranking_dict.get('ranked_products', ranking_dict.get('rankedProducts', []))

        return _normalize_ranked_products(ranked_products, vendor_analysis, method="score")

    except Exception as e:
        logger.error("[RankingTool] Score-based ranking failed: %s", str(e), exc_info=True)
        return []

def _normalize_ranked_products(
    ranked_products: List[Dict[str, Any]], 
    vendor_analysis: Dict[str, Any],
    method: str = "llm"
) -> List[Dict[str, Any]]:
    """Normalize ranked products data structure."""
    normalized_products = []
    
    # Create lookup dictionaries
    pricing_lookup = {}
    description_lookup = {}
    standards_lookup = {}
    matched_reqs_lookup = {}
    
    if vendor_analysis and 'vendor_matches' in vendor_analysis:
        for match in vendor_analysis['vendor_matches']:
            key = (match.get('vendor', ''), match.get('productName', match.get('product_name', '')))
            pricing_lookup[key] = {
                'pricing_url': match.get('pricing_url', ''),
                'pricing_source': match.get('pricing_source', '')
            }
            description_lookup[key] = match.get('product_description', match.get('productDescription', ''))
            standards_lookup[key] = match.get('standards_compliance', match.get('standardsCompliance', {}))
            matched_reqs_lookup[key] = {
                'matched_requirements': match.get('matched_requirements', match.get('matchedRequirements', {})),
                'unmatched_requirements': match.get('unmatched_requirements', match.get('unmatchedRequirements', []))
            }

    for product in ranked_products:
        p_vendor = product.get('vendor', '')
        # LLM ranking prompt returns "model" field; also handle "product_name"/"productName"
        p_name = product.get('product_name', product.get('productName', product.get('model', product.get('model_number', ''))))

        # Retrieve info from lookups
        pricing_info = pricing_lookup.get((p_vendor, p_name), {})
        product_desc = description_lookup.get((p_vendor, p_name), '')
        standards_info = standards_lookup.get((p_vendor, p_name), {})
        reqs_info = matched_reqs_lookup.get((p_vendor, p_name), {})

        # Compute match for score-based
        if method == "score":
            score = product.get('match_score', product.get('matchScore', product.get('overall_score', 0)))
            computed_match = score >= 80
            requirements_match = product.get('requirements_match', product.get('requirementsMatch', computed_match))
            # LLM ranking prompt returns "strengths"; also handle "key_strengths"/"keyStrengths"
            key_strengths = product.get('key_strengths', product.get('keyStrengths', product.get('strengths', product.get('reasoning', ''))))
            concerns = product.get('concerns', product.get('limitations', ''))
        else:
            requirements_match = product.get('requirements_match', product.get('requirementsMatch', False))
            # LLM ranking prompt returns "strengths"; also handle "key_strengths"/"keyStrengths"
            key_strengths = _normalize_reasoning(product.get('key_strengths', product.get('keyStrengths', product.get('strengths', []))))
            concerns = _normalize_reasoning(product.get('concerns', product.get('limitations', [])))

        normalized = {
            'productName': p_name,
            'vendor': p_vendor,
            # LLM ranking prompt returns "model"; prefer model_family/modelFamily for series name
            'modelFamily': product.get('model_family', product.get('modelFamily', product.get('model', ''))),
            'overallScore': product.get('overall_score', product.get('overallScore', product.get('match_score', product.get('matchScore', 0)))),
            'matchScore': product.get('match_score', product.get('matchScore', product.get('overall_score', product.get('overallScore', 0)))),
            'requirementsMatch': requirements_match,
            'keyStrengths': key_strengths,
            'concerns': concerns,
            'recommendation': product.get('recommendation', ''),
            'productDescription': product_desc,
            'standardsCompliance': standards_info,
            'matchedRequirements': reqs_info.get('matched_requirements', {}),
            'unmatchedRequirements': reqs_info.get('unmatched_requirements', []),
            'pricingUrl': pricing_info.get('pricing_url', ''),
            'pricingSource': pricing_info.get('pricing_source', '')
        }
        normalized_products.append(normalized)

    return normalized_products


# ============================================================================
# TOOLS
# ============================================================================

@tool("rank_products", args_schema=RankProductsInput)
@timed_execution("TOOLS", threshold_ms=20000)
@debug_log("TOOLS", log_args=True, log_result=False)
def rank_products_tool(
    vendor_analysis: Dict[str, Any],
    requirements: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    use_llm_ranking: bool = True
) -> Dict[str, Any]:
    """
    Rank products based on vendor analysis results.
    Returns ordered list with detailed analysis.
    """
    logger.info("[RankingTool] Starting product ranking")
    logger.info("[RankingTool] Session: %s", session_id or "N/A")

    result = {
        "success": False,
        "session_id": session_id
    }

    try:
        # Step 1: Validate input
        vendor_matches = vendor_analysis.get('vendor_matches', [])

        if not vendor_matches:
            result['success'] = True
            result['overall_ranking'] = []
            result['top_product'] = None
            result['total_ranked'] = 0
            result['ranking_summary'] = "No products available to rank"
            return result

        # Step 2: Generate ranking
        overall_ranking = []
        if use_llm_ranking:
            overall_ranking = _rank_with_llm(vendor_analysis)
        
        if not overall_ranking:
             if use_llm_ranking:
                 logger.warning("[RankingTool] LLM ranking failed, using score-based fallback")
             overall_ranking = _rank_by_score(vendor_analysis)

        # Step 3: Sort and Finalize
        overall_ranking = sorted(
            overall_ranking,
            key=lambda x: x.get('overallScore', x.get('matchScore', 0)),
            reverse=True
        )

        for i, product in enumerate(overall_ranking):
            product['rank'] = i + 1

        top_product = overall_ranking[0] if overall_ranking else None

        # Build result
        result['success'] = True
        result['overall_ranking'] = overall_ranking
        result['top_product'] = top_product
        result['total_ranked'] = len(overall_ranking)

        # Generate summary
        if top_product:
            top_name = top_product.get('productName', 'Unknown')
            top_vendor = top_product.get('vendor', 'Unknown')
            top_score = top_product.get('overallScore', 0)
            result['ranking_summary'] = (
                f"Ranked {len(overall_ranking)} products. "
                f"Top recommendation: {top_name} by {top_vendor} "
                f"with {top_score}% match score"
            )
        else:
            result['ranking_summary'] = "No products ranked"

        logger.info("[RankingTool] Ranking complete: %s", result['ranking_summary'])
        return result

    except Exception as e:
        logger.error("[RankingTool] Ranking failed: %s", str(e), exc_info=True)
        result['success'] = False
        result['error'] = str(e)
        return result


@tool("judge_analysis", args_schema=JudgeAnalysisInput)
def judge_analysis_tool(
    original_requirements: Dict[str, Any],
    vendor_analysis: Dict[str, Any],
    strategy_rules: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate and judge the vendor analysis results.
    Checks for consistency, accuracy, and strategy compliance.
    """
    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)
        parser = JsonOutputParser()

        chain = prompt | llm | parser

        result = chain.invoke({
            "original_requirements": json.dumps(original_requirements, indent=2),
            "vendor_analysis": json.dumps(vendor_analysis, indent=2),
            "strategy_rules": json.dumps(strategy_rules, indent=2) if strategy_rules else "No specific strategy rules"
        })

        return {
            "success": True,
            "is_valid": result.get("is_valid", False),
            "validation_score": result.get("validation_score", 0),
            "issues": result.get("issues", []),
            "approved_vendors": result.get("approved_vendors", []),
            "rejected_vendors": result.get("rejected_vendors", []),
            "validation_summary": result.get("validation_summary")
        }

    except Exception as e:
        logger.error(f"Analysis judging failed: {e}")
        return {
            "success": False,
            "is_valid": False,
            "issues": [{"type": "error", "description": str(e)}],
            "error": str(e)
        }


# Comparison Table Generation (Helper)
def generate_comparison_table(
    overall_ranking: List[Dict[str, Any]],
    requirements: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate a comparison table for ranked products."""
    if not overall_ranking:
        return {"columns": [], "rows": []}

    # Extract all unique parameters from requirements
    all_params = set()
    mandatory = requirements.get('mandatoryRequirements', requirements.get('mandatory', {}))
    all_params.update(mandatory.keys())
    optional = requirements.get('optionalRequirements', requirements.get('optional', {}))
    all_params.update(optional.keys())

    # Build columns
    columns = [
        {"key": "rank", "label": "Rank"},
        {"key": "vendor", "label": "Vendor"},
        {"key": "product_name", "label": "Product"},
        {"key": "match_score", "label": "Match %"},
    ]

    for param in sorted(all_params):
        columns.append({
            "key": param,
            "label": _format_field_name(param)
        })

    # Build rows
    rows = []
    for product in overall_ranking:
        row = {
            "rank": product.get('rank', 0),
            "vendor": product.get('vendor', 'Unknown'),
            "product_name": product.get('productName', 'Unknown'),
            "match_score": product.get('matchScore', 0),
        }
        rows.append(row)

    return {
        "columns": columns,
        "rows": rows
    }
