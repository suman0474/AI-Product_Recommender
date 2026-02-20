# search/prompts.py
# =============================================================================
# SEARCH DEEP AGENT PROMPTS
# =============================================================================
#
# Loads and provides access to prompts for the Search Deep Agent.
#
# =============================================================================

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Cache for loaded prompts
_PROMPTS_CACHE: Optional[Dict[str, str]] = None


def get_search_prompts() -> Dict[str, str]:
    """
    Load search deep agent prompts.

    Returns:
        Dictionary mapping section names to prompt content
    """
    global _PROMPTS_CACHE

    if _PROMPTS_CACHE is not None:
        return _PROMPTS_CACHE

    try:
        from common.prompts.prompt_loader import load_prompt_sections

        _PROMPTS_CACHE = load_prompt_sections("search_deep_agent_prompts")

        if _PROMPTS_CACHE:
            logger.info("[prompts] Loaded %d prompt sections from file", len(_PROMPTS_CACHE))
        else:
            logger.warning("[prompts] No sections loaded from file, using defaults")
            _PROMPTS_CACHE = _get_default_prompts()

        return _PROMPTS_CACHE

    except ImportError as e:
        logger.warning("[prompts] Prompt loader not available: %s, using defaults", e)
        _PROMPTS_CACHE = _get_default_prompts()
        return _PROMPTS_CACHE

    except Exception as exc:
        logger.error("[prompts] Failed to load prompts: %s", exc)
        _PROMPTS_CACHE = _get_default_prompts()
        return _PROMPTS_CACHE


def get_prompt(section: str) -> str:
    """
    Get a specific prompt section.

    Args:
        section: The section name

    Returns:
        Prompt content
    """
    prompts = get_search_prompts()
    return prompts.get(section, "")


def _get_default_prompts() -> Dict[str, str]:
    """Get default prompts as fallback."""
    return {
        "IDENTITY": """You are EnGenie, an expert AI assistant specializing in industrial instrumentation and process control equipment procurement.""",

        "PLANNING_PROTOCOL": """Analyze the user's query to determine the optimal search strategy:
- FAST: Simple queries with clear requirements
- FULL: Standard queries requiring complete analysis
- DEEP: Safety-critical queries requiring thorough validation""",

        "VALIDATION_GUIDANCE": """Extract the product type and requirements from the user's input.
Validate against the schema and identify missing mandatory fields.""",

        "VENDOR_ANALYSIS_GUIDANCE": """Analyze vendor products against the requirements.
Score each product on specification match, vendor reputation, and availability.""",

        "RANKING_GUIDANCE": """Rank products based on overall match quality.
Consider mandatory requirements (40%), optional features (20%), technical specs (15%), vendor factors (15%), and cost (10%).""",

        "RESPONSE_COMPOSER": """Compose a clear, professional response summarizing the search results.
Highlight the top recommendation and key differentiators.""",
    }


# =============================================================================
# PROMPT SECTION CONSTANTS
# =============================================================================

# Section names for easy reference
IDENTITY = "IDENTITY"
PLANNING_PROTOCOL = "PLANNING_PROTOCOL"
REFLECTION_PROTOCOL_VALIDATION = "REFLECTION_PROTOCOL_VALIDATION"
REFLECTION_PROTOCOL_ANALYSIS = "REFLECTION_PROTOCOL_ANALYSIS"
RESPONSE_PROTOCOL = "RESPONSE_PROTOCOL"
PLANNER = "PLANNER"
REASONER_VALIDATION = "REASONER_VALIDATION"
REASONER_ANALYSIS = "REASONER_ANALYSIS"
VALIDATION_GUIDANCE = "VALIDATION_GUIDANCE"
ADVANCED_PARAMS_GUIDANCE = "ADVANCED_PARAMS_GUIDANCE"
VENDOR_ANALYSIS_GUIDANCE = "VENDOR_ANALYSIS_GUIDANCE"
RANKING_GUIDANCE = "RANKING_GUIDANCE"
RESPONSE_COMPOSER = "RESPONSE_COMPOSER"
