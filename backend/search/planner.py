# search/planner.py
# =============================================================================
# SEARCH DEEP AGENT PLANNER
# =============================================================================
#
# Analyzes user input and creates an optimized execution plan.
# Similar to FlashPersonality in Solution Deep Agent.
#
# Strategy Selection:
#   FAST  - Simple queries, skip advanced params, relaxed thresholds
#   FULL  - Standard flow with all phases
#   DEEP  - Safety-critical queries, extra validation, strict thresholds
#
# =============================================================================

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Set

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Execution strategy for the search workflow."""
    FAST = "fast"      # Simple queries, optimized path
    FULL = "full"      # Standard flow with all phases
    DEEP = "deep"      # Safety-critical, extra validation


@dataclass
class SearchExecutionPlan:
    """
    Execution plan created by the SearchPlanner.

    Contains strategy selection, phases to run, quality thresholds,
    and hints for individual agents.
    """
    strategy: SearchStrategy
    phases_to_run: List[str]
    skip_advanced_params: bool
    max_vendor_retries: int
    quality_thresholds: Dict[str, int]
    tool_hints: Dict[str, Any]
    product_category: str
    has_safety_requirements: bool
    reasoning: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "strategy": self.strategy.value,
            "phases_to_run": self.phases_to_run,
            "skip_advanced_params": self.skip_advanced_params,
            "max_vendor_retries": self.max_vendor_retries,
            "quality_thresholds": self.quality_thresholds,
            "tool_hints": self.tool_hints,
            "product_category": self.product_category,
            "has_safety_requirements": self.has_safety_requirements,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
        }


class SearchPlanner:
    """
    Analyzes user input and creates an optimized execution plan.

    Dimensions Analyzed:
    1. Query specificity (length, detail level, spec count)
    2. Safety/compliance keywords (ATEX, SIL, IECEx)
    3. Product type clarity
    4. Specification richness
    5. Vendor preferences mentioned
    """

    # Safety keywords that trigger DEEP strategy
    SAFETY_KEYWORDS: Set[str] = {
        # Hazardous area certifications
        'atex', 'iecex', 'fm', 'csa', 'nepsi', 'kosha',
        # Safety integrity levels
        'sil', 'sil 1', 'sil 2', 'sil 3', 'sil 4',
        'functional safety', 'iec 61508', 'iec 61511',
        # Hazardous area classifications
        'zone 0', 'zone 1', 'zone 2', 'zone 20', 'zone 21', 'zone 22',
        'class i', 'class ii', 'class iii',
        'division 1', 'division 2', 'div 1', 'div 2',
        # Protection methods
        'flameproof', 'explosion proof', 'explosion-proof', 'ex-proof',
        'intrinsically safe', 'intrinsic safety', 'ex ia', 'ex ib',
        'ex d', 'ex e', 'ex n', 'ex p', 'ex m', 'ex o', 'ex q',
        # Classified areas
        'hazardous', 'hazardous area', 'classified area', 'hazloc',
        # Critical applications
        'safety critical', 'safety-critical', 'life safety',
        'nuclear', 'offshore', 'subsea',
    }

    # Product category keywords
    PRODUCT_CATEGORIES: Dict[str, Set[str]] = {
        "transmitter": {
            'transmitter', 'pressure transmitter', 'temperature transmitter',
            'level transmitter', 'flow transmitter', 'dp transmitter',
            'differential pressure', 'gauge pressure', 'absolute pressure',
        },
        "analyzer": {
            'analyzer', 'gas analyzer', 'liquid analyzer', 'ph analyzer',
            'conductivity analyzer', 'oxygen analyzer', 'nir analyzer',
            'spectrometer', 'chromatograph', 'gc', 'hplc',
        },
        "sensor": {
            'sensor', 'temperature sensor', 'pressure sensor', 'rtd',
            'thermocouple', 'thermowell', 'probe', 'electrode',
        },
        "valve": {
            'valve', 'control valve', 'globe valve', 'ball valve',
            'butterfly valve', 'gate valve', 'check valve', 'positioner',
        },
        "flowmeter": {
            'flowmeter', 'flow meter', 'coriolis', 'magnetic flowmeter',
            'vortex', 'ultrasonic flowmeter', 'turbine flowmeter',
            'mass flowmeter', 'volumetric',
        },
        "controller": {
            'controller', 'pid controller', 'plc', 'dcs', 'scada',
            'programmable controller', 'process controller',
        },
        "recorder": {
            'recorder', 'data logger', 'chart recorder', 'paperless recorder',
        },
    }

    # Specification patterns that indicate detailed requirements
    SPEC_PATTERNS: List[str] = [
        r'\d+\s*-\s*\d+\s*(psi|bar|mbar|kpa|mpa|pa)',  # Pressure ranges
        r'\d+\s*-\s*\d+\s*(°?[cf]|celsius|fahrenheit|kelvin)',  # Temp ranges
        r'\d+\s*-\s*\d+\s*(ma|mv|v|volt)',  # Electrical ranges
        r'\d+(\.\d+)?\s*%',  # Percentages (accuracy, etc.)
        r'(4-20|0-10|1-5)\s*ma',  # Standard signal ranges
        r'hart|profibus|foundation fieldbus|modbus|ethernet',  # Protocols
        r'(ip|nema)\s*\d+',  # Ingress protection
        r'(dn|nps|ansi)\s*\d+',  # Connection sizes
        r'(ss|stainless|316l?|hastelloy|monel|inconel|titanium)',  # Materials
        r'±\s*\d+(\.\d+)?',  # Accuracy with ± symbol
    ]

    def __init__(self, llm=None):
        """
        Initialize the SearchPlanner.

        Args:
            llm: Optional LLM for complex analysis (not used in rule-based planning)
        """
        self._llm = llm
        self._compiled_spec_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.SPEC_PATTERNS
        ]

    def plan(
        self,
        user_input: str,
        expected_product_type: Optional[str] = None,
        session_context: Optional[Dict[str, Any]] = None,
        skip_advanced_params: Optional[bool] = None,
    ) -> SearchExecutionPlan:
        """
        Create an execution plan based on query analysis.

        Args:
            user_input: Raw user requirement description
            expected_product_type: Hint for product type (from UI selection)
            session_context: Optional session context for continuity
            skip_advanced_params: Override for skipping advanced params

        Returns:
            SearchExecutionPlan with strategy and configuration
        """
        logger.info("[SearchPlanner] Analyzing query for execution plan")

        # Normalize input
        input_lower = user_input.lower()
        input_words = input_lower.split()

        # Analyze dimensions
        has_safety = self._detect_safety_requirements(input_lower)
        spec_count = self._count_specifications(user_input)
        product_category = self._detect_product_category(input_lower, expected_product_type)
        query_complexity = self._assess_query_complexity(user_input, spec_count)

        # Determine strategy
        strategy, reasoning = self._select_strategy(
            has_safety=has_safety,
            spec_count=spec_count,
            query_complexity=query_complexity,
            input_length=len(input_words),
        )

        # Build phases list
        phases = self._build_phases_list(strategy, skip_advanced_params)

        # Build quality thresholds
        thresholds = self._build_quality_thresholds(strategy)

        # Build tool hints
        tool_hints = self._build_tool_hints(
            strategy=strategy,
            has_safety=has_safety,
            product_category=product_category,
        )

        # Determine skip_advanced_params
        should_skip_params = skip_advanced_params if skip_advanced_params is not None else (
            strategy == SearchStrategy.FAST
        )

        plan = SearchExecutionPlan(
            strategy=strategy,
            phases_to_run=phases,
            skip_advanced_params=should_skip_params,
            max_vendor_retries=3 if strategy == SearchStrategy.DEEP else 1,
            quality_thresholds=thresholds,
            tool_hints=tool_hints,
            product_category=product_category,
            has_safety_requirements=has_safety,
            reasoning=reasoning,
            confidence=self._calculate_confidence(spec_count, query_complexity),
        )

        logger.info(
            "[SearchPlanner] Plan created: strategy=%s, phases=%d, safety=%s, category=%s",
            strategy.value,
            len(phases),
            has_safety,
            product_category,
        )

        return plan

    def _detect_safety_requirements(self, input_lower: str) -> bool:
        """Check for safety/compliance keywords in input."""
        for keyword in self.SAFETY_KEYWORDS:
            if keyword in input_lower:
                logger.debug("[SearchPlanner] Safety keyword detected: %s", keyword)
                return True
        return False

    def _count_specifications(self, user_input: str) -> int:
        """Count the number of specific values/specifications in input."""
        count = 0
        for pattern in self._compiled_spec_patterns:
            matches = pattern.findall(user_input)
            count += len(matches)
        return count

    def _detect_product_category(
        self,
        input_lower: str,
        expected_type: Optional[str],
    ) -> str:
        """Detect the product category from input or hint."""
        # Use expected type if provided
        if expected_type:
            expected_lower = expected_type.lower()
            for category, keywords in self.PRODUCT_CATEGORIES.items():
                if any(kw in expected_lower for kw in keywords):
                    return category

        # Detect from input
        for category, keywords in self.PRODUCT_CATEGORIES.items():
            if any(kw in input_lower for kw in keywords):
                return category

        return "unknown"

    def _assess_query_complexity(self, user_input: str, spec_count: int) -> str:
        """Assess overall query complexity."""
        word_count = len(user_input.split())

        if word_count < 10 and spec_count < 2:
            return "simple"
        elif word_count < 30 and spec_count < 5:
            return "moderate"
        else:
            return "complex"

    def _select_strategy(
        self,
        has_safety: bool,
        spec_count: int,
        query_complexity: str,
        input_length: int,
    ) -> tuple:
        """Select execution strategy based on analysis."""
        # Safety requirements always trigger DEEP
        if has_safety:
            return (
                SearchStrategy.DEEP,
                f"DEEP: Safety requirements detected (SIL/ATEX/hazardous area)"
            )

        # Complex queries with many specs use FULL
        if query_complexity == "complex" or spec_count >= 5:
            return (
                SearchStrategy.FULL,
                f"FULL: Complex query with {spec_count} specifications"
            )

        # Simple queries with few specs can use FAST
        if query_complexity == "simple" and spec_count < 2 and input_length < 15:
            return (
                SearchStrategy.FAST,
                f"FAST: Simple query ({input_length} words, {spec_count} specs)"
            )

        # Default to FULL
        return (
            SearchStrategy.FULL,
            f"FULL: Standard query ({query_complexity}, {spec_count} specs)"
        )

    def _build_phases_list(
        self,
        strategy: SearchStrategy,
        skip_advanced_params: Optional[bool],
    ) -> List[str]:
        """Build the list of phases to execute."""
        phases = ["plan", "validate", "collect_requirements"]

        # Add advanced params phase unless skipped
        should_skip = skip_advanced_params if skip_advanced_params is not None else (
            strategy == SearchStrategy.FAST
        )
        if not should_skip:
            phases.append("discover_params")

        # Always include core phases
        phases.extend(["analyze_vendors", "rank", "respond"])

        return phases

    def _build_quality_thresholds(self, strategy: SearchStrategy) -> Dict[str, int]:
        """Build quality thresholds based on strategy."""
        if strategy == SearchStrategy.DEEP:
            return {
                "min_match_score": 70,
                "min_vendor_matches": 2,
                "min_spec_coverage": 80,
                "judge_validation_threshold": 75,
            }
        elif strategy == SearchStrategy.FAST:
            return {
                "min_match_score": 50,
                "min_vendor_matches": 1,
                "min_spec_coverage": 60,
                "judge_validation_threshold": 60,
            }
        else:  # FULL
            return {
                "min_match_score": 60,
                "min_vendor_matches": 1,
                "min_spec_coverage": 70,
                "judge_validation_threshold": 70,
            }

    def _build_tool_hints(
        self,
        strategy: SearchStrategy,
        has_safety: bool,
        product_category: str,
    ) -> Dict[str, Any]:
        """Build hints for individual tools/agents."""
        return {
            "enable_ppi": True,
            "standards_depth": "deep" if has_safety else "shallow",
            "vendor_analysis_depth": "comprehensive" if strategy == SearchStrategy.DEEP else "standard",
            "ranking_criteria": "safety_first" if has_safety else "balanced",
            "product_category": product_category,
            "use_strategy_rag": True,
            "max_vendors_to_analyze": 20 if strategy == SearchStrategy.DEEP else 10,
        }

    def _calculate_confidence(self, spec_count: int, query_complexity: str) -> float:
        """Calculate confidence in the execution plan."""
        base_confidence = 0.7

        # More specs = higher confidence in understanding
        if spec_count >= 3:
            base_confidence += 0.15
        elif spec_count >= 1:
            base_confidence += 0.05

        # Complex queries may have ambiguity
        if query_complexity == "simple":
            base_confidence += 0.1
        elif query_complexity == "complex":
            base_confidence -= 0.1

        return min(0.95, max(0.5, base_confidence))


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_execution_plan(
    user_input: str,
    expected_product_type: Optional[str] = None,
    skip_advanced_params: Optional[bool] = None,
) -> SearchExecutionPlan:
    """
    Convenience function to create an execution plan.

    Args:
        user_input: Raw user requirement description
        expected_product_type: Hint for product type
        skip_advanced_params: Override for skipping advanced params

    Returns:
        SearchExecutionPlan
    """
    planner = SearchPlanner()
    return planner.plan(
        user_input=user_input,
        expected_product_type=expected_product_type,
        skip_advanced_params=skip_advanced_params,
    )
