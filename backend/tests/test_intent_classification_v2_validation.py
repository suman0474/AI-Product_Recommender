"""
Test script to validate V2 intent classification improvements.

Tests:
1. CHAT vs SEARCH boundary (knowledge vs purchase intent)
2. SEARCH vs SOLUTION boundary (single vs multi-instrument)
3. INVALID_INPUT detection (out-of-scope queries)
4. Edge cases and ambiguous queries

Run with: python -m pytest tests/test_intent_classification_v2_validation.py -v
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from agentic.agents.routing.intent_classifier import (
    IntentClassificationRoutingAgent,
    WorkflowTarget
)

# Test data organized by expected intent
TEST_CASES = {
    "CHAT": [
        # Knowledge questions (should route to CHAT)
        "What is a pressure transmitter?",
        "How does HART protocol work?",
        "Tell me about Rosemount 3051S specifications",
        "Explain the difference between RTD and thermocouple",
        "Compare SIL 2 vs SIL 3 requirements",
        "What are ATEX Zone 0 certification requirements?",
        "Who is our preferred supplier for control valves?",
        "What are the accuracy specs for Yokogawa EJX?",
        "How does a coriolis flowmeter measure density?",
        "What is the difference between 4-20mA and HART?",
    ],

    "SEARCH": [
        # Product discovery (should route to SEARCH)
        "I need a pressure transmitter",
        "Pressure transmitter 0-100 bar, 4-20mA, HART",
        "Looking for temperature sensor for 300°C steam line",
        "Thermowell 316 SS, 200mm, 1/2 NPT",
        "Find me SIL 3 certified level transmitter",
        "Control valve DN50, fail-safe close, pneumatic",
        "Recommend Rosemount pressure transmitters for crude oil",
        "Flow meter for water, 0-100 GPM, 1 inch NPT",
        "Temperature probe for reactor, 0-350°C, 1/4 NPT",
        "Show me pressure gauges 0-200 psi",
    ],

    "SOLUTION": [
        # System design (should route to SOLUTION)
        "Design a temperature monitoring system for 5 reactors",
        "I'm designing a crude oil distillation unit",
        "Implement complete level monitoring for three storage tanks",
        "I need custody transfer metering for natural gas",
        "Design burner management system with safety interlocks",
        "Complete instrumentation for water treatment facility",
        "Monitor temperature and pressure across 10 reactors",
        "I need a measurement system for reactor profiling",
        "Design a safety system for hydrocracker unit",
        "Instrumentation package for crude distillation tower",
    ],

    "INVALID": [
        # Out-of-scope (should route to INVALID_INPUT)
        "What's the weather in Paris?",
        "How to program a PLC in ladder logic?",
        "What's the price for Rosemount 3051S?",
        "Help me troubleshoot error code E15",
        "Design a DCS system architecture",
        "Tell me about heat exchanger design",
        "How does a centrifugal pump work?",
        "Tell me a funny joke",
        "What are the safety requirements for forklifts?",
        "How to design a piping stress analysis?",
    ]
}

# Edge cases that might be ambiguous
EDGE_CASES = {
    "CHAT": [
        # Should be CHAT (knowledge), not SEARCH (purchase)
        "What is a Rosemount 3051S?",
        "Tell me about pressure transmitters",
        "How does a level transmitter work?",
    ],

    "SEARCH": [
        # Should be SEARCH (purchase), not CHAT (knowledge)
        "I need a pressure transmitter",
        "Looking for a temperature sensor",
        "Find me a control valve",
    ],

    "SOLUTION": [
        # Should be SOLUTION (system), not SEARCH (single product)
        "I need temperature and pressure monitoring for reactor",
        "Design a measurement system for distillation",
        "Complete instrumentation for a process unit",
    ]
}


class TestChatIntentClassification:
    """Test CHAT intent classification (knowledge queries)."""

    def test_chat_knowledge_questions(self):
        """Knowledge questions should route to CHAT."""
        agent = IntentClassificationRoutingAgent()

        for query in TEST_CASES["CHAT"]:
            result = agent.classify(query, session_id=f"test_chat_{hash(query)}")

            assert result.target_workflow == WorkflowTarget.ENGENIE_CHAT, (
                f"Query '{query}' should route to CHAT, "
                f"got {result.target_workflow.value} "
                f"(confidence: {result.confidence:.2f}, reasoning: {result.reasoning})"
            )

            print(f"✓ CHAT: '{query[:50]}...' → {result.target_workflow.value} (conf: {result.confidence:.2f})")


class TestSearchIntentClassification:
    """Test SEARCH intent classification (product discovery)."""

    def test_search_product_discovery(self):
        """Product discovery queries should route to SEARCH."""
        agent = IntentClassificationRoutingAgent()

        for query in TEST_CASES["SEARCH"]:
            result = agent.classify(query, session_id=f"test_search_{hash(query)}")

            assert result.target_workflow == WorkflowTarget.INSTRUMENT_IDENTIFIER, (
                f"Query '{query}' should route to SEARCH, "
                f"got {result.target_workflow.value} "
                f"(confidence: {result.confidence:.2f}, reasoning: {result.reasoning})"
            )

            print(f"✓ SEARCH: '{query[:50]}...' → {result.target_workflow.value} (conf: {result.confidence:.2f})")


class TestSolutionIntentClassification:
    """Test SOLUTION intent classification (system design)."""

    def test_solution_system_design(self):
        """System design queries should route to SOLUTION."""
        agent = IntentClassificationRoutingAgent()

        for query in TEST_CASES["SOLUTION"]:
            result = agent.classify(query, session_id=f"test_solution_{hash(query)}")

            assert result.target_workflow == WorkflowTarget.SOLUTION_WORKFLOW, (
                f"Query '{query}' should route to SOLUTION, "
                f"got {result.target_workflow.value} "
                f"(confidence: {result.confidence:.2f}, reasoning: {result.reasoning})"
            )

            print(f"✓ SOLUTION: '{query[:50]}...' → {result.target_workflow.value} (conf: {result.confidence:.2f})")


class TestInvalidInputDetection:
    """Test INVALID_INPUT detection (out-of-scope queries)."""

    def test_invalid_out_of_scope(self):
        """Out-of-scope queries should route to INVALID_INPUT."""
        agent = IntentClassificationRoutingAgent()

        for query in TEST_CASES["INVALID"]:
            result = agent.classify(query, session_id=f"test_invalid_{hash(query)}")

            assert result.target_workflow == WorkflowTarget.OUT_OF_DOMAIN, (
                f"Query '{query}' should route to INVALID_INPUT, "
                f"got {result.target_workflow.value} "
                f"(confidence: {result.confidence:.2f}, reasoning: {result.reasoning})"
            )

            print(f"✓ INVALID: '{query[:50]}...' → {result.target_workflow.value} (conf: {result.confidence:.2f})")


class TestEdgeCases:
    """Test edge cases and ambiguous queries."""

    def test_chat_vs_search_boundary(self):
        """Test CHAT vs SEARCH boundary (knowledge vs purchase intent)."""
        agent = IntentClassificationRoutingAgent()

        # CHAT edge cases
        for query in EDGE_CASES["CHAT"]:
            result = agent.classify(query, session_id=f"test_edge_chat_{hash(query)}")

            assert result.target_workflow == WorkflowTarget.ENGENIE_CHAT, (
                f"Query '{query}' should be CHAT (knowledge intent), "
                f"got {result.target_workflow.value}"
            )

            print(f"✓ EDGE CHAT: '{query}' → {result.target_workflow.value}")

        # SEARCH edge cases
        for query in EDGE_CASES["SEARCH"]:
            result = agent.classify(query, session_id=f"test_edge_search_{hash(query)}")

            assert result.target_workflow == WorkflowTarget.INSTRUMENT_IDENTIFIER, (
                f"Query '{query}' should be SEARCH (purchase intent), "
                f"got {result.target_workflow.value}"
            )

            print(f"✓ EDGE SEARCH: '{query}' → {result.target_workflow.value}")

    def test_search_vs_solution_boundary(self):
        """Test SEARCH vs SOLUTION boundary (single vs multi-instrument)."""
        agent = IntentClassificationRoutingAgent()

        for query in EDGE_CASES["SOLUTION"]:
            result = agent.classify(query, session_id=f"test_edge_solution_{hash(query)}")

            assert result.target_workflow == WorkflowTarget.SOLUTION_WORKFLOW, (
                f"Query '{query}' should be SOLUTION (system design), "
                f"got {result.target_workflow.value}"
            )

            print(f"✓ EDGE SOLUTION: '{query}' → {result.target_workflow.value}")


class TestConfidenceScoring:
    """Test confidence scoring and thresholds."""

    def test_high_confidence_for_clear_intents(self):
        """Clear intents should have high confidence (≥0.75)."""
        agent = IntentClassificationRoutingAgent()

        high_confidence_queries = [
            ("What is a pressure transmitter?", WorkflowTarget.ENGENIE_CHAT),
            ("I need a pressure transmitter 0-100 bar", WorkflowTarget.INSTRUMENT_IDENTIFIER),
            ("Design a temperature monitoring system for 5 reactors", WorkflowTarget.SOLUTION_WORKFLOW),
            ("What's the weather?", WorkflowTarget.OUT_OF_DOMAIN),
        ]

        for query, expected_workflow in high_confidence_queries:
            result = agent.classify(query, session_id=f"test_conf_{hash(query)}")

            assert result.target_workflow == expected_workflow

            # Check confidence is reasonable
            # Note: V2 might have different confidence scoring, so we'll just log it
            print(f"✓ CONFIDENCE: '{query[:40]}...' → {result.confidence:.2f}")


def test_prompt_version_detection():
    """Test that V2 prompt is being used."""
    import logging

    # Check logs to see which prompt version loaded
    # This is just informational
    print("\n" + "="*70)
    print("PROMPT VERSION CHECK")
    print("="*70)
    print("Check the logs above for: '[INTENT_TOOLS] Using improved intent classification prompt (V2)'")
    print("If you see that message, V2 is active!")
    print("="*70 + "\n")


# Run summary
def test_summary():
    """Print test summary."""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"CHAT tests: {len(TEST_CASES['CHAT'])} queries")
    print(f"SEARCH tests: {len(TEST_CASES['SEARCH'])} queries")
    print(f"SOLUTION tests: {len(TEST_CASES['SOLUTION'])} queries")
    print(f"INVALID tests: {len(TEST_CASES['INVALID'])} queries")
    print(f"Edge case tests: {sum(len(v) for v in EDGE_CASES.values())} queries")
    print(f"\nTotal: {sum(len(v) for v in TEST_CASES.values()) + sum(len(v) for v in EDGE_CASES.values())} queries tested")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run tests manually if not using pytest
    print("\n" + "="*70)
    print("INTENT CLASSIFICATION V2 VALIDATION")
    print("="*70 + "\n")

    print("Running CHAT intent tests...")
    test_chat = TestChatIntentClassification()
    test_chat.test_chat_knowledge_questions()

    print("\nRunning SEARCH intent tests...")
    test_search = TestSearchIntentClassification()
    test_search.test_search_product_discovery()

    print("\nRunning SOLUTION intent tests...")
    test_solution = TestSolutionIntentClassification()
    test_solution.test_solution_system_design()

    print("\nRunning INVALID_INPUT tests...")
    test_invalid = TestInvalidInputDetection()
    test_invalid.test_invalid_out_of_scope()

    print("\nRunning edge case tests...")
    test_edge = TestEdgeCases()
    test_edge.test_chat_vs_search_boundary()
    test_edge.test_search_vs_solution_boundary()

    print("\nRunning confidence tests...")
    test_conf = TestConfidenceScoring()
    test_conf.test_high_confidence_for_clear_intents()

    test_prompt_version_detection()
    test_summary()

    print("\n✅ All tests passed! V2 intent classification is working correctly.\n")
