"""
Test script for RAG Classification Logic
Tests the updated intent classifier with SOLUTION category integration.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agentic.workflows.engenie_chat.engenie_chat_intent_agent import classify_query, DataSource


def test_classification():
    """Test the intent classifier with various queries."""
    
    test_cases = [
        # Solution Design queries (should route to SOLUTION)
        {
            "query": "Design a custody transfer metering skid",
            "expected": DataSource.SOLUTION,
            "description": "System design request"
        },
        {
            "query": "I need a complete control system for a boiler",
            "expected": DataSource.SOLUTION,
            "description": "Complete system request"
        },
        {
            "query": "Solution for fiscal metering",
            "expected": DataSource.SOLUTION,
            "description": "Solution keyword"
        },
        {
            "query": "Build a DCS system for refinery",
            "expected": DataSource.SOLUTION,
            "description": "Build system request"
        },
        
        # Product Search queries (should route to INDEX_RAG)
        {
            "query": "Rosemount 3051 specifications",
            "expected": DataSource.INDEX_RAG,
            "description": "Product spec query"
        },
        {
            "query": "I need a pressure transmitter datasheet",
            "expected": DataSource.INDEX_RAG,
            "description": "Datasheet request"
        },
        {
            "query": "Yokogawa EJX series features",
            "expected": DataSource.INDEX_RAG,
            "description": "Product features"
        },
        
        # Strategy queries (should route to STRATEGY_RAG)
        {
            "query": "Who is our preferred vendor for transmitters?",
            "expected": DataSource.STRATEGY_RAG,
            "description": "Vendor preference query"
        },
        {
            "query": "Approved suppliers for control valves",
            "expected": DataSource.STRATEGY_RAG,
            "description": "Approved suppliers"
        },
        
        # Standards queries (should route to STANDARDS_RAG)
        {
            "query": "What are the SIL 3 requirements?",
            "expected": DataSource.STANDARDS_RAG,
            "description": "SIL requirement query"
        },
        {
            "query": "ATEX Zone 1 certification requirements",
            "expected": DataSource.STANDARDS_RAG,
            "description": "ATEX query"
        },
        
        # Hybrid queries
        {
            "query": "Is Rosemount 3051 certified for SIL 3?",
            "expected": DataSource.HYBRID,
            "description": "Product + Standards"
        },
    ]
    
    print("=" * 80)
    print("RAG CLASSIFICATION TEST RESULTS")
    print("=" * 80)
    print()
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected = test_case["expected"]
        description = test_case["description"]
        
        # Run classification
        result, confidence, reasoning = classify_query(query)
        
        # Check if correct
        is_correct = result == expected
        status = "[PASS]" if is_correct else "[FAIL]"
        
        if is_correct:
            passed += 1
        else:
            failed += 1
        
        # Print result
        print(f"Test {i}: {status}")
        print(f"  Query: '{query}'")
        print(f"  Description: {description}")
        print(f"  Expected: {expected.value}")
        print(f"  Got: {result.value} (confidence: {confidence:.2f})")
        print(f"  Reasoning: {reasoning}")
        print()
    
    # Summary
    print("=" * 80)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)
    
    if failed == 0:
        print("SUCCESS: All tests passed! Classification is working correctly.")
    else:
        print(f"WARNING: {failed} test(s) failed. Review the results above.")
    
    return failed == 0


if __name__ == "__main__":
    success = test_classification()
    sys.exit(0 if success else 1)
