"""
Test Suite for Path 2 Features

Tests:
1. DomainValidator - Scope validation
2. Intent-Specific Confidence Thresholds
3. ClassificationMetrics - Performance tracking
4. Workflow Capability Validation

Run with: python -m pytest tests/test_path2_features.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from agentic.agents.routing.domain_validator import (
    DomainValidator,
    get_domain_validator,
    is_in_domain,
    validate_workflow_capability,
    get_reject_message
)
from agentic.agents.routing.classification_metrics import (
    ClassificationMetrics,
    get_classification_metrics
)
from agentic.agents.routing.intent_classifier import IntentConfig, WorkflowTarget


class TestDomainValidator:
    """Test Domain Validator for scope checking."""

    def test_in_scope_queries(self):
        """Test queries that should be accepted (in-domain)."""
        validator = DomainValidator()

        in_scope_queries = [
            # Instruments
            "What is a pressure transmitter?",
            "I need a temperature sensor",
            "Looking for a flow meter",
            "Control valve specifications",

            # Standards
            "What are SIL 3 requirements?",
            "ATEX Zone 0 certification",
            "IEC 61508 compliance",

            # Vendors
            "Rosemount 3051S datasheet",
            "Yokogawa EJX series",
            "Preferred suppliers for valves",

            # System design
            "Design a measurement system",
            "Complete instrumentation package",
        ]

        for query in in_scope_queries:
            is_valid, reason = validator.is_in_domain(query)
            assert is_valid, f"Query should be in-scope: '{query}'. Reason: {reason}"
            print(f"[OK] IN-SCOPE: {query[:50]}")

    def test_out_of_scope_queries(self):
        """Test queries that should be rejected (out-of-domain)."""
        validator = DomainValidator()

        out_of_scope_queries = [
            # PLC/SCADA
            "How to program a PLC in ladder logic?",
            "HMI design best practices",
            "SCADA programming tutorial",

            # Pricing/Commercial
            "What's the price for Rosemount 3051S?",
            "How much does a transmitter cost?",
            "Is this product in stock?",

            # Troubleshooting
            "How to troubleshoot error code E15?",
            "My transmitter is not working",
            "Maintenance procedure for valves",

            # Non-industrial
            "What's the weather in Paris?",
            "Tell me a funny joke",
            "Who won the World Cup?",

            # Mechanical (non-instrument)
            "Pump sizing calculation",
            "Heat exchanger design",
            "Compressor selection",
        ]

        for query in out_of_scope_queries:
            is_valid, reason = validator.is_in_domain(query)
            assert not is_valid, f"Query should be out-of-scope: '{query}'"
            print(f"[OK] OUT-OF-SCOPE: {query[:50]} -> {reason}")

    def test_workflow_capability_validation(self):
        """Test that workflows can/cannot handle specific query types."""
        validator = DomainValidator()

        # CHAT cannot handle purchase intent
        can_handle, reason = validator.validate_workflow_capability(
            "I need a pressure transmitter",
            "engenie_chat"
        )
        assert not can_handle
        assert "should route to SEARCH" in reason
        print(f"[OK] CHAT cannot handle purchase: {reason}")

        # CHAT cannot handle design intent
        can_handle, reason = validator.validate_workflow_capability(
            "Design a measurement system",
            "engenie_chat"
        )
        assert not can_handle
        assert "should route to SOLUTION" in reason
        print(f"[OK] CHAT cannot handle design: {reason}")

        # SEARCH cannot handle knowledge questions
        can_handle, reason = validator.validate_workflow_capability(
            "What is a pressure transmitter?",
            "instrument_identifier"
        )
        assert not can_handle
        assert "should route to CHAT" in reason
        print(f"[OK] SEARCH cannot handle knowledge: {reason}")

        # SOLUTION cannot handle knowledge questions (except "what is needed")
        can_handle, reason = validator.validate_workflow_capability(
            "What is a distillation unit?",
            "solution"
        )
        assert not can_handle
        assert "should route to CHAT" in reason
        print(f"[OK] SOLUTION cannot handle knowledge: {reason}")

    def test_reject_messages(self):
        """Test user-friendly reject messages."""
        validator = DomainValidator()

        # PLC-related
        message = validator.get_reject_message("Out-of-domain keyword detected: 'plc programming'")
        assert "PLC/SCADA programming" in message
        assert "automation engineering team" in message
        print(f"[OK] PLC reject message: {message[:80]}...")

        # Pricing-related
        message = validator.get_reject_message("Out-of-domain keyword detected: 'price'")
        assert "pricing information" in message
        assert "procurement team" in message
        print(f"[OK] Pricing reject message: {message[:80]}...")

        # Troubleshooting
        message = validator.get_reject_message("Out-of-domain keyword detected: 'troubleshoot'")
        assert "troubleshooting" in message
        assert "maintenance team" in message
        print(f"[OK] Troubleshooting reject message: {message[:80]}...")


class TestIntentSpecificConfidence:
    """Test intent-specific confidence thresholds."""

    def test_confidence_thresholds(self):
        """Test that each workflow has correct threshold."""
        thresholds = IntentConfig.CONFIDENCE_THRESHOLDS

        assert thresholds[WorkflowTarget.ENGENIE_CHAT] == 0.60
        assert thresholds[WorkflowTarget.INSTRUMENT_IDENTIFIER] == 0.75
        assert thresholds[WorkflowTarget.SOLUTION_WORKFLOW] == 0.80
        assert thresholds[WorkflowTarget.OUT_OF_DOMAIN] == 0.90

        print("[OK] All confidence thresholds set correctly")

    def test_should_accept_classification(self):
        """Test acceptance based on confidence thresholds."""

        # CHAT accepts lower confidence (0.60)
        assert IntentConfig.should_accept_classification(WorkflowTarget.ENGENIE_CHAT, 0.65)
        assert not IntentConfig.should_accept_classification(WorkflowTarget.ENGENIE_CHAT, 0.55)

        # SEARCH requires higher confidence (0.75)
        assert IntentConfig.should_accept_classification(WorkflowTarget.INSTRUMENT_IDENTIFIER, 0.80)
        assert not IntentConfig.should_accept_classification(WorkflowTarget.INSTRUMENT_IDENTIFIER, 0.70)

        # SOLUTION requires highest confidence (0.80)
        assert IntentConfig.should_accept_classification(WorkflowTarget.SOLUTION_WORKFLOW, 0.85)
        assert not IntentConfig.should_accept_classification(WorkflowTarget.SOLUTION_WORKFLOW, 0.75)

        # OUT_OF_DOMAIN requires very high confidence (0.90)
        assert IntentConfig.should_accept_classification(WorkflowTarget.OUT_OF_DOMAIN, 0.95)
        assert not IntentConfig.should_accept_classification(WorkflowTarget.OUT_OF_DOMAIN, 0.85)

        print("[OK] Confidence acceptance logic works correctly")

    def test_needs_disambiguation(self):
        """Test disambiguation detection."""

        # Low confidence for CHAT should trigger disambiguation
        needs_disamb, question = IntentConfig.needs_disambiguation(
            WorkflowTarget.ENGENIE_CHAT, 0.55
        )
        assert needs_disamb
        assert question is not None
        assert "learn" in question.lower() or "find" in question.lower()
        print(f"[OK] CHAT disambiguation: {question}")

        # Low confidence for SEARCH
        needs_disamb, question = IntentConfig.needs_disambiguation(
            WorkflowTarget.INSTRUMENT_IDENTIFIER, 0.70
        )
        assert needs_disamb
        assert "purchase" in question.lower() or "looking" in question.lower()
        print(f"[OK] SEARCH disambiguation: {question}")

        # Low confidence for SOLUTION
        needs_disamb, question = IntentConfig.needs_disambiguation(
            WorkflowTarget.SOLUTION_WORKFLOW, 0.75
        )
        assert needs_disamb
        assert "system" in question.lower() or "design" in question.lower()
        print(f"[OK] SOLUTION disambiguation: {question}")

        # High confidence should not need disambiguation
        needs_disamb, question = IntentConfig.needs_disambiguation(
            WorkflowTarget.ENGENIE_CHAT, 0.95
        )
        assert not needs_disamb
        assert question is None
        print("[OK] High confidence does not need disambiguation")


class TestClassificationMetrics:
    """Test Classification Metrics tracking."""

    def test_metrics_recording(self):
        """Test recording classifications."""
        metrics = ClassificationMetrics()
        metrics.clear()  # Start fresh

        # Record some classifications
        metrics.record_classification(
            query="What is a pressure transmitter?",
            intent="chat",
            confidence=0.95,
            target_workflow="engenie_chat",
            classification_time_ms=250.5,
            layer_used="semantic",
            is_solution=False,
            session_id="test_session"
        )

        metrics.record_classification(
            query="I need a transmitter 0-100 bar",
            intent="search",
            confidence=0.85,
            target_workflow="instrument_identifier",
            classification_time_ms=300.0,
            layer_used="pattern",
            is_solution=False,
            session_id="test_session"
        )

        # Check that recordings are stored
        recent = metrics.get_recent_classifications(count=2)
        assert len(recent) == 2
        assert recent[0]["intent"] == "chat"
        assert recent[1]["intent"] == "search"

        print(f"[OK] Recorded {len(recent)} classifications")

    def test_accuracy_by_layer(self):
        """Test layer statistics."""
        metrics = ClassificationMetrics()
        metrics.clear()

        # Record classifications with different layers
        for i in range(5):
            metrics.record_classification(
                query=f"query_{i}",
                intent="chat",
                confidence=0.90,
                target_workflow="engenie_chat",
                classification_time_ms=5.0,
                layer_used="pattern",
                is_solution=False
            )

        for i in range(3):
            metrics.record_classification(
                query=f"query_{i}",
                intent="search",
                confidence=0.80,
                target_workflow="instrument_identifier",
                classification_time_ms=250.0,
                layer_used="semantic",
                is_solution=False
            )

        for i in range(2):
            metrics.record_classification(
                query=f"query_{i}",
                intent="solution",
                confidence=0.75,
                target_workflow="solution",
                classification_time_ms=1500.0,
                layer_used="llm",
                is_solution=True
            )

        stats = metrics.get_accuracy_by_layer()

        assert "pattern" in stats
        assert stats["pattern"]["count"] == 5
        assert stats["pattern"]["avg_confidence"] == 0.90

        assert "semantic" in stats
        assert stats["semantic"]["count"] == 3
        assert stats["semantic"]["avg_confidence"] == 0.80

        assert "llm" in stats
        assert stats["llm"]["count"] == 2
        assert stats["llm"]["avg_confidence"] == 0.75

        print(f"[OK] Layer statistics: {stats}")

    def test_low_confidence_samples(self):
        """Test low-confidence sample collection."""
        metrics = ClassificationMetrics()
        metrics.clear()

        # Record mix of high and low confidence
        metrics.record_classification(
            query="high confidence query",
            intent="chat",
            confidence=0.95,
            target_workflow="engenie_chat",
            classification_time_ms=100.0,
            layer_used="semantic"
        )

        metrics.record_classification(
            query="low confidence query 1",
            intent="search",
            confidence=0.60,
            target_workflow="instrument_identifier",
            classification_time_ms=200.0,
            layer_used="llm"
        )

        metrics.record_classification(
            query="low confidence query 2",
            intent="solution",
            confidence=0.55,
            target_workflow="solution",
            classification_time_ms=300.0,
            layer_used="llm"
        )

        low_conf = metrics.get_low_confidence_samples(threshold=0.70)

        assert len(low_conf) == 2
        assert all(s["confidence"] < 0.70 for s in low_conf)

        print(f"[OK] Found {len(low_conf)} low-confidence samples")

    def test_misclassification_detection(self):
        """Test misclassification candidate detection."""
        metrics = ClassificationMetrics()
        metrics.clear()

        # Record potential misclassification: knowledge question routed to SEARCH
        metrics.record_classification(
            query="what is a pressure transmitter?",
            intent="search",
            confidence=0.80,
            target_workflow="instrument_identifier",
            classification_time_ms=200.0,
            layer_used="llm"
        )

        # Record potential misclassification: purchase routed to CHAT
        metrics.record_classification(
            query="i need a pressure transmitter",
            intent="chat",
            confidence=0.75,
            target_workflow="engenie_chat",
            classification_time_ms=150.0,
            layer_used="semantic"
        )

        # Record potential misclassification: design routed to CHAT
        metrics.record_classification(
            query="design a measurement system",
            intent="chat",
            confidence=0.70,
            target_workflow="engenie_chat",
            classification_time_ms=180.0,
            layer_used="semantic"
        )

        candidates = metrics.get_misclassification_candidates()

        assert len(candidates) >= 2  # At least 2 misclassifications detected

        print(f"[OK] Detected {len(candidates)} misclassification candidates")
        for c in candidates:
            print(f"     {c['issue']}: {c['query'][:40]}")

    def test_summary_statistics(self):
        """Test comprehensive summary generation."""
        metrics = ClassificationMetrics()
        metrics.clear()

        # Record diverse classifications
        for i in range(10):
            metrics.record_classification(
                query=f"query_{i}",
                intent="chat",
                confidence=0.85 + (i * 0.01),
                target_workflow="engenie_chat",
                classification_time_ms=200.0 + (i * 10),
                layer_used="semantic"
            )

        stats = metrics.get_summary_statistics()

        assert stats["total_classifications"] == 10
        assert "by_layer" in stats
        assert "by_workflow" in stats
        assert "avg_confidence" in stats
        assert "avg_time_ms" in stats

        print(f"[OK] Summary statistics: {stats['total_classifications']} classifications")
        print(f"     Average confidence: {stats['avg_confidence']:.2f}")
        print(f"     Average time: {stats['avg_time_ms']:.1f}ms")


def run_all_tests():
    """Run all tests manually (without pytest)."""
    print("\n" + "="*70)
    print("PATH 2 FEATURES TEST SUITE")
    print("="*70)

    print("\n--- Testing DomainValidator ---")
    test_domain = TestDomainValidator()
    test_domain.test_in_scope_queries()
    test_domain.test_out_of_scope_queries()
    test_domain.test_workflow_capability_validation()
    test_domain.test_reject_messages()

    print("\n--- Testing Intent-Specific Confidence ---")
    test_conf = TestIntentSpecificConfidence()
    test_conf.test_confidence_thresholds()
    test_conf.test_should_accept_classification()
    test_conf.test_needs_disambiguation()

    print("\n--- Testing ClassificationMetrics ---")
    test_metrics = TestClassificationMetrics()
    test_metrics.test_metrics_recording()
    test_metrics.test_accuracy_by_layer()
    test_metrics.test_low_confidence_samples()
    test_metrics.test_misclassification_detection()
    test_metrics.test_summary_statistics()

    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()
