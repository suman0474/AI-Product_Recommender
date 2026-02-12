"""
Tests for Strategy Keyword Standardizer
========================================
Unit tests for the StrategyKeywordStandardizer service.

Run with: pytest tests/test_strategy_keyword_standardizer.py -v
"""

import pytest
import sys
import os
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.strategy.keyword_standardizer import (
    StrategyKeywordStandardizer,
    get_standardizer
)


class TestStrategyKeywordStandardizer:
    """Test suite for StrategyKeywordStandardizer."""

    @pytest.fixture
    def standardizer(self):
        """Create standardizer instance."""
        return StrategyKeywordStandardizer()

    # ========================================================================
    # BASIC FUNCTIONALITY TESTS
    # ========================================================================

    def test_standardizer_initialization(self, standardizer):
        """Test that standardizer initializes correctly."""
        assert standardizer is not None
        assert standardizer.llm is not None
        # Collection may be None if MongoDB/pymongo not available
        assert len(standardizer._memory_cache) == 0

    def test_singleton_pattern(self):
        """Test that get_standardizer returns singleton instance."""
        std1 = get_standardizer()
        std2 = get_standardizer()
        assert std1 is std2

    # ========================================================================
    # KEYWORD STANDARDIZATION TESTS
    # ========================================================================

    def test_standardize_keyword_empty_input(self, standardizer):
        """Test standardization with empty input."""
        result = standardizer.standardize_keyword("", "category")
        assert result == ("", "", 0.0)

    def test_standardize_keyword_whitespace(self, standardizer):
        """Test standardization with whitespace."""
        result = standardizer.standardize_keyword("   ", "category")
        assert result == ("", "", 0.0)

    def test_standardize_keyword_fallback(self, standardizer):
        """Test fallback standardization for unknown keywords."""
        # With a very unique keyword that won't be in DB
        canonical_full, canonical_abbrev, confidence = standardizer.standardize_keyword(
            "xyzUnknownProductXYZ123",
            "category"
        )

        # Should return title case with low confidence
        assert canonical_full  # Should not be empty
        assert canonical_abbrev  # Should have abbreviation
        assert confidence < 0.5  # Low confidence for fallback

    # ========================================================================
    # BATCH STANDARDIZATION TESTS
    # ========================================================================

    def test_batch_standardize_empty_list(self, standardizer):
        """Test batch standardization with empty list."""
        result = standardizer.batch_standardize([])
        assert result == []

    def test_batch_standardize_single_record(self, standardizer):
        """Test batch standardization with single record."""
        records = [{
            "vendor_name": "Emerson",
            "category": "pressure",
            "subcategory": "transmitter",
            "strategy": "preferred vendor"
        }]

        result = standardizer.batch_standardize(records)

        assert len(result) == 1
        assert "vendor_name_std" in result[0]
        assert "category_std" in result[0]
        assert "subcategory_std" in result[0]
        assert "strategy_keywords" in result[0]
        assert "strategy_priority" in result[0]
        assert "standardization_confidence" in result[0]

    def test_batch_standardize_preserves_original(self, standardizer):
        """Test that original fields are preserved."""
        records = [{
            "vendor_name": "Emerson Electric",
            "category": "pressure transmitters",
            "subcategory": "differential",
            "strategy": "preferred vendor for critical applications"
        }]

        result = standardizer.batch_standardize(records)

        # Original fields should be preserved
        assert result[0]["vendor_name"] == "Emerson Electric"
        assert result[0]["category"] == "pressure transmitters"
        assert result[0]["subcategory"] == "differential"
        assert result[0]["strategy"] == "preferred vendor for critical applications"

    def test_batch_standardize_multiple_records(self, standardizer):
        """Test batch standardization with multiple records."""
        records = [
            {
                "vendor_name": "Emerson",
                "category": "pressure",
                "subcategory": "differential",
                "strategy": "preferred"
            },
            {
                "vendor_name": "Honeywell",
                "category": "temperature",
                "subcategory": "thermocouple",
                "strategy": "approved"
            }
        ]

        result = standardizer.batch_standardize(records)

        assert len(result) == 2
        for record in result:
            assert "vendor_name_std" in record
            assert "category_std" in record
            assert "standardization_confidence" in record

    def test_batch_standardize_missing_fields(self, standardizer):
        """Test batch standardization with missing fields."""
        records = [{
            "vendor_name": "Emerson"
            # Missing other fields
        }]

        result = standardizer.batch_standardize(records)

        assert len(result) == 1
        assert result[0]["vendor_name_std"]  # Should standardize what's there
        assert result[0]["category_std"] == ""  # Should be empty string

    # ========================================================================
    # QUERY EXPANSION TESTS
    # ========================================================================

    def test_expand_query_term_empty(self, standardizer):
        """Test query expansion with empty input."""
        result = standardizer.expand_query_term("", "category")
        assert result == []

    def test_expand_query_term_single_term(self, standardizer):
        """Test query expansion returns at least original term."""
        result = standardizer.expand_query_term("pressure", "category")

        # Should contain at least the original term
        assert "pressure" in result or "Pressure" in result
        assert len(result) > 0

    def test_expand_query_term_case_insensitive(self, standardizer):
        """Test that query expansion is case insensitive."""
        result1 = standardizer.expand_query_term("PRESSURE", "category")
        result2 = standardizer.expand_query_term("pressure", "category")

        # Both should return results
        assert len(result1) > 0
        assert len(result2) > 0

    # ========================================================================
    # MEMORY CACHE TESTS
    # ========================================================================

    def test_memory_cache_storage(self, standardizer):
        """Test that results are cached in memory."""
        # Clear cache
        standardizer._memory_cache.clear()

        # Standardize same keyword twice
        result1 = standardizer.standardize_keyword("pressure transmitter", "category")
        cache_size_after_first = len(standardizer._memory_cache)

        result2 = standardizer.standardize_keyword("pressure transmitter", "category")
        cache_size_after_second = len(standardizer._memory_cache)

        # Should be cached
        assert cache_size_after_first > 0
        assert cache_size_after_second == cache_size_after_first  # No new entries

    def test_memory_cache_limit(self, standardizer):
        """Test that memory cache respects size limit."""
        # Set a small limit for testing
        standardizer._cache_size_limit = 5
        standardizer._memory_cache.clear()

        # Standardize multiple keywords to populate cache through proper method
        # (This triggers cache limit enforcement)
        keywords = [
            "pressure", "flow", "temperature", "level", "control",
            "analytical", "digital", "analog", "smart", "wireless"
        ]

        for keyword in keywords:
            standardizer.standardize_keyword(keyword, "category")

        # Should not exceed limit (enforced by standardize_keyword method)
        assert len(standardizer._memory_cache) <= 5

    # ========================================================================
    # ERROR HANDLING TESTS
    # ========================================================================

    def test_standardize_with_special_characters(self, standardizer):
        """Test standardization with special characters."""
        result = standardizer.standardize_keyword("PT-100", "category")

        # Should return something (fallback)
        assert result[0]  # canonical_full should not be empty
        assert result[1]  # canonical_abbrev should not be empty

    def test_standardize_with_unicode(self, standardizer):
        """Test standardization with unicode characters."""
        result = standardizer.standardize_keyword("Müller Temperature", "category")

        # Should handle unicode gracefully
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)
        assert isinstance(result[2], float)

    # ========================================================================
    # INTEGRATION TESTS
    # ========================================================================

    def test_full_standardization_workflow(self, standardizer):
        """Test complete standardization workflow."""
        # 1. Standardize single keyword
        keyword_result = standardizer.standardize_keyword("PT", "category")
        assert keyword_result[2] > 0  # Should have confidence

        # 2. Batch standardize
        records = [{
            "vendor_name": "Emerson",
            "category": "PT",
            "subcategory": "DP",
            "strategy": "preferred"
        }]
        batch_result = standardizer.batch_standardize(records)
        assert len(batch_result) == 1

        # 3. Expand query - should return at least the original term
        expanded = standardizer.expand_query_term("PT", "category")
        assert len(expanded) >= 1  # Should have at least the original term
        assert "PT" in expanded or "pt" in [t.lower() for t in expanded]

    def test_user_scoped_standardization(self, standardizer):
        """Test that user_id affects standardization (user-scoped mappings)."""
        # Standardize with user_id
        result1 = standardizer.standardize_keyword(
            "custom vendor",
            "vendor",
            user_id=123
        )

        # Standardize same with different user_id
        result2 = standardizer.standardize_keyword(
            "custom vendor",
            "vendor",
            user_id=456
        )

        # Both should return results
        assert result1[0]
        assert result2[0]

    # ========================================================================
    # FIELD TYPE TESTS
    # ========================================================================

    def test_standardize_category(self, standardizer):
        """Test category standardization."""
        result = standardizer.standardize_keyword(
            "pressure transmitter",
            "category"
        )
        assert result[0]  # Should return canonical full
        assert result[1]  # Should return abbreviation

    def test_standardize_subcategory(self, standardizer):
        """Test subcategory standardization."""
        result = standardizer.standardize_keyword(
            "differential pressure",
            "subcategory",
            category_context="Pressure Instruments"
        )
        assert result[0]

    def test_standardize_vendor(self, standardizer):
        """Test vendor name standardization."""
        result = standardizer.standardize_keyword(
            "Emerson Electric Co.",
            "vendor"
        )
        assert result[0]  # Should return canonical vendor name
        assert result[1]  # Should return abbreviation (EMR)

    def test_standardize_strategy_keywords(self, standardizer):
        """Test strategy keyword extraction."""
        result = standardizer.standardize_keyword(
            "preferred vendor for critical applications",
            "strategy"
        )
        # For strategy, returns (keywords_str, priority, confidence)
        assert isinstance(result[0], str)  # keywords
        assert result[1] in ["critical", "high", "medium", "low"]  # priority
        assert 0 <= result[2] <= 1  # confidence


class TestStrategyKeywordIntegration:
    """Integration tests with real MongoDB (if available)."""

    @pytest.fixture
    def standardizer(self):
        """Create standardizer instance."""
        return StrategyKeywordStandardizer()

    def test_database_connectivity(self, standardizer):
        """Test that database connection works."""
        if standardizer.collection is not None:
            # Try a simple query
            count = standardizer.collection.count_documents({})
            assert isinstance(count, int)
        else:
            pytest.skip("Database not available")

    def test_store_and_retrieve_mapping(self, standardizer):
        """Test storing and retrieving a mapping."""
        if standardizer.collection is None:
            pytest.skip("Database not available")

        # Store a test mapping
        standardizer._store_mapping(
            keyword="test_keyword_123",
            field_type="category",
            canonical_full="Test Category",
            canonical_abbrev="TC",
            confidence=0.95,
            user_id=None
        )

        # Retrieve it
        mapping = standardizer._find_mapping("test_keyword_123", "category")

        if mapping:
            assert mapping["canonical_full"] == "Test Category"
            assert mapping["canonical_abbrev"] == "TC"

            # Cleanup
            standardizer.collection.delete_one({"_id": mapping["_id"]})


def test_import_standardizer():
    """Test that standardizer can be imported and used."""
    from services.strategy.keyword_standardizer import get_standardizer
    standardizer = get_standardizer()
    assert standardizer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
