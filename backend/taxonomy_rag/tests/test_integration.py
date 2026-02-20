import sys
import os
import pytest
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
# Add backend to sys.path so 'common' imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from backend.taxonomy_rag.normalization_agent import TaxonomyNormalizationAgent
from backend.taxonomy_rag.integration import TaxonomyIntegrationAdapter

class TestIntegration:

    def test_reverse_normalization_logic(self):
        """
        Manually test the reverse normalization logic since we can't easily mock
        the RAG memory in this test environment without complex setup.
        """
        # Mock memory with a known taxonomy
        mock_memory = MagicMock()
        mock_memory.get_taxonomy.return_value = {
            "instruments": [
                {"name": "Pressure Transmitter", "aliases": ["Diff Press", "PT"]},
                {"name": "Flow Meter",  "aliases": ["FM", "FlowIndicator"]}
            ],
            "accessories": []
        }
        
        agent = TaxonomyNormalizationAgent(memory=mock_memory)
        
        # Test input: Canonical "Pressure Transmitter"
        items = [
            {"canonical_name": "Pressure Transmitter", "name": "PT-01"},
            {"canonical_name": "Flow Meter", "name": "FM-01"},
            {"canonical_name": "Unknown Item", "name": "Custom"}
        ]
        
        # ACT
        result = agent.reverse_normalize(items)
        
        # ASSERT
        # 1. PT -> "PT" (shortest alias)
        assert result[0]["reverse_name"] == "PT"
        assert result[0]["reverse_match_source"] == "reverse_map"
        
        # 2. FM -> "FM"
        assert result[1]["reverse_name"] == "FM"
        
        # 3. Unknown -> "Unknown Item" (passthrough)
        assert result[2]["reverse_name"] == "Unknown Item"
        assert result[2]["reverse_match_source"] == "passthrough"

    def test_search_payload_preparation(self):
        """Test the adapter's ability to format data for product search."""
        
        # Input: Normalized items from Solution Agent
        normalized_items = [
            {
                "canonical_name": "Pressure Transmitter",
                "name": "Line 1 PT",
                "category": "Instrumentation",
                "quantity": 2,
                "specifications": {"Range": "0-10 bar"},
                "sample_input": "I need 2 PTs 0-10 bar",
                "taxonomy_matched": True
            },
            {
                # Fallback item (no canonical name)
                "name": "Custom Analyzer",
                "quantity": 1,
                "category": "Analysis",
                "specifications": {},
                "taxonomy_matched": False
            }
        ]
        
        # ACT
        payload = TaxonomyIntegrationAdapter.prepare_search_payload(
            normalized_items, 
            solution_name="Test Solution"
        )
        
        products = payload["required_products"]
        
        # ASSERT
        assert len(products) == 2
        
        # Check matched item
        p1 = products[0]
        assert p1["product_type"] == "Pressure Transmitter"
        assert p1["sample_input"] == "I need 2 PTs 0-10 bar"
        assert p1["taxonomy_matched"] is True
        assert p1["application"] == "Test Solution - Instrumentation"
        
        # Check fallback item
        p2 = products[1]
        assert p2["product_type"] == "Custom Analyzer" # Falls back to name
        assert p2["taxonomy_matched"] is False

