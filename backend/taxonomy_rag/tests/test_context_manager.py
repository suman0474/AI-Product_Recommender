import pytest
from unittest.mock import MagicMock, patch
import logging

from backend.taxonomy_rag.context_manager import TaxonomyContextManager
from backend.taxonomy_rag.normalization_agent import TaxonomyNormalizationAgent

# Setup logging
logging.basicConfig(level=logging.INFO)

class TestTaxonomyContextManager:
    def test_load_history(self):
        manager = TaxonomyContextManager(max_history=3)
        history = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "4"},
        ]
        manager.load_history(history)
        assert len(manager.history) == 3
        assert manager.history[0]["content"] == "2"

    def test_resolve_contextual_references_instrument(self):
        manager = TaxonomyContextManager()
        # Mock active entities
        manager.active_entities["instruments"] = [
            {"name": "Pressure Transmitter", "canonical_name": "Pressure Transmitter"}
        ]
        
        # Test generic reference resolution
        original = "Change the range of the transmitter to 0-10 bar"
        resolved = manager.resolve_contextual_references(original)
        
        # Should append context or clarify
        assert "Pressure Transmitter" in resolved
        assert "referring to" in resolved

    def test_resolve_no_change(self):
        manager = TaxonomyContextManager()
        original = "I need a new flow meter"
        resolved = manager.resolve_contextual_references(original)
        assert resolved == original

class TestNormalizationWithContext:
    @patch('backend.taxonomy_rag.rag.get_taxonomy_rag')
    def test_normalize_with_context_fallback(self, mock_get_rag):
        # Setup Agent
        agent = TaxonomyNormalizationAgent()
        
        # Mock RAG to return nothing initially to force fallback logic check
        mock_rag_instance = MagicMock()
        mock_rag_instance.retrieve.return_value = []
        mock_get_rag.return_value = mock_rag_instance
        
        # Mock Alias Map (empty)
        with patch.object(agent, '_get_alias_map', return_value={}):
            items = [{"name": "it", "category": "unknown"}]
            user_input = "Make the Pressure Transmitter housing stainless steel"
            history = []
            
            # Run normalization
            # The context manager inside should try to resolve 'it' using the active entity logic.
            # However, active entities are empty in a fresh context manager.
            # But the user input contains "Pressure Transmitter" which might be picked up 
            # if we implemented that robust heuristics. 
            
            # Actually, let's test the plumbing: ensuring `resolve_contextual_references` is called.
            # We can spy on TaxonomyContextManager.
            
            with patch('backend.taxonomy_rag.context_manager.TaxonomyContextManager.resolve_contextual_references') as mock_resolve:
                mock_resolve.return_value = "Pressure Transmitter"
                
                normalized = agent.normalize_with_context(items, user_input, history)
                
                # Check if it tried to normalize "Pressure Transmitter" instead of "it"
                # _resolve_term would be called with "Pressure Transmitter"
                # Since we mocked alias map to empty and RAG to empty, it won't match taxonomy,
                # but the `canonical_name` in result should reflect the resolution attempt if we look closely at how _resolve_term behaves (it returns raw as fallback).
                
                assert normalized[0]["canonical_name"] == "Pressure Transmitter"
                assert normalized[0]["context_resolved_from"] == "it"
