
import sys
import unittest
import os
import time

# Add backend directory to path
backend_path = os.path.join(os.getcwd(), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Import IntentClassificationRoutingAgent 
# We need to be careful with imports.
from agentic.agents.routing.intent_classifier import IntentClassificationRoutingAgent, WorkflowTarget

class TestConversationalIntents(unittest.TestCase):
    def setUp(self):
        self.agent = IntentClassificationRoutingAgent()
        
    def test_greeting_fast_path(self):
        """Test fast path for greetings."""
        query = "Hello"
        print(f"\nTesting Greeting: '{query}'")
        start = time.time()
        result = self.agent.classify(query)
        duration = (time.time() - start) * 1000
        
        print(f"Time: {duration:.2f}ms")
        print(f"Target: {result.target_workflow}")
        print(f"Direct Response: {result.direct_response}")
        
        self.assertEqual(result.target_workflow, WorkflowTarget.GREETING)
        self.assertEqual(result.intent, "greeting")
        self.assertIsNotNone(result.direct_response)
        self.assertTrue(result.extracted_info.get("fast_pattern_match"))

    def test_farewell_fast_path(self):
        """Test fast path for farewells."""
        query = "Goodbye"
        print(f"\nTesting Farewell: '{query}'")
        result = self.agent.classify(query)
        
        print(f"Target: {result.target_workflow}")
        print(f"Direct Response: {result.direct_response}")
        
        self.assertEqual(result.target_workflow, WorkflowTarget.CONVERSATIONAL)
        self.assertEqual(result.intent, "farewell")
        self.assertIsNotNone(result.direct_response)

    def test_gratitude_fast_path(self):
        """Test fast path for gratitude."""
        query = "Thank you so much"
        print(f"\nTesting Gratitude: '{query}'")
        result = self.agent.classify(query)
        
        print(f"Target: {result.target_workflow}")
        print(f"Direct Response: {result.direct_response}")
        
        self.assertEqual(result.target_workflow, WorkflowTarget.CONVERSATIONAL)
        self.assertEqual(result.intent, "gratitude")
        self.assertIsNotNone(result.direct_response)

    def test_help_fast_path(self):
        """Test fast path for help."""
        query = "What can you do?"
        print(f"\nTesting Help: '{query}'")
        result = self.agent.classify(query)
        
        print(f"Target: {result.target_workflow}")
        print(f"Direct Response: {result.direct_response}")
        
        self.assertEqual(result.target_workflow, WorkflowTarget.CONVERSATIONAL)
        self.assertEqual(result.intent, "help")
        self.assertIsNotNone(result.direct_response)
        
    def test_chitchat_fast_path(self):
        """Test fast path for chitchat."""
        query = "Who made you?"
        print(f"\nTesting Chitchat: '{query}'")
        result = self.agent.classify(query)
        
        print(f"Target: {result.target_workflow}")
        print(f"Direct Response: {result.direct_response}")
        
        self.assertEqual(result.target_workflow, WorkflowTarget.CONVERSATIONAL)
        self.assertEqual(result.intent, "chitchat")
        self.assertIsNotNone(result.direct_response)
        
    def test_gibberish_fast_path(self):
        """Test fast path for gibberish."""
        query = "asdfghjkl"
        print(f"\nTesting Gibberish: '{query}'")
        result = self.agent.classify(query)
        
        print(f"Target: {result.target_workflow}")
        print(f"Direct Response: {result.direct_response}")
        
        # Depending on implementation, gibberish might map to CONVERSATIONAL (chitchat/other)
        # or OUT_OF_DOMAIN. In my plan I mapped it to CONVERSATIONAL/gibberish for direct response.
        self.assertTrue(result.direct_response is not None)
        self.assertEqual(result.intent, "gibberish")

if __name__ == '__main__':
    unittest.main()
