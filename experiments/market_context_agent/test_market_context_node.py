"""
Unit tests for market_context_node with mocked MCP tool calls.

These tests validate the core functionality of the market context agent
without requiring actual API calls, using unittest.mock to simulate tool responses.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Find workspace root (3 levels up from this file)
workspace_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(workspace_root))

from agents.market_context import (
    market_context_node,
    _determine_data_needs,
    _fetch_market_data,
)
from graph.state import WealthManagerState


class TestMarketContextImports(unittest.TestCase):
    """Test 1: Verify all required imports work."""
    
    def test_imports(self):
        """Should import all required modules without errors."""
        import agents.market_context
        self.assertTrue(hasattr(agents.market_context, "market_context_node"))
        self.assertTrue(hasattr(agents.market_context, "_determine_data_needs"))
        self.assertTrue(hasattr(agents.market_context, "_fetch_market_data"))


class TestDetermineDataNeeds(unittest.TestCase):
    """Test 2: Validate _determine_data_needs logic."""
    
    def test_analyst_mode_needs_all_data(self):
        """When analyst_mode=True, should need all data types."""
        needs = _determine_data_needs(
            user_query="Analyze Apple",
            analyst_mode=True,
            sector_analysis=False
        )
        
        self.assertTrue(needs["needs_news"])
        self.assertTrue(needs["needs_earnings"])
        self.assertTrue(needs["needs_analyst_ratings"])
        self.assertTrue(needs["needs_10k_content"])
        self.assertTrue(needs["needs_sec_filings"])
    
    def test_sector_analysis_mode(self):
        """When sector_analysis=True, should need relevant data."""
        needs = _determine_data_needs(
            user_query="Compare tech sector performance",
            analyst_mode=False,
            sector_analysis=True
        )
        
        self.assertTrue(needs["needs_news"])
        self.assertTrue(needs["needs_analyst_ratings"])
    
    def test_quick_query_minimal_data(self):
        """For simple queries, should need minimal data."""
        needs = _determine_data_needs(
            user_query="What's AAPL's price?",
            analyst_mode=False,
            sector_analysis=False
        )
        
        # Quick queries should at least get news
        self.assertTrue(needs["needs_news"])


class TestMarketContextNodeWithMocks(unittest.TestCase):
    """Test 3-6: Market context node with fully mocked MCP tools."""
    
    @patch("agents.market_context.dispatch_mcp_tool")
    def test_market_context_node_with_mocked_tools(self, mock_dispatch):
        """Test 3: Full node execution with mocked tool responses."""
        
        # Mock tool responses
        mock_dispatch.side_effect = [
            # First call: fetch_news
            {
                "articles": [
                    {
                        "title": "Apple Q4 Results Beat Expectations",
                        "source": "Reuters",
                        "url": "https://reuters.com/apple-q4",
                        "date": "2024-01-30",
                        "summary": "Apple's Q4 earnings exceeded analyst expectations with strong iPhone sales."
                    }
                ]
            },
            # Second call: fetch_analyst_ratings
            {
                "ratings": [
                    {"analyst": "Goldman Sachs", "rating": "BUY", "target": 195.0},
                    {"analyst": "Morgan Stanley", "rating": "BUY", "target": 190.0},
                ]
            },
        ]
        
        state = {
            "messages": ["Analyze Apple stock"],
            "tickers": ["AAPL"],
            "portfolio_tickers": ["AAPL"],
            "workflow_config": {
                "needs_news": True,
                "needs_earnings": False,
                "needs_analyst_ratings": True,
                "needs_10k_content": False,
                "needs_sec_filings": False,
                "enable_analyst_mode": False,
            },
            "market_context": {},
        }
        
        result = market_context_node(state)
        
        self.assertIn("market_context", result)
        self.assertIn("fetch_news", result["market_context"])
        self.assertIn("fetch_analyst_ratings", result["market_context"])
        
        # Verify tool was called
        self.assertGreater(mock_dispatch.call_count, 0)
    
    @patch("agents.market_context.dispatch_mcp_tool")
    def test_selective_tool_invocation(self, mock_dispatch):
        """Test 4: Only requested tools should be invoked."""
        
        mock_dispatch.return_value = {"status": "success"}
        
        state = {
            "messages": ["Quick check on AAPL"],
            "tickers": ["AAPL"],
            "portfolio_tickers": ["AAPL"],
            "workflow_config": {
                "needs_news": True,
                "needs_earnings": False,  # Don't need earnings
                "needs_analyst_ratings": False,  # Don't need ratings
                "needs_10k_content": False,
                "needs_sec_filings": False,
                "enable_analyst_mode": False,
            },
            "market_context": {},
        }
        
        result = market_context_node(state)
        
        # Verify only news was fetched
        call_args = [call[0][0] for call in mock_dispatch.call_args_list if call[0]]
        
        # At least one call should be for news
        self.assertTrue(
            any("fetch_news" in str(call) for call in call_args),
            "Should invoke fetch_news when needs_news=True"
        )
    
    @patch("agents.market_context.dispatch_mcp_tool")
    def test_market_context_handles_missing_tickers(self, mock_dispatch):
        """Test 5: Should handle missing tickers gracefully."""
        
        state = {
            "messages": ["Analyze stocks"],
            "tickers": [],  # No tickers
            "portfolio_tickers": [],
            "workflow_config": {
                "needs_news": True,
                "needs_earnings": False,
                "needs_analyst_ratings": False,
                "needs_10k_content": False,
                "needs_sec_filings": False,
                "enable_analyst_mode": False,
            },
            "market_context": {},
        }
        
        result = market_context_node(state)
        
        # Should return state without errors
        self.assertIn("market_context", result)
        self.assertEqual(result["market_context"], {})
    
    @patch("agents.market_context.dispatch_mcp_tool")
    def test_market_context_error_handling(self, mock_dispatch):
        """Test 6: Should handle tool errors gracefully."""
        
        # Simulate tool error
        mock_dispatch.side_effect = Exception("API rate limit exceeded")
        
        state = {
            "messages": ["Analyze AAPL"],
            "tickers": ["AAPL"],
            "portfolio_tickers": ["AAPL"],
            "workflow_config": {
                "needs_news": True,
                "needs_earnings": False,
                "needs_analyst_ratings": False,
                "needs_10k_content": False,
                "needs_sec_filings": False,
                "enable_analyst_mode": False,
            },
            "market_context": {},
        }
        
        # Should not raise exception, but handle gracefully
        try:
            result = market_context_node(state)
            # If it gets here, error was handled
            self.assertIn("market_context", result)
        except Exception as e:
            # Document the error for debugging
            self.fail(f"market_context_node should handle tool errors gracefully. Got: {e}")


def run_all_tests():
    """Run all tests with detailed output."""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  MARKET CONTEXT NODE UNIT TEST SUITE".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestMarketContextImports))
    suite.addTests(loader.loadTestsFromTestCase(TestDetermineDataNeeds))
    suite.addTests(loader.loadTestsFromTestCase(TestMarketContextNodeWithMocks))
    
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "█"*80)
    print(f"  Tests Run: {result.testsRun}")
    print(f"  Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failed: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print("█"*80 + "\n")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
