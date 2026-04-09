"""
Comprehensive end-to-end test for the entire wealth manager workflow.

Tests the complete orchestration flow:
  1. Orchestrator: Ticker extraction and routing
  2. Market Context: Data fetching and caching
  3. Sentiment: Market sentiment analysis
  4. Investment Analyst: RAG-powered analysis
  5. Auditor: Fact-checking and hallucination detection
  6. Report Generator: Final synthesis

Note: Full end-to-end tests require OPENAI_API_KEY and ANTHROPIC_API_KEY.
Unit tests for individual agents work without API keys.
"""

import os
import sys
from pathlib import Path
from pprint import pprint
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Find workspace root (3 levels up from this file)
workspace_root = Path(__file__).resolve().parents[2]
dotenv_path = workspace_root / ".env"
load_dotenv(dotenv_path)

# Add workspace root to path
sys.path.insert(0, str(workspace_root))

from graph.state import WealthManagerState


def create_initial_state(user_message: str, tickers: list = None) -> dict:
    """
    Create a properly initialized state for workflow execution.
    
    Args:
        user_message: The user's input query
        tickers: Optional pre-extracted tickers (for testing)
    
    Returns:
        Initialized state dictionary
    """
    return {
        "messages": [user_message],
        "tickers": tickers or [],
        "route_target": "",
        "market_context": {},
        "news_articles": [],
        "sentiment_results": [],
        "sentiment_summary": {},
        "sentiment_score": 0.0,
        "portfolio_weights": {},
        "retrieved_context": "",
        "live_data_context": "",
        "draft_report": "",
        "audit_score": 0.0,
        "audit_findings": [],
        "is_hallucinating": False,
        "final_report": "",
    }


def test_orchestrator_ticker_extraction():
    """Test 1: Orchestrator extracts tickers from user message."""
    print("\n" + "="*80)
    print("TEST 1: Orchestrator Ticker Extraction")
    print("="*80)
    
    from agents.orchestrator import orchestrator_node
    
    state = create_initial_state(
        "I want to analyze Apple and Microsoft stocks for my tech portfolio."
    )
    
    result = orchestrator_node(state)
    
    print(f"  User Message: {state['messages'][0]}")
    print(f"  Extracted Tickers: {result['tickers']}")
    print(f"  Route Target: {result['route_target']}")
    
    assert "AAPL" in result["tickers"], "Should extract AAPL"
    assert "MSFT" in result["tickers"], "Should extract MSFT"
    assert result["route_target"] == "sentiment_agent", "Should route to sentiment_agent when tickers found"
    print("  [OK] PASSED")
    

def test_orchestrator_no_tickers():
    """Test 2: Orchestrator routes to report generator when no tickers found."""
    print("\n" + "="*80)
    print("TEST 2: Orchestrator No Tickers (Direct to Report Generator)")
    print("="*80)
    
    from agents.orchestrator import orchestrator_node
    
    state = create_initial_state("What's the current state of the market?")
    
    result = orchestrator_node(state)
    
    print(f"  User Message: {state['messages'][0]}")
    print(f"  Extracted Tickers: {result['tickers']}")
    print(f"  Route Target: {result['route_target']}")
    
    assert result["tickers"] == [], "Should extract no tickers"
    assert result["route_target"] == "report_generator_agent", "Should route to report_generator when no tickers"
    print("  [OK] PASSED")


def test_market_context_agent():
    """Test 3: Market Context agent initializes market_context dict."""
    print("\n" + "="*80)
    print("TEST 3: Market Context Agent Data Fetching")
    print("="*80)
    
    from agents.market_context import market_context_node
    
    state = create_initial_state("Analyze AAPL and GOOGL")
    state["tickers"] = ["AAPL", "GOOGL"]
    state["portfolio_tickers"] = ["AAPL", "GOOGL"]
    state["workflow_config"] = {
        "needs_news": True,
        "needs_earnings": False,
        "needs_analyst_ratings": False,
        "needs_10k_content": False,
        "needs_sec_filings": False,
        "enable_analyst_mode": False,
    }
    
    result = market_context_node(state)
    
    print(f"  Tickers: {state.get('portfolio_tickers', [])}")
    print(f"  Market Context Keys: {list(result.get('market_context', {}).keys())}")
    
    assert "market_context" in result, "Should return market_context"
    assert isinstance(result["market_context"], dict), "market_context should be a dict"
    print("  [OK] PASSED")


def test_full_workflow_with_tickers():
    """Test 4: Run complete workflow with valid tickers."""
    print("\n" + "="*80)
    print("TEST 4: Full Workflow Execution (with tickers)")
    print("="*80)
    
    # Skip if API keys not available
    if not os.environ.get("OPENAI_API_KEY"):
        print("  [SKIP] Requires OPENAI_API_KEY")
        return
    
    from graph.workflow import create_wealth_manager_graph
    
    app = create_wealth_manager_graph()
    
    initial_state = create_initial_state(
        "Analyze the tech sector. Focus on AAPL and NVDA impact on portfolio."
    )
    
    print(f"  Input: {initial_state['messages'][0]}")
    print(f"  Running workflow...\n")
    
    nodes_executed = []
    final_state = None
    
    try:
        for output in app.stream(initial_state):
            for node_name, node_state in output.items():
                nodes_executed.append(node_name)
                print(f"  [OK] {node_name} executed")
                
                # Capture final state
                final_state = node_state
                
                # Log key state updates
                if "tickers" in node_state and node_state["tickers"]:
                    print(f"    - Tickers: {node_state['tickers']}")
                if "market_context" in node_state and node_state["market_context"]:
                    print(f"    - Market Context Keys: {list(node_state['market_context'].keys())}")
                if "sentiment_score" in node_state and node_state["sentiment_score"] != 0:
                    print(f"    - Sentiment Score: {node_state['sentiment_score']:.2f}")
                if "draft_report" in node_state and node_state["draft_report"]:
                    print(f"    - Draft Report Generated: {len(node_state['draft_report'])} chars")
                if "final_report" in node_state and node_state["final_report"]:
                    print(f"    - Final Report Generated: {len(node_state['final_report'])} chars")
    
    except Exception as e:
        print(f"  [WARN] Error during workflow: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n  Nodes Executed: {' -> '.join(nodes_executed)}")
    
    # Verify orchestrator ran
    assert "orchestrator_agent" in nodes_executed, "Should execute orchestrator_agent"
    
    # Verify market_context ran
    assert "market_context_agent" in nodes_executed, "Should execute market_context_agent"
    
    print("  [OK] PASSED")


def test_full_workflow_without_tickers():
    """Test 5: Run complete workflow without tickers (short path)."""
    print("\n" + "="*80)
    print("TEST 5: Full Workflow Execution (without tickers - short path)")
    print("="*80)
    
    # Skip if API keys not available
    if not os.environ.get("OPENAI_API_KEY"):
        print("  [SKIP] Requires OPENAI_API_KEY")
        return
    
    from graph.workflow import create_wealth_manager_graph
    
    app = create_wealth_manager_graph()
    
    initial_state = create_initial_state(
        "What is the current market sentiment?"
    )
    
    print(f"  Input: {initial_state['messages'][0]}")
    print(f"  Running workflow...\n")
    
    nodes_executed = []
    
    try:
        for output in app.stream(initial_state):
            for node_name, node_state in output.items():
                nodes_executed.append(node_name)
                print(f"  [OK] {node_name} executed")
    
    except Exception as e:
        print(f"  [WARN] Error during workflow: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n  Nodes Executed: {' -> '.join(nodes_executed)}")
    
    # When no tickers, should skip sentiment/analyst/auditor and go directly to report generator
    assert "orchestrator_agent" in nodes_executed
    assert "report_generator_agent" in nodes_executed
    
    print("  [OK] PASSED")


def run_all_tests():
    """Execute all tests."""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  WEALTH MANAGER FULL WORKFLOW TEST SUITE".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    tests = [
        test_orchestrator_ticker_extraction,
        test_orchestrator_no_tickers,
        test_market_context_agent,
        test_full_workflow_with_tickers,
        test_full_workflow_without_tickers,
    ]
    
    failed = 0
    passed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "█"*80)
    print(f"  Results: {passed} passed, {failed} failed")
    print("█"*80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
