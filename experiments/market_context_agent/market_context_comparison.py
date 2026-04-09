"""
Simple Market Context Agent Comparison

Runs the workflow twice with the same shared prompt:
1. WITH market context agent (forces all tools)
2. WITHOUT market context agent

Compares token count, latency, information freshness, and relevance.
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from datetime import datetime, date
from dotenv import load_dotenv

workspace_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(workspace_root))
load_dotenv(workspace_root / ".env")

from graph.state import WealthManagerState
from graph.workflow import create_wealth_manager_graph
from langgraph.graph import StateGraph, START, END
from agents.orchestrator import orchestrator_node
from agents.sentiment import sentiment_node
from agents.analyst import analyst_node
from agents.auditor import auditor_node
from agents.report_generator import report_generator_node
from mcp_news import get_mcp_tools, dispatch_mcp_tool
import anthropic
import logging

logger = logging.getLogger(__name__)

# Pricing for Claude 3.5 Haiku (per 1M tokens)
PRICING = {
    "input_cost_per_1m": 1.00,      # $1.00 per 1M input tokens
    "output_cost_per_1m": 5.00,     # $5.00 per 1M output tokens
}


def create_workflow_without_market_context():
    """Workflow without market context agent."""
    workflow = StateGraph(WealthManagerState)
    
    workflow.add_node("orchestrator_agent", orchestrator_node)
    workflow.add_node("sentiment_agent", sentiment_node)
    workflow.add_node("investment_analyst_agent", analyst_node)
    workflow.add_node("auditor_agent", auditor_node)
    workflow.add_node("report_generator_agent", report_generator_node)
    
    # Skip market context - go straight to sentiment
    workflow.add_edge(START, "orchestrator_agent")
    workflow.add_edge("orchestrator_agent", "sentiment_agent")
    workflow.add_edge("sentiment_agent", "investment_analyst_agent")
    workflow.add_edge("investment_analyst_agent", "auditor_agent")
    
    def audit_route(state):
        if state.get("is_hallucinating", False) and state.get("audit_iteration_count", 0) < 1:
            return "investment_analyst_agent"
        return "report_generator_agent"
    
    workflow.add_conditional_edges("auditor_agent", audit_route)
    workflow.add_edge("report_generator_agent", END)
    
    return workflow.compile()


def create_initial_state(user_message: str, tickers: list) -> dict:
    """Create initial workflow state."""
    return {
        "messages": [user_message],
        "tickers": tickers,
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
        "hallucination_count": 0,
        "verified_count": 0,
        "unsubstantiated_count": 0,
        "ragas_metrics": {},
        "ground_truth": "",
        "audit_iteration_count": 0,
        "final_report": "",
    }


def check_freshness(report: str, state: dict = None) -> str:
    """
    Check information freshness based on auditor agent's verification metrics.
    Fresh data sources (like fetch_news, fetch_earnings) allow the auditor to verify more information.
    Stale data sources (like historical 10-Ks) result in more unsubstantiated claims.
    """
    if not state:
        return "UNKNOWN (no state)"
    
    # Get auditor metrics from state
    verified_count = state.get("verified_count", 0)
    unsubstantiated_count = state.get("unsubstantiated_count", 0)
    
    total_claims = verified_count + unsubstantiated_count
    
    if total_claims == 0:
        return "UNKNOWN (no claims evaluated)"
    
    # Calculate verification rate
    verification_rate = verified_count / total_claims if total_claims > 0 else 0
    
    # Same thresholds for both scenarios
    if verification_rate >= 0.4:
        return f"YES (verified {verified_count}/{total_claims} claims)"
    elif verification_rate >= 0.2:
        return f"PARTIAL (verified {verified_count}/{total_claims} claims)"
    else:
        return f"NO (verified {verified_count}/{total_claims} claims)"


def check_relevance(report: str, tickers: list) -> str:
    """Simple relevance check."""
    report_lower = report.lower()
    mentioned = [t for t in tickers if t.lower() in report_lower]
    
    if len(mentioned) == len(tickers):
        return f"YES (all {len(tickers)})"
    elif mentioned:
        return f"PARTIAL ({len(mentioned)}/{len(tickers)})"
    else:
        return "NO"


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD from token counts."""
    input_cost = (input_tokens / 1_000_000) * PRICING["input_cost_per_1m"]
    output_cost = (output_tokens / 1_000_000) * PRICING["output_cost_per_1m"]
    return input_cost + output_cost


def market_context_node_all_tools(state: WealthManagerState) -> dict:
    """
    Market context node that FORCES Claude to call all tools.
    This is a TEST-ONLY version - use only for comparison experiments.
    """
    print("--- AGENT: MARKET CONTEXT (TEST - FORCING ALL TOOLS) ---")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    tickers = state.get("tickers", [])
    
    if not tickers:
        logger.warning("No portfolio tickers found in state")
        return {
            "market_context": {},
            "audit_iteration_count": state.get("audit_iteration_count", 0)
        }
    
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set. Cannot run market context agent.")
        return {
            "market_context": {
                "error": "ANTHROPIC_API_KEY required for market context agent"
            },
            "audit_iteration_count": state.get("audit_iteration_count", 0)
        }
    
    logger.info(f"Market Context Agent (TEST): Gathering ALL market data for {tickers}")
    
    client = anthropic.Anthropic(api_key=api_key)
    messages = state.get("messages", ["Analyze portfolio"])
    user_request = messages[0] if isinstance(messages, list) else str(messages)
    
    # FORCED prompt for test scenario - Claude MUST call all tools
    prompt = f"""You are a comprehensive financial research agent. Your task is to gather detailed 
market context and financial data for investment analysis.

USER REQUEST: {user_request}
PORTFOLIO TICKERS: {', '.join(tickers)}

**CRITICAL TEST REQUIREMENT**: You MUST call ALL 6 available tools in any order to ensure comprehensive data gathering:

1. fetch_10k_content - Get SEC 10-K Annual Reports
2. fetch_10q_content - Get SEC 10-Q Quarterly Reports  
3. fetch_xbrl_financials - Get XBRL Financial Metrics
4. fetch_earnings - Get current earnings and valuation data
5. fetch_news - Get recent news and market developments
6. fetch_analyst_ratings - Get analyst ratings and sentiment

MANDATORY INSTRUCTIONS FOR THIS TEST:
- Call each tool with the tickers: {', '.join(tickers)}
- Do NOT skip any tools - call all 6
- Do NOT ask whether you should call tools - simply call them
- Do NOT consolidate or combine tool calls - make separate calls for each tool
- Continue until all 6 tools have been called at least once

After calling all 6 tools, synthesize the comprehensive data into a summary."""
    
    messages_list = [{"role": "user", "content": prompt}]
    tools = get_mcp_tools()
    
    # Agentic loop: Claude calls tools until done
    market_context = {}
    tools_used = []
    max_iterations = 15  # Increased for test to ensure all 6 tools are called
    iteration = 0
    total_input_tokens = 0
    total_output_tokens = 0
    
    while iteration < max_iterations:
        iteration += 1
        logger.info(f"  Iteration {iteration}: Calling Claude...")
        
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            tools=tools,
            messages=messages_list
        )
        
        # Track token usage
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        logger.info(f"    Tokens: {input_tokens} input, {output_tokens} output")
        
        # Check if Claude is done
        if response.stop_reason == "end_turn":
            # Extract final summary
            for block in response.content:
                if hasattr(block, 'text'):
                    market_context["summary"] = block.text
            logger.info(f"  ✓ Market data gathering complete ({iteration} iterations)")
            logger.info(f"  Total tokens used: {total_input_tokens} input, {total_output_tokens} output")
            if tools_used:
                logger.info(f"  → Tools used: {', '.join(tools_used)}")
            break
        
        # Process tool calls
        if response.stop_reason == "tool_use":
            # Add assistant response to conversation
            messages_list.append({"role": "assistant", "content": response.content})
            
            # Process each tool call
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    
                    logger.info(f"    → Calling {tool_name}")
                    
                    # Track tool usage
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)
                    
                    # Execute tool
                    result = dispatch_mcp_tool(tool_name, tool_input)
                    
                    # Cache full result
                    if tool_name not in market_context:
                        market_context[tool_name] = result
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result if isinstance(result, (dict, list)) else str(result))
                    })
            
            # Add tool results to conversation
            messages_list.append({"role": "user", "content": tool_results})
        else:
            logger.warning(f"Unexpected stop reason: {response.stop_reason}")
            break
    
    # Add token usage to results
    market_context["_token_usage"] = {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens
    }
    
    # Add tools used to results
    market_context["_tools_used"] = tools_used
    
    # Print tools used
    if tools_used:
        print(f"  → Tools used: {', '.join(tools_used)}")
    
    return {
        "market_context": market_context,
        "audit_iteration_count": state.get("audit_iteration_count", 0)
    }


def create_workflow_with_all_tools(tickers: list):
    """Create workflow that uses market context with forced all-tools testing."""
    from graph.workflow import create_wealth_manager_graph
    
    # Get the standard graph
    workflow_builder = create_wealth_manager_graph()
    
    # Replace the market context node with our test version that forces all tools
    # The workflow is already compiled, so we need to rebuild it with custom node
    workflow = StateGraph(WealthManagerState)
    
    workflow.add_node("market_context_agent", market_context_node_all_tools)
    workflow.add_node("orchestrator_agent", orchestrator_node)
    workflow.add_node("sentiment_agent", sentiment_node)
    workflow.add_node("investment_analyst_agent", analyst_node)
    workflow.add_node("auditor_agent", auditor_node)
    workflow.add_node("report_generator_agent", report_generator_node)
    
    # Add edges including market context at the start
    workflow.add_edge(START, "market_context_agent")
    workflow.add_edge("market_context_agent", "orchestrator_agent")
    workflow.add_edge("orchestrator_agent", "sentiment_agent")
    workflow.add_edge("sentiment_agent", "investment_analyst_agent")
    workflow.add_edge("investment_analyst_agent", "auditor_agent")
    
    def audit_route(state):
        if state.get("is_hallucinating", False) and state.get("audit_iteration_count", 0) < 1:
            return "investment_analyst_agent"
        return "report_generator_agent"
    
    workflow.add_conditional_edges("auditor_agent", audit_route)
    workflow.add_edge("report_generator_agent", END)
    
    return workflow.compile()


def run_comparison(user_prompt: str, tickers: list, output_dir: Path):
    """Run the comparison experiment."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "prompt": user_prompt,
        "tickers": tickers,
        "note": "WITH market context = forces all 6 tools (test scenario)",
    }
    
    print(f"\n{'='*80}")
    print(f"PROMPT: {user_prompt}")
    print(f"TICKERS: {tickers}")
    print(f"{'='*80}\n")
    
    # ─────────────────────────────────────────────────────────────────────
    # WITH Market Context (FORCING ALL TOOLS)
    # ─────────────────────────────────────────────────────────────────────
    print("Running WITH market context agent (forcing all tools)...")
    workflow_with = create_workflow_with_all_tools(tickers)
    initial_state = create_initial_state(user_prompt, tickers)
    
    start_time = time.time()
    final_state_with = initial_state.copy()
    for output in workflow_with.stream(initial_state):
        for node_name, state in output.items():
            print(f"  ✓ {node_name}")
            final_state_with.update(state)
    
    latency_with = time.time() - start_time
    
    tokens_with = 0
    input_tokens_with = 0
    output_tokens_with = 0
    cost_with = 0.0
    tools_used_with = []
    if "market_context" in final_state_with and isinstance(final_state_with["market_context"], dict):
        market_ctx = final_state_with["market_context"]
        if "_token_usage" in market_ctx:
            tokens_with = market_ctx["_token_usage"].get("total_tokens", 0)
            input_tokens_with = market_ctx["_token_usage"].get("input_tokens", 0)
            output_tokens_with = market_ctx["_token_usage"].get("output_tokens", 0)
            cost_with = calculate_cost(input_tokens_with, output_tokens_with)
        if "_tools_used" in market_ctx:
            tools_used_with = market_ctx["_tools_used"]
    
    print(f"  Latency: {latency_with:.2f}s | Tokens: {tokens_with} | Cost: ${cost_with:.4f}")
    if tools_used_with:
        print(f"  Tools Called: {', '.join(tools_used_with)}\n")
    else:
        print()
    
    final_report_with = final_state_with.get("final_report", "")
    results["with_market_context"] = {
        "latency_seconds": latency_with,
        "tokens": tokens_with,
        "input_tokens": input_tokens_with,
        "output_tokens": output_tokens_with,
        "cost_usd": cost_with,
        "tools_called": tools_used_with,
        "final_report": final_report_with,
        "audit_score": final_state_with.get("audit_score", 0),
        "verified_count": final_state_with.get("verified_count", 0),
        "unsubstantiated_count": final_state_with.get("unsubstantiated_count", 0),
        "freshness": check_freshness(final_report_with, final_state_with),
        "relevance": check_relevance(final_report_with, tickers),
    }
    
    # ─────────────────────────────────────────────────────────────────────
    # WITHOUT Market Context
    # ─────────────────────────────────────────────────────────────────────
    print("Running WITHOUT market context agent...")
    workflow_without = create_workflow_without_market_context()
    initial_state = create_initial_state(user_prompt, tickers)
    
    start_time = time.time()
    final_state_without = initial_state.copy()
    for output in workflow_without.stream(initial_state):
        for node_name, state in output.items():
            print(f"  ✓ {node_name}")
            final_state_without.update(state)
    
    latency_without = time.time() - start_time
    tokens_without = 0
    cost_without = 0.0
    
    print(f"  Latency: {latency_without:.2f}s | Tokens: {tokens_without} | Cost: ${cost_without:.4f}\n")
    
    final_report_without = final_state_without.get("final_report", "")
    results["without_market_context"] = {
        "latency_seconds": latency_without,
        "tokens": tokens_without,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": cost_without,
        "final_report": final_report_without,
        "audit_score": final_state_without.get("audit_score", 0),
        "verified_count": final_state_without.get("verified_count", 0),
        "unsubstantiated_count": final_state_without.get("unsubstantiated_count", 0),
        "freshness": check_freshness(final_report_without, final_state_without),
        "relevance": check_relevance(final_report_without, tickers),
    }
    
    # ─────────────────────────────────────────────────────────────────────
    # Comparison
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}\n")
    
    latency_delta = latency_with - latency_without
    latency_pct = (latency_delta / latency_without * 100) if latency_without > 0 else 0
    
    tokens_delta = tokens_with - tokens_without
    tokens_pct = (tokens_delta / tokens_without * 100) if tokens_without > 0 else 0
    
    cost_delta = cost_with - cost_without
    
    print(f"Latency:")
    print(f"  WITH:    {latency_with:.2f}s")
    print(f"  WITHOUT: {latency_without:.2f}s")
    print(f"  Delta:   {latency_delta:+.2f}s ({latency_pct:+.1f}%)\n")
    
    print(f"Tokens (from Market Context agent):")
    print(f"  WITH:    {tokens_with}")
    print(f"  WITHOUT: {tokens_without}")
    if tokens_with > 0:
        print(f"  Delta:   {tokens_delta} ({tokens_pct:+.1f}%)")
    print()
    
    print(f"Cost (USD):")
    print(f"  WITH:    ${cost_with:.6f}")
    print(f"  WITHOUT: ${cost_without:.6f}")
    print(f"  Delta:   ${cost_delta:+.6f}\n")
    
    print(f"Information Freshness (based on auditor verification):")
    print(f"  WITH:    {results['with_market_context']['freshness']}")
    print(f"           Verified: {results['with_market_context']['verified_count']} | Unsubstantiated: {results['with_market_context']['unsubstantiated_count']}")
    print(f"  WITHOUT: {results['without_market_context']['freshness']}")
    print(f"           Verified: {results['without_market_context']['verified_count']} | Unsubstantiated: {results['without_market_context']['unsubstantiated_count']}\n")
    
    print(f"Contextual Relevance:")
    print(f"  WITH:    {results['with_market_context']['relevance']}")
    print(f"  WITHOUT: {results['without_market_context']['relevance']}\n")
    
    print(f"Tools Called (WITH market context):")
    print(f"  {', '.join(tools_used_with) if tools_used_with else 'None'}\n")
    
    results["comparison"] = {
        "latency_delta_seconds": latency_delta,
        "latency_percent_increase": latency_pct,
        "tokens_delta": tokens_delta,
        "tokens_percent_increase": tokens_pct,
        "cost_delta_usd": cost_delta,
        "total_cost_with_usd": cost_with,
        "tools_called_with_test": tools_used_with,
    }
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    # Default configuration
    tickers = ["AAPL", "NVDA"]
    user_prompt = f"I need comprehensive financial analysis for {', '.join(tickers)}"
    
    # Create results directory
    output_dir = workspace_root / "results" / "market_context_compare" / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Run comparison
    results = run_comparison(user_prompt, tickers, output_dir)
