"""
Market Context Agent

This agent runs early in the workflow to fetch and cache all relevant market data
(news, earnings, analyst ratings, SEC filings, 10-K content) once, making it
available to all downstream agents via the shared state.

Claude autonomously decides which tools to call based on the user's request and
portfolio tickers, eliminating redundant API calls and ensuring all agents work
with consistent, up-to-date market context.
"""

import os
import json
import anthropic
from dotenv import load_dotenv
from graph.state import WealthManagerState
from mcp_news import get_mcp_tools, dispatch_mcp_tool
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def _summarize_tool_result(tool_name: str, result: dict) -> dict:
    """
    Summarize tool results to reduce context window bloat in agentic loops.
    
    Keeps full data in market_context but sends compressed summaries back to Claude
    to prevent hitting token limits during multi-turn conversations.
    
    Args:
        tool_name: Name of the tool that was called
        result: Full result from the tool
        
    Returns:
        Summarized version of the result
    """
    if "error" in result:
        return result  # Keep errors as-is
    
    # Summarize by tool type
    if tool_name == "fetch_news":
        articles = result.get("articles", [])
        if len(articles) > 10:
            # Keep only top 10 articles
            articles = articles[:10]
            return {
                "articles": articles,
                "count": len(articles),
                "note": f"Showing top 10 of {result.get('count', len(articles))} articles"
            }
        return result
    

    elif tool_name == "fetch_10k_content":
        # Compress 10-K content - keep only summaries
        if isinstance(result, dict):
            compressed = {}
            for key, value in result.items():
                if isinstance(value, dict) and "summary" in value:
                    # Use summary instead of full text
                    compressed[key] = {"summary": value["summary"][:500]}
                elif isinstance(value, str):
                    # Truncate strings
                    compressed[key] = value[:300]
                else:
                    compressed[key] = value
            return {
                "content": compressed,
                "note": "10-K content compressed. Full data cached."
            }
        return result
    
    elif tool_name == "fetch_earnings":
        # Earnings data is usually structured - keep as-is but cap size
        if isinstance(result, dict):
            keys = list(result.keys())[:20]  # Cap to 20 tickers
            return {k: result[k] for k in keys} if len(result) > 20 else result
        return result
    
    elif tool_name == "fetch_analyst_ratings":
        # Keep analyst ratings - usually lean
        return result
    
    # Default: return as-is
    return result


def market_context_node(state: WealthManagerState) -> dict:
    """
    Fetch and cache market context data for the portfolio companies.
    
    Claude autonomously decides which MCP tools to call based on:
    - User's initial request (state["messages"])
    - Portfolio tickers (state["portfolio_tickers"])
    - Workflow configuration
    
    Runs an agentic loop until Claude completes the data gathering task.
    All results are cached in state["market_context"] for downstream agents.
    
    Args:
        state: Current workflow state containing portfolio tickers, user messages, config
        
    Returns:
        Updated state with market_context field populated
    """
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
    
    logger.info(f"Market Context Agent: Gathering market data for {tickers}")
    
    client = anthropic.Anthropic(api_key=api_key)
    messages = state.get("messages", ["Analyze portfolio"])
    user_request = messages[0] if isinstance(messages, list) else str(messages)
    
    # Build prompt for Claude to intelligently fetch data
    prompt = f"""You are a market research agent. Your task is to gather comprehensive 
market context for investment analysis.

USER REQUEST: {user_request}
PORTFOLIO TICKERS: {', '.join(tickers)}

Use the available tools to fetch relevant data for these tickers:
1. Recent news and market developments (fetch_news)
2. Current earnings, PE ratios, EPS data (fetch_earnings)
3. Analyst ratings and sentiment (fetch_analyst_ratings)
4. Detailed 10-K content if needed for long-term analysis (fetch_10k_content)

Decide which tools are most relevant based on the user's request.
Fetch data efficiently - avoid redundant calls.
Synthesize the gathered data into a clear summary."""
    
    messages_list = [{"role": "user", "content": prompt}]
    tools = get_mcp_tools()
    
    # Agentic loop: Claude calls tools until done
    market_context = {}
    max_iterations = 3
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
                    
                    # Execute tool
                    result = dispatch_mcp_tool(tool_name, tool_input)
                    
                    # Cache full result
                    if tool_name not in market_context:
                        market_context[tool_name] = result
                    
                    # Summarize result for conversation (to avoid context bloat)
                    summarized_result = _summarize_tool_result(tool_name, result)
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(summarized_result)
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
    
    return {
        "market_context": market_context,
        "audit_iteration_count": state.get("audit_iteration_count", 0)
    }
