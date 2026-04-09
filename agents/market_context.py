"""
Market Context Agent

This agent runs early in the workflow to fetch and cache all relevant market data
(news, earnings, analyst ratings, SEC filings, 10-K content) once, making it
available to all downstream agents via the shared state.

Claude autonomously decides which tools to call based on the user's request and
portfolio tickers, eliminating redundant API calls and ensuring all agents work
with consistent, up-to-date market context.

Uses Model Context Protocol (MCP) with Anthropic SDK for proper tool integration.
"""

import os
import json
import subprocess
import time
import asyncio
import anthropic
from dotenv import load_dotenv
from graph.state import WealthManagerState
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Global MCP server process
_mcp_server_process = None


def _start_mcp_server():
    """
    Start the MCP server as a subprocess.
    
    Returns:
        subprocess.Popen: The server process
    """
    global _mcp_server_process
    
    if _mcp_server_process and _mcp_server_process.poll() is None:
        logger.info("MCP server already running")
        return _mcp_server_process
    
    logger.info("Starting MCP server...")
    try:
        _mcp_server_process = subprocess.Popen(
            ["python", "mcp_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        time.sleep(2)  # Give server time to start
        logger.info("✓ MCP server started (PID: {})".format(_mcp_server_process.pid))
        return _mcp_server_process
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        raise


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
    Fetch and cache market context data for the portfolio companies using MCP.
    
    Uses the Model Context Protocol with Anthropic SDK:
    - Starts MCP server as subprocess
    - Connects via MCPClient (Option 1: proper MCP integration)
    - Fetches tools from MCP server
    - Claude autonomously decides which tools to call
    
    Args:
        state: Current workflow state containing portfolio tickers, user messages, config
        
    Returns:
        Updated state with market_context field populated
    """
    print("--- AGENT: MARKET CONTEXT (MCP) ---")
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
    
    # Start MCP server
    try:
        _start_mcp_server()
    except Exception as e:
        logger.warning(f"Could not start MCP server: {e}. Falling back to direct mode.")
    
    logger.info(f"Market Context Agent (MCP): Gathering market data for {tickers}")
    
    client = anthropic.Anthropic(api_key=api_key)
    messages = state.get("messages", ["Analyze portfolio"])
    user_request = messages[0] if isinstance(messages, list) else str(messages)
    
    # Check if anthropic SDK has MCPClient (Option 1: Recommended)
    try:
        from anthropic.mcp import MCPClient
        logger.info("✓ Using Anthropic SDK's MCPClient (proper MCP integration)")
        
        # Create MCP client pointing to our server
        # The MCP server is running on stdio, so we use it directly
        mcp_client = MCPClient(
            server="stdio",
            command=["python", "mcp_server.py"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        # Fetch tools from MCP server
        tools = []
        try:
            tools_from_server = asyncio.run(mcp_client.get_tools())
            tools = tools_from_server if tools_from_server else []
            logger.info(f"  Fetched {len(tools)} tools from MCP server")
        except Exception as e:
            logger.warning(f"Could not fetch tools from MCP server: {e}")
            tools = []
        
        use_mcp_client = True
    except (ImportError, Exception) as e:
        logger.info(f"⚠ MCPClient not available: {e}. Using direct dispatch mode")
        # Fall back to direct mode (old behavior)
        from mcp_news import get_mcp_tools, dispatch_mcp_tool
        tools = get_mcp_tools()
        use_mcp_client = False
        mcp_client = None
    
    # Build prompt for Claude
    prompt = f"""You are a comprehensive financial research agent. Your task is to gather detailed 
market context and financial data for investment analysis.

USER REQUEST: {user_request}
PORTFOLIO TICKERS: {', '.join(tickers)}

Use the available tools strategically:
1. SEC 10-K Annual Reports - For comprehensive financial statements and risk factors
2. SEC 10-Q Quarterly Reports - For quarterly performance tracking
3. XBRL Financial Metrics - For structured financial data
4. Recent news - For market sentiment and catalysts
5. Earnings data - For valuation metrics
6. Analyst ratings - For professional perspectives

Decide which tools are most relevant. Fetch data efficiently. Synthesize into a clear summary."""
    
    messages_list = [{"role": "user", "content": prompt}]
    
    # Agentic loop
    market_context = {}
    tools_used = []
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
        
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        logger.info(f"    Tokens: {input_tokens} input, {output_tokens} output")
        
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, 'text'):
                    market_context["summary"] = block.text
            logger.info(f"  ✓ Market data gathering complete ({iteration} iterations)")
            if tools_used:
                logger.info(f"  → Tools used: {', '.join(tools_used)}")
            break
        
        if response.stop_reason == "tool_use":
            messages_list.append({"role": "assistant", "content": response.content})
            tool_results = []
            
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    
                    logger.info(f"    → Calling {tool_name}")
                    
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)
                    
                    # Execute tool
                    if use_mcp_client:
                        try:
                            result = asyncio.run(mcp_client.call_tool(tool_name, tool_input))
                        except Exception as e:
                            logger.error(f"MCP tool execution failed: {e}")
                            result = {"error": str(e)}
                    else:
                        result = dispatch_mcp_tool(tool_name, tool_input)
                    
                    if tool_name not in market_context:
                        market_context[tool_name] = result
                    
                    summarized_result = _summarize_tool_result(tool_name, result)
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(summarized_result)
                    })
            
            messages_list.append({"role": "user", "content": tool_results})
        else:
            logger.warning(f"Unexpected stop reason: {response.stop_reason}")
            break
    
    market_context["_token_usage"] = {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens
    }
    market_context["_tools_used"] = tools_used
    market_context["_mcp_enabled"] = use_mcp_client
    
    if tools_used:
        print(f"  → Tools used: {', '.join(tools_used)}")
    print(f"  → MCP mode: {'Enabled' if use_mcp_client else 'Fallback (direct dispatch)'}")
    
    return {
        "market_context": market_context,
        "audit_iteration_count": state.get("audit_iteration_count", 0)
    }
