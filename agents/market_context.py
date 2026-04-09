"""
Market Context Agent

This agent runs early in the workflow to fetch and cache all relevant market data
(news, earnings, analyst ratings, SEC filings, 10-K content) once, making it
available to all downstream agents via the shared state.

Claude autonomously decides which tools to call based on the user's request and
portfolio tickers, eliminating redundant API calls and ensuring all agents work
with consistent, up-to-date market context.

Uses Model Context Protocol (MCP) with official MCP Python SDK for proper integration.
"""

import os
import json
from anthropic import Anthropic
import asyncio
from dotenv import load_dotenv
from graph.state import WealthManagerState
import logging
import httpx

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
    Fetch and cache market context data using official MCP Python SDK.
    
    Connects to MCP server via stdio and uses Claude with MCP tools autonomously.
    
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
    
    # Run the async MCP operation
    try:
        market_context_result = asyncio.run(_fetch_market_context_with_mcp(state, api_key))
        return market_context_result
    except Exception as e:
        logger.error(f"Error running market context with MCP: {e}")
        return {
            "market_context": {"error": str(e)},
            "audit_iteration_count": state.get("audit_iteration_count", 0)
        }


async def _fetch_market_context_with_mcp(state: WealthManagerState, api_key: str) -> dict:
    """
    Async helper to fetch market context using HTTP calls to MCP server.
    
    Connects to MCP server via HTTP, fetches tools, and runs Claude agentic loop.
    """
    tickers = state.get("tickers", [])
    messages = state.get("messages", ["Analyze portfolio"])
    user_request = messages[0] if isinstance(messages, list) else str(messages)
    
    logger.info(f"Market Context Agent (HTTP): Gathering market data for {tickers}")
    
    # Connect to MCP server via HTTP
    mcp_server_url = "http://localhost:3000"
    
    try:
        logger.info(f"Connecting to MCP server at {mcp_server_url}...")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Check server health
            logger.info("Step 1: Checking server health...")
            health_resp = await client.get(f"{mcp_server_url}/health")
            if health_resp.status_code != 200:
                raise Exception(f"Server health check failed: {health_resp.status_code}")
            logger.info("  ✓ Server is healthy")
            
            # List available tools
            logger.info("Step 2: Fetching available tools...")
            tools_resp = await client.get(f"{mcp_server_url}/tools")
            tools_resp.raise_for_status()
            tools_data = tools_resp.json()
            
            # Extract tools from response
            if isinstance(tools_data, dict) and "tools" in tools_data:
                tools = tools_data["tools"]
            else:
                tools = tools_data if isinstance(tools_data, list) else []
            
            logger.info(f"  ✓ Found {len(tools)} tools: {', '.join([t['name'] for t in tools])}")
            
            logger.info("Step 3: Starting Claude agentic loop...")
            # Run Claude agentic loop
            market_context = await _run_claude_with_mcp_http(
                client, mcp_server_url, api_key, tickers, user_request, tools
            )
            
            logger.info("Step 4: Claude loop completed")
            market_context["_mcp_enabled"] = True
            market_context["_mcp_transport"] = "HTTP"
            return {
                "market_context": market_context,
                "audit_iteration_count": state.get("audit_iteration_count", 0)
            }
        
    except Exception as e:
        logger.error(f"Failed to connect to MCP server at {mcp_server_url}: {e}", exc_info=True)
        logger.info("Make sure to start the MCP HTTP server in a separate terminal:")
        logger.info("  python mcp_http_server.py")
        raise


async def _run_claude_with_mcp_http(
    http_client: httpx.AsyncClient,
    mcp_url: str,
    api_key: str,
    tickers: list,
    user_request: str,
    tools: list
) -> dict:
    """
    Run Claude agentic loop with HTTP-based MCP tool calls.
    
    Claude autonomously decides which tools to call via HTTP.
    """
    client = Anthropic(api_key=api_key)
    
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
    market_context = {}
    tools_used = []
    max_iterations = 3
    iteration = 0
    total_input_tokens = 0
    total_output_tokens = 0
    
    while iteration < max_iterations:
        iteration += 1
        logger.info(f"  → Claude iteration {iteration}/{max_iterations}")
        
        try:
            # Call Claude with timeout
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                tools=tools,
                messages=messages_list,
                timeout=30.0
            )
            
            logger.info(f"    → Response: stop_reason={response.stop_reason}")
            
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            logger.info(f"    → Tokens: {input_tokens}in + {output_tokens}out")
            
            if response.stop_reason == "end_turn":
                logger.info("    → Claude finished")
                for block in response.content:
                    if hasattr(block, 'text'):
                        market_context["summary"] = block.text
                        logger.info(f"    → Summary ({len(block.text)} chars)")
                logger.info(f"  ✓ Complete in {iteration} iteration(s)")
                if tools_used:
                    logger.info(f"  → Tools called: {', '.join(tools_used)}")
                break
            
            if response.stop_reason == "tool_use":
                logger.info(f"    → Processing tool calls...")
                messages_list.append({"role": "assistant", "content": response.content})
                tool_results = []
                
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        
                        logger.info(f"    → Calling {tool_name}...")
                        
                        if tool_name not in tools_used:
                            tools_used.append(tool_name)
                        
                        # Call tool via HTTP with timeout
                        try:
                            result_resp = await asyncio.wait_for(
                                http_client.post(
                                    f"{mcp_url}/call",
                                    json={"name": tool_name, "arguments": tool_input},
                                    timeout=20.0
                                ),
                                timeout=25.0
                            )
                            result_resp.raise_for_status()
                            result_data = result_resp.json()
                            
                            # Extract result from HTTP response
                            if not result_data.get("success"):
                                error_msg = result_data.get("error", "Unknown error")
                                logger.error(f"      ✗ {tool_name} failed: {error_msg}")
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": json.dumps({"error": error_msg})
                                })
                                continue
                            
                            # Get the result dict
                            result_dict = result_data.get("result", {})
                            logger.info(f"      ✓ {tool_name} returned successfully")
                            
                            if tool_name not in market_context:
                                market_context[tool_name] = result_dict
                            
                            # Summarize for Claude
                            summarized = _summarize_tool_result(tool_name, result_dict)
                            
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(summarized) if isinstance(summarized, dict) else str(summarized)
                            })
                        except asyncio.TimeoutError:
                            logger.error(f"      ✗ {tool_name} timed out")
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps({"error": f"Tool timed out"})
                            })
                        except Exception as e:
                            logger.error(f"      ✗ {tool_name} failed: {e}")
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps({"error": str(e)})
                            })
                
                logger.info(f"    → Sending {len(tool_results)} results back to Claude")
                messages_list.append({"role": "user", "content": tool_results})
            else:
                logger.warning(f"    → Unexpected stop reason: {response.stop_reason}")
                break
                
        except Exception as e:
            logger.error(f"  ✗ Claude call failed at iteration {iteration}: {e}", exc_info=True)
            raise
    
    market_context["_token_usage"] = {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens
    }
    market_context["_tools_used"] = tools_used
    
    if tools_used:
        logger.info(f"✓ Tools successfully called: {', '.join(tools_used)}")
    logger.info(f"✓ MCP HTTP mode enabled")
    
    return market_context
