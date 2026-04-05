"""
MCP Tool Dispatcher

Routes tool calls from Claude to appropriate implementations.
Handles execution and error handling.
"""

import json
from typing import Dict, Any
from mcp_news.implementations import (
    fetch_news,
    fetch_earnings,
    fetch_analyst_ratings,
    fetch_sec_filings
)


def dispatch_mcp_tool(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the requested MCP tool and return results.
    
    Claude will inspect these results and decide on next steps.
    
    Args:
        tool_name: Name of the tool to execute
        tool_input: Input parameters for the tool
    
    Returns:
        Dict with tool execution results or error information
    """
    try:
        if tool_name == "fetch_news":
            return fetch_news(**tool_input)
        
        elif tool_name == "fetch_earnings":
            return fetch_earnings(**tool_input)
        
        elif tool_name == "fetch_analyst_ratings":
            return fetch_analyst_ratings(**tool_input)
        
        elif tool_name == "fetch_sec_filings":
            return fetch_sec_filings(**tool_input)
        
        else:
            return {
                "error": f"Unknown tool: {tool_name}",
                "available_tools": [
                    "fetch_news",
                    "fetch_earnings",
                    "fetch_analyst_ratings",
                    "fetch_sec_filings"
                ]
            }
    
    except Exception as e:
        return {
            "error": f"Tool execution failed for {tool_name}",
            "details": str(e),
            "tool_name": tool_name
        }
