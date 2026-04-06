"""
MCP Server - Wealth Manager Financial Data Tools

Exposes MCP tools for autonomous financial data fetching.
This server implements the Model Context Protocol (MCP) standard,
allowing Claude and other MCP clients to call financial data tools.

Usage:
    python mcp_server.py

The server communicates via stdio and should be registered in:
    - Claude Desktop: ~/.config/Claude/claude_desktop_config.json
    - or equivalent MCP client configuration

Architecture:
    Server (this file)
        ↓
    Tool Definitions (tools.py)
        ↓
    Tool Implementations (implementations.py)
        ↓
    External APIs (Finnhub, yfinance, SEC Edgar)
"""

import asyncio
import json
import sys
from typing import Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try to import MCP, provide helpful error if not installed
try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    from mcp import stdio_server
except ImportError as e:
    print("ERROR: mcp package not installed", file=sys.stderr)
    print(f"Import error details: {e}", file=sys.stderr)
    print("Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

from mcp_news.tools import get_mcp_tools
from mcp_news.dispatcher import dispatch_mcp_tool


# Initialize MCP Server
server = Server("wealth-manager-financial-data")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    Return list of available MCP tools.
    Claude will see these tools and can invoke them.
    """
    tools = get_mcp_tools()
    
    # Convert to MCP Tool format
    mcp_tools = []
    for tool in tools:
        mcp_tools.append(
            Tool(
                name=tool["name"],
                description=tool["description"],
                inputSchema=tool["input_schema"],
            )
        )
    
    return mcp_tools


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    Execute a tool invoked by Claude.
    Routes tool calls to appropriate implementations via dispatcher.
    """
    try:
        # Dispatch to implementation
        result = dispatch_mcp_tool(name, arguments)
        
        # Format result as text for Claude
        result_text = json.dumps(result, indent=2, default=str)
        
        return [TextContent(type="text", text=result_text)]
    
    except Exception as e:
        error_msg = f"Tool execution error: {str(e)}"
        return [TextContent(type="text", text=error_msg)]


async def main():
    """
    Start the MCP server.
    
    The server listens on stdin/stdout and waits for tool calls from Claude.
    This is the main entry point when registered with Claude or other MCP clients.
    """
    print("Starting Wealth Manager MCP Server...", file=sys.stderr)
    print("Available tools:", file=sys.stderr)
    
    # List available tools
    tools = await list_tools()
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}", file=sys.stderr)
    
    print("Ready to receive tool calls from Claude", file=sys.stderr)
    
    # Run server with stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
