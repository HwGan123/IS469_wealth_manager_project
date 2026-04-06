"""
MCP News Module

Provides Model Context Protocol (MCP) tools for autonomous financial data fetching.
Allows Claude to autonomously fetch news, earnings, analyst ratings, and SEC filings.

Main Components:
- tools.py: MCP tool schema definitions
- implementations.py: Tool function implementations
- dispatcher.py: Tool execution dispatcher
"""

from mcp_news.tools import get_mcp_tools
from mcp_news.dispatcher import dispatch_mcp_tool
from mcp_news.implementations import (
    fetch_news,
    fetch_earnings,
    fetch_analyst_ratings,
    fetch_sec_filings
)

__all__ = [
    "get_mcp_tools",
    "dispatch_mcp_tool",
    "fetch_news",
    "fetch_earnings",
    "fetch_analyst_ratings",
    "fetch_sec_filings",
]
