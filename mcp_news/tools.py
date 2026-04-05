"""
MCP Tool Definitions

Defines the schema for tools available to Claude via MCP protocol.
These tools enable Claude to autonomously fetch live financial data.
"""


def get_mcp_tools():
    """
    Define MCP tools that Claude can autonomously call.
    
    Returns:
        List of tool dictionaries with name, description, and input_schema.
    """
    return [
        {
            "name": "fetch_news",
            "description": "Fetch recent financial news for specific stock tickers using Finnhub API. Returns headlines with sentiment scores, categories, and publication details.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock tickers (e.g., ['AAPL', 'NVDA'])"
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days to look back for news (default: 7, max: 90)",
                        "default": 7
                    }
                },
                "required": ["tickers"]
            }
        },
        {
            "name": "fetch_earnings",
            "description": "Fetch current earnings data, PE ratios, EPS, forward guidance, and profit margins for tickers.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock tickers (e.g., ['AAPL', 'NVDA'])"
                    }
                },
                "required": ["tickers"]
            }
        },
        {
            "name": "fetch_analyst_ratings",
            "description": "Fetch current analyst ratings, price targets, recommendation consensus, and sentiment for tickers.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock tickers (e.g., ['AAPL', 'NVDA'])"
                    }
                },
                "required": ["tickers"]
            }
        },
        {
            "name": "fetch_sec_filings",
            "description": "Fetch recent SEC filings (10-K, 10-Q, 8-K) for companies from SEC Edgar database.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock tickers"
                    },
                    "filing_type": {
                        "type": "string",
                        "description": "Type of filing: 10-K (annual), 10-Q (quarterly), 8-K (event)",
                        "enum": ["10-K", "10-Q", "8-K"],
                        "default": "10-K"
                    }
                },
                "required": ["tickers"]
            }
        }
    ]
