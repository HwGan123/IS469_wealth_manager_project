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
            "name": "fetch_10k_content",
            "description": "Fetch SEC 10-K with actual document content. Returns key sections (MD&A, Risk Factors, Financial Summary) to provide meaningful context for investment analysis without overwhelming with full documents.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock tickers (e.g., ['AAPL', 'NVDA'])"
                    },
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["md_and_a", "risk_factors", "business_overview", "financial_summary"]
                        },
                        "description": "Which sections to extract from 10-K (default: all)",
                        "default": ["md_and_a", "risk_factors", "financial_summary"]
                    }
                },
                "required": ["tickers"]
            }
        },
        {
            "name": "fetch_10q_content",
            "description": "Fetch SEC 10-Q (quarterly report) with actual document content. Returns key sections (MD&A, Risk Factors, Financial Summary) for quarterly monitoring and responsive analysis without overwhelming with full documents.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock tickers (e.g., ['AAPL', 'NVDA'])"
                    },
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["md_and_a", "risk_factors", "business_overview", "financial_summary"]
                        },
                        "description": "Which sections to extract from 10-Q (default: all)",
                        "default": ["md_and_a", "risk_factors", "financial_summary"]
                    }
                },
                "required": ["tickers"]
            }
        },
        {
            "name": "fetch_xbrl_financials",
            "description": "Fetch structured financial metrics directly from SEC XBRL filings (machine-readable format). Returns key financial ratios and metrics (Revenue, Net Income, EPS, ROE, Debt-to-Equity, etc.) in compact JSON format. ~500 tokens per company (vs 3000+ for document prose). Token-efficient for frequent agent calls.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock tickers (e.g., ['AAPL', 'NVDA'])"
                    },
                    "filing_type": {
                        "type": "string",
                        "enum": ["10-K", "10-Q"],
                        "description": "Filing type: Use '10-K' for annual reports and full-year analysis. Use '10-Q' for quarterly reports and tracking quarterly trends. Defaults to '10-K' if not specified.",
                        "default": "10-K"
                    }
                },
                "required": ["tickers"]
            }
        }
    ]
