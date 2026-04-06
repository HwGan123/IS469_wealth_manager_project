"""
MCP Tool Implementations

Actual implementations of tools available to Claude.
Each function fetches real financial data from APIs.
"""

import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict


# ─────────────────────────────────────────────────────────────────────────
# NEWS FETCHING (via Finnhub)
# ─────────────────────────────────────────────────────────────────────────

def fetch_news(tickers: List[str], days_back: int = 7) -> Dict:
    """
    Fetch company news for tickers using Finnhub API.
    
    Finnhub provides high-quality financial news aggregated from multiple sources:
    - Company announcements
    - Earnings news
    - Merger/acquisition news
    - Regulatory filings
    - Analyst reports
    
    Args:
        tickers: List of stock tickers (e.g., ["AAPL", "NVDA"])
        days_back: Number of days to look back (default: 7, Finnhub limit: 90)
    
    Returns:
        Dict with 'articles' list and 'count', or error dict
    """
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        return {
            "error": "FINNHUB_API_KEY not configured in environment",
            "solution": "Add to .env: FINNHUB_API_KEY=your_key_from_finnhub.io"
        }
    
    articles = []
    from_date = (datetime.now() - timedelta(days=days_back)).date().isoformat()
    to_date = datetime.now().date().isoformat()
    
    for ticker in tickers:
        try:
            # Finnhub company news endpoint
            resp = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={
                    "symbol": ticker,
                    "from": from_date,
                    "to": to_date,
                    "token": api_key,
                },
                timeout=10,
            )
            resp.raise_for_status()
            
            data = resp.json()
            
            # Handle API errors
            if isinstance(data, dict) and "error" in data:
                articles.append({
                    "ticker": ticker,
                    "error": data.get("error", "Unknown error"),
                    "title": f"[Error fetching {ticker} news]"
                })
                continue
            
            # Parse articles
            for article in data if isinstance(data, list) else []:
                articles.append({
                    "ticker": ticker,
                    "title": article.get("headline", ""),
                    "description": article.get("summary", ""),
                    "content": article.get("summary", ""),  # Finnhub doesn't provide full content
                    "url": article.get("url", ""),
                    "published": article.get("datetime", ""),
                    "source": article.get("source", ""),
                    "category": article.get("category", ""),
                    "sentiment": article.get("sentiment"),  # Finnhub provides sentiment scores
                    "image": article.get("image", ""),
                })
        except requests.RequestException as e:
            articles.append({
                "ticker": ticker,
                "error": str(e),
                "title": f"[Error fetching {ticker} news]"
            })
    
    return {
        "articles": articles,
        "count": len(articles),
        "source": "Finnhub",
        "days_back": days_back,
        "from_date": from_date,
        "to_date": to_date
    }


# ─────────────────────────────────────────────────────────────────────────
# EARNINGS FETCHING
# ─────────────────────────────────────────────────────────────────────────

def fetch_earnings(tickers: List[str]) -> Dict:
    """
    Fetch earnings data for tickers using yfinance.
    
    Args:
        tickers: List of stock tickers
    
    Returns:
        Dict with 'earnings' list and 'count', or error dict
    """
    try:
        import yfinance as yf
    except ImportError:
        return {"error": "yfinance not installed. Run: pip install yfinance"}
    
    earnings_data = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            
            earnings_data.append({
                "ticker": ticker,
                "pe_ratio": info.get("trailingPE"),
                "trailing_eps": info.get("trailingEps"),
                "forward_eps": info.get("forwardEps"),
                "earnings_date": info.get("earningsDate"),
                "profit_margin": info.get("profitMargins"),
                "revenue_ttm": info.get("totalRevenue"),
                "market_cap": info.get("marketCap"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
            })
        except Exception as e:
            earnings_data.append({
                "ticker": ticker,
                "error": str(e)
            })
    
    return {
        "earnings": earnings_data,
        "count": len(earnings_data),
        "source": "yfinance"
    }


# ─────────────────────────────────────────────────────────────────────────
# ANALYST RATINGS FETCHING
# ─────────────────────────────────────────────────────────────────────────

def fetch_analyst_ratings(tickers: List[str]) -> Dict:
    """
    Fetch analyst ratings and sentiment for tickers using yfinance.
    
    Args:
        tickers: List of stock tickers
    
    Returns:
        Dict with 'ratings' list and 'count', or error dict
    """
    try:
        import yfinance as yf
    except ImportError:
        return {"error": "yfinance not installed. Run: pip install yfinance"}
    
    ratings = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            
            ratings.append({
                "ticker": ticker,
                "recommendation": info.get("recommendationKey", "N/A"),
                "target_price": info.get("targetMeanPrice"),
                "number_of_analysts": info.get("numberOfAnalystRatings"),
                "buy_count": info.get("recommendationCount", {}).get("buy"),
                "hold_count": info.get("recommendationCount", {}).get("hold"),
                "sell_count": info.get("recommendationCount", {}).get("sell"),
                "current_price": info.get("currentPrice"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            })
        except Exception as e:
            ratings.append({
                "ticker": ticker,
                "error": str(e)
            })
    
    return {
        "ratings": ratings,
        "count": len(ratings),
        "source": "yfinance"
    }


# ─────────────────────────────────────────────────────────────────────────
# SEC FILINGS FETCHING
# ─────────────────────────────────────────────────────────────────────────

def fetch_sec_filings(tickers: List[str], filing_type: str = "10-K") -> Dict:
    """
    Fetch SEC filings metadata for tickers.
    
    NOTE: This is a stub implementation. For production use, integrate
    with SEC Edgar API (https://www.sec.gov/cgi-bin/browse-edgar)
    
    Args:
        tickers: List of stock tickers
        filing_type: Type of filing ("10-K", "10-Q", "8-K")
    
    Returns:
        Dict with 'filings' list, noting this is stub data
    """
    filings = []
    
    for ticker in tickers:
        filings.append({
            "ticker": ticker,
            "filing_type": filing_type,
            "status": "stub_implementation",
            "note": "SEC Edgar integration needed for production",
            "next_steps": [
                "Integrate SEC Edgar API",
                "Cache filing documents",
                "Extract key sections (MD&A, Risk Factors, etc.)"
            ]
        })
    
    return {
        "filings": filings,
        "count": len(filings),
        "source": "SEC Edgar (stub)",
        "note": "This is a placeholder. Implement SEC Edgar API for full functionality."
    }
