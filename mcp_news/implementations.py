"""
MCP Tool Implementations

Actual implementations of tools available to Claude.
Each function fetches real financial data from APIs.
"""

import os
import json as json_lib
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Cache for company ticker data to avoid repeated downloads
_COMPANY_TICKERS_CACHE = None
_CACHE_LOAD_TIME = None


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
    Fetch SEC filing metadata (URLs, dates, accession numbers) for tickers.
    
    Uses the official SEC REST API (data.sec.gov/submissions/) for direct access
    to filing history without downloading large documents.
    No API key needed - SEC data APIs are open access.
    
    Args:
        tickers: List of stock tickers (e.g., ["AAPL", "NVDA"])
        filing_type: Type of filing ("10-K", "10-Q", "8-K")
    
    Returns:
        Dict with 'filings' list containing URLs, dates, and metadata
    """
    filings = []
    
    # Load ticker to CIK mapping from SEC's official source
    ticker_to_cik = _load_sec_company_tickers()
    
    for ticker in tickers:
        try:
            # Get CIK (Central Index Key) from SEC's official mapping
            cik = ticker_to_cik.get(ticker.upper())
            if not cik:
                # Try fetching CIK dynamically if not in mapping
                cik = _fetch_cik_from_sec(ticker)
                if not cik:
                    filings.append({
                        "ticker": ticker,
                        "error": f"Could not find CIK for {ticker}",
                        "status": "not_found"
                    })
                    continue
            
            # Query official SEC REST API (data.sec.gov)
            # This endpoint contains the complete filing history for a company
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            headers = _get_sec_request_headers()
            
            data = _sec_api_request(url, None, headers)
            
            if "error" in data:
                filings.append({
                    "ticker": ticker,
                    "error": data["error"],
                    "status": "api_error"
                })
                continue
            
            # Extract company name and tickers from response
            company_name = data.get("name", ticker)
            
            # Parse filing history from the new columnar data structure
            filing_history = _parse_sec_filings_response(data, filing_type, cik)
            
            if filing_history:
                for filing_data in filing_history[:3]:  # Get latest 3 filings
                    filings.append({
                        "ticker": ticker,
                        "company_name": company_name,
                        "filing_type": filing_type,
                        "filing_date": filing_data.get("filing_date", ""),
                        "report_date": filing_data.get("report_date", ""),
                        "accession_number": filing_data.get("accession_number", ""),
                        "url": filing_data.get("url", ""),
                        "status": "found"
                    })
            else:
                filings.append({
                    "ticker": ticker,
                    "company_name": company_name,
                    "status": "no_filings_found",
                    "filing_type": filing_type
                })
            
            # Be respectful to SEC API - add small delay between requests
            time.sleep(0.5)
        
        except Exception as e:
            filings.append({
                "ticker": ticker,
                "error": str(e),
                "status": "error"
            })
    
    return {
        "filings": filings,
        "count": len(filings),
        "source": "SEC REST API (data.sec.gov/submissions)",
        "filing_type": filing_type,
        "note": "Metadata only - URLs can be used to access full documents"
    }


def _get_sec_request_headers() -> dict:
    """
    Get proper HTTP headers for SEC Edgar API requests following official guidelines.
    
    Returns:
        Dict with User-Agent and Accept-Encoding headers
    """
    return {
        "User-Agent": "Wealth Manager Investment Analysis Tool contact@company.com",
        "Accept-Encoding": "gzip, deflate"
    }


def _load_sec_company_tickers() -> dict:
    """
    Load ticker to CIK mapping from SEC's official company_tickers.json file.
    
    This is the authoritative source for ticker-CIK associations.
    Caches the data to avoid repeated downloads.
    
    Returns:
        Dict mapping ticker symbols (uppercase) to CIK numbers
    """
    global _COMPANY_TICKERS_CACHE, _CACHE_LOAD_TIME
    
    # Return cached data if available and not too old (1 hour)
    if _COMPANY_TICKERS_CACHE is not None:
        cache_age = time.time() - _CACHE_LOAD_TIME
        if cache_age < 3600:  # 1 hour
            return _COMPANY_TICKERS_CACHE
    
    try:
        # Download SEC's official company tickers file
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = _get_sec_request_headers()
        
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        
        data = resp.json()
        
        # Transform SEC data format: {number: {"cik_str": 123, "ticker": "AAPL", ...}}
        # Into: {"AAPL": "0000000123", ...}
        ticker_to_cik = {}
        for entry in data.values():
            ticker = entry.get("ticker", "").upper()
            cik = str(entry.get("cik_str", "")).zfill(10)
            if ticker and cik:
                ticker_to_cik[ticker] = cik
        
        # Cache the result
        _COMPANY_TICKERS_CACHE = ticker_to_cik
        _CACHE_LOAD_TIME = time.time()
        
        logger.info(f"Loaded {len(ticker_to_cik)} company tickers from SEC")
        return ticker_to_cik
    
    except Exception as e:
        logger.warning(f"Failed to load SEC company tickers: {e}. Using fallback mapping.")
        # Return minimal fallback mapping
        return {
            "AAPL": "0000320193",
            "MSFT": "0000789019",
            "GOOG": "0001652044",
            "GOOGL": "0001652044",
            "NVDA": "0001045810",
            "TSLA": "0001318605",
            "AMZN": "0001018724",
            "META": "0001326801",
            "AMD": "0000002488",
        }


def _sec_api_request(url: str, params: dict, headers: dict, max_retries: int = 3) -> dict:
    """
    Make a request to SEC Edgar API with retry logic and exponential backoff.
    
    Args:
        url: SEC Edgar API endpoint
        params: Query parameters
        headers: Request headers
        max_retries: Maximum number of retry attempts
    
    Returns:
        JSON response or error dict
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if resp.status_code in [503, 429]:  # Service Unavailable or Too Many Requests
                if attempt < max_retries - 1:
                    wait_time = 3 ** attempt  # Exponential backoff: 1s, 3s, 9s
                    logger.warning(f"SEC API rate limited (HTTP {resp.status_code}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            return {"error": str(e)}
        except requests.ConnectionError as e:
            if attempt < max_retries - 1:
                wait_time = 3 ** attempt
                logger.warning(f"Connection error. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            return {"error": f"Connection failed: {str(e)}"}
        except Exception as e:
            return {"error": str(e)}
    
    return {"error": "Max retries exceeded - SEC Edgar API may be temporarily unavailable"}


def _parse_sec_filings_response(data: dict, filing_type: str, cik: str) -> List[Dict]:
    """
    Parse the columnar SEC filing data from data.sec.gov/submissions response.
    
    The SEC REST API returns filing history in a compact columnar format where
    accessionNumber, filingDate, form, and reportDate are parallel arrays.
    Elements at the same index correspond to the same filing.
    
    Args:
        data: JSON response from data.sec.gov/submissions/CIK##.json
        filing_type: Type of filing to filter (e.g., "10-K", "10-Q", "8-K")
        cik: Company's CIK number (for constructing URLs)
    
    Returns:
        List of dicts with filing_date, report_date, accession_number, url
    """
    filings = []
    
    try:
        # Extract the filings object containing the columnar data
        filings_obj = data.get("filings", {})
        recent = filings_obj.get("recent", {})
        
        if not recent:
            logger.warning(f"No recent filings found for CIK {cik}")
            return filings
        
        # Get the parallel arrays
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        forms = recent.get("form", [])
        report_dates = recent.get("reportDate", [])
        
        # Iterate through the arrays and extract matching filings
        for idx, form in enumerate(forms):
            # Check if this filing matches the requested type
            if form.strip() == filing_type.strip():
                accession_num = accession_numbers[idx] if idx < len(accession_numbers) else ""
                filing_date = filing_dates[idx] if idx < len(filing_dates) else ""
                report_date = report_dates[idx] if idx < len(report_dates) else ""
                
                if accession_num:
                    # Construct the filing URL
                    # Format: https://www.sec.gov/cgibin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K&dateb=&owner=exclude&count=100
                    # Or: https://www.sec.gov/Archives/edgar/{cik}/{accession_num_with_dashes}/{accession_num}-index.htm
                    # Simple approach: use the accession number to construct Archives URL
                    accession_stripped = accession_num.replace("-", "")
                    accession_formatted = f"{accession_stripped[:10]}-{accession_stripped[10:12]}-{accession_stripped[12:]}"
                    
                    url = f"https://www.sec.gov/Archives/edgar/{cik}/{accession_formatted}/{accession_num}-index.htm"
                    
                    filings.append({
                        "filing_date": filing_date,
                        "report_date": report_date,
                        "accession_number": accession_num,
                        "url": url
                    })
        
        logger.info(f"Found {len(filings)} {filing_type} filings for CIK {cik}")
        return filings
    
    except Exception as e:
        logger.error(f"Error parsing SEC filings response for CIK {cik}: {e}")
        return filings


def _fetch_cik_from_sec(ticker: str) -> str:
    """
    Dynamically fetch CIK for a ticker from SEC Edgar company search.
    Fallback for tickers not in company_tickers.json mapping.
    """
    try:
        url = "https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            "company": ticker,
            "owner": "exclude",
            "action": "getcompany",
            "output": "json"
        }
        
        headers = _get_sec_request_headers()
        
        data = _sec_api_request(url, params, headers)
        
        if "cik_lookup" in data and data["cik_lookup"]:
            cik = str(data["cik_lookup"][0]["CIK_str"]).zfill(10)
            logger.info(f"Found CIK {cik} for ticker {ticker}")
            return cik
    except Exception as e:
        logger.warning(f"Failed to fetch CIK for {ticker}: {e}")
    
    return None
