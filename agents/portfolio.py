import yfinance as yf
import pandas as pd
from pypfopt import black_litterman, risk_models, expected_returns, EfficientFrontier
from graph.state import WealthManagerState

def portfolio_node(state: WealthManagerState):
    print("--- ⚖️ AGENT: PORTFOLIO OPTIMIZATION (Black-Litterman) ---")

    # 1. Define the Universe (The stocks we are allowed to buy)
    # For a Tech HNW client, we use these pillars:
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    
    # 2. Fetch Historical Data (Last 2 years)
    # This provides the 'Prior' (what the market usually does)
    data = yf.download(tickers, period="2y")["Adj Close"]
    
    # 3. Calculate Market Priors
    # S = Covariance matrix (Risk/Volatility)
    # mu = Average historical returns
    s = risk_models.sample_cov(data)
    mu = expected_returns.mean_historical_return(data)

    # 4. Integrate Sentiment as a 'View'
    # We translate the sentiment_score (-1 to 1) into a return 'tilt'
    # If sentiment is 0.8 (Bullish), we add a 5% 'View' to the expected return
    sentiment_tilt = state["sentiment_score"] * 0.05
    
    # We apply this tilt to all assets in our tech basket
    viewdict = {ticker: mu[ticker] + sentiment_tilt for ticker in tickers}

    # 5. Run Black-Litterman Model
    # This mathematically blends Market Wisdom + AI Sentiment
    bl = black_litterman.BlackLittermanModel(s, absolute_views=viewdict)
    rets_bl = bl.bl_returns()

    # 6. Optimize for Maximum Sharpe Ratio (Best Return per unit of Risk)
    ef = EfficientFrontier(rets_bl, s)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    # 7. Update State
    return {
        "portfolio_weights": dict(cleaned_weights),
        "messages": [f"Portfolio Rebalanced. Allocation: {dict(cleaned_weights)}"]
    }