from typing import TypedDict, Annotated, List, Union, Dict
import operator

class WealthManagerState(TypedDict):
    # 'messages' stores the full conversation history for context retention
    messages: Annotated[list, operator.add]
    
    # Domain-specific state variables
    tickers: List[str]             # Stock tickers to analyze
    news_articles: List[Dict]      # From Sentiment Agent (NewsAPI)
    sentiment_results: List[Dict]  # Sentiment scores per article
    sentiment_summary: Dict        # Aggregated sentiment by ticker
    sentiment_score: float         # Overall sentiment score (-1.0 to 1.0)
    portfolio_weights: Dict        # From Portfolio Optimization Agent
    retrieved_context: str         # 10-K chunks from RAG Pipeline
    live_data_context: str         # Live news/earnings from MCP web scraper
    draft_report: str              # From Investment Analyst Agent
    audit_score: float             # Quantitative score from Auditor Agent
    is_hallucinating: bool         # Boolean flag for the self-correction loop