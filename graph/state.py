from typing import Annotated, Dict, List
import operator


class WealthManagerState(Dict):
    # ── Conversation history ───────────────────────────────────────────────────
    messages: Annotated[list, operator.add]

    # ── Orchestrator ───────────────────────────────────────────────────────────
    tickers: List[str]              # ticker symbols extracted from user input
    route_target: str               # next node chosen by orchestrator

    # ── Sentiment agent ────────────────────────────────────────────────────────
    news_articles:     List[dict]   # raw articles from NewsAPI (tool 1 output)
    sentiment_results: List[dict]   # per-headline inference results (tool 2 output)
    sentiment_summary: dict         # aggregated scores + human-readable summary
    sentiment_score:   float        # overall scalar score (-1 bearish → +1 bullish)

    # ── Investment analyst agent ──────────────────────────────────────────────
    retrieved_context: str          # RAG context chunks
    draft_report:      str          # generated markdown report

    # ── Auditor agent ─────────────────────────────────────────────────────────
    audit_score:           float           # quantitative audit score (0-1)
    audit_findings:        List[Dict]      # structured findings from auditor
    is_hallucinating:      bool            # self-correction loop flag
    hallucination_count:   int             # number of hallucinations detected
    verified_count:        int             # number of verified claims
    unsubstantiated_count: int             # number of unsubstantiated claims
    ragas_metrics:         Dict            # RAGAS scores (faithfulness, relevancy, recall)
    ground_truth:          str             # expected/correct answer for comparison

    # ── Report generator agent ────────────────────────────────────────────────
    final_report: str               # final user-facing combined report
    
    # ── Market Context Agent ───────────────────────────────────────────────────
    market_context: dict            # cached market data (news, earnings, ratings, 10-K, filings)
    
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
