from typing import Annotated, Dict, List
import operator


def take_max(a, b):
    """Reducer that takes the maximum value between two numbers."""
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


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
    sentiment_logs:    List[str]    # detailed runtime logs from sentiment pipeline
    sentiment_start_date: str       # optional ISO date "YYYY-MM-DD" filter start
    sentiment_end_date:   str       # optional ISO date "YYYY-MM-DD" filter end

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
    audit_iteration_count: Annotated[int, take_max]  # counts audit-analyst loop iterations, takes max to preserve increments
    # ── Report generator agent ────────────────────────────────────────────────
    final_report: str               # final user-facing combined report
    
    # ── Market Context Agent ───────────────────────────────────────────────────
    market_context: dict            # cached market data (news, earnings, ratings, 10-K, filings)
    
    # Domain-specific state variables
    portfolio_weights: Dict        # From Portfolio Optimization Agent
    live_data_context: str         # Live news/earnings from MCP web scraper
