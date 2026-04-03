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
    audit_score:      float         # quantitative audit score
    audit_findings:   List[Dict]    # structured findings from auditor
    is_hallucinating: bool          # self-correction loop flag

    # ── Report generator agent ────────────────────────────────────────────────
    final_report: str               # final user-facing combined report
