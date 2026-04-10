"""
graph/architecture_experiment/state.py
=======================================
Shared state definition for the three architecture-comparison workflows:
  - sequential_workflow.py
  - conditional_workflow.py       (coming next)
  - orchestration_worker_workflow.py  (coming next)

Design notes
------------
* Compatible with all existing agent nodes (market_context, analyst, sentiment,
  auditor, report_generator).  Agents call state.get(key, default), so any
  dict-like object works — no circular imports needed.
* `messages` uses operator.add so each node's output messages are appended
  rather than replaced.
* `audit_iteration_count` uses take_max so the counter can only increase even
  when LangGraph merges partial state updates in a conditional loop.
* `route_target` is kept here so the conditional workflow variant can reuse
  this same state class without modification.
"""

from __future__ import annotations

import operator
from typing import Annotated, Dict, List

# Reuse the same take_max function object as WealthManagerState so that
# LangGraph does not treat audit_iteration_count as having conflicting reducers
# when both state classes are registered in the same graph.
from graph.state import take_max  # noqa: F401


# ── State class ───────────────────────────────────────────────────────────────

class ArchExperimentState(Dict):
    """
    Typed state dictionary for the architecture-experiment workflow variants.

    All fields follow the same naming convention as the production
    WealthManagerState so that the real agent nodes (imported from agents/)
    can read and write state without any modification.
    """

    # ── Conversation & routing ─────────────────────────────────────────────────
    messages: Annotated[list, operator.add]
    """Accumulated conversation / agent log messages (append-only)."""

    tickers: List[str]
    """Ticker symbols extracted before the graph starts (no orchestrator node)."""

    route_target: str
    """Next-node name written by orchestrator; used only in the conditional variant."""

    # ── Market context agent ───────────────────────────────────────────────────
    market_context: dict
    """Raw cached results from every MCP tool call (news, earnings, ratings, 10-K)."""

    live_data_context: str
    """Human-readable summary formatted from market_context for the analyst."""

    # ── Investment analyst agent ───────────────────────────────────────────────
    retrieved_context: str
    """10-K text chunks retrieved from the ChromaDB vector store."""

    draft_report: str
    """Markdown investment report produced by the analyst."""

    portfolio_weights: Dict
    """Optional ticker→weight mapping; empty dict when not provided."""

    # ── Sentiment agent ────────────────────────────────────────────────────────
    news_articles: List[dict]
    """Raw news articles fetched per ticker."""

    sentiment_results: List[dict]
    """Per-headline FinBERT inference results."""

    sentiment_summary: dict
    """Aggregated sentiment scores and human-readable explanation."""

    sentiment_score: float
    """Overall scalar sentiment: -1.0 (very bearish) → +1.0 (very bullish)."""

    # ── Auditor agent ──────────────────────────────────────────────────────────
    audit_score: float
    """Quantitative quality score (0–1); 0.35 = REJECTED, 0.9 = APPROVED."""

    audit_findings: List[Dict]
    """Structured list of per-claim fact-check results."""

    is_hallucinating: bool
    """True when auditor determines the draft contains unverified claims."""

    hallucination_count: int
    verified_count: int
    unsubstantiated_count: int

    ragas_metrics: Dict
    """RAGAS scores: faithfulness, answer_relevancy, context_recall."""

    ground_truth: str
    """Optional reference answer used to compute RAGAS context_recall."""

    audit_iteration_count: Annotated[int, take_max]
    """Counts how many audit→re-run loops have occurred (max wins on merge)."""

    # ── Report generator agent ─────────────────────────────────────────────────
    final_report: str
    """Final markdown report returned to the user."""

    # ── Orchestrator-worker variant ────────────────────────────────────────────
    next_worker: str
    """Next worker node name dispatched by the orchestrator agent.
    Written on every orchestrator iteration; read by the conditional edge
    that routes control to the appropriate worker."""

    orchestrator_iteration: Annotated[int, take_max]
    """How many times the orchestrator agent has been called in this run.
    Uses take_max so the counter only moves forward even on merge."""

    # ── Experiment trace (populated by the harness instrumentation layer) ──────
    trace: Annotated[list, operator.add]
    """Ordered list of per-node trace entries injected by the experiment
    harness wrappers.  Uses operator.add so each node's entry is *appended*
    rather than replaced, giving the full execution path in final_state."""
