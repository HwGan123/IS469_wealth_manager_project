"""
graph/architecture_experiment/conditional_workflow.py
=====================================================
Conditional workflow architecture variant.

Node execution order — four possible paths determined by the router
---------------------------------------------------------------------

  docs_only path (Q1-type: ticker IN VDB, fundamental/filing query)
  ─────────────────────────────────────────────────────────────────
  START → conditional_router → investment_analyst_agent
                                       │
                                       ▼
                                 auditor_agent ──[halluc?]──► investment_analyst_agent
                                       │
                                       └── pass ──► report_generator_agent → END

  market_only path (Q2-type: live/sentiment query, pure market data)
  ─────────────────────────────────────────────────────────────────
  START → conditional_router → market_context_agent
                                       │
                                       ▼
                                 sentiment_agent → report_generator_agent → END

  market_analysis path (Q4-type: ticker NOT IN VDB, analysis required)
  ────────────────────────────────────────────────────────────────────
  START → conditional_router → market_context_agent
                                       │
                                       ▼
                                 sentiment_agent
                                       │
                                       ▼
                              investment_analyst_agent
                                       │
                                       ▼
                                 auditor_agent ──[halluc?]──► market_context_agent
                                       │
                                       └── pass ──► report_generator_agent → END

  hybrid_analysis path (Q3-type: ticker IN VDB, needs both live + docs)
  ──────────────────────────────────────────────────────────────────────
  START → conditional_router → market_context_agent
                                       │
                                       ▼
                              investment_analyst_agent (uses VDB + market data)
                                       │
                                       ▼
                                 auditor_agent ──[halluc?]──► market_context_agent
                                       │
                                       └── pass ──► report_generator_agent → END

Key design decisions
---------------------
* VDB_TICKERS: the local ChromaDB only has 10-K filings for NVDA, AMZN,
  AAPL, GOOGL, MSFT.  Any ticker outside this set forces a market_context
  data source (market_only or market_analysis depending on whether the query
  also needs investment analysis).
* conditional_router is a zero-LLM-call rule-based node.  It inspects:
    1. Whether any ticker is outside VDB_TICKERS → market-based path
    2. Query keywords to classify: docs_only / market_only / hybrid_analysis
    3. If ticker outside VDB but query needs analysis → market_analysis
* route_target encodes which of the four paths to take.  It is written once
  (by conditional_router) and read by every subsequent conditional edge.
* auditor_agent only runs after investment_analyst_agent.
  The market_only path (Q2) skips both analyst and auditor entirely.
* MAX_AUDIT_ITERATIONS caps the hallucination loop at 2 re-runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

# ── Ensure project root is importable when this file is run directly ──────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END

from graph.architecture_experiment.state import ArchExperimentState
from agents.market_context import market_context_node
from agents.analyst import analyst_node
from agents.sentiment_agent import sentiment_node
from agents.auditor import auditor_node
from agents.report_generator import report_generator_node


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_AUDIT_ITERATIONS: int = 2
"""Maximum hallucination-correction loops before forcing report generation."""

# Tickers whose 10-K filings are indexed in the local ChromaDB vector store.
VDB_TICKERS: frozenset = frozenset({"NVDA", "AMZN", "AAPL", "GOOGL", "MSFT"})

# ── Keyword sets ──────────────────────────────────────────────────────────────

# Keywords that indicate the user wants LIVE / current market information.
# Presence → prefer market_context as the data source.
_MARKET_ONLY_WORDS: frozenset = frozenset({
    "current", "latest", "today", "recent", "news",
    "market", "price", "sentiment", "bullish", "bearish",
})

# Keywords that indicate filing / fundamental analysis from documents.
# Presence → prefer VDB (investment_analyst RAG) as the data source.
_DOCS_ONLY_WORDS: frozenset = frozenset({
    "10k", "10-k", "10q", "10-q",
    "risk", "risks", "fundamentals",
    "revenue", "valuation", "profitability", "profit",
    "earnings", "margin", "margins", "quarterly", "filing",
})
_DOCS_ONLY_PHRASES: tuple = (
    "annual report", "risk factors", "business model",
    "long term", "long-term",
)

# Keywords that indicate the user wants BOTH live context AND fundamental
# analysis — a comprehensive investment-level recommendation.
_HYBRID_WORDS: frozenset = frozenset({
    "recommendation", "recommendations",
})
_HYBRID_PHRASES: tuple = (
    "long-term investment", "investment potential",
    "should i invest", "compare for investment",
)


# ── Routing helper ────────────────────────────────────────────────────────────

def _determine_route(state: dict) -> str:
    """
    Pure rule-based routing — no LLM calls.

    Returns one of four route_target strings:
      "docs_only"        → investment_analyst_agent (RAG, no live data)
      "market_only"      → market_context + sentiment → report (no analyst)
      "market_analysis"  → market_context + sentiment + analyst + auditor
      "hybrid_analysis"  → market_context + analyst + auditor (no sentiment)

    Rules applied in order
    ----------------------
    1. If any ticker is outside VDB_TICKERS:
         - Query needs analysis keywords → "market_analysis"
         - Otherwise                     → "market_only"
    2. All tickers are in VDB (or no tickers given):
         - Hybrid keywords present   → "hybrid_analysis"
         - Docs-only keywords present → "docs_only"
         - Market-only keywords present → "market_only"
         - Default (tickers present) → "hybrid_analysis"
         - Default (no tickers)      → "docs_only"
    """
    tickers = [t.upper() for t in state.get("tickers", [])]
    messages = state.get("messages", [])
    query = " ".join(str(m) for m in messages).lower()
    words = set(query.split())

    # ── Step 1: keyword presence ──────────────────────────────────────────────
    # Use substring matching so that punctuation-adjacent words like "10k,"
    # "risks?" or "recommendations." still match their keyword.
    has_market_only = any(kw in query for kw in _MARKET_ONLY_WORDS)

    has_docs_only = any(kw in query for kw in _DOCS_ONLY_WORDS) or any(
        phrase in query for phrase in _DOCS_ONLY_PHRASES
    )

    has_hybrid = any(kw in query for kw in _HYBRID_WORDS) or any(
        phrase in query for phrase in _HYBRID_PHRASES
    )

    # "analysis needed" = either docs_only keywords or hybrid keywords
    has_analysis = has_docs_only or has_hybrid

    # ── Step 2: VDB coverage check ────────────────────────────────────────────
    any_outside_vdb = any(t not in VDB_TICKERS for t in tickers) if tickers else False

    if any_outside_vdb:
        # Ticker not in VDB → must use live market data as the data source.
        # Still run analyst if the query requests investment-level analysis.
        if has_analysis:
            return "market_analysis"
        return "market_only"

    # ── Step 3: all tickers covered by VDB (or no tickers) ───────────────────
    if has_hybrid:
        return "hybrid_analysis"
    # Both doc signals AND live signals → need VDB + market data
    if has_docs_only and has_market_only:
        return "hybrid_analysis"
    if has_docs_only:
        return "docs_only"
    if has_market_only:
        return "market_only"

    # Default: tickers present but no clear keyword signal → hybrid analysis
    return "hybrid_analysis" if tickers else "docs_only"


# ── Conditional router node ───────────────────────────────────────────────────

def conditional_router_node(state: ArchExperimentState) -> dict:
    """
    Entry node: classifies the query and sets ``route_target``.

    Inspects tickers (VDB coverage) and query keywords to choose one of:
      docs_only       | market_only | market_analysis | hybrid_analysis

    No LLM is called here; routing is entirely rule-based.
    """
    print("--- ROUTER: CONDITIONAL ---")

    route = _determine_route(state)

    tickers = state.get("tickers", [])
    msg_preview = str(state.get("messages", [""])[0])[:60]
    print(f"  tickers={tickers}  query='{msg_preview}...'")
    print(f"  → route_target = '{route}'")

    return {"route_target": route}


# ── After-market-context routing ──────────────────────────────────────────────

def _after_market_context_route(state: ArchExperimentState) -> str:
    """
    Decide what runs immediately after market_context_agent.

    hybrid_analysis   → sentiment_agent            (then → analyst → auditor)
    market_only       → sentiment_agent            (then → auditor, no analyst)
    market_analysis   → sentiment_agent            (then → analyst → auditor)
    """
    return "sentiment_agent"


# ── After-sentiment routing ───────────────────────────────────────────────────

def _after_sentiment_route(state: ArchExperimentState) -> str:
    """
    Decide what runs after sentiment_agent.

    market_only      → auditor_agent            (fact-check sentiment draft before report)
    market_analysis  → investment_analyst_agent  (proceed with full analysis)
    hybrid_analysis  → investment_analyst_agent  (VDB + market + sentiment → analyst)
    """
    route = state.get("route_target", "market_only")
    if route in ("market_analysis", "hybrid_analysis"):
        return "investment_analyst_agent"
    return "auditor_agent"


# ── Hallucination routing function ────────────────────────────────────────────

def hallucination_route(state: ArchExperimentState) -> str:
    """
    Routing function called after every auditor run.

    If hallucination detected and cap not reached, loop back to the
    appropriate re-entry point:
      docs_only       → investment_analyst_agent  (re-analyse with same RAG)
      hybrid_analysis → market_context_agent      (re-fetch live data first)
      market_analysis → market_context_agent      (re-fetch live data first)

    market_only reaches the auditor after sentiment_agent; on re-run it loops
    back to sentiment_agent (re-fetch + re-classify with same live data).
    """
    is_hallucinating: bool = state.get("is_hallucinating", False)
    iterations: int = state.get("audit_iteration_count", 0)

    if is_hallucinating and iterations < MAX_AUDIT_ITERATIONS:
        route = state.get("route_target", "docs_only")
        if route == "docs_only":
            loop_target = "investment_analyst_agent"
        elif route == "market_only":
            loop_target = "sentiment_agent"
        else:
            loop_target = "market_context_agent"
        print(
            f"  [conditional] Hallucination detected "
            f"(iteration {iterations}/{MAX_AUDIT_ITERATIONS}) — "
            f"looping back to {loop_target}."
        )
        return loop_target

    print(
        f"  [conditional] Audit passed or cap reached "
        f"(iteration {iterations}) — routing to report_generator_agent."
    )
    return "report_generator_agent"


# ── Graph factory ─────────────────────────────────────────────────────────────

def create_conditional_graph(
    node_overrides: Optional[Dict[str, Callable]] = None,
    state_schema=None,
):
    """
    Build and compile the conditional workflow LangGraph.

    Parameters
    ----------
    node_overrides:
        Optional dict mapping node names to replacement callables.  Used by
        the experiment harness to inject per-node timing and token-tracking
        wrappers without modifying this file.

        Valid keys:
          "conditional_router", "market_context_agent",
          "investment_analyst_agent", "sentiment_agent",
          "auditor_agent", "report_generator_agent"

    Returns
    -------
    A compiled ``CompiledGraph`` ready to call with ``.invoke()`` or
    ``.stream()``.

    Example
    -------
    >>> app = create_conditional_graph()
    >>> initial = make_initial_state("Based on AAPL's 10K, what are the risks?",
    ...                              tickers=["AAPL"])
    >>> result = app.invoke(initial)
    >>> print(result["final_report"])
    """
    nodes: Dict[str, Callable] = {
        "conditional_router":           conditional_router_node,
        "market_context_agent":         market_context_node,
        "investment_analyst_agent":     analyst_node,
        "sentiment_agent":              sentiment_node,
        "auditor_agent":                auditor_node,
        "report_generator_agent":       report_generator_node,
    }
    if node_overrides:
        nodes.update(node_overrides)

    workflow = StateGraph(state_schema if state_schema is not None else ArchExperimentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    for name, fn in nodes.items():
        workflow.add_node(name, fn)

    # ── Entry: router classifies the query ────────────────────────────────────
    workflow.add_edge(START, "conditional_router")

    # ── Router dispatches to starting node ────────────────────────────────────
    # docs_only     → investment_analyst_agent (RAG, no live data needed)
    # everything else → market_context_agent (fetch live data first)
    workflow.add_conditional_edges(
        "conditional_router",
        lambda s: (
            "investment_analyst_agent"
            if s.get("route_target") == "docs_only"
            else "market_context_agent"
        ),
        {
            "investment_analyst_agent": "investment_analyst_agent",
            "market_context_agent":     "market_context_agent",
        },
    )

    # ── After market_context: branch based on route ───────────────────────────
    # hybrid_analysis → skip sentiment, go straight to analyst (uses both VDB + market)
    # market_only / market_analysis → run sentiment first
    workflow.add_conditional_edges(
        "market_context_agent",
        _after_market_context_route,
        {
            "investment_analyst_agent": "investment_analyst_agent",
            "sentiment_agent":          "sentiment_agent",
        },
    )

    # ── After sentiment: branch based on route ────────────────────────────────
    # market_only     → auditor_agent (fact-check sentiment draft)
    # market_analysis → investment_analyst_agent (then auditor)
    workflow.add_conditional_edges(
        "sentiment_agent",
        _after_sentiment_route,
        {
            "auditor_agent":            "auditor_agent",
            "investment_analyst_agent": "investment_analyst_agent",
        },
    )

    # ── Analyst always feeds auditor ──────────────────────────────────────────
    workflow.add_edge("investment_analyst_agent", "auditor_agent")

    # ── Conditional: hallucination loop or finish ─────────────────────────────
    workflow.add_conditional_edges(
        "auditor_agent",
        hallucination_route,
        {
            "market_context_agent":     "market_context_agent",
            "investment_analyst_agent": "investment_analyst_agent",
            "sentiment_agent":          "sentiment_agent",
            "report_generator_agent":   "report_generator_agent",
        },
    )

    workflow.add_edge("report_generator_agent", END)

    return workflow.compile()


# ── Initial-state factory ─────────────────────────────────────────────────────

def make_initial_state(
    query: str,
    tickers: List[str],
    ground_truth: str = "",
    portfolio_weights: dict = None,
) -> dict:
    """
    Build a fully-initialised state dict for one conditional workflow run.

    Parameters
    ----------
    query:
        The user's natural-language investment question / request.
    tickers:
        Pre-extracted ticker symbols.  The conditional_router checks these
        against VDB_TICKERS to select the data-source path.
    ground_truth:
        Optional reference answer for RAGAS context-recall calculation.
    portfolio_weights:
        Optional ticker→weight mapping for portfolio-aware analysis.
    """
    return {
        # ── Input ──────────────────────────────────────────────────────────────
        "messages":              [query],
        "tickers":               list(tickers),
        "route_target":          "",        # set by conditional_router_node

        # ── Market context ──────────────────────────────────────────────────────
        "market_context":        {},
        "live_data_context":     "",

        # ── Analyst ─────────────────────────────────────────────────────────────
        "retrieved_context":     "",
        "draft_report":          "",
        "portfolio_weights":     portfolio_weights or {},

        # ── Sentiment ───────────────────────────────────────────────────────────
        "news_articles":         [],
        "sentiment_results":     [],
        "sentiment_summary":     {},
        "sentiment_score":       0.0,

        # ── Auditor ─────────────────────────────────────────────────────────────
        "audit_score":           0.0,
        "audit_findings":        [],
        "is_hallucinating":      False,
        "hallucination_count":   0,
        "verified_count":        0,
        "unsubstantiated_count": 0,
        "ragas_metrics":         {},
        "ground_truth":          ground_truth,
        "audit_iteration_count": 0,

        # ── Report ───────────────────────────────────────────────────────────────
        "final_report":          "",

        # ── Experiment trace (accumulated by harness wrappers) ─────────────────
        "trace":                 [],
    }


# ── Node registry (exported for experiment instrumentation) ──────────────────

CONDITIONAL_NODES: Dict[str, Callable] = {
    "conditional_router":           conditional_router_node,
    "market_context_agent":         market_context_node,
    "investment_analyst_agent":     analyst_node,
    "sentiment_agent":              sentiment_node,
    "auditor_agent":                auditor_node,
    "report_generator_agent":       report_generator_node,
}
"""
Ordered dict of {node_name: callable} for all nodes in this workflow.
Imported by the experiment harness to wrap nodes for timing/token tracking.
"""


# ── Quick smoke-test (run directly: python conditional_workflow.py) ───────────

if __name__ == "__main__":
    print("Building conditional workflow graph ...")
    app = create_conditional_graph()
    print("Graph compiled successfully.")
    print("\nMermaid diagram:")
    print(app.get_graph().draw_mermaid())

    # Quick routing tests (no LLM calls)
    _tests = [
        ("Based on AAPL's 10K, what are its main business risks?", ["AAPL"],
         "docs_only"),
        ("What is the current market sentiment for NVDA today?", ["NVDA"],
         "market_only"),
        ("Analyse AAPL and MSFT for long-term investment potential.", ["AAPL", "MSFT"],
         "hybrid_analysis"),
        ("Provide a comprehensive financial report on TSLA, including risks "
         "and recommendations.", ["TSLA"], "market_analysis"),
    ]
    print("\nRouting sanity checks:")
    for q, t, expected in _tests:
        state = {"messages": [q], "tickers": t}
        result = _determine_route(state)
        status = "✓" if result == expected else f"✗ (got {result!r})"
        print(f"  {status}  {expected!r:20s}  {q[:60]}")
