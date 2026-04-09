"""
graph/architecture_experiment/conditional_workflow.py
=====================================================
Conditional workflow architecture variant.

Node execution order
---------------------
  START
    │
    ▼
  conditional_router              ← lightweight node: writes route_target to state
    │                               ("market_context_agent" or "sentiment_agent")
    ├─ needs live data?
    │     YES ─────────────────────────────────────────────────────────────────┐
    │                                                                          │
    │                              market_context_agent                        │
    │                                     │                                    │
    │     NO ──────────────────────────── │ ────────────────────────────────┐ │
    │                                     ▼                                 │ │
    └────────────────────────►      sentiment_agent       ◄─────────────────┘ │
                                          │                                    │
                                          ▼                                    │
                                investment_analyst_agent                       │
                                          │                                    │
                                          ▼                                    │
                                    auditor_agent                              │
                                          │                                    │
                                          ▼ conditional: is_hallucinating?     │
                            ┌─ True, initial_route == "market_context_agent" ──┘
                            ├─ True, initial_route == "sentiment_agent" ────────┐
                            │                                                   │
                            │           sentiment_agent  ◄──────────────────────┘
                            │
                            └─ False ─► report_generator_agent ─► END

Key design decisions
---------------------
* conditional_router is a logic-only node (zero LLM calls).  It inspects the
  initial state — tickers and query keywords — to decide whether fetching live
  market data is worth the latency and token cost.
* route_target is written once (by conditional_router) and never overwritten
  by downstream nodes, so the hallucination-correction router can always read
  which branch the run took and loop back to the right place.
* sentiment_agent always runs BEFORE investment_analyst_agent so that the
  analyst can incorporate the bullish/bearish sentiment signal when writing
  the draft report.
* MAX_AUDIT_ITERATIONS caps the hallucination loop at 2 re-runs, matching the
  sequential variant, so comparisons between the two are fair.
* create_conditional_graph() accepts an optional node_overrides dict so the
  experiment harness can inject timing/token-tracking wrappers without
  modifying this file.
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

# Keywords whose presence in the query suggests live market data is valuable.
_LIVE_DATA_KEYWORDS: frozenset = frozenset({
    "current", "latest", "recent", "today", "now", "live",
    "price", "earnings", "news", "rating", "analyst",
    "quarter", "quarterly", "forecast", "guidance",
    "buy", "sell", "hold", "recommend",
})


# ── Live-data routing helper ──────────────────────────────────────────────────

def _needs_live_data(state: dict) -> bool:
    """
    Return True when the query or state suggests that fetching live market
    data (via MCP tools) will add value to the analysis.

    Heuristics (no LLM call — fast, zero extra tokens):
      1. Tickers are present → almost always want current prices/earnings.
      2. Query contains recognised live-data keywords.
      3. Fallback: False (use RAG/historical data only).
    """
    if state.get("tickers"):
        return True

    messages: list = state.get("messages", [])
    query: str = " ".join(str(m) for m in messages).lower()
    query_words: set = set(query.split())
    return bool(query_words & _LIVE_DATA_KEYWORDS)


# ── Conditional router node ───────────────────────────────────────────────────

def conditional_router_node(state: ArchExperimentState) -> dict:
    """
    Entry node: decides whether to fetch live market data or go directly
    to the investment analyst.

    Writes ``route_target`` to state.  This value is:
      - Read by the conditional edge leaving this node to select the next step.
      - Preserved throughout the run so the hallucination-correction router
        knows which branch to loop back to if the audit fails.

    No LLM is called here; routing is purely logic-based.
    """
    print("--- ROUTER: CONDITIONAL (Is live data needed?) ---")

    target: str = (
        "market_context_agent"
        if _needs_live_data(state)
        else "sentiment_agent"
    )

    reason: str = (
        f"tickers={state.get('tickers')} / keywords matched"
        if target == "market_context_agent"
        else "no tickers, no live-data keywords — using RAG only"
    )
    print(f"  → Routing to {target} ({reason})")

    return {"route_target": target}


# ── Hallucination routing function ────────────────────────────────────────────

def hallucination_route(state: ArchExperimentState) -> str:
    """
    Routing function called after every auditor run.

    If hallucination is detected and the iteration cap has not been reached,
    loop back to wherever the conditional_router originally sent the run
    (market_context_agent OR sentiment_agent).  This ensures:

      * Live-data path → re-fetches fresh market data, then re-runs sentiment
                         and analyst before auditing again.
      * RAG-only path  → re-runs sentiment → analyst with the same static
                         context (cheaper; market_context was never fetched).

    Returns
    -------
    "market_context_agent"   re-fetch + re-run sentiment → analyst (live-data branch)
    "sentiment_agent"        re-run sentiment → analyst only (RAG-only branch)
    "report_generator_agent" audit passed or cap reached
    """
    is_hallucinating: bool = state.get("is_hallucinating", False)
    iterations: int = state.get("audit_iteration_count", 0)

    if is_hallucinating and iterations < MAX_AUDIT_ITERATIONS:
        # Loop back to the same entry point used in this run
        loop_target: str = state.get("route_target", "sentiment_agent")
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
    >>> initial = make_initial_state("What is the sentiment for NVDA?",
    ...                              tickers=["NVDA"])
    >>> result = app.invoke(initial)
    >>> print(result["final_report"])
    """
    # Default node implementations (can be overridden for instrumentation)
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

    workflow = StateGraph(ArchExperimentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    for name, fn in nodes.items():
        workflow.add_node(name, fn)

    # ── Entry: router decides initial branch ──────────────────────────────────
    workflow.add_edge(START, "conditional_router")

    workflow.add_conditional_edges(
        "conditional_router",
        lambda s: s.get("route_target", "sentiment_agent"),
        {
            "market_context_agent": "market_context_agent",
            "sentiment_agent":      "sentiment_agent",
        },
    )

    # ── Live-data branch: market_context feeds into sentiment ─────────────────
    workflow.add_edge("market_context_agent", "sentiment_agent")

    # ── Shared linear segment: sentiment → analyst → auditor ──────────────────
    workflow.add_edge("sentiment_agent",          "investment_analyst_agent")
    workflow.add_edge("investment_analyst_agent", "auditor_agent")

    # ── Conditional: hallucination loop or finish ─────────────────────────────
    workflow.add_conditional_edges(
        "auditor_agent",
        hallucination_route,
        {
            "market_context_agent":   "market_context_agent",
            "sentiment_agent":        "sentiment_agent",
            "report_generator_agent": "report_generator_agent",
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
        Pre-extracted ticker symbols.  The conditional_router inspects this
        list to decide whether live market data is needed.  Pass an empty
        list for queries with no ticker mentions.
    ground_truth:
        Optional reference answer for RAGAS context-recall calculation.
    portfolio_weights:
        Optional ticker→weight mapping for portfolio-aware analysis.
    """
    return {
        # ── Input ──────────────────────────────────────────────────────────────
        "messages":              [query],
        "tickers":               list(tickers),
        "route_target":          "",         # set by conditional_router_node

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

        # ── Trace (appended by every agent node) ──────────────────────────────
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
Imported by the experiment harness to wrap nodes for timing/token tracking
without duplicating the node list in two places.
"""


# ── Quick smoke-test (run directly: python conditional_workflow.py) ───────────

if __name__ == "__main__":
    print("Building conditional workflow graph ...")
    app = create_conditional_graph()
    print("Graph compiled successfully.")
    print("\nMermaid diagram:")
    print(app.get_graph().draw_mermaid())
