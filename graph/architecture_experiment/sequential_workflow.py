"""
graph/architecture_experiment/sequential_workflow.py
=====================================================
Sequential workflow architecture variant.

Node execution order
---------------------
  START
    │
    ▼
  market_context_agent       ← fetches live market data (news, earnings, 10-K)
    │
    ▼
  investment_analyst_agent   ← writes draft investment report (RAG + market data)
    │
    ▼
  sentiment_agent            ← FinBERT sentiment analysis on news headlines
    │
    ▼
  auditor_agent              ← fact-checks draft; increments audit_iteration_count
    │
    ▼  conditional: is_hallucinating AND iteration < MAX_AUDIT_ITERATIONS?
    ├─ True  ──────────────────────────────────────────────────────────────────┐
    │                                                                          │
    └─ False ─► report_generator_agent ─► END                                 │
                                                                               │
         ◄────────────────────── loop back ───────────────────────────────────┘

Design notes
------------
* No orchestrator node — the caller must extract tickers and include them in
  the initial state (see `make_initial_state` below).
* Sentiment runs AFTER the analyst so that the report generator can reference
  both the analysis and the final sentiment score in one pass.
* The hallucination-correction loop targets market_context_agent (not analyst)
  so that fresh external data is fetched before the report is regenerated.
  This is more expensive but addresses the root cause when stale data is the
  source of hallucinated figures.
* MAX_AUDIT_ITERATIONS caps the loop at 2 re-runs to prevent infinite cycles;
  the auditor forces APPROVED on the final iteration regardless.
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
"""Maximum number of times the auditor may send the workflow back to
market_context for a re-run.  The auditor itself enforces this cap and forces
APPROVED on the final iteration, but the routing function also checks it as a
safety net."""


# ── Routing function ──────────────────────────────────────────────────────────

def _audit_route(state: ArchExperimentState) -> str:
    """
    Decide what happens after the auditor runs.

    Returns
    -------
    "market_context_agent"
        The draft is hallucinating AND the loop cap has not been reached.
        The workflow returns to market_context so fresh data can be fetched
        before the analyst rewrites the report.
    "report_generator_agent"
        Either the draft passed the audit, or we have hit the iteration cap.
    """
    is_hallucinating: bool = state.get("is_hallucinating", False)
    iterations: int = state.get("audit_iteration_count", 0)

    if is_hallucinating and iterations < MAX_AUDIT_ITERATIONS:
        print(
            f"  [sequential] Hallucination detected "
            f"(iteration {iterations}/{MAX_AUDIT_ITERATIONS}) — "
            "routing back to market_context_agent."
        )
        return "market_context_agent"

    print(
        f"  [sequential] Audit passed or iteration cap reached "
        f"(iteration {iterations}) — routing to report_generator_agent."
    )
    return "report_generator_agent"


# ── Graph factory ─────────────────────────────────────────────────────────────

def create_sequential_graph(
    node_overrides: Optional[Dict[str, Callable]] = None,
):
    """
    Build and compile the sequential workflow LangGraph.

    Parameters
    ----------
    node_overrides:
        Optional dict mapping node names to replacement callables.  Used by
        the experiment harness to inject per-node timing and token-tracking
        wrappers without modifying this file.

        Valid keys:
          "market_context_agent", "investment_analyst_agent",
          "sentiment_agent", "auditor_agent", "report_generator_agent"

    Returns
    -------
    A compiled ``CompiledGraph`` ready to call with ``.invoke()`` or
    ``.stream()``.

    Example
    -------
    >>> app = create_sequential_graph()
    >>> initial = make_initial_state("Analyze AAPL", tickers=["AAPL"])
    >>> result = app.invoke(initial)
    >>> print(result["final_report"])
    """
    # Default node implementations (can be overridden for instrumentation)
    nodes: Dict[str, Callable] = {
        "market_context_agent":       market_context_node,
        "investment_analyst_agent":   analyst_node,
        "sentiment_agent":            sentiment_node,
        "auditor_agent":              auditor_node,
        "report_generator_agent":     report_generator_node,
    }
    if node_overrides:
        nodes.update(node_overrides)

    workflow = StateGraph(ArchExperimentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    for name, fn in nodes.items():
        workflow.add_node(name, fn)

    # ── Linear edges (fixed sequential order) ─────────────────────────────────
    workflow.add_edge(START,                        "market_context_agent")
    workflow.add_edge("market_context_agent",       "investment_analyst_agent")
    workflow.add_edge("investment_analyst_agent",   "sentiment_agent")
    workflow.add_edge("sentiment_agent",            "auditor_agent")

    # ── Conditional: hallucination loop or finish ─────────────────────────────
    workflow.add_conditional_edges(
        "auditor_agent",
        _audit_route,
        {
            "market_context_agent":   "market_context_agent",
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
    portfolio_weights: dict | None = None,
) -> dict:
    """
    Build a fully-initialised state dict for one sequential workflow run.

    Parameters
    ----------
    query:
        The user's natural-language investment question / request.
    tickers:
        Stock ticker symbols to analyse (e.g. ``["AAPL", "MSFT"]``).
        These must be pre-extracted; there is no orchestrator node in this
        workflow variant to do it automatically.
    ground_truth:
        Optional reference answer used by the auditor to calculate RAGAS
        context-recall.  Leave empty if not available.
    portfolio_weights:
        Optional dict mapping ticker → portfolio weight (e.g.
        ``{"AAPL": 0.6, "MSFT": 0.4}``).  Passed to the analyst for
        portfolio-aware report generation.
    """
    return {
        # ── Input ──────────────────────────────────────────────────────────────
        "messages":              [query],
        "tickers":               list(tickers),
        "route_target":          "",         # unused in sequential; included for
                                             # state-class compatibility
        # ── Market context (populated by market_context_agent) ─────────────────
        "market_context":        {},
        "live_data_context":     "",

        # ── Analyst (populated by investment_analyst_agent) ────────────────────
        "retrieved_context":     "",
        "draft_report":          "",
        "portfolio_weights":     portfolio_weights or {},

        # ── Sentiment (populated by sentiment_agent) ───────────────────────────
        "news_articles":         [],
        "sentiment_results":     [],
        "sentiment_summary":     {},
        "sentiment_score":       0.0,

        # ── Auditor (populated by auditor_agent) ───────────────────────────────
        "audit_score":           0.0,
        "audit_findings":        [],
        "is_hallucinating":      False,
        "hallucination_count":   0,
        "verified_count":        0,
        "unsubstantiated_count": 0,
        "ragas_metrics":         {},
        "ground_truth":          ground_truth,
        "audit_iteration_count": 0,

        # ── Report (populated by report_generator_agent) ───────────────────────
        "final_report":          "",

        # ── Experiment trace (accumulated by harness wrappers) ─────────────────
        "trace":                 [],
    }


# ── Node registry (exported for experiment instrumentation) ──────────────────

SEQUENTIAL_NODES: Dict[str, Callable] = {
    "market_context_agent":     market_context_node,
    "investment_analyst_agent": analyst_node,
    "sentiment_agent":          sentiment_node,
    "auditor_agent":            auditor_node,
    "report_generator_agent":   report_generator_node,
}
"""
Ordered dict of {node_name: callable} for all nodes in this workflow.
Imported by the experiment harness to wrap nodes for timing/token tracking
without duplicating the node list in two places.
"""


# ── Quick smoke-test (run directly: python sequential_workflow.py) ────────────

if __name__ == "__main__":
    print("Building sequential workflow graph ...")
    app = create_sequential_graph()
    print("Graph compiled successfully.")
    print("\nMermaid diagram:")
    print(app.get_graph().draw_mermaid())
