"""
graph/architecture_experiment/orchestrator_workflow.py
=======================================================
Orchestrator-Worker workflow architecture variant.

Topology — hub-and-spoke
-------------------------
  START
    │
    ▼
  orchestrator_agent  ◄─────────────────────────────────────────────────────┐
    │                                                                         │
    │  (conditional: state["next_worker"])                                    │
    ├─► market_context_agent       ──────────────────────────────────────────┤
    ├─► sentiment_agent            ──────────────────────────────────────────┤
    ├─► investment_analyst_agent   ──────────────────────────────────────────┤
    ├─► auditor_agent              ──────────────────────────────────────────┘
    │
    └─► report_generator_agent  ─► END

How it works
------------
1. The orchestrator_agent is called first.  It uses an LLM (gpt-4o-mini) to
   inspect the user query and decide which specialist worker to invoke.
2. After any worker finishes its task, control returns to orchestrator_agent.
3. The orchestrator re-reads the updated state (what has already been done,
   what the auditor found, whether hallucination was detected) and decides the
   next action — which could be another worker, a re-run, or the final report.
4. This repeats until orchestrator_agent dispatches report_generator_agent,
   which ends the workflow.

Key differences from sequential / conditional
----------------------------------------------
* The orchestrator has full visibility of the state after every worker run
  and can adapt its plan dynamically (e.g. retry analyst if hallucinating,
  skip sentiment if not requested, skip auditor for simple queries).
* Worker order is not fixed in the graph — the LLM decides.
* A safety cap (MAX_ORCHESTRATOR_ITERATIONS) prevents runaway loops.

create_orchestrator_graph() accepts an optional node_overrides dict so the
experiment harness can inject timing/token-tracking wrappers.
"""

from __future__ import annotations

import sys
import textwrap
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

MAX_ORCHESTRATOR_ITERATIONS: int = 8
"""
Safety cap on total orchestrator iterations per run.
The orchestrator forces a route to report_generator_agent when this is reached.
Set high enough to allow: market_context + analyst + sentiment + auditor
+ one potential re-run of each (4 + 4 = 8 iterations with orchestrator calls).
"""

_VALID_WORKERS: frozenset = frozenset({
    "market_context_agent",
    "sentiment_agent",
    "investment_analyst_agent",
    "auditor_agent",
    "report_generator_agent",
})

# Tickers whose 10-K filings are indexed in the local ChromaDB VDB.
# If ALL tickers are in this set the orchestrator may use investment_analyst
# with RAG retrieval.  If any ticker is outside, it must rely on
# market_context_agent for external/live data retrieval.
_VDB_TICKERS: frozenset = frozenset({"NVDA", "AMZN", "AAPL", "GOOGL", "MSFT"})


# ── LLM prompts ───────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
    You are the central orchestrator of a wealth management AI system.

    Your role is to understand the user's request and dynamically coordinate
    specialist workers — one at a time — until the user's question is fully
    answered.  After each worker completes, you will be called again with the
    updated state so you can decide what to do next.

    AVAILABLE WORKERS
    -----------------
    market_context_agent
        Fetches live/current market data via external tools:
        recent news, earnings, analyst ratings, and SEC 10-K filings.
        Use when the query needs current prices, news, sentiment, OR when
        a ticker is NOT in the local vector database (see VDB TICKERS below).

    sentiment_agent
        Runs fine-tuned sentiment analysis on news headlines for the tickers.
        Use when the query asks about market sentiment, bullish/bearish signals,
        OR as a step before investment_analyst_agent when market data is the
        primary source (ticker not in VDB, or live-data query).

    investment_analyst_agent
        Writes a comprehensive investment analysis report using RAG retrieval
        from the local 10-K vector database AND any market data already in state.
        MUST run before auditor_agent.

    auditor_agent
        Fact-checks the investment analysis and computes RAGAS metrics.
        ONLY run this AFTER investment_analyst_agent has produced a draft.
        If hallucination is detected, re-run market_context_agent (for live-data
        queries) or investment_analyst_agent (for docs-only queries) before the
        final report.

    report_generator_agent
        Formats and delivers the final report.  Call this LAST — once
        analysis (and auditing) is complete.  Ends the entire workflow.

    VDB TICKERS (local 10-K filings are only available for these tickers)
    ----------------------------------------------------------------------
    NVDA, AMZN, AAPL, GOOGL, MSFT

    If the query involves a ticker NOT in this list (e.g. TSLA, META, AMD),
    there are no local filings — use market_context_agent for data retrieval.

    ROUTING PATTERNS
    ----------------
    Follow the pattern that best matches the user query:

    Pattern A — docs_only (ticker in VDB, fundamental/filing query)
      Keywords: 10K, annual report, risk factors, business model, fundamentals,
                revenue, valuation
      Sequence: investment_analyst_agent → auditor_agent → report_generator_agent

    Pattern B — market_only (live/sentiment query, NO investment analysis needed)
      Keywords: current, latest, today, recent, news, market, price,
                sentiment, bullish, bearish
      Sequence: market_context_agent → sentiment_agent → report_generator_agent

    Pattern C — market_analysis (ticker NOT in VDB, OR comprehensive report)
      Condition: any ticker outside VDB list, or query needs full analysis
      Keywords: comprehensive, risks, recommendations, financial report
      Sequence: market_context_agent → sentiment_agent
                  → investment_analyst_agent → auditor_agent → report_generator_agent

    Pattern D — hybrid_analysis (ticker in VDB, needs BOTH live + fundamentals)
      Keywords: long-term investment, investment potential, should I invest,
                recommendation, compare for investment
      Sequence: market_context_agent → investment_analyst_agent
                  → auditor_agent → report_generator_agent

    HARD RULES
    ----------
    1. NEVER call a worker that already completed successfully unless a re-run
       is needed (hallucination detected, iteration still low).
    2. NEVER call auditor_agent before investment_analyst_agent has run.
    3. Call report_generator_agent only when analysis (+ audit) is complete,
       or when the iteration cap is near.

    Respond with ONLY the exact worker name, nothing else.
    Valid responses: market_context_agent | sentiment_agent |
                     investment_analyst_agent | auditor_agent |
                     report_generator_agent
""").strip()

_HUMAN_PROMPT = textwrap.dedent("""
    USER QUERY   : {query}
    TICKERS      : {tickers}
    TICKERS IN VDB  : {tickers_in_vdb}
    TICKERS OUT VDB : {tickers_out_vdb}
    ITERATION    : {iteration} / {max_iter}

    COMPLETED WORKERS : {completed}

    CURRENT STATE
    -------------
    Live market data gathered : {has_market_context}
    Sentiment analysis done   : {has_sentiment}
    Draft report written      : {has_draft}
    Audit completed           : {has_audit}
    Hallucination detected    : {is_hallucinating}
    Audit score               : {audit_score}

    Which worker should run next?
""").strip()


# ── Orchestrator agent node ───────────────────────────────────────────────────

def orchestrator_agent_node(state: ArchExperimentState) -> dict:
    """
    Central orchestrator — called at the start and after every worker finishes.

    Inspects the current state to determine what has already been done, then
    uses GPT-4o-mini to decide which worker to invoke next.  The decision is
    written to state["next_worker"] and the conditional edge routes there.

    Falls back to deterministic logic if the LLM call fails.
    """
    print("--- AGENT: ORCHESTRATOR (dynamic planner) ---")

    iteration: int = state.get("orchestrator_iteration", 0) + 1

    # ── Safety cap: force final report when close to limit ────────────────────
    if iteration > MAX_ORCHESTRATOR_ITERATIONS:
        print(f"  Iteration cap ({MAX_ORCHESTRATOR_ITERATIONS}) reached — finalising.")
        return {
            "next_worker":            "report_generator_agent",
            "orchestrator_iteration": iteration,
            "messages":               [f"Orchestrator: cap reached — routing to reporter."],
        }

    # ── Build a human-readable state summary for the LLM ─────────────────────
    completed: List[str] = []
    if state.get("market_context"):
        completed.append("market_context_agent")
    if state.get("sentiment_summary"):
        completed.append("sentiment_agent")
    if state.get("draft_report"):
        completed.append("investment_analyst_agent")
    if state.get("audit_score", 0.0) > 0.0:
        completed.append("auditor_agent")

    messages: list = state.get("messages", [])
    query: str = str(messages[0]) if messages else ""
    tickers: List[str] = state.get("tickers", [])
    tickers_upper = [t.upper() for t in tickers]
    in_vdb = [t for t in tickers_upper if t in _VDB_TICKERS]
    out_vdb = [t for t in tickers_upper if t not in _VDB_TICKERS]

    human_text = _HUMAN_PROMPT.format(
        query=query,
        tickers=", ".join(tickers) if tickers else "(none)",
        tickers_in_vdb=", ".join(in_vdb) if in_vdb else "(none)",
        tickers_out_vdb=", ".join(out_vdb) if out_vdb else "(none)",
        iteration=iteration,
        max_iter=MAX_ORCHESTRATOR_ITERATIONS,
        completed=", ".join(completed) if completed else "none",
        has_market_context=bool(state.get("market_context")),
        has_sentiment=bool(state.get("sentiment_summary")),
        has_draft=bool(state.get("draft_report")),
        has_audit=state.get("audit_score", 0.0) > 0.0,
        is_hallucinating=state.get("is_hallucinating", False),
        audit_score=round(state.get("audit_score", 0.0), 2),
    )

    # ── LLM call ──────────────────────────────────────────────────────────────
    next_worker: str = _fallback_decision(state, completed)

    try:
        from langchain_openai import ChatOpenAI           # noqa: E402
        from langchain_core.messages import SystemMessage, HumanMessage  # noqa: E402

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_text),
        ])
        candidate: str = response.content.strip().lower().replace(" ", "_")

        if candidate in _VALID_WORKERS:
            next_worker = candidate
        else:
            print(f"  [warn] LLM returned unknown worker '{candidate}' — using fallback.")

    except Exception as exc:
        print(f"  [warn] Orchestrator LLM call failed ({exc}) — using fallback decision.")

    print(f"  → Iteration {iteration}: dispatching {next_worker}")

    return {
        "next_worker":            next_worker,
        "orchestrator_iteration": iteration,
        "messages":               [f"Orchestrator (iter {iteration}): → {next_worker}"],
    }


def _fallback_decision(state: dict, completed: List[str]) -> str:
    """
    Rule-based fallback used when the LLM call fails or returns an invalid
    worker name.  Mirrors the four routing patterns in the system prompt.

    Pattern A (docs_only):   no live data needed → analyst → auditor → report
    Pattern B (market_only): live-data/sentiment → market → sentiment → report
    Pattern C (market_analysis): ticker out of VDB → market → sentiment → analyst → auditor
    Pattern D (hybrid_analysis): ticker in VDB, hybrid → market → analyst → auditor
    """
    tickers = [t.upper() for t in state.get("tickers", [])]
    messages = state.get("messages", [])
    query = " ".join(str(m) for m in messages).lower()

    any_out_vdb = any(t not in _VDB_TICKERS for t in tickers) if tickers else False
    is_sentiment_query = any(
        kw in query for kw in ("sentiment", "bullish", "bearish", "current",
                               "latest", "today", "recent", "news", "price")
    )
    needs_analysis = any(
        kw in query for kw in ("risk", "risks", "recommendation", "recommendations",
                               "fundamentals", "10k", "annual", "valuation",
                               "comprehensive", "investment potential",
                               "long-term", "should i invest")
    )

    # ── Pattern B: pure sentiment / market-only query ─────────────────────────
    # No analyst needed; fetch market data, run sentiment, then audit the draft.
    is_market_only = is_sentiment_query and not needs_analysis and not any_out_vdb

    if is_market_only:
        if "market_context_agent" not in completed:
            return "market_context_agent"
        if "sentiment_agent" not in completed:
            return "sentiment_agent"
        if "auditor_agent" not in completed:
            return "auditor_agent"
        return "report_generator_agent"

    # ── Patterns C & D / A: analyst + auditor required ───────────────────────
    # Fetch live data first for out-of-VDB tickers (C) or hybrid queries (D).
    # Docs-only queries (A) can skip market_context entirely.
    needs_market_context = any_out_vdb or (
        not any_out_vdb and needs_analysis and is_sentiment_query
    ) or (
        tickers and not needs_analysis  # hybrid default when tickers present
    )

    if needs_market_context and "market_context_agent" not in completed:
        return "market_context_agent"

    # Run sentiment between market_context and analyst only for market_analysis (C)
    if any_out_vdb and "sentiment_agent" not in completed and \
            "market_context_agent" in completed:
        return "sentiment_agent"

    if "investment_analyst_agent" not in completed:
        return "investment_analyst_agent"

    if "auditor_agent" not in completed:
        return "auditor_agent"

    # Re-run on hallucination if budget allows
    if state.get("is_hallucinating") and state.get("orchestrator_iteration", 0) < 6:
        if any_out_vdb:
            return "market_context_agent"
        return "investment_analyst_agent"

    return "report_generator_agent"


# ── Routing function ──────────────────────────────────────────────────────────

def _dispatch_route(state: ArchExperimentState) -> str:
    """Read next_worker from state; default to report_generator on unknown value."""
    worker = state.get("next_worker", "report_generator_agent")
    if worker not in _VALID_WORKERS:
        print(f"  [warn] Unknown next_worker='{worker}' — defaulting to reporter.")
        return "report_generator_agent"
    return worker


# ── Graph factory ─────────────────────────────────────────────────────────────

def create_orchestrator_graph(
    node_overrides: Optional[Dict[str, Callable]] = None,
):
    """
    Build and compile the orchestrator-worker hub-and-spoke LangGraph.

    Parameters
    ----------
    node_overrides:
        Optional dict mapping node names to replacement callables.  Used by
        the experiment harness to inject per-node timing and token-tracking
        wrappers without modifying this file.

        Valid keys:
          "orchestrator_agent", "market_context_agent",
          "investment_analyst_agent", "sentiment_agent",
          "auditor_agent", "report_generator_agent"

    Returns
    -------
    A compiled ``CompiledGraph`` ready to call with ``.invoke()`` or
    ``.stream()``.

    Example
    -------
    >>> app = create_orchestrator_graph()
    >>> initial = make_initial_state("Analyze AAPL and MSFT", tickers=["AAPL", "MSFT"])
    >>> result = app.invoke(initial)
    >>> print(result["final_report"])
    """
    nodes: Dict[str, Callable] = {
        "orchestrator_agent":         orchestrator_agent_node,
        "market_context_agent":       market_context_node,
        "investment_analyst_agent":   analyst_node,
        "sentiment_agent":            sentiment_node,
        "auditor_agent":              auditor_node,
        "report_generator_agent":     report_generator_node,
    }
    if node_overrides:
        nodes.update(node_overrides)

    workflow = StateGraph(ArchExperimentState)

    # ── Register all nodes ────────────────────────────────────────────────────
    for name, fn in nodes.items():
        workflow.add_node(name, fn)

    # ── Entry: always start at orchestrator ───────────────────────────────────
    workflow.add_edge(START, "orchestrator_agent")

    # ── Orchestrator dispatches to any worker ─────────────────────────────────
    workflow.add_conditional_edges(
        "orchestrator_agent",
        _dispatch_route,
        {
            "market_context_agent":     "market_context_agent",
            "sentiment_agent":          "sentiment_agent",
            "investment_analyst_agent": "investment_analyst_agent",
            "auditor_agent":            "auditor_agent",
            "report_generator_agent":   "report_generator_agent",
        },
    )

    # ── Workers return to orchestrator (hub-and-spoke) ────────────────────────
    workflow.add_edge("market_context_agent",       "orchestrator_agent")
    workflow.add_edge("sentiment_agent",            "orchestrator_agent")
    workflow.add_edge("investment_analyst_agent",   "orchestrator_agent")
    workflow.add_edge("auditor_agent",              "orchestrator_agent")

    # ── Report generator is terminal ──────────────────────────────────────────
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
    Build a fully-initialised state dict for one orchestrator-worker run.

    Parameters
    ----------
    query:
        The user's natural-language investment question / request.
    tickers:
        Pre-extracted ticker symbols.  Passed to the orchestrator so it can
        decide whether live market data is needed (no separate router node).
    ground_truth:
        Optional reference answer for RAGAS context-recall.
    portfolio_weights:
        Optional ticker→weight mapping for portfolio-aware analysis.
    """
    return {
        # ── Input ──────────────────────────────────────────────────────────────
        "messages":              [query],
        "tickers":               list(tickers),
        "route_target":          "",   # unused in this variant

        # ── Orchestrator state ──────────────────────────────────────────────────
        "next_worker":           "",
        "orchestrator_iteration": 0,

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

ORCHESTRATOR_NODES: Dict[str, Callable] = {
    "orchestrator_agent":           orchestrator_agent_node,
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


# ── Quick smoke-test (run directly: python orchestrator_workflow.py) ──────────

if __name__ == "__main__":
    print("Building orchestrator-worker workflow graph ...")
    app = create_orchestrator_graph()
    print("Graph compiled successfully.")
    print("\nMermaid diagram:")
    print(app.get_graph().draw_mermaid())
