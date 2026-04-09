"""
WealthMind AI — FastAPI backend
Wraps the LangGraph multi-agent workflow and streams agent progress
via Server-Sent Events (SSE).

Run from the project root:
    uvicorn backend.server:app --reload --port 8000
"""

import asyncio
import json
import os
import sys

# ── Make sure project root is on sys.path so agents/* can be imported ─────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

from graph.workflow import create_wealth_manager_graph

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="WealthMind AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State fields we care about sending to clients ─────────────────────────────
STREAM_FIELDS = {
    "tickers",
    "sentiment_score",
    "sentiment_summary",
    "sentiment_result",
    "sentiment_logs",
    "sentiment_results",
    "audit_score",
    "is_hallucinating",
    "audit_findings",
    "hallucination_count",
    "verified_count",
    "unsubstantiated_count",
    "ragas_metrics",
    "draft_report",
    "final_report",
    "live_data_context",
    "route_target",
    "portfolio_weights",
}

# Node → friendly display name for the step indicator
NODE_LABELS = {
    "orchestrator_agent": "Orchestrator",
    "market_context_agent": "Market Context",
    "sentiment_agent": "Sentiment",
    "investment_analyst_agent": "Analyst",
    "auditor_agent": "Auditor",
    "report_generator_agent": "Report",
}


# ── Request schema ─────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    message: str
    tickers: List[str] = []


# ── SSE helper ─────────────────────────────────────────────────────────────────
def sse(payload: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


# ── Core streaming generator ───────────────────────────────────────────────────
async def stream_workflow(message: str, tickers: List[str]):
    """
    Runs the LangGraph workflow and yields SSE events:
      - node_start   : a node just became active
      - node_complete: a node finished (with partial state)
      - complete     : entire pipeline done
      - error        : something went wrong
    """
    graph = create_wealth_manager_graph()
    sse_loop = asyncio.get_running_loop()

    log_queue: asyncio.Queue = asyncio.Queue()
    graph_queue: asyncio.Queue = asyncio.Queue()

    initial_state = {
        "messages": [message],
        "tickers": tickers,
        "news_articles": [],
        "sentiment_results": [],
        "sentiment_summary": {},
        "sentiment_score": 0.0,
        "portfolio_weights": {},
        "retrieved_context": "",
        "live_data_context": "",
        "draft_report": "",
        "audit_score": 0.0,
        "is_hallucinating": False,
        "__sse_log_queue": log_queue,
        "__sse_loop": sse_loop,
    }

    # Signal that the first node is about to start
    yield sse(
        {
            "type": "node_start",
            "node": "orchestrator_agent",
            "label": NODE_LABELS["orchestrator_agent"],
        }
    )

    # Map: once we see node N complete, we know the next likely node
    # (exact next depends on conditional edges — we just broadcast node_start optimistically)
    NEXT_NODE = {
        "orchestrator_agent": "market_context_agent",
        "market_context_agent": "sentiment_agent",
        "sentiment_agent": "investment_analyst_agent",
        "investment_analyst_agent": "auditor_agent",
        "auditor_agent": "report_generator_agent",
    }

    async def _run_graph() -> None:
        try:
            async for event in graph.astream(initial_state):
                await graph_queue.put({"kind": "graph", "event": event})
            await graph_queue.put({"kind": "complete"})
        except Exception as exc:  # pragma: no cover
            await graph_queue.put({"kind": "error", "message": str(exc)})

    graph_task = asyncio.create_task(_run_graph())

    try:
        graph_done = False
        while True:
            get_log_task = asyncio.create_task(log_queue.get())
            get_graph_task = asyncio.create_task(graph_queue.get())

            done, pending = await asyncio.wait(
                {get_log_task, get_graph_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

            if get_log_task in done:
                log_item = get_log_task.result()
                if isinstance(log_item, dict):
                    yield sse(
                        {
                            "type": "agent_log",
                            "node": log_item.get("node", "sentiment_agent"),
                            "message": str(log_item.get("message", "")),
                        }
                    )
                continue

            graph_item = get_graph_task.result()
            kind = graph_item.get("kind")

            if kind == "error":
                yield sse(
                    {
                        "type": "error",
                        "message": graph_item.get("message", "Unknown error"),
                    }
                )
                break

            if kind == "complete":
                graph_done = True
            elif kind == "graph":
                event = graph_item["event"]
                for node_name, state_update in event.items():
                    # Strip state to only fields the UI needs
                    state_snapshot = {
                        k: v
                        for k, v in state_update.items()
                        if k in STREAM_FIELDS and v not in (None, "", [], {})
                    }

                    yield sse(
                        {
                            "type": "node_complete",
                            "node": node_name,
                            "label": NODE_LABELS.get(node_name, node_name),
                            "state": state_snapshot,
                        }
                    )

                    # Let the UI know what's running next
                    next_node = NEXT_NODE.get(node_name)
                    if next_node:
                        # For auditor: check is_hallucinating to decide real next node
                        if node_name == "auditor_agent":
                            if state_update.get("is_hallucinating"):
                                next_node = "investment_analyst_agent"
                            else:
                                next_node = "report_generator_agent"

                        yield sse(
                            {
                                "type": "node_start",
                                "node": next_node,
                                "label": NODE_LABELS.get(next_node, next_node),
                            }
                        )

            # Exit only after graph completed and no pending live logs remain.
            if graph_done and log_queue.empty():
                yield sse({"type": "complete"})
                break

            await asyncio.sleep(0)

    finally:
        if not graph_task.done():
            graph_task.cancel()


# ── API endpoints ──────────────────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    """
    Start the wealth-manager pipeline.
    Returns a streaming SSE response so clients can track each agent step.
    """
    return StreamingResponse(
        stream_workflow(request.message, request.tickers),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx proxy buffering
            "Connection": "keep-alive",
        },
    )


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "WealthMind AI"}


@app.get("/api/best-workflow")
async def best_workflow():
    """
    Returns the recommended workflow architecture derived from the
    experiment benchmarks in experiments/architecture_workflow/results.json.

    Selection criteria (in priority order):
      1. Lowest node_failure_rate
      2. Highest average judge quality score
      3. Lowest average latency (as tiebreaker)
    """
    import json as _json
    from pathlib import Path as _Path
    from collections import defaultdict as _dd

    results_path = (
        _Path(ROOT) / "experiments" / "architecture_workflow" / "results.json"
    )
    try:
        with open(results_path) as f:
            runs = _json.load(f)
    except FileNotFoundError:
        return {
            "workflow": "sequential",
            "reason": "results.json not found — using default",
        }

    by_wf = _dd(list)
    for r in runs:
        by_wf[r["workflow"]].append(r)

    def avg(vals):
        return sum(vals) / len(vals) if vals else 0.0

    summary = {}
    for wf, wf_runs in by_wf.items():
        quality = avg(
            [
                avg(list(r["judge_scores"].values()))
                for r in wf_runs
                if r.get("judge_scores")
            ]
        )
        summary[wf] = {
            "fail_rate": avg([r["node_failure_rate"] for r in wf_runs]),
            "quality": round(quality, 4),
            "latency_ms": round(avg([r["total_latency_ms"] for r in wf_runs]), 0),
            "tokens": round(avg([r["total_tokens"] for r in wf_runs]), 0),
        }

    best = min(
        summary.items(),
        key=lambda x: (x[1]["fail_rate"], -x[1]["quality"], x[1]["latency_ms"]),
    )
    return {
        "workflow": best[0],
        "metrics": best[1],
        "all": summary,
    }
