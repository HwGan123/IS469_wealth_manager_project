"""
WealthMind AI — FastAPI backend
Wraps the LangGraph multi-agent workflow and streams agent progress
via Server-Sent Events (SSE).

Run from the project root:
    uvicorn backend.server:app --reload --port 8000
"""

import asyncio
import io
import json
import os
import re
import sys

# Load .env BEFORE any project imports so model paths / API keys are available
from dotenv import load_dotenv
load_dotenv()

# ── Make sure project root is on sys.path so agents/* can be imported ─────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List

FRONTEND_DIR = os.path.join(ROOT, "frontend")

from graph.architecture_experiment.conditional_workflow import (
    create_conditional_graph,
    make_initial_state,
)
from graph.state import WealthManagerState

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
    "audit_iteration_count",
    "ragas_metrics",
    "draft_report",
    "final_report",
    "live_data_context",
    "route_target",
    "portfolio_weights",
}

# Node → friendly display name for the step indicator
NODE_LABELS = {
    "conditional_router": "Router",
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


# ── Print capture — forwards print() output to the SSE log queue ──────────────
class _PrintCapture(io.TextIOBase):
    """
    Replaces sys.stdout for the duration of a workflow run.
    Each complete line is:
      1. Still written to the real terminal.
      2. Forwarded to log_queue as an agent_log item so the frontend shows it.

    Node routing: each agent opens with a "--- AGENT: XXX ---" header that we
    parse to keep track of which panel the subsequent lines belong to.
    """

    _AGENT_HEADER = re.compile(
        r"-{2,}\s*(?:[\U0001F300-\U0001FFFF\u2600-\u27BF]\s*)*AGENT:\s*(.+?)\s*(?:\(.*\))?\s*-{2,}",
        re.IGNORECASE,
    )
    _KEYWORD_TO_NODE = {
        "ORCHESTRATOR": "orchestrator_agent",
        "MARKET CONTEXT": "market_context_agent",
        "MARKET": "market_context_agent",
        "SENTIMENT": "sentiment_agent",
        "INVESTMENT ANALYST": "investment_analyst_agent",
        "ANALYST": "investment_analyst_agent",
        "AUDITOR": "auditor_agent",
        "REPORT GENERATOR": "report_generator_agent",
        "PORTFOLIO": "orchestrator_agent",
    }

    def __init__(
        self,
        log_queue: asyncio.Queue,
        sse_loop: asyncio.AbstractEventLoop,
        real_stdout,
    ):
        self._queue = log_queue
        self._loop = sse_loop
        self._real = real_stdout
        self._buf = ""
        self._node = "orchestrator_agent"

    # TextIOBase interface -------------------------------------------------
    def write(self, text: str) -> int:
        if not isinstance(text, str):
            text = str(text)
        self._real.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line)
        return len(text)

    def flush(self):
        self._real.flush()

    @property
    def encoding(self):
        return getattr(self._real, "encoding", "utf-8")

    # Internal -------------------------------------------------------------
    def _emit(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return

        # Detect agent header → update active node
        m = self._AGENT_HEADER.search(stripped)
        if m:
            label = m.group(1).upper()
            for key, node in self._KEYWORD_TO_NODE.items():
                if key in label:
                    self._node = node
                    break

        item = {"node": self._node, "message": line.rstrip()}
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except Exception:
            try:
                self._queue.put_nowait(item)
            except Exception:
                pass


# ── Core streaming generator ───────────────────────────────────────────────────
async def stream_workflow(message: str, tickers: List[str]):
    """
    Runs the LangGraph workflow and yields SSE events:
      - node_start   : a node just became active
      - node_complete: a node finished (with partial state)
      - complete     : entire pipeline done
      - error        : something went wrong
    """
    graph = create_conditional_graph(state_schema=WealthManagerState)
    sse_loop = asyncio.get_running_loop()

    log_queue: asyncio.Queue = asyncio.Queue()
    graph_queue: asyncio.Queue = asyncio.Queue()

    initial_state = make_initial_state(message, tickers)
    initial_state["__sse_log_queue"] = log_queue
    initial_state["__sse_loop"] = sse_loop

    # Signal that the first node is about to start
    yield sse(
        {
            "type": "node_start",
            "node": "conditional_router",
            "label": NODE_LABELS["conditional_router"],
        }
    )

    # Track the route chosen by conditional_router so next-node prediction is accurate
    _route_target: str = ""

    def _predict_next_node(node_name: str, state_update: dict) -> str | None:
        """Predict which node runs next based on conditional routing rules."""
        if node_name == "conditional_router":
            return (
                "investment_analyst_agent"
                if _route_target == "docs_only"
                else "market_context_agent"
            )
        if node_name == "market_context_agent":
            return (
                "investment_analyst_agent"
                if _route_target == "hybrid_analysis"
                else "sentiment_agent"
            )
        if node_name == "sentiment_agent":
            return (
                "investment_analyst_agent"
                if _route_target == "market_analysis"
                else "auditor_agent"
            )
        if node_name == "investment_analyst_agent":
            return "auditor_agent"
        if node_name == "auditor_agent":
            if state_update.get("is_hallucinating"):
                if _route_target == "docs_only":
                    return "investment_analyst_agent"
                if _route_target == "market_only":
                    return "sentiment_agent"
                return "market_context_agent"
            return "report_generator_agent"
        return None

    async def _run_graph() -> None:
        try:
            async for event in graph.astream(initial_state):
                await graph_queue.put({"kind": "graph", "event": event})
            await graph_queue.put({"kind": "complete"})
        except Exception as exc:  # pragma: no cover
            await graph_queue.put({"kind": "error", "message": str(exc)})

    # ── Redirect print() → log_queue for the duration of this run ────────────
    _real_stdout = sys.stdout
    sys.stdout = _PrintCapture(log_queue, sse_loop, _real_stdout)

    graph_task = asyncio.create_task(_run_graph())

    try:
        graph_done = False
        _last_log_key: tuple | None = None  # dedup consecutive identical lines
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
                    key = (log_item.get("node"), log_item.get("message"))
                    if key != _last_log_key:  # skip exact back-to-back duplicates
                        _last_log_key = key
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
                    # Track route_target set by conditional_router
                    if state_update.get("route_target"):
                        _route_target = state_update["route_target"]

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

                    # Let the UI know what's running next (conditional prediction)
                    next_node = _predict_next_node(node_name, state_update)
                    if next_node:
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
        sys.stdout = _real_stdout  # always restore, even on error
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


@app.get("/")
async def serve_frontend():
    """Serve the frontend single-page application."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


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
