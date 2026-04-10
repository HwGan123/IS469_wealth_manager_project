"""
workflow_experiment.py
======================
Experiment harness that measures and compares three agentic workflow
architectures defined in graph/architecture_experiment/.

  sequential           → sequential_workflow.py
  conditional          → conditional_workflow.py
  orchestration_worker → orchestrator_workflow.py

Metrics collected per run
--------------------------
1. End-to-end latency (ms)
2. Total token count (input + output) across all LLM calls
3. Average input / output tokens per node
4. LLM-judged quality: correctness, relevance, groundedness,
   guideline adherence, safety  (each 0–1)
5. Node failure rate + MCP tool-error count

How to run
----------
    # From the project root:
    python -m experiments.architecture_workflow.workflow_experiment

    # Or from the file's own directory:
    cd experiments/architecture_workflow
    python workflow_experiment.py

    # Common flags:
    python -m experiments.architecture_workflow.workflow_experiment --save
    python -m experiments.architecture_workflow.workflow_experiment \\
        --workflows sequential conditional \\
        --queries "Analyze AAPL" "Is NVDA a buy?"
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  TOKEN TRACKING
#     Monkey-patches OpenAI and Anthropic at the API level BEFORE any agent
#     module is imported, so every LLM call is captured automatically.
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        return self

    def copy(self) -> "TokenUsage":
        return TokenUsage(self.input_tokens, self.output_tokens)


class _TokenCollector:
    """Per-node token accumulator.  Reset before each workflow run."""

    def __init__(self) -> None:
        self._buckets: Dict[str, TokenUsage] = {}
        self._current_node: str = "__global__"

    def reset(self) -> None:
        self._buckets.clear()
        self._current_node = "__global__"

    def set_node(self, name: str) -> None:
        self._current_node = name
        self._buckets.setdefault(name, TokenUsage())

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self._buckets.setdefault(self._current_node, TokenUsage())
        self._buckets[self._current_node].input_tokens += input_tokens
        self._buckets[self._current_node].output_tokens += output_tokens

    def snapshot_node(self, name: str) -> TokenUsage:
        return self._buckets.get(name, TokenUsage()).copy()

    @property
    def total(self) -> TokenUsage:
        t = TokenUsage()
        for u in self._buckets.values():
            t += u
        return t


TOKEN_COLLECTOR = _TokenCollector()


def _patch_openai() -> None:
    try:
        import openai.resources.chat.completions as _mod

        _orig = _mod.Completions.create

        def _patched(self, *args, **kwargs):
            resp = _orig(self, *args, **kwargs)
            if hasattr(resp, "usage") and resp.usage:
                TOKEN_COLLECTOR.add(
                    resp.usage.prompt_tokens, resp.usage.completion_tokens
                )
            return resp

        _mod.Completions.create = _patched
    except Exception as exc:
        print(f"[warn] Could not patch OpenAI: {exc}")


def _patch_anthropic() -> None:
    try:
        import anthropic.resources.messages as _mod

        _orig = _mod.Messages.create

        def _patched(self, *args, **kwargs):
            resp = _orig(self, *args, **kwargs)
            if hasattr(resp, "usage") and resp.usage:
                TOKEN_COLLECTOR.add(resp.usage.input_tokens, resp.usage.output_tokens)
            return resp

        _mod.Messages.create = _patched
    except Exception as exc:
        print(f"[warn] Could not patch Anthropic: {exc}")


_patch_openai()
_patch_anthropic()


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  WORKFLOW IMPORTS  (from graph/architecture_experiment/)
#     All graph factories, state factories, and node registries live in the
#     workflow files.  This file only orchestrates measurement.
# ═══════════════════════════════════════════════════════════════════════════════

_IMPORT_ERRORS: Dict[str, str] = {}


def _safe_import(module: str, symbol: str, default=None):
    """
    Import ``symbol`` from ``module``.
    On failure: records the error, returns ``default`` if provided, otherwise
    returns a stub callable that raises RuntimeError when called.
    """
    try:
        import importlib

        return getattr(importlib.import_module(module), symbol)
    except Exception as exc:
        _IMPORT_ERRORS[symbol] = str(exc)
        print(f"[warn] Could not import {symbol} from {module}: {exc}")
        if default is not None:
            return default

        err_msg = str(exc)  # capture before Python deletes `exc` at end of except block

        def _stub(*_a, **_kw):  # noqa: accept any call signature
            raise RuntimeError(f"'{symbol}' unavailable (import failed: {err_msg})")

        _stub.__name__ = symbol
        return _stub


_SEQ = "graph.architecture_experiment.sequential_workflow"
_COND = "graph.architecture_experiment.conditional_workflow"
_ORCH = "graph.architecture_experiment.orchestrator_workflow"

# Graph factories
create_sequential_graph = _safe_import(_SEQ, "create_sequential_graph")
create_conditional_graph = _safe_import(_COND, "create_conditional_graph")
create_orchestrator_graph = _safe_import(_ORCH, "create_orchestrator_graph")

# State factories
make_sequential_state = _safe_import(_SEQ, "make_initial_state")
make_conditional_state = _safe_import(_COND, "make_initial_state")
make_orchestrator_state = _safe_import(_ORCH, "make_initial_state")

# Node registries  {node_name: callable}  — single source of truth per workflow
SEQUENTIAL_NODES = _safe_import(_SEQ, "SEQUENTIAL_NODES", default={})
CONDITIONAL_NODES = _safe_import(_COND, "CONDITIONAL_NODES", default={})
ORCHESTRATOR_NODES = _safe_import(_ORCH, "ORCHESTRATOR_NODES", default={})

# Ticker extraction (regex-only, zero LLM calls, zero extra tokens)
_extract_tickers = _safe_import("agents.orchestrator", "_extract_tickers_regex")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  METRICS DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class NodeResult:
    name: str
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    success: bool = True
    error: Optional[str] = None
    tool_calls: int = 0
    tool_errors: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class RunResult:
    workflow: str
    query: str
    total_latency_ms: float = 0.0
    nodes: List[NodeResult] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    judge_scores: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    final_report: str = ""
    retrieved_context: str = ""
    market_context_skipped: bool = True
    orchestrator_iterations: int = 0
    audit_retries: int = 0
    trace: List = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def node_failure_rate(self) -> float:
        return (
            sum(1 for n in self.nodes if not n.success) / len(self.nodes)
            if self.nodes
            else 0.0
        )

    @property
    def tool_error_count(self) -> int:
        return sum(n.tool_errors for n in self.nodes)

    @property
    def avg_input_tokens_per_node(self) -> float:
        return self.total_input_tokens / len(self.nodes) if self.nodes else 0.0

    @property
    def avg_output_tokens_per_node(self) -> float:
        return self.total_output_tokens / len(self.nodes) if self.nodes else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  INSTRUMENTATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _count_tool_errors(node_name: str, state_updates: dict) -> Tuple[int, int]:
    """Return (tool_calls, tool_errors) for the market_context node."""
    if node_name != "market_context_agent":
        return 0, 0
    ctx = state_updates.get("market_context", {})
    tools = ctx.get("_tools_used", [])
    errors = sum(1 for t in tools if isinstance(ctx.get(t), dict) and "error" in ctx[t])
    return len(tools), errors


def _run_node(name: str, fn: Callable, state: dict) -> Tuple[dict, NodeResult]:
    """Run one agent node; capture timing, tokens, and errors."""
    TOKEN_COLLECTOR.set_node(name)
    before = TOKEN_COLLECTOR.snapshot_node(name)
    t0 = time.perf_counter()
    nr = NodeResult(name=name)
    updates: dict = {}

    try:
        updates = fn(state) or {}
    except Exception as exc:
        nr.success = False
        nr.error = traceback.format_exc(limit=4)
        print(f"  [ERROR] {name}: {exc}")

    nr.latency_ms = (time.perf_counter() - t0) * 1000
    after = TOKEN_COLLECTOR.snapshot_node(name)
    nr.input_tokens = after.input_tokens - before.input_tokens
    nr.output_tokens = after.output_tokens - before.output_tokens
    nr.tool_calls, nr.tool_errors = _count_tool_errors(name, updates)
    return updates, nr


def _make_wrappers(
    node_map: Dict[str, Callable],
    node_results: List[NodeResult],
) -> Dict[str, Callable]:
    """Wrap every node in node_map with timing + token tracking."""
    _SNAP_KEYS = (
        "tickers",
        "draft_report",
        "sentiment_summary",
        "audit_score",
        "is_hallucinating",
        "market_context",
    )

    def _wrap(name: str, fn: Callable) -> Callable:
        def _instrumented(state: dict) -> dict:
            updates, nr = _run_node(name, fn, state)
            node_results.append(nr)
            # Inject a trace entry unless the node already wrote one (e.g. orchestrator)
            if "trace" not in updates:
                in_snap = {
                    k: state.get(k)
                    for k in _SNAP_KEYS
                    if state.get(k) not in (None, {}, [], "")
                }
                out_snap = {
                    k: ("<long-text>" if isinstance(v, str) and len(v) > 300 else v)
                    for k, v in updates.items()
                    if k != "trace"
                }
                updates["trace"] = [
                    {
                        "agent": name,
                        "input_snapshot": in_snap,
                        "output_snapshot": out_snap,
                    }
                ]
            return updates

        _instrumented.__name__ = name
        return _instrumented

    return {n: _wrap(n, fn) for n, fn in node_map.items()}


def _invoke_graph(app, initial_state: dict, run: RunResult) -> dict:
    """Invoke a compiled LangGraph app; capture errors into run."""
    try:
        return app.invoke(initial_state)
    except Exception as exc:
        run.success = False
        run.error = traceback.format_exc(limit=4)
        print(f"  [ERROR] graph invoke: {exc}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  WORKFLOW RUNNERS
#     A single generic helper drives all three; each public function is a
#     one-liner that passes the right graph/state/nodes from the imports above.
# ═══════════════════════════════════════════════════════════════════════════════


def _run_workflow(
    name: str,
    create_graph_fn: Callable,
    make_state_fn: Callable,
    node_map: Dict[str, Callable],
    query: str,
) -> RunResult:
    TOKEN_COLLECTOR.reset()
    run = RunResult(workflow=name, query=query)
    node_results: List[NodeResult] = []
    tickers = _extract_tickers(query)

    t0 = time.perf_counter()
    try:
        app = create_graph_fn(node_overrides=_make_wrappers(node_map, node_results))
        final_state = _invoke_graph(app, make_state_fn(query, tickers), run)
    except Exception as exc:
        run.success = False
        run.error = str(exc)
        print(f"  [ERROR] '{name}' workflow unavailable: {exc}")
        final_state = {}

    run.total_latency_ms = (time.perf_counter() - t0) * 1000
    run.nodes = node_results
    run.success = run.success and all(n.success for n in node_results)
    totals = TOKEN_COLLECTOR.total
    run.total_input_tokens = totals.input_tokens
    run.total_output_tokens = totals.output_tokens
    run.final_report = final_state.get("final_report", "")
    run.retrieved_context = final_state.get("retrieved_context", "")
    run.market_context_skipped = not bool(final_state.get("market_context"))
    run.orchestrator_iterations = final_state.get("orchestrator_iteration", 0)
    run.audit_retries = final_state.get("audit_iteration_count", 0)
    run.trace = final_state.get("trace", [])
    return run


def run_sequential(query: str) -> RunResult:
    """Sequential: market_context → sentiment → analyst → auditor → reporter."""
    return _run_workflow(
        "sequential",
        create_sequential_graph,
        make_sequential_state,
        SEQUENTIAL_NODES,
        query,
    )


def run_conditional(query: str) -> RunResult:
    """Conditional: router decides live-data path; auditor loop returns to entry."""
    return _run_workflow(
        "conditional",
        create_conditional_graph,
        make_conditional_state,
        CONDITIONAL_NODES,
        query,
    )


def run_orchestration_worker(query: str) -> RunResult:
    """Orchestration-worker: LLM orchestrator dispatches workers dynamically."""
    return _run_workflow(
        "orchestration_worker",
        create_orchestrator_graph,
        make_orchestrator_state,
        ORCHESTRATOR_NODES,
        query,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  LLM JUDGE  (tokens NOT counted in workflow metrics — call after capturing)
# ═══════════════════════════════════════════════════════════════════════════════

_JUDGE_SYSTEM = textwrap.dedent("""
    You are a strict evaluator of AI-generated financial reports.
    Score the report on these five dimensions (0-100 integer each):
      correctness        – internally consistent facts; no contradictions
      relevance          – directly addresses the user query
      groundedness       – claims justified by the provided context
      guideline_adherence – follows standard report structure
      safety             – no harmful advice or regulatory violations
    Return ONLY a JSON object, no extra text.
    Example: {{"correctness":85,"relevance":90,"groundedness":78,
              "guideline_adherence":92,"safety":100}}
""").strip()

_JUDGE_KEYS = [
    "correctness",
    "relevance",
    "groundedness",
    "guideline_adherence",
    "safety",
]


def llm_judge(query: str, report: str, context: str = "") -> Dict[str, float]:
    """Score a report on five quality dimensions (0–1 each) using GPT-4o-mini."""
    zero = {k: 0.0 for k in _JUDGE_KEYS}
    if not report.strip():
        return zero
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _JUDGE_SYSTEM),
                (
                    "human",
                    "USER QUERY:\n{query}\n\nCONTEXT:\n{context}\n\nREPORT:\n{report}",
                ),
            ]
        )
        resp = (prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)).invoke(
            {
                "query": query,
                "context": context[:3000] or "(none)",
                "report": report[:6000],
            }
        )
        match = re.search(r"\{.*?\}", resp.content, re.DOTALL)
        if match:
            raw: Dict[str, Any] = json.loads(match.group())
            return {k: float(raw.get(k, 0)) / 100.0 for k in _JUDGE_KEYS}
    except Exception as exc:
        print(f"  [judge] failed: {exc}")
    return zero


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_QUERIES: List[str] = [
    "Analyze AAPL for long-term investment potential.",
    "What is the current market sentiment for NVDA?",
    "Provide a comprehensive financial report on TSLA, including risks and recommendations.",
    "Based on AAPL's 10K, what are its main business risks?",
]

WORKFLOW_RUNNERS: Dict[str, Callable[[str], RunResult]] = {
    "sequential": run_sequential,
    "conditional": run_conditional,
    "orchestration_worker": run_orchestration_worker,
}


def run_experiment(
    queries: Optional[List[str]] = None,
    workflows: Optional[List[str]] = None,
    checkpoint_path: Optional[str] = None,
) -> List[RunResult]:
    """Run every (workflow, query) pair; judge each result; return all RunResults.

    If checkpoint_path is given, completed results are saved after each run so
    the experiment can be resumed after a crash (pass --resume to skip already-
    completed pairs).
    """
    queries = queries or DEFAULT_QUERIES
    runners = {
        k: v for k, v in WORKFLOW_RUNNERS.items() if workflows is None or k in workflows
    }

    # ── Load existing checkpoint ───────────────────────────────────────────────
    results: List[RunResult] = []
    done: set = set()          # (workflow, query) pairs already completed
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            with open(checkpoint_path, encoding="utf-8") as fh:
                ckpt = json.load(fh)
            for entry in ckpt:
                r = RunResult(workflow=entry["workflow"], query=entry["query"])
                r.total_latency_ms = entry.get("total_latency_ms", 0.0)
                r.total_input_tokens = entry.get("total_input_tokens", 0)
                r.total_output_tokens = entry.get("total_output_tokens", 0)
                r.node_failure_rate = entry.get("node_failure_rate", 0.0)
                r.tool_error_count = entry.get("tool_error_count", 0)
                r.market_context_skipped = entry.get("market_context_skipped", False)
                r.orchestrator_iterations = entry.get("orchestrator_iterations", 0)
                r.audit_retries = entry.get("audit_retries", 0)
                r.judge_scores = entry.get("judge_scores", {})
                r.success = entry.get("success", False)
                r.error = entry.get("error")
                results.append(r)
                done.add((entry["workflow"], entry["query"]))
            print(f"  ✓ Resumed from checkpoint: {len(done)} run(s) already done")
        except Exception as exc:
            print(f"  ⚠ Could not load checkpoint ({exc}); starting fresh")

    for query in queries:
        print(f"\n{'=' * 72}\nQUERY: {query}\n{'=' * 72}")
        for wf_name, runner in runners.items():
            if (wf_name, query) in done:
                print(f"\n── Workflow: {wf_name.upper()} ── [skipped — already in checkpoint]")
                continue

            print(f"\n── Workflow: {wf_name.upper()} ──")
            result = runner(query)

            print("  Running LLM quality judge ...")
            result.judge_scores = llm_judge(
                query, result.final_report, result.retrieved_context
            )
            results.append(result)
            _print_run_summary(result)

            # ── Checkpoint after every completed run ───────────────────────────
            if checkpoint_path:
                save_results(results, path=checkpoint_path)
                print(f"  ✓ Checkpoint saved ({len(results)} run(s))")

    return results


def _print_run_summary(r: RunResult) -> None:
    status = "OK" if r.success else "FAILED"
    print(
        f"  [{status}] {r.total_latency_ms:.0f} ms  "
        f"tokens={r.total_tokens} (in={r.total_input_tokens}/out={r.total_output_tokens})  "
        f"nodes={len(r.nodes)}  fail={r.node_failure_rate:.2f}"
    )
    if r.nodes:
        seq = " → ".join(n.name for n in r.nodes)
        print(f"  Node sequence: {seq}")
    if r.judge_scores:
        print(
            "  Judge → " + "  ".join(f"{k}={v:.2f}" for k, v in r.judge_scores.items())
        )
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  RESULTS REPORTING
# ═══════════════════════════════════════════════════════════════════════════════


def _avg(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def print_comparison_table(results: List[RunResult]) -> None:
    """Aggregated comparison table (averaged over all queries) + per-query rows."""
    by_wf: Dict[str, List[RunResult]] = defaultdict(list)
    for r in results:
        by_wf[r.workflow].append(r)

    cols = [
        ("Workflow", "<", 25),
        ("N", ">", 3),
        ("Lat(ms)", ">", 10),
        ("Tokens", ">", 8),
        ("In/node", ">", 8),
        ("Out/node", ">", 9),
        ("FailRate", ">", 9),
        ("ToolErr", ">", 8),
        ("Correct", ">", 8),
        ("Relev", ">", 6),
        ("Ground", ">", 7),
        ("Guide", ">", 6),
        ("Safety", ">", 7),
    ]
    fmt = lambda t, a, w: f"{t:{a}{w}}"
    header = "  ".join(fmt(h, a, w) for h, a, w in cols)
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}\nWORKFLOW COMPARISON\n{'=' * len(header)}")
    print(header)
    print(sep)

    for wf, runs in by_wf.items():
        j = lambda k: _avg([r.judge_scores.get(k, 0) for r in runs])
        row = {
            "Workflow": wf,
            "N": str(len(runs)),
            "Lat(ms)": f"{_avg([r.total_latency_ms for r in runs]):.0f}",
            "Tokens": f"{_avg([r.total_tokens for r in runs]):.0f}",
            "In/node": f"{_avg([r.avg_input_tokens_per_node for r in runs]):.0f}",
            "Out/node": f"{_avg([r.avg_output_tokens_per_node for r in runs]):.0f}",
            "FailRate": f"{_avg([r.node_failure_rate for r in runs]):.3f}",
            "ToolErr": f"{_avg([float(r.tool_error_count) for r in runs]):.1f}",
            "Correct": f"{j('correctness'):.3f}",
            "Relev": f"{j('relevance'):.3f}",
            "Ground": f"{j('groundedness'):.3f}",
            "Guide": f"{j('guideline_adherence'):.3f}",
            "Safety": f"{j('safety'):.3f}",
        }
        print("  ".join(fmt(row[h], a, w) for h, a, w in cols))

    print(f"{'=' * len(header)}\n\nPER-QUERY BREAKDOWN\n{sep}")
    for r in results:
        scores = "  ".join(f"{k[:5]}={v:.2f}" for k, v in r.judge_scores.items())
        print(
            f"  {r.workflow:<25}  {r.query[:48]:<50}  "
            f"lat={r.total_latency_ms:>7.0f}ms  tok={r.total_tokens:>6}  {scores}"
        )
        if r.nodes:
            seq = " → ".join(n.name for n in r.nodes)
            print(f"  {'':25}  node sequence: {seq}")


def print_node_sequence_table(results: List[RunResult]) -> None:
    """Per-query table showing node sequence, re-runs, and audit loops per workflow."""
    by_query: Dict[str, List[RunResult]] = defaultdict(list)
    for r in results:
        by_query[r.query].append(r)

    WF_W, SEQ_W, RERUN_W, AUDIT_W = 22, 90, 8, 11

    for query, runs in by_query.items():
        print(f"\nQuery: {query}")
        sep = "-" * (WF_W + SEQ_W + RERUN_W + AUDIT_W + 6)
        print(sep)
        print(
            f"  {'Workflow':<{WF_W}}  {'Node sequence':<{SEQ_W}}  "
            f"{'Re-runs':>{RERUN_W}}  {'Audit loops':>{AUDIT_W}}"
        )
        print(sep)
        for r in runs:
            seq = " → ".join(n.name for n in r.nodes) if r.nodes else "(no nodes ran)"
            # Re-runs: nodes that appeared more than once (loop-backs)
            seen: set = set()
            reruns = 0
            for n in r.nodes:
                if n.name in seen:
                    reruns += 1
                seen.add(n.name)
            rerun_str = str(reruns) if reruns else "none"
            audit_loops = sum(1 for n in r.nodes if n.name == "auditor_agent")
            print(
                f"  {r.workflow:<{WF_W}}  {seq:<{SEQ_W}}  "
                f"{rerun_str:>{RERUN_W}}  {audit_loops:>{AUDIT_W}}"
            )
        print()


def save_results(results: List[RunResult], path: Optional[str] = None) -> str:
    """Serialise all RunResults to JSON; also save per-run trace files."""
    _base = PROJECT_ROOT / "experiments" / "architecture_workflow"
    path = path or str(_base / "results.json")

    data = [
        {
            "workflow": r.workflow,
            "query": r.query,
            "total_latency_ms": round(r.total_latency_ms, 2),
            "total_tokens": r.total_tokens,
            "total_input_tokens": r.total_input_tokens,
            "total_output_tokens": r.total_output_tokens,
            "avg_input_tok_per_node": round(r.avg_input_tokens_per_node, 1),
            "avg_output_tok_per_node": round(r.avg_output_tokens_per_node, 1),
            "node_failure_rate": round(r.node_failure_rate, 4),
            "tool_error_count": r.tool_error_count,
            "market_context_skipped": r.market_context_skipped,
            "orchestrator_iterations": r.orchestrator_iterations,
            "audit_retries": r.audit_retries,
            "judge_scores": {k: round(v, 4) for k, v in r.judge_scores.items()},
            "success": r.success,
            "error": r.error,
            "nodes": [
                {
                    "name": n.name,
                    "latency_ms": round(n.latency_ms, 2),
                    "input_tokens": n.input_tokens,
                    "output_tokens": n.output_tokens,
                    "tool_calls": n.tool_calls,
                    "tool_errors": n.tool_errors,
                    "success": n.success,
                    "error": n.error,
                }
                for n in r.nodes
            ],
        }
        for r in results
    ]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    print(f"\nResults saved → {path}")

    # ── Save per-run traces ────────────────────────────────────────────────────
    traces_dir = _base / "traces"
    traces_dir.mkdir(exist_ok=True)
    # Group by workflow to assign query_index per workflow
    _wf_counter: Dict[str, int] = {}
    for r in results:
        idx = _wf_counter.get(r.workflow, 0)
        _wf_counter[r.workflow] = idx + 1
        trace_path = traces_dir / f"run_{r.workflow}_{idx}.json"
        with open(trace_path, "w", encoding="utf-8") as fh:
            json.dump(
                {"workflow": r.workflow, "query": r.query, "trace": r.trace},
                fh,
                indent=2,
                default=str,
            )
    print(f"Traces saved → {traces_dir}/")

    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Agentic Workflow Architecture Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python -m experiments.architecture_workflow.workflow_experiment
              python -m experiments.architecture_workflow.workflow_experiment --save
              python -m experiments.architecture_workflow.workflow_experiment \\
                  --workflows sequential conditional \\
                  --queries "Analyze AAPL" "Is NVDA a buy?"
        """),
    )
    parser.add_argument("--queries", nargs="*", help="Custom test queries")
    parser.add_argument(
        "--workflows",
        nargs="*",
        choices=list(WORKFLOW_RUNNERS),
        help="Workflows to run (default: all three)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to experiments/architecture_workflow/results.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom JSON output path (implies --save)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (results.json), skipping already-completed runs",
    )
    args = parser.parse_args()

    _base = PROJECT_ROOT / "experiments" / "architecture_workflow"
    checkpoint = str(args.output or _base / "results.json") if (args.save or args.output or args.resume) else None

    results = run_experiment(queries=args.queries, workflows=args.workflows, checkpoint_path=checkpoint)
    print_node_sequence_table(results)
    print_comparison_table(results)

    if args.save or args.output:
        save_results(results, path=args.output)

