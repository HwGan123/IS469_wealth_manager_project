"""
qualitative_experiment.py
=========================
Qualitative comparison of the three workflow architectures.

Complements workflow_experiment.py (quantitative metrics) with:
  1. Execution path analysis   – node sequence, routing decisions, iteration counts
  2. Report structure scoring  – section count, depth, word count, coverage flags
  3. LLM pairwise comparison   – side-by-side report quality judgment (A vs B)
  4. Audit findings diff       – what the auditor flagged per workflow
  5. Routing correctness check – does conditional correctly skip live data for no-ticker queries?

How to run
----------
    # From the project root:
    python -m experiments.architecture_workflow.qualitative_experiment

    # Save results to qualitative_results.json:
    python -m experiments.architecture_workflow.qualitative_experiment --save

    # Subset of workflows or queries:
    python -m experiments.architecture_workflow.qualitative_experiment \\
        --workflows sequential conditional \\
        --queries "Analyze AAPL" "What is dollar-cost averaging?"
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


# ── Re-use token tracking and node machinery from the quantitative harness ────
# Import BEFORE any agent modules so patching takes effect.
from experiments.architecture_workflow.workflow_experiment import (
    TOKEN_COLLECTOR,
    NodeResult,
    RunResult,
    _make_wrappers,
    _invoke_graph,
    _extract_tickers,
    create_sequential_graph,
    create_conditional_graph,
    create_orchestrator_graph,
    make_sequential_state,
    make_conditional_state,
    make_orchestrator_state,
    SEQUENTIAL_NODES,
    CONDITIONAL_NODES,
    ORCHESTRATOR_NODES,
    llm_judge,
    WORKFLOW_RUNNERS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  EXTENDED RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QualRunResult(RunResult):
    """RunResult extended with qualitative fields captured from the final state."""

    # Audit / hallucination detail
    audit_score: float = 0.0
    audit_findings: List[str] = field(default_factory=list)
    is_hallucinating: bool = False
    hallucination_count: int = 0
    verified_count: int = 0
    audit_iteration_count: int = 0

    # Routing / orchestration metadata
    route_target: str = ""          # conditional workflow: which branch was taken
    orchestrator_iteration: int = 0  # orchestrator workflow: total LLM decisions made

    # Sentiment summary
    sentiment_score: float = 0.0
    sentiment_label: str = ""

    # Report structure (populated by analyze_report_structure)
    structure: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  QUALITATIVE WORKFLOW RUNNER
#     Like _run_workflow in the quantitative harness but returns QualRunResult
#     and captures extra fields from the final graph state.
# ═══════════════════════════════════════════════════════════════════════════════

def _run_qual_workflow(
    name: str,
    create_graph_fn: Callable,
    make_state_fn: Callable,
    node_map: Dict[str, Callable],
    query: str,
) -> QualRunResult:
    TOKEN_COLLECTOR.reset()
    run = QualRunResult(workflow=name, query=query)
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

    run.total_latency_ms    = (time.perf_counter() - t0) * 1000
    run.nodes               = node_results
    run.success             = run.success and all(n.success for n in node_results)
    totals                  = TOKEN_COLLECTOR.total
    run.total_input_tokens  = totals.input_tokens
    run.total_output_tokens = totals.output_tokens

    # Standard fields
    run.final_report      = final_state.get("final_report", "")
    run.retrieved_context = final_state.get("retrieved_context", "")

    # Extended qualitative fields
    run.audit_score           = final_state.get("audit_score", 0.0)
    run.audit_findings        = final_state.get("audit_findings", [])
    run.is_hallucinating      = final_state.get("is_hallucinating", False)
    run.hallucination_count   = final_state.get("hallucination_count", 0)
    run.verified_count        = final_state.get("verified_count", 0)
    run.audit_iteration_count = final_state.get("audit_iteration_count", 0)
    run.route_target          = final_state.get("route_target", "")
    run.orchestrator_iteration = final_state.get("orchestrator_iteration", 0)

    sentiment_summary     = final_state.get("sentiment_summary", {})
    run.sentiment_score   = final_state.get("sentiment_score", 0.0)
    run.sentiment_label   = sentiment_summary.get("overall_label", "")

    return run


def run_qual_sequential(query: str) -> QualRunResult:
    return _run_qual_workflow("sequential", create_sequential_graph,
                              make_sequential_state, SEQUENTIAL_NODES, query)

def run_qual_conditional(query: str) -> QualRunResult:
    return _run_qual_workflow("conditional", create_conditional_graph,
                              make_conditional_state, CONDITIONAL_NODES, query)

def run_qual_orchestrator(query: str) -> QualRunResult:
    return _run_qual_workflow("orchestration_worker", create_orchestrator_graph,
                              make_orchestrator_state, ORCHESTRATOR_NODES, query)


QUAL_RUNNERS: Dict[str, Callable[[str], QualRunResult]] = {
    "sequential":           run_qual_sequential,
    "conditional":          run_qual_conditional,
    "orchestration_worker": run_qual_orchestrator,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  REPORT STRUCTURE SCORER  (zero LLM calls — purely regex/heuristic)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_report_structure(report: str) -> Dict[str, Any]:
    """
    Return a dict of structural signals extracted from the final report text.
    No LLM is used — all checks are regex or string operations.
    """
    if not report.strip():
        return {"empty": True}

    lines = report.splitlines()

    # Section headers (markdown # or ##, or ALL-CAPS lines ≥ 4 chars)
    h1 = [l for l in lines if re.match(r"^#\s", l)]
    h2 = [l for l in lines if re.match(r"^##\s", l)]
    allcaps_headers = [l for l in lines if re.match(r"^[A-Z][A-Z\s\-]{3,}$", l.strip())]

    # Bullet / list items
    bullets = [l for l in lines if re.match(r"^\s*[-*•]\s", l)]

    # Numerical figures — dollar amounts, percentages, multipliers
    figures = re.findall(
        r"(\$[\d,]+(?:\.\d+)?[BMK]?|[\d,]+(?:\.\d+)?%|\d+x|\b\d{4}\b)", report
    )

    # Content coverage flags
    text_lower = report.lower()
    has_risk        = bool(re.search(r"\brisk[s]?\b",          text_lower))
    has_recommend   = bool(re.search(r"\brecommend|buy|sell|hold\b", text_lower))
    has_sentiment   = bool(re.search(r"\bsentiment\b",         text_lower))
    has_valuation   = bool(re.search(r"\bp/e|price.to.earnings|valuation\b", text_lower))
    has_financials  = bool(re.search(r"\brevenue|earnings|eps|ebitda\b", text_lower))
    has_conclusion  = bool(re.search(r"\bconclusion|summary|in summary\b", text_lower))

    word_count = len(report.split())

    return {
        "word_count":        word_count,
        "h1_sections":       len(h1),
        "h2_sections":       len(h2),
        "allcaps_headers":   len(allcaps_headers),
        "total_sections":    len(h1) + len(h2) + len(allcaps_headers),
        "bullet_points":     len(bullets),
        "numerical_figures": len(figures),
        "has_risk":          has_risk,
        "has_recommendation": has_recommend,
        "has_sentiment":     has_sentiment,
        "has_valuation":     has_valuation,
        "has_financials":    has_financials,
        "has_conclusion":    has_conclusion,
        "section_headers":   [l.strip() for l in (h1 + h2 + allcaps_headers)][:15],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  LLM PAIRWISE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

_PAIRWISE_SYSTEM = textwrap.dedent("""
    You are an expert evaluator of AI-generated investment research reports.
    You will be given two reports (A and B) produced by different AI workflows
    for the same user query.  Compare them on these dimensions:

    1. investor_utility   – actionability and usefulness to an investor
    2. structural_clarity – organisation, headings, logical flow
    3. risk_coverage      – depth and specificity of risk discussion
    4. sentiment_integration – how well market sentiment is woven into analysis
    5. factual_grounding  – evidence cited; avoidance of unsupported claims

    For each dimension score A and B on 1–5 (5 = excellent).
    Then give an overall_winner ("A", "B", or "tie") and a 2–3 sentence
    reasoning that highlights the single most important difference.

    Return ONLY a JSON object.  No extra text.
    Example:
    {{
      "investor_utility":        {{"A": 4, "B": 3}},
      "structural_clarity":      {{"A": 3, "B": 5}},
      "risk_coverage":           {{"A": 4, "B": 4}},
      "sentiment_integration":   {{"A": 2, "B": 4}},
      "factual_grounding":       {{"A": 5, "B": 3}},
      "overall_winner": "A",
      "reasoning": "Report A …"
    }}
""").strip()


def pairwise_llm_compare(
    query: str,
    report_a: str,
    label_a: str,
    report_b: str,
    label_b: str,
) -> Dict[str, Any]:
    """
    Ask GPT-4o-mini to compare two reports side-by-side.
    Returns a dict with per-dimension scores and overall_winner.
    On failure, returns an empty dict.
    """
    if not report_a.strip() or not report_b.strip():
        return {"error": "one or both reports are empty"}

    empty = {"error": "llm call failed"}
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        human_text = textwrap.dedent(f"""
            USER QUERY: {query}

            === REPORT A ({label_a}) ===
            {report_a[:4000]}

            === REPORT B ({label_b}) ===
            {report_b[:4000]}
        """).strip()

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        resp = llm.invoke([
            SystemMessage(content=_PAIRWISE_SYSTEM),
            HumanMessage(content=human_text),
        ])
        match = re.search(r"\{.*\}", resp.content, re.DOTALL)
        if match:
            result = json.loads(match.group())
            result["label_a"] = label_a
            result["label_b"] = label_b
            return result
    except Exception as exc:
        print(f"  [pairwise] failed: {exc}")
    return empty


def run_all_pairwise(
    query: str,
    results_by_wf: Dict[str, QualRunResult],
) -> List[Dict[str, Any]]:
    """Run all three pairwise comparisons (A vs B, A vs C, B vs C)."""
    wf_names = list(results_by_wf.keys())
    comparisons = []
    for i in range(len(wf_names)):
        for j in range(i + 1, len(wf_names)):
            na, nb = wf_names[i], wf_names[j]
            ra, rb = results_by_wf[na], results_by_wf[nb]
            print(f"  Pairwise: {na} vs {nb} ...")
            cmp = pairwise_llm_compare(
                query,
                ra.final_report, na,
                rb.final_report, nb,
            )
            cmp["query"] = query
            comparisons.append(cmp)
    return comparisons


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  EXECUTION PATH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def execution_path_summary(run: QualRunResult) -> Dict[str, Any]:
    """
    Summarise the execution path for one run.
    Returns node sequence, duplicates (re-runs), and routing metadata.
    """
    node_sequence = [n.name for n in run.nodes]
    node_counts: Dict[str, int] = defaultdict(int)
    for name in node_sequence:
        node_counts[name] += 1

    rerun_nodes = {n: c for n, c in node_counts.items() if c > 1}

    return {
        "workflow":             run.workflow,
        "query":                run.query,
        "node_sequence":        node_sequence,
        "node_counts":          dict(node_counts),
        "rerun_nodes":          rerun_nodes,
        "total_nodes_executed": len(node_sequence),
        "unique_nodes":         len(node_counts),
        "audit_iterations":     run.audit_iteration_count,
        "route_target":         run.route_target,         # conditional only
        "orchestrator_iters":   run.orchestrator_iteration,  # orchestrator only
        "hallucination_loop":   run.audit_iteration_count > 0 and bool(rerun_nodes),
    }


def routing_correctness_check(
    results_by_query: Dict[str, Dict[str, QualRunResult]]
) -> List[Dict[str, Any]]:
    """
    Check whether the conditional workflow correctly skipped market_context_agent
    for queries that contain no stock tickers.
    """
    checks = []
    for query, wf_results in results_by_query.items():
        cond = wf_results.get("conditional")
        if not cond:
            continue

        has_ticker_in_query = bool(_extract_tickers(query))
        mc_ran = any(n.name == "market_context_agent" for n in cond.nodes)
        route = cond.route_target

        # Correct if: ticker present → market_context ran, no ticker → skipped
        if has_ticker_in_query:
            correct = mc_ran
            expected = "market_context_agent should run (ticker present)"
        else:
            correct = not mc_ran
            expected = "market_context_agent should be skipped (no ticker)"

        checks.append({
            "query":             query,
            "has_ticker":        has_ticker_in_query,
            "route_target":      route,
            "market_context_ran": mc_ran,
            "routing_correct":   correct,
            "expected_behavior": expected,
        })
    return checks


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  PRINTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_W = 120  # total print width


def _bar(char: str = "═") -> str:
    return char * _W


def print_execution_paths(
    results_by_query: Dict[str, Dict[str, QualRunResult]]
) -> None:
    print(f"\n{_bar()}\nEXECUTION PATH ANALYSIS\n{_bar()}")

    for query, wf_results in results_by_query.items():
        print(f"\nQuery: {query[:90]}")
        print("-" * _W)

        col_w = 35
        header = f"  {'Workflow':<25}  {'Node sequence':<{col_w*2}}  {'Re-runs':<12}  {'Audit loops'}"
        print(header)

        for wf, run in wf_results.items():
            summary = execution_path_summary(run)
            seq = " → ".join(summary["node_sequence"])
            reruns = ", ".join(f"{n}×{c}" for n, c in summary["rerun_nodes"].items()) or "none"
            print(
                f"  {wf:<25}  {seq:<{col_w*2}}  {reruns:<12}  "
                f"{summary['audit_iterations']}"
            )


def print_routing_check(checks: List[Dict[str, Any]]) -> None:
    print(f"\n{_bar()}\nROUTING CORRECTNESS CHECK (conditional workflow)\n{_bar()}")
    print(f"  {'Query':<55}  {'Ticker?':<8}  {'Route':<28}  {'MC ran?':<8}  {'Correct?'}")
    print(f"  {'-'*55}  {'-'*8}  {'-'*28}  {'-'*8}  {'-'*8}")
    for c in checks:
        print(
            f"  {c['query'][:55]:<55}  "
            f"{'yes' if c['has_ticker'] else 'no':<8}  "
            f"{c['route_target']:<28}  "
            f"{'yes' if c['market_context_ran'] else 'no':<8}  "
            f"{'✓' if c['routing_correct'] else '✗'}"
        )


def print_audit_findings(
    results_by_query: Dict[str, Dict[str, QualRunResult]]
) -> None:
    print(f"\n{_bar()}\nAUDIT FINDINGS DIFF\n{_bar()}")
    for query, wf_results in results_by_query.items():
        print(f"\nQuery: {query[:90]}")
        print("-" * _W)
        for wf, run in wf_results.items():
            status = (
                f"HALLUCINATED (iter={run.audit_iteration_count})"
                if run.is_hallucinating else "PASSED"
            )
            print(f"  [{wf:<25}]  audit={run.audit_score:.2f}  {status}  "
                  f"verified={run.verified_count}  hallucinated={run.hallucination_count}")
            if run.audit_findings:
                for finding in run.audit_findings[:3]:
                    finding_str = str(finding)[:110]
                    print(f"      • {finding_str}")
                if len(run.audit_findings) > 3:
                    print(f"      … +{len(run.audit_findings) - 3} more findings")


def print_report_structure_comparison(
    results_by_query: Dict[str, Dict[str, QualRunResult]]
) -> None:
    print(f"\n{_bar()}\nREPORT STRUCTURE COMPARISON\n{_bar()}")

    struct_cols = [
        ("Workflow", "<", 25),
        ("Words",    ">",  6),
        ("Sections", ">",  8),
        ("Bullets",  ">",  7),
        ("Figures",  ">",  7),
        ("Risk",     ">",  5),
        ("Rec",      ">",  5),
        ("Sent",     ">",  5),
        ("Valuation",">",  9),
        ("Fin",      ">",  4),
        ("Concl",    ">",  6),
    ]
    hdr = "  ".join(f"{h:{a}{w}}" for h, a, w in struct_cols)

    for query, wf_results in results_by_query.items():
        print(f"\nQuery: {query[:90]}")
        print(f"  {hdr}")
        print(f"  {'-' * len(hdr)}")
        for wf, run in wf_results.items():
            s = run.structure
            if not s or s.get("empty"):
                print(f"  {wf:<25}  (no report)")
                continue
            row = {
                "Workflow":  wf,
                "Words":     str(s.get("word_count", 0)),
                "Sections":  str(s.get("total_sections", 0)),
                "Bullets":   str(s.get("bullet_points", 0)),
                "Figures":   str(s.get("numerical_figures", 0)),
                "Risk":      "✓" if s.get("has_risk") else "✗",
                "Rec":       "✓" if s.get("has_recommendation") else "✗",
                "Sent":      "✓" if s.get("has_sentiment") else "✗",
                "Valuation": "✓" if s.get("has_valuation") else "✗",
                "Fin":       "✓" if s.get("has_financials") else "✗",
                "Concl":     "✓" if s.get("has_conclusion") else "✗",
            }
            print("  " + "  ".join(f"{row[h]:{a}{w}}" for h, a, w in struct_cols))


def print_pairwise_results(all_comparisons: List[Dict[str, Any]]) -> None:
    print(f"\n{_bar()}\nLLM PAIRWISE COMPARISON RESULTS\n{_bar()}")

    dimensions = [
        "investor_utility", "structural_clarity",
        "risk_coverage", "sentiment_integration", "factual_grounding",
    ]

    for cmp in all_comparisons:
        if "error" in cmp:
            print(f"\n  [{cmp.get('label_a','?')} vs {cmp.get('label_b','?')}] ERROR: {cmp['error']}")
            continue

        la, lb = cmp.get("label_a", "A"), cmp.get("label_b", "B")
        print(f"\n  {la}  vs  {lb}  |  Query: {cmp.get('query','')[:70]}")
        print(f"  {'-'*90}")

        # Per-dimension scores
        for dim in dimensions:
            scores = cmp.get(dim, {})
            sa, sb = scores.get("A", scores.get(la, "?")), scores.get("B", scores.get(lb, "?"))
            winner = la if (isinstance(sa, (int, float)) and isinstance(sb, (int, float)) and sa > sb) else \
                     lb if (isinstance(sa, (int, float)) and isinstance(sb, (int, float)) and sb > sa) else "tie"
            print(f"    {dim:<26}  {la}={sa}  {lb}={sb}  → {winner}")

        print(f"\n  Overall winner: {cmp.get('overall_winner', '?')}")
        print(f"  Reasoning: {cmp.get('reasoning', '(none)')}")


def print_side_by_side_reports(
    results_by_query: Dict[str, Dict[str, QualRunResult]],
    max_chars: int = 800,
) -> None:
    """Print the opening of each final report side-by-side for manual inspection."""
    print(f"\n{_bar()}\nFINAL REPORT EXCERPTS (first {max_chars} chars)\n{_bar()}")
    for query, wf_results in results_by_query.items():
        print(f"\nQuery: {query[:90]}")
        for wf, run in wf_results.items():
            excerpt = run.final_report[:max_chars].replace("\n", "\n    ") if run.final_report else "(empty)"
            print(f"\n  ── {wf.upper()} ──")
            print(f"    {excerpt}")
            if len(run.final_report) > max_chars:
                print(f"    … [{len(run.final_report) - max_chars} chars omitted]")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

# Curated query set designed to expose qualitative differences:
#   - Ticker query       → all workflows fetch live data
#   - No-ticker query    → conditional should skip market_context; sequential won't
#   - Sentiment query    → tests how each workflow prioritises the sentiment node
#   - Multi-ticker query → tests orchestrator's adaptive dispatch

DEFAULT_QUAL_QUERIES: List[str] = [
    "Analyze AAPL for long-term investment potential.",
    "What is dollar-cost averaging and when should I use it?",
    "What is the current market sentiment for NVDA?",
    "Compare MSFT and GOOG — which is a better buy right now?",
]


def run_qualitative_experiment(
    queries: Optional[List[str]] = None,
    workflows: Optional[List[str]] = None,
    skip_pairwise: bool = False,
) -> Tuple[Dict[str, Dict[str, QualRunResult]], List[Dict[str, Any]]]:
    """
    Run every (workflow, query) pair with rich state capture.

    Returns
    -------
    results_by_query : {query: {workflow: QualRunResult}}
    all_comparisons  : flat list of pairwise comparison dicts
    """
    queries = queries or DEFAULT_QUAL_QUERIES
    runners = {k: v for k, v in QUAL_RUNNERS.items()
               if workflows is None or k in workflows}

    results_by_query: Dict[str, Dict[str, QualRunResult]] = {}
    all_comparisons: List[Dict[str, Any]] = []

    for query in queries:
        print(f"\n{'='*_W}\nQUERY: {query}\n{'='*_W}")
        results_by_query[query] = {}

        for wf_name, runner in runners.items():
            print(f"\n── Workflow: {wf_name.upper()} ──")
            run = runner(query)

            print("  Running LLM quality judge ...")
            run.judge_scores = llm_judge(query, run.final_report, run.retrieved_context)

            print("  Scoring report structure ...")
            run.structure = analyze_report_structure(run.final_report)

            results_by_query[query][wf_name] = run

            status = "OK" if run.success else "FAILED"
            print(f"  [{status}] {run.total_latency_ms:.0f} ms  "
                  f"tokens={run.total_tokens}  nodes={len(run.nodes)}  "
                  f"audit={run.audit_score:.2f}  hallucinating={run.is_hallucinating}")

        if not skip_pairwise and len(runners) >= 2:
            print(f"\n── Pairwise LLM comparisons for this query ──")
            comparisons = run_all_pairwise(query, results_by_query[query])
            all_comparisons.extend(comparisons)

    return results_by_query, all_comparisons


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def save_qualitative_results(
    results_by_query: Dict[str, Dict[str, QualRunResult]],
    all_comparisons: List[Dict[str, Any]],
    path: Optional[str] = None,
) -> str:
    path = path or str(
        PROJECT_ROOT / "experiments" / "architecture_workflow" / "qualitative_results.json"
    )

    output: Dict[str, Any] = {
        "runs": [],
        "pairwise_comparisons": all_comparisons,
        "routing_checks": routing_correctness_check(results_by_query),
    }

    for query, wf_results in results_by_query.items():
        for wf, run in wf_results.items():
            output["runs"].append({
                "workflow":             run.workflow,
                "query":                run.query,
                "success":              run.success,
                "error":                run.error,
                # Performance
                "total_latency_ms":     round(run.total_latency_ms, 2),
                "total_tokens":         run.total_tokens,
                # Quality scores
                "judge_scores":         {k: round(v, 4) for k, v in run.judge_scores.items()},
                # Qualitative fields
                "audit_score":          round(run.audit_score, 4),
                "audit_findings":       run.audit_findings,
                "is_hallucinating":     run.is_hallucinating,
                "hallucination_count":  run.hallucination_count,
                "verified_count":       run.verified_count,
                "audit_iteration_count": run.audit_iteration_count,
                "route_target":         run.route_target,
                "orchestrator_iteration": run.orchestrator_iteration,
                "sentiment_score":      run.sentiment_score,
                "sentiment_label":      run.sentiment_label,
                # Report content
                "final_report":         run.final_report,
                "report_structure":     run.structure,
                # Execution path
                "execution_path": execution_path_summary(run),
                # Node details
                "nodes": [
                    {
                        "name":          n.name,
                        "latency_ms":    round(n.latency_ms, 2),
                        "input_tokens":  n.input_tokens,
                        "output_tokens": n.output_tokens,
                        "tool_calls":    n.tool_calls,
                        "tool_errors":   n.tool_errors,
                        "success":       n.success,
                        "error":         n.error,
                    }
                    for n in run.nodes
                ],
            })

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"\nQualitative results saved → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Qualitative Workflow Architecture Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python -m experiments.architecture_workflow.qualitative_experiment
              python -m experiments.architecture_workflow.qualitative_experiment --save
              python -m experiments.architecture_workflow.qualitative_experiment \\
                  --workflows sequential conditional \\
                  --no-pairwise
        """),
    )
    parser.add_argument("--queries",      nargs="*", help="Custom test queries")
    parser.add_argument("--workflows",    nargs="*",
                        choices=list(QUAL_RUNNERS),
                        help="Workflows to run (default: all three)")
    parser.add_argument("--no-pairwise", action="store_true",
                        help="Skip LLM pairwise comparison (faster, cheaper)")
    parser.add_argument("--save",         action="store_true",
                        help="Save results to qualitative_results.json")
    parser.add_argument("--output",       type=str, default=None,
                        help="Custom JSON output path (implies --save)")
    parser.add_argument("--no-excerpts",  action="store_true",
                        help="Skip printing full report excerpts")
    args = parser.parse_args()

    results_by_query, all_comparisons = run_qualitative_experiment(
        queries=args.queries,
        workflows=args.workflows,
        skip_pairwise=args.no_pairwise,
    )

    # ── Print all qualitative analyses ───────────────────────────────────────
    print_execution_paths(results_by_query)
    print_routing_check(routing_correctness_check(results_by_query))
    print_audit_findings(results_by_query)
    print_report_structure_comparison(results_by_query)

    if not args.no_pairwise and all_comparisons:
        print_pairwise_results(all_comparisons)

    if not args.no_excerpts:
        print_side_by_side_reports(results_by_query)

    if args.save or args.output:
        save_qualitative_results(results_by_query, all_comparisons, path=args.output)
