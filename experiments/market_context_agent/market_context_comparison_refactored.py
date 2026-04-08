"""
Market Context Agent Comparison Experiment (Refactored)

Compares two workflow variants:
A: Normal flow WITH market context agent
B: Workflow WITHOUT market context agent intervention

Uses LLM as a Judge (5 dimensions) to evaluate report quality at the end.
Tracks efficiency metrics (latency, tokens, cost) and quality metrics.
"""

import os
import sys
import json
import time
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

workspace_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(workspace_root))
load_dotenv(workspace_root / ".env")

from graph.state import WealthManagerState
from graph.workflow import create_wealth_manager_graph
from langgraph.graph import StateGraph, START, END
from agents.orchestrator import orchestrator_node
from agents.market_context import market_context_node
from agents.sentiment import sentiment_node
from agents.analyst import analyst_node
from agents.auditor import auditor_node
from agents.report_generator import report_generator_node
from mcp_news import get_mcp_tools, dispatch_mcp_tool
import anthropic
import logging

logger = logging.getLogger(__name__)

# Pricing for Claude 3.5 Haiku (per 1M tokens)
PRICING = {
    "input_cost_per_1m": 1.00,      # $1.00 per 1M input tokens
    "output_cost_per_1m": 5.00,     # $5.00 per 1M output tokens
}


# ═══════════════════════════════════════════════════════════════════════════
# WORKFLOW CREATION
# ═══════════════════════════════════════════════════════════════════════════

def create_workflow_with_market_context():
    """
    VARIANT A: Normal workflow WITH market context agent.
    Includes market context as the first node after orchestrator.
    """
    workflow = StateGraph(WealthManagerState)
    
    workflow.add_node("orchestrator_agent", orchestrator_node)
    workflow.add_node("market_context_agent", market_context_node)
    workflow.add_node("sentiment_agent", sentiment_node)
    workflow.add_node("investment_analyst_agent", analyst_node)
    workflow.add_node("auditor_agent", auditor_node)
    workflow.add_node("report_generator_agent", report_generator_node)
    
    # Orchestrator -> Market Context -> Sentiment -> Analyst -> Auditor -> Report
    workflow.add_edge(START, "orchestrator_agent")
    workflow.add_edge("orchestrator_agent", "market_context_agent")
    workflow.add_edge("market_context_agent", "sentiment_agent")
    workflow.add_edge("sentiment_agent", "investment_analyst_agent")
    workflow.add_edge("investment_analyst_agent", "auditor_agent")
    
    def audit_route(state):
        if state.get("is_hallucinating", False) and state.get("audit_iteration_count", 0) < 1:
            return "investment_analyst_agent"
        return "report_generator_agent"
    
    workflow.add_conditional_edges("auditor_agent", audit_route)
    workflow.add_edge("report_generator_agent", END)
    
    return workflow.compile()


def create_workflow_without_market_context():
    """
    VARIANT B: Agents on their own WITHOUT market context agent intervention.
    Skips market context agent entirely, goes directly to orchestrator.
    """
    workflow = StateGraph(WealthManagerState)
    
    workflow.add_node("orchestrator_agent", orchestrator_node)
    workflow.add_node("sentiment_agent", sentiment_node)
    workflow.add_node("investment_analyst_agent", analyst_node)
    workflow.add_node("auditor_agent", auditor_node)
    workflow.add_node("report_generator_agent", report_generator_node)
    
    # No market context - start directly with orchestrator
    workflow.add_edge(START, "orchestrator_agent")
    workflow.add_edge("orchestrator_agent", "sentiment_agent")
    workflow.add_edge("sentiment_agent", "investment_analyst_agent")
    workflow.add_edge("investment_analyst_agent", "auditor_agent")
    
    def audit_route(state):
        if state.get("is_hallucinating", False) and state.get("audit_iteration_count", 0) < 1:
            return "investment_analyst_agent"
        return "report_generator_agent"
    
    workflow.add_conditional_edges("auditor_agent", audit_route)
    workflow.add_edge("report_generator_agent", END)
    
    return workflow.compile()


def create_initial_state(user_message: str, tickers: list) -> dict:
    """Create initial workflow state."""
    return {
        "messages": [user_message],
        "tickers": tickers,
        "route_target": "",
        "market_context": {},
        "news_articles": [],
        "sentiment_results": [],
        "sentiment_summary": {},
        "sentiment_score": 0.0,
        "portfolio_weights": {},
        "retrieved_context": "",
        "live_data_context": "",
        "draft_report": "",
        "audit_score": 0.0,
        "audit_findings": [],
        "is_hallucinating": False,
        "hallucination_count": 0,
        "verified_count": 0,
        "unsubstantiated_count": 0,
        "ragas_metrics": {},
        "ground_truth": "",
        "audit_iteration_count": 0,
        "final_report": "",
    }


# ═══════════════════════════════════════════════════════════════════════════
# LLM AS A JUDGE (5 DIMENSIONS)
# ═══════════════════════════════════════════════════════════════════════════

def judge_report_quality(variant_name: str, report: str, original_data: dict, state: dict) -> dict:
    """
    Uses LLM to score wealth manager report on 5 dimensions.
    
    Dimensions:
    1. Coherence (1-5): Professional tone, logical narrative structure
    2. Completeness (1-5): All tickers mentioned, key metrics included, findings present
    3. Financial Accuracy (1-5): Correct terminology, realistic analysis, no obvious errors
    4. Analysis Depth (1-5): Thoroughness of insights, quality of reasoning, substantiation
    5. Portfolio Relevance (1-5): How well-tailored to specific holdings, actionable insights
    
    Returns dict with scores (1-5), feedback, and overall quality score.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set. Cannot run judge.")
        return {
            "coherence": 0, "completeness": 0, "financial_accuracy": 0,
            "analysis_depth": 0, "portfolio_relevance": 0,
            "feedback": "Error: ANTHROPIC_API_KEY not set",
            "overall_quality": 0.0
        }
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Get context from state
    verified_count = state.get("verified_count", 0)
    unsubstantiated_count = state.get("unsubstantiated_count", 0)
    audit_score = state.get("audit_score", 0)
    tickers = state.get("tickers", [])
    
    judge_prompt = f"""You are a Senior Wealth Management Auditor and Expert Report Evaluator.
Grade the following investment report from Variant "{variant_name}" using a comprehensive rubric.

═══════════════════════════════════════════════════════════════════════
SOURCE DATA (Client Information)
═══════════════════════════════════════════════════════════════════════
Tickers: {', '.join(tickers)}
Original Request: {state.get('messages', ['N/A'])[0] if state.get('messages') else 'N/A'}

Audit Context (Data Quality Indicators):
- Verified Claims: {verified_count}
- Unsubstantiated Claims: {unsubstantiated_count}
- Audit Score (0-100): {audit_score}

═══════════════════════════════════════════════════════════════════════
REPORT TO EVALUATE
═══════════════════════════════════════════════════════════════════════
{report}

═══════════════════════════════════════════════════════════════════════
GRADING RUBRIC (Score 1-5, where 5 is excellent)
═══════════════════════════════════════════════════════════════════════

1. COHERENCE (Professional Tone & Narrative Logic)
   5 = Exceptional: Highly professional, well-structured, flows naturally
   4 = Strong: Professional tone, clear logic, minor organizational issues
   3 = Adequate: Mostly professional, generally logical, some clarity issues
   2 = Weak: Somewhat unprofessional or disjointed
   1 = Poor: Unprofessional or incoherent

2. COMPLETENESS (Coverage of All Key Elements)
   5 = Exceptional: All tickers mentioned, all key metrics included, comprehensive findings
   4 = Strong: All tickers + most metrics, good coverage of findings
   3 = Adequate: Most tickers + metrics mentioned, some findings present
   2 = Weak: Missing several tickers or key metrics
   1 = Poor: Major omissions of tickers, metrics, or findings

3. FINANCIAL ACCURACY (Correctness & Realism)
   5 = Exceptional: All terminology correct, realistic analysis, no errors
   4 = Strong: Mostly accurate, appropriate analysis, minor inaccuracies
   3 = Adequate: Generally correct, acceptable analysis
   2 = Weak: Some terminology errors or questionable analysis
   1 = Poor: Major inaccuracies or unrealistic claims

4. ANALYSIS DEPTH (Thoroughness & Reasoning Quality)
   5 = Exceptional: Deep insights, strong substantiation, excellent recommendations
   4 = Strong: Good depth, well-reasoned, clear recommendations
   3 = Adequate: Moderate depth, acceptable reasoning
   2 = Weak: Shallow analysis, limited reasoning
   1 = Poor: Very shallow or poorly reasoned

5. PORTFOLIO RELEVANCE (Tailoring to Specific Holdings)
   5 = Exceptional: Highly tailored, actionable insights specific to each ticker
   4 = Strong: Well-tailored, specific insights for most holdings
   3 = Adequate: Somewhat tailored, some generic content
   2 = Weak: Mostly generic, little tailoring to specific holdings
   1 = Poor: Generic, not tailored to portfolio

═══════════════════════════════════════════════════════════════════════
REQUIRED OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════
Return ONLY valid JSON (no markdown, no explanation, just the JSON object):
{{
  "coherence": <1-5>,
  "completeness": <1-5>,
  "financial_accuracy": <1-5>,
  "analysis_depth": <1-5>,
  "portfolio_relevance": <1-5>,
  "feedback": "<2-3 sentence summary of strengths and weaknesses>"
}}
"""
    
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": judge_prompt}]
        )
        
        # Extract JSON from response
        response_text = response.content[0].text if response.content else "{}"
        
        # Use regex to find JSON block (in case LLM adds extra text)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
        else:
            logger.warning(f"Judge response not in JSON format: {response_text}")
            scores = {
                "coherence": 0, "completeness": 0, "financial_accuracy": 0,
                "analysis_depth": 0, "portfolio_relevance": 0,
                "feedback": "Failed to parse judge response"
            }
        
        # Calculate overall quality score
        dimension_scores = [
            scores.get("coherence", 0),
            scores.get("completeness", 0),
            scores.get("financial_accuracy", 0),
            scores.get("analysis_depth", 0),
            scores.get("portfolio_relevance", 0),
        ]
        overall_quality = round(sum(dimension_scores) / len(dimension_scores), 2)
        scores["overall_quality"] = overall_quality
        
        return scores
        
    except Exception as e:
        logger.error(f"Error in judge: {e}")
        return {
            "coherence": 0, "completeness": 0, "financial_accuracy": 0,
            "analysis_depth": 0, "portfolio_relevance": 0,
            "feedback": f"Judge error: {str(e)}",
            "overall_quality": 0.0
        }


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def check_freshness(state: dict) -> str:
    """Check information freshness based on auditor verification metrics."""
    verified_count = state.get("verified_count", 0)
    unsubstantiated_count = state.get("unsubstantiated_count", 0)
    total_claims = verified_count + unsubstantiated_count
    
    if total_claims == 0:
        return "UNKNOWN (0 claims evaluated)"
    
    verification_rate = verified_count / total_claims
    
    if verification_rate >= 0.4:
        return f"YES ({verified_count}/{total_claims})"
    elif verification_rate >= 0.2:
        return f"PARTIAL ({verified_count}/{total_claims})"
    else:
        return f"NO ({verified_count}/{total_claims})"


def check_relevance(report: str, tickers: list) -> str:
    """Check ticker mentions in report."""
    report_lower = report.lower()
    mentioned = [t for t in tickers if t.lower() in report_lower]
    
    if len(mentioned) == len(tickers):
        return f"YES ({len(tickers)}/{len(tickers)})"
    elif mentioned:
        return f"PARTIAL ({len(mentioned)}/{len(tickers)})"
    else:
        return "NO (0/{len(tickers)})"


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD from token counts."""
    input_cost = (input_tokens / 1_000_000) * PRICING["input_cost_per_1m"]
    output_cost = (output_tokens / 1_000_000) * PRICING["output_cost_per_1m"]
    return input_cost + output_cost


# ═══════════════════════════════════════════════════════════════════════════
# MAIN COMPARISON EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

def run_variant(variant_name: str, workflow, initial_state: dict) -> tuple[dict, float]:
    """
    Run a single variant and return (final_state, latency).
    """
    print(f"\n  Running {variant_name}...")
    
    start_time = time.time()
    final_state = initial_state.copy()
    
    for output in workflow.stream(initial_state):
        for node_name, state in output.items():
            logger.info(f"    ✓ {node_name}")
            final_state.update(state)
    
    latency = time.time() - start_time
    
    return final_state, latency


def run_comparison_experiment(user_prompt: str, tickers: list, output_dir: Path):
    """Run the full comparison experiment with both variants."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"MARKET CONTEXT AGENT COMPARISON EXPERIMENT")
    print(f"{'='*80}")
    print(f"Prompt: {user_prompt}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"{'='*80}\n")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "prompt": user_prompt,
        "tickers": tickers,
        "variants": {}
    }
    
    # ─────────────────────────────────────────────────────────────────────
    # VARIANT A: WITH Market Context Agent
    # ─────────────────────────────────────────────────────────────────────
    print("VARIANT A: WITH Market Context Agent")
    print("-" * 80)
    
    workflow_a = create_workflow_with_market_context()
    initial_state_a = create_initial_state(user_prompt, tickers)
    
    state_a, latency_a = run_variant("Variant A (WITH Market Context)", workflow_a, initial_state_a)
    
    # Extract metrics
    final_report_a = state_a.get("final_report", "")
    audit_score_a = state_a.get("audit_score", 0)
    verified_a = state_a.get("verified_count", 0)
    unsubstantiated_a = state_a.get("unsubstantiated_count", 0)
    
    # Run judge
    print("  Running LLM Judge...")
    judge_scores_a = judge_report_quality("Variant A (WITH Market Context)", final_report_a, 
                                          {"prompt": user_prompt, "tickers": tickers}, state_a)
    
    print(f"  ✓ Latency: {latency_a:.2f}s")
    print(f"  ✓ Audit Score: {audit_score_a:.2f}/100")
    print(f"  ✓ Freshness: {check_freshness(state_a)}")
    print(f"  ✓ Relevance: {check_relevance(final_report_a, tickers)}")
    if judge_scores_a.get("overall_quality"):
        print(f"  ✓ Overall Quality (Judge): {judge_scores_a['overall_quality']}/5.0")
    
    results["variants"]["A_with_market_context"] = {
        "latency_seconds": latency_a,
        "audit_score": audit_score_a,
        "verified_claims": verified_a,
        "unsubstantiated_claims": unsubstantiated_a,
        "freshness": check_freshness(state_a),
        "relevance": check_relevance(final_report_a, tickers),
        "judge_scores": {
            "coherence": judge_scores_a.get("coherence", 0),
            "completeness": judge_scores_a.get("completeness", 0),
            "financial_accuracy": judge_scores_a.get("financial_accuracy", 0),
            "analysis_depth": judge_scores_a.get("analysis_depth", 0),
            "portfolio_relevance": judge_scores_a.get("portfolio_relevance", 0),
            "overall_quality": judge_scores_a.get("overall_quality", 0),
            "feedback": judge_scores_a.get("feedback", "")
        },
        "report_preview": final_report_a[:500] + "..." if len(final_report_a) > 500 else final_report_a
    }
    
    # ─────────────────────────────────────────────────────────────────────
    # VARIANT B: WITHOUT Market Context Agent
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("VARIANT B: WITHOUT Market Context Agent")
    print("-" * 80)
    
    workflow_b = create_workflow_without_market_context()
    initial_state_b = create_initial_state(user_prompt, tickers)
    
    state_b, latency_b = run_variant("Variant B (WITHOUT Market Context)", workflow_b, initial_state_b)
    
    # Extract metrics
    final_report_b = state_b.get("final_report", "")
    audit_score_b = state_b.get("audit_score", 0)
    verified_b = state_b.get("verified_count", 0)
    unsubstantiated_b = state_b.get("unsubstantiated_count", 0)
    
    # Run judge
    print("  Running LLM Judge...")
    judge_scores_b = judge_report_quality("Variant B (WITHOUT Market Context)", final_report_b,
                                          {"prompt": user_prompt, "tickers": tickers}, state_b)
    
    print(f"  ✓ Latency: {latency_b:.2f}s")
    print(f"  ✓ Audit Score: {audit_score_b:.2f}/100")
    print(f"  ✓ Freshness: {check_freshness(state_b)}")
    print(f"  ✓ Relevance: {check_relevance(final_report_b, tickers)}")
    if judge_scores_b.get("overall_quality"):
        print(f"  ✓ Overall Quality (Judge): {judge_scores_b['overall_quality']}/5.0")
    
    results["variants"]["B_without_market_context"] = {
        "latency_seconds": latency_b,
        "audit_score": audit_score_b,
        "verified_claims": verified_b,
        "unsubstantiated_claims": unsubstantiated_b,
        "freshness": check_freshness(state_b),
        "relevance": check_relevance(final_report_b, tickers),
        "judge_scores": {
            "coherence": judge_scores_b.get("coherence", 0),
            "completeness": judge_scores_b.get("completeness", 0),
            "financial_accuracy": judge_scores_b.get("financial_accuracy", 0),
            "analysis_depth": judge_scores_b.get("analysis_depth", 0),
            "portfolio_relevance": judge_scores_b.get("portfolio_relevance", 0),
            "overall_quality": judge_scores_b.get("overall_quality", 0),
            "feedback": judge_scores_b.get("feedback", "")
        },
        "report_preview": final_report_b[:500] + "..." if len(final_report_b) > 500 else final_report_b
    }
    
    # ─────────────────────────────────────────────────────────────────────
    # COMPARISON TABLE
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    comparison_data = {
        "Metric": [
            "Latency (s)",
            "Audit Score (0-100)",
            "Verified Claims",
            "Unsubstantiated Claims",
            "Freshness",
            "Relevance",
            "",
            "Judge: Coherence",
            "Judge: Completeness",
            "Judge: Financial Accuracy",
            "Judge: Analysis Depth",
            "Judge: Portfolio Relevance",
            "Judge: Overall Quality (avg)"
        ],
        "Variant A (WITH)": [
            f"{latency_a:.2f}",
            f"{audit_score_a:.1f}",
            f"{verified_a}",
            f"{unsubstantiated_a}",
            check_freshness(state_a),
            check_relevance(final_report_a, tickers),
            "",
            judge_scores_a.get("coherence", 0),
            judge_scores_a.get("completeness", 0),
            judge_scores_a.get("financial_accuracy", 0),
            judge_scores_a.get("analysis_depth", 0),
            judge_scores_a.get("portfolio_relevance", 0),
            judge_scores_a.get("overall_quality", 0)
        ],
        "Variant B (WITHOUT)": [
            f"{latency_b:.2f}",
            f"{audit_score_b:.1f}",
            f"{verified_b}",
            f"{unsubstantiated_b}",
            check_freshness(state_b),
            check_relevance(final_report_b, tickers),
            "",
            judge_scores_b.get("coherence", 0),
            judge_scores_b.get("completeness", 0),
            judge_scores_b.get("financial_accuracy", 0),
            judge_scores_b.get("analysis_depth", 0),
            judge_scores_b.get("portfolio_relevance", 0),
            judge_scores_b.get("overall_quality", 0)
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # Calculate deltas
    print(f"\n{'─'*80}")
    print("DELTA (A - B)")
    print(f"{'─'*80}\n")
    
    latency_delta = latency_a - latency_b
    latency_pct = (latency_delta / latency_b * 100) if latency_b > 0 else 0
    
    audit_delta = audit_score_a - audit_score_b
    quality_delta = judge_scores_a.get("overall_quality", 0) - judge_scores_b.get("overall_quality", 0)
    
    print(f"Latency Delta:        {latency_delta:+.2f}s ({latency_pct:+.1f}%)")
    print(f"Audit Score Delta:    {audit_delta:+.1f}")
    print(f"Quality Score Delta:  {quality_delta:+.2f}")
    print(f"Verified Claims Delta: {verified_a - verified_b:+d}")
    
    results["comparison"] = {
        "latency_delta_seconds": latency_delta,
        "latency_percent_change": latency_pct,
        "audit_score_delta": audit_delta,
        "quality_score_delta": quality_delta,
        "verified_claims_delta": verified_a - verified_b
    }
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save comparison table
    csv_file = output_dir / "comparison.csv"
    df_comparison.to_csv(csv_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"✓ Results saved to: {results_file}")
    print(f"✓ Comparison CSV:   {csv_file}")
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    # Configuration
    tickers = ["AAPL", "NVDA"]
    user_prompt = f"Provide comprehensive financial analysis for {', '.join(tickers)} with investment recommendations."
    
    # Create results directory
    output_dir = workspace_root / "results" / "market_context_compare" / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Run experiment
    results = run_comparison_experiment(user_prompt, tickers, output_dir)
