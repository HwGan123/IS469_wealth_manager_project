"""
Market Context Agent Comparison Experiment (Refactored)

Compares three workflow variants:
A: Normal flow WITH market context agent (Claude agentic loop - intelligent tool selection)
B (PARALLEL): Workflow WITH parallel market context (all 6 tools fetched in parallel - brute force)
C (NO AGENT): Workflow WITHOUT market context agent intervention (baseline)

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from contextlib import contextmanager
from typing import Dict

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

try:
    from langchain_community.callbacks.manager import get_openai_callback
    HAS_LANGCHAIN_CB = True
except ImportError:
    HAS_LANGCHAIN_CB = False
    logger.warning("LangChain callback not available - OpenAI token tracking will be limited")

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# TOKEN TRACKING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

# Global tracking for tokens across all agents (WORKFLOW ONLY - judge excluded)
_token_tracker: Dict[str, Dict] = {
    "anthropic": {"input_tokens": 0, "output_tokens": 0},
    "openai": {"input_tokens": 0, "output_tokens": 0}
}

# Separate tracking for judge tokens (NOT included in overall variant metrics)
_judge_token_tracker: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

def reset_token_tracker():
    """Reset the global token tracker (workflow tokens only)."""
    global _token_tracker
    _token_tracker = {
        "anthropic": {"input_tokens": 0, "output_tokens": 0},
        "openai": {"input_tokens": 0, "output_tokens": 0}
    }

def reset_judge_tracker():
    """Reset the judge token tracker (judge evaluation only)."""
    global _judge_token_tracker
    _judge_token_tracker = {"input_tokens": 0, "output_tokens": 0}

def track_anthropic_tokens(input_tokens: int, output_tokens: int):
    """Track Anthropic token usage (workflow only)."""
    _token_tracker["anthropic"]["input_tokens"] += input_tokens
    _token_tracker["anthropic"]["output_tokens"] += output_tokens

def track_openai_tokens(input_tokens: int, output_tokens: int):
    """Track OpenAI token usage (workflow agents ONLY - judge excluded)."""
    _token_tracker["openai"]["input_tokens"] += input_tokens
    _token_tracker["openai"]["output_tokens"] += output_tokens

def track_judge_tokens(input_tokens: int, output_tokens: int):
    """Track OpenAI judge token usage (SEPARATE - not in overall metrics)."""
    _judge_token_tracker["input_tokens"] += input_tokens
    _judge_token_tracker["output_tokens"] += output_tokens

def get_tracked_tokens() -> Dict:
    """Get the current token tracking snapshot (workflow only - judge excluded)."""
    return {
        "anthropic_market_context": _token_tracker["anthropic"].copy(),
        "openai_agents": _token_tracker["openai"].copy(),
    }

def get_judge_tokens() -> Dict:
    """Get the judge token tracking snapshot (evaluation only - separate). Returns dict with default 0 if empty."""
    return {
        "input_tokens": _judge_token_tracker.get("input_tokens", 0),
        "output_tokens": _judge_token_tracker.get("output_tokens", 0)
    }

@contextmanager
def track_openai_usage():
    """Context manager to track OpenAI token usage via LangChain."""
    if HAS_LANGCHAIN_CB:
        with get_openai_callback() as cb:
            yield cb
    else:
        # Fallback - no tracking
        class DummyCallback:
            completion_tokens = 0
            prompt_tokens = 0
        yield DummyCallback()

# Pricing for all models (per 1M tokens)
PRICING = {
    "anthropic": {
        "input_cost_per_1m": 1.00,      # Claude 3.5 Haiku: $1.00 per 1M input tokens
        "output_cost_per_1m": 5.00,     # Claude 3.5 Haiku: $5.00 per 1M output tokens
    },
    "openai": {
        "gpt-4o": {
            "input_cost_per_1m": 15.00,     # GPT-4o: $15 per 1M input tokens
            "output_cost_per_1m": 60.00,    # GPT-4o: $60 per 1M output tokens
        },
        "gpt-4o-mini": {
            "input_cost_per_1m": 0.15,      # GPT-4o-mini: $0.15 per 1M input tokens
            "output_cost_per_1m": 0.60,     # GPT-4o-mini: $0.60 per 1M output tokens
        }
    }
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
    VARIANT C (NO AGENT): Agents on their own WITHOUT market context agent intervention.
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


def parallel_market_context_node(state: WealthManagerState) -> dict:
    """
    VARIANT B (PARALLEL): Market context with PARALLEL tool fetching.
    Fetches all 6 MCP tools in parallel instead of using Claude's agentic loop.
    
    Philosophy: No intelligence about which tools to call - just fetch everything at once.
    Expected: 65% faster (~3-5s), same data quality as Variant A.
    Trade-off: Wasteful if user only needs specific data, but maximum parallel efficiency.
    """
    print("--- AGENT: MARKET CONTEXT (PARALLEL MODE) ---")
    tickers = state.get("tickers", [])
    
    if not tickers:
        logger.warning("No portfolio tickers found in state")
        return {
            "market_context": {
                "_tools_used": [],
                "_token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "_parallel_latency": 0.0
            },
            "audit_iteration_count": state.get("audit_iteration_count", 0)
        }
    
    logger.info(f"Market Context Agent (Parallel): Fetching all 6 tools for {tickers}")
    
    # Define all tools to fetch in parallel
    tools_to_fetch = [
        ("fetch_news", {"tickers": tickers, "days_back": 7}),
        ("fetch_earnings", {"tickers": tickers}),
        ("fetch_analyst_ratings", {"tickers": tickers}),
        ("fetch_10k_content", {"tickers": tickers, "sections": ["MD&A", "Risk Factors", "Financial Summary"]}),
        ("fetch_10q_content", {"tickers": tickers, "sections": ["Management Discussion", "Financial Highlights"]}),
        ("fetch_xbrl_financials", {"tickers": tickers, "filing_type": "10-K"}),
    ]
    
    market_context = {}
    start_parallel = time.time()
    
    # Parallel execution with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(dispatch_mcp_tool, tool_name, tool_input): tool_name 
            for tool_name, tool_input in tools_to_fetch
        }
        
        for future in as_completed(futures):
            tool_name = futures[future]
            try:
                result = future.result(timeout=30)
                market_context[tool_name] = result
                logger.info(f"  ✓ {tool_name} completed")
            except Exception as e:
                logger.error(f"  ✗ {tool_name} failed: {e}")
                market_context[tool_name] = {"error": str(e)}
    
    parallel_latency = time.time() - start_parallel
    
    # Record parallel execution metrics
    tools_used = [name for name, _ in tools_to_fetch]
    market_context["_tools_used"] = tools_used
    market_context["_parallel_latency"] = parallel_latency
    
    # Note: Parallel execution doesn't use Claude, so no token usage to track
    market_context["_token_usage"] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "note": "Parallel execution - no Claude agentic loop"
    }
    
    logger.info(f"  ✓ Parallel market data gathering complete ({parallel_latency:.2f}s)")
    logger.info(f"  → All 6 tools fetched in parallel")
    
    return {
        "market_context": market_context,
        "audit_iteration_count": state.get("audit_iteration_count", 0)
    }


def create_workflow_with_parallel_market_context():
    """
    VARIANT B (PARALLEL): Normal workflow with PARALLEL market context agent.
    Fetches all 6 tools in parallel instead of using Claude's agentic reasoning.
    
    Trade-off: Faster execution, but wastes API calls on unnecessary tools.
    """
    workflow = StateGraph(WealthManagerState)
    
    workflow.add_node("orchestrator_agent", orchestrator_node)
    workflow.add_node("market_context_agent", parallel_market_context_node)
    workflow.add_node("sentiment_agent", sentiment_node)
    workflow.add_node("investment_analyst_agent", analyst_node)
    workflow.add_node("auditor_agent", auditor_node)
    workflow.add_node("report_generator_agent", report_generator_node)
    
    # Orchestrator -> Parallel Market Context -> Sentiment -> Analyst -> Auditor -> Report
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

def judge_report_quality(variant_name: str, report: str, original_data: dict, state: dict, tools_used_count: int = None) -> dict:
    """
    Uses GPT-4o-mini to score wealth manager report on 8 dimensions.
    
    Core Quality Dimensions (1-5):
    1. Coherence: Professional tone, logical narrative structure
    2. Completeness: All tickers mentioned, key metrics included, findings present
    3. Financial Accuracy: Correct terminology, realistic analysis, no obvious errors
    4. Analysis Depth: Thoroughness of insights, quality of reasoning, substantiation
    5. Portfolio Relevance: How well-tailored to specific holdings, actionable insights
    
    Data Quality Dimensions (1-5) - Shows impact of market context vs baseline:
    6. Data Recency: How current/recent is the information? (reflects market context agent)
    7. Market Context Integration: Integration of market developments, earnings, news
    
    Efficiency Dimension (1-5) - Rewards smart tool selection:
    8. Execution Efficiency: Optimal tool usage vs brute-force data collection
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set. Cannot run judge.")
        return {
            "coherence": 0, "completeness": 0, "financial_accuracy": 0,
            "analysis_depth": 0, "portfolio_relevance": 0,
            "data_recency": 0, "market_context_integration": 0,
            "execution_efficiency": 0,
            "feedback": "Error: OPENAI_API_KEY not set",
            "overall_quality": 0.0
        }
    
    from langchain_openai import ChatOpenAI
    
    client = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    
    # Get context from state
    verified_count = state.get("verified_count", 0)
    unsubstantiated_count = state.get("unsubstantiated_count", 0)
    audit_score = state.get("audit_score", 0)
    tickers = state.get("tickers", [])
    
    # Get tools usage info from market_context or default
    market_context = state.get("market_context", {})
    tools_used = market_context.get("_tools_used", [])
    tools_count = len(tools_used) if tools_used else (tools_used_count or 0)
    
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

Execution Context (Efficiency Metrics):
- Tools Used: {tools_count}/6 total available tools
- Approach: {'Smart selection' if tools_count < 6 and tools_count > 0 else ('Full brute-force' if tools_count == 6 else 'No external tools')}

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

6. DATA RECENCY (Currency of Information)
   5 = Exceptional: Uses recent/current market data, recent earnings, latest news
   4 = Strong: Mostly recent data, some current market references
   3 = Adequate: Mix of recent and slightly dated information
   2 = Weak: Mostly general information, few current market references
   1 = Poor: Appears to use outdated or stale information

7. MARKET CONTEXT INTEGRATION (Use of Market Developments)
   5 = Exceptional: Seamlessly integrates news, earnings, analyst ratings, market trends
   4 = Strong: Good integration of market context with fundamentals
   3 = Adequate: Some market context mentioned, mostly fundamental analysis
   2 = Weak: Minimal market context, mostly generic financial analysis
   1 = Poor: No integration of market developments or current events

8. EXECUTION EFFICIENCY (Optimal vs Brute-Force Data Collection)
   5 = Exceptional: Smart tool selection achieved high-quality report with minimal tools (<3)
   4 = Strong: Efficient tool usage (3-4 tools) with comprehensive analysis
   3 = Adequate: All tools used but report quality justifies the overhead
   2 = Weak: Brute-force approach (all 6 tools) with minimal incremental value
   1 = Poor: Excessive data collection with no efficiency consideration
   Note: Score based on whether tool usage appears necessary and justified.

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
  "data_recency": <1-5>,
  "market_context_integration": <1-5>,
  "execution_efficiency": <1-5>,
  "feedback": "<2-3 sentence summary of strengths and weaknesses>"
}}
"""
    
    try:
        with track_openai_usage() as openai_cb:
            response = client.invoke(judge_prompt)
            
            # Track judge OpenAI tokens (SEPARATE - not in variant overall metrics)
            if openai_cb and hasattr(openai_cb, 'completion_tokens') and hasattr(openai_cb, 'prompt_tokens'):
                try:
                    track_judge_tokens(
                        getattr(openai_cb, 'prompt_tokens', 0),
                        getattr(openai_cb, 'completion_tokens', 0)
                    )
                except Exception as token_error:
                    logger.debug(f"Could not track judge tokens: {token_error}")
        
        # Extract JSON from response
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Use regex to find JSON block (in case LLM adds extra text)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
        else:
            logger.warning(f"Judge response not in JSON format: {response_text}")
            scores = {
                "coherence": 0, "completeness": 0, "financial_accuracy": 0,
                "analysis_depth": 0, "portfolio_relevance": 0,
                "data_recency": 0, "market_context_integration": 0,
                "execution_efficiency": 0,
                "feedback": "Failed to parse judge response"
            }
        
        # Calculate overall quality score (average of all 8 dimensions)
        dimension_scores = [
            scores.get("coherence", 0),
            scores.get("completeness", 0),
            scores.get("financial_accuracy", 0),
            scores.get("analysis_depth", 0),
            scores.get("portfolio_relevance", 0),
            scores.get("data_recency", 0),
            scores.get("market_context_integration", 0),
            scores.get("execution_efficiency", 0),
        ]
        overall_quality = round(sum(dimension_scores) / len(dimension_scores), 2)
        scores["overall_quality"] = overall_quality
        
        return scores
        
    except Exception as e:
        logger.error(f"Error in judge: {e}")
        return {
            "coherence": 0, "completeness": 0, "financial_accuracy": 0,
            "analysis_depth": 0, "portfolio_relevance": 0,
            "data_recency": 0, "market_context_integration": 0,
            "execution_efficiency": 0,
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


def get_total_tokens_for_variant(variant_tokens: Dict) -> tuple:
    """Extract total token counts and costs from variant token usage dict."""
    anthropic_mc = variant_tokens.get("anthropic_market_context", {})
    openai = variant_tokens.get("openai_agents", {})
    
    anthropic_input = anthropic_mc.get("input_tokens", 0)
    anthropic_output = anthropic_mc.get("output_tokens", 0)
    anthropic_total = anthropic_input + anthropic_output
    
    openai_input = openai.get("input_tokens", 0)
    openai_output = openai.get("output_tokens", 0)
    openai_total = openai_input + openai_output
    
    total = anthropic_total + openai_total
    
    # Format as string for display
    return (
        total,
        f"Anthropic: {anthropic_total} | OpenAI: {openai_total}",
        anthropic_total,
        openai_total,
        anthropic_input + openai_input,
        anthropic_output + openai_output
    )


def print_token_summary(tokens_dict: Dict):
    """Pretty print token summary across all providers (workflow tokens only - judge excluded)."""
    anthropic_mc = tokens_dict.get("anthropic_market_context", {})
    openai = tokens_dict.get("openai_agents", {})
    
    anthropic_total = anthropic_mc.get("input_tokens", 0) + anthropic_mc.get("output_tokens", 0)
    openai_total = openai.get("input_tokens", 0) + openai.get("output_tokens", 0)
    
    print(f"  ✓ Tokens (Workflow - Judge Excluded):")
    print(f"      Anthropic Market Context:")
    if anthropic_mc.get("input_tokens", 0) > 0 or anthropic_mc.get("output_tokens", 0) > 0:
        print(f"        - {anthropic_mc.get('input_tokens', 0)} input, {anthropic_mc.get('output_tokens', 0)} output")
    else:
        print(f"        - 0 (no agent or agent inactive)")
    print(f"      OpenAI Agents (Analyst/Auditor/Orchestrator):")
    if openai.get("input_tokens", 0) > 0 or openai.get("output_tokens", 0) > 0:
        print(f"        - {openai.get('input_tokens', 0)} input, {openai.get('output_tokens', 0)} output")
    else:
        print(f"        - 0")
    print(f"      Total Workflow: {anthropic_total + openai_total}")
    
    # Calculate and show costs
    costs = calculate_total_cost(tokens_dict)
    judge_tokens = get_judge_tokens()
    judge_cost = calculate_judge_cost(judge_tokens["input_tokens"], judge_tokens["output_tokens"])
    
    print(f"  ✓ Costs (Workflow Only - Judge Separate):")
    if costs["anthropic_market_context"] > 0:
        print(f"      - Anthropic: ${costs['anthropic_market_context']:.6f}")
    if costs["openai_agents"] > 0:
        print(f"      - OpenAI Agents: ${costs['openai_agents']:.6f}")
    print(f"      - Workflow Total: ${costs['total']:.6f}")
    
    if judge_cost > 0:
        print(f"  ✓ Judge Evaluation Cost (Separate): ${judge_cost:.6f}")


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


def calculate_anthropic_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for Anthropic tokens."""
    input_cost = (input_tokens / 1_000_000) * PRICING["anthropic"]["input_cost_per_1m"]
    output_cost = (output_tokens / 1_000_000) * PRICING["anthropic"]["output_cost_per_1m"]
    return round(input_cost + output_cost, 4)


def calculate_openai_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini") -> float:
    """Calculate cost for OpenAI tokens."""
    if model not in PRICING["openai"]:
        model = "gpt-4o-mini"  # Default to cheaper model
    
    pricing = PRICING["openai"][model]
    input_cost = (input_tokens / 1_000_000) * pricing["input_cost_per_1m"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_cost_per_1m"]
    return round(input_cost + output_cost, 4)


def calculate_judge_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for judge evaluation (gpt-4o-mini)."""
    return calculate_openai_cost(input_tokens, output_tokens, "gpt-4o-mini")


def calculate_total_cost(variant_tokens: Dict) -> Dict:
    """
    Calculate total costs for a variant across all providers (WORKFLOW ONLY - JUDGE EXCLUDED).
    
    Args:
        variant_tokens: Dict with keys like "anthropic_market_context", "openai_agents"
                       These are workflow tokens ONLY - judge tokens are tracked separately
    
    Returns:
        Dict with cost breakdown and total (workflow only)
    """
    costs = {
        "anthropic_market_context": 0.0,
        "openai_agents": 0.0,
        "total": 0.0
    }
    
    if "anthropic_market_context" in variant_tokens:
        tokens = variant_tokens["anthropic_market_context"]
        costs["anthropic_market_context"] = calculate_anthropic_cost(
            tokens.get("input_tokens", 0),
            tokens.get("output_tokens", 0)
        )
    
    if "openai_agents" in variant_tokens:
        tokens = variant_tokens["openai_agents"]
        # OpenAI agents ONLY (judge tokens are separate and not included here)
        # Estimate mix: analyst uses gpt-4o, auditor + orchestrator use gpt-4o-mini
        # Conservative estimate: 40% gpt-4o (analyst), 60% gpt-4o-mini (auditor + orchestrator)
        openai_cost_mini = calculate_openai_cost(
            int(tokens.get("input_tokens", 0) * 0.6),
            int(tokens.get("output_tokens", 0) * 0.6),
            "gpt-4o-mini"
        )
        openai_cost_4o = calculate_openai_cost(
            int(tokens.get("input_tokens", 0) * 0.4),
            int(tokens.get("output_tokens", 0) * 0.4),
            "gpt-4o"
        )
        costs["openai_agents"] = openai_cost_mini + openai_cost_4o
    
    costs["total"] = round(
        costs["anthropic_market_context"] +
        costs["openai_agents"],
        4
    )
    
    return costs


# ═══════════════════════════════════════════════════════════════════════════
# MAIN COMPARISON EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

def run_variant(variant_name: str, workflow, initial_state: dict) -> tuple[dict, float, Dict]:
    """
    Run a single variant and return (final_state, latency, token_usage_dict).
    
    Token usage DOES NOT include judge tokens (judge is a separate evaluator, not part of variant efficiency).
    
    Token usage includes (WORKFLOW ONLY):
    - anthropic_market_context: Tokens from market context agent (Anthropic)
    - openai_agents: Tokens from analyst, auditor, orchestrator (OpenAI) - NOT judge
    
    Judge tokens are tracked separately and can be retrieved via get_judge_tokens()
    """
    print(f"\n  Running {variant_name}...")
    
    # Reset token tracking for this variant
    reset_token_tracker()
    
    start_time = time.time()
    final_state = initial_state.copy()
    
    # Run workflow with OpenAI token tracking
    with track_openai_usage() as openai_cb:
        for output in workflow.stream(final_state):
            for node_name, state in output.items():
                logger.info(f"    ✓ {node_name}")
                final_state.update(state)
    
    # Track OpenAI tokens if callback is available
    if openai_cb and hasattr(openai_cb, 'completion_tokens'):
        track_openai_tokens(
            openai_cb.prompt_tokens,
            openai_cb.completion_tokens
        )
    
    latency = time.time() - start_time
    
    # Get all tracked tokens
    all_tokens = get_tracked_tokens()
    
    # Also capture market_context tokens (Anthropic)
    if final_state.get("market_context"):
        mc_tokens = final_state["market_context"].get("_token_usage", {})
        if mc_tokens.get("input_tokens", 0) > 0 or mc_tokens.get("output_tokens", 0) > 0:
            all_tokens["anthropic_market_context"] = {
                "input_tokens": mc_tokens.get("input_tokens", 0),
                "output_tokens": mc_tokens.get("output_tokens", 0),
            }
    
    return final_state, latency, all_tokens


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
    # VARIANT A: WITH Market Context Agent (Agentic - Smart Tool Selection)
    # ─────────────────────────────────────────────────────────────────────
    print("VARIANT A: WITH Market Context Agent (Agentic - Smart Tool Selection)")
    print("-" * 80)
    
    workflow_a = create_workflow_with_market_context()
    initial_state_a = create_initial_state(user_prompt, tickers)
    
    state_a, latency_a, tokens_a = run_variant("Variant A (Agentic Selection)", workflow_a, initial_state_a)
    
    # Extract market context info for efficiency scoring
    market_context_a = state_a.get("market_context", {})
    tools_used_a = market_context_a.get("_tools_used", [])
    
    # Extract metrics
    final_report_a = state_a.get("final_report", "")
    
    # Run judge
    print("  Running LLM Judge...")
    judge_scores_a = judge_report_quality("Variant A (Agentic)", final_report_a, 
                                          {"prompt": user_prompt, "tickers": tickers}, state_a, len(tools_used_a))
    
    print(f"  ✓ Latency: {latency_a:.2f}s")
    print_token_summary(tokens_a)
    print(f"  ✓ Relevance: {check_relevance(final_report_a, tickers)}")
    if judge_scores_a.get("overall_quality"):
        print(f"  ✓ Overall Quality (Judge): {judge_scores_a['overall_quality']}/5.0")
    
    costs_a = calculate_total_cost(tokens_a)
    
    results["variants"]["A_agentic"] = {
        "latency_seconds": latency_a,
        "token_usage": tokens_a,
        "costs": costs_a,
        "relevance": check_relevance(final_report_a, tickers),
        "judge_scores": {
            "coherence": judge_scores_a.get("coherence", 0),
            "completeness": judge_scores_a.get("completeness", 0),
            "financial_accuracy": judge_scores_a.get("financial_accuracy", 0),
            "analysis_depth": judge_scores_a.get("analysis_depth", 0),
            "portfolio_relevance": judge_scores_a.get("portfolio_relevance", 0),
            "data_recency": judge_scores_a.get("data_recency", 0),
            "market_context_integration": judge_scores_a.get("market_context_integration", 0),
            "execution_efficiency": judge_scores_a.get("execution_efficiency", 0),
            "overall_quality": judge_scores_a.get("overall_quality", 0),
            "feedback": judge_scores_a.get("feedback", "")
        },
        "report_preview": final_report_a[:500] + "..." if len(final_report_a) > 500 else final_report_a
    }
    
    # ─────────────────────────────────────────────────────────────────────
    # VARIANT B (PARALLEL): WITH Parallel Market Context (Brute-Force All Tools)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("VARIANT B (PARALLEL): WITH Parallel Market Context (Brute-Force)")
    print("-" * 80)
    
    workflow_b = create_workflow_with_parallel_market_context()
    initial_state_b = create_initial_state(user_prompt, tickers)
    
    state_b, latency_b, tokens_b = run_variant("Variant B (Parallel)", workflow_b, initial_state_b)
    
    # Extract parallel execution info
    market_context_b = state_b.get("market_context", {})
    tools_used_b = market_context_b.get("_tools_used", [])
    parallel_latency_b = market_context_b.get("_parallel_latency", 0.0)
    
    # Extract metrics
    final_report_b = state_b.get("final_report", "")
    
    # Run judge
    print("  Running LLM Judge...")
    judge_scores_b = judge_report_quality("Variant B (Parallel)", final_report_b,
                                          {"prompt": user_prompt, "tickers": tickers}, state_b, len(tools_used_b))
    
    print(f"  ✓ Total Latency: {latency_b:.2f}s")
    print(f"  ✓ Parallel Fetch Time: {parallel_latency_b:.2f}s")
    print_token_summary(tokens_b)
    print(f"  ✓ Relevance: {check_relevance(final_report_b, tickers)}")
    print(f"  ✓ Tools Fetched: {len(tools_used_b)}/6 (all in parallel)")
    if judge_scores_b.get("overall_quality"):
        print(f"  ✓ Overall Quality (Judge): {judge_scores_b['overall_quality']}/5.0")
    
    costs_b = calculate_total_cost(tokens_b)
    
    results["variants"]["B_parallel"] = {
        "latency_seconds": latency_b,
        "parallel_fetch_latency": parallel_latency_b,
        "token_usage": tokens_b,
        "costs": costs_b,
        "relevance": check_relevance(final_report_b, tickers),
        "tools_fetched": len(tools_used_b),
        "execution_mode": "Parallel (all 6 tools)",
        "judge_scores": {
            "coherence": judge_scores_b.get("coherence", 0),
            "completeness": judge_scores_b.get("completeness", 0),
            "financial_accuracy": judge_scores_b.get("financial_accuracy", 0),
            "analysis_depth": judge_scores_b.get("analysis_depth", 0),
            "portfolio_relevance": judge_scores_b.get("portfolio_relevance", 0),
            "data_recency": judge_scores_b.get("data_recency", 0),
            "market_context_integration": judge_scores_b.get("market_context_integration", 0),
            "execution_efficiency": judge_scores_b.get("execution_efficiency", 0),
            "overall_quality": judge_scores_b.get("overall_quality", 0),
            "feedback": judge_scores_b.get("feedback", "")
        },
        "report_preview": final_report_b[:500] + "..." if len(final_report_b) > 500 else final_report_b
    }
    
    # ─────────────────────────────────────────────────────────────────────
    # VARIANT C (NO AGENT): WITHOUT Market Context Agent (Baseline)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("VARIANT C (NO AGENT): WITHOUT Market Context Agent (Baseline)")
    print("-" * 80)
    
    workflow_c = create_workflow_without_market_context()
    initial_state_c = create_initial_state(user_prompt, tickers)
    
    state_c, latency_c, tokens_c = run_variant("Variant C (No Agent)", workflow_c, initial_state_c)
    
    # For Variant C, no market context agent was used
    tools_used_c = []
    
    # Extract metrics
    final_report_c = state_c.get("final_report", "")
    
    # Run judge
    print("  Running LLM Judge...")
    judge_scores_c = judge_report_quality("Variant C (No Agent)", final_report_c,
                                          {"prompt": user_prompt, "tickers": tickers}, state_c, len(tools_used_c))
    
    print(f"  ✓ Latency: {latency_c:.2f}s")
    print_token_summary(tokens_c)
    print(f"  ✓ Relevance: {check_relevance(final_report_c, tickers)}")
    if judge_scores_c.get("overall_quality"):
        print(f"  ✓ Overall Quality (Judge): {judge_scores_c['overall_quality']}/5.0")
    
    costs_c = calculate_total_cost(tokens_c)
    
    results["variants"]["C_no_agent"] = {
        "latency_seconds": latency_c,
        "token_usage": tokens_c,
        "costs": costs_c,
        "relevance": check_relevance(final_report_c, tickers),
        "judge_scores": {
            "coherence": judge_scores_c.get("coherence", 0),
            "completeness": judge_scores_c.get("completeness", 0),
            "financial_accuracy": judge_scores_c.get("financial_accuracy", 0),
            "analysis_depth": judge_scores_c.get("analysis_depth", 0),
            "portfolio_relevance": judge_scores_c.get("portfolio_relevance", 0),
            "data_recency": judge_scores_c.get("data_recency", 0),
            "market_context_integration": judge_scores_c.get("market_context_integration", 0),
            "execution_efficiency": judge_scores_c.get("execution_efficiency", 0),
            "overall_quality": judge_scores_c.get("overall_quality", 0),
            "feedback": judge_scores_c.get("feedback", "")
        },
        "report_preview": final_report_c[:500] + "..." if len(final_report_c) > 500 else final_report_c
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
            "Total Tokens (Breakdown)",
            "Anthropic Tokens",
            "OpenAI Tokens",
            "Combined Input Tokens",
            "Combined Output Tokens",
            "Relevance",
            "Tools Fetched",
            "",
            "Judge: Coherence",
            "Judge: Completeness",
            "Judge: Financial Accuracy",
            "Judge: Analysis Depth",
            "Judge: Portfolio Relevance",
            "Judge: Data Recency",
            "Judge: Market Context Integration",
            "Judge: Execution Efficiency",
            "Judge: Overall Quality (avg)"
        ],
        "Variant A (Agentic)": [
            f"{latency_a:.2f}",
            get_total_tokens_for_variant(tokens_a)[1],
            f"{get_total_tokens_for_variant(tokens_a)[2]}",
            f"{get_total_tokens_for_variant(tokens_a)[3]}",
            f"{get_total_tokens_for_variant(tokens_a)[4]}",
            f"{get_total_tokens_for_variant(tokens_a)[5]}",
            check_relevance(final_report_a, tickers),
            "Smart Selection",
            "",
            judge_scores_a.get("coherence", 0),
            judge_scores_a.get("completeness", 0),
            judge_scores_a.get("financial_accuracy", 0),
            judge_scores_a.get("analysis_depth", 0),
            judge_scores_a.get("portfolio_relevance", 0),
            judge_scores_a.get("data_recency", 0),
            judge_scores_a.get("market_context_integration", 0),
            judge_scores_a.get("execution_efficiency", 0),
            judge_scores_a.get("overall_quality", 0)
        ],
        "Variant B (Parallel)": [
            f"{latency_b:.2f}",
            get_total_tokens_for_variant(tokens_b)[1],
            f"{get_total_tokens_for_variant(tokens_b)[2]}",
            f"{get_total_tokens_for_variant(tokens_b)[3]}",
            f"{get_total_tokens_for_variant(tokens_b)[4]}",
            f"{get_total_tokens_for_variant(tokens_b)[5]}",
            check_relevance(final_report_b, tickers),
            f"{len(tools_used_b)}/6",
            "",
            judge_scores_b.get("coherence", 0),
            judge_scores_b.get("completeness", 0),
            judge_scores_b.get("financial_accuracy", 0),
            judge_scores_b.get("analysis_depth", 0),
            judge_scores_b.get("portfolio_relevance", 0),
            judge_scores_b.get("data_recency", 0),
            judge_scores_b.get("market_context_integration", 0),
            judge_scores_b.get("execution_efficiency", 0),
            judge_scores_b.get("overall_quality", 0)
        ],
        "Variant C (No Agent)": [
            f"{latency_c:.2f}",
            get_total_tokens_for_variant(tokens_c)[1],
            f"{get_total_tokens_for_variant(tokens_c)[2]}",
            f"{get_total_tokens_for_variant(tokens_c)[3]}",
            f"{get_total_tokens_for_variant(tokens_c)[4]}",
            f"{get_total_tokens_for_variant(tokens_c)[5]}",
            check_relevance(final_report_c, tickers),
            "None",
            "",
            judge_scores_c.get("coherence", 0),
            judge_scores_c.get("completeness", 0),
            judge_scores_c.get("financial_accuracy", 0),
            judge_scores_c.get("analysis_depth", 0),
            judge_scores_c.get("portfolio_relevance", 0),
            judge_scores_c.get("data_recency", 0),
            judge_scores_c.get("market_context_integration", 0),
            judge_scores_c.get("execution_efficiency", 0),
            judge_scores_c.get("overall_quality", 0)
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # Calculate deltas
    print(f"\n{'─'*80}")
    print("DELTA COMPARISONS")
    print(f"{'─'*80}\n")
    
    # Get total tokens for each variant
    total_tokens_a = get_total_tokens_for_variant(tokens_a)[0]
    total_tokens_b = get_total_tokens_for_variant(tokens_b)[0]
    total_tokens_c = get_total_tokens_for_variant(tokens_c)[0]
    
    # A vs C (Agentic vs No Tools)
    print("A vs C (Agentic Selection vs No Agent):")
    latency_delta_ac = latency_a - latency_c
    latency_pct_ac = (latency_delta_ac / latency_c * 100) if latency_c > 0 else 0
    token_delta_ac = total_tokens_a - total_tokens_c
    quality_delta_ac = judge_scores_a.get("overall_quality", 0) - judge_scores_c.get("overall_quality", 0)
    cost_delta_ac = calculate_total_cost(tokens_a)["total"] - calculate_total_cost(tokens_c)["total"]
    
    print(f"  Latency Delta:        {latency_delta_ac:+.2f}s ({latency_pct_ac:+.1f}%)")
    print(f"  Token Delta:          {token_delta_ac:+d} tokens")
    print(f"  Cost Delta:           ${cost_delta_ac:+.6f}")
    print(f"  Quality Score Delta:  {quality_delta_ac:+.2f}")
    
    # B vs C (Parallel vs No Tools)
    print("\nB vs C (Parallel Execution vs No Agent):")
    latency_delta_bc = latency_b - latency_c
    latency_pct_bc = (latency_delta_bc / latency_c * 100) if latency_c > 0 else 0
    token_delta_bc = total_tokens_b - total_tokens_c
    quality_delta_bc = judge_scores_b.get("overall_quality", 0) - judge_scores_c.get("overall_quality", 0)
    cost_delta_bc = calculate_total_cost(tokens_b)["total"] - calculate_total_cost(tokens_c)["total"]
    
    print(f"  Latency Delta:        {latency_delta_bc:+.2f}s ({latency_pct_bc:+.1f}%)")
    print(f"  Token Delta:          {token_delta_bc:+d} tokens")
    print(f"  Cost Delta:           ${cost_delta_bc:+.6f}")
    print(f"  Quality Score Delta:  {quality_delta_bc:+.2f}")
    
    # A vs B (Agentic vs Parallel)
    print("\nA vs B (Agentic Selection vs Parallel Execution):")
    latency_delta_ab = latency_a - latency_b
    latency_pct_ab = (latency_delta_ab / latency_b * 100) if latency_b > 0 else 0
    token_delta_ab = total_tokens_a - total_tokens_b
    quality_delta_ab = judge_scores_a.get("overall_quality", 0) - judge_scores_b.get("overall_quality", 0)
    cost_delta_ab = calculate_total_cost(tokens_a)["total"] - calculate_total_cost(tokens_b)["total"]
    
    print(f"  Latency Delta:        {latency_delta_ab:+.2f}s ({latency_pct_ab:+.1f}%) [B is {'faster' if latency_delta_ab > 0 else 'slower'}]")
    print(f"  Token Delta:          {token_delta_ab:+d} tokens")
    print(f"  Cost Delta:           ${cost_delta_ab:+.6f}")
    print(f"  Quality Score Delta:  {quality_delta_ab:+.2f}")
    print(f"  Parallel Fetch Time:  {parallel_latency_b:.2f}s (part of B's total)")
    
    results["comparison"] = {
        "A_vs_C": {
            "latency_delta_seconds": latency_delta_ac,
            "latency_percent_change": latency_pct_ac,
            "token_delta": token_delta_ac,
            "cost_delta": cost_delta_ac,
            "quality_score_delta": quality_delta_ac
        },
        "B_vs_C": {
            "latency_delta_seconds": latency_delta_bc,
            "latency_percent_change": latency_pct_bc,
            "token_delta": token_delta_bc,
            "cost_delta": cost_delta_bc,
            "quality_score_delta": quality_delta_bc
        },
        "A_vs_B": {
            "latency_delta_seconds": latency_delta_ab,
            "latency_percent_change": latency_pct_ab,
            "token_delta": token_delta_ab,
            "cost_delta": cost_delta_ab,
            "quality_score_delta": quality_delta_ab,
            "parallel_fetch_time": parallel_latency_b
        }
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
    # Configuration - Single test case focusing on earnings
    user_prompt = "Quick earnings outlook on AAPL and NVDA - focus on recent earnings and analyst expectations"
    tickers = []  # Let orchestrator auto-detect from message
    
    # Create results directory
    output_dir = workspace_root / "results" / "market_context_compare" / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Run experiment
    results = run_comparison_experiment(user_prompt, tickers, output_dir)
