"""
Orchestrator agent — entry point for the wealth manager graph.

Responsibilities:
  1. Parse the user message to extract ticker symbols
  2. Determine which workflow path to invoke (currently always the full pipeline)
  3. Populate state fields that downstream agents depend on (tickers, etc.)

Ticker extraction uses a two-stage approach:
  - Stage 1: regex scan for known uppercase symbols (fast, no API needed)
  - Stage 2: optional LLM extraction when OPENAI_API_KEY is set
"""

import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from graph.state import WealthManagerState
except ModuleNotFoundError:
    # Allow direct execution (python agents/orchestrator.py) by adding repo root.
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from graph.state import WealthManagerState

# ── Known ticker registry (extend as needed) ──────────────────────────────────
KNOWN_TICKERS: set[str] = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX",
    "AMD", "INTC", "ORCL", "CRM", "ADBE", "PYPL", "UBER", "LYFT", "SNAP",
    "JPM", "GS", "BAC", "WFC", "C", "MS", "BLK", "AXP",
    "JNJ", "PFE", "MRK", "ABBV", "LLY", "UNH",
    "XOM", "CVX", "COP", "SLB",
    "BRK", "V", "MA", "HD", "WMT", "TGT", "COST", "NKE",
    "DIS", "CMCSA", "T", "VZ",
    "SPY", "QQQ", "IWM", "GLD", "SLV", "BTC", "ETH",
}

_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")


def _extract_tickers_regex(text: str) -> list[str]:
    """Return known tickers found in text, preserving order, deduped."""
    seen: set[str] = set()
    result: list[str] = []
    for match in _TICKER_RE.finditer(text):
        t = match.group(1)
        if t in KNOWN_TICKERS and t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _extract_tickers_llm(text: str) -> list[str]:
    """
    Use an LLM to extract tickers when OPENAI_API_KEY is available.
    Falls back to regex on any error.
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Extract all stock ticker symbols mentioned in the user message. "
             "Return only a comma-separated list of uppercase tickers (e.g. AAPL, TSLA). "
             "If none are found, return an empty string."),
            ("human", "{text}"),
        ])
        response = (prompt | llm).invoke({"text": text})
        raw = response.content.strip()
        if not raw:
            return []
        return [t.strip().upper() for t in raw.split(",") if t.strip()]
    except Exception:
        return _extract_tickers_regex(text)


def extract_tickers(text: str) -> list[str]:
    """Extract tickers from text. Uses LLM if available, regex otherwise."""
    if os.environ.get("OPENAI_API_KEY"):
        tickers = _extract_tickers_llm(text)
        # Fill any gaps with regex (LLM may miss some)
        regex_tickers = _extract_tickers_regex(text)
        seen = set(tickers)
        for t in regex_tickers:
            if t not in seen:
                tickers.append(t)
        return tickers
    return _extract_tickers_regex(text)


# ── LangGraph node ─────────────────────────────────────────────────────────────

def orchestrator_node(state: WealthManagerState) -> dict:
    """
    Orchestrator entry node.
    Reads the latest user message, extracts tickers, and seeds the state
    for downstream agents.
    """
    print("--- 🎯 AGENT: ORCHESTRATOR ---")

    messages = state.get("messages") or []
    latest = str(messages[-1]) if messages else ""

    tickers = extract_tickers(latest)
    route_target = "sentiment_agent" if tickers else "report_generator_agent"

    if tickers:
        print(f"  Detected tickers: {tickers}")
    else:
        print("  No tickers detected — routing directly to report generator.")

    return {
        "tickers":  tickers,
        "route_target": route_target,
        "audit_iteration_count": 0,
        "messages": [
            f"Orchestrator: extracted tickers {tickers} from input.",
            f"Orchestrator: next route -> {route_target}",
        ],
    }
