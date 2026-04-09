#!/usr/bin/env python3
"""
Experiment: Hallucination Detection by Context Type
=====================================================

Uses ONLY Hybrid RAG experiment output (50 questions) to test:
1. 10-K ONLY - Combined historical 10-K from AAPL, AMZN, GOOG, MSFT, NVDA (All context)
2. 10-K + LIVE DATA - All 10K context + real-time market data (Finnhub news)
3. LIVE DATA ONLY - Only real-time market data from Finnhub (most recent 7 days)

Key Optimization: Live data is fetched ONCE and cached (no 50 API calls!)

Result: Compare which context source best detects hallucinations

Hypothesis: 10K + Live Data should provide best hallucination detection
because it has both historical accuracy and real-time market context.
Trade-off: More context = slower audits, but better detection accuracy
"""

import json
import os
import sys
import csv
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime
import signal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.auditor import AuditorAgent, RAGASCalculator
from mcp_news.implementations import fetch_news
from dotenv import load_dotenv

load_dotenv()



# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_hybrid_rag_output(filepath: str) -> List[Dict]:
    """Load Hybrid RAG output (50 questions with answers and contexts)."""
    try:
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        print(f"✓ Loaded {len(data)} Hybrid RAG results from {filepath}")
        return data
    except Exception as e:
        print(f"✗ Error loading Hybrid RAG output: {e}")
        return []


def get_10k_context() -> str:
    """
    Load combined historical 10-K data from rag/data/processed folder.
    
    Sources: AAPL, AMZN, GOOG, MSFT, NVDA (comprehensive tech sector context)
    Chunks: All available chunks from each company (up to 50K char limit)
    
    Returns: Combined historical 10-K context for hallucination detection
    """
    rag_data_path = Path(__file__).parent.parent.parent / "rag" / "data" / "processed"
    
    context = ""
    # Load 10K chunks from multiple tech companies for comprehensive context
    chunk_files = [
        "aapl_10k_chunks.jsonl",
        "amzn_10k_cleaned_chunks.jsonl",
        "goog_10k_cleaned_chunks.jsonl",
        "msft_10k_cleaned_chunks.jsonl",
        "nvda_10k_cleaned_chunks.jsonl"
    ]
    
    for chunk_file in chunk_files:
        filepath = rag_data_path / chunk_file
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    chunk = json.loads(line)
                    text = chunk.get("text", "")
                    if text:
                        context += text + "\n\n---\n\n"
        except FileNotFoundError:
            continue
        except Exception as e:
            continue
    
    return context[:50000]  # Truncate to 50K chars to keep context manageable


def get_live_data_context(ticker: str = "AAPL") -> str:
    """
    Fetch live market data from Finnhub for any ticker.
    
    Parameters:
    - ticker: Stock ticker symbol (e.g., "AAPL", "AMZN", "GOOG", "MSFT", "NVDA")
    
    Returns: Formatted news articles from the last 7 days for the given ticker
    
    Used by: run_experiment() to fetch live data for all tech companies
    """
    try:
        news_data = fetch_news([ticker], days_back=7)
        
        if "error" in news_data:
            return f"[Live data unavailable: {news_data['error']}]"
        
        articles = news_data.get("articles", [])
        if not articles:
            return "[No recent news articles available]"
        
        # Format news articles as context
        live_context = "Recent Market News and Updates:\n\n"
        for article in articles[:5]:  # Use top 5 most recent articles
            live_context += f"Title: {article.get('headline', 'N/A')}\n"
            live_context += f"Summary: {article.get('summary', 'N/A')[:200]}...\n"
            live_context += f"Source: {article.get('source', 'N/A')} | {article.get('datetime', 'N/A')}\n\n"
        
        return live_context
    except Exception as e:
        return f"[Error fetching live data: {str(e)}]"


# ─────────────────────────────────────────────────────────────────────────────
# AUDITOR VARIANT TESTING
# ─────────────────────────────────────────────────────────────────────────────

def create_auditor(context_type: str) -> AuditorAgent:
    """
    Create auditor instance for hallucination detection.
    
    Uses GPT-4o-mini with temperature=0 for deterministic auditing.
    (Temperature set internally in AuditorAgent.__init__)
    """
    # AuditorAgent automatically loads OPENAI_API_KEY from environment
    auditor = AuditorAgent()
    return auditor


def audit_with_context(
    auditor: AuditorAgent,
    question: str,
    draft_answer: str,
    context: str,
    context_type: str,
    timeout: int = 60
) -> Dict:
    """
    Run auditor with specific context and return results.
    
    Parameters:
    - auditor: AuditorAgent instance (GPT-4o-mini, temperature=0)
    - question: Original question from Hybrid RAG dataset
    - draft_answer: RAG-generated answer to audit
    - context: Reference context (10-K, live data, or combined)
    - context_type: Label for this context variant ("10k_only", "10k_live", "live_only")
    - timeout: Max seconds to wait for API response (default 60)
    
    Process:
    1. Extract claims from draft answer
    2. Verify each claim against context
    3. Calculate RAGAS metrics (faithfulness, answer_relevancy, context_recall)
    4. Return hallucination count and status
    """
    try:
        # Run audit with timeout handling
        result = auditor.audit_draft(
            context=context,
            draft=draft_answer
        )
        
        # Extract hallucination detection from AuditReport
        hallucination_count = result.hallucination_count
        is_hallucinating = hallucination_count > 0
        
        return {
            "question": question,
            "context_type": context_type,
            "draft_answer": draft_answer,
            "is_hallucinating": is_hallucinating,
            "hallucination_count": hallucination_count,
            "audit_result": result
        }
    except Exception as e:
        print(f"✗ Audit error ({context_type}): {str(e)[:100]}")
        return {
            "question": question,
            "context_type": context_type,
            "is_hallucinating": False,
            "hallucination_count": 0,
            "error": str(e)
        }


def run_experiment(hybrid_output: List[Dict]) -> Dict:
    """
    Run full experiment: compare hallucination detection across 3 context variants.
    
    Tests each of the 50 Hybrid RAG questions with 3 different context sources:
    1. 10-K ONLY: Combined historical 10-K from AAPL, AMZN, GOOG, MSFT, NVDA (all available)
    2. 10-K + LIVE: Combined 10-K + Finnhub news (last 7 days) - live data cached
    3. LIVE DATA ONLY: Only Finnhub news (last 7 days, no historical context) - live data cached
    
    Live data is fetched ONCE and cached to avoid 50 API calls
    
    Returns: Dict with hallucination detection results per variant
    """
    print("\n" + "="*80)
    print("STARTING EXPERIMENT: Hallucination Detection by Context Type")
    print("Using Hybrid RAG output (50 questions) evaluated with 3 context sources")
    print("="*80)
    
    # Load 10K context once (it's the same for all 50 questions)
    context_10k = get_10k_context()
    print(f"\n✓ Loaded combined historical 10K context: {len(context_10k):,} characters")
    print("  Sources: AAPL, AMZN, GOOG, MSFT, NVDA (all available chunks)")
    
    # Fetch and cache live data once for ALL companies instead of 50 times
    print("\n✓ Fetching live market data for all companies (cached for all 50 questions)...")
    tickers = ["AAPL", "AMZN", "GOOG", "MSFT", "NVDA"]
    context_live = ""
    for ticker in tickers:
        ticker_news = get_live_data_context(ticker)
        if ticker_news and "[Live data unavailable" not in ticker_news and "[No recent" not in ticker_news:
            context_live += f"[{ticker}]\n{ticker_news}\n"
    
    # Keep combined context for maximum accuracy
    context_combined = f"{context_10k}\n\n[LIVE MARKET DATA]\n{context_live}"
    
    auditor_10k = create_auditor("10k_only")
    auditor_hybrid = create_auditor("10k_live")
    auditor_live = create_auditor("live_only")
    
    results = {
        "10k_only": {"detected": 0, "total": 0, "hallucinations": []},
        "10k_live": {"detected": 0, "total": 0, "hallucinations": []},
        "live_only": {"detected": 0, "total": 0, "hallucinations": []},
    }
    
    # Test each question with all 3 context variants (using cached live data)
    for idx in range(len(hybrid_output)):
        hybrid_result = hybrid_output[idx]
        question = hybrid_result.get("question", "")
        draft_answer = hybrid_result.get("answer", "")
        
        print(f"\n[{idx + 1}/{len(hybrid_output)}] Q: {question[:80]}...")
        
        # Test variant 1: 10-K only
        print("  Testing: 10-K context only...", end=" ", flush=True)
        result_10k = audit_with_context(
            auditor_10k,
            question,
            draft_answer,
            context_10k,
            "10k_only"
        )
        results["10k_only"]["total"] += 1
        if result_10k.get("is_hallucinating"):
            results["10k_only"]["detected"] += 1
            results["10k_only"]["hallucinations"].append({
                "question": question,
                "count": result_10k.get("hallucination_count", 0)
            })
        print(f"{'✗ HALLUCINATION' if result_10k.get('is_hallucinating') else '✓ OK'}")
        
        # Test variant 2: 10-K + Live Data (using combined context)
        print("  Testing: 10-K + Live Data...", end=" ", flush=True)
        result_hybrid = audit_with_context(
            auditor_hybrid,
            question,
            draft_answer,
            context_combined,
            "10k_live"
        )
        results["10k_live"]["total"] += 1
        if result_hybrid.get("is_hallucinating"):
            results["10k_live"]["detected"] += 1
            results["10k_live"]["hallucinations"].append({
                "question": question,
                "count": result_hybrid.get("hallucination_count", 0)
            })
        print(f"{'✗ HALLUCINATION' if result_hybrid.get('is_hallucinating') else '✓ OK'}")
        
        # Test variant 3: Live Data only (using cached data)
        print("  Testing: Live Data only...", end=" ", flush=True)
        result_live = audit_with_context(
            auditor_live,
            question,
            draft_answer,
            context_live,
            "live_only"
        )
        results["live_only"]["total"] += 1
        if result_live.get("is_hallucinating"):
            results["live_only"]["detected"] += 1
            results["live_only"]["hallucinations"].append({
                "question": question,
                "count": result_live.get("hallucination_count", 0)
            })
        print(f"{'✗ HALLUCINATION' if result_live.get('is_hallucinating') else '✓ OK'}")
    
    return results


def generate_csv_report(results: Dict, output_path: str):
    """
    Generate CSV report comparing hallucination detection across context types.
    
    3 Context Variants:
    - 10k_only: Combined historical 10-K (AAPL, AMZN, GOOG, MSFT, NVDA)
    - 10k_live: Combined 10-K + Finnhub news (last 7 days)
    - live_only: Finnhub news only (real-time market data, last 7 days)
    
    Metrics:
    - Total Questions: 50 from Hybrid RAG output
    - Hallucinations Detected: Count of drafts flagged as hallucinating
    - Detection Rate: % of drafts with hallucinations detected
    - Quality Score: 100% - Detection Rate (higher = better, means fewer hallucinations)
    
    Output: CSV file to results/context_variant_experiment/
    """
    csv_file = f"{output_path}/context_variant_results.csv"
    
    # Calculate statistics
    stats = []
    for context_type in ["10k_only", "10k_live", "live_only"]:
        data = results[context_type]
        total = data["total"]
        detected = data["detected"]
        detection_rate = (detected / total * 100) if total > 0 else 0
        avg_hallucinations = sum(h["count"] for h in data["hallucinations"]) / detected if detected > 0 else 0
        
        stats.append({
            "Context Type": context_type.replace("_", " ").title(),
            "Total Questions": total,
            "Hallucinations Detected": detected,
            "Detection Rate (%)": f"{detection_rate:.1f}%",
            "Avg Hallucinations per Question": f"{avg_hallucinations:.2f}",
            "Quality Score (higher is better)": f"{(100 - detection_rate):.1f}"
        })
    
    # Write CSV
    os.makedirs(output_path, exist_ok=True)
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)
    
    print(f"\n✓ CSV report saved to {csv_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    for row in stats:
        print(f"\n{row['Context Type']}:")
        print(f"  Total Questions: {row['Total Questions']}")
        print(f"  Hallucinations Detected: {row['Hallucinations Detected']}")
        print(f"  Detection Rate: {row['Detection Rate (%)']}")
        print(f"  Avg Hallucinations: {row['Avg Hallucinations per Question']}")
        print(f"  Quality Score: {row['Quality Score (higher is better)']}")
    
    # Determine winner
    print("\n" + "="*80)
    detection_rates = {s["Context Type"]: float(s["Detection Rate (%)"].rstrip('%')) for s in stats}
    best_detection = max(detection_rates, key=detection_rates.get)
    best_quality = max(detection_rates, key=lambda x: 100 - detection_rates[x])
    
    print(f"\n🏆 Best Hallucination Detection: {best_detection}")
    print(f"🥇 Best Overall Quality: {best_quality} (lowest hallucination rate)")
    print("="*80)




def main():
    """
    Main experiment runner.
    
    Steps:
    1. Load Hybrid RAG output (50 questions with answers)
    2. Run hallucination detection audit for each question with 3 context variants
    3. Generate CSV report comparing hallucination detection rates
    4. Output results to: results/context_variant_experiment/context_variant_results.csv
    """
    # Load Hybrid RAG output
    hybrid_file = "results/rag_compare/hybrid/hybrid_details.jsonl"
    hybrid_output = load_hybrid_rag_output(hybrid_file)
    
    if not hybrid_output:
        print("✗ Failed to load Hybrid RAG output. Exiting.")
        sys.exit(1)
    
    # Run experiment
    results = run_experiment(hybrid_output)
    
    # Generate CSV report
    output_dir = "results/context_variant_experiment"
    generate_csv_report(results, output_dir)


if __name__ == "__main__":
    main()
