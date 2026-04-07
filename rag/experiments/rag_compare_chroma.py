"""
RAG Benchmark with MCP - Using Pre-Built Chroma DB

Uses existing Chroma embeddings, so no sentence-transformers needed.
Compares baseline 10-K retrieval vs MCP-augmented retrieval.

Usage:
    python rag/experiments/rag_compare_chroma.py \
        --qa rag/data/manual_qa_template.jsonl \
        --output-dir results/rag_compare_mcp \
        --tickers AAPL NVDA GOOG MSFT
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from mcp_news.dispatcher import dispatch_mcp_tool


def load_qa(qa_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with qa_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fetch_mcp_context(tickers: list[str], days_back: int = 7) -> str:
    """Fetch live financial data from MCP server and return as combined context."""
    
    print(f"\n📡 Fetching MCP live data for tickers: {tickers}")
    
    try:
        # Fetch news
        print("  - Fetching news...")
        news_result = dispatch_mcp_tool("fetch_news", {
            "tickers": tickers,
            "days_back": days_back
        })
        
        # Fetch earnings
        print("  - Fetching earnings...")
        earnings_result = dispatch_mcp_tool("fetch_earnings", {
            "tickers": tickers
        })
        
        # Fetch analyst ratings
        print("  - Fetching analyst ratings...")
        ratings_result = dispatch_mcp_tool("fetch_analyst_ratings", {
            "tickers": tickers
        })
        
        context_parts = []
        
        # Add news
        if isinstance(news_result, dict) and "articles" in news_result:
            context_parts.append("=== RECENT FINANCIAL NEWS ===")
            for article in news_result["articles"][:5]:
                context_parts.append(f"- {article.get('title', '')} ({article.get('ticker', '')})")
                context_parts.append(f"  Sentiment: {article.get('sentiment_score', 'N/A')}")
        
        # Add earnings
        if isinstance(earnings_result, dict) and "earnings" in earnings_result:
            context_parts.append("\n=== EARNINGS DATA ===")
            for earning in earnings_result["earnings"]:
                context_parts.append(f"{earning.get('ticker')}: PE={earning.get('pe_ratio', 'N/A')}, EPS={earning.get('eps', 'N/A')}")
        
        # Add ratings
        if isinstance(ratings_result, dict) and "ratings" in ratings_result:
            context_parts.append("\n=== ANALYST RATINGS ===")
            for rating in ratings_result["ratings"][:3]:
                context_parts.append(f"{rating.get('ticker')}: {rating.get('rating', 'N/A')} (Target: {rating.get('target_price', 'N/A')})")
        
        combined_context = "\n".join(context_parts) if context_parts else ""
        print(f"✓ Fetched MCP data")
        return combined_context
        
    except Exception as e:
        print(f"⚠️  Error fetching MCP data: {e}")
        return ""


def context_hit(contexts: list[str], keywords: list[str]) -> bool:
    if not keywords:
        return False
    merged = "\n".join(contexts).lower()
    return any(keyword.lower() in merged for keyword in keywords)


def run_variant(
    variant: str,
    vectorstore: Chroma,
    qa_rows: list[dict[str, Any]],
    k: int,
    mcp_context: str = "",
) -> dict[str, Any]:
    """Run a RAG variant."""
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    rows = []
    hit_count = 0
    reciprocal_ranks = []
    precision_scores = []

    for qa in qa_rows:
        question = qa["question"]
        
        # Retrieve from Chroma
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        
        # Add MCP context if variant is "mcp"
        if variant == "mcp" and mcp_context:
            contexts = [mcp_context] + contexts[:k-1]
        
        keywords = qa.get("gold_context_keywords", [])
        hit = context_hit(contexts, keywords)
        if hit:
            hit_count += 1
        
        # Compute MRR
        mrr_found = False
        for rank, context in enumerate(contexts, start=1):
            if any(kw.lower() in context.lower() for kw in keywords):
                reciprocal_ranks.append(1.0 / rank)
                mrr_found = True
                break
        if not mrr_found:
            reciprocal_ranks.append(0.0)
        
        # Compute Precision@K
        relevant_in_topk = sum(1 for ctx in contexts if any(kw.lower() in ctx.lower() for kw in keywords))
        precision_at_k = relevant_in_topk / max(k, 1)
        precision_scores.append(precision_at_k)

        rows.append({
            "variant": variant,
            "id": qa.get("id", ""),
            "question": question,
            "contexts": contexts[:3],
            "ground_truth": qa.get("ground_truth", ""),
            "context_hit": hit,
        })

    recall_at_k = hit_count / max(len(qa_rows), 1)
    precision_avg = sum(precision_scores) / max(len(precision_scores), 1)
    mean_reciprocal_rank = sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1)
    f1_score = 2 * (precision_avg * recall_at_k) / max(precision_avg + recall_at_k, 1e-6)

    return {
        "variant": variant,
        "rows": rows,
        "summary": {
            "variant": variant,
            "precision_at_k": round(precision_avg, 4),
            "recall_at_k": round(recall_at_k, 4),
            "f1_score": round(f1_score, 4),
            "accuracy": round(recall_at_k, 4),
            "mrr": round(mean_reciprocal_rank, 4),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="RAG Benchmark with Chroma DB + MCP")
    parser.add_argument("--qa", type=Path, required=True, help="Path to QA JSONL file")
    parser.add_argument("--output-dir", type=Path, default=Path("results/rag_compare_mcp"), help="Output directory")
    parser.add_argument("--db-path", type=Path, default=Path("./chroma_db"), help="Path to Chroma DB")
    parser.add_argument("--k", type=int, default=5, help="Top-k for retrieval")
    parser.add_argument("--tickers", nargs="+", default=["AAPL"], help="Stock tickers for MCP fetch")
    parser.add_argument("--days-back", type=int, default=7, help="Days to look back for news")
    parser.add_argument("--skip-mcp", action="store_true", help="Skip MCP fetching")

    args = parser.parse_args()

    # Load QA data
    qa_rows = load_qa(args.qa)
    print(f"✓ Loaded {len(qa_rows)} QA pairs")

    # Load Chroma DB
    print(f"📚 Loading Chroma DB from {args.db_path}...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = Chroma(
        persist_directory=str(args.db_path),
        embedding_function=embeddings,
        collection_name="sec_10k"
    )
    print(f"✓ Loaded Chroma DB")

    # Fetch MCP context
    mcp_context = "" if args.skip_mcp else fetch_mcp_context(args.tickers, args.days_back)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run variants
    print("\n🚀 Running RAG variants...")
    variants_to_run = ["baseline"] if args.skip_mcp else ["baseline", "mcp"]
    results = []

    for variant in variants_to_run:
        print(f"  - Running {variant.upper()}...")
        result = run_variant(
            variant=variant,
            vectorstore=vectorstore,
            qa_rows=qa_rows,
            k=args.k,
            mcp_context=mcp_context if variant == "mcp" else "",
        )
        results.append(result)
        summary = result['summary']
        print(f"    Precision@{args.k}: {summary['precision_at_k']}, Recall: {summary['recall_at_k']}, F1: {summary['f1_score']}")

    # Save results
    print(f"\n💾 Saving results to {args.output_dir}...")
    
    # Summary CSV
    with (args.output_dir / "comparison_summary.csv").open("w", newline="") as f:
        fieldnames = ["variant", "precision_at_k", "recall_at_k", "f1_score", "accuracy", "mrr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result["summary"])
    
    # Summary JSON
    summary_data = [result["summary"] for result in results]
    with (args.output_dir / "comparison_summary.json").open("w") as f:
        json.dump(summary_data, f, indent=2)

    # Detailed results
    if results:
        with (args.output_dir / "baseline_details.jsonl").open("w") as f:
            for row in results[0]["rows"]:
                f.write(json.dumps(row) + "\n")

    print(f"✓ Results saved!")
    print(f"\n📊 Summary:")
    for result in results:
        summary = result["summary"]
        print(f"  {summary['variant'].upper()}:")
        print(f"    Precision@{args.k}: {summary['precision_at_k']}")
        print(f"    Recall@{args.k}: {summary['recall_at_k']}")
        print(f"    F1-Score: {summary['f1_score']}")
        print(f"    MRR: {summary['mrr']}")


if __name__ == "__main__":
    main()
