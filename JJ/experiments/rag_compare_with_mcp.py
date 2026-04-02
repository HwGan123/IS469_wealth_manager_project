"""
RAG Benchmark with MCP Live Data Integration

Extends rag_compare.py to include MCP-fetched live financial data (news, earnings, analyst ratings).
Compares MCP-augmented RAG against baseline dense retrieval.

Usage:
    python JJ/experiments/rag_compare_with_mcp.py \
        --chunks JJ/aapl_10k_chunks.jsonl \
        --qa JJ/data/manual_qa_template.jsonl \
        --output-dir results/rag_compare_mcp \
        --tickers AAPL NVDA GOOG \
        --days-back 7
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mcp_news.dispatcher import dispatch_mcp_tool


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def contextualize_chunks(chunks: list[dict[str, Any]], window: int = 1) -> list[str]:
    texts = [chunk["text"] for chunk in chunks]
    contextualized: list[str] = []
    for index in range(len(texts)):
        left = max(0, index - window)
        right = min(len(texts), index + window + 1)
        contextualized.append("\n\n".join(texts[left:right]))
    return contextualized


def load_qa(qa_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with qa_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fetch_mcp_context(tickers: list[str], days_back: int = 7) -> dict[str, str]:
    """
    Fetch live financial data from MCP server.
    
    Args:
        tickers: List of stock tickers (e.g., ["AAPL", "NVDA"])
        days_back: Number of days to look back for news
    
    Returns:
        Dict mapping ticker to concatenated live data context
    """
    mcp_context = {}
    
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
        
        # Aggregate by ticker
        for ticker in tickers:
            context_parts = []
            
            # Add news for this ticker
            if isinstance(news_result, dict) and "articles" in news_result:
                ticker_news = [a for a in news_result["articles"] if a.get("ticker") == ticker]
                if ticker_news:
                    context_parts.append("=== RECENT NEWS ===")
                    for article in ticker_news[:3]:  # Top 3 articles
                        context_parts.append(f"- {article.get('title', '')}")
                        context_parts.append(f"  Sentiment: {article.get('sentiment_score', 'N/A')}")
            
            # Add earnings for this ticker
            if isinstance(earnings_result, dict) and "earnings" in earnings_result:
                ticker_earnings = [e for e in earnings_result["earnings"] if e.get("ticker") == ticker]
                if ticker_earnings:
                    context_parts.append("\n=== EARNINGS DATA ===")
                    for earning in ticker_earnings:
                        context_parts.append(f"PE Ratio: {earning.get('pe_ratio', 'N/A')}")
                        context_parts.append(f"EPS: {earning.get('eps', 'N/A')}")
                        context_parts.append(f"Forward EPS: {earning.get('forward_eps', 'N/A')}")
            
            # Add analyst ratings for this ticker
            if isinstance(ratings_result, dict) and "ratings" in ratings_result:
                ticker_ratings = [r for r in ratings_result["ratings"] if r.get("ticker") == ticker]
                if ticker_ratings:
                    context_parts.append("\n=== ANALYST RATINGS ===")
                    for rating in ticker_ratings[:2]:  # Top 2 ratings
                        context_parts.append(f"Target Price: {rating.get('target_price', 'N/A')}")
                        context_parts.append(f"Rating: {rating.get('rating', 'N/A')} ({rating.get('num_analysts', 0)} analysts)")
            
            if context_parts:
                mcp_context[ticker] = "\n".join(context_parts)
        
        print(f"✓ Fetched MCP data for {len(mcp_context)} tickers")
        return mcp_context
        
    except Exception as e:
        print(f"⚠️  Error fetching MCP data: {e}")
        return {}


def try_build_bm25(corpus_tokens: list[list[str]]):
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi(corpus_tokens)
    except Exception:
        return None


@dataclass
class DenseRetriever:
    texts: list[str]
    model_name: str

    def __post_init__(self):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as error:
            raise RuntimeError(
                "sentence-transformers is required for dense retrieval. "
                "Install it with: pip install sentence-transformers"
            ) from error

        self.model = SentenceTransformer(self.model_name)
        vectors = self.model.encode(self.texts, normalize_embeddings=True)
        self.matrix = vectors.tolist() if hasattr(vectors, "tolist") else vectors

    def search(self, query: str, k: int) -> list[int]:
        query_vector = self.model.encode(query, normalize_embeddings=True)
        query_vector = query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector

        scored: list[tuple[float, int]] = []
        for index, vec in enumerate(self.matrix):
            score = sum(a * b for a, b in zip(query_vector, vec))
            scored.append((score, index))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [index for _, index in scored[:k]]


def get_llm(model_name: str):
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return None

    if not os.getenv("OPENAI_API_KEY"):
        return None

    return ChatOpenAI(model=model_name, temperature=0.0)


def generate_answer(question: str, contexts: list[str], llm) -> str:
    if llm is None:
        return contexts[0][:700] if contexts else ""

    joined = "\n\n".join(contexts[:3])
    prompt = (
        "You are an investment analyst. Answer using only the retrieved context. "
        "If evidence is insufficient, say so explicitly.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{joined}"
    )
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else ""


def context_hit(retrieved_contexts: list[str], keywords: list[str]) -> bool:
    if not keywords:
        return False
    merged = "\n".join(retrieved_contexts).lower()
    return any(keyword.lower() in merged for keyword in keywords)


def run_variant(
    variant: str,
    chunks: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
    k: int,
    baseline_model: str,
    llm_model: str,
    mcp_context: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run a RAG variant (baseline or MCP-augmented)."""
    
    raw_texts = [chunk["text"] for chunk in chunks]
    contextual_texts = contextualize_chunks(chunks)
    llm = get_llm(llm_model)

    if variant == "mcp":
        # MCP-augmented variant
        if not mcp_context:
            print("⚠️  MCP context not available, falling back to baseline")
            variant = "baseline"
        else:
            # Prepend MCP context to first document
            augmented_texts = raw_texts.copy()
            if augmented_texts and mcp_context:
                # Combine all ticker contexts
                combined_mcp = "\n\n".join(mcp_context.values())
                augmented_texts[0] = f"=== LIVE MARKET DATA ===\n{combined_mcp}\n\n{augmented_texts[0]}"
            
            dense = DenseRetriever(augmented_texts, baseline_model)
            retrieve = lambda q: dense.search(q, k)
            active_texts = augmented_texts

    if variant == "baseline":
        dense = DenseRetriever(raw_texts, baseline_model)
        retrieve = lambda q: dense.search(q, k)
        active_texts = raw_texts

    rows: list[dict[str, Any]] = []
    hit_count = 0
    reciprocal_ranks = []
    precision_scores = []

    for qa in qa_rows:
        question = qa["question"]
        top_indices = retrieve(question)
        contexts = [active_texts[index] for index in top_indices]
        answer = generate_answer(question, contexts, llm)

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
            "answer": answer,
            "contexts": contexts,
            "ground_truth": qa.get("ground_truth", ""),
            "retrieved_indices": top_indices,
            "context_hit": hit,
        })

    recall_at_k = hit_count / max(len(qa_rows), 1)
    precision_at_k = sum(precision_scores) / max(len(precision_scores), 1)
    mean_reciprocal_rank = sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1)
    f1_score = 2 * (precision_at_k * recall_at_k) / max(precision_at_k + recall_at_k, 1e-6)
    accuracy = recall_at_k

    return {
        "variant": variant,
        "rows": rows,
        "summary": {
            "variant": variant,
            "precision_at_k": round(precision_at_k, 4),
            "recall_at_k": round(recall_at_k, 4),
            "f1_score": round(f1_score, 4),
            "accuracy": round(accuracy, 4),
            "mrr": round(mean_reciprocal_rank, 4),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="RAG Benchmark with MCP Integration")
    parser.add_argument("--chunks", type=Path, required=True, help="Path to chunks JSONL file")
    parser.add_argument("--qa", type=Path, required=True, help="Path to QA JSONL file")
    parser.add_argument("--output-dir", type=Path, default=Path("results/rag_compare_mcp"), help="Output directory")
    parser.add_argument("--k", type=int, default=5, help="Top-k for retrieval")
    parser.add_argument("--tickers", nargs="+", default=["AAPL"], help="Stock tickers for MCP fetch")
    parser.add_argument("--days-back", type=int, default=7, help="Days to look back for news")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--skip-mcp", action="store_true", help="Skip MCP and only run baseline")

    args = parser.parse_args()

    # Load data
    chunks = load_chunks(args.chunks)
    qa_rows = load_qa(args.qa)
    print(f"✓ Loaded {len(chunks)} chunks and {len(qa_rows)} QA pairs")

    # Fetch MCP data
    mcp_context = {} if args.skip_mcp else fetch_mcp_context(args.tickers, args.days_back)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run variants
    print("\n🚀 Running RAG variants...")
    variants_to_run = ["baseline"] if args.skip_mcp else ["baseline", "mcp"]
    results = []

    for variant in variants_to_run:
        print(f"\n  - Running {variant}...")
        result = run_variant(
            variant=variant,
            chunks=chunks,
            qa_rows=qa_rows,
            k=args.k,
            baseline_model="sentence-transformers/all-MiniLM-L6-v2",
            llm_model=args.llm_model,
            mcp_context=mcp_context if variant == "mcp" else None,
        )
        results.append(result)
        print(f"    ✓ {variant.upper()}: Precision={result['summary']['precision_at_k']}, Recall={result['summary']['recall_at_k']}, F1={result['summary']['f1_score']}")

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

    # Detailed results for first variant
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
