from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


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


class BM25Retriever:
    def __init__(self, texts: list[str]):
        self.texts = texts
        self.tokens = [tokenize(text) for text in texts]
        self.model = try_build_bm25(self.tokens)

    def search(self, query: str, k: int) -> list[int]:
        query_tokens = tokenize(query)

        if self.model is not None:
            scores = self.model.get_scores(query_tokens)
            ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
            return [index for index, _ in ranked[:k]]

        query_set = set(query_tokens)
        fallback_scores: list[tuple[float, int]] = []
        for index, doc_tokens in enumerate(self.tokens):
            overlap = len(query_set.intersection(doc_tokens))
            fallback_scores.append((float(overlap), index))
        fallback_scores.sort(key=lambda item: item[0], reverse=True)
        return [index for _, index in fallback_scores[:k]]


def reciprocal_rank_fusion(*rank_lists: list[int], k: int = 60) -> list[int]:
    score_map: dict[int, float] = {}
    for ranking in rank_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            score_map[doc_id] = score_map.get(doc_id, 0.0) + 1.0 / (k + rank)
    return [doc_id for doc_id, _ in sorted(score_map.items(), key=lambda item: item[1], reverse=True)]


def rerank_with_cohere(question: str, retrieved_indices: list[int], chunks: list[dict[str, Any]], model_name: str = "rerank-v3.5") -> list[int]:
    """Rerank retrieved chunks using Cohere rerank API"""
    try:
        import cohere
    except ImportError:
        print("Cohere package not installed. Install with: pip install cohere")
        return retrieved_indices

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("COHERE_API_KEY not set. Skipping reranking.")
        return retrieved_indices

    client = cohere.Client(api_key)

    # Prepare documents for reranking
    docs = [chunks[idx]["text"] for idx in retrieved_indices]

    try:
        response = client.rerank(model=model_name, query=question, documents=docs, top_n=len(docs))

        # Reorder indices based on reranking
        reranked_indices = []
        for item in response.results:
            reranked_indices.append(retrieved_indices[item.index])

        return reranked_indices

    except Exception as e:
        print(f"Cohere reranking failed: {e}. Using original ranking.")
        return retrieved_indices


def rerank_with_cross_encoder(question: str, retrieved_indices: list[int], chunks: list[dict[str, Any]], model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1") -> list[int]:
    """Rerank retrieved chunks using local cross-encoder model (no API required)"""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("sentence-transformers package not installed. Install with: pip install sentence-transformers")
        return retrieved_indices

    try:
        cross_encoder = CrossEncoder(model_name)
        
        # Prepare documents for reranking
        docs = [chunks[idx]["text"] for idx in retrieved_indices]
        
        # Create query-document pairs
        pairs = [[question, doc] for doc in docs]
        
        # Score pairs
        scores = cross_encoder.predict(pairs)
        
        # Sort by score and get reranked indices
        scored_indices = list(zip(range(len(retrieved_indices)), scores))
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        reranked_indices = [retrieved_indices[i] for i, _ in scored_indices]
        return reranked_indices

    except Exception as e:
        print(f"Cross-encoder reranking failed: {e}. Using original ranking.")
        return retrieved_indices


def rerank_with_groq(question: str, retrieved_indices: list[int], chunks: list[dict[str, Any]], model_name: str = "mixtral-8x7b-32768") -> list[int]:
    """Rerank retrieved chunks using Groq LLM API (uses LLM to score relevance)"""
    try:
        from groq import Groq
    except ImportError:
        print("Groq package not installed. Install with: pip install groq")
        return retrieved_indices

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set. Skipping Groq reranking.")
        return retrieved_indices

    client = Groq(api_key=api_key)
    
    # Use full document text for better ranking (Groq has higher token limits)
    docs = [chunks[idx]["text"][:2000] for idx in retrieved_indices]  # Increased from 500 to 2000
    
    # Create prompt asking Groq to rank documents by relevance
    doc_text = "\n\n".join([f"[{i}] {doc[:800]}" for i, doc in enumerate(docs)])
    prompt = f"""You are a financial document ranking expert. For the given question, rank these documents by how relevant they are to answering it.

Question: {question}

Documents:
{doc_text}

Respond with ONLY a comma-separated list of indices in order of relevance (best first). Example: 2,0,1,3
No explanations. No markdown. Just indices."""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50,
        )
        
        ranking_text = response.choices[0].message.content.strip()
        print(f"Groq ranking response: {ranking_text}")  # Debug output
        
        # Parse indices from response
        ranked_positions = []
        for token in ranking_text.replace(",", " ").split():
            try:
                pos = int(token.strip())
                if 0 <= pos < len(retrieved_indices):
                    ranked_positions.append(pos)
            except ValueError:
                continue
        
        if ranked_positions:
            reranked = [retrieved_indices[pos] for pos in ranked_positions]
            # Add any missing indices at the end (shouldn't happen but safety check)
            for idx in retrieved_indices:
                if idx not in reranked:
                    reranked.append(idx)
            return reranked[:len(retrieved_indices)]
        else:
            print("Failed to parse Groq ranking response. Using original ranking.")
            return retrieved_indices

    except Exception as e:
        print(f"Groq reranking failed: {e}. Using original ranking.")
        return retrieved_indices


def get_llm(model_name: str):
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return None

    if not os.getenv("OPENAI_API_KEY"):
        return None

    return ChatOpenAI(model=model_name, temperature=0.0)


def generate_hyde_query(question: str, llm) -> str:
    if llm is None:
        return question

    prompt = (
        "Write a short hypothetical answer passage for retrieving supporting 10-K evidence.\n"
        f"Question: {question}"
    )
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else question


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


def run_ragas(results: list[dict[str, Any]], llm_model_name: str) -> dict[str, float] | None:
    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
        from langchain_openai import ChatOpenAI
    except Exception as e:
        print(f"  [RAGAS] Skipping — missing package: {e}")
        return None

    if not os.getenv("OPENAI_API_KEY"):
        print("  [RAGAS] Skipping — OPENAI_API_KEY not set")
        return None

    samples = []
    for row in results:
        if not row.get("ground_truth") or not row.get("answer"):
            continue
        samples.append(
            SingleTurnSample(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=row["contexts"],
                reference=row["ground_truth"],
            )
        )

    if not samples:
        print("  [RAGAS] Skipping — no valid samples (check ground_truth field in your QA file)")
        return None

    print(f"  [RAGAS] Evaluating {len(samples)} samples...")
    try:
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(model=llm_model_name, temperature=0.0)
        )
        dataset = EvaluationDataset(samples=samples)

        scores = evaluate(
            dataset=dataset,
            metrics=[
                LLMContextRecall(),
                Faithfulness(),
                FactualCorrectness(),
            ],
            llm=evaluator_llm,
        )

        score_dict = scores.to_pandas().mean(numeric_only=True).to_dict()

        if "faithfulness" in score_dict:
            score_dict["hallucination_rate"] = round(1.0 - float(score_dict["faithfulness"]), 4)

        final = {key: round(float(val), 4) for key, val in score_dict.items()}
        print(f"  [RAGAS] Done: {final}")
        return final

    except Exception as e:
        import traceback
        print(f"  [RAGAS] Evaluation failed: {e}")
        traceback.print_exc()
        return None


def run_variant(
    variant: str,
    chunks: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
    k: int,
    baseline_model: str,
    finance_model: str,
    llm_model: str,
    reranker: str,
    cohere_model: str,
    groq_model: str,
) -> dict[str, Any]:
    raw_texts = [chunk["text"] for chunk in chunks]
    contextual_texts = contextualize_chunks(chunks)
    llm = get_llm(llm_model)

    # Determine if this variant uses reranking
    use_rerank = variant.endswith("+rerank")
    base_variant = variant.replace("+rerank", "") if use_rerank else variant

    if base_variant == "baseline":
        dense = DenseRetriever(raw_texts, baseline_model)
        retrieve = lambda q: dense.search(q, k)
        active_texts = raw_texts
    elif base_variant == "hyde":
        dense = DenseRetriever(raw_texts, baseline_model)

        def retrieve(query: str):
            hyde_query = generate_hyde_query(query, llm)
            return dense.search(hyde_query, k)

        active_texts = raw_texts
    elif base_variant == "hybrid":
        dense = DenseRetriever(contextual_texts, baseline_model)
        bm25 = BM25Retriever(contextual_texts)

        def retrieve(query: str):
            dense_rank = dense.search(query, max(k, 15))
            bm25_rank = bm25.search(query, max(k, 15))
            fused = reciprocal_rank_fusion(dense_rank, bm25_rank)
            return fused[:k]

        active_texts = contextual_texts
    elif base_variant == "finance":
        dense = DenseRetriever(contextual_texts, finance_model)
        retrieve = lambda q: dense.search(q, k)
        active_texts = contextual_texts
    else:
        raise ValueError(f"Unknown variant: {base_variant}")

    rows: list[dict[str, Any]] = []
    hit_count = 0
    reciprocal_ranks = []
    precision_scores = []

    for qa in qa_rows:
        question = qa["question"]

        # Initial retrieval
        top_indices = retrieve(question)

        # Apply reranking if requested
        if use_rerank:
            if reranker == "cohere":
                top_indices = rerank_with_cohere(question, top_indices, chunks, cohere_model)
            elif reranker == "groq":
                top_indices = rerank_with_groq(question, top_indices, chunks, groq_model)
            elif reranker == "cross-encoder":
                top_indices = rerank_with_cross_encoder(question, top_indices, chunks)
            else:
                print(f"Unknown reranker '{reranker}'; using original ranking.")

        contexts = [active_texts[index] for index in top_indices]
        answer = generate_answer(question, contexts, llm)

        keywords = qa.get("gold_context_keywords", [])
        hit = context_hit(contexts, keywords)
        if hit:
            hit_count += 1

        # Compute MRR (Mean Reciprocal Rank)
        keyword_set = set(kw.lower() for kw in keywords)
        merged_context = "\n".join(contexts).lower()
        mrr_found = False
        for rank, context in enumerate(contexts, start=1):
            if any(kw in context.lower() for kw in keywords):
                reciprocal_ranks.append(1.0 / rank)
                mrr_found = True
                break
        if not mrr_found:
            reciprocal_ranks.append(0.0)

        # Compute Precision@K (fraction of top-k with keywords)
        relevant_in_topk = sum(1 for ctx in contexts if any(kw.lower() in ctx.lower() for kw in keywords))
        precision_at_k = relevant_in_topk / max(k, 1)
        precision_scores.append(precision_at_k)

        rows.append(
            {
                "variant": variant,
                "id": qa.get("id", ""),
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": qa.get("ground_truth", ""),
                "retrieved_indices": top_indices,
                "context_hit": hit,
            }
        )

    recall_at_k = hit_count / max(len(qa_rows), 1)
    precision_at_k = sum(precision_scores) / max(len(precision_scores), 1)
    mean_reciprocal_rank = sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1)
    f1_score = 2 * (precision_at_k * recall_at_k) / max(precision_at_k + recall_at_k, 1e-6)
    accuracy = recall_at_k  # Hit rate
    ragas_scores = run_ragas(rows, llm_model_name=llm_model)

    return {
        "variant": variant,
        "rows": rows,
        "summary": {
            "variant": variant,
            "num_questions": len(qa_rows),
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "mrr": mean_reciprocal_rank,
            "ragas": ragas_scores,
        },
    }


def write_outputs(output_dir: Path, all_results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for result in all_results:
        variant = result["variant"]

        detail_path = output_dir / f"{variant}_details.jsonl"
        with detail_path.open("w", encoding="utf-8") as handle:
            for row in result["rows"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary = result["summary"]
        ragas_scores = summary.get("ragas") or {}
        summary_rows.append(
            {
                "variant": variant,
                "num_questions": summary["num_questions"],
                "precision_at_k": summary.get("precision_at_k", ""),
                "recall_at_k": summary.get("recall_at_k", ""),
                "f1_score": summary.get("f1_score", ""),
                "accuracy": summary.get("accuracy", ""),
                "mrr": summary.get("mrr", ""),
            }
        )

    csv_path = output_dir / "comparison_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "num_questions",
                "precision_at_k",
                "recall_at_k",
                "f1_score",
                "accuracy",
                "mrr",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = output_dir / "comparison_summary.json"
    json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare RAG variants with Cohere reranking for investment analysis.")
    parser.add_argument("--chunks", type=Path, default=Path("JJ/data/processed/aapl_10k_chunks.jsonl"))
    parser.add_argument("--qa", type=Path, default=Path("JJ/data/manual_qa_template.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/rag_compare_rerank"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--variants",
        type=str,
        default="baseline+rerank,hyde+rerank,hybrid+rerank,finance+rerank",
        help="Comma-separated list: baseline+rerank, hyde+rerank, hybrid+rerank, finance+rerank",
    )
    parser.add_argument("--baseline-embedding", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--finance-embedding",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Set this to your finance-finetuned embedding checkpoint.",
    )
    parser.add_argument("--llm-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--reranker", type=str, default="cross-encoder", choices=["none", "cohere", "groq", "cross-encoder"], help="Choose reranker for +rerank variants")
    parser.add_argument("--cohere-model", type=str, default="rerank-v4.0-pro")
    parser.add_argument("--groq-model", type=str, default="llama-3.1-70b-versatile", help="Groq model for reranking (e.g., llama-3.1-70b-versatile, llama-3.1-405b-versatile)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunks = load_chunks(args.chunks)
    qa_rows = load_qa(args.qa)
    variants = [variant.strip() for variant in args.variants.split(",") if variant.strip()]

    all_results = []
    for variant in variants:
        print(f"Running variant: {variant}")
        result = run_variant(
            variant=variant,
            chunks=chunks,
            qa_rows=qa_rows,
            k=args.k,
            baseline_model=args.baseline_embedding,
            finance_model=args.finance_embedding,
            llm_model=args.llm_model,
            reranker=args.reranker,
            cohere_model=args.cohere_model,
            groq_model=args.groq_model,
        )
        all_results.append(result)
        summary = result['summary']
        print(f"\n[{variant}] Metrics@{args.k}:")
        print(f"  Precision: {summary.get('precision_at_k', 0):.3f}")
        print(f"  Recall:    {summary.get('recall_at_k', 0):.3f}")
        print(f"  F1-Score:  {summary.get('f1_score', 0):.3f}")
        print(f"  Accuracy:  {summary.get('accuracy', 0):.3f}")
        print(f"  MRR:       {summary.get('mrr', 0):.3f}")

        ragas_scores = summary.get("ragas")
        if ragas_scores:
            print(f"  RAGAS: {ragas_scores}")
        else:
            print(f"  RAGAS: skipped (missing package/API key)")

    write_outputs(args.output_dir, all_results)
    print(f"Saved outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()