from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


DEFAULT_CHUNKS_PATH = Path(__file__).resolve().parent / "aapl_10k_chunks.jsonl"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Load .env from project root (or current working directory)
load_dotenv()


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing chunks file: {path}. Run JJ/rag.py first.")

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No chunks found in {path}")
    return rows


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def retrieve(question: str, chunks: list[dict], model: SentenceTransformer, top_k: int) -> list[dict]:
    texts = [row["text"] for row in chunks]

    doc_embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    # Cosine similarity using normalized vectors = dot product
    scores = doc_embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, index in enumerate(top_indices, start=1):
        results.append(
            {
                "rank": rank,
                "score": float(scores[index]),
                "chunk": chunks[int(index)],
            }
        )
    return results


def rerank_with_cross_encoder(question: str, retrieved: list[dict], model_name: str) -> list[dict]:
    from sentence_transformers import CrossEncoder

    cross_encoder = CrossEncoder(model_name)
    pairs = [[question, row["chunk"]["text"]] for row in retrieved]
    scores = cross_encoder.predict(pairs)

    reranked = []
    for row, score in zip(retrieved, scores):
        reranked.append({**row, "rerank_score": float(score)})

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    for idx, item in enumerate(reranked, start=1):
        item["rank"] = idx
    return reranked


def rerank_with_cohere(question: str, retrieved: list[dict], model_name: str = "rerank-v3.5") -> list[dict]:
    import cohere

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY is not set.")

    client = cohere.Client(api_key)
    docs = [row["chunk"]["text"] for row in retrieved]

    response = client.rerank(model=model_name, query=question, documents=docs, top_n=len(docs))

    reranked = []
    for idx, item in enumerate(response.results, start=1):
        base = retrieved[item.index]
        reranked.append({**base, "rerank_score": float(item.relevance_score), "rank": idx})
    return reranked


def print_results(question: str, results: list[dict], max_chars: int = 300) -> None:
    print(f"\nQuestion: {question}")
    for row in results:
        chunk = row["chunk"]
        chunk_id = chunk.get("id", "unknown")
        text = chunk["text"].replace("\n", " ")
        text_preview = text[:max_chars] + ("..." if len(text) > max_chars else "")

        score_part = f"score={row['score']:.4f}"
        if "rerank_score" in row:
            score_part += f", rerank={row['rerank_score']:.4f}"

        print(f"\n[{row['rank']}] {chunk_id} ({score_part})")
        print(text_preview)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple RAG retrieval over Apple 10-K chunks.")
    parser.add_argument("--question", required=True, help="Question to retrieve context for")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument(
        "--reranker",
        choices=["none", "cross-encoder", "cohere"],
        default="none",
        help="Optional reranker after initial embedding retrieval",
    )
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER)
    parser.add_argument("--cohere-model", default="rerank-v3.5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunks = load_chunks(args.chunks)
    model = SentenceTransformer(args.embed_model)

    retrieved = retrieve(args.question, chunks, model, args.top_k)

    if args.reranker == "cross-encoder":
        retrieved = rerank_with_cross_encoder(args.question, retrieved, args.cross_encoder_model)
    elif args.reranker == "cohere":
        retrieved = rerank_with_cohere(args.question, retrieved, args.cohere_model)

    print_results(args.question, retrieved)


if __name__ == "__main__":
    main()
