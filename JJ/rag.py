from __future__ import annotations

from pathlib import Path
import json
import math
import re
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
CLEAN_TXT_PATH = BASE_DIR / "aapl_10k_clean.txt"
DB_PATH = BASE_DIR.parent / "chroma_db"
CHUNKS_PATH = BASE_DIR / "aapl_10k_chunks.jsonl"
EMBEDDINGS_PATH = BASE_DIR / "aapl_10k_embeddings.json"
COLLECTION_NAME = "sec_10k"


def load_clean_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cleaned 10-K text: {path}. Run JJ/data_download.py and the HTML cleaner first."
        )
    return path.read_text(encoding="utf-8").strip()


def chunk_document(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[dict[str, Any]]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: list[dict[str, Any]] = []

    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append({"text": current})

        if len(paragraph) <= chunk_size:
            current = paragraph
            continue

        start = 0
        step = max(chunk_size - overlap, 1)
        while start < len(paragraph):
            end = start + chunk_size
            chunks.append({"text": paragraph[start:end]})
            start += step
        current = ""

    if current:
        chunks.append({"text": current})

    for index, chunk in enumerate(chunks):
        chunk["id"] = f"aapl-10k-2025-{index:05d}"
        chunk["metadata"] = {
            "source": "AAPL_10K_2025",
            "company": "Apple",
            "filing_type": "10-K",
            "fiscal_year": "2025",
            "chunk_index": index,
        }

    return chunks


def save_chunks(chunks: list[dict[str, Any]]) -> None:
    with CHUNKS_PATH.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as error:
        raise RuntimeError(
            "sentence-transformers is required for embedding generation. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from error

    return SentenceTransformer(model_name), model_name


def build_embeddings(chunks: list[dict[str, Any]], model, model_name: str) -> list[list[float]]:
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = vectors.tolist() if hasattr(vectors, "tolist") else vectors
    payload = {
        "model": model_name,
        "count": len(chunks),
        "ids": [chunk["id"] for chunk in chunks],
        "metadatas": [chunk["metadata"] for chunk in chunks],
        "embeddings": embeddings,
    }
    EMBEDDINGS_PATH.write_text(json.dumps(payload), encoding="utf-8")
    return embeddings


def try_build_chroma_index(chunks: list[dict[str, Any]], embeddings: list[list[float]] | None) -> bool:
    if embeddings is None:
        return False

    try:
        import chromadb
        from chromadb.config import Settings
    except Exception:
        return False

    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH), settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    try:
        collection.delete(ids=ids)
    except Exception:
        pass

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return True


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def sanity_check_retrieval(chunks: list[dict[str, Any]], embeddings: list[list[float]], model) -> None:

    query = "What are Apple's key risk factors and growth drivers?"
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    query_vector = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding

    scored = []
    for chunk, vector in zip(chunks, embeddings):
        scored.append((cosine_similarity(query_vector, vector), chunk["text"]))
    scored.sort(key=lambda item: item[0], reverse=True)

    print("\n--- Retrieval sanity check ---")
    print(f"Query: {query}")
    for index, (score, text) in enumerate(scored[:2], start=1):
        preview = text[:240].replace("\n", " ")
        print(f"Result {index} (score={score:.4f}): {preview}...")


def main() -> None:
    text = load_clean_text(CLEAN_TXT_PATH)
    chunks = chunk_document(text)
    save_chunks(chunks)

    model, model_name = load_embedding_model()
    embeddings = build_embeddings(chunks, model, model_name)
    has_chroma = try_build_chroma_index(chunks, embeddings)

    print(f"Loaded text length: {len(text):,} chars")
    print(f"Created chunks: {len(chunks):,}")
    print(f"Saved chunk dataset at: {CHUNKS_PATH}")

    print(f"Saved embeddings at: {EMBEDDINGS_PATH}")

    if has_chroma:
        print(f"Persisted Chroma vector DB at: {DB_PATH}")
    else:
        print("Chroma index not created (install chromadb if you want vector DB persistence).")

    sanity_check_retrieval(chunks, embeddings, model)


if __name__ == "__main__":
    main()