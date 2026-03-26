from __future__ import annotations

from pathlib import Path
import json
import math
import re
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
CLEANED_FILES = [
    BASE_DIR / "aapl_10k_clean.txt",
    BASE_DIR / "amzn_10k_cleaned.txt",
    BASE_DIR / "goog_10k_cleaned.txt",
    BASE_DIR / "msft_10k_cleaned.txt",
    BASE_DIR / "nvda_10k_cleaned.txt",
]
DB_PATH = BASE_DIR.parent / "chroma_db"
COLLECTION_NAME = "sec_10k"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 150


def load_clean_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cleaned text: {path}. Run JJ/data_download.py and the HTML cleaner first."
        )
    return path.read_text(encoding="utf-8").strip()


def parse_source_metadata(path: Path) -> tuple[str, str, str, str]:
    stem = path.stem.lower()
    ticker = stem.split("_")[0]
    known = {
        "aapl": ("AAPL", "Apple", "10-K", "2025"),
        "amzn": ("AMZN", "Amazon", "10-K", "2025"),
        "goog": ("GOOG", "Google", "10-K", "2025"),
        "msft": ("MSFT", "Microsoft", "10-K", "2025"),
        "nvda": ("NVDA", "Nvidia", "10-K", "2025"),
    }
    if ticker in known:
        return known[ticker]

    fiscal_year = "2025"
    year_match = re.search(r"(20\d{2})", stem)
    if year_match:
        fiscal_year = year_match.group(1)

    company = ticker.upper()
    filing_type = "10-K" if "10k" in stem or "10-k" in stem else "unknown"
    return (ticker.upper(), company, filing_type, fiscal_year)


def chunk_document(
    text: str,
    source: str,
    company: str,
    filing_type: str,
    fiscal_year: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict[str, Any]]:
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
        chunk["id"] = f"{source}-{fiscal_year}-{index:05d}"
        chunk["metadata"] = {
            "source": f"{source}_{filing_type}_{fiscal_year}",
            "company": company,
            "filing_type": filing_type,
            "fiscal_year": fiscal_year,
            "chunk_index": index,
        }

    return chunks


def save_chunks(chunks: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
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


def build_embeddings(
    chunks: list[dict[str, Any]],
    model,
    model_name: str,
    embeddings_path: Path,
) -> list[list[float]]:
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
    embeddings_path.write_text(json.dumps(payload), encoding="utf-8")
    return embeddings


def try_build_chroma_index(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]] | None,
    collection_name: str = COLLECTION_NAME,
) -> bool:
    if embeddings is None:
        return False

    try:
        import chromadb
        from chromadb.config import Settings
    except Exception:
        return False

    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH), settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name=collection_name)

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
    model, model_name = load_embedding_model()

    for clean_path in CLEANED_FILES:
        if not clean_path.exists():
            print(f"Warning: cleaned file missing, skipping: {clean_path}")
            continue

        source, company, filing_type, fiscal_year = parse_source_metadata(clean_path)
        text = load_clean_text(clean_path)

        chunks = chunk_document(text, source, company, filing_type, fiscal_year)

        chunks_path = clean_path.with_name(f"{clean_path.stem}_chunks.jsonl")
        save_chunks(chunks, chunks_path)

        embeddings_path = clean_path.with_name(f"{clean_path.stem}_embeddings.json")
        embeddings = build_embeddings(chunks, model, model_name, embeddings_path)

        collection_name = f"{COLLECTION_NAME}_{source.lower()}"
        has_chroma = try_build_chroma_index(chunks, embeddings, collection_name=collection_name)

        print(f"\nProcessed {clean_path.name}")
        print(f"  source: {source} company: {company} filing: {filing_type} year: {fiscal_year}")
        print(f"  loaded text length: {len(text):,} chars")
        print(f"  created chunks: {len(chunks):,}")
        print(f"  saved chunks at: {chunks_path}")
        print(f"  saved embeddings at: {embeddings_path}")
        print(f"  chroma collection: {collection_name} (persisted: {has_chroma})")

        sanity_check_retrieval(chunks, embeddings, model)


if __name__ == "__main__":
    main()