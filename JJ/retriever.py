"""
Reusable RAG Retriever Module

Centralizes vector store management for:
- Analyst agent (existing)
- RAG Q&A agent (new)
- Future agents that need document retrieval

Uses local HuggingFace embeddings (free, CPU-friendly) with ChromaDB persistence.
"""

from pathlib import Path
from typing import Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class RagRetriever:
    """Manages document retrieval from ChromaDB with Apple 10-K data."""

    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "sec_10k"):
        """
        Initialize the RAG retriever.

        Args:
            db_path: Path to ChromaDB persistence directory.
            collection_name: Name of the ChromaDB collection (default: "sec_10k" from rag.py).
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None

    def load_or_build_vector_store(self, k: int = 3) -> None:
        """
        Load or build the ChromaDB vector store with local embeddings.

        Args:
            k: Number of top results to retrieve (default: 3).
        """
        # Initialize local embeddings (CPU-friendly)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False}
        )

        # Connect to ChromaDB (creates if doesn't exist, or loads existing)
        self.vectorstore = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )

        # Create retriever with k results
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

        # Verify data is loaded
        collection = self.vectorstore._collection
        doc_count = collection.count() if hasattr(collection, 'count') else 0
        print(f"✓ ChromaDB loaded from {self.db_path} (collection: '{self.collection_name}'): {doc_count} documents")

    def retrieve(
        self,
        query: str,
        k: int = 3
    ) -> list[dict]:
        """
        Retrieve top-k documents from the vector store.

        Args:
            query: User query or search text.
            k: Number of results to return (overrides initialization default).

        Returns:
            List of dicts with 'text' and 'metadata' keys.
            Format: [
                {
                    'text': 'chunk text...',
                    'metadata': {'chunk_index': 0, 'source': 'AAPL_10K_2025', ...}
                },
                ...
            ]
        """
        if self.retriever is None:
            raise RuntimeError(
                "Vector store not loaded. Call load_or_build_vector_store() first."
            )

        # Temporarily update k if different
        if k != 3:
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            )

        # Invoke retriever and format results
        docs = self.retriever.invoke(query)
        results = []

        for doc in docs:
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, "metadata") else {},
            })

        return results

    def retrieve_with_scores(
        self,
        query: str,
        k: int = 3
    ) -> list[dict]:
        """
        Retrieve top-k documents WITH similarity scores.

        Args:
            query: User query or search text.
            k: Number of results to return.

        Returns:
            List of dicts with 'text', 'score', and 'metadata' keys.
            Scores range from 0 to 1 (higher = more relevant).
        """
        if self.vectorstore is None:
            raise RuntimeError(
                "Vector store not loaded. Call load_or_build_vector_store() first."
            )

        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        results = []

        for doc, score in docs_with_scores:
            results.append({
                "text": doc.page_content,
                "score": score,
                "metadata": doc.metadata if hasattr(doc, "metadata") else {},
            })

        return results

    def format_context(
        self,
        results: list[dict],
        include_scores: bool = False,
        include_metadata: bool = True
    ) -> str:
        """
        Format retrieved chunks for LLM consumption.

        Args:
            results: Output from retrieve() or retrieve_with_scores().
            include_scores: If True, include similarity scores in output.
            include_metadata: If True, include source metadata (chunk_id, filing_type, etc.).

        Returns:
            Formatted string ready for LLM prompt.
        """
        formatted_chunks = []

        for i, result in enumerate(results, start=1):
            chunk_text = result.get("text", "").strip()
            score = result.get("score")
            metadata = result.get("metadata", {})

            # Build chunk header
            header = f"[Excerpt {i}]"
            if include_scores and score is not None:
                header += f" (relevance: {score:.3f})"
            if include_metadata and metadata:
                chunk_id = metadata.get("chunk_index", "?")
                company = metadata.get("company", "?")
                filing_type = metadata.get("filing_type", "?")
                header += f" | {company} {filing_type} (chunk #{chunk_id})"

            formatted_chunks.append(f"{header}\n{chunk_text}")

        return "\n\n".join(formatted_chunks)

    def get_stats(self) -> dict:
        """
        Get statistics about the loaded vector store.

        Returns:
            Dict with document count, model name, db path.
        """
        if self.vectorstore is None:
            return {"status": "not loaded"}

        collection = self.vectorstore._collection
        doc_count = collection.count() if hasattr(collection, 'count') else 0

        return {
            "model": self.model_name,
            "db_path": self.db_path,
            "document_count": doc_count,
        }


# ============================================================================
# Module-level convenience instance (optional, for quick access)
# ============================================================================

_default_retriever: Optional[RagRetriever] = None


def get_default_retriever(k: int = 3) -> RagRetriever:
    """
    Get or create a singleton default retriever instance.

    Args:
        k: Number of results to retrieve.

    Returns:
        RagRetriever instance, ready for use.
    """
    global _default_retriever

    if _default_retriever is None:
        _default_retriever = RagRetriever()
        _default_retriever.load_or_build_vector_store(k=k)

    return _default_retriever


# ============================================================================
# CLI Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing RagRetriever...\n")

    retriever = RagRetriever()
    retriever.load_or_build_vector_store(k=3)

    # Test retrieve with scores
    query = "What are Apple's key risk factors and growth drivers?"
    results = retriever.retrieve_with_scores(query, k=3)

    print(f"Query: {query}\n")
    formatted = retriever.format_context(results, include_scores=True)
    print(formatted)

    # Print stats
    stats = retriever.get_stats()
    print(f"\n✓ Stats: {stats}")
