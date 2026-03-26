# JJ Folder

This folder is a mini pipeline for preparing Apple 2025 10-K data for RAG experiments.

## Files

- `data_download.py`  
  Downloads Apple’s 2025 10-K HTML filing from SEC EDGAR and saves it as `aapl_10k_2025.htm`.

- `aapl_10k_2025.htm`  
  Raw filing HTML document.

- `rag.py`  
  Builds a RAG-ready dataset from `aapl_10k_clean.txt` by:
  - splitting into chunks,
  - saving chunk dataset to `aapl_10k_chunks.jsonl`,
  - optionally generating embeddings,
  - optionally building a Chroma index.

- `aapl_10k_clean.txt`  
  Cleaned plain-text version of the filing (HTML/iXBRL removed).

- `aapl_10k_chunks.jsonl`  
  Chunked dataset for retrieval experiments. Each line contains:
  - `text`
  - `id`
  - `metadata` (source, company, filing type, year, chunk index)

- `simple_rag.py`
  Simple RAG retriever based on Topic 7 notebook flow:
  - bi-encoder embeddings (`all-MiniLM-L6-v2`)
  - cosine similarity retrieval
  - optional reranking (`cross-encoder` or `cohere`)

## Typical Workflow

1. Install dependencies
  pip install -r requirements.txt

2. Download filing:
   ```bash
   python JJ/data_download.py
   ```

3. Ensure `aapl_10k_clean.txt` is available (from your cleaning step).

4. Build chunks / optional embeddings:
   ```bash
   python JJ/rag.py
   ```

5. Run simple RAG retrieval:
  ```bash
  python JJ/simple_rag.py --question "What are Apple key risk factors?" --top-k 5
  ```

6. Optional reranking:
  ```bash
  python JJ/simple_rag.py --question "What are Apple key risk factors?" --top-k 5 --reranker cross-encoder
  ```

  For Cohere rerank, set `COHERE_API_KEY` first and use:
  ```bash
  python JJ/simple_rag.py --question "What are Apple key risk factors?" --top-k 5 --reranker cohere
  ```

## Notes

- `JJ` is currently a standalone data-prep area for RAG experiments.
- Main project retrieval can reuse `JJ/aapl_10k_chunks.jsonl` or the generated vector DB artifacts.
