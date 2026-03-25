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

## Typical Workflow

1. Download filing:
   ```bash
   python JJ/data_download.py
   ```

2. Ensure `aapl_10k_clean.txt` is available (from your cleaning step).

3. Build chunks / optional embeddings:
   ```bash
   python JJ/rag.py
   ```

## Notes

- `JJ` is currently a standalone data-prep area for RAG experiments.
- Main project retrieval can reuse `JJ/aapl_10k_chunks.jsonl` or the generated vector DB artifacts.
