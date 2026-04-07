# IS469 Wealth Manager Project - RAG Investment Analysis

## Overview
Multi-variant RAG (Retrieval Augmented Generation) system for investment analysis on financial documents (10-K filings). Compares baseline dense retrieval against advanced variants (HyDE, hybrid BM25+dense, finance-tuned embeddings) with comprehensive quantitative evaluation metrics.

## Quick Start

### Environment Setup
```bash
# Create RAG environment (Python 3.9+)
python3 -m venv .venv_rag
.venv_rag/bin/pip install -q sentence-transformers==2.7.0 rank-bm25==0.2.2 numpy\<2

# Or activate existing setup
source .venv_rag/bin/activate
```

### Run Baseline RAG (50-Question Evaluation)
```bash
.venv_rag/bin/python JJ/experiments/rag_compare.py \
  --chunks JJ/aapl_10k_chunks.jsonl \
  --qa JJ/data/manual_qa_template.jsonl \
  --output-dir results/rag_compare \
  --variants baseline \
  --k 5
```

## Results

### Baseline RAG Performance (Recall@5)
| Metric | Value | Definition |
|--------|-------|-----------|
| **Precision@5** | 0.452 | Among top-5 chunks, % containing gold keywords |
| **Recall@5** | 0.840 | % of questions with ≥1 keyword match in top-5 |
| **F1-Score** | 0.588 | Harmonic mean of precision & recall |
| **Accuracy** | 0.840 | Hit rate (questions answered) |
| **MRR** | 0.622 | Mean Reciprocal Rank of first relevant result |

**Interpretation:**
- ✅ Strong recall (84%): Retrieves relevant docs for most queries
- ⚠️ Moderate precision (45.2%): ~half of top-5 chunks match keywords
- ℹ️ First relevant result appears at avg position 1.6 in top-5

### Run All Variants
```bash
.venv_rag/bin/python JJ/experiments/rag_compare.py \
  --variants baseline,hyde,hybrid,finance \
  --qa JJ/data/manual_qa_template.jsonl \
  --output-dir results/rag_compare \
  --k 5
```

**Variant Options:**
- `baseline` – Dense retrieval only (`all-MiniLM-L6-v2`)
- `hyde` – Hypothetical Document Embeddings (requires `OPENAI_API_KEY`)
- `hybrid` – Reciprocal Rank Fusion over BM25 + dense retrieval
- `finance` – Finance-tuned embedding model (provide via `--finance-embedding`)

## Reranking with Cross-Encoder

### Run All Variants with Reranking
```bash
.venv_rag/bin/python JJ/experiments/rag_compare_rerank.py \
  --variants baseline,hyde,hybrid,finance \
  --llm-model gpt-3.5-turbo
```

### Run Specific Variants with Reranking
```bash
# Hybrid with reranking
.venv_rag/bin/python JJ/experiments/rag_compare_rerank.py \
  --variants hybrid \
  --llm-model gpt-3.5-turbo

# Finance with reranking
.venv_rag/bin/python JJ/experiments/rag_compare_rerank.py \
  --variants finance \
  --llm-model gpt-3.5-turbo
```

### Reranking Options
- `--reranker cross-encoder` (default, local reranking - no API cost)
- `--reranker none` (baseline without reranking)
- `--k 5` (retrieve top 5 documents before reranking)
- `--output-dir results/rag_compare_rerank_cross_encoder` (auto-created)

### Results Comparison
Reranking uses local cross-encoder (`mmarco-mMiniLMv2-L12-H384-v1`) to reorder retrieved documents. Results saved to:
- `results/rag_compare_rerank_cross_encoder/comparison_summary.json` (metrics)
- `results/rag_compare_rerank_cross_encoder/{variant}+rerank_details.jsonl` (per-question details)

**Recommendation:**
- 🏆 **Best overall**: Hybrid (no reranking) – F1: 0.8348
- 🥈 **Alternative**: Finance + reranking – F1: 0.8103 with improved MRR (0.8400)
- Reranking adds ~2-3 seconds overhead per 50 questions

## Output Files
- `results/rag_compare/comparison_summary.csv` – Metrics per variant (Precision, Recall, F1, Accuracy, MRR)
- `results/rag_compare/comparison_summary.json` – Same metrics in JSON format
- `results/rag_compare/baseline_details.jsonl` – Per-question results with retrieved text

## Data Pipeline

### Preparation Steps
1. **Download**: Apple 10-K HTML → `JJ/aapl_10k_2025.htm`
2. **Clean**: Extract readable text → `JJ/aapl_10k_clean.txt`
3. **Chunk**: Split into 1000-char paragraphs (150-char overlap) → `JJ/aapl_10k_chunks.jsonl`
4. **Embed**: Generate 384-dim normalized embeddings → `JJ/aapl_10k_clean_embeddings.json`

### QA Dataset
- **Location**: `JJ/data/manual_qa_template.jsonl`
- **Format**: JSON Lines (one entry per line)
- **Schema**: `{"id", "question", "ground_truth", "gold_context_keywords"}`
- **Size**: 50 representative questions covering risk, revenue, compliance, operations

## Optional: RAGAS Evaluation (Requires OpenAI API)
For hallucination rate, context precision, and context recall metrics:

```bash
# Set OpenAI API key
export OPENAI_API_KEY="sk-..."

# Install RAGAS (optional)
.venv_rag/bin/pip install ragas datasets langchain-openai

# Re-run with RAGAS enabled
.venv_rag/bin/python JJ/experiments/rag_compare.py \
  --variants baseline \
  --llm-model gpt-4o-mini
```

## Project Structure
```
.
├── JJ/
│   ├── aapl_10k_clean.txt              # Cleaned 10-K text
│   ├── aapl_10k_chunks.jsonl           # Paragraph chunks
│   ├── aapl_10k_clean_embeddings.json  # Dense embeddings
│   ├── data/
│   │   └── manual_qa_template.jsonl    # 50-question eval set
│   ├── experiments/
│   │   └── rag_compare.py              # Variant comparison harness
│   ├── rag.py                          # Embedding pipeline
│   └── simple_rag.py                   # Single-query RAG tester
├── agents/                             # LLM agents (portfolio, analyst)
├── graph/                              # LangGraph workflow orchestration
└── results/                            # Outputs (not tracked)
```

## Metric Definitions

- **Precision@K**: Fraction of top-k chunks containing gold keywords
- **Recall@K**: Fraction of questions with ≥1 keyword match in top-k results
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Accuracy**: Same as Recall@K (hit rate)
- **MRR (Mean Reciprocal Rank)**: 1/average_position(first_relevant), range [0,1]
- **Faithfulness** (RAGAS): LLM evaluation of answer consistency with context
- **Context Precision** (RAGAS): % of retrieved context relevant to question
- **Context Recall** (RAGAS): % of relevant info in corpus retrieved
- **Hallucination Rate**: 1 - Faithfulness