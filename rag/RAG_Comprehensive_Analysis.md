# Comprehensive RAG Experiment Analysis

## Overview
This document provides a detailed analysis of 8 RAG (Retrieval-Augmented Generation) experiments conducted on Apple 10-K financial data. The experiments compare four RAG variants (baseline, hyde, hybrid, finance) with and without reranking using Groq's Llama 3 70B model.

## Experiment Setup
- **Dataset**: Apple 10-K financial document chunks
- **QA Pairs**: 50 manually curated questions
- **K**: Top 5 retrieved documents
- **Embedding Models**: 
  - Baseline/HyDE/Hybrid: all-MiniLM-L6-v2
  - Finance: all-mpnet-base-v2
- **LLM**: GPT-5.4-nano for answer generation
- **Reranking**: Groq Llama-3-70B-8192 for LLM-based reranking

## Non-Rerank Results

| Variant   | Precision@5 | Recall@5 | F1-Score | Accuracy | MRR      |
|-----------|-------------|----------|----------|----------|----------|
| Baseline  | 0.452      | 0.84    | 0.588   | 0.84    | 0.622   |
| HyDE      | 0.507      | 0.82    | 0.626   | 0.82    | 0.613   |
| Hybrid    | 0.764      | 0.92    | 0.835   | 0.92    | 0.818   |
| Finance   | 0.724      | 0.92    | 0.810   | 0.92    | 0.827   |

### Key Insights (Non-Rerank)
- **Hybrid** performs best across all metrics, with 76.4% precision, 92% recall, and 83.5% F1-score.
- **Finance** variant shows strong performance (72.4% precision, 92% recall), slightly below hybrid.
- **HyDE** improves precision over baseline (50.7% vs 45.2%) but has lower recall (82% vs 84%).
- **Baseline** has the lowest precision but decent recall and MRR.

## Rerank Results (Groq)

| Variant          | Precision@5 | Recall@5 | F1-Score | Accuracy | MRR      |
|------------------|-------------|----------|----------|----------|----------|
| Baseline+Rerank | 0.452      | 0.84    | 0.588   | 0.84    | 0.622   |
| HyDE+Rerank     | 0.452      | 0.84    | 0.588   | 0.84    | 0.622   |
| Hybrid+Rerank   | 0.764      | 0.92    | 0.835   | 0.92    | 0.818   |
| Finance+Rerank  | 0.724      | 0.92    | 0.810   | 0.92    | 0.827   |

### Key Insights (Rerank)
- **No performance improvement** observed with Groq reranking.
- Results are identical to non-rerank experiments.
- This suggests the reranking process may not have been applied or failed silently.

## Comparison: Non-Rerank vs Rerank

| Variant   | Non-Rerank F1 | Rerank F1 | Change |
|-----------|----------------|-----------|--------|
| Baseline  | 0.588         | 0.588    | 0.000  |
| HyDE      | 0.626         | 0.588    | -0.038 |
| Hybrid    | 0.835         | 0.835    | 0.000  |
| Finance   | 0.810         | 0.810    | 0.000  |

### Analysis
- **No significant changes** between non-rerank and rerank results.
- **HyDE+Rerank** shows a slight degradation (F1 from 0.626 to 0.588), but this may be due to identical metrics suggesting no reranking occurred.
- **Hybrid and Finance** variants maintain their strong performance.

## Overall Rankings

### By F1-Score (Non-Rerank)
1. Hybrid (0.835)
2. Finance (0.810)
3. HyDE (0.626)
4. Baseline (0.588)

### By F1-Score (Rerank)
1. Hybrid+Rerank (0.835)
2. Finance+Rerank (0.810)
3. Baseline+Rerank (0.588)
4. HyDE+Rerank (0.588)

## Conclusions
1. **Hybrid retrieval** consistently outperforms other variants, combining dense and sparse retrieval effectively.
2. **Finance-tuned embeddings** provide strong performance, second only to hybrid.
3. **HyDE** offers moderate improvement in precision but may not be worth the complexity for this dataset.
4. **Groq reranking** did not provide benefits in this experiment, possibly due to implementation issues or the nature of the financial QA task.

## Recommendations
- **Primary Recommendation**: Use Hybrid retrieval for production RAG systems on financial documents.
- **Further Investigation**: Debug the reranking implementation or try different reranking approaches (e.g., Cohere, custom models).
- **Optimization**: Focus on improving embedding quality and retrieval strategies rather than reranking for this use case.
- **Next Steps**: Test on larger datasets or different document types to validate findings.

## Technical Notes
- All experiments used k=5 for retrieval evaluation.
- Metrics calculated on 50 QA pairs from Apple 10-K data.
- Reranking was implemented using Groq's Llama 3 70B model for document relevance scoring.
- No faithfulness, context precision, or hallucination metrics were collected in these experiments.