# RAG Auditor Integration Guide

## Overview
The auditor now integrates with RAG outputs to provide comprehensive claim verification and RAGAS metrics for all generated answers.

## Quick Start

### 1. Audit RAG Outputs
```bash
# Deactivate conda (if using anaconda)
conda deactivate

# Activate virtual environment
source .venv/bin/activate

# Load API keys from .env file
set -a && source .env && set +a

# Run audit on baseline RAG variants (all 4: baseline, hyde, hybrid, finance)
.venv/bin/python audit_rag_outputs.py
```

### 2. Output
Generates `results/auditor_compare.py/rag_audit_report.json` with:
- Per-variant audit summaries
- Per-question claim verification details
- RAGAS metrics (faithfulness, answer_relevancy, context_recall)
- Comparative rankings across variants
- Hallucination detection and counts

## What Gets Audited

The script reads `.jsonl` files from RAG outputs and audits:
- **Hybrid** (best performing variant)
- **Finance** (best with reranking)
- **HyDE** (hypothetical document expansion)
- **Baseline** (dense retrieval only)

Source: `results/rag_compare/{variant}/{variant}_details.jsonl`

## Audit Report Structure

```json
{
  "audit_source": "results/rag_compare",
  "variants_audited": [
    {
      "variant": "hybrid",
      "total_entries": 50,
      "overall_status": "PASSED",
      "hallucination_rate": 0.08,
      "total_hallucinations": 4,
      "total_verified": 32,
      "total_unsubstantiated": 14,
      "ragas_averages": {
        "faithfulness": 0.847,
        "answer_relevancy": 0.756,
        "context_recall": 0.823
      },
      "audit_details": [
        {
          "question_id": 1,
          "question": "What are the major risk factors...",
          "status": "APPROVED",
          "hallucination_count": 0,
          "verified_count": 5,
          "unsubstantiated_count": 0,
          "ragas_metrics": {
            "faithfulness": 0.85,
            "answer_relevancy": 0.72,
            "context_recall": 0.88
          },
          "findings_summary": "0 hallucinations, 5 verified, 0 unsubstantiated"
        }
      ]
    }
  ],
  "comparative_summary": {
    "best_variant_by_faithfulness": {
      "variant": "hybrid",
      "score": 0.847
    },
    "worst_hallucination_rate": {
      "variant": "hybrid",
      "rate": 0.08,
      "count": 4
    },
    "variant_rankings": [
      {
        "variant": "hybrid",
        "average_ragas_score": 0.809,
        "hallucination_rate": 0.08,
        "status": "PASSED"
      },
      {
        "variant": "finance",
        "average_ragas_score": 0.798,
        "hallucination_rate": 0.10,
        "status": "PASSED"
      }
    ]
  }
}
```

## Interpreting Results

### RAGAS Metrics

| Metric | Range | Interpretation |
|--------|-------|-----------------|
| **Faithfulness** | 0-1 | How faithful is the answer to context? <br/> 0.9-1.0 = Excellent, 0.7-0.9 = Good |
| **Answer Relevancy** | 0-1 | How well does answer address question? <br/> 0.9-1.0 = Perfect, 0.5-0.7 = Partial |
| **Context Recall** | 0-1 | How much relevant info was retrieved? <br/> 0.9-1.0 = Nearly all, 0.5-0.7 = About half |

### Claim Status

- **VERIFIED**: Claim exactly matches information in retrieved context
- **HALLUCINATION**: Claim contradicts context or is fabricated
- **UNSUBSTANTIATED**: Claim not mentioned but not contradicted

### Overall Status

- **PASSED**: < 20% hallucination rate AND faithfulness > 70%
- **FAILED**: Otherwise

## Integration with LangGraph

### Reading RAG Outputs into State

```python
from graph.state import WealthManagerState
import json

# Load RAG output entry
entry = json.loads(rag_output_line)

state = WealthManagerState()
state["draft_report"] = entry["answer"]
state["retrieved_context"] = "\n\n---\n\n".join(entry["contexts"])
state["ground_truth"] = entry.get("ground_truth", "")

# Auditor reads these fields
from agents.auditor import auditor_node
audit_result = auditor_node(state)
```

### Updated State Fields

New fields available in `WealthManagerState`:
```python
ragas_metrics: Dict            # {"faithfulness": 0.85, ...}
hallucination_count: int       # Number of hallucinations found
verified_count: int            # Number of verified claims
unsubstantiated_count: int     # Number of unsubstantiated claims
ground_truth: str              # Expected answer for comparison
```

## Advanced Usage

### Audit Only Specific Variants

```python
from audit_rag_outputs import RAGAuditor

auditor = RAGAuditor()

# Audit hybrid variant only
result = auditor.audit_variant("hybrid", 
    Path("results/rag_compare/hybrid/hybrid_details.jsonl"))
```

### Audit with Reranking Results

```python
# Audit reranked variants
result = auditor.audit_variant("hybrid+rerank",
    Path("results/rag_compare_rerank_cross_encoder/hybrid+rerank_details.jsonl"))
```

### Access Per-Question Audits

```python
audit_report = json.load(open("results/rag_audit_report.json"))

for variant in audit_report["variants_audited"]:
    print(f"\n{variant['variant'].upper()}:")
    for audit in variant["audit_details"]:
        print(f"  Q{audit['question_id']}: {audit['status']}")
        print(f"    Hallucinations: {audit['hallucination_count']}")
        print(f"    Faithfulness: {audit['ragas_metrics']['faithfulness']:.1%}")
```

## Performance & Cost

- **Speed**: ~5-10 seconds per variant (50 questions)
- **Cost**: ~$0.50-1.00 per variant audit using gpt-4o-mini
- **Accuracy**: Highest with detailed context (4-5 chunks per question)

## Key Insights from RAG Audit

### Best Performers (from baseline rag_compare)

1. **Hybrid** (Recommended)
   - Combines dense + BM25 retrieval
   - Highest faithfulness
   - No API overhead
   - Best F1 score: 0.8348

2. **Finance** (With Reranking)
   - Finance-tuned embeddings
   - Benefits from cross-encoder reranking
   - Better ranking order (higher MRR)
   - F1 with rerank: 0.8103

### Common Hallucinations Detected

- Certainty language ("always", "guaranteed", "risk-free")
- Numerical inaccuracies (wrong revenue/debt figures)
- Logic gaps (e.g., high growth claims despite risk factors)
- Contradictions in sequential claims

## Generating Fresh Audits

```bash
# First, run RAG comparisons
python JJ/experiments/rag_compare.py --variants baseline,hyde,hybrid,finance

# Then audit the outputs
python audit_rag_outputs.py

# Results saved to: results/rag_audit_report.json
```

## Troubleshooting

### Issue: "OPENAI_API_KEY not set"
```bash
export OPENAI_API_KEY="sk-..."
```

### Issue: "No .jsonl files found"
Make sure RAG variants have been run:
```bash
python JJ/experiments/rag_compare.py --variants hybrid,finance
```

### Issue: Audit taking too long
- Check API rate limits
- Reduce number of entries audited
- Run on specific variants only

## Next Steps

1. Integrate auditor into LangGraph workflow for real-time validation
2. Use audit results to automatically retry failed analyses
3. Export audit report for compliance documentation
4. Track hallucination trends across model versions
