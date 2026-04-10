# Enhanced Auditor Agent - Implementation Guide

## Overview
The Auditor Agent has been enhanced with GPT-4o-mini, RAGAS metrics, and comprehensive claim verification to detect hallucinations and validate financial analysis.

## Key Features

### 1. **GPT-4o-mini Powered Auditing**
- Uses OpenAI's GPT-4o-mini model (faster, more cost-effective than GPT-4)
- Temperature set to 0 for deterministic, accurate auditing
- Structured output using Pydantic models for reliability

### 2. **RAGAS Metrics Integration**
Three key metrics from the Retrieval-Augmented Generation Assessment Suite:

#### **Faithfulness Score (0-1)**
- Measures how faithful the generated answer is to the retrieved context
- Checks if numerical claims in the answer appear in the source documents
- Perfect score (1.0) means all claims are grounded in context

#### **Answer Relevancy Score (0-1)**
- Measures if the answer addresses the original question
- Checks keyword overlap between question and answer
- Ignores common stop words for more meaningful analysis

#### **Context Recall Score (0-1)**
- Measures how much relevant information from the context was retrieved
- Checks if key terms from ground truth appear in retrieved chunks
- Ensures sufficient context was provided for proper answer generation

### 3. **Claim Extraction & Verification**
Process:
1. **Extract specific factual claims** from Investment Analyst's output
   - Identifies concrete facts (numbers, dates, percentages, facts)
   - Filters out subjective opinions and analysis
   
2. **Cross-check each claim** against 10-K context
   - Verifies numerical accuracy
   - Checks for contradictions
   - Identifies unsupported claims
   
3. **Classify each claim** into three categories:
   - **VERIFIED**: Exactly matches information in source
   - **HALLUCINATION**: Contradicts source or is fabricated
   - **UNSUBSTANTIATED**: Not mentioned but not contradicted

### 4. **Hallucination Detection**
Flags unsupported or contradictory claims:
- Certainty-based language ("guaranteed", "certain", "risk-free", "always", "never fails")
- Numerical claims inconsistent with context
- Logic gaps and contradictions
- Missing source justification

### 5. **Comprehensive Output**

```python
{
  "status": "APPROVED" | "REJECTED",
  "hallucination_count": int,
  "verified_count": int,
  "unsubstantiated_count": int,
  "ragas_metrics": {
    "faithfulness": 0.0-1.0,
    "answer_relevancy": 0.0-1.0,
    "context_recall": 0.0-1.0
  },
  "audit_findings": [
    {
      "claim": str,
      "status": "VERIFIED" | "HALLUCINATION" | "UNSUBSTANTIATED",
      "source_quote": str,
      "evidence_strength": "STRONG" | "WEAK" | "NONE",
      "correction": str  # If hallucinated
    }
  ],
  "summary_notes": str
}
```

## Implementation Details

### File: `agents/auditor.py`

#### Class: `RAGASCalculator`
Static methods for calculating RAGAS metrics:
```python
# Extract numerical values from text
RAGASCalculator.extract_numbers(text: str) -> List[str]

# Calculate faithfulness score
RAGASCalculator.calculate_faithfulness(answer: str, context: str) -> float

# Calculate answer relevancy
RAGASCalculator.calculate_answer_relevancy(answer: str, question: str) -> float

# Calculate context recall
RAGASCalculator.calculate_context_recall(context: str, ground_truth: str) -> float
```

#### Class: `AuditorAgent`
Main auditing class:
```python
# Initialize with GPT-4o-mini
agent = AuditorAgent(api_key="sk-...")

# Extract factual claims from draft
claims = agent.extract_claims(draft: str) -> List[str]

# Verify single claim against context
fact_check, strength = agent.verify_claim_against_context(
  claim: str,
  context: str
) -> Tuple[FactCheck, float]

# Calculate RAGAS metrics
metrics = agent.calculate_ragas_metrics(
  draft: str,
  context: str,
  ground_truth: str = ""
) -> RAGASMetrics

# Comprehensive audit
report = agent.audit_draft(
  draft: str,
  context: str,
  ground_truth: str = ""
) -> AuditReport
```

#### Function: `auditor_node(state: WealthManagerState) -> dict`
Integrates with LangGraph workflow:
- Reads Investment Analyst output from state
- Reads retrieved 10-K context
- Performs comprehensive audit
- Updates state with findings and metrics
- Returns structured audit results

## LangGraph Integration

### Iterative Graph with Self-Correction
File: `graph/auditor_experiment/iterative_graph.py`

```
Investment Analyst → Auditor → Reporter → END
     ↓                  ↓
     └──────────────────┘ (if REJECTED & loop_count < 2)
```

Flow:
1. Investment Analyst generates draft
2. Auditor verifies claims and calculates RAGAS metrics
3. If approved (hallucination_rate < 20% AND faithfulness > 70%):
   - Continue to Reporter
4. If rejected AND loop_count < 2:
   - Return to Investment Analyst for revision
5. Otherwise:
   - Continue to Reporter regardless

## Usage Examples

### Basic Audit
```python
from agents.auditor import AuditorAgent

agent = AuditorAgent()
report = agent.audit_draft(
    draft="Apple reported $394 billion revenue...",
    context="Apple reported net sales of $394.3 billion in fiscal 2024..."
)
print(f"Status: {report.status}")
print(f"Hallucinations: {report.hallucination_count}")
print(f"RAGAS Faithfulness: {report.ragas_metrics.faithfulness:.1%}")
```

### In LangGraph
```python
from graph.auditor_experiment.iterative_graph import build_iterative

graph = build_iterative()
result = graph.invoke({
    "ticker": "AAPL",
    "context": "10-K text...",
    "draft": "Analysis...",
    "loop_count": 0
})
```

### Test with Demo Data
```bash
# Run the test script
python test_auditor_enhanced.py
```

## RAGAS Metrics Interpretation

### Faithfulness Score
- **0.9-1.0**: Excellent - Answer is well-grounded in context
- **0.7-0.9**: Good - Minor unsupported claims
- **0.5-0.7**: Fair - Multiple unsupported claims
- **0.0-0.5**: Poor - Answer barely grounded in context

### Answer Relevancy Score  
- **0.9-1.0**: Perfectly addresses the question
- **0.7-0.9**: Mostly relevant with minor deviations
- **0.5-0.7**: Partially relevant
- **0.0-0.5**: Mostly irrelevant

### Context Recall Score
- **0.9-1.0**: Retrieved almost all relevant information
- **0.7-0.9**: Retrieved most relevant information
- **0.5-0.7**: Retrieved about half the relevant information
- **0.0-0.5**: Retrieved very little relevant information

## API Requirements

- **OpenAI API Key**: Required for GPT-4o-mini auditing
  ```bash
  export OPENAI_API_KEY="sk-..."
  ```

## Fallback Mode

If OPENAI_API_KEY is not set, the system uses a fallback audit that:
- Detects certainty-based language patterns
- Checks context overlap
- Calculates basic RAGAS metrics
- Returns structured output for compatibility

## Performance Notes

- **Speed**: ~5-10 seconds per audit (including API calls)
- **Cost**: ~$0.05-0.10 per audit using gpt-4o-mini
- **Accuracy**: Improves with longer, more detailed context

## Future Enhancements

1. **Multi-language Support**: Extend to non-English financial documents
2. **Domain-Specific Metrics**: Add finance-specific hallucination patterns
3. **Batch Auditing**: Process multiple claims in parallel
4. **Explanation Generation**: Provide more detailed feedback for failures
5. **Fine-tuning**: Custom model training on financial audit data

## RAG Output Integration

### Script: `audit_rag_outputs.py`
Comprehensive auditing of RAG system outputs with RAGAS metrics.

**Usage:**
```bash
# Setup environment (one time)
python3 -m venv .venv
conda deactivate
source .venv/bin/activate
pip install -q pydantic langchain-openai python-dotenv numpy==1.26.4

# Run the auditor
set -a && source .env && set +a
.venv/bin/python audit_rag_outputs.py
```

**Features:**
- Reads `.jsonl` files from `results/rag_compare/{variant}/`
- Audits all 4 variants: hybrid, finance, hyde, baseline
- Generates comparative rankings
- Detects hallucinations across 50 questions per variant
- Exports full audit report to `results/auditor_compare.py/rag_audit_report.json`

**Output:**
```json
{
  "variants_audited": [
    {
      "variant": "hybrid",
      "hallucination_rate": 0.0,
      "ragas_averages": {
        "faithfulness": 0.98,
        "answer_relevancy": 0.03,
        "context_recall": 0.823
      },
      "audit_details": [...]
    }
  ],
  "comparative_summary": {...}
}
```

### Updated State Fields

LangGraph `WealthManagerState` now includes:
- `ragas_metrics`: Dict with faithfulness, answer_relevancy, context_recall
- `hallucination_count`: Number of hallucinations detected
- `verified_count`: Number of verified claims
- `unsubstantiated_count`: Number of unsubstantiated claims
- `ground_truth`: Expected answer for comparison
