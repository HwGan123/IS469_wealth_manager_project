# IS469 Wealth Manager Project

Enterprise-grade investment analysis system combining **Retrieval-Augmented Generation (RAG)**, **LLM-powered agents**, and **AI auditing** for hallucination detection and compliance.

## 🎯 Project Overview

This system analyzes financial documents (10-K filings) using a pipeline of specialized agents:

```
📄 Financial Document (10-K)
    ↓
🔍 RAG System (4 Retrieval Variants)
    ├─ Dense Retrieval (baseline)
    ├─ HyDE (Hypothetical Document Expansion)
    ├─ Hybrid (Dense + BM25)
    └─ Finance-Tuned Embeddings
    ↓
🤖 Investment Analyst Agent
    └─ Generates investment analysis
    ↓
✅ Auditor Agent (Enhanced)
    ├─ Extracts & verifies claims
    ├─ Calculates RAGAS metrics
    └─ Flags hallucinations
    ↓
📊 Report Generator Agent
    └─ Final investment recommendation
```

## 📦 Core Components

### 1. **RAG System** (`rag/` folder)
Multi-variant retrieval system for financial document analysis:

- **Baseline**: Dense embeddings (`all-MiniLM-L6-v2`)
- **HyDE**: Query expansion via OpenAI
- **Hybrid**: Reciprocal Rank Fusion (Dense + BM25)
- **Finance**: Finance-tuned embeddings (`all-mpnet-base-v2`)

**Key Features:**
- ChromaDB vector database (persistent storage)
- 50-question evaluation dataset on Apple 10-K
- Cross-encoder reranking (optional)
- Comprehensive metrics: Precision, Recall, F1, MRR, RAGAS

**Best Performers:**
- 🏆 **Hybrid (no rerank)**: F1 = 0.8348
- 🥈 **Finance (with rerank)**: F1 = 0.8103

👉 See: [rag/README.md](rag/README.md) for RAG usage

### 2. **Auditor Agent** (`agents/auditor.py`)
GPT-4o-mini powered hallucination detection with RAGAS metrics:

**Capabilities:**
- ✅ Claim extraction from generated text
- ✅ Cross-reference against source documents
- ✅ RAGAS metrics: faithfulness, answer_relevancy, context_recall
- ✅ Evidence strength classification: STRONG/WEAK/NONE
- ✅ Hallucination count & verification report

**Output:**
```json
{
  "status": "APPROVED/REJECTED",
  "hallucination_count": 2,
  "verified_count": 8,
  "unsubstantiated_count": 1,
  "ragas_metrics": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.78,
    "context_recall": 0.92
  }
}
```

👉 See: [AUDITOR_IMPLEMENTATION.md](AUDITOR_IMPLEMENTATION.md) for details

### 3. **RAG Output Auditor** (`audit_rag_outputs.py`)
Batch audits all RAG variant outputs (.jsonl files):

```bash
export OPENAI_API_KEY="sk-..."
python audit_rag_outputs.py
```

**Output:** `results/rag_audit_report.json`
- Per-variant hallucination rates
- RAGAS averages
- Comparative rankings
- Per-question audit details

👉 See: [RAG_AUDITOR_GUIDE.md](RAG_AUDITOR_GUIDE.md) for usage

### 4. **LangGraph Workflow** (`graph/`)
Orchestrates multi-agent pipeline with state management:

- `graph/state.py`: Unified state schema
- `graph/workflow.py`: Main computation graph
- `graph/auditor_experiment/`: Iterative graphs with self-correction loop

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.9+ (tested on 3.9, 3.14)
python3 --version

# Create .env file in project root with API keys
# (File: .env)
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...  # Optional
GROQ_API_KEY=...    # Optional
```

**Note:** If using Anaconda/conda, deactivate it first to avoid dependency conflicts:
```bash
conda deactivate
```

### 1. Run RAG Evaluation
```bash
# Setup environment
python3 -m venv .venv_rag
source .venv_rag/bin/activate
pip install -q sentence-transformers==2.7.0 rank-bm25==0.2.2

# Run all 4 variants (baseline, hyde, hybrid, finance)
python rag/experiments/rag_compare.py \
  --variants baseline,hyde,hybrid,finance
```

### 2. Audit RAG Outputs

**Setup virtual environment:**
```bash
# Create and activate virtual environment
python3 -m venv .venv
conda deactivate  # If using conda/anaconda
source .venv/bin/activate

# Install all required dependencies
pip install -q pydantic langchain-openai python-dotenv
pip install -q numpy==1.26.4 sentence-transformers==2.7.0
pip install -q rank-bm25 chromadb tiktoken ragas
```

**Run the audit:**
```bash
# Load API keys from .env and run auditor
conda deactivate  # Deactivate conda if active
source .venv/bin/activate
set -a && source .env && set +a
.venv/bin/python audit_rag_outputs.py
```

**Results:** `results/auditor_compare.py/rag_audit_report.json` with per-variant metrics

**Audit Results Summary:**
```
⭐ Best Overall: HYBRID (51.1% RAGAS score)
✓ Lowest Hallucination: All variants 0.0%
📊 Faithfulness: Baseline 98.6% → Hybrid 98.0% → Finance 98.3% → HyDE 87.0%
```

**Required Dependencies:**
- `pydantic` - Structured data validation
- `langchain-openai` - GPT-4o-mini integration for auditing
- `python-dotenv` - Load API keys from .env
- `numpy==1.26.4` - NumPy version compatibility (NOT 2.x)
- `sentence-transformers==2.7.0` - Embeddings
- `rank-bm25` - BM25 retrieval
- `chromadb` - Vector database
- `tiktoken` - Token counting
- `ragas` - RAGAS metrics evaluation

### 3. Use in LangGraph Workflow
```python
from graph.auditor_experiment.iterative_graph import build_iterative

# Build workflow with audit loop
graph = build_iterative()

# Invoke with state
result = graph.invoke({
    "ticker": "AAPL",
    "context": "...",  # 10-K text
    "draft": "...",    # Investment analysis
    "loop_count": 0
})

print(result["audit_results"])
# {
#   "status": "APPROVED",
#   "hallucination_count": 0,
#   "ragas_metrics": {...}
# }
```

## � Complete Dependency Reference

### Core Framework Dependencies
```bash
# LangChain & LLM Integration
pip install langchain==0.2.0
pip install langchain-openai==0.1.7
pip install langgraph==0.0.55

# Environment & Utilities
pip install python-dotenv==1.0.1
pip install anthropic==0.86.0
```

### RAG & Vector Database
```bash
# Dense & Sparse Retrieval
pip install sentence-transformers==2.7.0
pip install rank-bm25==0.2.2

# Vector Storage
pip install chromadb==1.4.0
pip install pypdf==4.2.0
pip install tiktoken==0.7.0
```

### Auditor & Quality Assessment
```bash
# Auditor Dependencies
pip install pydantic>=2.0.0      # For structured outputs
pip install ragas==0.1.9          # RAGAS metrics
pip install datasets==2.19.1      # For evaluation

# Critical: Compatibility Pins
pip install numpy==1.26.4         # NumPy 1.x (NOT 2.x - causes torch/transformers conflicts)
pip install transformers==4.41.0  # Requires numpy<2
pip install torch==2.2.2          # PyTorch
```

### Optional: Financial Analysis & Sentiment
```bash
# Financial Data & Portfolio
pip install yfinance==1.2.0
pip install PyPortfolioOpt==1.5.5
pip install pandas==2.2.2

# Sentiment Analysis
pip install sentencepiece==0.2.0

# Dependency Fix
pip install protobuf==4.25.3
```

### Quick Install (All At Once)
```bash
pip install -r requirements.txt
```

**⚠️ IMPORTANT NOTES:**
- **NumPy Version**: Must use `numpy==1.26.4` (NOT 2.x) due to PyTorch/transformers compatibility
- **Conda Conflicts**: Deactivate conda/anaconda before activating `.venv` to avoid mixing package systems
- **Python Version**: Tested on Python 3.9+ (3.14.3 confirmed working)
- **Virtual Environment**: Always use `.venv` over system Python to isolate dependencies

## �📊 Architecture

### State Flow
```
WealthManagerState
├── messages: List                    # Conversation history
├── tickers: List[str]               # Stock symbols
├── sentiment_score: float           # From Sentiment Agent
├── retrieved_context: str           # From RAG System ✨
├── draft_report: str               # From Investment Analyst
├── audit_score: float              # From Auditor ✨
├── hallucination_count: int        # From Auditor ✨
├── ragas_metrics: Dict             # RAGAS scores ✨
└── final_report: str               # From Report Generator
```

### Agents
| Agent | Input | Output | Technology |
|-------|-------|--------|------------|
| **Investment Analyst** | `retrieved_context` | `draft_report` | RAG + LLM |
| **Auditor** | `draft_report`, `retrieved_context` | `audit_findings`, `ragas_metrics` | GPT-4o-mini |
| **Report Generator** | `draft_report`, `audit_findings` | `final_report` | LLM |
| **Sentiment** | News articles | `sentiment_score` | FinBERT |
| **Portfolio** | Financial data | `portfolio_weights` | PyPortfolioOpt |

## 📈 Results Summary

### RAG Performance (50-question Apple 10-K eval)

| Variant | Precision | Recall | F1 | RAGAS Faithfulness |
|---------|-----------|--------|----|--------------------|
| Hybrid | 0.764 | 0.920 | **0.8348** ⭐ | 0.847 |
| Finance | 0.724 | 0.920 | 0.8103 | 0.816 |
| Baseline | 0.452 | 0.840 | 0.5877 | 0.768 |
| HyDE | 0.460 | 0.840 | 0.5970 | 0.714 |

**Recommendation:** Use **Hybrid variant** (no reranking needed)
- Best F1 score
- No API overhead
- Highest consistency
- Ready for production

### Auditor Performance (From Latest Audit)
- ✅ **Zero Hallucinations**: All 4 variants scored 0.0% hallucination rate
- 🏆 **Best Overall**: Hybrid variant (51.1% RAGAS score)
- 📈 **Faithfulness**: Baseline 98.6% → Finance 98.3% → Hybrid 98.0% → HyDE 87.0%
- 📊 **Context Recall**: Hybrid 52.4% (best) → Finance 49.1% → Baseline 40.9% → HyDE 40.5%
- ⚡ **Speed**: ~5-10 minutes per full 4-variant audit (200 total questions)
- 💰 **Cost**: ~$1-2 per full audit using gpt-4o-mini

## 📚 Documentation

| File | Purpose |
|------|---------|
| [rag/README.md](rag/README.md) | RAG system setup & usage |
| [AUDITOR_IMPLEMENTATION.md](AUDITOR_IMPLEMENTATION.md) | Auditor agent architecture |
| [RAG_AUDITOR_GUIDE.md](RAG_AUDITOR_GUIDE.md) | Batch auditing & integration |
| [RAG_Comprehensive_Analysis.md](rag/RAG_Comprehensive_Analysis.md) | RAG evaluation methodology |

## 🔧 Key Commands

### Setup (One Time)
```bash
# Create virtual environment
python3 -m venv .venv

# Deactivate conda if active
conda deactivate

# Activate venv
source .venv/bin/activate

# Install all dependencies
pip install -q pydantic langchain-openai python-dotenv numpy==1.26.4 sentence-transformers==2.7.0

# Install RAG & auditor dependencies  
pip install -q rank-bm25 chromadb tiktoken ragas
```

### RAG System
```bash
# Activate venv first
conda deactivate && source .venv/bin/activate

# Run all variants (baseline, hyde, hybrid, finance)
python rag/experiments/rag_compare.py --variants baseline,hyde,hybrid,finance

# Run with reranking
python rag/experiments/rag_compare.py --reranker cross-encoder

# Single query testing
python rag/simple_rag.py "What are Apple's revenue streams?"
```

### Auditor
```bash
# Setup: Activate venv and load environment
conda deactivate
source .venv/bin/activate
set -a && source .env && set +a

# Run full audit (all 4 variants, 200 questions total)
.venv/bin/python audit_rag_outputs.py

# Generate CSV summary table
python generate_audit_csv.py

# View audit report
cat results/auditor_compare.py/rag_audit_report.json | python -m json.tool

# View CSV table
cat results/auditor_compare.py/audit_summary.csv
```

### LangGraph
```bash
# Activate venv first
conda deactivate && source .venv/bin/activate

# Run workflow with audit loop
python graph/auditor_experiment/iterative_graph.py
```

## 📁 Project Structure

```
.
├── rag/                         # RAG System
│   ├── experiments/
│   │   ├── rag_compare.py      # Baseline variants
│   │   └── rag_compare_rerank.py # With reranking
│   ├── data/
│   │   ├── processed/          # Chunked data, embeddings
│   │   └── manual_qa_template.jsonl
│   ├── rag.py, simple_rag.py, retriever.py
│   └── README.md               # RAG documentation
│
├── agents/                      # LLM Agents
│   ├── auditor.py              # ✨ Enhanced auditor with RAGAS
│   ├── investment.py           # Investment analyst
│   ├── reporter.py             # Report generator
│   ├── sentiment.py            # Sentiment analysis
│   └── portfolio.py            # Portfolio optimization
│
├── graph/                       # LangGraph Workflows
│   ├── state.py                # ✨ Updated state schema
│   ├── workflow.py
│   └── auditor_experiment/
│       ├── baseline_graph.py
│       ├── iterative_graph.py   # With self-correction loop
│       └── oneshot_graph.py
│
├── results/                     # RAG Results & Audits
│   ├── rag_compare/            # Baseline outputs
│   └── auditor_compare.py/     # ✨ Audit results folder
│       ├── rag_audit_report.json
│       └── audit_summary.csv
│
├── audit_rag_outputs.py        # ✨ Batch auditor script
├── generate_audit_csv.py       # ✨ CSV report generator
├── AUDITOR_IMPLEMENTATION.md   # ✨ Auditor docs
├── RAG_AUDITOR_GUIDE.md        # ✨ Integration guide
└── requirements.txt            # Dependencies
```

## ⚙️ Configuration

### Environment Variables
```bash
# Required
export OPENAI_API_KEY="sk-..."           # For auditor & HyDE

# Optional
export GROQ_API_KEY="sk-..."             # For Groq models
export NEWSAPI_KEY="..."                 # For sentiment analysis
```

### Key Settings (in code)
- **RAG K**: Top-k documents to retrieve (default: 5)
- **Auditor Temperature**: 0 (deterministic)
- **LLM Model**: gpt-4o-mini (cost-optimized)
- **Reranker**: cross-encoder (local, no API)

## 🧪 Testing

```bash
# Test RAG system
python rag/verify_setup.py

# Test auditor
python test_auditor_enhanced.py

# Test MCP integration
python test_mcp_rag_integration.py

# Generate final results CSV
python generate_final_csv.py
```

## 📊 Performance Benchmarks

| Operation | Time | Cost |
|-----------|------|------|
| RAG evaluation (50 Q) | 2-3 min | ~$0.10 |
| Auditor batch (50 Q, single variant) | 5-10 min | ~$0.50 |
| Reranking (50 Q) | +2-3 sec | Free (local) |
| Full workflow (end-to-end) | ~15 min | ~$1.00 |

## 🔐 Security & Compliance

- ✅ API keys via environment variables (never in code)
- ✅ Hallucination detection for compliance
- ✅ Audit trail (detailed claim verification logs)
- ✅ Source attribution (all claims cited)
- ✅ RAGAS metrics for quality assurance

## 🚧 Future Roadmap

- [ ] Real-time validation in LangGraph
- [ ] Auto-retry on failed audits
- [ ] Multi-document RAG (multiple 10-Ks)
- [ ] Fine-tuned models for financial domain
- [ ] Visualization dashboard for audit results
- [ ] Compliance report generation

## 📞 Support

For issues or questions:
1. Check relevant README in `rag/` folder
2. See `AUDITOR_IMPLEMENTATION.md` for auditor questions
3. Review `RAG_AUDITOR_GUIDE.md` for integration issues

## 📄 License

[Add license information]

---

**Last Updated:** April 7, 2026  
**Status:** ✅ Production Ready with Enhanced Auditing