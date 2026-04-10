# WealthMind AI — IS469 Wealth Manager

Enterprise-grade investment analysis system combining **Retrieval-Augmented Generation (RAG)**, **fine-tuned Llama-3.2 sentiment analysis**, **LLM-powered agents**, and **hallucination auditing** — served through a real-time streaming web UI.

---

## Architecture

```
User Query
    ↓
Orchestrator Agent        — extracts tickers from natural language
    ↓
Market Context Agent      — fetches news, earnings, analyst ratings (MCP tools)
    ↓
Sentiment Agent           — Llama-3.2-3B-Instruct (fine-tuned, HuggingFace)
    ↓
Investment Analyst Agent  — RAG over SEC 10-K filings (ChromaDB + BM25)
    ↓
Auditor Agent             — hallucination detection with RAGAS metrics
    ↓
Report Generator Agent    — final investment recommendation
```

The pipeline runs inside a **LangGraph** state machine and streams progress to the frontend via **Server-Sent Events (SSE)**.

---

## Project Structure

```
.
├── main.py                   # CLI entry point (local + remote modes)
├── pyproject.toml            # uv-managed dependencies
├── langgraph.json            # LangGraph server config
├── .env.example              # Required environment variables
│
├── agents/                   # LangGraph node implementations
│   ├── orchestrator.py       # Ticker extraction
│   ├── market_context.py     # MCP-powered news/earnings/ratings
│   ├── sentiment_agent.py    # Fine-tuned Llama-3.2 sentiment
│   ├── analyst.py            # RAG-based investment analysis
│   ├── auditor.py            # Hallucination detection (RAGAS)
│   ├── report_generator.py   # Final report composition
│   └── portfolio.py          # Portfolio optimization
│
├── graph/                    # LangGraph workflow
│   ├── state.py              # Unified WealthManagerState schema
│   ├── workflow.py           # Graph compilation & conditional edges
│   └── auditor_experiment/   # Iterative audit-loop variants
│
├── rag/                      # Retrieval-Augmented Generation
│   ├── rag.py                # 4 retrieval variants (baseline/HyDE/hybrid/finance)
│   ├── retriever.py          # Vector store helpers
│   ├── simple_rag.py         # Single-query interface
│   ├── data/processed/       # Cleaned 10-K filings (AAPL/AMZN/GOOG/MSFT/NVDA)
│   ├── experiments/          # RAG benchmarking scripts
│   └── README.md             # RAG-specific documentation
│
├── mcp_news/                 # Model Context Protocol — financial data tools
│   ├── tools.py              # Tool schema definitions
│   ├── implementations.py    # Finnhub / SEC Edgar integrations
│   └── dispatcher.py         # Tool call router
│
├── backend/                  # FastAPI web server
│   ├── server.py             # SSE streaming API (POST /api/analyze)
│   └── mcp_http_server.py    # Standalone HTTP MCP service
│
├── frontend/
│   └── index.html            # Single-page web UI (no build step required)
│
├── scripts/                  # Utility & batch scripts
│   ├── audit_rag_outputs.py  # Batch-audit all RAG variants
│   ├── generate_audit_csv.py # Export audit results to CSV
│   ├── generate_final_csv.py # Final comparison table
│   ├── data_ingestion.py     # Load documents into ChromaDB
│   ├── check_reranking.py    # Verify cross-encoder setup
│   └── verify_setup.py       # Dependency health check
│
├── tests/                    # Integration & unit tests
│   ├── test_auditor_enhanced.py
│   ├── test_mcp_server.py
│   ├── test_mcp_rag_integration.py
│   ├── test_http_server.py
│   └── test_env.py
│
├── docs/                     # Extended documentation
│   ├── AUDITOR_IMPLEMENTATION.md
│   ├── RAG_AUDITOR_GUIDE.md
│   ├── MCP_SERVER_SETUP.md
│   └── MCP_HTTP_SETUP.md
│
├── experiments/              # Research notebooks & training runs
│   ├── sentiment_agent/      # Llama / Qwen / FinBERT training notebooks
│   ├── auditor_agent/
│   ├── architecture_workflow/
│   └── market_context_agent/
│
├── results/                  # Experiment output artefacts (.jsonl, .csv)
└── chroma_db/                # Persistent ChromaDB vector store
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 2. Clone & install

```bash
git clone https://github.com/<org>/IS469_wealth_manager_project.git
cd IS469_wealth_manager_project

# Create .venv and install all dependencies from pyproject.toml
uv sync
```

> **No `pip install` or `requirements.txt` needed.** `uv sync` reads `pyproject.toml` and the pinned `uv.lock` file.

### 3. Configure environment

```bash
cp .env.example .env
# Open .env and fill in your API keys
```

Required keys:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude / MCP market context agent |
| `OPENAI_API_KEY` | Orchestrator, auditor, HyDE RAG |
| `FINNHUB_API_KEY` | Live financial news |
| `NEWS_API_KEY` | Supplementary news feed |
| `LLAMA_SENTIMENT_MODEL_PATH` | HuggingFace repo or local path (default: `lunn1212/llama3.2`) |

Optional:

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Only needed for private HuggingFace repos |
| `GROQ_API_KEY` | Alternative LLM provider |
| `COHERE_API_KEY` | Cohere reranker |

### 4. Run the web interface

```bash
# Start backend (from project root)
uv run uvicorn backend.server:app --reload --port 8000

# Open frontend in browser
open frontend/index.html
```

The frontend connects to `http://localhost:8000`. Enter a query like *"Should I invest in AAPL and NVDA?"* and watch the agents run in real time.

### 5. Run CLI mode

```bash
uv run python main.py
# or against a running LangGraph dev server:
uv run python main.py remote
```

---

## Sentiment Model

The sentiment agent uses a **Llama-3.2-3B-Instruct** model fine-tuned on financial news, hosted on HuggingFace at [`lunn1212/llama3.2`](https://huggingface.co/lunn1212/llama3.2).

The model is loaded on first inference (lazy, thread-safe). Set `LLAMA_SENTIMENT_MODEL_PATH` in `.env` to override — accepts any HuggingFace repo ID or absolute local path:

```bash
# HuggingFace (default, public repo — no token needed)
LLAMA_SENTIMENT_MODEL_PATH=lunn1212/llama3.2

# Local checkpoint (teammates with downloaded weights)
LLAMA_SENTIMENT_MODEL_PATH=/absolute/path/to/llama_3.2_3b_instruct_saved

# Private repo
LLAMA_SENTIMENT_MODEL_PATH=org/private-model
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

Device selection is automatic: CUDA > Apple MPS > CPU.

---

## RAG System

4 retrieval variants over SEC 10-K filings (AAPL, AMZN, GOOG, MSFT, NVDA):

| Variant | Method | F1 |
|---|---|---|
| **Hybrid** | Dense + BM25 (RRF) | **0.835** |
| Finance | Finance-tuned embeddings | 0.810 |
| HyDE | Query expansion via GPT | 0.597 |
| Baseline | Dense only | 0.588 |

```bash
# Run all 4 variants (50-question Apple 10-K eval)
uv run python rag/experiments/rag_compare.py --variants baseline,hyde,hybrid,finance

# Single query
uv run python rag/simple_rag.py "What are Apple's main revenue streams?"
```

See [rag/README.md](rag/README.md) for ingestion and evaluation details.

---

## Auditor Agent

GPT-4o-mini powered hallucination detection using RAGAS metrics:

| Metric | Best Variant |
|---|---|
| Faithfulness | Baseline 98.6% |
| Context Recall | Hybrid 52.4% |
| Hallucination Rate | 0% across all variants |

```bash
# Batch audit all 4 RAG variant outputs
uv run python scripts/audit_rag_outputs.py

# Export CSV summary
uv run python scripts/generate_audit_csv.py
```

See [docs/RAG_AUDITOR_GUIDE.md](docs/RAG_AUDITOR_GUIDE.md) for details.

---

## MCP Tools (Market Context)

The market context agent calls financial APIs via **Model Context Protocol**:

| Tool | Source | Data |
|---|---|---|
| `fetch_news` | Finnhub | Company news |
| `fetch_earnings` | Finnhub | Earnings dates & results |
| `fetch_analyst_ratings` | Finnhub | Price targets & buy/sell ratings |
| `fetch_sec_filings` | SEC Edgar | 10-K / 10-Q filings |

To run the MCP tools as a standalone HTTP service:

```bash
uv run uvicorn backend.mcp_http_server:app --port 8001
# POST http://localhost:8001/call {"name": "fetch_news", "arguments": {"tickers": ["AAPL"]}}
```

See [docs/MCP_SERVER_SETUP.md](docs/MCP_SERVER_SETUP.md) and [docs/MCP_HTTP_SETUP.md](docs/MCP_HTTP_SETUP.md).

---

## Tests

```bash
# Run all tests
uv run pytest tests/

# Individual tests
uv run python tests/test_auditor_enhanced.py
uv run python tests/test_mcp_server.py
```

---

## Key Commands Reference

```bash
# Install / sync dependencies
uv sync

# Start web server
uv run uvicorn backend.server:app --reload --port 8000

# CLI pipeline run
uv run python main.py

# RAG evaluation
uv run python rag/experiments/rag_compare.py --variants baseline,hyde,hybrid,finance

# Batch audit
uv run python scripts/audit_rag_outputs.py

# Verify environment
uv run python scripts/verify_setup.py
```

---

## Documentation

| File | Topic |
|---|---|
| [rag/README.md](rag/README.md) | RAG setup, evaluation, and ingestion |
| [docs/AUDITOR_IMPLEMENTATION.md](docs/AUDITOR_IMPLEMENTATION.md) | Auditor architecture & RAGAS metrics |
| [docs/RAG_AUDITOR_GUIDE.md](docs/RAG_AUDITOR_GUIDE.md) | Batch auditing workflow |
| [docs/MCP_SERVER_SETUP.md](docs/MCP_SERVER_SETUP.md) | MCP tool setup & debugging |
| [docs/MCP_HTTP_SETUP.md](docs/MCP_HTTP_SETUP.md) | HTTP MCP transport setup |
| [rag/RAG_Comprehensive_Analysis.md](rag/RAG_Comprehensive_Analysis.md) | Detailed retrieval analysis |

---

## Security

- All API keys stored in `.env` — never committed (`.gitignore` enforced)
- `.env.example` contains only placeholder values
- Hallucination detection provides an audit trail for all generated claims
