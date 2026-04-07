# MCP News Module - Integration Testing Guide

This directory contains the MCP (Model Context Protocol) tools for automated financial data fetching, enabling RAG systems to augment retrieval results with live market data.

## Overview

The `mcp_news` module provides tools to:
- **Fetch News** - Recent financial news from Finnhub API
- **Fetch Earnings** - Current earnings data, PE ratios, EPS, profit margins
- **Fetch Analyst Ratings** - Analyst recommendations, price targets, consensus ratings
- **Fetch SEC Filings** - Recent 10-K, 10-Q, 8-K filings from SEC Edgar

### Module Structure

```
mcp_news/
├── __init__.py          # Module exports
├── tools.py             # MCP tool schema definitions
├── implementations.py   # API call implementations
├── dispatcher.py        # Tool execution router
└── README.md           # This file
```

---

## Setup

### 1. Install Dependencies

Ensure all requirements are installed:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `requests` - HTTP API calls
- `python-dotenv` - Environment variable loading
- `langchain-community` - RAG integration
- `chromadb` - Vector database for RAG
- `sentence-transformers` - Embeddings

### 2. Set Finnhub API Key

Get your free API key from [finnhub.io](https://finnhub.io).

**.env File**
Create a `.env` file in the project root:
```
FINNHUB_API_KEY=your_finnhub_api_key_here
```

### 3. Verify Setup

```bash
python verify_setup.py
```

---

## Running Integration Tests

There are **2 main test suites** for MCP integration:

### Test Suite 1: MCP + RAG Integration Tests

**Purpose:** Verify MCP tools work correctly and can augment RAG results

**File:** `test_mcp_rag_integration.py`

**Run:**
```bash
cd d:\Codes and Repos\IS469_wealth_manager_project
python test_mcp_rag_integration.py
```

**What It Tests:**

1. **TEST 1: RAG Retrieval Only (10-K documents)**
   - Loads Chroma vector database
   - Retrieves chunks from Apple 10-K filing
   - Tests basic RAG functionality

2. **TEST 2: MCP Live Data Fetching**
   - Fetches recent financial news via `fetch_news`
   - Fetches earnings data via `fetch_earnings`
   - Fetches analyst ratings via `fetch_analyst_ratings`
   - Validates API responses

3. **TEST 3: RAG + MCP Integration**
   - Retrieves 10-K chunks from Chroma
   - Fetches live market data via MCP
   - Combines both contexts into unified prompt

4. **TEST 4: RAG Benchmark Query**
   - Simulates benchmark scenario
   - Shows baseline (RAG only) vs augmented (RAG + MCP)
   - Displays side-by-side comparison

**Expected Output:**
```
======================================================================
  MCP + RAG INTEGRATION TEST SUITE
======================================================================

TEST 1: RAG Retrieval Only (10-K documents)
TEST 2: MCP Live Data Fetching
TEST 3: RAG + MCP Integration (Combined Context)
TEST 4: RAG Benchmark Query (Like rag_compare_chroma.py)

======================================================================
  ✅ ALL INTEGRATION TESTS PASSED
======================================================================
```

**Troubleshooting:**
- If `FINNHUB_API_KEY` error: Set environment variable (see Setup section)
- If Chroma DB not found: Run `rag/rag.py` first to create embeddings
- If import errors: Run `pip install -r requirements.txt`

---

### Test Suite 2: RAG Benchmark with MCP Augmentation

**Purpose:** Benchmark RAG performance with and without MCP data augmentation

**File:** `rag/experiments/rag_compare_chroma.py`

**Run - Minimal:**
```bash
cd d:\Codes and Repos\IS469_wealth_manager_project
python rag/experiments/rag_compare_chroma.py \
    --qa rag/data/manual_qa_template.jsonl
```

**Run - Full Benchmark:**
```bash
python rag/experiments/rag_compare_chroma.py \
    --qa rag/data/manual_qa_template.jsonl \
    --output-dir results/rag_compare_mcp \
    --tickers AAPL NVDA GOOG MSFT \
    --k 5 \
    --days-back 7
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--qa` | (required) | Path to QA JSONL dataset |
| `--output-dir` | `results/rag_compare_mcp` | Where to save results |
| `--db-path` | `./chroma_db` | Path to Chroma vector database |
| `--tickers` | `AAPL` | Stock tickers for MCP fetch (space-separated) |
| `--k` | 5 | Top-k chunks to retrieve |
| `--days-back` | 7 | Days of news history to fetch |
| `--skip-mcp` | (flag) | Skip MCP augmentation (baseline only) |

**What It Does:**

1. **Loads QA Dataset** - Reads manual_qa_template.jsonl with questions and ground truth
2. **Loads Chroma DB** - Initializes vector database with 10-K embeddings
3. **Fetches MCP Data** - Retrieves live news, earnings, analyst ratings
4. **Runs Two Variants:**
   - **Baseline**: Pure RAG (top-k chunks from 10-K)
   - **MCP-Augmented**: Prepends live data + top-(k-1) RAG chunks
5. **Evaluates Metrics:**
   - Precision@K - Relevance of top-k results
   - Recall@K - Coverage of ground truth
   - F1 Score - Harmonic mean
   - MRR (Mean Reciprocal Rank) - Ranking quality

**Output Files:**

```
results/rag_compare_mcp/
├── comparison_summary.csv      # Side-by-side metrics
├── comparison_summary.json     # Same as JSON
└── baseline_details.jsonl      # Per-query breakdown
```

**Sample Output:**
```
variant,precision_at_k,recall_at_k,f1_score,accuracy,mrr
baseline,0.7234,0.6821,0.7021,0.6821,0.8234
mcp,0.7892,0.7456,0.7670,0.7456,0.8567
```

**Expected Improvements:**
- MCP variant should show 3-5% improvement in recall
- Better MRR score (live data ranks higher)
- Higher precision for time-sensitive questions

---

## How MCP Tools Work

### Quick Example

```python
from mcp_news.dispatcher import dispatch_mcp_tool

# Fetch news
news = dispatch_mcp_tool("fetch_news", {
    "tickers": ["AAPL", "NVDA"],
    "days_back": 7
})

# Response structure
{
    "articles": [
        {
            "ticker": "AAPL",
            "title": "Apple Reports Record Q1 Earnings",
            "source": "Bloomberg",
            "sentiment": 0.85,
            "url": "https://...",
            "published": "2024-01-30"
        },
        ...
    ],
    "count": 15,
    "source": "Finnhub",
    "days_back": 7
}
```

### Tool Inputs & Outputs

#### 1. fetch_news
```python
dispatch_mcp_tool("fetch_news", {
    "tickers": ["AAPL"],      # Required: list of stock symbols
    "days_back": 7            # Optional: default 7, max 90
})
```
**Returns:** Recent news articles with sentiment scores

#### 2. fetch_earnings
```python
dispatch_mcp_tool("fetch_earnings", {
    "tickers": ["AAPL", "NVDA"]
})
```
**Returns:** PE ratios, EPS, profit margins, forward guidance

#### 3. fetch_analyst_ratings
```python
dispatch_mcp_tool("fetch_analyst_ratings", {
    "tickers": ["AAPL", "NVDA"]
})
```
**Returns:** Analyst consensus, price targets, recommendation sentiment

#### 4. fetch_sec_filings (no API key yet)
```python
dispatch_mcp_tool("fetch_sec_filings", {
    "tickers": ["AAPL"],
    "filing_type": "10-K"     # Options: "10-K", "10-Q", "8-K"
})
```
**Returns:** Recent SEC Edgar filings with document links

---

## Advanced Usage

### Running Tests with Custom Tickers

```bash
# Test with different companies
python rag/experiments/rag_compare_chroma.py \
    --qa rag/data/manual_qa_template.jsonl \
    --tickers TSLA MSFT AMZN \
    --output-dir results/rag_compare_custom
```

### Baseline Only (Skip MCP)

For performance comparison without MCP overhead:

```bash
python rag/experiments/rag_compare_chroma.py \
    --qa rag/data/manual_qa_template.jsonl \
    --skip-mcp
```

### Custom Top-K

Test different retrieval depths:

```bash
# Retrieve top-3 instead of top-5
python rag/experiments/rag_compare_chroma.py \
    --qa rag/data/manual_qa_template.jsonl \
    --k 3
```

### Extended News History

Analyze impact of historical depth:

```bash
# Look back 30 days instead of 7
python rag/experiments/rag_compare_chroma.py \
    --qa rag/data/manual_qa_template.jsonl \
    --days-back 30
```

---

## Troubleshooting

### Error: "FINNHUB_API_KEY not configured"
**Solution:** Set environment variable
```bash
$env:FINNHUB_API_KEY="your_api_key"
```

### Error: "Chroma DB not found"
**Solution:** Create embeddings first
```bash
python rag/rag.py
```

### Error: "ModuleNotFoundError: mcp_news"
**Solution:** Run from project root
```bash
cd d:\Codes and Repos\IS469_wealth_manager_project
python test_mcp_rag_integration.py
```

### Slow API Responses
- Finnhub free tier has rate limits (~60 requests/minute)
- Consider caching results for repeated queries
- Use `--skip-mcp` flag to test RAG performance independently

### Empty MCP Results
- Free Finnhub API may have limited data for some tickers
- Try major cap stocks: AAPL, MSFT, GOOGL, AMZN, NVDA
- Verify API key is valid at [finnhub.io](https://finnhub.io/dashboard)

---

## Performance Notes

### Typical Runtimes

| Test | Time | Depends On |
|------|------|-----------|
| `test_mcp_rag_integration.py` | 30-60s | API latency, Chroma loading |
| `rag_compare_chroma.py` (baseline) | 20-40s | QA dataset size, DB size |
| `rag_compare_chroma.py` (with MCP) | 40-80s | +API calls for news/earnings/ratings |

## Next Steps

After successfully running tests:

1. **Review Results** - Check `results/rag_compare_mcp/comparison_summary.csv`
2. **Integrate into Agents** - Use MCP in `agents/analyst.py` for live analysis
3. **Build MCP Server** - Expose as real MCP server for Claude integration
4. **Optimize Prompt** - Fine-tune how MCP data is combined with RAG results

---