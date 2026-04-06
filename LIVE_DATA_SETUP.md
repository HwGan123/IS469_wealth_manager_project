## MCP Live Data Integration Setup

Your analyst agent now fetches **live financial data** (news + earnings) and combines it with static 10-K filings for enhanced analysis.

### Architecture

```
sentiment_agent (NewsAPI)
       ↓
analyst_agent
   ├─ 10-K RAG (chromadb)
   └─ LIVE DATA (MCP web scraper)
       ├─ News (NewsAPI)
       └─ Earnings (yfinance)
       ↓
portfolio_agent
       ↓
auditor_agent
```

### Setup

#### 1. **Environment Variables** (`.env`)

```bash
# OpenAI API
OPENAI_API_KEY=sk-...

# NewsAPI (https://newsapi.org)
NEWS_API_KEY=your_newsapi_key

# Optional: for detailed sentiment analysis
SENTIMENT_MODEL=finbert
```

#### 2. **Install Dependencies**

The setup already includes required packages in `requirements.txt`:
- `requests` — for API calls
- `yfinance` — for earnings data
- `langchain` — for orchestration
- `chromadb` — for embeddings

#### 3. **Initialize Live Data Vectorstore**

The first time you run the workflow, live data will be embedded and stored in ChromaDB:

```bash
python main.py
```

This will:
- Fetch news for tickers (sentiment_agent)
- Fetch earnings data (analyst_agent)
- Embed both using HuggingFace embeddings
- Store in separate ChromaDB collection (`live_financial_data`)

### How It Works

#### Data Flow in Analyst Agent

1. **Static Context** (10-K)
   ```python
   docs = retriever.invoke(search_query)  # Retrieves 10-K chunks
   context = "\n\n".join([doc.page_content for doc in docs])
   ```

2. **Live Data** (Real-time)
   ```python
   live_data_retriever.refresh(tickers)  # Fetches news + earnings
   live_docs = live_data_retriever.retrieve(live_query, k=5)
   live_context = "\n\n".join([doc["text"] for doc in live_docs])
   ```

3. **Combined Analysis**
   - Prompt receives both `context` (10-K) and `live_context` (live data)
   - Claude synthesizes insights across historical + current data
   - Flags contradictions between 10-K guidance and recent developments

### Configuration Options

#### In `JJ/live_data_retriever.py`

```python
# Adjust how much historical data to fetch
days_back=7  # Change to 14, 30, etc.

# Adjust retrieval results
k=5  # Number of live data documents to retrieve
```

#### In `agents/analyst.py`

```python
# Adjust sensitivity to live data vs 10-K
temperature=0.2  # Lower = more conservative, Higher = more creative

# Add custom data sources
# Example: fetch from SEC Edgar, Yahoo Finance, etc.
```

### What Data Is Fetched

**From NewsAPI:**
- Company news (last 7 days by default)
- Consolidated by ticker
- Metadata: title, description, source, publish date

**From yfinance:**
- Current earnings data
- P/E ratio, EPS, forward guidance
- Profit margins, revenue per share

### Example Output

The analyst agent now produces reports that reference:

```
## Executive Summary
NVIDIA shows strong bullish signals with recent earnings beat
(Q4 guidance +$500M) aligned with positive sentiment (+0.75 score).

### Recent Market Developments
- NVDA Q4 earnings beat: AI data center revenue +40% YoY
- Analyst rating upgrade from 3 major firms
- Stock price +15% since last 10-K filing

### Risk Assessment  
Despite gains, 10-K cites "competitive pressure from AMD" —
Recent news shows AMD's new Instinct 8 GPU line launching Q2.
```

### Troubleshooting

**"NEWS_API_KEY not set"**
→ Add to `.env` file: `NEWS_API_KEY=your_key`

**"Vector store not loaded"**
→ Run `live_data_retriever.load_or_build_vector_store()` first

**"Error fetching live data"**
→ News API rate limits (500 req/day free tier)
→ yfinance might be rate-limited by Yahoo
→ Agent gracefully continues with static 10-K context only

### Next Steps

- **Add more data sources**: SEC Edgar filings, Bloomberg API, competitor data
- **Tune embedding model**: Switch from `all-MiniLM-L6-v2` to `all-mpnet-base-v2` for better quality
- **Implement caching**: Store fetched data to avoid repeated API calls
- **Add sentiment weighting**: Give higher weight to recent news in analysis

### File Organization

```
analyst.py                    ← Imports live_data_retriever
JJ/
  ├─ live_data_retriever.py   ← NEW: MCP web scraper interface
  ├─ rag.py                   ← Existing 10-K chunking
  └─ retriever.py             ← Existing 10-K retrieval
graph/
  └─ state.py                 ← Updated with live_data_context field
```
