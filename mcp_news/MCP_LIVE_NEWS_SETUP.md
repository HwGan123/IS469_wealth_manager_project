## MCP Live Data Integration Setup

Your analyst agent now fetches **live financial data** (news + earnings) and combines it with static 10-K filings for enhanced analysis.

### Architecture

```
sentiment_agent (NewsAPI)
       ↓
analyst_agent
   ├─ 10-K RAG (chromadb)
   └─ LIVE DATA (MCP tools with APIs)
       ├─ News (Finnhub via MCP)
       ├─ Earnings (yfinance via MCP)
       ├─ Analyst Ratings (yfinance via MCP)
       └─ SEC Filings (SEC Edgar via MCP)
       ↓
portfolio_agent
       ↓
auditor_agent
```

### Setup

#### 1. **Environment Variables** (`.env`)

```bash
# OpenAI API (for analyst LLM)
OPENAI_API_KEY=sk-...

# Anthropic API (for Claude MCP agentic loop)
ANTHROPIC_API_KEY=sk-ant-...

# Finnhub API (for company news and financial data)
# Get key from: https://finnhub.io
FINNHUB_API_KEY=your_finnhub_key
```

#### 2. **Install Dependencies**

The setup already includes required packages in `requirements.txt`:
- `requests` — for API calls
- `yfinance` — for earnings data
- `langchain` — for orchestration
- `chromadb` — for embeddings

#### 3. **Run the Analyst Agent**

When you run the analyst node, it will autonomously fetch live data via MCP:

```bash
python main.py
```

This will:
- Initialize the 10-K RAG vector store from ChromaDB
- Claude calls MCP tools to fetch real-time news, earnings, ratings, and SEC filings
- Claude synthesizes the live data into a structured summary
- Combine static 10-K context with live data for analysis
- Generate investment insights that cite both historical and current information

### How It Works

#### Data Flow in Analyst Agent

1. **Static Context** (10-K)
   ```python
   docs = retriever.invoke(search_query)  # Retrieves 10-K chunks from ChromaDB
   rag_context = "\n\n".join([doc.page_content for doc in docs])
   ```

2. **Live Data** (MCP Tools - Agentic)
   ```python
   # Claude autonomously calls MCP tools to fetch live data
   tools = get_mcp_tools()  # fetch_news, fetch_earnings, fetch_analyst_ratings, fetch_sec_filings
   
   response = anthropic_client.messages.create(
       model="claude-3-5-sonnet-20241022",
       tools=tools,
       messages=messages
   )
   
   # Claude inspects tool results and calls more tools as needed
   # Continues until it has gathered sufficient data
   live_data_summary = result_from_claude
   ```

3. **Combined Analysis**
   - Both `rag_context` (10-K) and `live_data_summary` (live data) are passed to final LLM prompt
   - Claude synthesizes insights across historical + current data
   - Flags contradictions between 10-K guidance and recent market developments

### Configuration Options

#### In `agents/analyst.py` (Main Analyst Logic)

```python
# Adjust sensitivity to live data vs 10-K analysis
analyst_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)  
# Lower temp = more conservative, Higher = more creative

# Adjust Claude's data gathering behavior
response = anthropic_client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4096,  # Increase for longer data summaries
    tools=tools,
    messages=messages
)

# Control max iterations for Claude's agentic loop
max_iterations = 5  # Change to allow more tool calls
```

#### In `mcp_news/implementations.py` (Tool Implementations)

```python
# Adjust data source parameters
days_back = 7  # Change to 14, 30, etc. for historical data
k = 5  # Number of results to return per tool call

# Add or modify which tools Claude can call
# Current tools:
#   - fetch_news: Finnhub (financial news with sentiment scoring)
#   - fetch_earnings: yfinance (valuation metrics)
#   - fetch_analyst_ratings: yfinance (analyst consensus)
#   - fetch_sec_filings: SEC Edgar (regulatory documents)
```

### What Data Is Fetched

**From Finnhub:**
- Company news and announcements (last 7 days by default)
- Earnings news, M&A activity, regulatory filings
- Analyst reports and market sentiment
- Sentiment scores included with each article

**From yfinance:**
- Current earnings data
- P/E ratio, EPS (trailing and forward), PEG ratio
- Analyst ratings and price targets
- 52-week highs/lows, market cap

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

**"FINNHUB_API_KEY not set"**
→ Add to `.env` file: `FINNHUB_API_KEY=your_key` (Get at https://finnhub.io)

**"ANTHROPIC_API_KEY not set"**
→ Add to `.env` file: `ANTHROPIC_API_KEY=your_key`

**"Tool returned error result"**
→ Check that all API keys are set in `.env` (FINNHUB_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY)
→ Verify internet connectivity for API calls

**"API rate limits hit"**
→ Finnhub: 60 API calls/minute (free tier), 250 calls/minute (pro)
→ yfinance: May be rate-limited by Yahoo Finance
→ Agent gracefully retries or continues with static 10-K context only

### Next Steps

- **Add more MCP tools**: Extend `mcp_news/implementations.py` with additional data sources (Bloomberg, proprietary APIs, web scraping)
- **Improve Claude's data synthesis**: Adjust the prompt in `agents/analyst.py` to ask Claude for specific analysis types
- **Implement caching**: Cache MCP tool results to avoid repeated API calls for the same ticker within a session
- **Optimize 10-K embedding model**: For RAG only — tune the HuggingFace embedding model in `JJ/retriever.py` for better 10-K semantic search

### File Organization

```
agents/
  └─ analyst.py               ← Orchestrates RAG + MCP tool calling
mcp_news/
  ├─ dispatcher.py            ← Routes Claude's tool calls to implementations
  ├─ implementations.py        ← Actual tool logic (fetch_news, fetch_earnings, etc.)
  ├─ tools.py                 ← Tool definitions for Claude
  └─ __init__.py              ← Exports get_mcp_tools() and dispatch_mcp_tool()
JJ/
  ├─ rag.py                   ← 10-K chunking logic
  ├─ retriever.py             ← 10-K vector store retrieval
  └─ simple_rag.py            ← Alternative RAG implementation
graph/
  ├─ state.py                 ← WealthManagerState with live_data_context
  └─ workflow.py              ← Graph orchestration
```
