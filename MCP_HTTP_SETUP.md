# MCP HTTP Server Setup

## Overview

This project uses the **Model Context Protocol (MCP)** with an **HTTP-based transport** to enable Claude autonomous tool calling for financial data fetching.

## Architecture

```
Terminal 1 (MCP Server)           Terminal 2 (Main Workflow)
┌─────────────────────────┐       ┌──────────────────────────┐
│ python mcp_http_server  │       │ python main.py           │
│                         │       │                          │
│ - HTTP server on :3000  │◄─────►│ - market_context_agent   │
│ - 6 financial tools     │ HTTP  │ - Calls tools via HTTP   │
│ - Tool dispatcher       │       │ - Claude agentic loop    │
└─────────────────────────┘       └──────────────────────────┘
```

## Quick Start

### Terminal 1: Start the MCP Server
```bash
python mcp_http_server.py
```

Expected output:
```
2026-04-09 17:51:46,400 - mcp-http-server - INFO - Starting MCP HTTP Server
2026-04-09 17:51:46,401 - mcp-http-server - INFO - Available tools (6):
  - fetch_news
  - fetch_earnings
  - fetch_analyst_ratings
  - fetch_10k_content
  - fetch_10q_content
  - fetch_xbrl_financials
2026-04-09 17:51:46,402 - mcp-http-server - INFO - Uvicorn running on http://127.0.0.1:3000
```

### Terminal 2: Run the Workflow
```bash
python main.py
```

The workflow will automatically:
1. Connect to the HTTP server at `http://localhost:3000`
2. Fetch available tools
3. Run Claude agentic loop with MCP tool calling
4. Fetch market data for portfolio tickers
5. Continue with sentiment, analyst, auditor, and report generation

## Testing

### Test HTTP Server Connection
```bash
python test_http_server.py
```

This verifies:
- ✓ Server health check
- ✓ Tools list endpoint
- ✓ Tool call endpoint

## Files

### Core Components
- **`mcp_http_server.py`** - HTTP server providing MCP tools
- **`agents/market_context.py`** - Agent that connects to HTTP server and calls Claude with tools
- **`test_http_server.py`** - Quick test to verify setup

### API Endpoints
All endpoints run on `http://localhost:3000`:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET    | `/health` | Server health check |
| GET    | `/tools` | List available tools |
| POST   | `/call` | Call a tool |

### Tool Call Example
```python
import httpx

response = httpx.post(
    "http://localhost:3000/call",
    json={
        "name": "fetch_earnings",
        "arguments": {"ticker": "AAPL"}
    }
)

result = response.json()
# {
#     "success": True,
#     "name": "fetch_earnings",
#     "result": { ... }
# }
```

## Key Features

✅ **Two-Terminal Setup** - MCP server runs independently in one terminal, main workflow in another

✅ **HTTP Transport** - More reliable than stdio for subprocess communication

✅ **6 Financial Tools** - News, earnings, analyst ratings, 10-K, 10-Q, XBRL financials

✅ **Claude Autonomous Tool Calling** - Claude decides which tools to call via MCP

✅ **Tool Result Summarization** - Compresses results to prevent token bloat in Claude loop

✅ **Multi-Iteration Support** - Claude can call tools up to 3 times with results fed back

## Dependencies

```
- mcp>=0.1.0
- starlette>=0.35.0
- uvicorn>=0.27.0
- httpx>=0.24.0
- anthropic>=0.28.0
```

Install via: `pip install -r requirements.txt`

## Troubleshooting

### Main.py can't connect to server
- Ensure `python mcp_http_server.py` is running in another terminal
- Check server is listening on `http://localhost:3000`
- Run `python test_http_server.py` to verify connectivity

### Market context taking too long
- Market context calls multiple tools (10-K, 10-Q, XBRL, news, earnings, ratings)
- Each API call takes 1-3 seconds
- Multiple iterations with Claude adds thinking time
- Total time: 30-60 seconds is normal

### Tool timeouts
- Default timeout: 20 seconds per tool call
- Some SEC filings may take longer
- Configure in `agents/market_context.py` line ~295

## Architecture Decisions

1. **HTTP over Stdio** - More reliable for long-running API calls, no buffer issues
2. **Two-Terminal Design** - Keeps MCP server isolated, easier to debug
3. **Tool Summarization** - Prevents Claude context window bloat from large SEC filings
4. **Agentic Loop (3 iterations)** - Claude can refine tool calls with results

## Next Steps

- Monitor server logs in Terminal 1 to see tool execution
- Check main workflow logs in Terminal 2
- Analyze market_context results in final report
