# MCP Server Setup Guide

This guide explains how to set up the Wealth Manager MCP Server and integrate it with Claude Desktop.

## What is an MCP Server?

An **MCP (Model Context Protocol) Server** exposes tools to Claude via a standardized interface. Instead of calling Python functions directly, Claude can request these tools autonomously.

**Before (Local Dispatch):**
```
Your Python Code
    ↓
dispatch_mcp_tool("fetch_news", {...})
    ↓
Returns data
```

**After (MCP Server):**
```
Claude Desktop
    ↓ (MCP protocol via stdio)
MCP Server (mcp_server.py)
    ↓ (Python function calls)
dispatch_mcp_tool("fetch_news", {...})
    ↓
Returns data to Claude
```

---

## Installation

### Step 1: Install MCP Library

```bash
pip install mcp
```

### Step 2: Verify Installation

```bash
python -c "import mcp; print('MCP installed successfully')"
```

---

## Configuration for Claude Desktop

### Step 1: Find Claude Desktop Config

The configuration location depends on your OS:

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**macOS:**
```
~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Linux:**
```
~/.config/Claude/claude_desktop_config.json
```

### Step 2: Edit Configuration

Open `claude_desktop_config.json` and add the MCP server:

```json
{
  "mcpServers": {
    "wealth-manager": {
      "command": "python",
      "args": [
        "C:\\Users\\YOUR_USERNAME\\path\\to\\IS469_wealth_manager_project\\mcp_server.py"
      ],
      "env": {
        "FINNHUB_API_KEY": "your_finnhub_api_key_here",
        "PYTHONPATH": "C:\\Users\\YOUR_USERNAME\\path\\to\\IS469_wealth_manager_project"
      }
    }
  }
}
```

### Step 3: Replace Paths

Update these placeholders:
- `YOUR_USERNAME` - Your Windows username
- `path\to\IS469_wealth_manager_project` - Actual path to the project

**Example (Windows):**
```json
{
  "mcpServers": {
    "wealth-manager": {
      "command": "python",
      "args": [
        "D:\\Codes and Repos\\IS469_wealth_manager_project\\mcp_server.py"
      ],
      "env": {
        "FINNHUB_API_KEY": "sk_free_abc123xyz...",
        "PYTHONPATH": "D:\\Codes and Repos\\IS469_wealth_manager_project"
      }
    }
  }
}
```

### Step 4: Get Your Finnhub API Key

1. Go to [finnhub.io](https://finnhub.io)
2. Sign up for free account
3. Copy your API key from dashboard
4. Paste into `FINNHUB_API_KEY` in config

### Step 5: Restart Claude Desktop

Close and reopen Claude Desktop app to load the new configuration.

---

## Testing the MCP Server

### Option 1: Manual Server Start (Testing)

Start the server directly in a terminal:

```bash
cd d:\Codes and Repos\IS469_wealth_manager_project
python mcp_server.py
```

Expected output:
```
Starting Wealth Manager MCP Server...
Available tools:
  - fetch_news: Fetch recent financial news for specific stock tickers
  - fetch_earnings: Fetch current earnings data, PE ratios, EPS
  - fetch_analyst_ratings: Fetch current analyst ratings, price targets
  - fetch_sec_filings: Fetch recent SEC filings
Ready to receive tool calls from Claude
```

This will hang waiting for input (normal - it's listening for MCP protocol messages).

Press `Ctrl+C` to stop.

### Option 2: Test if Claude Can See the Tools

In Claude Desktop, ask:

```
What tools do you have available?
```

Claude should list:
- fetch_news
- fetch_earnings
- fetch_analyst_ratings
- fetch_sec_filings

If no tools appear, check:
1. Server is running
2. Configuration path is correct
3. FINNHUB_API_KEY is set
4. Claude Desktop was restarted after config change

### Option 3: Request a Tool Call

Ask Claude to fetch live data:

```
Can you fetch the latest news about Apple (AAPL)?
```

Claude will:
1. Recognize you want news about Apple
2. Call `fetch_news` with `{"tickers": ["AAPL"], "days_back": 7}`
3. Return results as context
4. Answer your question with the fresh data

---

## How It Works

### Architecture

```
claude_desktop_config.json describes server
    ↓
Claude Desktop launches: python mcp_server.py
    ↓
mcp_server.py starts MCP Server
    ↓
Server advertises tools (list_tools())
    ↓
Claude sees tools and can invoke them
    ↓
User asks Claude a question
    ↓
Claude calls dispatch_mcp_tool()
    ↓
dispatch_mcp_tool routes to implementation
    ↓
API calls (Finnhub, yfinance, etc.)
    ↓
Results returned to Claude
    ↓
Claude answers user's question with fresh data
```

### Key Files

| File | Purpose |
|------|---------|
| `mcp_server.py` | MCP server entry point |
| `mcp_news/tools.py` | Tool schema definitions |
| `mcp_news/dispatcher.py` | Route tool calls |
| `mcp_news/implementations.py` | Actual API calls |
| `claude_desktop_config.json` | Claude registration |

---

## Troubleshooting

### Error: "mcp module not found"

**Solution:** Install MCP
```bash
pip install mcp
```

### Error: "Command not found: python"

**Solution:** Use full path to Python in config
```json
{
  "command": "C:\\Python311\\python.exe",
  "args": ["D:\\...\\mcp_server.py"]
}
```

### Error: "FINNHUB_API_KEY not configured"

**Solution:** Add to config file
```json
"env": {
  "FINNHUB_API_KEY": "your_api_key_here"
}
```

### Claude doesn't see the tools

**Checklist:**
1. ✅ Server running? (check terminal for "Starting...")
2. ✅ Config path correct? (check filename and location)
3. ✅ Claude restarted? (close and reopen)
4. ✅ JSON syntax valid? (use JSON linter)
5. ✅ PYTHONPATH set? (needed for imports)

### Server crashes on startup

Check stderr output for errors:
```bash
python mcp_server.py 2>&1
```

Common issues:
- Missing dependencies: `pip install -r requirements.txt`
- Wrong path: Check `PYTHONPATH` in config
- Import error: Verify `mcp_news` module exists

---

## Advanced Configuration

### Multiple MCP Servers

You can register multiple servers in the same config:

```json
{
  "mcpServers": {
    "wealth-manager": {
      "command": "python",
      "args": ["D:\\...\\mcp_server.py"],
      "env": {"FINNHUB_API_KEY": "..."}
    },
    "another-server": {
      "command": "python",
      "args": ["D:\\...\\another_server.py"],
      "env": {}
    }
  }
}
```

### Environment Variables

Pass any environment variables the server needs:

```json
{
  "env": {
    "FINNHUB_API_KEY": "sk_free_...",
    "DEBUG": "true",
    "LOG_LEVEL": "INFO"
  }
}
```

### Custom Python Executable

If you use a virtual environment:

```json
{
  "command": "D:\\venv\\Scripts\\python.exe",
  "args": ["D:\\...\\mcp_server.py"]
}
```

---

## Next Steps

1. **Install MCP:** `pip install mcp`
2. **Get Finnhub API Key:** [finnhub.io](https://finnhub.io)
3. **Update Config:** Add server to `claude_desktop_config.json`
4. **Restart Claude:** Close and reopen app
5. **Test:** Ask Claude to fetch data

---

## References

- [MCP Documentation](https://modelcontextprotocol.io/)
- [Claude Integration Guide](https://claude.ai/docs)
- [Finnhub API Docs](https://finnhub.io/docs/api)

---

## Questions?

If the server isn't working:
1. Check all paths are absolute (not relative)
2. Verify FINNHUB_API_KEY is valid
3. Ensure Python can import `mcp` and `mcp_news`
4. Review Claude app logs for error messages
5. Try running `mcp_server.py` directly first
