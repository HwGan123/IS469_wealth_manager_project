# Market Context Agent Experiments

## Overview

The Market Context Agent is an intelligent data gathering component that uses Claude with MCP tools to fetch relevant financial market data for investment analysis.

**Key Features:**
- Claude intelligently decides which tools to call based on user request
- Full results cached for downstream agents
- Context-aware result summarization to prevent token overflow
- Minimal token usage (~7-8.5K per workflow)
- Cost-effective ($0.009-0.012 per workflow at current Haiku pricing)

## Test Suite

This directory contains three complementary test files:

### 1. **test_agent.py** - Integration Testing with Token Tracking
Tests the complete market context agent with real API calls and cost estimation.

```bash
python test_agent.py
```

**Tests included:**
- `test_market_context_minimal()`: 2-ticker analysis (~7.4K tokens, $0.009)
- `test_market_context_detailed()`: 4-ticker comprehensive analysis (~8.5K tokens, $0.012)

**What it validates:**
- Token usage accuracy
- Cost model correctness (Haiku 4.5 pricing)
- Multiple ticker handling without context overflow
- Rate limit compliance (70-second pause between tests)

**Requirements:** ANTHROPIC_API_KEY set in .env

### 2. **test_market_context_node.py** - Unit Testing with Mocks
Tests the market context node with mocked MCP tool calls.

```bash
python test_market_context_node.py
```

**Tests included:**
- Imports validation
- `_determine_data_needs()` logic
- Selective tool invocation
- Missing ticker handling
- Error handling

**What it validates:**
- Core functionality without API calls
- Tool selection logic
- State management
- Error resilience

**Requirements:** None (uses unittest.mock)

### 3. **test_full_workflow.py** - End-to-End Integration Testing
Tests the complete wealth manager workflow including orchestrator, market context, sentiment, analyst, auditor, and report generator.

```bash
python test_full_workflow.py
```

**Tests included:**
- Orchestrator ticker extraction
- Orchestrator no-tickers routing
- Market context initialization
- Full workflow with tickers
- Full workflow without tickers

**What it validates:**
- Workflow graph connectivity
- State management across agents
- Routing logic
- End-to-end data flow

**Requirements:** OPENAI_API_KEY and ANTHROPIC_API_KEY set in .env (for full tests)

## Running All Tests

Create a simple test runner:

```python
# run_all_tests.py
import subprocess
import sys

tests = [
    "test_market_context_node.py",      # Unit tests (fast, no API)
    "test_agent.py",                    # Integration tests (real API)
    "test_full_workflow.py",            # End-to-end tests (full workflow)
]

for test in tests:
    print(f"\n{'='*80}")
    print(f"Running {test}...")
    print('='*80)
    result = subprocess.run([sys.executable, test])
    if result.returncode != 0:
        print(f"FAILED: {test}")
        sys.exit(1)

print("\nAll tests passed!")
```

## Architecture

### Flow
```
User Request
    ↓
Market Context Agent
    ├─ Iteration 1: Claude decides which tools to call
    │   ├─ Call fetch_news
    │   ├─ Call fetch_earnings
    │   ├─ Call fetch_analyst_ratings
    │   ├─ Call fetch_sec_filings (optional)
    │   └─ Call fetch_10k_content (optional)
    │
    ├─ Full results cached in market_context[tool_name]
    │
    ├─ Iteration 2: Summarized results sent to Claude
    │   └─ Claude synthesizes into summary
    │
    └─ Returns to workflow:
        ├─ market_context["fetch_news"] = [480 articles - FULL]
        ├─ market_context["fetch_earnings"] = {...}
        ├─ market_context["fetch_analyst_ratings"] = {...}
        ├─ market_context["fetch_sec_filings"] = {...}
        └─ market_context["summary"] = "Claude's synthesis"
```

### Result Summarization (Option 4)

To prevent context overflow, full tool results are cached but summarized versions are sent to Claude:

- **fetch_news**: Top 10 articles shown (out of full ~480)
- **fetch_sec_filings**: Top 5 metadata shown (full data cached)
- **fetch_10k_content**: Truncated summaries shown (full content cached)
- **fetch_earnings**: Full data shown (usually lean)
- **fetch_analyst_ratings**: Full data shown (usually lean)

**Benefits:**
- Downstream agents always get complete data
- Claude conversation stays efficient (<10K tokens total)
- No context window overflow errors
- Allows 4+ ticker analysis without issues

## Testing

### Run Tests

```bash
cd experiments/market_context_agent
python test_agent.py
```

### Test Scenarios

**Test 1: Quick Check (2 tickers)**
- User Request: "Quick sentiment check on tech stocks"
- Tickers: AAPL, NVDA
- Tools Called: fetch_news, fetch_analyst_ratings, fetch_earnings
- Tokens: ~6.5K input, ~1K output
- Cost: ~$0.009
- Time: ~15 seconds

**Test 2: Detailed Analysis (4 tickers)**
- User Request: "Analyze tech sector volatility impact focusing on earnings and sentiment"
- Tickers: AAPL, GOOGL, MSFT, NVDA
- Tools Called: fetch_news, fetch_earnings, fetch_analyst_ratings
- Tokens: ~7K input, ~1.5K output
- Cost: ~$0.012
- Time: ~20 seconds

### Token Usage Tracking

All tests include real token tracking from Claude API responses:
```
Iteration 1: 1437 input, 223 output
Iteration 2: 5025 input, 733 output
─────────────────────────────────
Total: 6462 input, 956 output
Cost: $0.008994
```

## Implementation Details

### Files

- **agents/market_context.py** - Main agent implementation
  - `market_context_node()` - Entry point
  - `_summarize_tool_result()` - Context management function
  
- **mcp_news/implementations.py** - Tool implementations
  - `fetch_news()` - Uses Finnhub API
  - `fetch_earnings()` - Uses yfinance
  - `fetch_analyst_ratings()` - Uses Finnhub API
  - `fetch_sec_filings()` - Uses SEC Edgar API
  - `fetch_10k_content()` - Uses SEC Edgar + RAG

### Configuration

Uses Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) for cost efficiency.

**Pricing:**
- Input: $0.80 per 1M tokens
- Output: $4.00 per 1M tokens

## Integration with Workflow

The market context agent runs early in the workflow and provides cached data to all downstream agents:

```python
# In workflow.py
def create_wealth_manager_graph():
    ...
    # Market context runs first
    graph.add_node("market_context", market_context_node)
    
    # All other agents use the cached data
    graph.add_node("sentiment", sentiment_agent)
    graph.add_node("analyst", analyst_agent)
    graph.add_node("auditor", auditor_agent)
    graph.add_node("report_generator", report_generator_node)
```

## Results

### Test Results (April 7, 2026)

```
TEST 1: Quick Check (2 tickers)
✓ Completed in 2 iterations
✓ Tokens: 6,462 input, 956 output
✓ Cost: $0.008994
✓ Data: 480 articles, analyst ratings, earnings
✓ Status: PASS

TEST 2: Detailed Analysis (4 tickers)
✓ Completed in 2 iterations
✓ Tokens: 6,935 input, 1,568 output
✓ Cost: $0.011820
✓ Data: 949 articles, analyst ratings, earnings
✓ Status: PASS

Overall: ✅ ALL TESTS PASSED
```

## Cost Projections

| Scale | Approx. Cost |
|-------|-------------|
| 1,000 workflows | $10-12 |
| 10,000 workflows | $100-125 |
| 100,000 workflows | $1,000-1,250 |

## Future Improvements

1. **Caching** - Cache results by ticker/date to avoid redundant API calls
2. **Batch processing** - Fetch data for multiple tickers in single API call
3. **Selective calling** - Add config to enable/disable specific tools based on workflow needs
4. **Custom summarization** - Allow downstream agents to request full vs summarized data

## Troubleshooting

### Context Window Exceeded (200K tokens)

If you get a 400 error about prompt being too long:

1. Disable analyst mode: `workflow_config={"enable_analyst_mode": False}`
2. Reduce ticker count from 4 to 2
3. Skip SEC filings if not needed

Result summarization (Option 4) is already implemented to prevent this.

### Rate Limit Errors (429)

Anthropic has rate limits: 50K tokens/minute for some tiers.

- Tests automatically wait 70 seconds between runs
- Space out production workflows if hitting limits
- Contact Anthropic sales for higher limits

### Missing API Keys

Ensure `.env` file has:
```
ANTHROPIC_API_KEY=...
FINNHUB_API_KEY=...
```

## References

- [Anthropic Models](https://docs.anthropic.com/en/docs/about/models/overview)
- [Finnhub API](https://finnhub.io/docs/api)
- [SEC EDGAR API](https://www.sec.gov/edgar.shtml)
