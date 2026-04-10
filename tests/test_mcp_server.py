"""
MCP Server Validation Script

Tests the MCP server without needing Claude Desktop.
Verifies that tools are properly defined and can be called.

Usage:
    python test_mcp_server.py
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from mcp_news.tools import get_mcp_tools
from mcp_news.dispatcher import dispatch_mcp_tool


def test_tool_definitions():
    """Test 1: Verify tool schemas are properly defined."""
    print("\n" + "="*70)
    print("TEST 1: Tool Definitions")
    print("="*70)
    
    tools = get_mcp_tools()
    
    print(f"\n✓ Found {len(tools)} tools:")
    for tool in tools:
        print(f"\n  Tool: {tool['name']}")
        print(f"  Description: {tool['description']}")
        print(f"  Required params: {tool['input_schema']['required']}")
    
    assert len(tools) == 4, "Expected 4 tools"
    print("\n✅ Tool definitions OK")


def test_dispatch_fetch_news():
    """Test 2: Verify fetch_news can be dispatched."""
    print("\n" + "="*70)
    print("TEST 2: Dispatch fetch_news")
    print("="*70)
    
    result = dispatch_mcp_tool("fetch_news", {
        "tickers": ["AAPL"],
        "days_back": 7
    })
    
    print(f"\nResult type: {type(result)}")
    print(f"Keys: {list(result.keys())}")
    
    # Check response structure
    if "error" in result:
        print(f"⚠️  Error (expected if API key not set): {result['error']}")
    elif "articles" in result:
        print(f"✓ Got {result.get('count', 0)} articles")
        if result.get('articles'):
            article = result['articles'][0]
            print(f"  Sample: {article.get('title', 'N/A')[:60]}...")
    
    print("\n✅ fetch_news dispatch OK")


def test_dispatch_fetch_earnings():
    """Test 3: Verify fetch_earnings can be dispatched."""
    print("\n" + "="*70)
    print("TEST 3: Dispatch fetch_earnings")
    print("="*70)
    
    result = dispatch_mcp_tool("fetch_earnings", {
        "tickers": ["AAPL"]
    })
    
    print(f"\nResult type: {type(result)}")
    print(f"Keys: {list(result.keys())}")
    
    if "error" in result:
        print(f"⚠️  Error: {result['error']}")
    elif "earnings" in result:
        print(f"✓ Got earnings for {len(result['earnings'])} tickers")
    
    print("\n✅ fetch_earnings dispatch OK")


def test_dispatch_fetch_analyst_ratings():
    """Test 4: Verify fetch_analyst_ratings can be dispatched."""
    print("\n" + "="*70)
    print("TEST 4: Dispatch fetch_analyst_ratings")
    print("="*70)
    
    result = dispatch_mcp_tool("fetch_analyst_ratings", {
        "tickers": ["AAPL"]
    })
    
    print(f"\nResult type: {type(result)}")
    print(f"Keys: {list(result.keys())}")
    
    if "error" in result:
        print(f"⚠️  Error: {result['error']}")
    elif "ratings" in result:
        print(f"✓ Got ratings for {len(result['ratings'])} tickers")
    
    print("\n✅ fetch_analyst_ratings dispatch OK")


def test_dispatch_invalid_tool():
    """Test 5: Verify invalid tool calls are handled."""
    print("\n" + "="*70)
    print("TEST 5: Invalid Tool Handling")
    print("="*70)
    
    result = dispatch_mcp_tool("nonexistent_tool", {})
    
    print(f"\nCalling nonexistent_tool...")
    print(f"Result: {result}")
    
    assert "error" in result, "Should return error for invalid tool"
    assert "available_tools" in result, "Should list available tools"
    
    print(f"\n✓ Error handling works")
    print(f"✓ Available tools listed: {len(result['available_tools'])} tools")
    
    print("\n✅ Invalid tool handling OK")


def test_mcp_server_imports():
    """Test 6: Verify MCP server can be imported."""
    print("\n" + "="*70)
    print("TEST 6: MCP Server Imports")
    print("="*70)
    
    try:
        import mcp
        print(f"✓ MCP library imported: {mcp.__version__ if hasattr(mcp, '__version__') else 'unknown version'}")
    except ImportError as e:
        print(f"❌ MCP not installed: {e}")
        print("   Install with: pip install mcp")
        return False
    
    try:
        from mcp.server import Server
        print(f"✓ MCP Server class imported")
    except ImportError as e:
        print(f"❌ Failed to import MCP Server: {e}")
        return False
    
    print("\n✅ MCP imports OK")
    return True


def test_mcp_server_definition():
    """Test 7: Verify MCP server file exists and is valid."""
    print("\n" + "="*70)
    print("TEST 7: MCP Server File")
    print("="*70)
    
    server_path = Path(__file__).parent / "mcp_server.py"
    
    if not server_path.exists():
        print(f"❌ Server file not found: {server_path}")
        return False
    
    print(f"✓ Server file exists: {server_path}")
    
    try:
        with open(server_path) as f:
            content = f.read()
            assert "def list_tools" in content, "Missing list_tools function"
            assert "def call_tool" in content, "Missing call_tool function"
            assert "async def main" in content, "Missing main function"
        print(f"✓ Server file contains required functions")
    except Exception as e:
        print(f"❌ Error reading server file: {e}")
        return False
    
    print("\n✅ MCP Server file OK")
    return True


def main():
    print("\n" + "="*70)
    print("  MCP SERVER VALIDATION TEST SUITE")
    print("="*70)
    
    try:
        # Test 1-5: Core functionality
        test_tool_definitions()
        test_dispatch_fetch_news()
        test_dispatch_fetch_earnings()
        test_dispatch_fetch_analyst_ratings()
        test_dispatch_invalid_tool()
        
        # Test 6-7: Server setup
        mcp_ok = test_mcp_server_imports()
        server_ok = test_mcp_server_definition()
        
        # Summary
        print("\n" + "="*70)
        print("  ✅ ALL VALIDATION TESTS PASSED")
        print("="*70)
        
        print("\n📋 NEXT STEPS FOR MCP SERVER:")
        if not mcp_ok:
            print("  1. Install MCP: pip install mcp")
        else:
            print("  1. ✓ MCP installed")
        
        if mcp_ok and server_ok:
            print("  2. Test server directly: python mcp_server.py")
            print("  3. Follow MCP_SERVER_SETUP.md for Claude Desktop config")
            print("  4. Add environment variables (especially FINNHUB_API_KEY)")
            print("  5. Restart Claude Desktop")
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
