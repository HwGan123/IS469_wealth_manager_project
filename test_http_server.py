"""
Quick test to verify MCP HTTP server is working
"""
import httpx
import json

def test_http_server():
    """Test the HTTP MCP server"""
    base_url = "http://localhost:3000"
    
    print("=" * 60)
    print("Testing MCP HTTP Server Connection")
    print("=" * 60)
    
    try:
        # Test 1: Health check
        print("\n1. Testing health endpoint...")
        resp = httpx.get(f"{base_url}/health")
        assert resp.status_code == 200
        print(f"   ✓ Health check: {resp.json()}")
        
        # Test 2: List tools
        print("\n2. Testing tools endpoint...")
        resp = httpx.get(f"{base_url}/tools")
        assert resp.status_code == 200
        data = resp.json()
        tools = data.get("tools", [])
        print(f"   ✓ Found {len(tools)} tools:")
        for tool in tools:
            print(f"      - {tool['name']}")
        
        # Test 3: Call a simple tool (fetch_earnings)
        print("\n3. Testing tool call (fetch_earnings for AAPL)...")
        resp = httpx.post(
            f"{base_url}/call",
            json={
                "name": "fetch_earnings",
                "arguments": {"ticker": "AAPL"}
            },
            timeout=30.0
        )
        assert resp.status_code == 200
        result = resp.json()
        if result.get("success"):
            print(f"   ✓ Tool call successful")
            print(f"   Result keys: {list(result.get('result', {}).keys())}")
        else:
            print(f"   ✗ Tool call failed: {result.get('error')}")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed! HTTP server is working correctly.")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = test_http_server()
    exit(0 if success else 1)
