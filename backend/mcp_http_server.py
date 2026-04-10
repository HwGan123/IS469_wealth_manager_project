"""
HTTP Server for MCP Tools - Wealth Manager

Runs as a persistent service in one terminal.
Other processes connect to it via HTTP.

Usage:
    python mcp_http_server.py

Then access tools via:
    GET http://localhost:3000/tools - List available tools
    POST http://localhost:3000/call - Call a tool
"""

import json
import sys
from dotenv import load_dotenv
from typing import Dict, Any
import logging

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mcp-http-server")

try:
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("ERROR: starlette and uvicorn required for HTTP server", file=sys.stderr)
    print("Install with: pip install starlette uvicorn", file=sys.stderr)
    sys.exit(1)

from mcp_news.tools import get_mcp_tools
from mcp_news.dispatcher import dispatch_mcp_tool


# Tool definitions cache
_tools_cache = None


def get_tools_list() -> list:
    """Get list of available tools."""
    global _tools_cache
    if _tools_cache is None:
        _tools_cache = get_mcp_tools()
    return _tools_cache


async def list_tools(request):
    """Endpoint: GET /tools - List available tools."""
    tools = get_tools_list()
    return JSONResponse({
        "success": True,
        "tools": tools,
        "count": len(tools)
    })


async def call_tool(request):
    """Endpoint: POST /call - Call a tool."""
    try:
        data = await request.json()
        tool_name = data.get("name")
        arguments = data.get("arguments", {})
        
        if not tool_name:
            return JSONResponse(
                {"success": False, "error": "Missing 'name' parameter"},
                status_code=400
            )
        
        logger.info(f"Calling tool: {tool_name}")
        logger.info(f"  Arguments: {str(arguments)[:100]}...")
        
        try:
            # Call the tool via dispatcher
            result = dispatch_mcp_tool(tool_name, arguments)
            
            logger.info(f"  Success! Result size: {len(json.dumps(result))} bytes")
            
            return JSONResponse({
                "success": True,
                "name": tool_name,
                "result": result
            })
            
        except Exception as e:
            logger.error(f"  Tool error: {e}", exc_info=True)
            return JSONResponse(
                {
                    "success": False,
                    "name": tool_name,
                    "error": str(e)
                },
                status_code=500
            )
    
    except Exception as e:
        logger.error(f"Request parsing error: {e}", exc_info=True)
        return JSONResponse(
            {"success": False, "error": f"Invalid request: {str(e)}"},
            status_code=400
        )


async def health(request):
    """Endpoint: GET /health - Server health check."""
    return JSONResponse({"status": "OK", "service": "mcp-http-server"})


# Create Starlette app
app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/tools", list_tools, methods=["GET"]),
        Route("/call", call_tool, methods=["POST"]),
    ]
)

# Add CORS middleware for cross-origin requests
app = CORSMiddleware(
    app,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def main():
    """Start the HTTP server."""
    logger.info("=" * 60)
    logger.info("Starting MCP HTTP Server")
    logger.info("=" * 60)
    
    # List available tools
    tools = get_tools_list()
    logger.info(f"Available tools ({len(tools)}):")
    for tool in tools:
        logger.info(f"  - {tool['name']}: {tool['description']}")
    
    logger.info("=" * 60)
    logger.info("Starting HTTP server on http://127.0.0.1:3000")
    logger.info("Endpoints:")
    logger.info("  GET  http://localhost:3000/health - Health check")
    logger.info("  GET  http://localhost:3000/tools  - List tools")
    logger.info("  POST http://localhost:3000/call   - Call a tool")
    logger.info("=" * 60)
    
    # Run uvicorn server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=3000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    import asyncio
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
