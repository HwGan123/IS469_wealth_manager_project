import os
import sys
from dotenv import load_dotenv

# Load environment variables (API Keys) FIRST, before any agent imports
load_dotenv()

from graph.workflow import create_wealth_manager_graph

def main():
    print("--- 🚀 Initializing AI Wealth Manager System ---")
    
    # Compile the Graph
    app = create_wealth_manager_graph()
    
    # Initial input for the system
    # In a real-world case, this would be a specific client request
    initial_input = {
        "messages": ["Should I invest more in AAPL and NVDA given the current AI boom? What about tech sector volatility?"],
        "tickers": ["AAPL", "NVDA", "GOOG", "MSFT"],
        "news_articles": [],
        "sentiment_results": [],
        "sentiment_summary": {},
        "sentiment_score": 0.0,
        "portfolio_weights": {},
        "retrieved_context": "",
        "live_data_context": "",
        "draft_report": "",
        "audit_score": 0.0,
        "is_hallucinating": False,
        "audit_iteration_count": 0
    }

    # Execute the workflow
    # Using stream allows you to see the output of each agent in real-time
    for output in app.stream(initial_input):
        for key, value in output.items():
            print(f"\n--- Node '{key}' finished ---")
            if "messages" in value:
                print(f"Latest Update: {value['messages'][-1]}")


def run_remote(url: str = "http://localhost:2024"):
    """Stream from a running LangGraph dev server (model stays warm between runs)."""
    try:
        from langgraph_sdk import get_sync_client
    except ImportError:
        print("langgraph-sdk not installed. Run: uv add langgraph-sdk")
        sys.exit(1)

    print(f"--- AI Wealth Manager (remote mode → {url}) ---")
    client = get_sync_client(url=url)

    thread = client.threads.create()
    for chunk in client.runs.stream(
        thread["thread_id"],
        "wealth_manager",
        input=INITIAL_INPUT,
        stream_mode="updates",
    ):
        if chunk.data and isinstance(chunk.data, dict):
            for key, value in chunk.data.items():
                print(f"\n--- Node '{key}' finished ---")
                if isinstance(value, dict) and "messages" in value:
                    msgs = value["messages"]
                    if msgs:
                        print(f"Latest Update: {msgs[-1]}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "local"
    if mode == "remote":
        run_remote()
    else:
        main()