import os
from dotenv import load_dotenv
from graph.workflow import create_wealth_manager_graph

# Load environment variables (API Keys)
load_dotenv()

def main():
    print("--- 🚀 Initializing AI Wealth Manager System ---")
    
    # Compile the Graph
    app = create_wealth_manager_graph()
    
    # Initial input for the system
    # In a real-world case, this would be a specific client request
    initial_input = {
        "messages": ["Analyze the impact of recent tech sector volatility on a high-growth portfolio."],
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
        "is_hallucinating": False
    }

    # Execute the workflow
    # Using stream allows you to see the output of each agent in real-time
    for output in app.stream(initial_input):
        for key, value in output.items():
            print(f"\n--- Node '{key}' finished ---")
            if "messages" in value:
                print(f"Latest Update: {value['messages'][-1]}")

if __name__ == "__main__":
    main()