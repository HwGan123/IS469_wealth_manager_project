import os
import json
import anthropic
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from graph.state import WealthManagerState
from mcp_news import get_mcp_tools, dispatch_mcp_tool

# Load environment variables
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# TOGGLE: Switch between MCP live data fetching and cached market context
# ─────────────────────────────────────────────────────────────────────────────
USE_MCP_TOOLS = False  # Set to True to use Anthropic MCP tool calling for live data
                       # Set to False to use cached market_context from market_context_agent


# Setup local embeddings for 10-K retrieval
model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

# Connect to the 10-K Vector Store (use absolute path)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "chroma_db"

vectorstore = Chroma(persist_directory=str(DB_PATH), embedding_function=hf_embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# Setup the Analyst LLM
analyst_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# Setup Anthropic client for MCP-based live data fetching (used only if USE_MCP_TOOLS=True)
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def _format_market_context(market_context: dict) -> str:
    """Format cached market context data into a readable summary."""
    if not market_context:
        return "No market context data available."
    
    summary_parts = []
    
    # Add the summary if present
    if "summary" in market_context:
        summary_parts.append(market_context["summary"])
    
    # Add key data points
    if "fetch_news" in market_context:
        articles = market_context["fetch_news"].get("articles", [])
        if articles:
            summary_parts.append(f"\n**Recent News ({len(articles)} articles):**")
            for article in articles[:5]:
                summary_parts.append(f"- {article.get('title', 'N/A')}")
    
    if "fetch_earnings" in market_context:
        summary_parts.append(f"\n**Earnings Data:**\n{json.dumps(market_context['fetch_earnings'], indent=2)}")
    
    if "fetch_analyst_ratings" in market_context:
        summary_parts.append(f"\n**Analyst Ratings:**\n{json.dumps(market_context['fetch_analyst_ratings'], indent=2)}")
    
    return "\n".join(summary_parts)


def analyst_node(state: WealthManagerState):
    """
    Investment analyst node that combines:
    - Historical 10-K context from RAG (vector store)
    - Cached market data from market_context_agent
    - Sentiment analysis from sentiment_agent
    """
    print("--- AGENT: INVESTMENT ANALYST (RAG + MCP LIVE DATA) ---")

    # Extract tickers from portfolio or state
    tickers = list(state.get("portfolio_weights", {}).keys()) or state.get("tickers", [])
    if not tickers:
        print("  ⚠️  No tickers in state. Skipping analysis.")
        return {
            "draft_report": "Awaiting portfolio data.",
            "retrieved_context": "",
            "live_data_context": "",
            "messages": ["Analyst: Awaiting portfolio data"],
            # Preserve audit-related state for the audit loop
            "audit_iteration_count": state.get("audit_iteration_count", 0),
            "is_hallucinating": state.get("is_hallucinating", False),
            "audit_score": state.get("audit_score", 0.0),
            "hallucination_count": state.get("hallucination_count", 0),
            "verified_count": state.get("verified_count", 0),
            "unsubstantiated_count": state.get("unsubstantiated_count", 0),
            "ragas_metrics": state.get("ragas_metrics", {}),
            "audit_findings": state.get("audit_findings", [])
        }

    print(f"  Analyzing: {tickers}")
    print(f"  Sentiment Score: {state.get('sentiment_score', 0.0)}")

    # Retrieve 10-K context from ticker-specific collections
    search_query = f"Financial risks, growth drivers, and strategy"
    all_docs = []
    
    # Try to fetch from ticker-specific collections
    for ticker in tickers:
        try:
            collection_name = f"sec_10k_{ticker.lower()}"
            ticker_vectorstore = Chroma(
                persist_directory=str(DB_PATH),
                collection_name=collection_name,
                embedding_function=hf_embeddings
            )
            ticker_retriever = ticker_vectorstore.as_retriever(search_kwargs={"k": 3})
            ticker_docs = ticker_retriever.invoke(search_query)
            all_docs.extend(ticker_docs)
        except Exception as e:
            pass
    
    # Fallback to generic sec_10k collection if no ticker-specific docs found
    if not all_docs:
        try:
            fallback_vectorstore = Chroma(
                persist_directory=str(DB_PATH),
                collection_name="sec_10k",
                embedding_function=hf_embeddings
            )
            fallback_retriever = fallback_vectorstore.as_retriever(search_kwargs={"k": 3})
            all_docs = fallback_retriever.invoke(search_query)
        except Exception as e:
            pass
    
    rag_context = "\n\n".join([doc.page_content for doc in all_docs])
    print(f"  📚 Retrieved {len(all_docs)} documents ({len(rag_context)} chars)")

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITIONAL: Fetch live data based on USE_MCP_TOOLS toggle
    # ─────────────────────────────────────────────────────────────────────────
    if USE_MCP_TOOLS:
        # Autonomously fetch live data via MCP
        print("  Invoking Claude with MCP tools for live data...")
        
        mcp_prompt = f"""
        You are an investment research assistant with access to live financial data tools.
        
        I need you to gather comprehensive live data for this portfolio analysis:
        
        PORTFOLIO TICKERS: {', '.join(tickers)}
        
        Please use the available tools to fetch:
        1. Recent news and market developments
        2. Current earnings data and valuation metrics
        3. Current analyst ratings and sentiment
        4. Any other relevant real-time data
        
        For each query, call the appropriate tools to gather complete information.
        Then synthesize the data into a structured summary.
        """
        
        messages = [{"role": "user", "content": mcp_prompt}]
        tools = get_mcp_tools()
        
        # Agentic loop: Claude calls tools until it's done
        live_data_summary = ""
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            response = anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                tools=tools,
                messages=messages
            )
            
            # Check if Claude is done
            if response.stop_reason == "end_turn":
                # Extract final response
                for block in response.content:
                    if hasattr(block, 'text'):
                        live_data_summary = block.text
                break
            
            # Process tool calls
            if response.stop_reason == "tool_use":
                # Add assistant response to conversation
                messages.append({"role": "assistant", "content": response.content})
                
                # Process each tool call
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        
                        print(f"    → Calling {tool_name}({list(tool_input.keys())})")
                        
                        # Execute tool
                        result = dispatch_mcp_tool(tool_name, tool_input)
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result)
                        })
                
                # Add tool results to conversation
                messages.append({"role": "user", "content": tool_results})
            else:
                break
        
        print(f"  ✓ MCP tools fetched live data ({iteration} calls)")
    else:
        # Use cached market context from market_context_agent
        market_context = state.get("market_context", {})
        live_data_summary = _format_market_context(market_context)
        print(f"  ✓ Using cached market context ({len(market_context)} keys)")

    # Generate final analysis
    final_prompt = ChatPromptTemplate.from_template("""
    You are a Senior Investment Analyst synthesizing a professional investment report.
    
    PORTFOLIO CONTEXT:
    - Tickers: {tickers}
    - Portfolio Weights: {weights}
    - Current Sentiment Score: {sentiment}
    
    STATIC CONTEXT (10-K Historical Data):
    {rag_context}
    
    LIVE MARKET DATA (Real-time News, Earnings, Ratings):
    {live_data}
    
    INSTRUCTIONS:
    1. Synthesize insights from both historical (10-K) and live market data
    2. Highlight recent catalysts (earnings beats, news, rating changes)
    3. If sentiment is bearish, emphasize risks and headwinds
    4. If sentiment is bullish, emphasize growth drivers and opportunities
    5. Flag contradictions between 10-K guidance and recent developments
    6. Provide specific, actionable recommendations
    
    Output a STRUCTURED MARKDOWN report with sections:
    - Executive Summary
    - Portfolio Overview
    - Company-by-Company Analysis
    - Recent Developments & Catalysts
    - Risk Assessment
    - Investment Recommendations
    """)

    chain = final_prompt | analyst_llm
    final_response = chain.invoke({
        "tickers": ", ".join(tickers),
        "sentiment": state.get("sentiment_score", 0.0),
        "weights": state.get("portfolio_weights", {}),
        "rag_context": rag_context,
        "live_data": live_data_summary
    })

    return {
        "draft_report": final_response.content,
        "retrieved_context": rag_context,
        "live_data_context": live_data_summary,
        "messages": [final_response.content],
        "tickers": tickers,
        # Preserve audit-related state for the audit loop
        "audit_iteration_count": state.get("audit_iteration_count", 0),
        "is_hallucinating": state.get("is_hallucinating", False),
        "audit_score": state.get("audit_score", 0.0),
        "hallucination_count": state.get("hallucination_count", 0),
        "verified_count": state.get("verified_count", 0),
        "unsubstantiated_count": state.get("unsubstantiated_count", 0),
        "ragas_metrics": state.get("ragas_metrics", {}),
        "audit_findings": state.get("audit_findings", [])
    }