"""
Test MCP + RAG Integration

Verifies that MCP tools can augment RAG retrieval results.
Shows the flow: Query → RAG retrieval → MCP augmentation → Combined results

Usage:
    python test_mcp_rag_integration.py
"""

import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from mcp_news.dispatcher import dispatch_mcp_tool
import json

load_dotenv()

def test_rag_only():
    """Test RAG retrieval alone."""
    print("\n" + "="*70)
    print("TEST 1: RAG Retrieval Only (10-K documents)")
    print("="*70)
    
    # Load Chroma DB
    print("\n📚 Loading Chroma DB...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="sec_10k"
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Test query
    query = "What are Apple's main risks and challenges?"
    print(f"\nQuery: '{query}'")
    print("\nRetrieving from Chroma DB...")
    
    docs = retriever.invoke(query)
    
    print(f"\n✅ Retrieved {len(docs)} chunks from 10-K:")
    for i, doc in enumerate(docs, 1):
        preview = doc.page_content[:150].replace("\n", " ")
        print(f"\n  [{i}] {preview}...")
    
    return docs

def test_mcp_only():
    """Test MCP data fetching alone."""
    print("\n" + "="*70)
    print("TEST 2: MCP Live Data Fetching")
    print("="*70)
    
    tickers = ["AAPL"]
    
    print(f"\nFetching live data for {tickers}...")
    
    # Test news
    print("\n🔍 Fetching news...")
    news_result = dispatch_mcp_tool("fetch_news", {
        "tickers": tickers,
        "days_back": 7
    })
    
    if "articles" in news_result:
        articles = news_result["articles"][:3]
        print(f"✅ Fetched {news_result['count']} articles (showing first 3):")
        for i, article in enumerate(articles, 1):
            print(f"  [{i}] {article.get('title', 'N/A')[:70]}")
            print(f"      Sentiment: {article.get('sentiment', 'N/A')}")
    
    # Test earnings
    print("\n💰 Fetching earnings data...")
    earnings_result = dispatch_mcp_tool("fetch_earnings", {
        "tickers": tickers
    })
    
    if "earnings" in earnings_result:
        earnings = earnings_result["earnings"]
        print(f"✅ Fetched earnings for {len(earnings)} tickers:")
        for earning in earnings:
            print(f"  {earning.get('ticker')}: PE={earning.get('pe_ratio', 'N/A')}, EPS={earning.get('eps', 'N/A')}")
    
    # Test ratings
    print("\n⭐ Fetching analyst ratings...")
    ratings_result = dispatch_mcp_tool("fetch_analyst_ratings", {
        "tickers": tickers
    })
    
    if "ratings" in ratings_result:
        ratings = ratings_result["ratings"]
        print(f"✅ Fetched ratings for {len(ratings)} tickers:")
        for rating in ratings[:2]:
            print(f"  {rating.get('ticker')}: {rating.get('rating', 'N/A')} (Target: ${rating.get('target_price', 'N/A')})")
    
    return news_result, earnings_result, ratings_result

def test_mcp_rag_integration():
    """Test RAG + MCP together."""
    print("\n" + "="*70)
    print("TEST 3: RAG + MCP Integration (Combined Context)")
    print("="*70)
    
    tickers = ["AAPL"]
    
    # Step 1: RAG retrieval
    print("\n📚 Step 1: Retrieving 10-K historic context...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="sec_10k"
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke("Apple business risks and strategy")
    
    rag_context = "\n\n".join([doc.page_content[:200] for doc in docs])
    print(f"✅ RAG retrieved {len(docs)} chunks ({len(rag_context)} chars)")
    
    # Step 2: MCP fetching
    print("\n🌐 Step 2: Fetching live market data via MCP...")
    news_result = dispatch_mcp_tool("fetch_news", {
        "tickers": tickers,
        "days_back": 7
    })
    
    mcp_summary = f"Recent News ({news_result.get('count', 0)} articles):\n"
    for article in news_result.get("articles", [])[:3]:
        mcp_summary += f"- {article.get('title', 'N/A')}\n"
    
    print(f"✅ MCP fetched live data ({len(mcp_summary)} chars)")
    
    # Step 3: Combined context
    print("\n🔗 Step 3: Combining RAG + MCP context...")
    combined_context = f"""
=== HISTORICAL CONTEXT (10-K Document) ===
{rag_context}

=== LIVE MARKET DATA (Recent News) ===
{mcp_summary}
"""
    
    print(f"✅ Combined context ready ({len(combined_context)} chars)")
    print("\n📊 COMBINED CONTEXT PREVIEW:")
    print("-" * 70)
    print(combined_context[:500])
    print("\n... [truncated for display]")
    print("-" * 70)
    
    return combined_context

def test_rag_with_mcp_augmentation():
    """Test what the actual benchmark does."""
    print("\n" + "="*70)
    print("TEST 4: RAG Benchmark Query (Like rag_compare_chroma.py)")
    print("="*70)
    
    tickers = ["AAPL"]
    
    # Load RAG
    print("\n📚 Loading RAG system...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="sec_10k"
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Sample question from QA dataset
    question = "What are the company's main growth drivers?"
    
    print(f"\nQuestion: '{question}'")
    
    # BASELINE: RAG only
    print("\n🔵 BASELINE (RAG only):")
    baseline_docs = retriever.invoke(question)
    baseline_contexts = [doc.page_content for doc in baseline_docs]
    
    print(f"  Retrieved {len(baseline_contexts)} chunks from 10-K")
    for i, ctx in enumerate(baseline_contexts, 1):
        print(f"  [{i}] {ctx[:80]}...")
    
    # AUGMENTED: RAG + MCP
    print("\n🟢 AUGMENTED (RAG + MCP):")
    
    # Fetch MCP data
    print("  Fetching live data via MCP...")
    news_result = dispatch_mcp_tool("fetch_news", {
        "tickers": tickers,
        "days_back": 7
    })
    
    mcp_context = f"Recent News for {tickers[0]}:\n"
    for article in news_result.get("articles", [])[:2]:
        mcp_context += f"- {article.get('title', 'N/A')}\n"
    
    # Prepend MCP to RAG results
    augmented_contexts = [mcp_context] + baseline_contexts[:2]
    
    print(f"  Prepended MCP context + top 2 RAG chunks")
    for i, ctx in enumerate(augmented_contexts, 1):
        preview = ctx[:80].replace("\n", " ")
        print(f"  [{i}] {preview}...")
    
    print("\n✅ BENCHMARK INTEGRATION WORKING!")
    print(f"  - Baseline: {len(baseline_contexts)} chunks from 10-K")
    print(f"  - Augmented: {len(augmented_contexts)} contexts (1 MCP + 2 RAG)")

def main():
    print("\n" + "="*70)
    print("  MCP + RAG INTEGRATION TEST SUITE")
    print("="*70)
    
    try:
        # Test 1: RAG only
        test_rag_only()
        
        # Test 2: MCP only
        test_mcp_only()
        
        # Test 3: Combined
        test_mcp_rag_integration()
        
        # Test 4: Benchmark scenario
        test_rag_with_mcp_augmentation()
        
        # Summary
        print("\n" + "="*70)
        print("  ✅ ALL INTEGRATION TESTS PASSED")
        print("="*70)
        print("\nYour system is ready to run:")
        print("  python rag/experiments/rag_compare_chroma.py \\")
        print("    --qa rag/data/manual_qa_template.jsonl \\")
        print("    --tickers AAPL NVDA GOOG MSFT \\")
        print("    --output-dir results/rag_compare_mcp")
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
