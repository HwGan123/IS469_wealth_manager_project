import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from graph.state import WealthManagerState

# 1. Setup FREE Local Embeddings
# 'all-MiniLM-L6-v2' is fast, accurate, and only ~100MB.
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'} # Change to 'cuda' if you have a GPU
encode_kwargs = {'normalize_embeddings': False}

hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 2. Connect to the Vector Store
DB_PATH = "./chroma_db"
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=hf_embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Setup the Analyst LLM (GPT-4o still requires a key, but costs are minimal)
analyst_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

def analyst_node(state: WealthManagerState):
    print("--- ✍️ AGENT: INVESTMENT ANALYST (ADVANCED RAG) ---")

    # A. Query Expansion: Instead of the raw user prompt, we create a search query
    # based on the portfolio weights and sentiment.
    tickers = ", ".join(state["portfolio_weights"].keys())
    search_query = f"Financial risks, growth drivers, and 10-K analysis for {tickers} in 2024"

    # B. Retrieval Step (Using Free Local Embeddings)
    docs = retriever.invoke(search_query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # C. The Analyst Prompt
    prompt = ChatPromptTemplate.from_template("""
    You are a Senior Investment Analyst. Synthesize a professional report.
    
    DATA INPUTS:
    - Sentiment Score: {sentiment}
    - Portfolio Weights: {weights}
    - 10-K Research Context: {context}
    
    INSTRUCTIONS:
    1. Relate the 10-K risks directly to the portfolio weights. 
    2. If sentiment is bearish, highlight the 'Risk Factors' from the context.
    3. If sentiment is bullish, highlight 'Management Discussion' growth drivers.
    
    Output a structured markdown report.
    """)

    # D. Generation
    chain = prompt | analyst_llm
    response = chain.invoke({
        "sentiment": state["sentiment_score"],
        "weights": state["portfolio_weights"],
        "context": context
    })

    # E. Update State
    return {
        "draft_report": response.content,
        "retrieved_context": context, # Crucial for the Auditor to check later!
        "messages": [response.content]
    }