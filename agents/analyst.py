import os

from graph.state import WealthManagerState

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
except Exception:
    Chroma = None
    HuggingFaceEmbeddings = None
    ChatOpenAI = None
    ChatPromptTemplate = None


def _build_context_from_retriever(tickers: list[str]) -> str:
    if not (Chroma and HuggingFaceEmbeddings):
        return ""

    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        hf_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=hf_embeddings,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        query = (
            "Financial risks, growth drivers, and 10-K analysis for "
            + (", ".join(tickers) if tickers else "the selected portfolio")
        )
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception:
        return ""


def _draft_with_llm(sentiment_score: float, tickers: list[str], context: str) -> str:
    if not (ChatOpenAI and ChatPromptTemplate) or not os.getenv("OPENAI_API_KEY"):
        return ""

    try:
        analyst_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
        prompt = ChatPromptTemplate.from_template(
            """
You are a Senior Investment Analyst. Write a concise professional markdown report.

Inputs:
- Sentiment Score: {sentiment}
- Tickers: {tickers}
- Context: {context}

Requirements:
1. Provide an executive summary.
2. Highlight key risks and growth drivers.
3. Tie recommendations directly to sentiment and context.
"""
        )
        chain = prompt | analyst_llm
        response = chain.invoke(
            {
                "sentiment": sentiment_score,
                "tickers": ", ".join(tickers) if tickers else "N/A",
                "context": context or "No external context retrieved.",
            }
        )
        return str(response.content)
    except Exception:
        return ""


def _fallback_report(sentiment_score: float, tickers: list[str], context: str) -> str:
    sentiment_bucket = (
        "bullish" if sentiment_score > 0.15 else "bearish" if sentiment_score < -0.15 else "neutral"
    )
    lines = [
        "# Investment Analyst Report",
        "",
        "## Executive Summary",
        f"Coverage tickers: {', '.join(tickers) if tickers else 'N/A'}.",
        f"Current aggregated sentiment signal is {sentiment_bucket} ({sentiment_score:+.2f}).",
        "",
        "## Risk and Opportunity View",
        "- Risks: monitor earnings quality, guidance changes, and macro sensitivity.",
        "- Opportunities: focus on firms with resilient cash flow and defensible margins.",
        "",
        "## Suggested Positioning",
        "- Keep sizing aligned with risk budget and diversification constraints.",
        "- Reassess exposures if sentiment weakens materially or guidance deteriorates.",
    ]
    if context:
        lines.extend(["", "## Retrieved Context", context[:2000]])
    return "\n".join(lines)


def analyst_node(state: WealthManagerState) -> dict:
    print("--- AGENT: INVESTMENT ANALYST ---")

    tickers = state.get("tickers") or []
    sentiment_score = float(state.get("sentiment_score", 0.0) or 0.0)

    context = _build_context_from_retriever(tickers)
    draft_report = _draft_with_llm(sentiment_score, tickers, context)
    if not draft_report:
        draft_report = _fallback_report(sentiment_score, tickers, context)

    return {
        "retrieved_context": context,
        "draft_report": draft_report,
        "messages": ["Investment analyst produced a draft report."],
    }