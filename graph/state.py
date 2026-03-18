from typing import TypedDict, Annotated, List, Union, Dict
import operator

class WealthManagerState(TypedDict):
    # 'messages' stores the full conversation history for context retention
    messages: Annotated[list, operator.add]
    
    # Domain-specific state variables
    sentiment_score: float         # From FinBERT Sentiment Agent
    portfolio_weights: Dict        # From Portfolio Optimization Agent
    retrieved_context: str         # Chunks from the Real RAG Pipeline
    draft_report: str              # From Investment Analyst Agent
    audit_score: float             # Quantitative score from Auditor Agent
    is_hallucinating: bool         # Boolean flag for the self-correction loop