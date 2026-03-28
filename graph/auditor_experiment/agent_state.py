# agents/state.py
from typing import TypedDict, List, Optional


#Need to standardize 
class AgentState(TypedDict):
    ticker: str
    context: str           # The RAG-retrieved 10-K chunks
    sentiment_score: float
    draft: str             # The current version of the investment report
    audit_results: dict    # The JSON output from auditor.py
    final_report: str
    loop_count: int        # Tracking K=2 for Experiment 3