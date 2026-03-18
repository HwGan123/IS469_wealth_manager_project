from langgraph.graph import StateGraph, START, END
from graph.state import WealthManagerState
from agents.sentiment import sentiment_node
from agents.portfolio import portfolio_node
from agents.analyst import analyst_node
from agents.auditor import auditor_node

def create_wealth_manager_graph():
    # Initialize the StateGraph with our custom State
    workflow = StateGraph(WealthManagerState)

    # 1. Add the 4 AI Agent Nodes
    workflow.add_node("sentiment_agent", sentiment_node)
    workflow.add_node("portfolio_agent", portfolio_node)
    workflow.add_node("analyst_agent", analyst_node)
    workflow.add_node("auditor_agent", auditor_node)

    # 2. Define the linear flow
    workflow.add_edge(START, "sentiment_agent")
    workflow.add_edge("sentiment_agent", "portfolio_agent")
    workflow.add_edge("portfolio_agent", "analyst_agent")
    workflow.add_edge("analyst_agent", "auditor_agent")

    # 3. Add Conditional Logic (The Self-Correction Loop)
    # This addresses the risk of hallucinations by implementing a guardrail [cite: 21]
    workflow.add_conditional_edges(
        "auditor_agent",
        lambda state: "analyst_agent" if state["is_hallucinating"] else END
    )

    return workflow.compile()