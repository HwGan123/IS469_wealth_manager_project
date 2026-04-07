from langgraph.graph import StateGraph, START, END
from graph.state import WealthManagerState
from agents.orchestrator import orchestrator_node
from agents.market_context import market_context_node
from agents.sentiment import sentiment_node
from agents.analyst import analyst_node
from agents.auditor import auditor_node
from agents.report_generator import report_generator_node


def create_wealth_manager_graph():
    workflow = StateGraph(WealthManagerState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    workflow.add_node("orchestrator_agent", orchestrator_node)
    workflow.add_node("market_context_agent", market_context_node)
    workflow.add_node("sentiment_agent", sentiment_node)
    workflow.add_node("investment_analyst_agent", analyst_node)
    workflow.add_node("auditor_agent", auditor_node)
    workflow.add_node("report_generator_agent", report_generator_node)

    # ── Linear flow ────────────────────────────────────────────────────────────
    # orchestrator routes to market_context to fetch market data
    workflow.add_edge(START, "orchestrator_agent")
    workflow.add_edge("orchestrator_agent", "market_context_agent")
    
    # market_context conditionally routes to sentiment or directly to report generator
    workflow.add_conditional_edges(
        "market_context_agent",
        lambda state: state.get("route_target", "sentiment_agent"),
    )

    # Main analysis flow
    workflow.add_edge("sentiment_agent", "investment_analyst_agent")
    workflow.add_edge("investment_analyst_agent", "auditor_agent")

    # ── Self-correction loop ───────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "auditor_agent",
        lambda state: "investment_analyst_agent" if state.get("is_hallucinating") else "report_generator_agent",
    )

    workflow.add_edge("report_generator_agent", END)

    return workflow.compile()
