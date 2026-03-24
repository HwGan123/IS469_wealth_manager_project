from langgraph.graph import StateGraph, START, END
from graph.state import WealthManagerState
from agents.orchestrator import orchestrator_node
from agents.sentiment import sentiment_node
from agents.analyst import analyst_node
from agents.auditor import auditor_node


def create_wealth_manager_graph():
    workflow = StateGraph(WealthManagerState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    workflow.add_node("orchestrator",    orchestrator_node)
    workflow.add_node("sentiment_agent", sentiment_node)
    workflow.add_node("analyst_agent",   analyst_node)
    workflow.add_node("auditor_agent",   auditor_node)

    # ── Linear flow ────────────────────────────────────────────────────────────
    # orchestrator → sentiment → analyst → auditor
    workflow.add_edge(START,             "orchestrator")
    workflow.add_edge("orchestrator",    "sentiment_agent")
    workflow.add_edge("sentiment_agent", "analyst_agent")
    workflow.add_edge("analyst_agent",   "auditor_agent")

    # ── Self-correction loop ───────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "auditor_agent",
        lambda state: "analyst_agent" if state.get("is_hallucinating") else END,
    )

    return workflow.compile()
