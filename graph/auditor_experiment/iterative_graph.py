from langgraph.graph import StateGraph, END
from agents.investment import investment_node
from agents.auditor import auditor_node
from agents.reporter import reporter_node
from graph.auditor_experiment.agent_state import AgentState

def should_continue(state):
    # This is the "Router" function
    if state["audit_results"].get("status") == "APPROVED":
        return "approved"
    elif state.get("loop_count", 0) >= 2:
        return "max_loops_reached"
    else:
        return "needs_revision"

def build_iterative():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("investor", investment_node)
    workflow.add_node("auditor", auditor_node)
    workflow.add_node("reporter", reporter_node)
    
    workflow.set_entry_point("investor")
    workflow.add_edge("investor", "auditor")
    
    # The Conditional Loop
    workflow.add_conditional_edges(
        "auditor",
        should_continue,
        {
            "approved": "reporter",
            "needs_revision": "investor", # Loop back to rewrite
            "max_loops_reached": "reporter"
        }
    )
    
    workflow.add_edge("reporter", END)
    
    return workflow.compile()