from langgraph.graph import StateGraph, END
from agents.investment import investment_node
from agents.auditor import auditor_node
from agents.reporter import reporter_node
from graph.auditor_experiment.agent_state import AgentState

def build_oneshot():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("investor", investment_node)
    workflow.add_node("auditor", auditor_node)
    workflow.add_node("reporter", reporter_node)
    
    workflow.set_entry_point("investor")
    workflow.add_edge("investor", "auditor")
    
    # Logic: Even if it fails, it goes to reporter, 
    # but the reporter will see the "REJECTED" status in state.
    workflow.add_edge("auditor", "reporter")
    workflow.add_edge("reporter", END)
    
    return workflow.compile()