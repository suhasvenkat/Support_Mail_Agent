"""LangGraph workflow definition."""

from langgraph.graph import StateGraph, END

from src.graph.state import AgentState
from src.nodes import classifier, kb_retriever, responder, escalator, followup


def build_graph() -> StateGraph:
    """
    Build the LangGraph workflow for email processing.

    Complete workflow:
        1. classifier      → Classify email intent
        2. kb_retriever    → Search knowledge base for relevant docs
        3. responder       → Draft response using KB context
        4. escalator       → Decide escalation and send email
        5. followup        → Schedule follow-ups if escalated

    Edges:
        input → classifier → kb_retriever → responder → escalator → followup → output
    """
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("classifier", classifier.classify_email)
    workflow.add_node("kb_retriever", kb_retriever.retrieve_knowledge)
    workflow.add_node("responder", responder.draft_response)
    workflow.add_node("escalator", escalator.escalate_or_send)
    workflow.add_node("followup", followup.schedule_followup)

    # Set entry point
    workflow.set_entry_point("classifier")

    # Add edges (sequential flow)
    workflow.add_edge("classifier", "kb_retriever")
    workflow.add_edge("kb_retriever", "responder")
    workflow.add_edge("responder", "escalator")
    workflow.add_edge("escalator", "followup")
    workflow.add_edge("followup", END)

    return workflow.compile()
