"""Agent state definition for LangGraph workflow."""

from typing import Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    State shared across all nodes in the LangGraph workflow.

    Represents the complete lifecycle of an email from receipt to sending and follow-up.

    Attributes:
        email_id: Unique identifier for the email
        sender: Email sender address
        subject: Email subject
        body: Email body/content
        intent: Classified intent (billing, technical, general, complaint, urgent)
        kb_results: Retrieved knowledge base results (list of relevant docs)
        draft_response: AI-generated draft response
        confidence: Confidence score for the response (0.0-1.0)
        should_escalate: Whether issue needs human escalation
        final_response: Final response (either AI-generated or escalated)
        email_sent: Whether the response was sent successfully
        followup_scheduled: Whether a follow-up was scheduled (for escalated issues)
    """

    email_id: str
    sender: str
    subject: str
    body: str
    intent: Optional[str] = None
    kb_results: Optional[list] = None
    draft_response: Optional[str] = None
    confidence: Optional[float] = None
    should_escalate: Optional[bool] = None
    final_response: Optional[str] = None
    email_sent: Optional[bool] = None
    followup_scheduled: Optional[bool] = None
