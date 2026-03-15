"""Escalation decision and final response node."""

from src.graph.state import AgentState
from src.services.email_service import EmailService


def escalate_or_send(state: AgentState) -> AgentState:
    """
    Decide whether to escalate to human or send AI response.

    Escalation triggers:
    - Confidence < 0.6
    - Intent is "urgent" or "complaint"
    - No KB results found for complex issues

    Args:
        state: Current agent state with draft response and confidence

    Returns:
        Updated state with should_escalate and final_response
    """
    try:
        # Escalation decision logic
        confidence = state.get("confidence", 0.0)
        intent = state.get("intent", "general")
        kb_results = state.get("kb_results", [])

        # Escalate if: low confidence OR urgent/complaint intent
        should_escalate = (
            confidence < 0.6
            or intent in ["urgent", "complaint"]
            or (intent == "technical" and not kb_results)
        )

        state["should_escalate"] = should_escalate

        # Determine final response
        if should_escalate:
            state["final_response"] = (
                "Thank you for your email. Due to the complexity of your request, "
                "our team will review it and get back to you within 24 hours."
            )
        else:
            state["final_response"] = state.get("draft_response", "")

        # Send email (can be mocked or real based on configuration)
        email_service = EmailService()
        success = email_service.send_email(
            recipient=state["sender"],
            subject=f"Re: {state['subject']}",
            body=state["final_response"],
            escalated=should_escalate,
        )

        if not success:
            print(f"[WARN] Failed to send email to {state['sender']}")

    except Exception as e:
        print(f"[ERROR] Escalation/send failed: {e}")
        state["should_escalate"] = True
        state["final_response"] = (
            "We encountered an error processing your request. "
            "Our support team will reach out to you shortly."
        )

    return state
