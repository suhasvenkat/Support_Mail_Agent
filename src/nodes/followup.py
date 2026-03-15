"""Follow-up scheduling node."""

from src.graph.state import AgentState
from src.services.followup_service import FollowUpService


def schedule_followup(state: AgentState) -> AgentState:
    """
    Schedule follow-up emails for escalated or complex issues.

    For escalated emails, schedule a follow-up check-in with customer.

    Args:
        state: Current agent state with escalation status

    Returns:
        Updated state with followup_scheduled flag
    """
    try:
        if state.get("should_escalate"):
            followup_service = FollowUpService()

            # Schedule follow-up for escalated issues (1 day)
            success = followup_service.schedule_followup(
                email_id=state["email_id"],
                recipient=state["sender"],
                reason="escalated",
                days=1,
            )

            state["followup_scheduled"] = success
        else:
            state["followup_scheduled"] = False

    except Exception as e:
        print(f"[ERROR] Follow-up scheduling failed: {e}")
        state["followup_scheduled"] = False

    return state
