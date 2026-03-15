"""Follow-up scheduling service."""

from typing import Optional
from datetime import datetime, timedelta


class FollowUpService:
    """
    Service for scheduling follow-up emails and maintaining follow-up foundation.

    Tracks escalated issues and schedules reminder follow-ups.
    """

    def __init__(self):
        """Initialize follow-up service."""
        # In-memory storage for demo (TODO: use database)
        self.followups = {}

    def schedule_followup(
        self,
        email_id: str,
        recipient: str,
        reason: str,
        days: int = 1,
    ) -> bool:
        """
        Schedule a follow-up email.

        Args:
            email_id: Original email ID
            recipient: Recipient email address
            reason: Reason for follow-up (e.g., "escalated", "awaiting response")
            days: Days until follow-up (default: 1)

        Returns:
            True if scheduled successfully
        """
        try:
            followup_date = datetime.now() + timedelta(days=days)

            self.followups[email_id] = {
                "recipient": recipient,
                "reason": reason,
                "scheduled_at": datetime.now(),
                "followup_at": followup_date,
                "status": "pending",
            }

            print(
                f"[INFO] Follow-up scheduled for {email_id} on {followup_date.strftime('%Y-%m-%d')}"
            )
            return True

        except Exception as e:
            print(f"[ERROR] Failed to schedule follow-up: {e}")
            return False

    def get_pending_followups(self) -> dict:
        """
        Get all pending follow-ups.

        Returns:
            Dictionary of pending follow-ups
        """
        pending = {
            email_id: details
            for email_id, details in self.followups.items()
            if details["status"] == "pending" and details["followup_at"] <= datetime.now()
        }
        return pending

    def mark_completed(self, email_id: str) -> bool:
        """
        Mark a follow-up as completed.

        Args:
            email_id: Email ID of follow-up

        Returns:
            True if marked successfully
        """
        if email_id in self.followups:
            self.followups[email_id]["status"] = "completed"
            return True
        return False
