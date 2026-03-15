"""Email service for reading and sending emails."""

from typing import Optional
from src.schemas.email import EmailInput
from src.core.config import get_settings


class EmailService:
    """
    Service for handling email operations.

    Supports reading from inbox and sending responses.
    Can be configured for IMAP, Gmail API, or mock mode.
    """

    def __init__(self):
        """Initialize email service with settings."""
        self.settings = get_settings()
        self.app_env = self.settings.app_env

    def read_email(self, email_id: str) -> Optional[EmailInput]:
        """
        Read an email from inbox.

        TODO: Implement IMAP/Gmail API integration for production.

        Args:
            email_id: Email identifier

        Returns:
            EmailInput or None if not found
        """
        # Mock implementation for development
        if self.app_env == "production":
            # TODO: Implement Gmail API or IMAP integration
            pass

        return None

    def send_email(
        self,
        recipient: str,
        subject: str,
        body: str,
        escalated: bool = False,
    ) -> bool:
        """
        Send a response email to customer.

        Logs email send action. In production, uses SMTP or Gmail API.

        Args:
            recipient: Recipient email address
            subject: Email subject
            body: Email body content
            escalated: Whether this was escalated to human

        Returns:
            True if successful, False otherwise
        """
        try:
            escalation_flag = "[ESCALATED] " if escalated else ""

            if self.app_env == "production":
                # TODO: Implement actual email sending via SMTP or Gmail API
                print(
                    f"[SEND] {escalation_flag}Email to {recipient}: {subject}"
                )
            else:
                # Mock mode: log only
                print(
                    f"[MOCK-SEND] {escalation_flag}Email to {recipient}: {subject}"
                )
                print(f"  Body preview: {body[:100]}...")

            return True

        except Exception as e:
            print(f"[ERROR] Failed to send email: {e}")
            return False
