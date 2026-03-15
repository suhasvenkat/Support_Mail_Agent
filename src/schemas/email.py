"""Email data schemas."""

from pydantic import BaseModel, EmailStr
from typing import Optional


class EmailInput(BaseModel):
    """Input email schema."""

    email_id: Optional[str] = None
    sender: EmailStr
    subject: str
    body: str

    class Config:
        json_schema_extra = {
            "example": {
                "email_id": "email_123",
                "sender": "user@example.com",
                "subject": "Payment issue",
                "body": "I was charged twice for my subscription.",
            }
        }


class ClassifiedEmail(BaseModel):
    """Email after classification."""

    email_id: str
    intent: str
    confidence: float = 0.0


class EmailOutput(BaseModel):
    """Output email response schema."""

    email_id: str
    recipient: EmailStr
    subject: str
    body: str
    escalated: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "email_id": "email_123",
                "recipient": "user@example.com",
                "subject": "Re: Payment issue",
                "body": "Thank you for reporting this. Our team is looking into it.",
                "escalated": False,
            }
        }
