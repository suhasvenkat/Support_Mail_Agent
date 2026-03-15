"""Email processing routes."""

from datetime import datetime
from fastapi import APIRouter, HTTPException
from src.schemas.email import EmailInput, EmailOutput
from src.graph.workflow import build_graph
from src.utils.id_generator import generate_id

router = APIRouter()

# In-memory storage for demo (TODO: use database)
# Store complete workflow results, not just EmailOutput
processed_emails = {}


@router.post("/process", response_model=EmailOutput)
async def process_email(email: EmailInput) -> EmailOutput:
    """
    Process an incoming support email through the complete LangGraph workflow.

    Workflow:
        1. Classify intent (billing, technical, general, complaint, urgent)
        2. Search knowledge base for relevant documentation
        3. Draft response using LLM with KB context
        4. Escalate to human if confidence is low or issue is complex
        5. Send final reply to customer
        6. Schedule follow-ups for escalated issues

    Args:
        email: Input email data (sender, subject, body)

    Returns:
        EmailOutput with AI-generated response or escalation message
    """
    try:
        # Generate email ID if not provided
        if not email.email_id:
            email.email_id = generate_id()

        # Build and execute the LangGraph workflow
        graph = build_graph()

        # Convert to state dict for graph execution
        initial_state = {
            "email_id": email.email_id,
            "sender": email.sender,
            "subject": email.subject,
            "body": email.body,
        }

        # Execute workflow (synchronous invocation)
        result = graph.invoke(initial_state)

        # Build response from workflow result
        response = EmailOutput(
            email_id=result.get("email_id"),
            recipient=result.get("sender"),
            subject=f"Re: {result.get('subject')}",
            body=result.get("final_response", ""),
            escalated=result.get("should_escalate", False),
        )

        # Store complete workflow result with metadata for detailed view
        processed_emails[response.email_id] = {
            "email": response.model_dump(),
            "workflow": {
                "intent": result.get("intent"),
                "confidence": result.get("confidence", 0),
                "kb_results": result.get("kb_results", []),
                "should_escalate": result.get("should_escalate", False),
                "followup_scheduled": result.get("followup_scheduled", False),
            },
            "input": {
                "sender": email.sender,
                "subject": email.subject,
                "body": email.body,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        return response

    except Exception as e:
        print(f"[ERROR] Email processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/")
async def list_emails():
    """
    List all processed emails.

    Returns:
        Dictionary of email_id -> Email details
    """
    return {
        "total": len(processed_emails),
        "emails": processed_emails,
    }


@router.get("/details/{email_id}")
async def get_email_details(email_id: str):
    """
    Get detailed processing results for an email.

    Args:
        email_id: Email identifier

    Returns:
        Complete workflow execution details

    Raises:
        HTTPException: If email not found
    """
    if email_id not in processed_emails:
        raise HTTPException(status_code=404, detail=f"Email {email_id} not found")

    return processed_emails[email_id]


@router.get("/{email_id}", response_model=EmailOutput)
async def get_email(email_id: str):
    """
    Get a specific processed email response.

    Args:
        email_id: Email identifier

    Returns:
        EmailOutput response with agent's reply

    Raises:
        HTTPException: If email not found
    """
    if email_id not in processed_emails:
        raise HTTPException(status_code=404, detail=f"Email {email_id} not found")

    return EmailOutput(**processed_emails[email_id]["email"])
