"""Email intent classification node."""

from src.graph.state import AgentState
from src.core.llm import get_llm
from src.prompts.classify import CLASSIFY_PROMPT


def classify_email(state: AgentState) -> AgentState:
    """
    Classify the intent of an incoming email.

    Uses LLM to categorize email into: billing, technical, general, complaint, or urgent.

    Args:
        state: Current agent state with email subject and body

    Returns:
        Updated state with intent classification
    """
    try:
        llm = get_llm(temperature=0.0)  # Low temp for deterministic classification

        # Format prompt with email body
        prompt = CLASSIFY_PROMPT.format(email_body=f"Subject: {state['subject']}\n\n{state['body']}")

        # Call LLM
        response = llm.invoke(prompt)
        intent = response.content.strip().lower()

        # Validate intent is one of expected categories
        valid_intents = ["billing", "technical", "general", "complaint", "urgent"]
        if intent not in valid_intents:
            intent = "general"  # Default fallback

        state["intent"] = intent
    except Exception as e:
        print(f"[ERROR] Classification failed: {e}")
        state["intent"] = "general"

    return state
