"""Response drafting node."""

from src.graph.state import AgentState
from src.core.llm import get_llm
from src.prompts.respond import RESPOND_PROMPT


def draft_response(state: AgentState) -> AgentState:
    """
    Draft an AI-generated response based on KB results and email content.

    Uses LLM to compose a professional, helpful response leveraging knowledge base info.

    Args:
        state: Current agent state with KB results and intent classification

    Returns:
        Updated state with draft_response and confidence score (0.0-1.0)
    """
    try:
        llm = get_llm(temperature=0.7)  # Moderate temp for natural responses

        # Format KB context from FAISS results
        kb_context = ""
        if state.get("kb_results"):
            kb_items = []
            for i, doc in enumerate(state["kb_results"][:3], 1):
                # FAISS results have 'content', 'metadata', and 'similarity'
                content = doc.get("content", str(doc))
                similarity = doc.get("similarity", 0)

                # Format with similarity score
                kb_items.append(
                    f"[Document {i} - Relevance: {similarity:.1%}]\n{content[:200]}..."
                )
            kb_context = "\n\n".join(kb_items)
        else:
            kb_context = "No relevant articles found in knowledge base."

        # Format prompt with context
        prompt = RESPOND_PROMPT.format(
            email_body=state["body"],
            intent=state.get("intent", "general"),
            kb_context=kb_context,
        )

        # Call LLM to draft response
        response = llm.invoke(prompt)
        draft = response.content.strip()

        state["draft_response"] = draft

        # Confidence scoring: higher if KB results found
        if state.get("kb_results"):
            state["confidence"] = 0.85
        else:
            state["confidence"] = 0.60

    except Exception as e:
        print(f"[ERROR] Response drafting failed: {e}")
        state["draft_response"] = (
            "Thank you for contacting us. Our team is reviewing your request and will respond shortly."
        )
        state["confidence"] = 0.3

    return state
