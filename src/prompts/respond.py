"""Response drafting prompt template."""

from langchain.prompts import PromptTemplate

RESPOND_PROMPT = PromptTemplate(
    input_variables=["email_body", "intent", "kb_context"],
    template="""You are a professional customer support agent. Draft a helpful, empathetic response to the customer's email.

CUSTOMER EMAIL:
{email_body}

ISSUE TYPE: {intent}

RELEVANT DOCUMENTATION:
{kb_context}

INSTRUCTIONS:
- Write a professional, friendly response
- Address the customer's concern directly
- Use information from the knowledge base if available
- Keep response concise (2-3 sentences)
- Include next steps or reference to docs if helpful
- Be empathetic and acknowledge their issue

RESPONSE:""",
)
