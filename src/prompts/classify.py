"""Email classification prompt template."""

from langchain.prompts import PromptTemplate

CLASSIFY_PROMPT = PromptTemplate(
    input_variables=["email_body"],
    template="""You are a customer support email classifier. Classify the intent of the email into exactly ONE category.

Categories:
- billing: Questions or issues about pricing, payments, invoices, subscriptions
- technical: Technical issues, bugs, errors, feature requests
- general: General questions about products/services
- complaint: Customer complaints, negative feedback
- urgent: Time-sensitive or high-priority issues

Email:
{email_body}

Respond with ONLY the category name (one word, lowercase):""",
)
