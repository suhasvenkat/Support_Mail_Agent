"""Mock LLM service - Realistic responses without API calls"""

from typing import Optional


class MockLLM:
    """
    Generates realistic mock responses based on email content and intent.
    Perfect for testing without OpenAI API calls.
    """

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature

    def _classify_intent(self, text: str) -> str:
        """Classify email intent based on keywords."""
        text_lower = text.lower()

        # Urgent/Complaint keywords (check first - highest priority)
        urgent_keywords = ["urgent", "asap", "immediately", "now", "emergency", "critical"]
        complaint_keywords = ["terrible", "hate", "angry", "frustrated", "worst", "never", "unacceptable", "outraged"]

        if any(kw in text_lower for kw in urgent_keywords):
            return "urgent"
        if any(kw in text_lower for kw in complaint_keywords):
            return "complaint"

        # Billing keywords
        billing_keywords = ["charged", "charge", "payment", "refund", "bill", "invoice", "subscription", "price", "cost", "money", "card"]
        if any(kw in text_lower for kw in billing_keywords):
            return "billing"

        # Technical keywords
        technical_keywords = ["crash", "error", "bug", "broken", "not working", "doesn't work", "issue", "problem", "fail", "feature", "request"]
        if any(kw in text_lower for kw in technical_keywords):
            return "technical"

        # Password/Account keywords
        account_keywords = ["password", "reset", "forgot", "access", "account", "login", "sign in"]
        if any(kw in text_lower for kw in account_keywords):
            return "general"

        # Default
        return "general"

    def invoke(self, prompt_or_messages) -> object:
        """Mock LLM that returns realistic responses."""
        # Handle both string prompts and message lists
        if isinstance(prompt_or_messages, str):
            user_message = prompt_or_messages
        elif isinstance(prompt_or_messages, list) and len(prompt_or_messages) > 0:
            user_message = prompt_or_messages[-1].get("content", "") if isinstance(prompt_or_messages[-1], dict) else str(prompt_or_messages[-1])
        else:
            user_message = ""

        # Generate response based on content
        response_text = self._generate_response(user_message)

        # Return in same format as real LLM
        return MockMessage(content=response_text)

    def _generate_response(self, prompt: str) -> str:
        """Generate realistic mock response based on prompt."""
        prompt_lower = prompt.lower()

        # Check if this is a classification task (looking for specific keywords in prompt)
        if "classify" in prompt_lower and "categories:" in prompt_lower:
            # Extract ONLY the email content, not the category definitions
            # Email content comes after "Email:\n" marker
            if "email:\n" in prompt_lower:
                email_section = prompt_lower.split("email:\n", 1)[1]
                # Remove the final instruction line
                email_content = email_section.split("respond with only")[0].strip()
            else:
                email_content = prompt_lower
            return self._classify_intent(email_content)

        # Billing-related
        if "charged" in prompt_lower or "payment" in prompt_lower or "refund" in prompt_lower or "billing" in prompt_lower:
            return (
                "Thank you for reaching out regarding your billing concern. "
                "Double charges can occasionally occur due to network delays or subscription updates. "
                "Our team reviews all charges within 24 hours. "
                "Please reply with your transaction ID and account information, "
                "and we'll investigate immediately. "
                "Refunds are typically processed within 3-5 business days. "
                "We appreciate your patience."
            )

        # Technical issues
        elif "crash" in prompt_lower or "error" in prompt_lower or "bug" in prompt_lower:
            return (
                "We're sorry you're experiencing technical issues. "
                "Here are some troubleshooting steps that often help:\n\n"
                "1. Clear your app cache (Settings > Apps > [App Name] > Storage > Clear Cache)\n"
                "2. Restart your device\n"
                "3. Check if there's an app update available in your app store\n"
                "4. Try reinstalling the app if the issue persists\n\n"
                "If these steps don't resolve the issue, please reply with:\n"
                "- Your device type and OS version\n"
                "- When the problem started\n"
                "- Steps to reproduce the issue\n\n"
                "Our technical team will prioritize your case."
            )

        # Password/Account access
        elif "password" in prompt_lower or "reset" in prompt_lower or "access" in prompt_lower:
            return (
                "We can help you reset your password. "
                "Please click the 'Forgot Password' link on the login page. "
                "You'll receive an email with a reset link within a few minutes. "
                "The link expires after 24 hours for security. "
                "If you don't see the email, check your spam folder. "
                "If you're still having trouble, reply with your email address "
                "and we'll send you a new reset link manually."
            )

        # Complaints/Urgent
        elif any(word in prompt_lower for word in ["urgent", "terrible", "hate", "angry", "worst"]):
            return (
                "We understand your frustration and sincerely apologize. "
                "Your feedback is important to us. "
                "Due to the complexity of your concerns, "
                "we're escalating this to our senior support team. "
                "Someone will contact you within 24 hours to resolve this. "
                "We value your business and want to make this right."
            )

        # General inquiries
        else:
            return (
                "Thank you for contacting us. "
                "We've received your inquiry and will review it shortly. "
                "If you have any additional details to share, please reply to this message. "
                "We're here to help and will get back to you as soon as possible."
            )


class MockMessage:
    """Mock message object that mimics LangChain message format."""

    def __init__(self, content: str):
        self.content = content

    def __str__(self):
        return self.content
