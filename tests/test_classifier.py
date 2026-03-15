"""Tests for email classifier node."""

import pytest
from src.graph.state import AgentState
from src.nodes.classifier import classify_email


@pytest.fixture
def sample_state() -> AgentState:
    """Create a sample agent state for testing."""
    return AgentState(
        email_id="test_001",
        sender="user@example.com",
        subject="Billing question",
        body="How much is the premium plan?",
    )


def test_classify_email(sample_state):
    """Test email classification node."""
    # TODO: Implement test with actual classification
    result = classify_email(sample_state)
    assert result["intent"] is not None
    assert isinstance(result["intent"], str)
