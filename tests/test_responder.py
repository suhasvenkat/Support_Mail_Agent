"""Tests for response drafting node."""

import pytest
from src.graph.state import AgentState
from src.nodes.responder import draft_response


@pytest.fixture
def sample_state() -> AgentState:
    """Create a sample agent state for testing."""
    return AgentState(
        email_id="test_001",
        sender="user@example.com",
        subject="Technical issue",
        body="App crashes on startup",
        intent="technical",
        kb_results=[],
    )


def test_draft_response(sample_state):
    """Test response drafting node."""
    # TODO: Implement test with actual response generation
    result = draft_response(sample_state)
    assert result["draft_response"] is not None
    assert isinstance(result["confidence"], float)
