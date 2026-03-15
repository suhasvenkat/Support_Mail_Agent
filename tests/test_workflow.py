"""Tests for LangGraph workflow."""

import pytest
from src.graph.state import AgentState
from src.graph.workflow import build_graph


@pytest.fixture
def sample_email() -> AgentState:
    """Create a sample email for workflow testing."""
    return AgentState(
        email_id="test_001",
        sender="user@example.com",
        subject="Test email",
        body="This is a test email.",
    )


def test_build_graph():
    """Test graph construction."""
    # TODO: Implement workflow execution and assertion
    graph = build_graph()
    assert graph is not None


@pytest.mark.asyncio
async def test_workflow_execution(sample_email):
    """Test end-to-end workflow execution."""
    # TODO: Implement async workflow execution test
    # graph = build_graph()
    # result = await graph.ainvoke(sample_email)
    # assert result["final_response"] is not None
    pass
