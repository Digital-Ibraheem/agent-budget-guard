"""Test BudgetedSession functionality."""

import pytest
from unittest.mock import Mock, MagicMock
from agentguard import BudgetedSession, BudgetExceededError


def test_session_initialization():
    """Test session initializes correctly."""
    session = BudgetedSession(budget_usd=5.0)

    assert session.get_budget() == 5.0
    assert session.get_total_spent() == 0.0
    assert session.get_remaining_budget() == 5.0


def test_wrap_openai():
    """Test wrapping OpenAI client."""
    session = BudgetedSession(budget_usd=5.0)

    mock_client = Mock()
    wrapped = session.wrap_openai(mock_client)

    assert wrapped is not None
    assert hasattr(wrapped, 'chat')


def test_budget_exceeded_on_estimate():
    """Test that budget exceeded error is raised on estimate."""
    session = BudgetedSession(budget_usd=0.01)  # Very small budget

    mock_client = Mock()

    # Mock the chat.completions structure
    mock_completions = Mock()
    mock_chat = Mock()
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    wrapped = session.wrap_openai(mock_client)

    # Try to make a call that would exceed budget
    with pytest.raises(BudgetExceededError):
        wrapped.chat.completions.create(
            model="gpt-4o",  # Expensive model
            messages=[{"role": "user", "content": "Write a long essay " * 100}],
            max_tokens=1000
        )


def test_successful_call_tracking():
    """Test that successful calls are tracked."""
    session = BudgetedSession(budget_usd=5.0)

    mock_client = Mock()

    # Mock response with usage data
    mock_response = Mock()
    mock_response.model = "gpt-4o-mini"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20

    # Setup mock chain
    mock_completions = Mock()
    mock_completions.create = Mock(return_value=mock_response)
    mock_chat = Mock()
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    wrapped = session.wrap_openai(mock_client)

    # Make call
    response = wrapped.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=20
    )

    # Should have tracked some spend
    assert session.get_total_spent() > 0
    assert session.get_remaining_budget() < 5.0

    # Response should be returned
    assert response == mock_response


def test_get_summary():
    """Test getting budget summary."""
    session = BudgetedSession(budget_usd=10.0)

    summary = session.get_summary()

    assert summary['budget'] == 10.0
    assert summary['spent'] == 0.0
    assert summary['reserved'] == 0.0
    assert summary['remaining'] == 10.0
    assert summary['utilization_percent'] == 0.0


def test_reset():
    """Test resetting session."""
    session = BudgetedSession(budget_usd=10.0)

    # Manually adjust tracker (simulating some spend)
    session._tracker._spent = 3.0

    assert session.get_total_spent() == 3.0

    session.reset()

    assert session.get_total_spent() == 0.0
    assert session.get_remaining_budget() == 10.0


def test_batch_tier():
    """Test creating session with batch tier."""
    session = BudgetedSession(budget_usd=5.0, tier="batch")

    mock_client = Mock()
    wrapped = session.wrap_openai(mock_client)

    assert wrapped is not None
    assert wrapped._tier == "batch"


def test_on_budget_exceeded_callback():
    """Test that callback is called instead of raising."""
    captured = []
    session = BudgetedSession(
        budget_usd=0.01,
        on_budget_exceeded=lambda e: captured.append(e),
    )

    mock_client = Mock()
    mock_completions = Mock()
    mock_chat = Mock()
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    wrapped = session.wrap_openai(mock_client)

    # This would normally raise BudgetExceededError
    result = wrapped.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Write a long essay " * 100}],
        max_tokens=1000
    )

    assert result is None
    assert len(captured) == 1
    assert isinstance(captured[0], BudgetExceededError)


def test_on_warning_callback():
    """Test that warning callbacks fire at thresholds."""
    warnings = []
    session = BudgetedSession(
        budget_usd=0.001,
        on_warning=lambda w: warnings.append(w),
        warning_thresholds=[50],
    )

    mock_client = Mock()
    mock_response = Mock()
    mock_response.model = "gpt-4o-mini"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20

    mock_completions = Mock()
    mock_completions.create = Mock(return_value=mock_response)
    mock_chat = Mock()
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    wrapped = session.wrap_openai(mock_client)

    # Make calls until warning fires
    for _ in range(50):
        try:
            wrapped.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=20
            )
        except BudgetExceededError:
            break

    # Should have fired the 50% threshold exactly once
    threshold_50 = [w for w in warnings if w["threshold"] == 50]
    assert len(threshold_50) == 1
    assert threshold_50[0]["budget"] == 0.001


def test_openai_classmethod():
    """Test the one-liner classmethod."""
    from unittest.mock import patch

    mock_openai_cls = Mock()
    mock_openai_instance = Mock()
    mock_openai_cls.return_value = mock_openai_instance

    with patch("agentguard.session.OpenAI", mock_openai_cls, create=True):
        # Patch the import inside the classmethod
        import agentguard.session as session_mod
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        with patch.dict("sys.modules", {"openai": Mock(OpenAI=mock_openai_cls)}):
            client = BudgetedSession.openai(budget_usd=5.0, api_key="test-key")

    assert client is not None
    assert client.session is not None
    assert client.session.get_budget() == 5.0
