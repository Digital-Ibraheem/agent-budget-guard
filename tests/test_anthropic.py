"""Tests for Anthropic provider and wrapper."""

import sys
from unittest.mock import Mock, patch

import pytest

from agent_budget_guard import BudgetedSession, BudgetExceededError
from agent_budget_guard.cost.pricing import PricingTable
from agent_budget_guard.providers.anthropic_provider import AnthropicProvider
from agent_budget_guard.wrappers.anthropic import AnthropicClientWrapper


# ---------------------------------------------------------------------------
# AnthropicProvider unit tests
# ---------------------------------------------------------------------------

class TestAnthropicProvider:
    def setup_method(self):
        self.provider = AnthropicProvider()

    def test_estimate_cost_with_max_tokens(self):
        messages = [{"role": "user", "content": "Hello there"}]
        cost = self.provider.estimate_cost(
            messages=messages,
            model="claude-haiku-4-5",
            max_tokens=100,
        )
        assert cost > 0

    def test_estimate_cost_without_max_tokens(self):
        messages = [{"role": "user", "content": "Hello there"}]
        cost = self.provider.estimate_cost(
            messages=messages,
            model="claude-haiku-4-5",
        )
        # Should default to at least 1024 output tokens
        assert cost > 0

    def test_estimate_cost_larger_model_is_more_expensive(self):
        messages = [{"role": "user", "content": "Hello"}]
        haiku_cost = self.provider.estimate_cost(
            messages=messages, model="claude-haiku-4-5", max_tokens=100
        )
        opus_cost = self.provider.estimate_cost(
            messages=messages, model="claude-opus-4-6", max_tokens=100
        )
        assert opus_cost > haiku_cost

    def test_estimate_cost_content_blocks(self):
        """Content can be a list of block dicts (tool use, etc.)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is 2+2?"},
                ],
            }
        ]
        cost = self.provider.estimate_cost(
            messages=messages, model="claude-haiku-4-5", max_tokens=50
        )
        assert cost > 0

    def test_calculate_cost_uses_response_usage(self):
        mock_response = Mock()
        mock_response.model = "claude-haiku-4-5"
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        cost = self.provider.calculate_cost(mock_response)

        pricing = PricingTable(provider="anthropic")
        expected = (
            (100 / 1000) * pricing.get_input_price("claude-haiku-4-5")
            + (50 / 1000) * pricing.get_output_price("claude-haiku-4-5")
        )
        assert abs(cost - expected) < 1e-10

    def test_calculate_cost_model_override(self):
        """model kwarg takes precedence over response.model."""
        mock_response = Mock()
        mock_response.model = "claude-haiku-4-5"
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        cost_haiku = self.provider.calculate_cost(mock_response, model="claude-haiku-4-5")
        cost_opus = self.provider.calculate_cost(mock_response, model="claude-opus-4-6")

        assert cost_opus > cost_haiku

    def test_pricing_table_provider(self):
        pricing = self.provider.get_pricing_table()
        assert isinstance(pricing, PricingTable)
        # Spot-check a known Claude model
        price = pricing.get_input_price("claude-haiku-4-5")
        assert price == 0.0008


# ---------------------------------------------------------------------------
# AnthropicClientWrapper integration tests (mocked SDK)
# ---------------------------------------------------------------------------

def _make_mock_anthropic_response(model="claude-haiku-4-5", input_tokens=10, output_tokens=20):
    resp = Mock()
    resp.model = model
    resp.usage = Mock()
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    return resp


def _make_wrapped_client(budget_usd=5.0):
    """Return (session, wrapped_client) with a mocked underlying SDK client."""
    session = BudgetedSession(budget_usd=budget_usd)

    mock_sdk_client = Mock()
    mock_messages = Mock()
    mock_sdk_client.messages = mock_messages

    wrapped = session.wrap_anthropic(mock_sdk_client)
    return session, wrapped, mock_messages


class TestAnthropicClientWrapper:
    def test_successful_call_tracked(self):
        session, wrapped, mock_messages = _make_wrapped_client(budget_usd=5.0)

        mock_response = _make_mock_anthropic_response(input_tokens=10, output_tokens=20)
        mock_messages.create.return_value = mock_response

        response = wrapped.messages.create(
            model="claude-haiku-4-5",
            max_tokens=20,
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert response is mock_response
        assert session.get_total_spent() > 0
        assert session.get_remaining_budget() < 5.0

    def test_budget_exceeded_raises(self):
        session, wrapped, mock_messages = _make_wrapped_client(budget_usd=0.000001)

        with pytest.raises(BudgetExceededError):
            wrapped.messages.create(
                model="claude-opus-4-6",
                max_tokens=1000,
                messages=[{"role": "user", "content": "Write me a book"}],
            )

    def test_budget_exceeded_callback(self):
        captured = []
        session = BudgetedSession(
            budget_usd=0.000001,
            on_budget_exceeded=lambda e: captured.append(e),
        )
        mock_sdk = Mock()
        wrapped = session.wrap_anthropic(mock_sdk)

        result = wrapped.messages.create(
            model="claude-opus-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": "Write me a book"}],
        )

        assert result is None
        assert len(captured) == 1
        assert isinstance(captured[0], BudgetExceededError)

    def test_api_failure_rolls_back(self):
        session, wrapped, mock_messages = _make_wrapped_client(budget_usd=5.0)
        mock_messages.create.side_effect = RuntimeError("network error")

        with pytest.raises(RuntimeError, match="network error"):
            wrapped.messages.create(
                model="claude-haiku-4-5",
                max_tokens=20,
                messages=[{"role": "user", "content": "Hi"}],
            )

        # No spend should have been committed
        assert session.get_total_spent() == 0.0
        assert session.get_reserved() == 0.0

    def test_warning_callback_fires(self):
        warnings = []
        session = BudgetedSession(
            budget_usd=0.001,
            on_warning=lambda w: warnings.append(w),
            warning_thresholds=[50],
        )
        mock_sdk = Mock()
        mock_messages = Mock()
        mock_sdk.messages = mock_messages
        wrapped = session.wrap_anthropic(mock_sdk)

        mock_messages.create.return_value = _make_mock_anthropic_response(
            input_tokens=10, output_tokens=20
        )

        for _ in range(50):
            try:
                wrapped.messages.create(
                    model="claude-haiku-4-5",
                    max_tokens=20,
                    messages=[{"role": "user", "content": "Hi"}],
                )
            except BudgetExceededError:
                break

        threshold_50 = [w for w in warnings if w["threshold"] == 50]
        assert len(threshold_50) == 1

    def test_non_messages_attrs_forwarded(self):
        """Attributes other than .messages should pass through to the SDK client."""
        session = BudgetedSession(budget_usd=5.0)
        mock_sdk = Mock()
        mock_sdk.some_attr = "hello"
        wrapped = session.wrap_anthropic(mock_sdk)

        assert wrapped.some_attr == "hello"

    def test_session_attached(self):
        mock_sdk = Mock()
        mock_sdk_cls = Mock(return_value=mock_sdk)

        with patch.dict("sys.modules", {"anthropic": Mock(Anthropic=mock_sdk_cls)}):
            client = BudgetedSession.anthropic(budget_usd=3.0, api_key="test")

        assert client.session is not None
        assert client.session.get_budget() == 3.0

    def test_anthropic_import_error(self):
        """Without anthropic installed, a clear ImportError is raised."""
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic"):
                BudgetedSession.anthropic(budget_usd=5.0)
