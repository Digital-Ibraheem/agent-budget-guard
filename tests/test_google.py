"""Tests for Google Gemini provider and wrapper."""

import sys
from unittest.mock import Mock, patch

import pytest

from agent_budget_guard import BudgetedSession, BudgetExceededError
from agent_budget_guard.cost.pricing import PricingTable
from agent_budget_guard.providers.google_provider import GoogleProvider
from agent_budget_guard.wrappers.google import GoogleClientWrapper


# ---------------------------------------------------------------------------
# GoogleProvider unit tests
# ---------------------------------------------------------------------------

class TestGoogleProvider:
    def setup_method(self):
        self.provider = GoogleProvider()

    def test_estimate_cost_string_contents(self):
        cost = self.provider.estimate_cost(
            messages="Tell me a joke.",
            model="gemini-2.0-flash",
            max_tokens=100,
        )
        assert cost > 0

    def test_estimate_cost_list_of_strings(self):
        cost = self.provider.estimate_cost(
            messages=["Hello", "How are you?"],
            model="gemini-2.0-flash",
            max_tokens=100,
        )
        assert cost > 0

    def test_estimate_cost_openai_style_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "model", "content": "Hi there!"},
        ]
        cost = self.provider.estimate_cost(
            messages=messages,
            model="gemini-2.0-flash",
            max_tokens=100,
        )
        assert cost > 0

    def test_estimate_cost_google_style_parts(self):
        contents = [
            {"role": "user", "parts": [{"text": "What is 2+2?"}]},
        ]
        cost = self.provider.estimate_cost(
            messages=contents,
            model="gemini-2.0-flash",
            max_tokens=50,
        )
        assert cost > 0

    def test_estimate_cost_without_max_tokens_defaults(self):
        cost = self.provider.estimate_cost(
            messages="Short prompt.",
            model="gemini-2.0-flash",
        )
        assert cost > 0

    def test_estimate_cost_pro_more_expensive_than_flash(self):
        flash = self.provider.estimate_cost(
            messages="Hello", model="gemini-2.0-flash", max_tokens=100
        )
        pro = self.provider.estimate_cost(
            messages="Hello", model="gemini-1.5-pro", max_tokens=100
        )
        assert pro > flash

    def test_calculate_cost_uses_usage_metadata(self):
        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50

        cost = self.provider.calculate_cost(mock_response, model="gemini-2.0-flash")

        pricing = PricingTable(provider="google")
        expected = (
            (100 / 1000) * pricing.get_input_price("gemini-2.0-flash")
            + (50 / 1000) * pricing.get_output_price("gemini-2.0-flash")
        )
        assert abs(cost - expected) < 1e-10

    def test_calculate_cost_requires_model(self):
        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5

        with pytest.raises(ValueError, match="model must be provided"):
            self.provider.calculate_cost(mock_response)

    def test_versioned_model_resolves(self):
        """gemini-2.0-flash-001 should resolve to gemini-2.0-flash."""
        cost = self.provider.estimate_cost(
            messages="Hi", model="gemini-2.0-flash-001", max_tokens=50
        )
        assert cost > 0

    def test_pricing_table_provider(self):
        pricing = self.provider.get_pricing_table()
        assert isinstance(pricing, PricingTable)
        price = pricing.get_input_price("gemini-2.0-flash")
        assert price == 0.0001


# ---------------------------------------------------------------------------
# GoogleClientWrapper integration tests (mocked SDK)
# ---------------------------------------------------------------------------

def _make_mock_google_response(prompt_tokens=10, candidates_tokens=20):
    resp = Mock()
    resp.usage_metadata = Mock()
    resp.usage_metadata.prompt_token_count = prompt_tokens
    resp.usage_metadata.candidates_token_count = candidates_tokens
    return resp


def _make_wrapped_client(budget_usd=5.0):
    session = BudgetedSession(budget_usd=budget_usd)
    mock_sdk = Mock()
    mock_models = Mock()
    mock_sdk.models = mock_models
    wrapped = session.wrap_google(mock_sdk)
    return session, wrapped, mock_models


class TestGoogleClientWrapper:
    def test_successful_call_tracked(self):
        session, wrapped, mock_models = _make_wrapped_client(budget_usd=5.0)

        mock_models.generate_content.return_value = _make_mock_google_response(10, 20)

        response = wrapped.models.generate_content(
            model="gemini-2.0-flash",
            contents="Hello!",
        )

        assert response is not None
        assert session.get_total_spent() > 0
        assert session.get_remaining_budget() < 5.0

    def test_budget_exceeded_raises(self):
        session, wrapped, mock_models = _make_wrapped_client(budget_usd=0.000001)

        with pytest.raises(BudgetExceededError):
            wrapped.models.generate_content(
                model="gemini-1.5-pro",
                contents="Write me a novel " * 200,
            )

    def test_budget_exceeded_callback(self):
        captured = []
        session = BudgetedSession(
            budget_usd=0.000001,
            on_budget_exceeded=lambda e: captured.append(e),
        )
        mock_sdk = Mock()
        wrapped = session.wrap_google(mock_sdk)

        result = wrapped.models.generate_content(
            model="gemini-1.5-pro",
            contents="Write me a novel " * 200,
        )

        assert result is None
        assert len(captured) == 1
        assert isinstance(captured[0], BudgetExceededError)

    def test_api_failure_rolls_back(self):
        session, wrapped, mock_models = _make_wrapped_client(budget_usd=5.0)
        mock_models.generate_content.side_effect = RuntimeError("API error")

        with pytest.raises(RuntimeError, match="API error"):
            wrapped.models.generate_content(
                model="gemini-2.0-flash",
                contents="Hello",
            )

        assert session.get_total_spent() == 0.0
        assert session.get_reserved() == 0.0

    def test_max_output_tokens_from_config_dict(self):
        """max_output_tokens extracted from config dict for estimation."""
        session, wrapped, mock_models = _make_wrapped_client(budget_usd=5.0)
        mock_models.generate_content.return_value = _make_mock_google_response(5, 10)

        wrapped.models.generate_content(
            model="gemini-2.0-flash",
            contents="Hi",
            config={"max_output_tokens": 50},
        )

        assert session.get_total_spent() > 0

    def test_max_output_tokens_from_config_object(self):
        """max_output_tokens extracted from a config object attribute."""
        config = Mock()
        config.max_output_tokens = 50

        session, wrapped, mock_models = _make_wrapped_client(budget_usd=5.0)
        mock_models.generate_content.return_value = _make_mock_google_response(5, 10)

        wrapped.models.generate_content(
            model="gemini-2.0-flash",
            contents="Hi",
            config=config,
        )

        assert session.get_total_spent() > 0

    def test_warning_callback_fires(self):
        warnings = []
        session = BudgetedSession(
            budget_usd=0.001,
            on_warning=lambda w: warnings.append(w),
            warning_thresholds=[50],
        )
        mock_sdk = Mock()
        mock_models = Mock()
        mock_sdk.models = mock_models
        wrapped = session.wrap_google(mock_sdk)

        # Use higher token counts so each call burns enough to cross the 50% threshold
        # within the loop.  gemini-2.0-flash: $0.0001/1K input, $0.0004/1K output.
        # 500 prompt + 500 candidates -> ~$0.000250/call; budget=$0.001 -> 50%≈2 calls.
        mock_models.generate_content.return_value = _make_mock_google_response(500, 500)

        for _ in range(50):
            try:
                wrapped.models.generate_content(
                    model="gemini-2.0-flash",
                    contents="Hi",
                )
            except BudgetExceededError:
                break

        threshold_50 = [w for w in warnings if w["threshold"] == 50]
        assert len(threshold_50) == 1

    def test_non_models_attrs_forwarded(self):
        session = BudgetedSession(budget_usd=5.0)
        mock_sdk = Mock()
        mock_sdk.some_attr = "value"
        wrapped = session.wrap_google(mock_sdk)

        assert wrapped.some_attr == "value"

    def test_other_models_methods_forwarded(self):
        """Methods other than generate_content (e.g. count_tokens) pass through."""
        session, wrapped, mock_models = _make_wrapped_client(budget_usd=5.0)
        mock_models.count_tokens.return_value = Mock(total_tokens=42)

        result = wrapped.models.count_tokens(model="gemini-2.0-flash", contents="Hi")
        assert result.total_tokens == 42

    def test_session_attached(self):
        mock_sdk = Mock()
        mock_sdk_cls = Mock(return_value=mock_sdk)
        mock_genai = Mock()
        mock_genai.Client = mock_sdk_cls

        with patch.dict("sys.modules", {"google": Mock(genai=mock_genai), "google.genai": mock_genai}):
            client = BudgetedSession.google(budget_usd=2.0, api_key="test")

        assert client.session is not None
        assert client.session.get_budget() == 2.0

    def test_google_import_error(self):
        """Without google-genai installed, a clear ImportError is raised."""
        with patch.dict("sys.modules", {"google": None}):
            with pytest.raises(ImportError, match="google-genai"):
                BudgetedSession.google(budget_usd=5.0)
