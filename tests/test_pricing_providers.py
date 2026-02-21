"""Tests for PricingTable multi-provider support."""

import pytest
from agent_budget_guard.cost.pricing import PricingTable
from agent_budget_guard.exceptions import PricingDataError


class TestPricingTableProviders:
    def test_openai_prices(self):
        pricing = PricingTable(provider="openai")
        assert pricing.get_input_price("gpt-4o-mini") == 0.00015
        assert pricing.get_output_price("gpt-4o-mini") == 0.0006
        # Default (no provider arg) also loads OpenAI
        assert PricingTable().get_input_price("gpt-4o-mini") == 0.00015

    def test_anthropic_prices(self):
        pricing = PricingTable(provider="anthropic")
        assert pricing.get_input_price("claude-haiku-4-5") == 0.0008
        assert pricing.get_output_price("claude-haiku-4-5") == 0.004

    def test_google_prices(self):
        pricing = PricingTable(provider="google")
        assert pricing.get_input_price("gemini-2.0-flash") == 0.0001
        assert pricing.get_output_price("gemini-2.0-flash") == 0.0004

    def test_anthropic_opus_more_expensive_than_haiku(self):
        pricing = PricingTable(provider="anthropic")
        haiku = pricing.get_input_price("claude-haiku-4-5")
        opus = pricing.get_input_price("claude-opus-4-6")
        assert opus > haiku

    def test_anthropic_model_alias(self):
        pricing = PricingTable(provider="anthropic")
        assert pricing.get_input_price("claude-sonnet-latest") == pricing.get_input_price("claude-sonnet-4-6")

    def test_anthropic_versioned_model_resolves(self):
        pricing = PricingTable(provider="anthropic")
        assert pricing.get_input_price("claude-3-5-sonnet-20241022") == pricing.get_input_price("claude-3-5-sonnet")

    def test_google_pro_more_expensive_than_flash(self):
        pricing = PricingTable(provider="google")
        assert pricing.get_input_price("gemini-1.5-pro") > pricing.get_input_price("gemini-2.0-flash")

    def test_google_versioned_model_resolves(self):
        pricing = PricingTable(provider="google")
        assert pricing.get_input_price("gemini-2.0-flash-001") == pricing.get_input_price("gemini-2.0-flash")

    def test_unknown_provider_raises(self):
        with pytest.raises(PricingDataError, match="Unknown provider"):
            PricingTable(provider="unknown_llm")

    def test_provider_tables_are_isolated(self):
        """Models from one provider should not appear in another provider's table."""
        with pytest.raises(PricingDataError):
            PricingTable(provider="anthropic").get_input_price("gpt-4o-mini")
        with pytest.raises(PricingDataError):
            PricingTable(provider="google").get_input_price("claude-haiku-4-5")

    def test_custom_config_path_still_works(self, tmp_path):
        """config_path kwarg overrides provider selection."""
        import json
        custom = {
            "models": {
                "my-model": {
                    "standard": {
                        "input_price_per_1k": 0.1,
                        "output_price_per_1k": 0.2,
                    }
                }
            },
            "model_aliases": {},
        }
        config_file = tmp_path / "custom_pricing.json"
        config_file.write_text(json.dumps(custom))

        pricing = PricingTable(config_path=str(config_file))
        assert pricing.get_input_price("my-model") == 0.1
        assert pricing.get_output_price("my-model") == 0.2
