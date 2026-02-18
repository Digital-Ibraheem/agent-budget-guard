"""Tests for PricingTable multi-provider support."""

import pytest
from agent_budget_guard.cost.pricing import PricingTable
from agent_budget_guard.exceptions import PricingDataError


class TestPricingTableProviders:
    def test_openai_loads_by_default(self):
        pricing = PricingTable()
        price = pricing.get_input_price("gpt-4o-mini")
        assert price == 0.00015

    def test_openai_explicit_provider(self):
        pricing = PricingTable(provider="openai")
        price = pricing.get_input_price("gpt-4o-mini")
        assert price == 0.00015

    def test_anthropic_provider(self):
        pricing = PricingTable(provider="anthropic")
        price = pricing.get_input_price("claude-haiku-4-5")
        assert price == 0.0008

    def test_anthropic_output_price(self):
        pricing = PricingTable(provider="anthropic")
        price = pricing.get_output_price("claude-haiku-4-5")
        assert price == 0.004

    def test_anthropic_opus_more_expensive_than_haiku(self):
        pricing = PricingTable(provider="anthropic")
        haiku = pricing.get_input_price("claude-haiku-4-5")
        opus = pricing.get_input_price("claude-opus-4-6")
        assert opus > haiku

    def test_anthropic_model_alias(self):
        pricing = PricingTable(provider="anthropic")
        # claude-sonnet-latest -> claude-sonnet-4-6
        alias_price = pricing.get_input_price("claude-sonnet-latest")
        direct_price = pricing.get_input_price("claude-sonnet-4-6")
        assert alias_price == direct_price

    def test_anthropic_versioned_model_resolves(self):
        pricing = PricingTable(provider="anthropic")
        # claude-3-5-sonnet-20241022 should resolve to claude-3-5-sonnet
        price = pricing.get_input_price("claude-3-5-sonnet-20241022")
        assert price == pricing.get_input_price("claude-3-5-sonnet")

    def test_google_provider(self):
        pricing = PricingTable(provider="google")
        price = pricing.get_input_price("gemini-2.0-flash")
        assert price == 0.0001

    def test_google_output_price(self):
        pricing = PricingTable(provider="google")
        price = pricing.get_output_price("gemini-2.0-flash")
        assert price == 0.0004

    def test_google_pro_more_expensive_than_flash(self):
        pricing = PricingTable(provider="google")
        flash = pricing.get_input_price("gemini-2.0-flash")
        pro = pricing.get_input_price("gemini-1.5-pro")
        assert pro > flash

    def test_google_versioned_model_resolves(self):
        pricing = PricingTable(provider="google")
        price = pricing.get_input_price("gemini-2.0-flash-001")
        assert price == pricing.get_input_price("gemini-2.0-flash")

    def test_unknown_provider_raises(self):
        with pytest.raises(PricingDataError, match="Unknown provider"):
            PricingTable(provider="unknown_llm")

    def test_openai_model_not_in_anthropic_table(self):
        pricing = PricingTable(provider="anthropic")
        with pytest.raises(PricingDataError):
            pricing.get_input_price("gpt-4o-mini")

    def test_anthropic_model_not_in_google_table(self):
        pricing = PricingTable(provider="google")
        with pytest.raises(PricingDataError):
            pricing.get_input_price("claude-haiku-4-5")

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
