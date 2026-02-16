"""Test pricing table functionality."""

import pytest
from agent_budget_guard.cost.pricing import PricingTable
from agent_budget_guard.exceptions import PricingDataError


def test_pricing_table_loads():
    """Test that pricing table loads successfully."""
    pricing = PricingTable()
    assert pricing is not None


def test_get_input_price():
    """Test getting input price for a model."""
    pricing = PricingTable()

    # GPT-4o-mini should be cheapest
    price = pricing.get_input_price("gpt-4o-mini")
    assert price == 0.00015

    # GPT-5.2 should be more expensive
    price_52 = pricing.get_input_price("gpt-5.2")
    assert price_52 > price


def test_get_output_price():
    """Test getting output price for a model."""
    pricing = PricingTable()

    price = pricing.get_output_price("gpt-4o-mini")
    assert price == 0.0006


def test_batch_tier_pricing():
    """Test batch tier pricing is cheaper."""
    pricing = PricingTable()

    standard = pricing.get_input_price("gpt-5.2", tier="standard")
    batch = pricing.get_input_price("gpt-5.2", tier="batch")

    assert batch < standard
    assert batch == standard / 2  # Batch is 50% discount


def test_model_alias():
    """Test that model aliases resolve correctly."""
    pricing = PricingTable()

    # gpt-4-0613 should resolve to gpt-4
    price = pricing.get_input_price("gpt-4-0613")
    gpt4_price = pricing.get_input_price("gpt-4")

    assert price == gpt4_price


def test_unknown_model_error():
    """Test that unknown models raise error."""
    pricing = PricingTable()

    with pytest.raises(PricingDataError):
        pricing.get_input_price("gpt-99-ultra")


def test_is_reasoning_model():
    """Test reasoning model detection."""
    pricing = PricingTable()

    # O-series should be detected as reasoning models
    assert pricing.is_reasoning_model("o1")
    assert pricing.is_reasoning_model("o3")
    assert pricing.is_reasoning_model("o4-mini")

    # Regular GPT models should not be
    assert not pricing.is_reasoning_model("gpt-5.2")
    assert not pricing.is_reasoning_model("gpt-4o")


def test_get_encoding():
    """Test getting model encoding."""
    pricing = PricingTable()

    # Newer models use o200k_base
    assert pricing.get_model_encoding("gpt-5.2") == "o200k_base"
    assert pricing.get_model_encoding("gpt-4o") == "o200k_base"

    # Older models use cl100k_base
    assert pricing.get_model_encoding("gpt-4") == "cl100k_base"
    assert pricing.get_model_encoding("gpt-3.5-turbo") == "cl100k_base"
