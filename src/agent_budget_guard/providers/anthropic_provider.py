"""Anthropic provider — cost estimation and calculation for Claude models."""

from typing import Any, Dict, List, Optional, Union

from ..cost.pricing import PricingTable
from .base import BaseProvider

# Conservative estimate: ~4 characters per token for English text
_CHARS_PER_TOKEN = 4

# Fixed overhead tokens per message (role, formatting markers)
_TOKENS_PER_MESSAGE = 4


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude models.

    Uses character-based token counting for pre-call estimation (conservative)
    and parses response.usage.input_tokens / response.usage.output_tokens for
    exact post-call cost calculation.
    """

    def __init__(self, pricing_config: Optional[str] = None) -> None:
        self._pricing = PricingTable(config_path=pricing_config, provider="anthropic")

    def _count_tokens(self, messages: List[Dict]) -> int:
        """Character-based token estimate for a list of Anthropic messages."""
        total = 0
        for message in messages:
            total += _TOKENS_PER_MESSAGE
            content = message.get("content", "")
            if isinstance(content, str):
                total += max(1, len(content) // _CHARS_PER_TOKEN)
            elif isinstance(content, list):
                # Content blocks: text, image, tool_use, tool_result, etc.
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", block.get("input", ""))
                        total += max(1, len(str(text)) // _CHARS_PER_TOKEN)
        return total

    def estimate_cost(
        self,
        messages: List[Dict],
        model: str,
        max_tokens: Optional[int] = None,
        tier: str = "standard",
    ) -> float:
        input_tokens = self._count_tokens(messages)

        if max_tokens is not None:
            output_tokens = max_tokens
        else:
            # Conservative: at least 1024 or 50% of input, whichever is larger
            output_tokens = max(1024, int(input_tokens * 0.5))

        input_price = self._pricing.get_input_price(model, tier=tier)
        output_price = self._pricing.get_output_price(model, tier=tier)

        return (input_tokens / 1000.0) * input_price + (output_tokens / 1000.0) * output_price

    def calculate_cost(
        self,
        response: Any,
        tier: str = "standard",
        model: Optional[str] = None,
    ) -> float:
        """Calculate actual cost from an Anthropic messages response.

        Expects response.usage.input_tokens and response.usage.output_tokens
        as returned by the Anthropic SDK.
        """
        actual_model = model or response.model
        input_tokens: int = response.usage.input_tokens
        output_tokens: int = response.usage.output_tokens

        input_price = self._pricing.get_input_price(actual_model, tier=tier)
        output_price = self._pricing.get_output_price(actual_model, tier=tier)

        return (input_tokens / 1000.0) * input_price + (output_tokens / 1000.0) * output_price

    def get_pricing_table(self) -> PricingTable:
        return self._pricing
