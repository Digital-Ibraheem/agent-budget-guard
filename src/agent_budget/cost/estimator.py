"""Cost estimation for OpenAI API calls before they are made."""

from typing import List, Dict, Any, Optional

from .pricing import PricingTable
from ..utils.tokens import count_message_tokens, estimate_completion_tokens


class CostEstimator:
    """Estimates the cost of an OpenAI API call before it's made.

    Uses tiktoken for accurate token counting and applies conservative
    estimates for output tokens to prevent budget overruns.

    Attributes:
        _pricing: PricingTable instance for looking up model prices
    """

    def __init__(self, pricing_table: PricingTable) -> None:
        """Initialize CostEstimator.

        Args:
            pricing_table: PricingTable instance with model pricing data
        """
        self._pricing = pricing_table

    def estimate_chat_completion_cost(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        tier: str = "standard"
    ) -> float:
        """Estimate the cost of a chat completion API call.

        This provides a conservative (slightly high) estimate to ensure
        we don't exceed the budget. For o-series models, it applies a
        4x multiplier to output tokens to account for hidden reasoning tokens.

        Args:
            model: Model name (e.g., "gpt-5.2", "gpt-4o-mini")
            messages: List of message dictionaries for the chat
            max_tokens: Maximum completion tokens (if specified by user)
            tier: Pricing tier ("standard" or "batch")

        Returns:
            Estimated cost in USD

        Raises:
            PricingDataError: If model pricing not found
            ValueError: If messages are invalid
        """
        # Get model encoding for token counting
        encoding_name = self._pricing.get_model_encoding(model)

        # Count input tokens using tiktoken
        input_tokens = count_message_tokens(messages, encoding_name)

        # Check if this is an o-series reasoning model
        is_reasoning = self._pricing.is_reasoning_model(model)

        # Estimate output tokens conservatively
        output_tokens = estimate_completion_tokens(
            max_tokens=max_tokens,
            input_tokens=input_tokens,
            model=model,
            is_reasoning_model=is_reasoning
        )

        # Get pricing
        input_price = self._pricing.get_input_price(model, tier=tier)
        output_price = self._pricing.get_output_price(model, tier=tier)

        # Calculate cost (prices are per 1K tokens)
        input_cost = (input_tokens / 1000.0) * input_price
        output_cost = (output_tokens / 1000.0) * output_price

        total_cost = input_cost + output_cost

        return total_cost

    def estimate_cost_with_breakdown(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        tier: str = "standard"
    ) -> Dict[str, Any]:
        """Estimate cost with detailed breakdown.

        Useful for debugging and understanding cost estimates.

        Args:
            model: Model name
            messages: List of message dictionaries
            max_tokens: Maximum completion tokens (if specified)
            tier: Pricing tier

        Returns:
            Dictionary with breakdown including:
                - total_cost: Total estimated cost in USD
                - input_tokens: Number of input tokens
                - output_tokens: Estimated output tokens
                - input_cost: Cost of input tokens
                - output_cost: Cost of output tokens
                - is_reasoning_model: Whether this uses hidden reasoning tokens
        """
        encoding_name = self._pricing.get_model_encoding(model)
        input_tokens = count_message_tokens(messages, encoding_name)
        is_reasoning = self._pricing.is_reasoning_model(model)

        output_tokens = estimate_completion_tokens(
            max_tokens=max_tokens,
            input_tokens=input_tokens,
            model=model,
            is_reasoning_model=is_reasoning
        )

        input_price = self._pricing.get_input_price(model, tier=tier)
        output_price = self._pricing.get_output_price(model, tier=tier)

        input_cost = (input_tokens / 1000.0) * input_price
        output_cost = (output_tokens / 1000.0) * output_price

        return {
            "total_cost": input_cost + output_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "input_price_per_1k": input_price,
            "output_price_per_1k": output_price,
            "is_reasoning_model": is_reasoning,
            "model": model,
            "tier": tier
        }
