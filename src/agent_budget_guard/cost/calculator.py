"""Cost calculation from actual OpenAI API responses."""

from typing import Any

from .pricing import PricingTable


class CostCalculator:
    """Calculates the actual cost of an API call from the response.

    Uses the exact token counts from response.usage to calculate precise
    costs after the API call completes.

    Attributes:
        _pricing: PricingTable instance for looking up model prices
    """

    def __init__(self, pricing_table: PricingTable) -> None:
        """Initialize CostCalculator.

        Args:
            pricing_table: PricingTable instance with model pricing data
        """
        self._pricing = pricing_table

    def calculate_from_response(self, response: Any, tier: str = "standard") -> float:
        """Calculate the actual cost from an OpenAI API response.

        Args:
            response: OpenAI ChatCompletion response object
            tier: Pricing tier used ("standard" or "batch")

        Returns:
            Actual cost in USD

        Raises:
            PricingDataError: If model pricing not found
            AttributeError: If response doesn't have expected structure
        """
        # Extract model and token counts from response
        model = response.model
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        # Get pricing for this model
        input_price = self._pricing.get_input_price(model, tier=tier)
        output_price = self._pricing.get_output_price(model, tier=tier)

        # Calculate cost (prices are per 1K tokens)
        input_cost = (prompt_tokens / 1000.0) * input_price
        output_cost = (completion_tokens / 1000.0) * output_price

        total_cost = input_cost + output_cost

        return total_cost

    def calculate_with_breakdown(self, response: Any, tier: str = "standard") -> dict:
        """Calculate cost with detailed breakdown.

        Args:
            response: OpenAI ChatCompletion response object
            tier: Pricing tier used

        Returns:
            Dictionary with breakdown including:
                - total_cost: Total actual cost in USD
                - prompt_tokens: Number of input tokens used
                - completion_tokens: Number of output tokens used
                - input_cost: Cost of input tokens
                - output_cost: Cost of output tokens
                - model: Model used
                - tier: Pricing tier
        """
        model = response.model
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        input_price = self._pricing.get_input_price(model, tier=tier)
        output_price = self._pricing.get_output_price(model, tier=tier)

        input_cost = (prompt_tokens / 1000.0) * input_price
        output_cost = (completion_tokens / 1000.0) * output_price

        return {
            "total_cost": input_cost + output_cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "input_price_per_1k": input_price,
            "output_price_per_1k": output_price,
            "model": model,
            "tier": tier
        }
