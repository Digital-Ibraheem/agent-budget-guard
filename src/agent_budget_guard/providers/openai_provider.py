"""OpenAI provider — wraps existing CostEstimator and CostCalculator."""

from typing import Any, Dict, List, Optional

from ..cost.calculator import CostCalculator
from ..cost.estimator import CostEstimator
from ..cost.pricing import PricingTable
from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI models (GPT, o-series).

    Delegates to the existing CostEstimator and CostCalculator, keeping
    all OpenAI-specific logic (tiktoken, response.usage.prompt_tokens, etc.)
    in the same place it has always lived.
    """

    def __init__(self, pricing_config: Optional[str] = None) -> None:
        self._pricing = PricingTable(config_path=pricing_config, provider="openai")
        self._estimator = CostEstimator(self._pricing)
        self._calculator = CostCalculator(self._pricing)

    def estimate_cost(
        self,
        messages: List[Dict],
        model: str,
        max_tokens: Optional[int] = None,
        tier: str = "standard",
    ) -> float:
        return self._estimator.estimate_chat_completion_cost(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            tier=tier,
        )

    def calculate_cost(
        self,
        response: Any,
        tier: str = "standard",
        model: Optional[str] = None,
    ) -> float:
        return self._calculator.calculate_from_response(response, tier=tier)

    def get_pricing_table(self) -> PricingTable:
        return self._pricing
