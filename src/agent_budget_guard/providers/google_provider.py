"""Google provider — cost estimation and calculation for Gemini models."""

from typing import Any, Dict, List, Optional, Union

from ..cost.pricing import PricingTable
from .base import BaseProvider

# Conservative estimate: ~4 characters per token
_CHARS_PER_TOKEN = 4

# Fixed overhead tokens per turn (role label, separators)
_TOKENS_PER_TURN = 2


class GoogleProvider(BaseProvider):
    """Provider for Google Gemini models.

    Uses character-based token counting for pre-call estimation and parses
    response.usage_metadata for exact post-call cost calculation.

    Note: The model name is not included in GenerateContentResponse, so
    calculate_cost() requires the model to be passed explicitly.
    """

    def __init__(self, pricing_config: Optional[str] = None) -> None:
        self._pricing = PricingTable(config_path=pricing_config, provider="google")

    def _count_tokens_from_contents(self, contents: Any) -> int:
        """Character-based token estimate from various contents formats."""
        if contents is None:
            return 0

        if isinstance(contents, str):
            return max(1, len(contents) // _CHARS_PER_TOKEN)

        if isinstance(contents, (list, tuple)):
            total = 0
            for item in contents:
                if isinstance(item, str):
                    total += _TOKENS_PER_TURN + max(1, len(item) // _CHARS_PER_TOKEN)
                elif isinstance(item, dict):
                    total += _TOKENS_PER_TURN
                    # OpenAI-style: {"role": "user", "content": "..."}
                    content = item.get("content", "")
                    if isinstance(content, str):
                        total += max(1, len(content) // _CHARS_PER_TOKEN)
                    # Google-style: {"role": "user", "parts": [...]}
                    for part in item.get("parts", []):
                        if isinstance(part, str):
                            total += max(1, len(part) // _CHARS_PER_TOKEN)
                        elif isinstance(part, dict):
                            text = part.get("text", "")
                            total += max(1, len(str(text)) // _CHARS_PER_TOKEN)
                else:
                    # Unknown type — use string representation as fallback
                    total += max(1, len(str(item)) // _CHARS_PER_TOKEN)
            return total

        # Fallback for any other type
        return max(1, len(str(contents)) // _CHARS_PER_TOKEN)

    def estimate_cost(
        self,
        messages: List[Dict],
        model: str,
        max_tokens: Optional[int] = None,
        tier: str = "standard",
    ) -> float:
        # messages here may be the raw `contents` arg from generate_content
        input_tokens = self._count_tokens_from_contents(messages)

        if max_tokens is not None:
            output_tokens = max_tokens
        else:
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
        """Calculate actual cost from a Google GenerateContentResponse.

        Parses response.usage_metadata.prompt_token_count and
        response.usage_metadata.candidates_token_count.

        Args:
            response: GenerateContentResponse from google-genai SDK
            tier: Pricing tier
            model: Model name (required — not included in response)
        """
        if model is None:
            raise ValueError(
                "model must be provided to GoogleProvider.calculate_cost(); "
                "it is not included in the GenerateContentResponse."
            )

        metadata = response.usage_metadata
        input_tokens: int = metadata.prompt_token_count or 0
        output_tokens: int = metadata.candidates_token_count or 0

        input_price = self._pricing.get_input_price(model, tier=tier)
        output_price = self._pricing.get_output_price(model, tier=tier)

        return (input_tokens / 1000.0) * input_price + (output_tokens / 1000.0) * output_price

    def get_pricing_table(self) -> PricingTable:
        return self._pricing
