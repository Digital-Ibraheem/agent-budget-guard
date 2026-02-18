"""Abstract base class for LLM cost providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseProvider(ABC):
    """Abstract interface for provider-specific cost estimation and calculation.

    Each provider implementation handles:
    - Pre-call cost estimation (before the API request is made)
    - Post-call cost calculation (from the actual API response)
    """

    @abstractmethod
    def estimate_cost(
        self,
        messages: List[Dict],
        model: str,
        max_tokens: Optional[int] = None,
        tier: str = "standard",
    ) -> float:
        """Estimate the cost of an API call before it's made.

        Args:
            messages: List of message dicts (provider-specific format)
            model: Model identifier
            max_tokens: Maximum output tokens requested (if specified)
            tier: Pricing tier (e.g. "standard", "batch")

        Returns:
            Conservative cost estimate in USD
        """
        ...

    @abstractmethod
    def calculate_cost(
        self,
        response: Any,
        tier: str = "standard",
        model: Optional[str] = None,
    ) -> float:
        """Calculate the actual cost from an API response.

        Args:
            response: Provider-specific response object
            tier: Pricing tier used for the call
            model: Model name (required for providers that don't include it
                   in the response, e.g. Google)

        Returns:
            Actual cost in USD
        """
        ...
