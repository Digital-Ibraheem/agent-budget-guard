"""Main entry point for budget-controlled OpenAI API sessions."""

from typing import Any, Optional

from .tracking.tracker import SpendTracker
from .cost.pricing import PricingTable
from .cost.estimator import CostEstimator
from .cost.calculator import CostCalculator
from .wrappers.openai import OpenAIClientWrapper


class BudgetedSession:
    """Manages budget tracking across multiple OpenAI API calls.

    This is the main entry point for using the agent-budget library.
    Create a session with a budget limit, wrap your OpenAI client, and
    all API calls will be automatically budget-enforced.

    Example:
        >>> from openai import OpenAI
        >>> from agent_budget import BudgetedSession
        >>>
        >>> # Create session with $5 budget
        >>> session = BudgetedSession(budget_usd=5.00)
        >>>
        >>> # Wrap your OpenAI client
        >>> client = session.wrap_openai(OpenAI())
        >>>
        >>> # Make API calls - budget is automatically enforced
        >>> response = client.chat.completions.create(
        ...     model="gpt-4o-mini",
        ...     max_tokens=100,
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>>
        >>> # Check spending
        >>> print(f"Spent: ${session.get_total_spent():.4f}")
        >>> print(f"Remaining: ${session.get_remaining_budget():.4f}")

    Attributes:
        _tracker: SpendTracker for thread-safe budget tracking
        _pricing: PricingTable with model pricing data
        _estimator: CostEstimator for pre-call estimates
        _calculator: CostCalculator for post-call actual costs
    """

    def __init__(
        self,
        budget_usd: float,
        pricing_config: Optional[str] = None,
        tier: str = "standard"
    ) -> None:
        """Initialize a budgeted session.

        Args:
            budget_usd: Total budget limit in USD (e.g., 5.00 for $5)
            pricing_config: Optional path to custom pricing JSON file.
                          If None, uses default pricing from package.
            tier: Pricing tier to use ("standard" or "batch").
                 Default is "standard". Use "batch" for Batch API calls.

        Raises:
            ValueError: If budget is negative
            PricingDataError: If pricing config cannot be loaded
        """
        self._tracker = SpendTracker(budget_usd)
        self._pricing = PricingTable(config_path=pricing_config)
        self._estimator = CostEstimator(self._pricing)
        self._calculator = CostCalculator(self._pricing)
        self._tier = tier

    def wrap_openai(self, client: Any, tier: Optional[str] = None) -> OpenAIClientWrapper:
        """Wrap an OpenAI client with budget enforcement.

        Args:
            client: OpenAI client instance (from openai.OpenAI())
            tier: Optional pricing tier override for this client.
                 If None, uses the session's tier.

        Returns:
            Wrapped OpenAI client with budget enforcement

        Example:
            >>> from openai import OpenAI
            >>> client = session.wrap_openai(OpenAI())
            >>>
            >>> # Or with batch pricing
            >>> batch_client = session.wrap_openai(OpenAI(), tier="batch")
        """
        effective_tier = tier if tier is not None else self._tier

        return OpenAIClientWrapper(
            client=client,
            tracker=self._tracker,
            estimator=self._estimator,
            calculator=self._calculator,
            tier=effective_tier
        )

    def get_total_spent(self) -> float:
        """Get the total amount spent so far.

        This does NOT include pending reservations (in-flight API calls).

        Returns:
            Amount spent in USD
        """
        return self._tracker.get_spent()

    def get_remaining_budget(self) -> float:
        """Get the remaining budget available.

        This accounts for both spent amounts and currently reserved
        amounts (in-flight API calls).

        Returns:
            Remaining budget in USD
        """
        return self._tracker.get_remaining()

    def get_budget(self) -> float:
        """Get the total budget for this session.

        Returns:
            Total budget in USD
        """
        return self._tracker.get_budget()

    def get_reserved(self) -> float:
        """Get the amount currently reserved for in-flight API calls.

        Returns:
            Amount reserved in USD
        """
        return self._tracker.get_reserved()

    def reset(self) -> None:
        """Reset spent and reserved amounts to zero.

        WARNING: This does not cancel in-flight API calls. Only use this
        when you're sure no API calls are pending and you want to reuse
        the session with the same budget.

        Example:
            >>> session = BudgetedSession(budget_usd=5.00)
            >>> # ... make some API calls ...
            >>> session.reset()  # Start fresh with same $5 budget
        """
        self._tracker.reset()

    def get_summary(self) -> dict:
        """Get a summary of budget usage.

        Returns:
            Dictionary with:
                - budget: Total budget in USD
                - spent: Amount spent in USD
                - reserved: Amount reserved (in-flight calls) in USD
                - remaining: Remaining budget in USD
                - utilization: Percentage of budget used (spent + reserved)
        """
        budget = self._tracker.get_budget()
        spent = self._tracker.get_spent()
        reserved = self._tracker.get_reserved()
        remaining = self._tracker.get_remaining()

        utilization = ((spent + reserved) / budget * 100) if budget > 0 else 0.0

        return {
            "budget": budget,
            "spent": spent,
            "reserved": reserved,
            "remaining": remaining,
            "utilization_percent": utilization
        }
