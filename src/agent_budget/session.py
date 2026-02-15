"""Main entry point for budget-controlled OpenAI API sessions."""

from typing import Any, Callable, List, Optional

from .tracking.tracker import SpendTracker
from .cost.pricing import PricingTable
from .cost.estimator import CostEstimator
from .cost.calculator import CostCalculator
from .wrappers.openai import OpenAIClientWrapper

DEFAULT_WARNING_THRESHOLDS = [30, 80, 95]


class BudgetedSession:
    """Manages budget tracking across multiple OpenAI API calls.

    This is the main entry point for using the agent-budget library.

    Quick start (one-liner):
        >>> from agent_budget import BudgetedSession
        >>>
        >>> client = BudgetedSession.openai(
        ...     budget_usd=5.00,
        ...     on_budget_exceeded=lambda e: print(f"Budget hit: {e}"),
        ...     on_warning=lambda w: print(f"{w['threshold']}% budget used"),
        ... )
        >>> response = client.chat.completions.create(
        ...     model="gpt-4o-mini",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> print(client.session.get_summary())

    Manual setup:
        >>> from openai import OpenAI
        >>> from agent_budget import BudgetedSession
        >>>
        >>> session = BudgetedSession(budget_usd=5.00)
        >>> client = session.wrap_openai(OpenAI())
    """

    def __init__(
        self,
        budget_usd: float,
        pricing_config: Optional[str] = None,
        tier: str = "standard",
        on_budget_exceeded: Optional[Callable] = None,
        on_warning: Optional[Callable] = None,
        warning_thresholds: Optional[List[int]] = None,
    ) -> None:
        """Initialize a budgeted session.

        Args:
            budget_usd: Total budget limit in USD (e.g., 5.00 for $5)
            pricing_config: Optional path to custom pricing JSON file.
                          If None, uses default pricing from package.
            tier: Pricing tier to use ("standard" or "batch").
            on_budget_exceeded: Optional callback when budget is exceeded.
                If set, called with the BudgetExceededError and create()
                returns None instead of raising.
            on_warning: Optional callback when utilization crosses a threshold.
                Called with a dict: {"threshold": int, "spent": float,
                "remaining": float, "budget": float}.
            warning_thresholds: Utilization % levels that trigger on_warning.
                Defaults to [30, 80, 95].

        Raises:
            ValueError: If budget is negative
            PricingDataError: If pricing config cannot be loaded
        """
        self._tracker = SpendTracker(budget_usd)
        self._pricing = PricingTable(config_path=pricing_config)
        self._estimator = CostEstimator(self._pricing)
        self._calculator = CostCalculator(self._pricing)
        self._tier = tier
        self._on_budget_exceeded = on_budget_exceeded
        self._on_warning = on_warning
        self._warning_thresholds = sorted(warning_thresholds or DEFAULT_WARNING_THRESHOLDS)
        self._fired_thresholds: set = set()

    @classmethod
    def openai(
        cls,
        budget_usd: float,
        api_key: Optional[str] = None,
        tier: str = "standard",
        on_budget_exceeded: Optional[Callable] = None,
        on_warning: Optional[Callable] = None,
        warning_thresholds: Optional[List[int]] = None,
        **openai_kwargs: Any,
    ) -> OpenAIClientWrapper:
        """One-liner: create a budget-enforced OpenAI client.

        Creates a BudgetedSession and wraps a new OpenAI client in one step.
        Uses OPENAI_API_KEY from environment if api_key is not provided.

        Args:
            budget_usd: Total budget limit in USD.
            api_key: Optional OpenAI API key. If None, uses env var.
            tier: Pricing tier ("standard" or "batch").
            on_budget_exceeded: Optional callback when budget is exceeded.
            on_warning: Optional callback at utilization thresholds.
            warning_thresholds: Utilization % levels for warnings.
                Defaults to [30, 80, 95].
            **openai_kwargs: Extra kwargs passed to OpenAI() constructor.

        Returns:
            Wrapped OpenAI client with budget enforcement.
            Access the session via client.session.
        """
        from openai import OpenAI

        session = cls(
            budget_usd=budget_usd,
            tier=tier,
            on_budget_exceeded=on_budget_exceeded,
            on_warning=on_warning,
            warning_thresholds=warning_thresholds,
        )

        client_kwargs = dict(openai_kwargs)
        if api_key is not None:
            client_kwargs["api_key"] = api_key

        wrapped = session.wrap_openai(OpenAI(**client_kwargs))
        wrapped.session = session
        return wrapped

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
        """
        effective_tier = tier if tier is not None else self._tier

        return OpenAIClientWrapper(
            client=client,
            tracker=self._tracker,
            estimator=self._estimator,
            calculator=self._calculator,
            tier=effective_tier,
            on_budget_exceeded=self._on_budget_exceeded,
            on_warning=self._on_warning,
            warning_thresholds=self._warning_thresholds,
            fired_thresholds=self._fired_thresholds,
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
