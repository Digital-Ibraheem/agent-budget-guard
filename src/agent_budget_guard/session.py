"""Main entry point for budget-controlled LLM API sessions."""

from typing import Any, Callable, List, Optional

from .tracking.tracker import SpendTracker
from .cost.pricing import PricingTable
from .cost.estimator import CostEstimator
from .cost.calculator import CostCalculator
from .wrappers.openai import OpenAIClientWrapper

DEFAULT_WARNING_THRESHOLDS = [30, 80, 95]


class BudgetedSession:
    """Manages budget tracking across multiple LLM API calls.

    This is the main entry point for using the agent-budget-guard library.
    Supports OpenAI, Anthropic, and Google Gemini providers.

    Quick start (one-liner):
        >>> from agent_budget_guard import BudgetedSession
        >>>
        >>> # OpenAI
        >>> client = BudgetedSession.openai(budget_usd=5.00)
        >>> response = client.chat.completions.create(
        ...     model="gpt-4o-mini",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>>
        >>> # Anthropic
        >>> client = BudgetedSession.anthropic(budget_usd=5.00)
        >>> response = client.messages.create(
        ...     model="claude-sonnet-4-6",
        ...     max_tokens=1024,
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ... )
        >>>
        >>> # Google Gemini
        >>> client = BudgetedSession.google(budget_usd=5.00)
        >>> response = client.models.generate_content(
        ...     model="gemini-2.0-flash",
        ...     contents="Hello!",
        ... )
        >>> print(client.session.get_summary())

    Manual setup:
        >>> from openai import OpenAI
        >>> from agent_budget_guard import BudgetedSession
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
                          If None, uses default OpenAI pricing from package.
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

    # ------------------------------------------------------------------ #
    # Factory class methods                                                #
    # ------------------------------------------------------------------ #

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

    @classmethod
    def anthropic(
        cls,
        budget_usd: float,
        api_key: Optional[str] = None,
        tier: str = "standard",
        on_budget_exceeded: Optional[Callable] = None,
        on_warning: Optional[Callable] = None,
        warning_thresholds: Optional[List[int]] = None,
        **anthropic_kwargs: Any,
    ) -> Any:
        """One-liner: create a budget-enforced Anthropic client.

        Creates a BudgetedSession and wraps a new Anthropic client in one step.
        Uses ANTHROPIC_API_KEY from environment if api_key is not provided.

        Args:
            budget_usd: Total budget limit in USD.
            api_key: Optional Anthropic API key. If None, uses env var.
            tier: Pricing tier ("standard").
            on_budget_exceeded: Optional callback when budget is exceeded.
            on_warning: Optional callback at utilization thresholds.
            warning_thresholds: Utilization % levels for warnings.
                Defaults to [30, 80, 95].
            **anthropic_kwargs: Extra kwargs passed to anthropic.Anthropic().

        Returns:
            Wrapped Anthropic client with budget enforcement.
            Access the session via client.session.
        """
        try:
            import anthropic as anthropic_sdk
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required to use BudgetedSession.anthropic(). "
                "Install it with: pip install agent-budget-guard[anthropic]"
            ) from exc

        session = cls(
            budget_usd=budget_usd,
            tier=tier,
            on_budget_exceeded=on_budget_exceeded,
            on_warning=on_warning,
            warning_thresholds=warning_thresholds,
        )

        client_kwargs = dict(anthropic_kwargs)
        if api_key is not None:
            client_kwargs["api_key"] = api_key

        wrapped = session.wrap_anthropic(anthropic_sdk.Anthropic(**client_kwargs))
        wrapped.session = session
        return wrapped

    @classmethod
    def google(
        cls,
        budget_usd: float,
        api_key: Optional[str] = None,
        tier: str = "standard",
        on_budget_exceeded: Optional[Callable] = None,
        on_warning: Optional[Callable] = None,
        warning_thresholds: Optional[List[int]] = None,
        **google_kwargs: Any,
    ) -> Any:
        """One-liner: create a budget-enforced Google Gemini client.

        Creates a BudgetedSession and wraps a new google.genai.Client in one
        step. Uses GOOGLE_API_KEY from environment if api_key is not provided.

        Args:
            budget_usd: Total budget limit in USD.
            api_key: Optional Google API key. If None, uses env var.
            tier: Pricing tier ("standard").
            on_budget_exceeded: Optional callback when budget is exceeded.
            on_warning: Optional callback at utilization thresholds.
            warning_thresholds: Utilization % levels for warnings.
                Defaults to [30, 80, 95].
            **google_kwargs: Extra kwargs passed to google.genai.Client().

        Returns:
            Wrapped Google genai client with budget enforcement.
            Access the session via client.session.
        """
        try:
            from google import genai as google_genai
        except ImportError as exc:
            raise ImportError(
                "The 'google-genai' package is required to use BudgetedSession.google(). "
                "Install it with: pip install agent-budget-guard[google]"
            ) from exc

        session = cls(
            budget_usd=budget_usd,
            tier=tier,
            on_budget_exceeded=on_budget_exceeded,
            on_warning=on_warning,
            warning_thresholds=warning_thresholds,
        )

        client_kwargs = dict(google_kwargs)
        if api_key is not None:
            client_kwargs["api_key"] = api_key

        wrapped = session.wrap_google(google_genai.Client(**client_kwargs))
        wrapped.session = session
        return wrapped

    # ------------------------------------------------------------------ #
    # Wrap methods                                                         #
    # ------------------------------------------------------------------ #

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

    def wrap_anthropic(self, client: Any, tier: Optional[str] = None) -> Any:
        """Wrap an Anthropic client with budget enforcement.

        Args:
            client: anthropic.Anthropic() client instance
            tier: Optional pricing tier override. If None, uses session tier.

        Returns:
            Wrapped Anthropic client with budget enforcement

        Example:
            >>> import anthropic
            >>> client = session.wrap_anthropic(anthropic.Anthropic())
        """
        from .providers.anthropic_provider import AnthropicProvider
        from .wrappers.anthropic import AnthropicClientWrapper

        effective_tier = tier if tier is not None else self._tier
        provider = AnthropicProvider()

        return AnthropicClientWrapper(
            client=client,
            tracker=self._tracker,
            provider=provider,
            tier=effective_tier,
            on_budget_exceeded=self._on_budget_exceeded,
            on_warning=self._on_warning,
            warning_thresholds=self._warning_thresholds,
            fired_thresholds=self._fired_thresholds,
        )

    def wrap_async_openai(self, client: Any, tier: Optional[str] = None) -> Any:
        """Wrap an openai.AsyncOpenAI() client with budget enforcement.

        Args:
            client: AsyncOpenAI client instance
            tier: Optional pricing tier override. If None, uses session tier.

        Returns:
            Wrapped async OpenAI client with budget enforcement
        """
        from .wrappers.openai_async import AsyncOpenAIClientWrapper

        effective_tier = tier if tier is not None else self._tier

        return AsyncOpenAIClientWrapper(
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

    def wrap_async_anthropic(self, client: Any, tier: Optional[str] = None) -> Any:
        """Wrap an anthropic.AsyncAnthropic() client with budget enforcement.

        Args:
            client: AsyncAnthropic client instance
            tier: Optional pricing tier override. If None, uses session tier.

        Returns:
            Wrapped async Anthropic client with budget enforcement
        """
        from .providers.anthropic_provider import AnthropicProvider
        from .wrappers.anthropic_async import AsyncAnthropicClientWrapper

        effective_tier = tier if tier is not None else self._tier
        provider = AnthropicProvider()

        return AsyncAnthropicClientWrapper(
            client=client,
            tracker=self._tracker,
            provider=provider,
            tier=effective_tier,
            on_budget_exceeded=self._on_budget_exceeded,
            on_warning=self._on_warning,
            warning_thresholds=self._warning_thresholds,
            fired_thresholds=self._fired_thresholds,
        )

    def wrap_async_google(self, client: Any, tier: Optional[str] = None) -> Any:
        """Wrap a google.genai.Client() with async budget enforcement.

        Uses client.aio.models internally for all async API calls.

        Args:
            client: google.genai.Client() instance
            tier: Optional pricing tier override. If None, uses session tier.

        Returns:
            Wrapped async Google client with budget enforcement
        """
        from .providers.google_provider import GoogleProvider
        from .wrappers.google_async import AsyncGoogleClientWrapper

        effective_tier = tier if tier is not None else self._tier
        provider = GoogleProvider()

        return AsyncGoogleClientWrapper(
            client=client,
            tracker=self._tracker,
            provider=provider,
            tier=effective_tier,
            on_budget_exceeded=self._on_budget_exceeded,
            on_warning=self._on_warning,
            warning_thresholds=self._warning_thresholds,
            fired_thresholds=self._fired_thresholds,
        )

    def wrap_google(self, client: Any, tier: Optional[str] = None) -> Any:
        """Wrap a Google genai client with budget enforcement.

        Args:
            client: google.genai.Client() instance
            tier: Optional pricing tier override. If None, uses session tier.

        Returns:
            Wrapped Google genai client with budget enforcement

        Example:
            >>> from google import genai
            >>> client = session.wrap_google(genai.Client())
        """
        from .providers.google_provider import GoogleProvider
        from .wrappers.google import GoogleClientWrapper

        effective_tier = tier if tier is not None else self._tier
        provider = GoogleProvider()

        return GoogleClientWrapper(
            client=client,
            tracker=self._tracker,
            provider=provider,
            tier=effective_tier,
            on_budget_exceeded=self._on_budget_exceeded,
            on_warning=self._on_warning,
            warning_thresholds=self._warning_thresholds,
            fired_thresholds=self._fired_thresholds,
        )

    # ------------------------------------------------------------------ #
    # Budget introspection                                                 #
    # ------------------------------------------------------------------ #

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
                - utilization_percent: Percentage of budget used (spent + reserved)
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
            "utilization_percent": utilization,
        }
