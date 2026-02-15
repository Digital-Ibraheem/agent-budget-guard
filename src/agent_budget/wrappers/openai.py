"""OpenAI client wrappers with budget enforcement."""

from typing import Any

from ..cost.estimator import CostEstimator
from ..cost.calculator import CostCalculator
from ..tracking.tracker import SpendTracker


class CompletionsWrapper:
    """Wraps chat.completions to intercept create() calls."""

    def __init__(
        self,
        original_completions: Any,
        tracker: SpendTracker,
        estimator: CostEstimator,
        calculator: CostCalculator,
        tier: str = "standard"
    ) -> None:
        """Initialize CompletionsWrapper.

        Args:
            original_completions: Original client.chat.completions object
            tracker: SpendTracker for budget enforcement
            estimator: CostEstimator for pre-call estimation
            calculator: CostCalculator for post-call actual cost
            tier: Pricing tier to use ("standard" or "batch")
        """
        self._original = original_completions
        self._tracker = tracker
        self._estimator = estimator
        self._calculator = calculator
        self._tier = tier

    def create(self, **kwargs: Any) -> Any:
        """Budget-enforced version of chat.completions.create().

        This method:
        1. Estimates the cost before the API call
        2. Checks and reserves budget atomically
        3. Makes the actual API call if budget allows
        4. Calculates actual cost from response
        5. Commits actual cost and releases reservation
        6. Returns the response to the caller

        If any step fails, the reservation is rolled back.

        Args:
            **kwargs: Arguments to pass to OpenAI chat.completions.create()

        Returns:
            OpenAI ChatCompletion response

        Raises:
            BudgetExceededError: If estimated cost exceeds remaining budget
            Any exceptions from the underlying OpenAI API call
        """
        # Extract parameters for cost estimation
        model = kwargs.get("model")
        messages = kwargs.get("messages", [])
        max_tokens = kwargs.get("max_tokens")

        # STEP 1: Estimate cost before call
        estimated_cost = self._estimator.estimate_chat_completion_cost(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            tier=self._tier
        )

        # STEP 2: Atomic budget check + reserve
        # This will raise BudgetExceededError if budget insufficient
        reservation_id = self._tracker.check_and_reserve(estimated_cost)

        try:
            # STEP 3: Make actual API call
            response = self._original.create(**kwargs)

            # STEP 4: Calculate actual cost from response
            actual_cost = self._calculator.calculate_from_response(
                response, tier=self._tier
            )

            # STEP 5: Commit actual cost, release reservation
            self._tracker.commit(reservation_id, actual_cost)

            # STEP 6: Return response to caller
            return response

        except Exception as e:
            # Rollback reservation on any error
            self._tracker.rollback(reservation_id)
            raise


class ChatWrapper:
    """Wraps client.chat namespace."""

    def __init__(
        self,
        original_chat: Any,
        tracker: SpendTracker,
        estimator: CostEstimator,
        calculator: CostCalculator,
        tier: str = "standard"
    ) -> None:
        """Initialize ChatWrapper.

        Args:
            original_chat: Original client.chat object
            tracker: SpendTracker for budget enforcement
            estimator: CostEstimator for pre-call estimation
            calculator: CostCalculator for post-call actual cost
            tier: Pricing tier to use
        """
        self._original = original_chat
        self._tracker = tracker
        self._estimator = estimator
        self._calculator = calculator
        self._tier = tier

    @property
    def completions(self) -> CompletionsWrapper:
        """Return wrapper for completions endpoint.

        Returns:
            CompletionsWrapper that intercepts create() calls
        """
        return CompletionsWrapper(
            self._original.completions,
            self._tracker,
            self._estimator,
            self._calculator,
            self._tier
        )


class OpenAIClientWrapper:
    """Main wrapper that mimics OpenAI client interface.

    This wraps an OpenAI client instance and provides the same interface,
    but with budget enforcement on all API calls.

    Example:
        >>> from openai import OpenAI
        >>> from agent_budget import BudgetedSession
        >>>
        >>> session = BudgetedSession(budget_usd=5.00)
        >>> client = session.wrap_openai(OpenAI())
        >>>
        >>> # Use like normal OpenAI client, but budget-protected
        >>> response = client.chat.completions.create(
        ...     model="gpt-4o-mini",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
    """

    def __init__(
        self,
        client: Any,
        tracker: SpendTracker,
        estimator: CostEstimator,
        calculator: CostCalculator,
        tier: str = "standard"
    ) -> None:
        """Initialize OpenAIClientWrapper.

        Args:
            client: Original OpenAI client instance
            tracker: SpendTracker for budget enforcement
            estimator: CostEstimator for pre-call estimation
            calculator: CostCalculator for post-call actual cost
            tier: Pricing tier to use ("standard" or "batch")
        """
        self._client = client
        self._tracker = tracker
        self._estimator = estimator
        self._calculator = calculator
        self._tier = tier

    @property
    def chat(self) -> ChatWrapper:
        """Return wrapper for chat endpoint.

        Returns:
            ChatWrapper that provides access to completions
        """
        return ChatWrapper(
            self._client.chat,
            self._tracker,
            self._estimator,
            self._calculator,
            self._tier
        )

    # Provide access to other client attributes/methods
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to underlying client.

        This allows accessing other OpenAI client features like
        embeddings, images, etc. (not budget-wrapped).

        Args:
            name: Attribute name

        Returns:
            Attribute from underlying client
        """
        return getattr(self._client, name)
