"""OpenAI client wrappers with budget enforcement."""

import threading
from typing import Any, Callable, List, Optional, Set

from ..cost.estimator import CostEstimator
from ..cost.calculator import CostCalculator
from ..exceptions import BudgetExceededError
from ..tracking.tracker import SpendTracker


class CompletionsWrapper:
    """Wraps chat.completions to intercept create() calls."""

    def __init__(
        self,
        original_completions: Any,
        tracker: SpendTracker,
        estimator: CostEstimator,
        calculator: CostCalculator,
        tier: str = "standard",
        on_budget_exceeded: Optional[Callable] = None,
        on_warning: Optional[Callable] = None,
        warning_thresholds: Optional[List[int]] = None,
        fired_thresholds: Optional[Set[int]] = None,
    ) -> None:
        self._original = original_completions
        self._tracker = tracker
        self._estimator = estimator
        self._calculator = calculator
        self._tier = tier
        self._on_budget_exceeded = on_budget_exceeded
        self._on_warning = on_warning
        self._warning_thresholds = warning_thresholds or []
        self._fired_thresholds = fired_thresholds if fired_thresholds is not None else set()
        self._threshold_lock = threading.Lock()

    def _check_warnings(self) -> None:
        """Fire warning callbacks for newly crossed utilization thresholds."""
        if not self._on_warning or not self._warning_thresholds:
            return

        budget = self._tracker.get_budget()
        if budget <= 0:
            return

        spent = self._tracker.get_spent()
        reserved = self._tracker.get_reserved()
        utilization = (spent + reserved) / budget * 100

        with self._threshold_lock:
            for threshold in self._warning_thresholds:
                if utilization >= threshold and threshold not in self._fired_thresholds:
                    self._fired_thresholds.add(threshold)
                    self._on_warning({
                        "threshold": threshold,
                        "spent": spent,
                        "remaining": budget - spent - reserved,
                        "budget": budget,
                    })

    def create(self, **kwargs: Any) -> Any:
        """Budget-enforced version of chat.completions.create().

        If on_budget_exceeded callback is set, returns None instead of
        raising BudgetExceededError.
        """
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
        try:
            reservation_id = self._tracker.check_and_reserve(estimated_cost)
        except BudgetExceededError as e:
            if self._on_budget_exceeded:
                self._on_budget_exceeded(e)
                return None
            raise

        try:
            # STEP 3: Make actual API call
            response = self._original.create(**kwargs)

            # STEP 4: Calculate actual cost from response
            actual_cost = self._calculator.calculate_from_response(
                response, tier=self._tier
            )

            # STEP 5: Commit actual cost, release reservation
            self._tracker.commit(reservation_id, actual_cost)

            # STEP 6: Check warning thresholds
            self._check_warnings()

            # STEP 7: Return response to caller
            return response

        except Exception:
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
        tier: str = "standard",
        on_budget_exceeded: Optional[Callable] = None,
        on_warning: Optional[Callable] = None,
        warning_thresholds: Optional[List[int]] = None,
        fired_thresholds: Optional[Set[int]] = None,
    ) -> None:
        self._original = original_chat
        self._tracker = tracker
        self._estimator = estimator
        self._calculator = calculator
        self._tier = tier
        self._on_budget_exceeded = on_budget_exceeded
        self._on_warning = on_warning
        self._warning_thresholds = warning_thresholds
        self._fired_thresholds = fired_thresholds

    @property
    def completions(self) -> CompletionsWrapper:
        return CompletionsWrapper(
            self._original.completions,
            self._tracker,
            self._estimator,
            self._calculator,
            self._tier,
            on_budget_exceeded=self._on_budget_exceeded,
            on_warning=self._on_warning,
            warning_thresholds=self._warning_thresholds,
            fired_thresholds=self._fired_thresholds,
        )


class OpenAIClientWrapper:
    """Main wrapper that mimics OpenAI client interface.

    Example:
        >>> from agent_budget import BudgetedSession
        >>>
        >>> client = BudgetedSession.openai(budget_usd=5.00)
        >>> response = client.chat.completions.create(
        ...     model="gpt-4o-mini",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> print(client.session.get_summary())
    """

    def __init__(
        self,
        client: Any,
        tracker: SpendTracker,
        estimator: CostEstimator,
        calculator: CostCalculator,
        tier: str = "standard",
        on_budget_exceeded: Optional[Callable] = None,
        on_warning: Optional[Callable] = None,
        warning_thresholds: Optional[List[int]] = None,
        fired_thresholds: Optional[Set[int]] = None,
    ) -> None:
        self._client = client
        self._tracker = tracker
        self._estimator = estimator
        self._calculator = calculator
        self._tier = tier
        self._on_budget_exceeded = on_budget_exceeded
        self._on_warning = on_warning
        self._warning_thresholds = warning_thresholds
        self._fired_thresholds = fired_thresholds
        self.session = None  # set by BudgetedSession.openai()

    @property
    def chat(self) -> ChatWrapper:
        return ChatWrapper(
            self._client.chat,
            self._tracker,
            self._estimator,
            self._calculator,
            self._tier,
            on_budget_exceeded=self._on_budget_exceeded,
            on_warning=self._on_warning,
            warning_thresholds=self._warning_thresholds,
            fired_thresholds=self._fired_thresholds,
        )

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to underlying client."""
        return getattr(self._client, name)
