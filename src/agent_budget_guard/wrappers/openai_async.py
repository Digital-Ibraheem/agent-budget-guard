"""Async OpenAI client wrapper with budget enforcement."""

import threading
from typing import Any, Callable, List, Optional, Set

from ..cost.estimator import CostEstimator
from ..cost.calculator import CostCalculator
from ..exceptions import BudgetExceededError
from ..tracking.tracker import SpendTracker


class AsyncCompletionsWrapper:
    """Wraps async chat.completions to intercept create() calls."""

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

    async def _stream_generator(self, raw_stream: Any, reservation_id: str, model: str):
        """Async generator that commits cost from the final chunk containing usage."""
        try:
            async for chunk in raw_stream:
                yield chunk
                if chunk.usage is not None:
                    input_price = self._estimator._pricing.get_input_price(model, self._tier)
                    output_price = self._estimator._pricing.get_output_price(model, self._tier)
                    actual_cost = (
                        (chunk.usage.prompt_tokens / 1000) * input_price
                        + (chunk.usage.completion_tokens / 1000) * output_price
                    )
                    self._tracker.commit(reservation_id, actual_cost)
                    self._check_warnings()
        finally:
            self._tracker.rollback(reservation_id)

    async def create(self, **kwargs: Any) -> Any:
        """Budget-enforced async version of chat.completions.create()."""
        model = kwargs.get("model")
        messages = kwargs.get("messages", [])
        max_tokens = kwargs.get("max_tokens")

        estimated_cost = self._estimator.estimate_chat_completion_cost(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            tier=self._tier,
        )

        try:
            reservation_id = self._tracker.check_and_reserve(estimated_cost)
        except BudgetExceededError as e:
            if self._on_budget_exceeded:
                self._on_budget_exceeded(e)
                return None
            raise

        try:
            if kwargs.get("stream"):
                kwargs.setdefault("stream_options", {})
                kwargs["stream_options"]["include_usage"] = True
                raw_stream = await self._original.create(**kwargs)
                return self._stream_generator(raw_stream, reservation_id, model)

            response = await self._original.create(**kwargs)
            actual_cost = self._calculator.calculate_from_response(response, tier=self._tier)
            self._tracker.commit(reservation_id, actual_cost)
            self._check_warnings()
            return response

        except Exception:
            self._tracker.rollback(reservation_id)
            raise


class AsyncChatWrapper:
    """Wraps async client.chat namespace."""

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
    def completions(self) -> AsyncCompletionsWrapper:
        return AsyncCompletionsWrapper(
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


class AsyncOpenAIClientWrapper:
    """Wraps an openai.AsyncOpenAI() client with budget enforcement.

    Example:
        >>> from agent_budget_guard import BudgetedSession
        >>>
        >>> client = BudgetedSession.async_openai(budget_usd=5.00)
        >>> response = await client.chat.completions.create(
        ...     model="gpt-4o-mini",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>>
        >>> # Streaming
        >>> async for chunk in await client.chat.completions.create(
        ...     model="gpt-4o-mini",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     stream=True,
        ... ):
        ...     print(chunk.choices[0].delta.content or "", end="")
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
        self.session = None  # set by BudgetedSession.async_openai()

    @property
    def chat(self) -> AsyncChatWrapper:
        return AsyncChatWrapper(
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
        return getattr(self._client, name)
