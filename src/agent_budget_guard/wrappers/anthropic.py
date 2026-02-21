"""Anthropic client wrapper with budget enforcement."""

import threading
from typing import Any, Callable, List, Optional, Set

from ..exceptions import BudgetExceededError
from ..providers.anthropic_provider import AnthropicProvider
from ..tracking.tracker import SpendTracker


class MessagesCreateWrapper:
    """Wraps client.messages to intercept create() calls."""

    def __init__(
        self,
        original_messages: Any,
        tracker: SpendTracker,
        provider: AnthropicProvider,
        tier: str = "standard",
        on_budget_exceeded: Optional[Callable] = None,
        on_warning: Optional[Callable] = None,
        warning_thresholds: Optional[List[int]] = None,
        fired_thresholds: Optional[Set[int]] = None,
    ) -> None:
        self._original = original_messages
        self._tracker = tracker
        self._provider = provider
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

    def _anthropic_stream_generator(self, raw_stream: Any, reservation_id: str, model: str):
        """Transparent generator that commits cost after message_delta event."""
        input_tokens = 0
        try:
            for event in raw_stream:
                yield event
                if event.type == "message_start":
                    input_tokens = event.message.usage.input_tokens
                elif event.type == "message_delta":
                    output_tokens = event.usage.output_tokens
                    input_price = self._provider._pricing.get_input_price(model, tier=self._tier)
                    output_price = self._provider._pricing.get_output_price(model, tier=self._tier)
                    actual_cost = (
                        (input_tokens / 1000.0) * input_price
                        + (output_tokens / 1000.0) * output_price
                    )
                    self._tracker.commit(reservation_id, actual_cost)
                    self._check_warnings()
        finally:
            # No-op if already committed; rolls back on early exit or exception
            self._tracker.rollback(reservation_id)

    def create(self, **kwargs: Any) -> Any:
        """Budget-enforced version of client.messages.create().

        If on_budget_exceeded callback is set, returns None instead of
        raising BudgetExceededError.
        """
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        max_tokens = kwargs.get("max_tokens")

        estimated_cost = self._provider.estimate_cost(
            messages=messages,
            model=model,
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
                raw_stream = self._original.create(**kwargs)
                return self._anthropic_stream_generator(raw_stream, reservation_id, model)

            response = self._original.create(**kwargs)

            actual_cost = self._provider.calculate_cost(response, tier=self._tier)
            self._tracker.commit(reservation_id, actual_cost)
            self._check_warnings()
            return response

        except Exception:
            self._tracker.rollback(reservation_id)
            raise


class AnthropicClientWrapper:
    """Wraps an anthropic.Anthropic() client with budget enforcement.

    Mirrors the OpenAIClientWrapper pattern: intercepts client.messages.create()
    and enforces the configured budget.

    Example:
        >>> from agent_budget_guard import BudgetedSession
        >>>
        >>> client = BudgetedSession.anthropic(budget_usd=5.00)
        >>> response = client.messages.create(
        ...     model="claude-sonnet-4-6",
        ...     max_tokens=1024,
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ... )
        >>> print(client.session.get_summary())
    """

    def __init__(
        self,
        client: Any,
        tracker: SpendTracker,
        provider: AnthropicProvider,
        tier: str = "standard",
        on_budget_exceeded: Optional[Callable] = None,
        on_warning: Optional[Callable] = None,
        warning_thresholds: Optional[List[int]] = None,
        fired_thresholds: Optional[Set[int]] = None,
    ) -> None:
        self._client = client
        self._tracker = tracker
        self._provider = provider
        self._tier = tier
        self._on_budget_exceeded = on_budget_exceeded
        self._on_warning = on_warning
        self._warning_thresholds = warning_thresholds
        self._fired_thresholds = fired_thresholds
        self.session = None  # set by BudgetedSession.anthropic()

    @property
    def messages(self) -> MessagesCreateWrapper:
        return MessagesCreateWrapper(
            self._client.messages,
            self._tracker,
            self._provider,
            self._tier,
            on_budget_exceeded=self._on_budget_exceeded,
            on_warning=self._on_warning,
            warning_thresholds=self._warning_thresholds,
            fired_thresholds=self._fired_thresholds,
        )

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the underlying Anthropic client."""
        return getattr(self._client, name)
