"""Google Gemini client wrapper with budget enforcement."""

import threading
from typing import Any, Callable, List, Optional, Set

from ..exceptions import BudgetExceededError
from ..providers.google_provider import GoogleProvider
from ..tracking.tracker import SpendTracker


class ModelsWrapper:
    """Wraps client.models to intercept generate_content() calls."""

    def __init__(
        self,
        original_models: Any,
        tracker: SpendTracker,
        provider: GoogleProvider,
        tier: str = "standard",
        on_budget_exceeded: Optional[Callable] = None,
        on_warning: Optional[Callable] = None,
        warning_thresholds: Optional[List[int]] = None,
        fired_thresholds: Optional[Set[int]] = None,
    ) -> None:
        self._original = original_models
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

    def generate_content(self, model: str, contents: Any, **kwargs: Any) -> Any:
        """Budget-enforced version of client.models.generate_content().

        Args:
            model: Gemini model name (e.g. "gemini-2.0-flash")
            contents: Prompt content — string, list of strings, or list of
                      Content dicts with "parts" key.
            **kwargs: Extra args forwarded to the underlying SDK call
                      (e.g. config=GenerateContentConfig(...)).

        If on_budget_exceeded callback is set, returns None instead of
        raising BudgetExceededError.
        """
        # Normalise contents into something the provider can count tokens for
        messages = contents if contents is not None else []

        # Extract max_output_tokens from config kwarg if present
        max_tokens: Optional[int] = None
        config = kwargs.get("config")
        if config is not None:
            if isinstance(config, dict):
                max_tokens = config.get("max_output_tokens")
            else:
                max_tokens = getattr(config, "max_output_tokens", None)

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
            response = self._original.generate_content(
                model=model, contents=contents, **kwargs
            )

            actual_cost = self._provider.calculate_cost(
                response, tier=self._tier, model=model
            )
            self._tracker.commit(reservation_id, actual_cost)
            self._check_warnings()
            return response

        except Exception:
            self._tracker.rollback(reservation_id)
            raise

    def _google_stream_generator(self, raw_stream: Any, reservation_id: str, model: str):
        """Transparent generator that commits cost from the last chunk's usage_metadata."""
        last_chunk = None
        try:
            for chunk in raw_stream:
                yield chunk
                last_chunk = chunk
            # Stream completed normally — commit from last chunk
            if last_chunk is not None:
                actual_cost = self._provider.calculate_cost(
                    last_chunk, tier=self._tier, model=model
                )
                self._tracker.commit(reservation_id, actual_cost)
                self._check_warnings()
        finally:
            # No-op if already committed; rolls back on early exit or exception
            self._tracker.rollback(reservation_id)

    def generate_content_stream(self, model: str, contents: Any, **kwargs: Any) -> Any:
        """Budget-enforced streaming version of client.models.generate_content_stream().

        Args:
            model: Gemini model name (e.g. "gemini-2.0-flash")
            contents: Prompt content — string, list of strings, or list of
                      Content dicts with "parts" key.
            **kwargs: Extra args forwarded to the underlying SDK call.

        If on_budget_exceeded callback is set, returns None instead of
        raising BudgetExceededError.
        """
        messages = contents if contents is not None else []

        max_tokens: Optional[int] = None
        config = kwargs.get("config")
        if config is not None:
            if isinstance(config, dict):
                max_tokens = config.get("max_output_tokens")
            else:
                max_tokens = getattr(config, "max_output_tokens", None)

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
            raw_stream = self._original.generate_content_stream(
                model=model, contents=contents, **kwargs
            )
            return self._google_stream_generator(raw_stream, reservation_id, model)

        except Exception:
            self._tracker.rollback(reservation_id)
            raise

    def __getattr__(self, name: str) -> Any:
        """Forward all other client.models calls (count_tokens, etc.) unchanged."""
        return getattr(self._original, name)


class GoogleClientWrapper:
    """Wraps a google.genai.Client() instance with budget enforcement.

    Mirrors the OpenAIClientWrapper pattern: intercepts
    client.models.generate_content() and enforces the configured budget.

    Example:
        >>> from agent_budget_guard import BudgetedSession
        >>>
        >>> client = BudgetedSession.google(budget_usd=5.00)
        >>> response = client.models.generate_content(
        ...     model="gemini-2.0-flash",
        ...     contents="Tell me a joke.",
        ... )
        >>> print(client.session.get_summary())
    """

    def __init__(
        self,
        client: Any,
        tracker: SpendTracker,
        provider: GoogleProvider,
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
        self.session = None  # set by BudgetedSession.google()

    @property
    def models(self) -> ModelsWrapper:
        return ModelsWrapper(
            self._client.models,
            self._tracker,
            self._provider,
            self._tier,
            on_budget_exceeded=self._on_budget_exceeded,
            on_warning=self._on_warning,
            warning_thresholds=self._warning_thresholds,
            fired_thresholds=self._fired_thresholds,
        )

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the underlying Google client."""
        return getattr(self._client, name)
