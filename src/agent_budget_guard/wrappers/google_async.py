"""Async Google Gemini client wrapper with budget enforcement."""

import threading
from typing import Any, Callable, List, Optional, Set

from ..exceptions import BudgetExceededError
from ..providers.google_provider import GoogleProvider
from ..tracking.tracker import SpendTracker


class AsyncModelsWrapper:
    """Wraps client.aio.models to intercept generate_content() calls."""

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

    async def _stream_generator(self, raw_stream: Any, reservation_id: str, model: str):
        """Async generator that commits cost from the last chunk's usage_metadata."""
        last_chunk = None
        try:
            async for chunk in raw_stream:
                yield chunk
                last_chunk = chunk
            if last_chunk is not None:
                actual_cost = self._provider.calculate_cost(
                    last_chunk, tier=self._tier, model=model
                )
                self._tracker.commit(reservation_id, actual_cost)
                self._check_warnings()
        finally:
            self._tracker.rollback(reservation_id)

    async def generate_content(self, model: str, contents: Any, **kwargs: Any) -> Any:
        """Budget-enforced async version of client.aio.models.generate_content()."""
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
            response = await self._original.generate_content(
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

    async def generate_content_stream(self, model: str, contents: Any, **kwargs: Any) -> Any:
        """Budget-enforced async streaming version of client.aio.models.generate_content_stream()."""
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
            return self._stream_generator(raw_stream, reservation_id, model)

        except Exception:
            self._tracker.rollback(reservation_id)
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class AsyncGoogleClientWrapper:
    """Wraps a google.genai.Client() instance with async budget enforcement.

    Uses client.aio.models for all async operations.

    Example:
        >>> from agent_budget_guard import BudgetedSession
        >>>
        >>> client = BudgetedSession.async_google(budget_usd=5.00)
        >>> response = await client.models.generate_content(
        ...     model="gemini-2.0-flash",
        ...     contents="Hello!",
        ... )
        >>>
        >>> # Streaming
        >>> async for chunk in await client.models.generate_content_stream(
        ...     model="gemini-2.0-flash",
        ...     contents="Hello!",
        ... ):
        ...     print(chunk.text, end="")
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
        self.session = None  # set by BudgetedSession.async_google()

    @property
    def models(self) -> AsyncModelsWrapper:
        return AsyncModelsWrapper(
            self._client.aio.models,
            self._tracker,
            self._provider,
            self._tier,
            on_budget_exceeded=self._on_budget_exceeded,
            on_warning=self._on_warning,
            warning_thresholds=self._warning_thresholds,
            fired_thresholds=self._fired_thresholds,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
