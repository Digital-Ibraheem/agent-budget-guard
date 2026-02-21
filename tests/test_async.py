"""Tests for async support across all three providers."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from agent_budget_guard import BudgetedSession, BudgetExceededError


# ---------------------------------------------------------------------------
# Async iterator helpers (mock streaming responses)
# ---------------------------------------------------------------------------

async def async_iter(items):
    for item in items:
        yield item


def _make_openai_usage(prompt_tokens=10, completion_tokens=20):
    usage = Mock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    return usage


def _make_openai_chunk(usage=None, content="hello"):
    chunk = Mock()
    chunk.usage = usage
    chunk.choices = [Mock()]
    chunk.choices[0].delta = Mock()
    chunk.choices[0].delta.content = content
    chunk.model = "gpt-4o-mini"
    return chunk


def _make_openai_response(model="gpt-4o-mini", prompt_tokens=10, completion_tokens=20):
    resp = Mock()
    resp.model = model
    resp.usage = Mock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


def _make_anthropic_event(event_type, **kwargs):
    event = Mock()
    event.type = event_type
    if event_type == "message_start":
        event.message = Mock()
        event.message.usage = Mock()
        event.message.usage.input_tokens = kwargs.get("input_tokens", 10)
    elif event_type == "message_delta":
        event.usage = Mock()
        event.usage.output_tokens = kwargs.get("output_tokens", 20)
    return event


def _make_anthropic_response(model="claude-haiku-4-5", input_tokens=10, output_tokens=20):
    resp = Mock()
    resp.model = model
    resp.usage = Mock()
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    return resp


def _make_google_chunk(prompt_token_count=10, candidates_token_count=20):
    chunk = Mock()
    chunk.usage_metadata = Mock()
    chunk.usage_metadata.prompt_token_count = prompt_token_count
    chunk.usage_metadata.candidates_token_count = candidates_token_count
    return chunk


# ---------------------------------------------------------------------------
# Async OpenAI tests
# ---------------------------------------------------------------------------

class TestAsyncOpenAI:
    def _make_session_and_client(self, budget_usd=5.0, **session_kwargs):
        session = BudgetedSession(budget_usd=budget_usd, **session_kwargs)
        mock_sdk = Mock()
        mock_completions = Mock()
        mock_sdk.chat = Mock()
        mock_sdk.chat.completions = mock_completions
        wrapped = session.wrap_async_openai(mock_sdk)
        return session, wrapped, mock_completions

    async def test_non_streaming_call_tracked(self):
        session, wrapped, mock_completions = self._make_session_and_client()
        mock_completions.create = AsyncMock(
            return_value=_make_openai_response(prompt_tokens=10, completion_tokens=20)
        )

        response = await wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert response is not None
        assert session.get_total_spent() > 0
        assert session.get_reserved() == 0.0

    async def test_non_streaming_budget_exceeded_raises(self):
        session, wrapped, mock_completions = self._make_session_and_client(budget_usd=0.000001)

        with pytest.raises(BudgetExceededError):
            await wrapped.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi"}],
            )

        mock_completions.create.assert_not_called()

    async def test_non_streaming_budget_exceeded_callback(self):
        captured = []
        session, wrapped, mock_completions = self._make_session_and_client(
            budget_usd=0.000001,
            on_budget_exceeded=lambda e: captured.append(e),
        )

        result = await wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert result is None
        assert len(captured) == 1

    async def test_non_streaming_api_error_rolls_back(self):
        session, wrapped, mock_completions = self._make_session_and_client()
        mock_completions.create = AsyncMock(side_effect=RuntimeError("api error"))

        with pytest.raises(RuntimeError, match="api error"):
            await wrapped.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert session.get_total_spent() == 0.0
        assert session.get_reserved() == 0.0

    async def test_streaming_chunks_yielded_transparently(self):
        session, wrapped, mock_completions = self._make_session_and_client()
        chunks = [
            _make_openai_chunk(usage=None, content="He"),
            _make_openai_chunk(usage=None, content="llo"),
            _make_openai_chunk(usage=_make_openai_usage(10, 20), content=None),
        ]
        mock_completions.create = AsyncMock(return_value=async_iter(chunks))

        gen = await wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        received = [chunk async for chunk in gen]
        assert received == chunks

    async def test_streaming_commits_on_final_chunk(self):
        session, wrapped, mock_completions = self._make_session_and_client()
        chunks = [
            _make_openai_chunk(usage=None),
            _make_openai_chunk(usage=_make_openai_usage(10, 20)),
        ]
        mock_completions.create = AsyncMock(return_value=async_iter(chunks))

        async for _ in await wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        ):
            pass

        assert session.get_total_spent() > 0
        assert session.get_reserved() == 0.0

    async def test_streaming_early_exit_rolls_back(self):
        session, wrapped, mock_completions = self._make_session_and_client()
        chunks = [
            _make_openai_chunk(usage=None, content="Hello"),
            _make_openai_chunk(usage=None, content=" World"),
            _make_openai_chunk(usage=_make_openai_usage(10, 20), content=None),
        ]
        mock_completions.create = AsyncMock(return_value=async_iter(chunks))

        gen = await wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        async for _ in gen:
            break
        await gen.aclose()

        assert session.get_total_spent() == 0.0
        assert session.get_reserved() == 0.0

    async def test_streaming_stream_options_auto_injected(self):
        session, wrapped, mock_completions = self._make_session_and_client()
        mock_completions.create = AsyncMock(
            return_value=async_iter([_make_openai_chunk(usage=_make_openai_usage(5, 10))])
        )

        async for _ in await wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        ):
            pass

        call_kwargs = mock_completions.create.call_args[1]
        assert call_kwargs.get("stream_options", {}).get("include_usage") is True

    async def test_factory_method(self):
        mock_sdk = Mock()
        mock_sdk_cls = Mock(return_value=mock_sdk)

        with patch.dict("sys.modules", {"openai": Mock(AsyncOpenAI=mock_sdk_cls)}):
            client = BudgetedSession.async_openai(budget_usd=3.0, api_key="test")

        assert client.session is not None
        assert client.session.get_budget() == 3.0


# ---------------------------------------------------------------------------
# Async Anthropic tests
# ---------------------------------------------------------------------------

class TestAsyncAnthropic:
    def _make_session_and_client(self, budget_usd=5.0, **session_kwargs):
        session = BudgetedSession(budget_usd=budget_usd, **session_kwargs)
        mock_sdk = Mock()
        mock_messages = Mock()
        mock_sdk.messages = mock_messages
        wrapped = session.wrap_async_anthropic(mock_sdk)
        return session, wrapped, mock_messages

    async def test_non_streaming_call_tracked(self):
        session, wrapped, mock_messages = self._make_session_and_client()
        mock_messages.create = AsyncMock(
            return_value=_make_anthropic_response(input_tokens=10, output_tokens=20)
        )

        response = await wrapped.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert response is not None
        assert session.get_total_spent() > 0
        assert session.get_reserved() == 0.0

    async def test_non_streaming_budget_exceeded_raises(self):
        session, wrapped, mock_messages = self._make_session_and_client(budget_usd=0.000001)

        with pytest.raises(BudgetExceededError):
            await wrapped.messages.create(
                model="claude-opus-4-6",
                max_tokens=1000,
                messages=[{"role": "user", "content": "Hi"}],
            )

        mock_messages.create.assert_not_called()

    async def test_non_streaming_api_error_rolls_back(self):
        session, wrapped, mock_messages = self._make_session_and_client()
        mock_messages.create = AsyncMock(side_effect=RuntimeError("api error"))

        with pytest.raises(RuntimeError):
            await wrapped.messages.create(
                model="claude-haiku-4-5",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert session.get_total_spent() == 0.0
        assert session.get_reserved() == 0.0

    async def test_streaming_events_yielded_transparently(self):
        session, wrapped, mock_messages = self._make_session_and_client()
        events = [
            _make_anthropic_event("message_start", input_tokens=10),
            _make_anthropic_event("content_block_delta"),
            _make_anthropic_event("message_delta", output_tokens=20),
            _make_anthropic_event("message_stop"),
        ]
        mock_messages.create = AsyncMock(return_value=async_iter(events))

        gen = await wrapped.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        received = [e async for e in gen]
        assert received == events

    async def test_streaming_commits_after_message_delta(self):
        session, wrapped, mock_messages = self._make_session_and_client()
        events = [
            _make_anthropic_event("message_start", input_tokens=10),
            _make_anthropic_event("message_delta", output_tokens=20),
            _make_anthropic_event("message_stop"),
        ]
        mock_messages.create = AsyncMock(return_value=async_iter(events))

        async for _ in await wrapped.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        ):
            pass

        assert session.get_total_spent() > 0
        assert session.get_reserved() == 0.0

    async def test_streaming_early_exit_rolls_back(self):
        session, wrapped, mock_messages = self._make_session_and_client()
        events = [
            _make_anthropic_event("message_start", input_tokens=10),
            _make_anthropic_event("content_block_delta"),
            _make_anthropic_event("message_delta", output_tokens=20),
        ]
        mock_messages.create = AsyncMock(return_value=async_iter(events))

        gen = await wrapped.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        async for _ in gen:
            break
        await gen.aclose()

        assert session.get_total_spent() == 0.0
        assert session.get_reserved() == 0.0

    async def test_factory_method(self):
        mock_sdk = Mock()
        mock_sdk_cls = Mock(return_value=mock_sdk)

        with patch.dict("sys.modules", {"anthropic": Mock(AsyncAnthropic=mock_sdk_cls)}):
            client = BudgetedSession.async_anthropic(budget_usd=3.0, api_key="test")

        assert client.session is not None
        assert client.session.get_budget() == 3.0


# ---------------------------------------------------------------------------
# Async Google tests
# ---------------------------------------------------------------------------

class TestAsyncGoogle:
    def _make_session_and_client(self, budget_usd=5.0, **session_kwargs):
        session = BudgetedSession(budget_usd=budget_usd, **session_kwargs)
        mock_sdk = Mock()
        mock_aio_models = Mock()
        mock_sdk.aio = Mock()
        mock_sdk.aio.models = mock_aio_models
        wrapped = session.wrap_async_google(mock_sdk)
        return session, wrapped, mock_aio_models

    async def test_non_streaming_call_tracked(self):
        session, wrapped, mock_models = self._make_session_and_client()
        mock_models.generate_content = AsyncMock(
            return_value=_make_google_chunk(prompt_token_count=10, candidates_token_count=20)
        )

        response = await wrapped.models.generate_content(
            model="gemini-2.0-flash",
            contents="Hello",
        )

        assert response is not None
        assert session.get_total_spent() > 0
        assert session.get_reserved() == 0.0

    async def test_non_streaming_budget_exceeded_raises(self):
        session, wrapped, mock_models = self._make_session_and_client(budget_usd=0.000001)

        with pytest.raises(BudgetExceededError):
            await wrapped.models.generate_content(
                model="gemini-2.0-flash",
                contents="Hello",
            )

        mock_models.generate_content.assert_not_called()

    async def test_non_streaming_api_error_rolls_back(self):
        session, wrapped, mock_models = self._make_session_and_client()
        mock_models.generate_content = AsyncMock(side_effect=RuntimeError("api error"))

        with pytest.raises(RuntimeError):
            await wrapped.models.generate_content(
                model="gemini-2.0-flash",
                contents="Hello",
            )

        assert session.get_total_spent() == 0.0
        assert session.get_reserved() == 0.0

    async def test_streaming_chunks_yielded_transparently(self):
        session, wrapped, mock_models = self._make_session_and_client()
        chunks = [
            _make_google_chunk(prompt_token_count=0, candidates_token_count=5),
            _make_google_chunk(prompt_token_count=10, candidates_token_count=20),
        ]
        mock_models.generate_content_stream = Mock(return_value=async_iter(chunks))

        gen = await wrapped.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents="Hello",
        )

        received = [chunk async for chunk in gen]
        assert received == chunks

    async def test_streaming_commits_from_last_chunk(self):
        session, wrapped, mock_models = self._make_session_and_client()
        chunks = [
            _make_google_chunk(prompt_token_count=0, candidates_token_count=5),
            _make_google_chunk(prompt_token_count=10, candidates_token_count=20),
        ]
        mock_models.generate_content_stream = Mock(return_value=async_iter(chunks))

        async for _ in await wrapped.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents="Hello",
        ):
            pass

        assert session.get_total_spent() > 0
        assert session.get_reserved() == 0.0

    async def test_streaming_early_exit_rolls_back(self):
        session, wrapped, mock_models = self._make_session_and_client()
        chunks = [
            _make_google_chunk(prompt_token_count=0, candidates_token_count=5),
            _make_google_chunk(prompt_token_count=10, candidates_token_count=20),
        ]
        mock_models.generate_content_stream = Mock(return_value=async_iter(chunks))

        gen = await wrapped.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents="Hello",
        )
        async for _ in gen:
            break
        await gen.aclose()

        assert session.get_total_spent() == 0.0
        assert session.get_reserved() == 0.0

    async def test_streaming_budget_exceeded_raises(self):
        session, wrapped, mock_models = self._make_session_and_client(budget_usd=0.000001)

        with pytest.raises(BudgetExceededError):
            await wrapped.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents="Hello",
            )

        mock_models.generate_content_stream.assert_not_called()
