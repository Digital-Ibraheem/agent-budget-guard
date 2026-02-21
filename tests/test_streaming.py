"""Tests for streaming support across all three providers."""

from unittest.mock import Mock

import pytest

from agent_budget_guard import BudgetedSession, BudgetExceededError


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

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


def _make_google_chunk(prompt_token_count=10, candidates_token_count=20):
    chunk = Mock()
    chunk.usage_metadata = Mock()
    chunk.usage_metadata.prompt_token_count = prompt_token_count
    chunk.usage_metadata.candidates_token_count = candidates_token_count
    return chunk


# ---------------------------------------------------------------------------
# OpenAI streaming tests
# ---------------------------------------------------------------------------

class TestOpenAIStreaming:
    def _make_session_and_client(self, budget_usd=5.0, **session_kwargs):
        session = BudgetedSession(budget_usd=budget_usd, **session_kwargs)
        mock_sdk = Mock()
        mock_completions = Mock()
        mock_sdk.chat = Mock()
        mock_sdk.chat.completions = mock_completions
        wrapped = session.wrap_openai(mock_sdk)
        return session, wrapped, mock_completions

    def test_stream_chunks_yielded_transparently(self):
        session, wrapped, mock_completions = self._make_session_and_client()
        chunks = [
            _make_openai_chunk(usage=None, content="He"),
            _make_openai_chunk(usage=None, content="llo"),
            _make_openai_chunk(usage=_make_openai_usage(10, 20), content=None),
        ]
        mock_completions.create.return_value = iter(chunks)

        result = wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        received = list(result)
        assert received == chunks

    def test_stream_commits_on_final_chunk_with_usage(self):
        session, wrapped, mock_completions = self._make_session_and_client()
        chunks = [
            _make_openai_chunk(usage=None),
            _make_openai_chunk(usage=_make_openai_usage(10, 20)),
        ]
        mock_completions.create.return_value = iter(chunks)

        for _ in wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        ):
            pass

        assert session.get_total_spent() > 0
        assert session.get_reserved() == 0.0

    def test_stream_early_exit_rolls_back(self):
        session, wrapped, mock_completions = self._make_session_and_client()
        chunks = [
            _make_openai_chunk(usage=None, content="Hello"),
            _make_openai_chunk(usage=None, content=" World"),
            _make_openai_chunk(usage=_make_openai_usage(10, 20), content=None),
        ]
        mock_completions.create.return_value = iter(chunks)

        gen = wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        # Consume only the first chunk then close
        next(gen)
        gen.close()

        assert session.get_total_spent() == 0.0
        assert session.get_reserved() == 0.0

    def test_stream_exception_rolls_back(self):
        session, wrapped, mock_completions = self._make_session_and_client()

        def bad_stream():
            yield _make_openai_chunk(usage=None)
            raise RuntimeError("stream error")

        mock_completions.create.return_value = bad_stream()

        gen = wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        with pytest.raises(RuntimeError, match="stream error"):
            for _ in gen:
                pass

        assert session.get_total_spent() == 0.0
        assert session.get_reserved() == 0.0

    def test_stream_options_include_usage_auto_injected(self):
        session, wrapped, mock_completions = self._make_session_and_client()
        chunks = [_make_openai_chunk(usage=_make_openai_usage(5, 10))]
        mock_completions.create.return_value = iter(chunks)

        for _ in wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        ):
            pass

        call_kwargs = mock_completions.create.call_args[1]
        assert call_kwargs.get("stream_options", {}).get("include_usage") is True

    def test_stream_options_not_overwritten_if_already_set(self):
        session, wrapped, mock_completions = self._make_session_and_client()
        chunks = [_make_openai_chunk(usage=_make_openai_usage(5, 10))]
        mock_completions.create.return_value = iter(chunks)

        for _ in wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            stream_options={"include_usage": True, "custom_key": "preserved"},
        ):
            pass

        call_kwargs = mock_completions.create.call_args[1]
        assert call_kwargs["stream_options"]["custom_key"] == "preserved"
        assert call_kwargs["stream_options"]["include_usage"] is True

    def test_stream_budget_exceeded_raises_before_api_call(self):
        session, wrapped, mock_completions = self._make_session_and_client(budget_usd=0.000001)

        with pytest.raises(BudgetExceededError):
            wrapped.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )

        mock_completions.create.assert_not_called()

    def test_stream_budget_exceeded_callback(self):
        captured = []
        session, wrapped, mock_completions = self._make_session_and_client(
            budget_usd=0.000001,
            on_budget_exceeded=lambda e: captured.append(e),
        )

        result = wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        assert result is None
        assert len(captured) == 1
        assert isinstance(captured[0], BudgetExceededError)
        mock_completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Anthropic streaming tests
# ---------------------------------------------------------------------------

class TestAnthropicStreaming:
    def _make_session_and_client(self, budget_usd=5.0, **session_kwargs):
        session = BudgetedSession(budget_usd=budget_usd, **session_kwargs)
        mock_sdk = Mock()
        mock_messages = Mock()
        mock_sdk.messages = mock_messages
        wrapped = session.wrap_anthropic(mock_sdk)
        return session, wrapped, mock_messages

    def test_stream_events_yielded_transparently(self):
        session, wrapped, mock_messages = self._make_session_and_client()
        events = [
            _make_anthropic_event("message_start", input_tokens=10),
            _make_anthropic_event("content_block_start"),
            _make_anthropic_event("content_block_delta"),
            _make_anthropic_event("message_delta", output_tokens=20),
            _make_anthropic_event("message_stop"),
        ]
        mock_messages.create.return_value = iter(events)

        result = wrapped.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        received = list(result)
        assert received == events

    def test_stream_commits_after_message_delta(self):
        session, wrapped, mock_messages = self._make_session_and_client()
        events = [
            _make_anthropic_event("message_start", input_tokens=10),
            _make_anthropic_event("message_delta", output_tokens=20),
            _make_anthropic_event("message_stop"),
        ]
        mock_messages.create.return_value = iter(events)

        for _ in wrapped.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        ):
            pass

        assert session.get_total_spent() > 0
        assert session.get_reserved() == 0.0

    def test_stream_early_exit_rolls_back(self):
        session, wrapped, mock_messages = self._make_session_and_client()
        events = [
            _make_anthropic_event("message_start", input_tokens=10),
            _make_anthropic_event("content_block_delta"),
            _make_anthropic_event("message_delta", output_tokens=20),
        ]
        mock_messages.create.return_value = iter(events)

        gen = wrapped.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        next(gen)
        gen.close()

        assert session.get_total_spent() == 0.0
        assert session.get_reserved() == 0.0

    def test_stream_exception_rolls_back(self):
        session, wrapped, mock_messages = self._make_session_and_client()

        def bad_stream():
            yield _make_anthropic_event("message_start", input_tokens=10)
            raise RuntimeError("stream error")

        mock_messages.create.return_value = bad_stream()

        gen = wrapped.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        with pytest.raises(RuntimeError, match="stream error"):
            for _ in gen:
                pass

        assert session.get_total_spent() == 0.0
        assert session.get_reserved() == 0.0

    def test_stream_budget_exceeded_raises_before_api_call(self):
        session, wrapped, mock_messages = self._make_session_and_client(budget_usd=0.000001)

        with pytest.raises(BudgetExceededError):
            wrapped.messages.create(
                model="claude-opus-4-6",
                max_tokens=1000,
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )

        mock_messages.create.assert_not_called()

    def test_stream_budget_exceeded_callback(self):
        captured = []
        session, wrapped, mock_messages = self._make_session_and_client(
            budget_usd=0.000001,
            on_budget_exceeded=lambda e: captured.append(e),
        )

        result = wrapped.messages.create(
            model="claude-opus-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        assert result is None
        assert len(captured) == 1
        assert isinstance(captured[0], BudgetExceededError)
        mock_messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# Google streaming tests
# ---------------------------------------------------------------------------

class TestGoogleStreaming:
    def _make_session_and_client(self, budget_usd=5.0, **session_kwargs):
        session = BudgetedSession(budget_usd=budget_usd, **session_kwargs)
        mock_sdk = Mock()
        mock_models = Mock()
        mock_sdk.models = mock_models
        wrapped = session.wrap_google(mock_sdk)
        return session, wrapped, mock_models

    def test_stream_chunks_yielded_transparently(self):
        session, wrapped, mock_models = self._make_session_and_client()
        chunks = [
            _make_google_chunk(prompt_token_count=0, candidates_token_count=3),
            _make_google_chunk(prompt_token_count=0, candidates_token_count=5),
            _make_google_chunk(prompt_token_count=10, candidates_token_count=20),
        ]
        mock_models.generate_content_stream.return_value = iter(chunks)

        result = wrapped.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents="Hello",
        )

        received = list(result)
        assert received == chunks

    def test_stream_commits_from_last_chunk(self):
        session, wrapped, mock_models = self._make_session_and_client()
        chunks = [
            _make_google_chunk(prompt_token_count=0, candidates_token_count=5),
            _make_google_chunk(prompt_token_count=10, candidates_token_count=20),
        ]
        mock_models.generate_content_stream.return_value = iter(chunks)

        for _ in wrapped.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents="Hello",
        ):
            pass

        assert session.get_total_spent() > 0
        assert session.get_reserved() == 0.0

    def test_stream_early_exit_rolls_back(self):
        session, wrapped, mock_models = self._make_session_and_client()
        chunks = [
            _make_google_chunk(prompt_token_count=0, candidates_token_count=5),
            _make_google_chunk(prompt_token_count=10, candidates_token_count=20),
        ]
        mock_models.generate_content_stream.return_value = iter(chunks)

        gen = wrapped.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents="Hello",
        )
        next(gen)
        gen.close()

        assert session.get_total_spent() == 0.0
        assert session.get_reserved() == 0.0

    def test_stream_exception_rolls_back(self):
        session, wrapped, mock_models = self._make_session_and_client()

        def bad_stream():
            yield _make_google_chunk(prompt_token_count=0, candidates_token_count=5)
            raise RuntimeError("stream error")

        mock_models.generate_content_stream.return_value = bad_stream()

        gen = wrapped.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents="Hello",
        )
        with pytest.raises(RuntimeError, match="stream error"):
            for _ in gen:
                pass

        assert session.get_total_spent() == 0.0
        assert session.get_reserved() == 0.0

    def test_stream_budget_exceeded_raises_before_api_call(self):
        session, wrapped, mock_models = self._make_session_and_client(budget_usd=0.000001)

        with pytest.raises(BudgetExceededError):
            wrapped.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents="Hello",
            )

        mock_models.generate_content_stream.assert_not_called()

    def test_stream_budget_exceeded_callback(self):
        captured = []
        session, wrapped, mock_models = self._make_session_and_client(
            budget_usd=0.000001,
            on_budget_exceeded=lambda e: captured.append(e),
        )

        result = wrapped.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents="Hello",
        )

        assert result is None
        assert len(captured) == 1
        assert isinstance(captured[0], BudgetExceededError)
        mock_models.generate_content_stream.assert_not_called()
