# Agent Budget Guard

Hard spending limits for LLM API calls. Prevents runaway agent costs.

Wraps **OpenAI**, **Anthropic**, and **Google Gemini** — drop-in replacement for each SDK client with budget enforcement and no other changes to your code.

## Install

```bash
pip install agent-budget-guard
```

## Quickstart

### OpenAI

```python
from agent_budget_guard import BudgetedSession

client = BudgetedSession.openai(budget_usd=5.00)

# Non-streaming — identical to normal OpenAI usage
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)

# Streaming — works the same way, cost tracked from final chunk
for chunk in client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

print(client.session.get_summary())
```

### Anthropic

```python
client = BudgetedSession.anthropic(budget_usd=5.00)

# Non-streaming
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.content[0].text)

# Streaming
for event in client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
):
    if event.type == "content_block_delta":
        print(event.delta.text, end="")
```

### Google Gemini

```python
client = BudgetedSession.google(budget_usd=5.00)

# Non-streaming
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Hello",
)
print(response.text)

# Streaming — Google uses a separate method (mirrors the underlying SDK)
for chunk in client.models.generate_content_stream(
    model="gemini-2.0-flash",
    contents="Hello",
):
    print(chunk.text, end="")
```

## API Keys

Set the standard environment variable for each provider:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
```

Or pass `api_key=` directly to any factory method.

## Manual Wrapping

If you already have a client instance, wrap it directly:

```python
from openai import OpenAI
from agent_budget_guard import BudgetedSession

session = BudgetedSession(budget_usd=5.00)
client = session.wrap_openai(OpenAI())
```

Same pattern for `wrap_anthropic()` and `wrap_google()`.

## Callbacks

```python
client = BudgetedSession.openai(
    budget_usd=5.00,
    on_budget_exceeded=lambda e: print(f"Budget hit: {e}"),
    on_warning=lambda w: print(f"{w['threshold']}% of budget used"),
    warning_thresholds=[50, 90],  # default: [30, 80, 95]
)
```

**`on_budget_exceeded`** — called when a request would exceed the budget. The call returns `None` instead of raising. Without this callback, a `BudgetExceededError` is raised.

**`on_warning`** — called when utilization crosses a threshold. Each threshold fires once per session. The callback receives:

```python
{
    "threshold": 50,       # which % threshold was crossed
    "spent": 2.51,         # total spent so far
    "remaining": 2.49,     # budget left
    "budget": 5.00         # total budget
}
```

## Concurrent Agents

All agents share the same budget pool with atomic reservation — no race conditions.

```python
import concurrent.futures
from agent_budget_guard import BudgetedSession, BudgetExceededError

client = BudgetedSession.openai(budget_usd=10.00)

def agent_task(task_id):
    for _ in range(10):
        try:
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Task {task_id}"}]
            )
        except BudgetExceededError:
            return

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(agent_task, range(5))
```

## How It Works

1. Estimates cost before the API call (token counting + model pricing)
2. Atomically reserves that amount from the budget
3. Makes the API call only if within budget
4. Calculates actual cost from the response (or final stream chunk)
5. Commits actual cost, releases the reservation
6. Fires warning callbacks if thresholds are crossed

`spent + reserved <= budget` at all times, even under concurrency.

## Session API

```python
client.session.get_total_spent()        # USD spent so far
client.session.get_remaining_budget()   # USD remaining (accounts for in-flight calls)
client.session.get_reserved()           # USD reserved for in-flight calls
client.session.get_budget()             # total budget
client.session.get_summary()            # dict with all of the above
client.session.reset()                  # reset to zero (don't use mid-flight)
```

## Supported Models

**OpenAI** — GPT-5.2, GPT-5.1, GPT-5-mini, GPT-5-nano, GPT-4.1, GPT-4.1-mini, GPT-4.1-nano, GPT-4o, GPT-4o-mini, o1, o1-pro, o3, o3-pro, o4-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo

Batch pricing: `BudgetedSession.openai(budget_usd=5.00, tier="batch")`

**Anthropic** — claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5, claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus, claude-3-sonnet, claude-3-haiku

**Google Gemini** — gemini-2.0-flash, gemini-2.0-flash-lite, gemini-2.0-pro, gemini-1.5-pro, gemini-1.5-flash, gemini-1.5-flash-8b

## Development

```bash
git clone https://github.com/Digital-Ibraheem/agent-budget-guard.git
cd agent-budget-guard
pip install -e ".[dev]"
pytest
```
