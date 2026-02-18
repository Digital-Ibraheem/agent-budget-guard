# Agent Budget Guard

Hard spending limits for LLM API calls. Prevents runaway agent costs.

Supports **OpenAI**, **Anthropic**, and **Google Gemini** — same interface, same thread-safety guarantees across all providers.

## Setup

```bash
pip install agent-budget-guard
```

## Usage

### OpenAI

```python
from agent_budget_guard import BudgetedSession

client = BudgetedSession.openai(
    budget_usd=5.00,
    on_budget_exceeded=lambda e: print(f"Budget hit: {e}"),
    on_warning=lambda w: print(f"{w['threshold']}% budget used"),
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)

print(response.choices[0].message.content)
print(client.session.get_summary())
```

`client` works exactly like a normal OpenAI client — same `client.chat.completions.create()` interface.

### Anthropic

```python
from agent_budget_guard import BudgetedSession

client = BudgetedSession.anthropic(budget_usd=5.00)

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)

print(response.content[0].text)
print(client.session.get_summary())
```

`client` works exactly like a normal `anthropic.Anthropic()` client.

### Google Gemini

```python
from agent_budget_guard import BudgetedSession

client = BudgetedSession.google(budget_usd=5.00)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Hello",
)

print(response.text)
print(client.session.get_summary())
```

`client` works exactly like a normal `google.genai.Client()`.

### API keys

Set the standard environment variable for each provider:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
```

Or pass `api_key=` directly to any factory method.

### Manual wrapping

If you already have a client instance, wrap it directly:

```python
import anthropic
from agent_budget_guard import BudgetedSession

session = BudgetedSession(budget_usd=5.00)
client = session.wrap_anthropic(anthropic.Anthropic())
```

Same pattern works for `wrap_openai()` and `wrap_google()`.

## Callbacks

**`on_budget_exceeded`** — Called when a request would exceed your budget. Makes the call return `None` instead of raising. Without it, a `BudgetExceededError` is raised.

**`on_warning`** — Called when utilization crosses a threshold. Fires at 30%, 80%, and 95% by default. Customize with `warning_thresholds=[50, 90]`. Each threshold fires once.

The callback receives a dict:

```python
{
    "threshold": 80,       # which % threshold was crossed
    "spent": 4.02,         # total spent so far
    "remaining": 0.98,     # budget left
    "budget": 5.00         # total budget
}
```

## Concurrent Agents

All agents share the same budget pool. Thread-safe across all providers.

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

1. Estimates cost before each API call using model-specific pricing and token counting
2. Atomically reserves budget (thread-safe lock prevents race conditions)
3. Makes the API call only if within budget
4. Calculates actual cost from the response
5. Commits actual cost and releases the reservation
6. Fires warning callbacks if utilization thresholds are crossed

Guarantees `spent + reserved <= budget` at all times, even under concurrency.

## Session API

```python
client.session.get_total_spent()        # USD spent so far
client.session.get_remaining_budget()   # USD remaining (accounts for in-flight calls)
client.session.get_summary()            # full breakdown dict
client.session.get_budget()             # total budget
client.session.get_reserved()           # USD reserved for in-flight calls
client.session.reset()                  # reset to zero
```

## Supported Models

### OpenAI
GPT-5.2, GPT-5.1, GPT-5-mini, GPT-5-nano, GPT-4.1, GPT-4.1-mini, GPT-4.1-nano, GPT-4o, GPT-4o-mini, o1, o1-pro, o3, o3-pro, o4-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo

Batch tier pricing: `BudgetedSession.openai(budget_usd=5.00, tier="batch")`

### Anthropic
claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5, claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus, claude-3-sonnet, claude-3-haiku

### Google Gemini
gemini-2.0-flash, gemini-2.0-flash-lite, gemini-2.0-pro, gemini-1.5-pro, gemini-1.5-flash, gemini-1.5-flash-8b

## Development

```bash
git clone https://github.com/Digital-Ibraheem/agent-budget-guard.git
cd agent-budget-guard
pip install -e ".[dev]"
pytest
```
