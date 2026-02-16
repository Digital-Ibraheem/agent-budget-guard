# AgentGuard

Hard spending limits for OpenAI API calls. Prevents runaway agent costs.

## Setup

```bash
pip install agent-budget-guard
```

Set your API key:

```bash
export OPENAI_API_KEY=sk-...
```

Or pass it from env: `BudgetedSession.openai(budget_usd=5.00, api_key=os.getenv("OPENAI_API_KEY"))`

## Usage

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

`client` works exactly like a normal OpenAI client — same `client.chat.completions.create()` interface. When spending would exceed your budget, the call is blocked before it hits the API.

## Callbacks

**`on_budget_exceeded`** — Called when a request would exceed your budget. Makes `create()` return `None` instead of raising. Without it, a `BudgetExceededError` is raised.

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

All agents share the same budget pool. Thread-safe.

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
5. Commits actual cost and releases reservation
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

GPT-5.2, GPT-5.1, GPT-5-mini, GPT-5-nano, GPT-4.1, GPT-4o, GPT-4o-mini, o1, o3, o3-pro, o4-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo

Batch tier pricing: `BudgetedSession.openai(budget_usd=5.00, tier="batch")`

## Development

```bash
git clone https://github.com/Digital-Ibraheem/agentguard.git
cd agentguard
pip install -e ".[dev]"
pytest
```
