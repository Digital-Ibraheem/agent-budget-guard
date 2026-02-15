# Agent Budget

Prevent runaway agent costs by enforcing hard spending limits on OpenAI API usage. Wrap your OpenAI client once and guarantee that total spend never exceeds a fixed USD budget.

---

## Quick Setup

```bash
pip install -e .
```

```python
from agent_budget import BudgetedSession

client = BudgetedSession.openai(budget_usd=5.00)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)

print(f"Spent: ${client.session.get_total_spent():.4f}")
print(client.session.get_summary())
```

If the request would exceed the remaining budget, a `BudgetExceededError` is raised before the API call is made.

---

## Callbacks

Handle budget events without try/except:

```python
client = BudgetedSession.openai(
    budget_usd=5.00,
    on_budget_exceeded=lambda e: print(f"Budget hit: {e}"),
    on_warning=lambda w: print(f"{w['threshold']}% budget used"),
)

# Returns None instead of raising when budget is exceeded
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
```

Warnings fire at 30%, 80%, and 95% utilization by default. Customize with `warning_thresholds=[50, 90]`.

---

## Why This Exists

Autonomous agents can accumulate significant API costs in minutes due to recursive loops, retries, or concurrent execution. This library prevents that by:

* Blocking calls before they exceed your remaining budget
* Supporting concurrent threads safely
* Estimating cost using model-specific pricing
* Recording actual spend after each successful request

---

## Features

* One-liner setup with `BudgetedSession.openai()`
* Hard budget enforcement
* Budget warning callbacks at configurable thresholds
* Budget exceeded callback (no try/except needed)
* Thread-safe reservation system
* Compatible with all current OpenAI models including GPT-5.2 and o-series
* Drop-in wrapper for the OpenAI Python client

---

## Concurrent Agents Example

```python
import concurrent.futures
from agent_budget import BudgetedSession, BudgetExceededError

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

All agents share the same budget pool.

---

## Manual Setup

If you need more control, use the two-step API:

```python
from openai import OpenAI
from agent_budget import BudgetedSession

session = BudgetedSession(budget_usd=5.00)
client = session.wrap_openai(OpenAI())
```

---

## Batch Tier (Reduced Cost)

If using OpenAI batch tier pricing:

```python
client = BudgetedSession.openai(budget_usd=5.00, tier="batch")
```

The pricing estimator will use batch rates automatically.

---

## O-Series Model Note

Models such as `o1`, `o3`, `o3-pro`, and `o4-mini` use internal reasoning tokens that increase cost relative to visible output. Allocate budget conservatively when using reasoning-focused models.

---

## Public API

```python
# One-liner setup
client = BudgetedSession.openai(budget_usd=5.00)

# Access session from client
client.session.get_total_spent()
client.session.get_remaining_budget()
client.session.get_summary()
client.session.get_budget()
client.session.get_reserved()
client.session.reset()
```

---

## How It Works

1. Estimate token usage before sending the request
2. Calculate estimated cost using model pricing
3. Atomically reserve budget
4. Execute API call if within limit
5. Record actual cost from response
6. Check warning thresholds and fire callbacks
7. Release reservation

This guarantees:

```
spent + reserved <= budget
```

at all times, even under concurrency.

---

## Supported Models

Compatible with current OpenAI models, including:

* GPT-5.2, GPT-5.1, GPT-5-mini, GPT-5-nano
* GPT-4.1, GPT-4o, GPT-4o-mini
* o1, o3, o3-pro, o4-mini
* gpt-4-turbo, gpt-4, gpt-3.5-turbo

Pricing tables should be kept aligned with official OpenAI API pricing.

---

## Development

```bash
git clone <repo>
cd agent-budget
pip install -e ".[dev]"
pytest
```
