"""Concurrent test: multiple threads sharing one budget."""

import os
import sys
import concurrent.futures
from pathlib import Path

from openai import OpenAI
from agent_budget import BudgetedSession, BudgetExceededError

# Load .env from project root
env_file = Path(__file__).resolve().parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


def agent_worker(client, session, agent_id):
    """Simulate an agent making repeated API calls."""
    calls = 0
    try:
        for _ in range(50):
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Agent {agent_id}: say hi"}],
                max_tokens=10,
            )
            calls += 1
    except BudgetExceededError:
        pass
    return agent_id, calls


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY first:")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    num_agents = 4
    budget = 0.01  # 5 cents shared across all agents
    print(f"=== Concurrent Test ({num_agents} agents, shared ${budget:.2f} budget) ===\n")

    session = BudgetedSession(budget_usd=budget)
    client = session.wrap_openai(OpenAI())

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_agents) as pool:
        futures = {
            pool.submit(agent_worker, client, session, i): i
            for i in range(num_agents)
        }
        for future in concurrent.futures.as_completed(futures):
            agent_id, calls = future.result()
            print(f"  Agent {agent_id}: made {calls} calls before budget cutoff")

    print(f"\n=== Final Summary ===")
    summary = session.get_summary()
    for k, v in summary.items():
        if k == "utilization_percent":
            print(f"  {k}: {v:.2f}%")
        else:
            print(f"  {k}: ${v:.6f}")

    total_budget = summary["budget"]
    total_spent = summary["spent"]
    if total_spent <= total_budget:
        print(f"\n  Budget respected! Spent ${total_spent:.6f} out of ${total_budget:.6f}")
    else:
        print(f"\n  BUG: Overspent! ${total_spent:.6f} > ${total_budget:.6f}")


if __name__ == "__main__":
    main()
