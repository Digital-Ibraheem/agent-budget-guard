"""Smoke test: single calls, budget drain, and budget enforcement."""

import os
import sys
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


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in .env or shell:")
        print("  echo 'OPENAI_API_KEY=sk-...' > .env")
        sys.exit(1)

    budget = 0.0001  # 5 cents
    print(f"=== Smoke Test (budget: ${budget:.2f}) ===\n")

    session = BudgetedSession(budget_usd=budget)
    client = session.wrap_openai(OpenAI())

    # --- Test 1: Single call ---
    print("[1] Single call")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
        max_tokens=20,
    )
    print(f"    Response: {response.choices[0].message.content}")
    print(f"    Spent so far: ${session.get_total_spent():.6f}")
    print(f"    Remaining:    ${session.get_remaining_budget():.6f}")
    print()

    # --- Test 2: Loop until budget runs out ---
    print("[2] Looping until budget is exhausted...")
    call_count = 0
    try:
        for i in range(200):
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Reply with the number {i}"}],
                max_tokens=10,
            )
            call_count += 1
            spent = session.get_total_spent()
            remaining = session.get_remaining_budget()
            print(f"    Call {call_count}: spent=${spent:.6f}  remaining=${remaining:.6f}")
    except BudgetExceededError as e:
        print(f"\n    Budget enforced after {call_count} calls!")
        print(f"    Estimated cost of blocked call: ${e.estimated_cost:.6f}")
        print(f"    Remaining at cutoff:            ${e.remaining:.6f}")

    # --- Summary ---
    print("\n=== Final Summary ===")
    summary = session.get_summary()
    for k, v in summary.items():
        if k == "utilization_percent":
            print(f"  {k}: {v:.2f}%")
        else:
            print(f"  {k}: ${v:.6f}")


if __name__ == "__main__":
    main()
