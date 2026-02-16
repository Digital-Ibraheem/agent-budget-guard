"""Smoke test: one-liner setup, callbacks, and budget warnings."""

import os
import sys
from pathlib import Path

from agent_budget_guard import BudgetedSession

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

    budget = 0.0001

    # One-liner setup with callbacks — no try/except needed
    client = BudgetedSession.openai(
        budget_usd=budget,
        on_budget_exceeded=lambda e: print(f"\n    BUDGET HIT: need ${e.estimated_cost:.6f}, have ${e.remaining:.6f}"),
        on_warning=lambda w: print(f"\n    WARNING: {w['threshold']}% budget used (${w['spent']:.6f} spent)"),
    )

    print(f"=== Smoke Test (budget: ${budget}) ===\n")

    # --- Test 1: Single call ---
    print("[1] Single call")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
        max_tokens=20,
    )
    print(f"    Response: {response.choices[0].message.content}")
    print(f"    Spent: ${client.session.get_total_spent():.6f}")
    print()

    # --- Test 2: Loop until budget runs out ---
    print("[2] Looping until budget is exhausted...")
    call_count = 0
    for i in range(200):
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Reply with the number {i}"}],
            max_tokens=10,
        )
        if result is None:
            print(f"    Stopped after {call_count} calls")
            break
        call_count += 1
        s = client.session.get_summary()
        print(f"    Call {call_count}: spent=${s['spent']:.6f}  remaining=${s['remaining']:.6f}")

    # --- Summary ---
    print("\n=== Final Summary ===")
    summary = client.session.get_summary()
    for k, v in summary.items():
        if k == "utilization_percent":
            print(f"  {k}: {v:.2f}%")
        else:
            print(f"  {k}: ${v:.6f}")


if __name__ == "__main__":
    main()
