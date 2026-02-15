"""Concurrent test: startup founders debating with a shared budget.

Three startup founders (AI, Hardware, Open Source) pitch ideas and
roast each other in a shared conversation. All share one budget —
when it runs out, the meeting ends mid-sentence.

Tests:
- Real-time progress output (printed as each call completes)
- Thread-safe budget enforcement across concurrent agents
- Graceful handling of rate limits and API errors
- Multi-turn conversation with growing context (increasing token costs)
"""

import os
import sys
import threading
import time
import concurrent.futures
from pathlib import Path

from agent_budget import BudgetedSession, BudgetExceededError

# Load .env from project root
env_file = Path(__file__).resolve().parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

# Thread-safe print and shared conversation
print_lock = threading.Lock()
conversation_lock = threading.Lock()
conversation = []  # shared message history all agents can see
debate_over = threading.Event()  # signal all agents to stop


FOUNDERS = {
    0: {
        "name": "Zara (AI Startup)",
        "system": "You are Zara, a confident AI startup founder who believes AI agents will replace every SaaS tool. "
                  "You speak in 2-3 sentences. You hype up AI but also roast the other founders' ideas. "
                  "Be witty and direct. Respond to what others have said.",
    },
    1: {
        "name": "Dev (Hardware Guy)",
        "system": "You are Dev, a hardware startup founder who thinks software people are delusional and "
                  "the real money is in chips and devices. You speak in 2-3 sentences. "
                  "You talk about margins, atoms vs bits, and roast software founders. "
                  "Be sarcastic and direct. Respond to what others have said.",
    },
    2: {
        "name": "Amina (Open Source)",
        "system": "You are Amina, an open source startup founder who believes proprietary software is a scam "
                  "and community-driven development always wins. You speak in 2-3 sentences. "
                  "You reference open source wins (Linux, PostgreSQL) and roast VC-funded hype. "
                  "Be sharp and direct. Respond to what others have said.",
    },
}


def log(msg):
    with print_lock:
        print(msg, flush=True)


def print_status(session):
    s = session.get_summary()
    log(
        f"         [budget] spent=${s['spent']:.6f}  "
        f"reserved=${s['reserved']:.6f}  "
        f"remaining=${s['remaining']:.6f}  "
        f"util={s['utilization_percent']:.1f}%"
    )


def founder_worker(client, session, agent_id, turn_event, next_event, rounds):
    """A founder agent that reads the shared conversation and adds to it."""
    phil = FOUNDERS[agent_id]
    calls = 0

    for round_num in range(rounds):
        # Wait for our turn
        turn_event.wait()
        turn_event.clear()

        if debate_over.is_set():
            break

        # Build messages: system prompt + shared conversation so far
        with conversation_lock:
            history = list(conversation)

        messages = [{"role": "system", "content": phil["system"]}]
        messages.extend(history)

        if not history:
            messages.append({
                "role": "user",
                "content": "You're at a startup meetup. Pitch your startup idea and explain why your approach is the future."
            })

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=150,
            )
            calls += 1
            reply = response.choices[0].message.content.strip()

            # Add to shared conversation
            with conversation_lock:
                conversation.append({
                    "role": "assistant",
                    "content": f"[{phil['name']}]: {reply}"
                })

            log(f"\n  {phil['name']} (turn {calls}):")
            log(f"    \"{reply}\"")
            print_status(session)

        except BudgetExceededError as e:
            log(f"\n  {phil['name']} | BUDGET HIT after {calls} calls "
                f"(need ${e.estimated_cost:.6f}, have ${e.remaining:.6f})")
            debate_over.set()
            next_event.set()  # unblock next agent so it can exit
            break
        except Exception as e:
            log(f"  {phil['name']} | ERROR: {type(e).__name__}: {e}")
            time.sleep(1)

        # Signal next philosopher's turn
        next_event.set()

    # Make sure next agent isn't stuck waiting
    next_event.set()
    return agent_id, calls


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in .env or shell:")
        print("  echo 'OPENAI_API_KEY=sk-...' > .env")
        sys.exit(1)

    num_agents = len(FOUNDERS)
    rounds = 12  # 12 rounds = 4 turns each (3 founders taking turns)
    budget = 0.02
    print(f"=== Startup Roast ({num_agents} founders, shared ${budget} budget, model=gpt-4o) ===")
    print(f"    {rounds} rounds of pitching and roasting\n")

    # One-liner setup with warnings
    client = BudgetedSession.openai(
        budget_usd=budget,
        on_warning=lambda w: log(f"\n    *** WARNING: {w['threshold']}% budget used ***"),
    )
    session = client.session

    print_status(session)

    # Turn-based signaling: each agent waits on its event, then signals the next
    turn_events = [threading.Event() for _ in range(num_agents)]
    turn_events[0].set()  # first philosopher starts

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_agents) as pool:
        futures = {
            pool.submit(
                founder_worker, client, session, i,
                turn_events[i], turn_events[(i + 1) % num_agents], rounds
            ): i
            for i in range(num_agents)
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                agent_id, calls = future.result()
                results.append((agent_id, calls))
            except Exception as e:
                log(f"  Agent future failed: {e}")

    print(f"\n{'='*60}")
    print(f"=== Results ===")
    total_calls = 0
    for agent_id, calls in sorted(results):
        print(f"  {FOUNDERS[agent_id]['name']}: {calls} contributions")
        total_calls += calls
    print(f"  Total API calls: {total_calls}")

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
        print(f"\n  PASS: Budget respected! Spent ${total_spent:.6f} out of ${total_budget:.6f}")
    else:
        print(f"\n  FAIL: Overspent! ${total_spent:.6f} > ${total_budget:.6f}")


if __name__ == "__main__":
    main()
