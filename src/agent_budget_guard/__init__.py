"""Agent Budget Guard - Budget-limited LLM API client wrapper.

Prevents runaway AI agents from burning money by enforcing hard dollar
budget limits on LLM API calls (OpenAI, Anthropic, Google Gemini).

Example:
    >>> from agent_budget_guard import BudgetedSession
    >>>
    >>> # OpenAI
    >>> client = BudgetedSession.openai(budget_usd=5.00)
    >>> response = client.chat.completions.create(
    ...     model="gpt-4o-mini",
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )
    >>>
    >>> # Anthropic (requires: pip install agent-budget-guard[anthropic])
    >>> client = BudgetedSession.anthropic(budget_usd=5.00)
    >>> response = client.messages.create(
    ...     model="claude-sonnet-4-6",
    ...     max_tokens=1024,
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ... )
    >>>
    >>> # Google Gemini (requires: pip install agent-budget-guard[google])
    >>> client = BudgetedSession.google(budget_usd=5.00)
    >>> response = client.models.generate_content(
    ...     model="gemini-2.0-flash",
    ...     contents="Hello!",
    ... )
    >>> print(client.session.get_summary())
"""

from .session import BudgetedSession
from .exceptions import BudgetError, BudgetExceededError, PricingDataError

__version__ = "0.2.0"

__all__ = [
    "BudgetedSession",
    "BudgetError",
    "BudgetExceededError",
    "PricingDataError",
]
