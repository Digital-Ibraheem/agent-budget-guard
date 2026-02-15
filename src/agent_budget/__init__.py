"""Agent Budget - Budget-limited LLM API client wrapper.

Prevents runaway AI agents from burning money by enforcing hard dollar
budget limits on OpenAI API calls.

Example:
    >>> from agent_budget import BudgetedSession
    >>>
    >>> client = BudgetedSession.openai(budget_usd=5.00)
    >>> response = client.chat.completions.create(
    ...     model="gpt-4o-mini",
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )
    >>> print(client.session.get_summary())
"""

from .session import BudgetedSession
from .exceptions import BudgetError, BudgetExceededError, PricingDataError

__version__ = "0.1.0"

__all__ = [
    "BudgetedSession",
    "BudgetError",
    "BudgetExceededError",
    "PricingDataError",
]
