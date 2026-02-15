"""Agent Budget - Budget-limited LLM API client wrapper.

Prevents runaway AI agents from burning money by enforcing hard dollar
budget limits on OpenAI API calls.

Example:
    >>> from openai import OpenAI
    >>> from agent_budget import BudgetedSession, BudgetExceededError
    >>>
    >>> # Create session with $5 budget
    >>> session = BudgetedSession(budget_usd=5.00)
    >>>
    >>> # Wrap OpenAI client
    >>> client = session.wrap_openai(OpenAI())
    >>>
    >>> # Make API calls - budget automatically enforced
    >>> try:
    ...     response = client.chat.completions.create(
    ...         model="gpt-4o-mini",
    ...         messages=[{"role": "user", "content": "Hello!"}]
    ...     )
    ... except BudgetExceededError as e:
    ...     print(f"Budget exceeded: {e}")
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
