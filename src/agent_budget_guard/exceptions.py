"""Custom exceptions for agent-budget-guard library."""


class BudgetError(Exception):
    """Base exception for budget-related errors."""

    pass


class BudgetExceededError(BudgetError):
    """Raised when a request would exceed the budget limit.

    This exception is raised BEFORE the API call is made, preventing
    budget overruns in real-time.

    Attributes:
        estimated_cost: The estimated cost of the request that would exceed the budget
        remaining: The remaining budget available when the error was raised
    """

    def __init__(self, message: str, estimated_cost: float, remaining: float) -> None:
        """Initialize BudgetExceededError.

        Args:
            message: Human-readable error message
            estimated_cost: Estimated cost of the blocked request in USD
            remaining: Remaining budget in USD
        """
        super().__init__(message)
        self.estimated_cost = estimated_cost
        self.remaining = remaining


class PricingDataError(BudgetError):
    """Raised when pricing data is missing or invalid.

    This typically occurs when:
    - pricing.json is malformed or missing
    - A model is used that isn't in the pricing configuration
    - Pricing data cannot be loaded from disk
    """

    pass
