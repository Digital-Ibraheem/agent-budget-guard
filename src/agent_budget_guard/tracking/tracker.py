"""Thread-safe budget tracking with reservation system."""

import threading
import uuid
from typing import Dict

from ..exceptions import BudgetExceededError


class SpendTracker:
    """Thread-safe budget tracker with reservation system.

    Prevents race conditions where multiple threads could exceed the budget
    by using a reservation system: budget is "reserved" during API calls
    and then committed or rolled back based on success/failure.

    This ensures that even with concurrent API calls, the budget is never
    exceeded.

    Attributes:
        _budget: Total budget in USD
        _spent: Amount actually spent so far
        _reserved: Amount currently reserved (pending API calls)
        _reservations: Map of reservation_id -> reserved amount
        _lock: Threading lock for atomic operations
    """

    def __init__(self, budget_usd: float) -> None:
        """Initialize SpendTracker.

        Args:
            budget_usd: Total budget in USD (e.g., 5.00 for $5)

        Raises:
            ValueError: If budget is negative
        """
        if budget_usd < 0:
            raise ValueError("Budget cannot be negative")

        self._budget = float(budget_usd)
        self._spent = 0.0
        self._reserved = 0.0
        self._reservations: Dict[str, float] = {}
        self._lock = threading.Lock()

    def check_and_reserve(self, estimated_cost: float) -> str:
        """Atomically check budget and reserve funds for an API call.

        This is the critical operation that prevents race conditions.
        Multiple threads calling this simultaneously will be serialized
        by the lock, ensuring only one can reserve at a time.

        Args:
            estimated_cost: Estimated cost of the API call in USD

        Returns:
            Reservation ID (UUID) to use for commit/rollback

        Raises:
            BudgetExceededError: If estimated cost would exceed remaining budget
        """
        with self._lock:  # ATOMIC OPERATION - prevents race conditions
            # Calculate remaining budget considering both spent and reserved
            remaining = self._budget - self._spent - self._reserved

            if estimated_cost > remaining:
                raise BudgetExceededError(
                    f"Estimated cost ${estimated_cost:.6f} would exceed "
                    f"remaining budget ${remaining:.6f}",
                    estimated_cost=estimated_cost,
                    remaining=remaining
                )

            # Reserve the budget
            reservation_id = str(uuid.uuid4())
            self._reserved += estimated_cost
            self._reservations[reservation_id] = estimated_cost

            return reservation_id

    def commit(self, reservation_id: str, actual_cost: float) -> None:
        """Commit a reservation and record the actual cost.

        Called after a successful API call to convert the reservation
        into actual spend.

        Args:
            reservation_id: Reservation ID from check_and_reserve()
            actual_cost: Actual cost from the API response in USD

        Raises:
            ValueError: If reservation_id not found
        """
        with self._lock:
            if reservation_id not in self._reservations:
                raise ValueError(f"Reservation {reservation_id} not found")

            # Release the reservation and record actual spend
            reserved_amount = self._reservations.pop(reservation_id)
            self._reserved -= reserved_amount
            self._spent += actual_cost

    def rollback(self, reservation_id: str) -> None:
        """Rollback a reservation after a failed API call.

        Called when an API call fails or is cancelled to release
        the reserved budget without recording any spend.

        Args:
            reservation_id: Reservation ID from check_and_reserve()

        Note:
            Does not raise an error if reservation not found (idempotent)
        """
        with self._lock:
            if reservation_id in self._reservations:
                reserved_amount = self._reservations.pop(reservation_id)
                self._reserved -= reserved_amount

    def get_spent(self) -> float:
        """Get the total amount spent so far.

        Returns:
            Amount spent in USD (not including pending reservations)
        """
        with self._lock:
            return self._spent

    def get_remaining(self) -> float:
        """Get the remaining budget available.

        This accounts for both spent and currently reserved amounts.

        Returns:
            Remaining budget in USD
        """
        with self._lock:
            return self._budget - self._spent - self._reserved

    def get_budget(self) -> float:
        """Get the total budget.

        Returns:
            Total budget in USD
        """
        with self._lock:
            return self._budget

    def get_reserved(self) -> float:
        """Get the total amount currently reserved.

        Returns:
            Amount reserved in USD (pending API calls)
        """
        with self._lock:
            return self._reserved

    def reset(self) -> None:
        """Reset spent and reservations to zero.

        WARNING: This does not cancel in-flight API calls. Only use this
        when you're sure no calls are pending.
        """
        with self._lock:
            self._spent = 0.0
            self._reserved = 0.0
            self._reservations.clear()
