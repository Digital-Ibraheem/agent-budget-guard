"""Test spend tracker functionality."""

import pytest
import threading
from agent_budget.tracking.tracker import SpendTracker
from agent_budget.exceptions import BudgetExceededError


def test_tracker_initialization():
    """Test tracker initializes correctly."""
    tracker = SpendTracker(budget_usd=10.0)
    assert tracker.get_budget() == 10.0
    assert tracker.get_spent() == 0.0
    assert tracker.get_remaining() == 10.0


def test_negative_budget_raises_error():
    """Test that negative budget raises ValueError."""
    with pytest.raises(ValueError):
        SpendTracker(budget_usd=-5.0)


def test_check_and_reserve():
    """Test budget reservation."""
    tracker = SpendTracker(budget_usd=10.0)

    # Reserve $3
    reservation_id = tracker.check_and_reserve(3.0)
    assert reservation_id is not None

    # Remaining should be $7
    assert tracker.get_remaining() == 7.0
    assert tracker.get_reserved() == 3.0


def test_commit_reservation():
    """Test committing a reservation."""
    tracker = SpendTracker(budget_usd=10.0)

    reservation_id = tracker.check_and_reserve(3.0)
    tracker.commit(reservation_id, actual_cost=2.5)

    # Should have spent $2.5, no reservations
    assert tracker.get_spent() == 2.5
    assert tracker.get_reserved() == 0.0
    assert tracker.get_remaining() == 7.5


def test_rollback_reservation():
    """Test rolling back a reservation."""
    tracker = SpendTracker(budget_usd=10.0)

    reservation_id = tracker.check_and_reserve(3.0)
    tracker.rollback(reservation_id)

    # Should have no spend, no reservations
    assert tracker.get_spent() == 0.0
    assert tracker.get_reserved() == 0.0
    assert tracker.get_remaining() == 10.0


def test_budget_exceeded_error():
    """Test that exceeding budget raises error."""
    tracker = SpendTracker(budget_usd=5.0)

    # Try to reserve more than budget
    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.check_and_reserve(6.0)

    assert exc_info.value.estimated_cost == 6.0
    assert exc_info.value.remaining == 5.0


def test_multiple_reservations():
    """Test multiple concurrent reservations."""
    tracker = SpendTracker(budget_usd=10.0)

    # Reserve $3, then $4
    res1 = tracker.check_and_reserve(3.0)
    res2 = tracker.check_and_reserve(4.0)

    # Should have $3 remaining
    assert tracker.get_remaining() == 3.0
    assert tracker.get_reserved() == 7.0

    # Third reservation for $5 should fail
    with pytest.raises(BudgetExceededError):
        tracker.check_and_reserve(5.0)

    # Commit first, should free up budget
    tracker.commit(res1, actual_cost=2.0)
    assert tracker.get_remaining() == 4.0


def test_thread_safety():
    """Test that tracker is thread-safe."""
    tracker = SpendTracker(budget_usd=10.0)
    successful_reservations = []
    failed_reservations = []

    def try_reserve():
        try:
            res_id = tracker.check_and_reserve(2.0)
            successful_reservations.append(res_id)
        except BudgetExceededError:
            failed_reservations.append(True)

    # Try to make 10 reservations of $2 each with only $10 budget
    # Only 5 should succeed
    threads = [threading.Thread(target=try_reserve) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly 5 should succeed (10 / 2 = 5)
    assert len(successful_reservations) == 5
    assert len(failed_reservations) == 5
    assert tracker.get_reserved() == 10.0
    assert tracker.get_remaining() == 0.0


def test_reset():
    """Test resetting tracker."""
    tracker = SpendTracker(budget_usd=10.0)

    res = tracker.check_and_reserve(3.0)
    tracker.commit(res, actual_cost=3.0)

    assert tracker.get_spent() == 3.0

    tracker.reset()

    assert tracker.get_spent() == 0.0
    assert tracker.get_reserved() == 0.0
    assert tracker.get_remaining() == 10.0
