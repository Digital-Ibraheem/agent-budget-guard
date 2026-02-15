"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def small_budget():
    """Fixture for small budget (useful for testing budget exceeded)."""
    return 0.01


@pytest.fixture
def medium_budget():
    """Fixture for medium budget (useful for normal tests)."""
    return 5.00


@pytest.fixture
def large_budget():
    """Fixture for large budget (useful for integration tests)."""
    return 20.00
