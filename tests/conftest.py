"""Shared test fixtures."""

from __future__ import annotations

import pytest

from scanner.universe import Universe


@pytest.fixture
def universe() -> Universe:
    """Load the default universe config."""
    return Universe()
