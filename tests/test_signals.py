"""Tests for signal base classes."""

from __future__ import annotations

import pytest

from signals.base import Direction, SignalEvent


class TestSignalEvent:
    def test_create_signal(self) -> None:
        signal = SignalEvent(
            ticker="NVDA",
            direction=Direction.LONG,
            strength=0.8,
            strategy_id="catalyst_capture",
        )
        assert signal.ticker == "NVDA"
        assert signal.direction == Direction.LONG
        assert signal.strength == 0.8

    def test_invalid_strength_raises(self) -> None:
        with pytest.raises(ValueError, match="Signal strength must be 0-1"):
            SignalEvent(
                ticker="NVDA",
                direction=Direction.LONG,
                strength=1.5,
                strategy_id="test",
            )
