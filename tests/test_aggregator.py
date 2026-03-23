"""Tests for signal aggregation and conflict resolution."""

from __future__ import annotations

from datetime import datetime

from signals.aggregator import SignalBus
from signals.base import Direction, SignalEvent


def _signal(
    ticker: str = "NVDA",
    direction: Direction = Direction.LONG,
    strength: float = 0.7,
    strategy_id: str = "test",
) -> SignalEvent:
    return SignalEvent(
        ticker=ticker,
        direction=direction,
        strength=strength,
        strategy_id=strategy_id,
        timestamp=datetime(2024, 1, 15),
    )


class TestSignalBus:
    def test_emit_and_resolve_single(self) -> None:
        bus = SignalBus()
        bus.emit(_signal())
        resolved = bus.resolve()
        assert len(resolved) == 1
        assert resolved[0].ticker == "NVDA"

    def test_conflict_suppresses_both(self) -> None:
        """Opposite directions on same ticker should suppress both."""
        bus = SignalBus()
        bus.emit(_signal(direction=Direction.LONG, strategy_id="catalyst"))
        bus.emit(_signal(direction=Direction.SHORT, strategy_id="breakout"))
        resolved = bus.resolve()
        assert len(resolved) == 0

    def test_same_direction_merges(self) -> None:
        """Same direction signals are merged with averaged strength."""
        bus = SignalBus()
        bus.emit(_signal(strength=0.8, strategy_id="catalyst"))
        bus.emit(_signal(strength=0.6, strategy_id="mean_rev"))
        resolved = bus.resolve()
        assert len(resolved) == 1
        assert resolved[0].strength == 0.7  # Average of 0.8 and 0.6
        assert "catalyst" in resolved[0].strategy_id
        assert "mean_rev" in resolved[0].strategy_id

    def test_different_tickers_independent(self) -> None:
        bus = SignalBus()
        bus.emit(_signal(ticker="NVDA"))
        bus.emit(_signal(ticker="AMD"))
        resolved = bus.resolve()
        assert len(resolved) == 2

    def test_max_positions_limit(self) -> None:
        bus = SignalBus(max_concurrent_positions=2)
        bus.set_open_positions({"AAPL"})  # 1 existing position
        bus.emit(_signal(ticker="NVDA", strength=0.9))
        bus.emit(_signal(ticker="AMD", strength=0.7))
        bus.emit(_signal(ticker="MSFT", strength=0.5))
        resolved = bus.resolve()
        assert len(resolved) == 1  # Only 1 slot available
        assert resolved[0].ticker == "NVDA"  # Highest strength wins

    def test_skip_existing_positions(self) -> None:
        bus = SignalBus()
        bus.set_open_positions({"NVDA"})
        bus.emit(_signal(ticker="NVDA"))
        resolved = bus.resolve()
        assert len(resolved) == 0

    def test_sorted_by_strength(self) -> None:
        bus = SignalBus()
        bus.emit(_signal(ticker="AMD", strength=0.5))
        bus.emit(_signal(ticker="NVDA", strength=0.9))
        bus.emit(_signal(ticker="MSFT", strength=0.7))
        resolved = bus.resolve()
        assert [s.ticker for s in resolved] == ["NVDA", "MSFT", "AMD"]

    def test_clear(self) -> None:
        bus = SignalBus()
        bus.emit(_signal())
        bus.clear()
        assert len(bus.signals) == 0

    def test_emit_batch(self) -> None:
        bus = SignalBus()
        bus.emit_batch([_signal(ticker="A"), _signal(ticker="B")])
        assert len(bus.signals) == 2

    def test_all_slots_full(self) -> None:
        bus = SignalBus(max_concurrent_positions=3)
        bus.set_open_positions({"A", "B", "C"})
        bus.emit(_signal(ticker="D"))
        resolved = bus.resolve()
        assert len(resolved) == 0
