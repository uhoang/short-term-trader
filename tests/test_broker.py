"""Tests for Alpaca broker client (mocked — no real API calls)."""

from __future__ import annotations

from datetime import datetime

from signals.base import Direction, SignalEvent
from signals.strategy import TradeParams


def _make_signal() -> SignalEvent:
    return SignalEvent(
        ticker="NVDA",
        direction=Direction.LONG,
        strength=0.8,
        strategy_id="test",
        timestamp=datetime(2024, 1, 15),
        metadata={
            "trade_params": TradeParams(
                entry_price=150.0,
                stop_loss_pct=0.07,
                take_profit_pct=0.15,
                max_hold_days=10,
            ),
            "sector": "semiconductors",
        },
    )


class TestAlpacaBrokerInit:
    def test_broker_imports(self) -> None:
        """Verify the broker module can be imported."""
        from live.broker import AlpacaBroker  # noqa: F401

        assert AlpacaBroker is not None


class TestPositionTracker:
    def test_open_and_close(self) -> None:
        from live.positions import PositionTracker

        tracker = PositionTracker(init_equity=100_000)
        signal = _make_signal()
        tracker.open_position(signal, fill_price=150.0, shares=100)

        assert "NVDA" in tracker.get_open()
        assert len(tracker.get_open()) == 1

        trade = tracker.close_position("NVDA", fill_price=160.0, reason="take_profit")
        assert trade is not None
        assert trade["pnl"] > 0
        assert len(tracker.get_open()) == 0
        assert len(tracker.get_closed()) == 1

    def test_portfolio_state(self) -> None:
        from live.positions import PositionTracker

        tracker = PositionTracker(init_equity=100_000)
        signal = _make_signal()
        tracker.open_position(signal, fill_price=150.0, shares=100)

        state = tracker.get_portfolio_state({"NVDA": 155.0})
        assert state.equity > 0
        assert "NVDA" in state.open_positions

    def test_daily_pnl(self) -> None:
        from live.positions import PositionTracker

        tracker = PositionTracker(init_equity=100_000)
        signal = _make_signal()
        tracker.open_position(signal, fill_price=150.0, shares=100)

        pnl = tracker.daily_pnl({"NVDA": 155.0})
        assert pnl["unrealized"] == 500.0  # (155-150) * 100

    def test_save_and_load(self, tmp_path) -> None:
        from live.positions import PositionTracker

        tracker = PositionTracker(init_equity=100_000)
        signal = _make_signal()
        tracker.open_position(signal, fill_price=150.0, shares=100)

        path = tmp_path / "positions.json"
        tracker.save(path)

        tracker2 = PositionTracker()
        tracker2.load(path)
        assert "NVDA" in tracker2.get_open()

    def test_close_nonexistent(self) -> None:
        from live.positions import PositionTracker

        tracker = PositionTracker()
        result = tracker.close_position("FAKE", 100.0)
        assert result is None
