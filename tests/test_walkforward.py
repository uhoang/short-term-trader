"""Tests for walk-forward validation framework."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.walkforward import WalkForward, WalkForwardConfig
from signals.base import Direction, SignalEvent
from signals.strategy import TradeParams


class TestWindowGeneration:
    def test_generates_windows(self) -> None:
        wf = WalkForward(config=WalkForwardConfig(is_months=18, oos_months=6, step_months=3))
        windows = wf.generate_windows("2018-01-01", "2025-01-01")
        assert len(windows) > 0

        # Check first window
        is_start, is_end, oos_start, oos_end = windows[0]
        assert is_start == pd.Timestamp("2018-01-01")
        assert oos_start == is_end  # OOS starts where IS ends

    def test_no_overlap_between_windows(self) -> None:
        wf = WalkForward(config=WalkForwardConfig(is_months=12, oos_months=6, step_months=6))
        windows = wf.generate_windows("2020-01-01", "2024-01-01")

        for i in range(len(windows)):
            is_start, is_end, oos_start, oos_end = windows[i]
            assert oos_start == is_end  # No gap between IS and OOS
            assert oos_end > oos_start

    def test_short_period_no_windows(self) -> None:
        wf = WalkForward(config=WalkForwardConfig(is_months=18, oos_months=6))
        windows = wf.generate_windows("2024-01-01", "2024-06-01")
        assert len(windows) == 0  # Period too short

    def test_step_size_affects_count(self) -> None:
        wf_3m = WalkForward(config=WalkForwardConfig(step_months=3))
        wf_6m = WalkForward(config=WalkForwardConfig(step_months=6))

        windows_3m = wf_3m.generate_windows("2018-01-01", "2025-01-01")
        windows_6m = wf_6m.generate_windows("2018-01-01", "2025-01-01")
        assert len(windows_3m) > len(windows_6m)


class TestWalkForwardRun:
    def test_runs_with_simple_scan_fn(self) -> None:
        """Test walk-forward with a trivial scan function."""
        np.random.seed(42)
        dates = pd.bdate_range("2020-01-02", periods=800)
        close = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 800)))

        prices = {
            "AAPL": pd.DataFrame(
                {
                    "Open": close * 0.999,
                    "High": close * 1.01,
                    "Low": close * 0.99,
                    "Close": close,
                    "Volume": np.full(800, 5e6),
                },
                index=dates,
            )
        }
        features = {"AAPL": prices["AAPL"].copy()}

        def scan_fn(feats: dict[str, pd.DataFrame], params: dict) -> list[SignalEvent]:
            """Simple scan: buy every 20th day."""
            signals = []
            for ticker, df in feats.items():
                for i in range(0, len(df), 20):
                    signals.append(
                        SignalEvent(
                            ticker=ticker,
                            direction=Direction.LONG,
                            strength=0.7,
                            strategy_id="test",
                            timestamp=df.index[i].to_pydatetime(),
                            metadata={
                                "trade_params": TradeParams(
                                    entry_price=100,
                                    stop_loss_pct=0.07,
                                    take_profit_pct=0.15,
                                    max_hold_days=10,
                                ),
                            },
                        )
                    )
            return signals

        wf = WalkForward(config=WalkForwardConfig(is_months=12, oos_months=6, step_months=6))
        results = wf.run(scan_fn, prices, features)

        assert len(results) > 0
        for wr in results:
            assert wr.oos_start > wr.is_start
            assert wr.regime in ("risk_on", "risk_off", "high_vol", "low_vol", "unknown")

    def test_regime_classification(self) -> None:
        wf = WalkForward()
        # High volatility prices
        np.random.seed(42)
        dates = pd.bdate_range("2023-01-02", periods=60)
        volatile = 100 + np.cumsum(np.random.normal(0, 5, 60))  # High vol

        prices = {
            "X": pd.DataFrame(
                {"Close": volatile, "Open": volatile, "High": volatile, "Low": volatile},
                index=dates,
            )
        }
        regime = wf._classify_regime(prices, pd.Timestamp("2023-01-02"), pd.Timestamp("2023-04-01"))
        assert regime in ("risk_on", "risk_off", "high_vol", "low_vol", "unknown")
