"""Tests for Volatility Breakout strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.base import Direction
from signals.breakout import BreakoutConfig, VolatilityBreakout


def _make_features(n: int = 200) -> pd.DataFrame:
    """Create synthetic features for breakout testing."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-02", periods=n)
    close = 100 + np.cumsum(np.random.normal(0, 1, n))

    df = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(n, 3_000_000.0),
            "bb_width": 0.08,  # Normal BB width
            "bb_pct_b": 0.5,  # Middle of band
            "atr_5": 0.02,
            "atr_20": 0.02,
        },
        index=dates,
    )
    return df


class TestVolatilityBreakout:
    def test_no_signal_in_normal_conditions(self) -> None:
        strategy = VolatilityBreakout()
        features = _make_features()
        signals = strategy.scan(features, "AMD")
        assert len(signals) == 0

    def test_long_breakout_signal(self) -> None:
        strategy = VolatilityBreakout()
        features = _make_features()

        # Set up squeeze then breakout at row 150
        # First create a squeeze (low bb_width)
        for i in range(60, 150):
            features.iloc[i, features.columns.get_loc("bb_width")] = 0.02

        # Breakout row: bb_width still at squeeze level, but high ATR ratio + volume
        idx = 150
        features.iloc[idx, features.columns.get_loc("bb_width")] = 0.02
        features.iloc[idx, features.columns.get_loc("bb_pct_b")] = 1.2  # Above upper band
        features.iloc[idx, features.columns.get_loc("atr_5")] = 0.04
        features.iloc[idx, features.columns.get_loc("atr_20")] = 0.02
        features.iloc[idx, features.columns.get_loc("Volume")] = 8_000_000.0

        signals = strategy.scan(features, "AMD")
        assert len(signals) >= 1
        assert signals[0].direction == Direction.LONG

    def test_short_breakout_signal(self) -> None:
        strategy = VolatilityBreakout()
        features = _make_features()

        for i in range(60, 150):
            features.iloc[i, features.columns.get_loc("bb_width")] = 0.02

        idx = 150
        features.iloc[idx, features.columns.get_loc("bb_width")] = 0.02
        features.iloc[idx, features.columns.get_loc("bb_pct_b")] = -0.2  # Below lower band
        features.iloc[idx, features.columns.get_loc("atr_5")] = 0.04
        features.iloc[idx, features.columns.get_loc("atr_20")] = 0.02
        features.iloc[idx, features.columns.get_loc("Volume")] = 8_000_000.0

        signals = strategy.scan(features, "AMD")
        assert len(signals) >= 1
        assert signals[0].direction == Direction.SHORT

    def test_no_signal_without_volume(self) -> None:
        strategy = VolatilityBreakout()
        features = _make_features()

        for i in range(60, 150):
            features.iloc[i, features.columns.get_loc("bb_width")] = 0.02

        idx = 150
        features.iloc[idx, features.columns.get_loc("bb_width")] = 0.02
        features.iloc[idx, features.columns.get_loc("bb_pct_b")] = 1.2
        features.iloc[idx, features.columns.get_loc("atr_5")] = 0.04
        features.iloc[idx, features.columns.get_loc("atr_20")] = 0.02
        # Volume stays at average — no spike
        signals = strategy.scan(features, "AMD")
        assert len(signals) == 0

    def test_custom_config(self) -> None:
        config = BreakoutConfig(atr_ratio_min=1.5, stop_loss_pct=0.04)
        strategy = VolatilityBreakout(config=config)
        assert strategy.config.atr_ratio_min == 1.5
