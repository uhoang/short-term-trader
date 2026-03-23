"""Tests for Catalyst Capture strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.base import Direction
from signals.catalyst import CatalystCapture, CatalystCaptureConfig


def _make_features(n: int = 200) -> pd.DataFrame:
    """Create synthetic features with known catalyst conditions."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-02", periods=n)
    close = 100 + np.cumsum(np.random.normal(0, 1, n))

    df = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.random.randint(2_000_000, 10_000_000, n).astype(float),
            "event_score": 0.1,  # Default low
            "atr_5": 0.02,
            "atr_20": 0.02,
            "bb_width": 0.05,
        },
        index=dates,
    )
    return df


class TestCatalystCapture:
    def test_no_signal_on_low_event_score(self) -> None:
        strategy = CatalystCapture()
        features = _make_features()
        signals = strategy.scan(features, "NVDA")
        assert len(signals) == 0

    def test_signal_fires_on_conditions(self) -> None:
        strategy = CatalystCapture()
        features = _make_features()

        # Set up conditions on specific rows (after warmup)
        idx = 150
        features.iloc[idx, features.columns.get_loc("event_score")] = 0.8
        features.iloc[idx, features.columns.get_loc("atr_5")] = 0.05
        features.iloc[idx, features.columns.get_loc("atr_20")] = 0.02
        # BB width must be > 1.3x the 6-month median; set high value
        features.iloc[idx, features.columns.get_loc("bb_width")] = 0.20

        signals = strategy.scan(features, "NVDA")
        assert len(signals) >= 1

        signal = signals[0]
        assert signal.ticker == "NVDA"
        assert signal.direction == Direction.LONG
        assert signal.strategy_id == "catalyst_capture"
        assert 0.0 < signal.strength <= 1.0
        assert "trade_params" in signal.metadata

    def test_custom_config(self) -> None:
        config = CatalystCaptureConfig(
            event_score_min=0.5,
            atr_ratio_min=2.0,
            stop_loss_pct=0.05,
        )
        strategy = CatalystCapture(config=config)
        assert strategy.config.event_score_min == 0.5
        assert strategy.config.stop_loss_pct == 0.05

    def test_strength_capped_at_1(self) -> None:
        strategy = CatalystCapture()
        features = _make_features()

        idx = 150
        features.iloc[idx, features.columns.get_loc("event_score")] = 1.5
        features.iloc[idx, features.columns.get_loc("atr_5")] = 0.05
        features.iloc[idx, features.columns.get_loc("atr_20")] = 0.02
        features.iloc[idx, features.columns.get_loc("bb_width")] = 0.20

        signals = strategy.scan(features, "NVDA")
        for s in signals:
            assert s.strength <= 1.0
