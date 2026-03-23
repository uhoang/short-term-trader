"""Tests for Mean Reversion strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.base import Direction
from signals.mean_reversion import MeanReversion, MeanReversionConfig


def _make_features(n: int = 100) -> pd.DataFrame:
    """Create synthetic features for mean reversion testing."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-02", periods=n)
    close = 100 + np.cumsum(np.random.normal(0, 1, n))

    df = pd.DataFrame(
        {
            "Close": close,
            "vwap_dev": 0.0,  # At VWAP
            "rsi_14": 50.0,  # Neutral
        },
        index=dates,
    )
    return df


class TestMeanReversion:
    def test_no_signal_in_neutral(self) -> None:
        strategy = MeanReversion()
        features = _make_features()
        signals = strategy.scan(features, "MSFT")
        assert len(signals) == 0

    def test_signal_fires_on_oversold(self) -> None:
        strategy = MeanReversion()
        features = _make_features()

        # Set up oversold conditions
        idx = 50
        features.iloc[idx, features.columns.get_loc("vwap_dev")] = -0.03
        features.iloc[idx, features.columns.get_loc("rsi_14")] = 25.0

        signals = strategy.scan(features, "MSFT")
        assert len(signals) == 1
        assert signals[0].direction == Direction.LONG
        assert signals[0].strategy_id == "mean_reversion"

    def test_no_signal_rsi_too_high(self) -> None:
        strategy = MeanReversion()
        features = _make_features()

        idx = 50
        features.iloc[idx, features.columns.get_loc("vwap_dev")] = -0.03
        features.iloc[idx, features.columns.get_loc("rsi_14")] = 45.0  # Too high

        signals = strategy.scan(features, "MSFT")
        assert len(signals) == 0

    def test_strength_scales_with_deviation(self) -> None:
        strategy = MeanReversion()
        features = _make_features()

        # Mild oversold
        features.iloc[50, features.columns.get_loc("vwap_dev")] = -0.025
        features.iloc[50, features.columns.get_loc("rsi_14")] = 28.0

        # Deep oversold
        features.iloc[60, features.columns.get_loc("vwap_dev")] = -0.05
        features.iloc[60, features.columns.get_loc("rsi_14")] = 20.0

        signals = strategy.scan(features, "CRM")
        assert len(signals) == 2
        assert signals[1].strength > signals[0].strength

    def test_sector_etf_filter(self) -> None:
        strategy = MeanReversion()
        features = _make_features()

        idx = 50
        features.iloc[idx, features.columns.get_loc("vwap_dev")] = -0.03
        features.iloc[idx, features.columns.get_loc("rsi_14")] = 25.0

        # Sector ETF RSI too low (unhealthy sector)
        etf_rsi = pd.Series(30.0, index=features.index, name="rsi_14")
        signals = strategy.scan(features, "MSFT", sector_etf_rsi=etf_rsi)
        assert len(signals) == 0

        # Sector ETF RSI healthy
        etf_rsi_healthy = pd.Series(55.0, index=features.index, name="rsi_14")
        signals = strategy.scan(features, "MSFT", sector_etf_rsi=etf_rsi_healthy)
        assert len(signals) == 1

    def test_custom_config(self) -> None:
        config = MeanReversionConfig(
            vwap_dev_threshold=-0.03,
            rsi_threshold=28.0,
        )
        strategy = MeanReversion(config=config)
        assert strategy.config.vwap_dev_threshold == -0.03
