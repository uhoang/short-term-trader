"""Tests for Sector Momentum / Pairs strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.base import Direction
from signals.momentum_pairs import MomentumPairsConfig, SectorMomentumPairs


def _make_sector_features(n_tickers: int = 8, n_days: int = 100) -> dict[str, pd.DataFrame]:
    """Create synthetic sector features with known momentum rankings."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    features: dict[str, pd.DataFrame] = {}

    for i in range(n_tickers):
        # Ticker i has momentum proportional to i (higher = stronger)
        mom = 0.01 * (i - n_tickers / 2)  # Range from negative to positive
        close = 100 + np.cumsum(np.random.normal(mom, 0.5, n_days))

        features[f"TICK{i}"] = pd.DataFrame(
            {
                "Close": close,
                "vol_adj_mom_60": mom + np.random.normal(0, 0.001, n_days),
                "sector_zscore": (i - n_tickers / 2) / 2 + np.random.normal(0, 0.1, n_days),
            },
            index=dates,
        )

    return features


class TestSectorMomentumPairs:
    def test_generates_long_and_short_signals(self) -> None:
        strategy = SectorMomentumPairs()
        sector_features = _make_sector_features()
        signals = strategy.scan_sector(sector_features, "semiconductors")

        assert len(signals) > 0

        longs = [s for s in signals if s.direction == Direction.LONG]
        shorts = [s for s in signals if s.direction == Direction.SHORT]
        assert len(longs) > 0
        assert len(shorts) > 0

    def test_top_tickers_are_long(self) -> None:
        strategy = SectorMomentumPairs(config=MomentumPairsConfig(top_n=2, bottom_n=2))
        sector_features = _make_sector_features()
        signals = strategy.scan_sector(sector_features, "semiconductors")

        # First rebalance signals
        first_date = signals[0].timestamp
        first_signals = [s for s in signals if s.timestamp == first_date]

        longs = [s for s in first_signals if s.direction == Direction.LONG]
        shorts = [s for s in first_signals if s.direction == Direction.SHORT]

        # Top momentum tickers should be in longs
        long_tickers = {s.ticker for s in longs}
        short_tickers = {s.ticker for s in shorts}
        assert len(long_tickers & short_tickers) == 0  # No overlap

    def test_rebalance_interval(self) -> None:
        config = MomentumPairsConfig(rebalance_days=15, top_n=2, bottom_n=2)
        strategy = SectorMomentumPairs(config=config)
        sector_features = _make_sector_features(n_days=100)
        signals = strategy.scan_sector(sector_features, "tech")

        # Should have signals at multiple rebalance dates
        unique_dates = {s.timestamp for s in signals}
        assert len(unique_dates) >= 2  # At least 2 rebalance points in 100 days

    def test_single_ticker_scan_returns_empty(self) -> None:
        strategy = SectorMomentumPairs()
        df = pd.DataFrame(
            {"Close": [100], "vol_adj_mom_60": [0.01]},
            index=pd.bdate_range("2023-01-02", periods=1),
        )
        signals = strategy.scan(df, "TICK0")
        assert len(signals) == 0

    def test_too_few_tickers(self) -> None:
        strategy = SectorMomentumPairs()
        # Only 2 tickers but need 3 long + 3 short
        features = _make_sector_features(n_tickers=2)
        signals = strategy.scan_sector(features, "tech")
        assert len(signals) == 0

    def test_metadata_contains_rank(self) -> None:
        strategy = SectorMomentumPairs()
        sector_features = _make_sector_features()
        signals = strategy.scan_sector(sector_features, "semiconductors")

        for s in signals:
            assert "rank" in s.metadata
            assert "momentum" in s.metadata
            assert "sector" in s.metadata
