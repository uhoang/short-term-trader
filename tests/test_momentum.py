"""Tests for momentum and mean reversion feature computations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.features.momentum import (
    compute_52w_proximity,
    compute_all_momentum_features,
    compute_rsi,
    compute_sector_relative_rsi,
    compute_sector_zscore,
    compute_vol_adjusted_momentum,
    compute_vwap_deviation,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Create synthetic OHLCV data."""
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range("2023-01-02", periods=n)
    returns = np.random.normal(0.0005, 0.02, n)
    close = 100 * np.exp(np.cumsum(returns))
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)

    return pd.DataFrame(
        {
            "Open": close * (1 + np.random.normal(0, 0.005, n)),
            "High": close * (1 + np.abs(np.random.normal(0, 0.01, n))),
            "Low": close * (1 - np.abs(np.random.normal(0, 0.01, n))),
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


class TestRSI:
    def test_rsi_bounds(self, sample_ohlcv: pd.DataFrame) -> None:
        rsi = compute_rsi(sample_ohlcv["Close"])
        valid = rsi.dropna()
        assert (valid >= 0).all(), "RSI should be >= 0"
        assert (valid <= 100).all(), "RSI should be <= 100"

    def test_rsi_name(self, sample_ohlcv: pd.DataFrame) -> None:
        rsi = compute_rsi(sample_ohlcv["Close"])
        assert rsi.name == "rsi_14"

    def test_rsi_around_50_for_random_walk(self, sample_ohlcv: pd.DataFrame) -> None:
        rsi = compute_rsi(sample_ohlcv["Close"]).dropna()
        # Random walk RSI should center near 50
        assert 30 < rsi.mean() < 70


class TestVolAdjMomentum:
    def test_output_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        mom = compute_vol_adjusted_momentum(sample_ohlcv["Close"])
        assert list(mom.columns) == ["vol_adj_mom_10", "vol_adj_mom_20", "vol_adj_mom_60"]

    def test_finite_values(self, sample_ohlcv: pd.DataFrame) -> None:
        mom = compute_vol_adjusted_momentum(sample_ohlcv["Close"])
        # After warmup period, values should be finite
        valid = mom.iloc[70:].dropna()
        assert np.isfinite(valid.values).all()


class TestVWAPDeviation:
    def test_centered_near_zero(self, sample_ohlcv: pd.DataFrame) -> None:
        vwap = compute_vwap_deviation(sample_ohlcv["Close"], sample_ohlcv["Volume"])
        valid = vwap.dropna()
        assert -0.10 < valid.mean() < 0.10

    def test_name(self, sample_ohlcv: pd.DataFrame) -> None:
        vwap = compute_vwap_deviation(sample_ohlcv["Close"], sample_ohlcv["Volume"])
        assert vwap.name == "vwap_dev"


class Test52WeekProximity:
    def test_output_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        prox = compute_52w_proximity(sample_ohlcv["Close"])
        expected = [
            "dist_52w_high",
            "dist_52w_low",
            "near_52w_high_2",
            "near_52w_high_5",
            "near_52w_high_10",
            "near_52w_low_2",
            "near_52w_low_5",
            "near_52w_low_10",
        ]
        for col in expected:
            assert col in prox.columns

    def test_high_distance_non_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        prox = compute_52w_proximity(sample_ohlcv["Close"])
        valid = prox["dist_52w_high"].dropna()
        assert (valid <= 0.001).all()  # Close can't exceed 52w high (small float tolerance)

    def test_low_distance_non_negative(self, sample_ohlcv: pd.DataFrame) -> None:
        prox = compute_52w_proximity(sample_ohlcv["Close"])
        valid = prox["dist_52w_low"].dropna()
        assert (valid >= -0.001).all()  # Close can't be below 52w low

    def test_flags_are_binary(self, sample_ohlcv: pd.DataFrame) -> None:
        prox = compute_52w_proximity(sample_ohlcv["Close"])
        for col in ["near_52w_high_2", "near_52w_low_5"]:
            valid = prox[col].dropna()
            assert set(valid.unique()).issubset({0, 1})


class TestSectorRelativeRSI:
    def test_relative_rsi_centered(self) -> None:
        np.random.seed(42)
        dates = pd.bdate_range("2023-01-02", periods=100)
        rsi_series = {
            "A": pd.Series(np.random.uniform(30, 70, 100), index=dates, name="rsi_14"),
            "B": pd.Series(np.random.uniform(30, 70, 100), index=dates, name="rsi_14"),
            "C": pd.Series(np.random.uniform(30, 70, 100), index=dates, name="rsi_14"),
        }
        sector_map = {"A": "tech", "B": "tech", "C": "tech"}

        result = compute_sector_relative_rsi(rsi_series, sector_map)
        assert len(result) == 3
        # Mean of relative RSI across sector should be near 0
        combined = pd.DataFrame({t: s for t, s in result.items()})
        assert abs(combined.mean(axis=1).mean()) < 5.0


class TestSectorZScore:
    def test_zscore_distribution(self) -> None:
        np.random.seed(42)
        dates = pd.bdate_range("2023-01-02", periods=200)
        returns_dict = {
            "A": pd.Series(np.random.normal(0, 0.02, 200), index=dates),
            "B": pd.Series(np.random.normal(0, 0.02, 200), index=dates),
            "C": pd.Series(np.random.normal(0, 0.02, 200), index=dates),
        }
        sector_map = {"A": "tech", "B": "tech", "C": "tech"}

        result = compute_sector_zscore(returns_dict, sector_map)
        assert len(result) == 3
        for series in result.values():
            valid = series.dropna()
            # Z-scores should be roughly standard normal
            assert abs(valid.mean()) < 1.0
            assert 0.3 < valid.std() < 3.0


class TestAllMomentumFeatures:
    def test_all_features_computed(self, sample_ohlcv: pd.DataFrame) -> None:
        features = compute_all_momentum_features(sample_ohlcv)
        expected = [
            "vol_adj_mom_10",
            "vol_adj_mom_20",
            "vol_adj_mom_60",
            "vwap_dev",
            "rsi_14",
            "dist_52w_high",
            "dist_52w_low",
        ]
        for col in expected:
            assert col in features.columns, f"Missing feature: {col}"
