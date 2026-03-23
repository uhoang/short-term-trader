"""Tests for volatility feature computations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.features.volatility import (
    compute_all_volatility_features,
    compute_atr,
    compute_bollinger_bands,
    compute_garman_klass,
    compute_historical_volatility,
    compute_log_returns,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Create synthetic OHLCV data with known properties."""
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range("2023-01-02", periods=n)

    # Generate realistic price series with known volatility
    returns = np.random.normal(0.0005, 0.02, n)  # ~32% annualized vol
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_ = close * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestLogReturns:
    def test_returns_correct_length(self, sample_ohlcv: pd.DataFrame) -> None:
        ret = compute_log_returns(sample_ohlcv["Close"])
        assert len(ret) == len(sample_ohlcv)
        assert pd.isna(ret.iloc[0])  # First return is NaN

    def test_returns_reasonable_range(self, sample_ohlcv: pd.DataFrame) -> None:
        ret = compute_log_returns(sample_ohlcv["Close"]).dropna()
        assert ret.abs().max() < 0.5  # No >50% daily moves in synthetic data


class TestHistoricalVolatility:
    def test_output_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        hv = compute_historical_volatility(sample_ohlcv["Close"])
        assert list(hv.columns) == ["hv_5", "hv_10", "hv_20", "hv_60"]

    def test_annualized_scale(self, sample_ohlcv: pd.DataFrame) -> None:
        hv = compute_historical_volatility(sample_ohlcv["Close"])
        # Synthetic data has ~32% vol; HV20 should be in 15-60% range
        hv20_mean = hv["hv_20"].dropna().mean()
        assert 0.10 < hv20_mean < 0.80

    def test_shorter_window_more_volatile(self, sample_ohlcv: pd.DataFrame) -> None:
        hv = compute_historical_volatility(sample_ohlcv["Close"])
        # HV5 std should be higher than HV60 std (more noisy)
        assert hv["hv_5"].dropna().std() > hv["hv_60"].dropna().std()


class TestATR:
    def test_output_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        atr = compute_atr(sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"])
        assert list(atr.columns) == ["atr_5", "atr_14", "atr_20"]

    def test_atr_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        atr = compute_atr(sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"])
        for col in atr.columns:
            assert (atr[col].dropna() >= 0).all()

    def test_normalized_by_price(self, sample_ohlcv: pd.DataFrame) -> None:
        atr = compute_atr(sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"])
        # Normalized ATR should be small (typically 1-5% of price)
        assert atr["atr_14"].dropna().mean() < 0.10


class TestBollingerBands:
    def test_output_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        bb = compute_bollinger_bands(sample_ohlcv["Close"])
        assert list(bb.columns) == ["bb_width", "bb_pct_b"]

    def test_pct_b_range(self, sample_ohlcv: pd.DataFrame) -> None:
        bb = compute_bollinger_bands(sample_ohlcv["Close"])
        pct_b = bb["bb_pct_b"].dropna()
        # Most values should be between -0.5 and 1.5
        assert pct_b.median() > 0.0
        assert pct_b.median() < 1.0

    def test_width_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        bb = compute_bollinger_bands(sample_ohlcv["Close"])
        assert (bb["bb_width"].dropna() > 0).all()


class TestGarmanKlass:
    def test_output_is_series(self, sample_ohlcv: pd.DataFrame) -> None:
        gk = compute_garman_klass(
            sample_ohlcv["Open"],
            sample_ohlcv["High"],
            sample_ohlcv["Low"],
            sample_ohlcv["Close"],
        )
        assert isinstance(gk, pd.Series)
        assert gk.name == "gk_vol"

    def test_gk_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        gk = compute_garman_klass(
            sample_ohlcv["Open"],
            sample_ohlcv["High"],
            sample_ohlcv["Low"],
            sample_ohlcv["Close"],
        )
        assert (gk.dropna() >= 0).all()


class TestAllVolatilityFeatures:
    def test_all_features_computed(self, sample_ohlcv: pd.DataFrame) -> None:
        features = compute_all_volatility_features(sample_ohlcv)
        expected = [
            "hv_5",
            "hv_10",
            "hv_20",
            "hv_60",
            "atr_5",
            "atr_14",
            "atr_20",
            "bb_width",
            "bb_pct_b",
            "gk_vol",
        ]
        for col in expected:
            assert col in features.columns, f"Missing feature: {col}"

    def test_same_index(self, sample_ohlcv: pd.DataFrame) -> None:
        features = compute_all_volatility_features(sample_ohlcv)
        assert features.index.equals(sample_ohlcv.index)
