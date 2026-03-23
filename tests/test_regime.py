"""Tests for regime detection."""

from __future__ import annotations

import pandas as pd

from signals.regime import REGIME_WEIGHTS, RegimeDetector


class TestRegimeDetector:
    def test_threshold_high_vol(self) -> None:
        detector = RegimeDetector(use_hmm=False)
        vol = pd.Series([0.35, 0.40, 0.45], name="hv_20")
        regimes = detector.predict(pd.Series([0.0] * 3), vol)
        assert (regimes == "high_vol").all()

    def test_threshold_low_vol(self) -> None:
        detector = RegimeDetector(use_hmm=False)
        vol = pd.Series([0.08, 0.10, 0.05], name="hv_20")
        regimes = detector.predict(pd.Series([0.0] * 3), vol)
        assert (regimes == "low_vol").all()

    def test_threshold_normal(self) -> None:
        detector = RegimeDetector(use_hmm=False)
        vol = pd.Series([0.18, 0.22, 0.20], name="hv_20")
        regimes = detector.predict(pd.Series([0.0] * 3), vol)
        assert (regimes == "normal").all()

    def test_get_current_regime(self) -> None:
        detector = RegimeDetector(use_hmm=False)
        vol = pd.Series([0.10, 0.15, 0.08])
        regime = detector.get_current_regime(vol)
        assert regime == "low_vol"

    def test_empty_volatility(self) -> None:
        detector = RegimeDetector(use_hmm=False)
        regime = detector.get_current_regime(pd.Series(dtype=float))
        assert regime == "normal"


class TestRegimeWeights:
    def test_high_vol_disables_momentum(self) -> None:
        weights = RegimeDetector.get_strategy_weights("high_vol")
        assert weights["sector_momentum"] == 0.0
        assert weights["mean_reversion"] > 1.0

    def test_low_vol_boosts_catalyst(self) -> None:
        weights = RegimeDetector.get_strategy_weights("low_vol")
        assert weights["catalyst_capture"] > 1.0

    def test_normal_all_equal(self) -> None:
        weights = RegimeDetector.get_strategy_weights("normal")
        assert all(w == 1.0 for w in weights.values())

    def test_unknown_returns_normal(self) -> None:
        weights = RegimeDetector.get_strategy_weights("unknown")
        assert all(w == 1.0 for w in weights.values())

    def test_all_regimes_have_all_strategies(self) -> None:
        strategies = {
            "catalyst_capture",
            "volatility_breakout",
            "mean_reversion",
            "sector_momentum",
        }
        for regime, weights in REGIME_WEIGHTS.items():
            assert set(weights.keys()) == strategies
