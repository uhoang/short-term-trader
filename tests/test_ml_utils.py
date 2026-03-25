"""Tests for ML utilities."""

from __future__ import annotations

import pandas as pd

from ml.utils import compute_market_state, load_ml_result, save_ml_result


class TestComputeMarketState:
    """Test market state aggregation."""

    def test_empty_features(self):
        state = compute_market_state({})
        assert state["avg_hv20"] == 0.20
        assert state["momentum_breadth"] == 0.50
        assert state["avg_rsi"] == 50.0

    def test_with_features(self):
        dates = pd.date_range("2024-01-01", periods=10)
        df = pd.DataFrame(
            {
                "hv_20": [0.25] * 10,
                "vol_adj_mom_20": [0.5, -0.3, 0.1, 0.8, -0.1, 0.2, 0.3, -0.5, 0.4, 0.6],
                "hv_5": [0.30] * 10,
                "hv_60": [0.20] * 10,
                "bb_width": [0.04] * 10,
                "rsi_14": [55.0] * 10,
            },
            index=dates,
        )
        state = compute_market_state({"AAPL": df})
        assert abs(state["avg_hv20"] - 0.25) < 0.01
        assert state["vol_term_structure"] == 0.30 / 0.20
        assert abs(state["avg_rsi"] - 55.0) < 0.01


class TestMLResultPersistence:
    """Test save/load of ML results."""

    def test_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ml.utils.ML_CONFIG_DIR", tmp_path)
        data = {"accuracy": 0.85, "auc": 0.92}
        save_ml_result("test_model", data)

        loaded = load_ml_result("test_model")
        assert loaded is not None
        assert loaded["accuracy"] == 0.85
        assert loaded["auc"] == 0.92

    def test_load_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ml.utils.ML_CONFIG_DIR", tmp_path)
        result = load_ml_result("nonexistent")
        assert result is None
