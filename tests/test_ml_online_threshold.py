"""Tests for online threshold adjustment."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ml.online_threshold import OnlineThresholdAdjuster, ThresholdState


class TestOnlineThresholdAdjuster:
    """Test EWMA success tracking and threshold adjustment."""

    def test_initial_update(self):
        adj = OnlineThresholdAdjuster(alpha=0.1, min_trades=5)
        state = adj.update("mean_reversion", True)
        assert state.strategy_id == "mean_reversion"
        assert state.ewma_success_rate == 1.0
        assert state.n_trades == 1

    def test_ewma_decay(self):
        adj = OnlineThresholdAdjuster(alpha=0.5, min_trades=2)
        adj.update("mean_reversion", True)  # EWMA = 1.0
        state = adj.update("mean_reversion", False)  # EWMA = 0.5*0 + 0.5*1.0 = 0.5
        assert abs(state.ewma_success_rate - 0.5) < 0.01

    def test_no_adjustment_below_min_trades(self):
        adj = OnlineThresholdAdjuster(alpha=0.1, min_trades=20)
        for _ in range(10):
            adj.update("catalyst_capture", False)  # All losses
        state = adj.get_state("catalyst_capture")
        assert state.current_multiplier == 1.0  # No adjustment yet

    def test_tighten_on_poor_performance(self):
        adj = OnlineThresholdAdjuster(alpha=0.5, min_trades=3, tighten_factor=1.15)
        adj._states["catalyst_capture"] = ThresholdState(
            strategy_id="catalyst_capture",
            baseline_success_rate=0.6,
            n_trades=2,
        )
        # Simulate many losses to drive EWMA below 80% of baseline
        for _ in range(10):
            adj.update("catalyst_capture", False)

        state = adj.get_state("catalyst_capture")
        assert state.current_multiplier == 1.15  # Tightened

    def test_loosen_on_good_performance(self):
        adj = OnlineThresholdAdjuster(alpha=0.5, min_trades=3, loosen_factor=0.90)
        adj._states["mean_reversion"] = ThresholdState(
            strategy_id="mean_reversion",
            baseline_success_rate=0.4,
            n_trades=2,
        )
        # Simulate many wins
        for _ in range(10):
            adj.update("mean_reversion", True)

        state = adj.get_state("mean_reversion")
        assert state.current_multiplier == 0.90  # Loosened

    def test_get_adjusted_params_tightened(self):
        adj = OnlineThresholdAdjuster(tighten_factor=1.2)
        adj._states["catalyst_capture"] = ThresholdState(
            strategy_id="catalyst_capture",
            current_multiplier=1.2,
            n_trades=30,
        )
        base = {"event_score_min": 0.3, "atr_ratio_min": 1.5, "stop_loss_pct": 0.07}
        adjusted = adj.get_adjusted_params("catalyst_capture", base)

        # event_score_min should increase (tighten)
        assert adjusted["event_score_min"] > base["event_score_min"]
        # atr_ratio_min should increase
        assert adjusted["atr_ratio_min"] > base["atr_ratio_min"]
        # stop_loss_pct is not a threshold param, should be unchanged
        assert adjusted["stop_loss_pct"] == base["stop_loss_pct"]

    def test_adjusted_params_clamped_to_bounds(self):
        adj = OnlineThresholdAdjuster(tighten_factor=5.0)  # extreme
        adj._states["catalyst_capture"] = ThresholdState(
            strategy_id="catalyst_capture",
            current_multiplier=5.0,
            n_trades=30,
        )
        base = {"event_score_min": 0.5, "atr_ratio_min": 2.0}
        adjusted = adj.get_adjusted_params("catalyst_capture", base)

        # Should be clamped to PARAM_SPACES bounds
        assert adjusted["event_score_min"] <= 0.6  # max from PARAM_SPACES
        assert adjusted["atr_ratio_min"] <= 2.5

    def test_calibrate_baseline(self):
        adj = OnlineThresholdAdjuster()
        backtest = {
            "trades": [
                {"strategy_id": "mean_reversion", "return_pct": 0.05},
                {"strategy_id": "mean_reversion", "return_pct": -0.02},
                {"strategy_id": "mean_reversion", "return_pct": 0.03},
                {"strategy_id": "catalyst_capture", "return_pct": -0.01},
                {"strategy_id": "catalyst_capture", "return_pct": -0.03},
            ]
        }
        adj.calibrate_baseline(backtest)

        mr_state = adj.get_state("mean_reversion")
        assert abs(mr_state.baseline_success_rate - 2 / 3) < 0.01

        cc_state = adj.get_state("catalyst_capture")
        assert cc_state.baseline_success_rate == 0.0

    def test_save_and_load_state(self):
        adj = OnlineThresholdAdjuster()
        adj.update("mean_reversion", True)
        adj.update("mean_reversion", False)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        adj.save_state(path)
        assert path.exists()

        adj2 = OnlineThresholdAdjuster()
        adj2.load_state(path)
        state = adj2.get_state("mean_reversion")
        assert state is not None
        assert state.n_trades == 2

        path.unlink()

    def test_no_adjustment_when_multiplier_neutral(self):
        adj = OnlineThresholdAdjuster()
        adj._states["breakout"] = ThresholdState(strategy_id="breakout", current_multiplier=1.0)
        base = {"atr_ratio_min": 1.5}
        adjusted = adj.get_adjusted_params("breakout", base)
        assert adjusted == base
