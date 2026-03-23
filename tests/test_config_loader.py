"""Tests for strategy config loader."""

from __future__ import annotations

import json

from signals.breakout import BreakoutConfig
from signals.catalyst import CatalystCaptureConfig
from signals.config_loader import (
    build_config,
    get_defaults,
    get_param_metadata,
    load_strategy_configs,
    save_strategy_configs,
)


class TestConfigLoader:
    def test_get_defaults(self) -> None:
        defaults = get_defaults()
        assert "catalyst_capture" in defaults
        assert "volatility_breakout" in defaults
        assert "mean_reversion" in defaults
        assert "sector_momentum" in defaults
        assert defaults["catalyst_capture"]["event_score_min"] == 0.3

    def test_save_and_load(self, tmp_path) -> None:
        path = tmp_path / "test_params.json"
        params = get_defaults()
        params["catalyst_capture"]["event_score_min"] = 0.5
        save_strategy_configs(params, path)

        loaded = load_strategy_configs(path)
        assert loaded["catalyst_capture"]["event_score_min"] == 0.5

    def test_load_missing_file_returns_defaults(self, tmp_path) -> None:
        path = tmp_path / "nonexistent.json"
        loaded = load_strategy_configs(path)
        assert loaded == get_defaults()

    def test_load_merges_with_defaults(self, tmp_path) -> None:
        """Saved file with subset of params gets missing ones from defaults."""
        path = tmp_path / "partial.json"
        with open(path, "w") as f:
            json.dump({"catalyst_capture": {"event_score_min": 0.99}}, f)

        loaded = load_strategy_configs(path)
        # Custom value preserved
        assert loaded["catalyst_capture"]["event_score_min"] == 0.99
        # Missing values filled from defaults
        assert loaded["catalyst_capture"]["stop_loss_pct"] == 0.07
        # Other strategies get full defaults
        assert "volatility_breakout" in loaded

    def test_build_config_catalyst(self) -> None:
        params = {"event_score_min": 0.5, "stop_loss_pct": 0.10}
        config = build_config("catalyst_capture", params)
        assert isinstance(config, CatalystCaptureConfig)
        assert config.event_score_min == 0.5
        assert config.stop_loss_pct == 0.10

    def test_build_config_breakout(self) -> None:
        params = {"atr_ratio_min": 1.8}
        config = build_config("volatility_breakout", params)
        assert isinstance(config, BreakoutConfig)
        assert config.atr_ratio_min == 1.8

    def test_build_config_ignores_extra_params(self) -> None:
        params = {"event_score_min": 0.5, "nonexistent_field": 999}
        config = build_config("catalyst_capture", params)
        assert config.event_score_min == 0.5

    def test_get_param_metadata(self) -> None:
        meta = get_param_metadata("catalyst_capture")
        names = [m["name"] for m in meta]
        assert "event_score_min" in names
        assert "stop_loss_pct" in names
        for m in meta:
            assert "label" in m
            assert "type" in m
            assert "default" in m
