"""Load and save strategy parameters to/from JSON config file."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import structlog

from signals.breakout import BreakoutConfig
from signals.catalyst import CatalystCaptureConfig
from signals.mean_reversion import MeanReversionConfig
from signals.momentum_pairs import MomentumPairsConfig

logger = structlog.get_logger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "strategy_params.json"

# Map strategy_id to config class
STRATEGY_CONFIGS: dict[str, type] = {
    "catalyst_capture": CatalystCaptureConfig,
    "volatility_breakout": BreakoutConfig,
    "mean_reversion": MeanReversionConfig,
    "sector_momentum": MomentumPairsConfig,
}

# Human-readable labels for each parameter
PARAM_LABELS: dict[str, str] = {
    "event_score_min": "Min Event Score",
    "atr_ratio_min": "Min ATR(5)/ATR(20) Ratio",
    "iv_rank_multiplier": "IV Rank Multiplier (BB width vs median)",
    "stop_loss_pct": "Stop Loss %",
    "take_profit_pct": "Take Profit %",
    "max_hold_days": "Max Hold Days",
    "volume_spike_min": "Min Volume Spike (vs 20d avg)",
    "bb_width_lookback": "BB Width Lookback (days)",
    "vwap_dev_threshold": "VWAP Deviation Threshold",
    "rsi_threshold": "RSI Threshold (buy below)",
    "sector_etf_rsi_min": "Sector ETF RSI Floor",
    "rebalance_days": "Rebalance Interval (days)",
    "top_n": "Long Top N per Sector",
    "bottom_n": "Short Bottom N per Sector",
}


def get_defaults() -> dict[str, dict[str, Any]]:
    """Get default config values for all strategies."""
    return {
        strategy_id: asdict(config_cls()) for strategy_id, config_cls in STRATEGY_CONFIGS.items()
    }


def load_strategy_configs(path: Path | str = CONFIG_PATH) -> dict[str, dict[str, Any]]:
    """Load strategy configs from JSON, falling back to defaults for missing values."""
    defaults = get_defaults()
    path = Path(path)

    if not path.exists():
        logger.info("no_config_file_using_defaults", path=str(path))
        return defaults

    with open(path) as f:
        saved = json.load(f)

    # Merge: saved values override defaults
    merged: dict[str, dict[str, Any]] = {}
    for strategy_id, default_params in defaults.items():
        saved_params = saved.get(strategy_id, {})
        merged[strategy_id] = {**default_params, **saved_params}

    return merged


def save_strategy_configs(
    configs: dict[str, dict[str, Any]],
    path: Path | str = CONFIG_PATH,
) -> None:
    """Save strategy configs to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(configs, f, indent=2)
    logger.info("strategy_configs_saved", path=str(path))


def build_config(strategy_id: str, params: dict[str, Any]) -> object:
    """Build a strategy config dataclass from a params dict."""
    config_cls = STRATEGY_CONFIGS.get(strategy_id)
    if config_cls is None:
        raise ValueError(f"Unknown strategy: {strategy_id}")

    # Only pass fields that exist on the dataclass
    valid_fields = {f.name for f in fields(config_cls)}
    filtered = {k: v for k, v in params.items() if k in valid_fields}
    return config_cls(**filtered)


def get_param_metadata(strategy_id: str) -> list[dict[str, Any]]:
    """Get metadata for each param: name, label, type, default, range hints."""
    config_cls = STRATEGY_CONFIGS.get(strategy_id)
    if config_cls is None:
        return []

    defaults = config_cls()
    result = []
    for f in fields(config_cls):
        default_val = getattr(defaults, f.name)
        result.append(
            {
                "name": f.name,
                "label": PARAM_LABELS.get(f.name, f.name),
                "type": "float" if isinstance(default_val, float) else "int",
                "default": default_val,
            }
        )
    return result
