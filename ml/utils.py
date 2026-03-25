"""Shared utilities for ML modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

ML_MODELS_DIR = Path(__file__).parent.parent / "warehouse" / "ml_models"
ML_CONFIG_DIR = Path(__file__).parent.parent / "config"


def ensure_dirs() -> None:
    """Create required directories."""
    ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ML_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def check_ml_dependency(name: str) -> bool:
    """Check if an optional ML library is importable."""
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def load_backtest_trades_with_features(
    backtest_path: Path,
    feature_store: object,
) -> pd.DataFrame:
    """Join backtest trades with feature data at entry time.

    Returns DataFrame with columns: all features + strategy_id + sector + direction +
    return_pct (target).
    """
    if not backtest_path.exists():
        return pd.DataFrame()

    with open(backtest_path) as f:
        data = json.load(f)

    trades = data.get("trades", [])
    if not trades:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for trade in trades:
        ticker = trade["ticker"]
        entry_date = trade["entry_date"][:10]

        try:
            features = feature_store.load(ticker)
        except Exception:
            continue

        if features.empty:
            continue

        # Find the feature row closest to (but not after) entry_date
        entry_ts = pd.Timestamp(entry_date)
        mask = features.index <= entry_ts
        if not mask.any():
            continue

        row = features.loc[mask].iloc[-1].to_dict()
        row["ticker"] = ticker
        row["strategy_id"] = trade.get("strategy", trade.get("strategy_id", "unknown"))
        row["sector"] = trade.get("sector", "")
        row["direction"] = trade.get("direction", "long")
        row["return_pct"] = trade.get("return_pct", 0)
        row["pnl"] = trade.get("pnl", 0)
        row["profitable"] = 1 if trade.get("return_pct", 0) > 0 else 0
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def compute_market_state(features: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Compute market-level features from per-ticker feature data.

    Returns dict with: avg_hv20, momentum_breadth, sector_dispersion,
    vol_term_structure, avg_bb_width, avg_rsi.
    """
    if not features:
        return {
            "avg_hv20": 0.20,
            "momentum_breadth": 0.50,
            "sector_dispersion": 0.0,
            "vol_term_structure": 1.0,
            "avg_bb_width": 0.05,
            "avg_rsi": 50.0,
        }

    hv20s = []
    mom_positive = 0
    mom_total = 0
    hv5s = []
    hv60s = []
    bb_widths = []
    rsis = []

    for ticker, df in features.items():
        if df.empty:
            continue
        last = df.iloc[-1]

        if "hv_20" in df.columns and not pd.isna(last.get("hv_20")):
            hv20s.append(float(last["hv_20"]))
        if "vol_adj_mom_20" in df.columns and not pd.isna(last.get("vol_adj_mom_20")):
            mom_total += 1
            if last["vol_adj_mom_20"] > 0:
                mom_positive += 1
        if "hv_5" in df.columns and not pd.isna(last.get("hv_5")):
            hv5s.append(float(last["hv_5"]))
        if "hv_60" in df.columns and not pd.isna(last.get("hv_60")):
            hv60s.append(float(last["hv_60"]))
        if "bb_width" in df.columns and not pd.isna(last.get("bb_width")):
            bb_widths.append(float(last["bb_width"]))
        if "rsi_14" in df.columns and not pd.isna(last.get("rsi_14")):
            rsis.append(float(last["rsi_14"]))

    avg_hv20 = float(np.mean(hv20s)) if hv20s else 0.20
    momentum_breadth = mom_positive / mom_total if mom_total > 0 else 0.50
    avg_hv5 = float(np.mean(hv5s)) if hv5s else avg_hv20
    avg_hv60 = float(np.mean(hv60s)) if hv60s else avg_hv20
    vol_term_structure = avg_hv5 / avg_hv60 if avg_hv60 > 0 else 1.0

    return {
        "avg_hv20": avg_hv20,
        "momentum_breadth": momentum_breadth,
        "sector_dispersion": 0.0,  # requires sector-level computation
        "vol_term_structure": vol_term_structure,
        "avg_bb_width": float(np.mean(bb_widths)) if bb_widths else 0.05,
        "avg_rsi": float(np.mean(rsis)) if rsis else 50.0,
    }


def save_ml_result(name: str, data: dict[str, Any]) -> Path:
    """Save an ML result to config/ml_{name}.json."""
    ensure_dirs()
    path = ML_CONFIG_DIR / f"ml_{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("ml_result_saved", name=name, path=str(path))
    return path


def load_ml_result(name: str) -> dict[str, Any] | None:
    """Load an ML result from config/ml_{name}.json."""
    path = ML_CONFIG_DIR / f"ml_{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
