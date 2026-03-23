"""Strategy 2: Volatility Breakout — 2-week breakout trades."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from signals.base import Direction, SignalEvent
from signals.strategy import Strategy, TradeParams


@dataclass
class BreakoutConfig:
    """Tunable parameters for Volatility Breakout strategy."""

    atr_ratio_min: float = 1.3  # ATR(5) / ATR(20) threshold
    volume_spike_min: float = 1.5  # Volume vs 20d average
    bb_width_lookback: int = 126  # 6 months for BB width minimum
    stop_loss_pct: float = 0.06
    max_hold_days: int = 10


class VolatilityBreakout(Strategy):
    """Breakout strategy: enter when BB squeezes then expands with volume."""

    strategy_id = "volatility_breakout"

    def __init__(self, config: BreakoutConfig | None = None) -> None:
        self.config = config or BreakoutConfig()

    def scan(self, features: pd.DataFrame, ticker: str) -> list[SignalEvent]:
        signals: list[SignalEvent] = []
        cfg = self.config

        required = ["bb_width", "bb_pct_b", "atr_5", "atr_20", "Volume", "Close"]
        if not all(c in features.columns for c in required):
            return signals

        # BB width at 6-month rolling minimum (squeeze detection)
        bb_min_126 = features["bb_width"].rolling(cfg.bb_width_lookback, min_periods=60).min()
        is_squeeze = features["bb_width"] <= bb_min_126 * 1.05  # Within 5% of min

        # ATR ratio (vol expansion)
        atr_ratio = features["atr_5"] / features["atr_20"].replace(0, np.nan)

        # Volume spike
        vol_avg_20 = features["Volume"].rolling(20, min_periods=10).mean()
        vol_ratio = features["Volume"] / vol_avg_20.replace(0, np.nan)

        for i in range(len(features)):
            row = features.iloc[i]
            squeeze = is_squeeze.iloc[i]
            atr_r = atr_ratio.iloc[i]
            vol_r = vol_ratio.iloc[i]
            bb_pct_b = row.get("bb_pct_b")

            if pd.isna(squeeze) or pd.isna(atr_r) or pd.isna(vol_r) or pd.isna(bb_pct_b):
                continue

            if not (squeeze and atr_r >= cfg.atr_ratio_min and vol_r >= cfg.volume_spike_min):
                continue

            # Direction based on BB position
            if bb_pct_b > 1.0:
                direction = Direction.LONG
            elif bb_pct_b < 0.0:
                direction = Direction.SHORT
            else:
                continue  # No breakout yet

            strength = min(abs(float(bb_pct_b) - 0.5) * 2, 1.0)
            close = float(row["Close"])

            signals.append(
                SignalEvent(
                    ticker=ticker,
                    direction=direction,
                    strength=strength,
                    strategy_id=self.strategy_id,
                    timestamp=features.index[i].to_pydatetime(),
                    metadata={
                        "trade_params": TradeParams(
                            entry_price=close,
                            stop_loss_pct=cfg.stop_loss_pct,
                            take_profit_pct=0.0,  # Exit on opposite band
                            max_hold_days=cfg.max_hold_days,
                        ),
                        "bb_pct_b": float(bb_pct_b),
                        "atr_ratio": float(atr_r),
                        "volume_ratio": float(vol_r),
                    },
                )
            )

        return signals

    def evaluate(self, features: pd.DataFrame, ticker: str) -> dict[str, object]:
        """Evaluate latest row with per-condition rationale."""
        cfg = self.config
        required = ["bb_width", "bb_pct_b", "atr_5", "atr_20", "Volume", "Close"]
        if features.empty or not all(c in features.columns for c in required):
            return {
                "strategy_id": self.strategy_id,
                "ticker": ticker,
                "triggered": False,
                "conditions": [],
                "signal": None,
            }

        bb_min_126 = features["bb_width"].rolling(cfg.bb_width_lookback, min_periods=60).min()
        is_squeeze = features["bb_width"] <= bb_min_126 * 1.05
        atr_ratio = features["atr_5"] / features["atr_20"].replace(0, np.nan)
        vol_avg_20 = features["Volume"].rolling(20, min_periods=10).mean()
        vol_ratio = features["Volume"] / vol_avg_20.replace(0, np.nan)

        row = features.iloc[-1]
        sq = bool(is_squeeze.iloc[-1]) if not pd.isna(is_squeeze.iloc[-1]) else False
        atr_r = float(atr_ratio.iloc[-1]) if not pd.isna(atr_ratio.iloc[-1]) else 0
        vol_r = float(vol_ratio.iloc[-1]) if not pd.isna(vol_ratio.iloc[-1]) else 0
        bb_b = float(row.get("bb_pct_b", 0.5)) if not pd.isna(row.get("bb_pct_b", 0.5)) else 0.5
        breakout = bb_b > 1.0 or bb_b < 0.0

        conditions = [
            {
                "name": "BB Squeeze",
                "value": sq,
                "threshold": True,
                "operator": "==",
                "passed": sq,
            },
            {
                "name": "ATR Ratio (5/20)",
                "value": round(atr_r, 3),
                "threshold": cfg.atr_ratio_min,
                "operator": ">=",
                "passed": atr_r >= cfg.atr_ratio_min,
            },
            {
                "name": "Volume Spike",
                "value": round(vol_r, 2),
                "threshold": cfg.volume_spike_min,
                "operator": ">=",
                "passed": vol_r >= cfg.volume_spike_min,
            },
            {
                "name": "BB Breakout (%B)",
                "value": round(bb_b, 3),
                "threshold": ">1.0 or <0.0",
                "operator": "outside",
                "passed": breakout,
            },
        ]
        triggered = all(c["passed"] for c in conditions)
        signal = self.scan_latest(features, ticker) if triggered else None

        # Direction depends on BB position
        if triggered:
            direction = "LONG" if bb_b > 1.0 else "SHORT"
        else:
            direction = "none"

        return {
            "strategy_id": self.strategy_id,
            "ticker": ticker,
            "triggered": triggered,
            "direction": direction,
            "conditions": conditions,
            "signal": signal,
        }
