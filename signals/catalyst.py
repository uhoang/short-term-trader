"""Strategy 1: Catalyst Capture Scanner — 2-week event-driven trades."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from signals.base import Direction, SignalEvent
from signals.strategy import Strategy, TradeParams


@dataclass
class CatalystCaptureConfig:
    """Tunable parameters for Catalyst Capture strategy."""

    event_score_min: float = 0.3
    atr_ratio_min: float = 1.5  # ATR(5) / ATR(20) threshold
    iv_rank_multiplier: float = 1.3  # BB width vs 6-month median
    stop_loss_pct: float = 0.07
    take_profit_pct: float = 0.15
    max_hold_days: int = 10


class CatalystCapture(Strategy):
    """Event-driven strategy: enter before catalysts when vol is elevated."""

    strategy_id = "catalyst_capture"

    def __init__(self, config: CatalystCaptureConfig | None = None) -> None:
        self.config = config or CatalystCaptureConfig()

    def scan(self, features: pd.DataFrame, ticker: str) -> list[SignalEvent]:
        signals: list[SignalEvent] = []
        cfg = self.config

        required = ["event_score", "atr_5", "atr_20", "bb_width", "Close"]
        if not all(c in features.columns for c in required):
            return signals

        # IV rank proxy: BB width relative to 6-month rolling median
        bb_median_126 = features["bb_width"].rolling(126, min_periods=60).median()
        iv_rank_proxy = features["bb_width"] / bb_median_126.replace(0, np.nan)

        # ATR ratio
        atr_ratio = features["atr_5"] / features["atr_20"].replace(0, np.nan)

        for i in range(len(features)):
            row = features.iloc[i]
            event_score = row.get("event_score", 0)
            atr_r = atr_ratio.iloc[i]
            iv_r = iv_rank_proxy.iloc[i]

            if pd.isna(event_score) or pd.isna(atr_r) or pd.isna(iv_r):
                continue

            if (
                event_score >= cfg.event_score_min
                and atr_r >= cfg.atr_ratio_min
                and iv_r >= cfg.iv_rank_multiplier
            ):
                strength = min(float(event_score), 1.0)
                entry_price = float(row["Close"]) * 1.005  # Limit at +0.5%

                signals.append(
                    SignalEvent(
                        ticker=ticker,
                        direction=Direction.LONG,
                        strength=strength,
                        strategy_id=self.strategy_id,
                        timestamp=features.index[i].to_pydatetime(),
                        metadata={
                            "trade_params": TradeParams(
                                entry_price=entry_price,
                                stop_loss_pct=cfg.stop_loss_pct,
                                take_profit_pct=cfg.take_profit_pct,
                                max_hold_days=cfg.max_hold_days,
                            ),
                            "event_score": event_score,
                            "atr_ratio": float(atr_r),
                            "iv_rank_proxy": float(iv_r),
                        },
                    )
                )

        return signals

    def evaluate(self, features: pd.DataFrame, ticker: str) -> dict[str, object]:
        """Evaluate latest row with per-condition rationale."""
        cfg = self.config
        required = ["event_score", "atr_5", "atr_20", "bb_width", "Close"]
        if features.empty or not all(c in features.columns for c in required):
            return {
                "strategy_id": self.strategy_id,
                "ticker": ticker,
                "triggered": False,
                "conditions": [],
                "signal": None,
            }

        bb_median_126 = features["bb_width"].rolling(126, min_periods=60).median()
        iv_rank_proxy = features["bb_width"] / bb_median_126.replace(0, np.nan)
        atr_ratio = features["atr_5"] / features["atr_20"].replace(0, np.nan)

        row = features.iloc[-1]
        ev = float(row.get("event_score", 0)) if not pd.isna(row.get("event_score", 0)) else 0
        atr_r = float(atr_ratio.iloc[-1]) if not pd.isna(atr_ratio.iloc[-1]) else 0
        iv_r = float(iv_rank_proxy.iloc[-1]) if not pd.isna(iv_rank_proxy.iloc[-1]) else 0

        conditions = [
            {
                "name": "Event Score",
                "value": round(ev, 3),
                "threshold": cfg.event_score_min,
                "operator": ">=",
                "passed": ev >= cfg.event_score_min,
            },
            {
                "name": "ATR Ratio (5/20)",
                "value": round(atr_r, 3),
                "threshold": cfg.atr_ratio_min,
                "operator": ">=",
                "passed": atr_r >= cfg.atr_ratio_min,
            },
            {
                "name": "IV Rank Proxy",
                "value": round(iv_r, 3),
                "threshold": cfg.iv_rank_multiplier,
                "operator": ">=",
                "passed": iv_r >= cfg.iv_rank_multiplier,
            },
        ]
        triggered = all(c["passed"] for c in conditions)
        signal = self.scan_latest(features, ticker) if triggered else None

        return {
            "strategy_id": self.strategy_id,
            "ticker": ticker,
            "triggered": triggered,
            "direction": "LONG" if triggered else "none",
            "conditions": conditions,
            "signal": signal,
        }
