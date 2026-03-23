"""Strategy 3: Mean Reversion — 1-month oversold bounce trades."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from signals.base import Direction, SignalEvent
from signals.strategy import Strategy, TradeParams


@dataclass
class MeanReversionConfig:
    """Tunable parameters for Mean Reversion strategy."""

    vwap_dev_threshold: float = -0.02  # VWAP deviation threshold
    rsi_threshold: float = 32.0  # RSI must be below this
    sector_etf_rsi_min: float = 45.0  # Sector ETF RSI floor (healthy sector)
    stop_loss_pct: float = 0.08
    max_hold_days: int = 22


class MeanReversion(Strategy):
    """Mean reversion: buy oversold stocks in healthy sectors."""

    strategy_id = "mean_reversion"

    def __init__(self, config: MeanReversionConfig | None = None) -> None:
        self.config = config or MeanReversionConfig()

    def scan(
        self,
        features: pd.DataFrame,
        ticker: str,
        sector_etf_rsi: pd.Series | None = None,
    ) -> list[SignalEvent]:
        signals: list[SignalEvent] = []
        cfg = self.config

        required = ["vwap_dev", "rsi_14", "Close"]
        if not all(c in features.columns for c in required):
            return signals

        for i in range(len(features)):
            row = features.iloc[i]
            vwap_dev = row.get("vwap_dev")
            rsi = row.get("rsi_14")
            close = row.get("Close")

            if pd.isna(vwap_dev) or pd.isna(rsi) or pd.isna(close):
                continue

            # Check sector ETF health if provided
            if sector_etf_rsi is not None:
                dt = features.index[i]
                if dt in sector_etf_rsi.index:
                    etf_rsi = sector_etf_rsi.loc[dt]
                    if pd.isna(etf_rsi) or etf_rsi < cfg.sector_etf_rsi_min:
                        continue
                else:
                    continue

            if vwap_dev <= cfg.vwap_dev_threshold and rsi <= cfg.rsi_threshold:
                # Strength scales with how oversold (deeper = stronger)
                strength = min(abs(float(vwap_dev)) / 0.04, 1.0)
                entry_price = float(close) * 0.997  # Limit at -0.3%

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
                                take_profit_pct=0.0,  # Exit at VWAP reversion
                                max_hold_days=cfg.max_hold_days,
                            ),
                            "vwap_dev": float(vwap_dev),
                            "rsi": float(rsi),
                        },
                    )
                )

        return signals

    def evaluate(
        self,
        features: pd.DataFrame,
        ticker: str,
        sector_etf_rsi: pd.Series | None = None,
    ) -> dict[str, object]:
        """Evaluate latest row with per-condition rationale."""
        cfg = self.config
        required = ["vwap_dev", "rsi_14", "Close"]
        if features.empty or not all(c in features.columns for c in required):
            return {
                "strategy_id": self.strategy_id,
                "ticker": ticker,
                "triggered": False,
                "conditions": [],
                "signal": None,
            }

        row = features.iloc[-1]
        vwap = float(row.get("vwap_dev", 0)) if not pd.isna(row.get("vwap_dev", 0)) else 0
        rsi = float(row.get("rsi_14", 50)) if not pd.isna(row.get("rsi_14", 50)) else 50

        conditions = [
            {
                "name": "VWAP Deviation",
                "value": round(vwap, 4),
                "threshold": cfg.vwap_dev_threshold,
                "operator": "<=",
                "passed": vwap <= cfg.vwap_dev_threshold,
            },
            {
                "name": "RSI(14)",
                "value": round(rsi, 1),
                "threshold": cfg.rsi_threshold,
                "operator": "<=",
                "passed": rsi <= cfg.rsi_threshold,
            },
        ]

        if sector_etf_rsi is not None and not sector_etf_rsi.empty:
            dt = features.index[-1]
            etf_val = float(sector_etf_rsi.loc[dt]) if dt in sector_etf_rsi.index else 0
            conditions.append(
                {
                    "name": "Sector ETF RSI",
                    "value": round(etf_val, 1),
                    "threshold": cfg.sector_etf_rsi_min,
                    "operator": ">=",
                    "passed": etf_val >= cfg.sector_etf_rsi_min,
                }
            )

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
