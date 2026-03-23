"""Strategy 4: Sector Momentum / Pairs — 3-month ranking-based trades."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import structlog

from signals.base import Direction, SignalEvent
from signals.strategy import Strategy, TradeParams

logger = structlog.get_logger(__name__)


@dataclass
class MomentumPairsConfig:
    """Tunable parameters for Sector Momentum strategy."""

    rebalance_days: int = 15  # Rebalance every N trading days
    top_n: int = 3  # Long top N per sector
    bottom_n: int = 3  # Short bottom N per sector
    stop_loss_pct: float = 0.08
    max_hold_days: int = 45


class SectorMomentumPairs(Strategy):
    """Rank stocks within sector, long top quintile, short bottom quintile."""

    strategy_id = "sector_momentum"

    def __init__(self, config: MomentumPairsConfig | None = None) -> None:
        self.config = config or MomentumPairsConfig()

    def scan(self, features: pd.DataFrame, ticker: str) -> list[SignalEvent]:
        """Single-ticker scan not meaningful for ranking strategy.

        Use scan_sector() instead for proper cross-ticker ranking.
        """
        return []

    def scan_sector(
        self,
        sector_features: dict[str, pd.DataFrame],
        sector_name: str,
    ) -> list[SignalEvent]:
        """Scan an entire sector, ranking stocks by vol-adjusted momentum.

        Args:
            sector_features: dict of ticker -> features DataFrame
            sector_name: name of the sector being scanned

        Returns:
            List of signals for the sector (longs and shorts).
        """
        signals: list[SignalEvent] = []
        cfg = self.config

        if len(sector_features) < cfg.top_n + cfg.bottom_n:
            return signals

        # Get common date index
        all_dates = None
        for df in sector_features.values():
            if "vol_adj_mom_60" not in df.columns:
                continue
            if all_dates is None:
                all_dates = df.index
            else:
                all_dates = all_dates.intersection(df.index)

        if all_dates is None or len(all_dates) == 0:
            return signals

        # Build momentum ranking matrix
        mom_matrix = pd.DataFrame(
            {
                ticker: df.loc[df.index.isin(all_dates), "vol_adj_mom_60"]
                for ticker, df in sector_features.items()
                if "vol_adj_mom_60" in df.columns
            }
        )

        if mom_matrix.empty:
            return signals

        # Scan at rebalance intervals
        for i in range(0, len(mom_matrix), cfg.rebalance_days):
            row = mom_matrix.iloc[i]
            valid = row.dropna()
            if len(valid) < cfg.top_n + cfg.bottom_n:
                continue

            ranked = valid.sort_values(ascending=False)
            longs = ranked.head(cfg.top_n)
            shorts = ranked.tail(cfg.bottom_n)

            dt = mom_matrix.index[i]
            ts = dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else datetime.now()

            for ticker, mom_val in longs.items():
                zscore = self._get_zscore(sector_features, ticker, dt)
                strength = min(abs(float(zscore)) / 2, 1.0) if not np.isnan(zscore) else 0.5
                close = self._get_close(sector_features, ticker, dt)

                signals.append(
                    SignalEvent(
                        ticker=str(ticker),
                        direction=Direction.LONG,
                        strength=strength,
                        strategy_id=self.strategy_id,
                        timestamp=ts,
                        metadata={
                            "trade_params": TradeParams(
                                entry_price=close,
                                stop_loss_pct=cfg.stop_loss_pct,
                                take_profit_pct=0.0,
                                max_hold_days=cfg.max_hold_days,
                            ),
                            "sector": sector_name,
                            "rank": int(list(ranked.index).index(ticker)) + 1,
                            "momentum": float(mom_val),
                        },
                    )
                )

            for ticker, mom_val in shorts.items():
                zscore = self._get_zscore(sector_features, ticker, dt)
                strength = min(abs(float(zscore)) / 2, 1.0) if not np.isnan(zscore) else 0.5
                close = self._get_close(sector_features, ticker, dt)

                signals.append(
                    SignalEvent(
                        ticker=str(ticker),
                        direction=Direction.SHORT,
                        strength=strength,
                        strategy_id=self.strategy_id,
                        timestamp=ts,
                        metadata={
                            "trade_params": TradeParams(
                                entry_price=close,
                                stop_loss_pct=cfg.stop_loss_pct,
                                take_profit_pct=0.0,
                                max_hold_days=cfg.max_hold_days,
                            ),
                            "sector": sector_name,
                            "rank": int(list(ranked.index).index(ticker)) + 1,
                            "momentum": float(mom_val),
                        },
                    )
                )

        return signals

    @staticmethod
    def _get_zscore(sector_features: dict[str, pd.DataFrame], ticker: str, dt: object) -> float:
        df = sector_features.get(str(ticker))
        if df is None or "sector_zscore" not in df.columns or dt not in df.index:
            return float("nan")
        return float(df.loc[dt, "sector_zscore"])

    @staticmethod
    def _get_close(sector_features: dict[str, pd.DataFrame], ticker: str, dt: object) -> float:
        df = sector_features.get(str(ticker))
        if df is None or "Close" not in df.columns or dt not in df.index:
            return 0.0
        return float(df.loc[dt, "Close"])

    def evaluate_sector(
        self,
        sector_features: dict[str, pd.DataFrame],
        sector_name: str,
    ) -> list[dict[str, object]]:
        """Evaluate all tickers in a sector with momentum ranking rationale."""
        cfg = self.config
        results: list[dict[str, object]] = []

        if len(sector_features) < cfg.top_n + cfg.bottom_n:
            return results

        # Get latest momentum values
        latest_mom: dict[str, float] = {}
        for ticker, df in sector_features.items():
            if "vol_adj_mom_60" in df.columns and not df.empty:
                val = df["vol_adj_mom_60"].iloc[-1]
                if not np.isnan(val):
                    latest_mom[ticker] = float(val)

        if len(latest_mom) < cfg.top_n + cfg.bottom_n:
            return results

        ranked = sorted(latest_mom.items(), key=lambda x: x[1], reverse=True)
        long_tickers = {t for t, _ in ranked[: cfg.top_n]}
        short_tickers = {t for t, _ in ranked[-cfg.bottom_n :]}

        for rank, (ticker, mom_val) in enumerate(ranked, 1):
            in_long = ticker in long_tickers
            in_short = ticker in short_tickers
            triggered = in_long or in_short
            direction = "LONG" if in_long else ("SHORT" if in_short else "none")

            results.append(
                {
                    "strategy_id": self.strategy_id,
                    "ticker": ticker,
                    "triggered": triggered,
                    "conditions": [
                        {
                            "name": "Momentum Rank",
                            "value": rank,
                            "threshold": f"Top {cfg.top_n} / Bottom {cfg.bottom_n}",
                            "operator": "rank",
                            "passed": triggered,
                        },
                        {
                            "name": "Vol-Adj Momentum (60d)",
                            "value": round(mom_val, 4),
                            "threshold": "relative",
                            "operator": "rank",
                            "passed": triggered,
                        },
                    ],
                    "signal": None,
                    "direction": direction,
                    "rank": rank,
                    "sector": sector_name,
                }
            )

        return results
