"""Abstract base class for trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
import structlog

from signals.base import SignalEvent

logger = structlog.get_logger(__name__)


@dataclass
class TradeParams:
    """Common trade parameters attached to signal metadata."""

    entry_price: float
    stop_loss_pct: float
    take_profit_pct: float
    max_hold_days: int


class Strategy(ABC):
    """Base class for all trading strategies."""

    strategy_id: str = "base"

    @abstractmethod
    def scan(self, features: pd.DataFrame, ticker: str) -> list[SignalEvent]:
        """Scan a single ticker's feature DataFrame for signals.

        Features DataFrame has OHLCV + computed features with DatetimeIndex.
        Signal features are already shifted by 1 day (no look-ahead).

        Returns list of SignalEvent for each date a signal fires.
        """
        ...

    def scan_latest(self, features: pd.DataFrame, ticker: str) -> SignalEvent | None:
        """Scan only the most recent row for a live signal.

        Convenience method for daily scanning.
        """
        if features.empty:
            return None
        signals = self.scan(features.iloc[[-1]], ticker)
        return signals[0] if signals else None

    def evaluate(self, features: pd.DataFrame, ticker: str) -> dict[str, object]:
        """Evaluate the latest row and return per-condition rationale.

        Returns dict with:
          - triggered: bool
          - conditions: list of {name, value, threshold, operator, passed}
          - signal: SignalEvent | None

        Subclasses should override this for strategy-specific conditions.
        """
        signal = self.scan_latest(features, ticker)
        return {
            "strategy_id": self.strategy_id,
            "ticker": ticker,
            "triggered": signal is not None,
            "direction": signal.direction.value.upper() if signal else "none",
            "conditions": [],
            "signal": signal,
        }
