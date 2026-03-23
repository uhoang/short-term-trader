"""Position tracking and P&L management."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import structlog

from signals.base import SignalEvent
from signals.risk_filter import PortfolioState

logger = structlog.get_logger(__name__)

POSITIONS_FILE = Path(__file__).parent.parent / "warehouse" / "positions.json"


class PositionTracker:
    """Tracks open positions and completed trades internally."""

    def __init__(self, init_equity: float = 1_000_000.0) -> None:
        self.init_equity = init_equity
        self._open: dict[str, dict] = {}
        self._closed: list[dict] = []
        self._cash = init_equity

    def open_position(
        self,
        signal: SignalEvent,
        fill_price: float,
        shares: float,
    ) -> None:
        """Record a new position entry."""
        cost = fill_price * shares
        self._cash -= cost

        self._open[signal.ticker] = {
            "ticker": signal.ticker,
            "direction": signal.direction.value,
            "strategy_id": signal.strategy_id,
            "entry_date": datetime.now().isoformat(),
            "entry_price": fill_price,
            "shares": shares,
            "sector": signal.metadata.get("sector", ""),
        }
        logger.info(
            "position_opened",
            ticker=signal.ticker,
            direction=signal.direction.value,
            price=fill_price,
            shares=shares,
        )

    def close_position(
        self,
        ticker: str,
        fill_price: float,
        reason: str = "signal_exit",
    ) -> dict | None:
        """Close an open position and record the trade."""
        if ticker not in self._open:
            logger.warning("no_open_position", ticker=ticker)
            return None

        pos = self._open.pop(ticker)
        entry_price = pos["entry_price"]
        shares = pos["shares"]

        if pos["direction"] == "long":
            pnl = (fill_price - entry_price) * shares
        else:
            pnl = (entry_price - fill_price) * shares

        return_pct = pnl / (entry_price * shares)
        self._cash += entry_price * shares + pnl

        trade = {
            **pos,
            "exit_date": datetime.now().isoformat(),
            "exit_price": fill_price,
            "return_pct": return_pct,
            "pnl": pnl,
            "exit_reason": reason,
        }
        self._closed.append(trade)

        logger.info(
            "position_closed",
            ticker=ticker,
            pnl=round(pnl, 2),
            return_pct=f"{return_pct:.2%}",
            reason=reason,
        )
        return trade

    def get_open(self) -> dict[str, dict]:
        """Return all open positions."""
        return dict(self._open)

    def get_closed(self) -> list[dict]:
        """Return all completed trades."""
        return list(self._closed)

    def get_portfolio_state(self, current_prices: dict[str, float] | None = None) -> PortfolioState:
        """Build PortfolioState for the risk filter."""
        position_value = 0.0
        open_positions: dict[str, dict[str, object]] = {}

        for ticker, pos in self._open.items():
            price = (
                current_prices.get(ticker, pos["entry_price"])
                if current_prices
                else pos["entry_price"]
            )
            value = price * pos["shares"]
            position_value += value
            open_positions[ticker] = {
                "sector": pos.get("sector", ""),
                "direction": pos["direction"],
                "value": value,
                "entry_date": pos["entry_date"],
            }

        equity = self._cash + position_value
        peak = max(self.init_equity, equity)

        return PortfolioState(
            equity=equity,
            peak_equity=peak,
            open_positions=open_positions,
        )

    def daily_pnl(self, current_prices: dict[str, float]) -> dict[str, float]:
        """Calculate today's unrealized + realized P&L."""
        unrealized = 0.0
        for ticker, pos in self._open.items():
            price = current_prices.get(ticker, pos["entry_price"])
            if pos["direction"] == "long":
                unrealized += (price - pos["entry_price"]) * pos["shares"]
            else:
                unrealized += (pos["entry_price"] - price) * pos["shares"]

        realized = sum(t["pnl"] for t in self._closed)
        return {
            "unrealized": unrealized,
            "realized": realized,
            "total": unrealized + realized,
        }

    def save(self, path: Path | str = POSITIONS_FILE) -> None:
        """Persist positions to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "cash": self._cash,
            "init_equity": self.init_equity,
            "open_positions": self._open,
            "closed_trades": self._closed,
            "saved_at": datetime.now().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, path: Path | str = POSITIONS_FILE) -> None:
        """Load positions from JSON."""
        path = Path(path)
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        self._cash = data.get("cash", self.init_equity)
        self.init_equity = data.get("init_equity", self.init_equity)
        self._open = data.get("open_positions", {})
        self._closed = data.get("closed_trades", [])
        logger.info("positions_loaded", open=len(self._open), closed=len(self._closed))
