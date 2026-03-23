"""Pure pandas/numpy backtesting engine with realistic fill simulation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import structlog

from signals.base import Direction, SignalEvent
from signals.strategy import TradeParams

logger = structlog.get_logger(__name__)


@dataclass
class Trade:
    """Completed trade record."""

    ticker: str
    direction: str  # "long" or "short"
    strategy_id: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    shares: float
    return_pct: float
    pnl: float
    hold_days: int
    exit_reason: str  # "stop_loss", "take_profit", "max_hold", "signal_exit"
    sector: str = ""


@dataclass
class BacktestConfig:
    """Backtest simulation parameters."""

    init_cash: float = 1_000_000.0
    slippage_pct: float = 0.0005  # 5 basis points
    commission_per_share: float = 0.005
    short_borrow_annual: float = 0.005  # 0.5% annual
    max_position_pct: float = 0.05  # 5% of NAV per position


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    equity_curve: pd.Series
    trades: list[Trade]
    daily_returns: pd.Series
    config: BacktestConfig

    def metrics(self) -> dict[str, float]:
        """Compute performance metrics."""
        returns = self.daily_returns.dropna()
        if len(returns) == 0:
            return {
                "total_return": 0,
                "annualized_return": 0,
                "annualized_volatility": 0,
                "sharpe": 0,
                "sortino": 0,
                "calmar": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "avg_trade_return": 0,
                "avg_hold_days": 0,
                "profit_factor": 0,
                "total_trades": 0,
            }

        # Annualized metrics
        ann_return = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Sortino (downside deviation)
        downside = returns[returns < 0]
        down_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
        sortino = ann_return / down_vol if down_vol > 0 else 0

        # Max drawdown
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        max_dd = drawdown.min()

        # Calmar
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        # Trade metrics
        trade_returns = [t.return_pct for t in self.trades]
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]

        win_rate = len(wins) / len(trade_returns) if trade_returns else 0
        avg_trade = np.mean(trade_returns) if trade_returns else 0
        avg_hold = np.mean([t.hold_days for t in self.trades]) if self.trades else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        total_return = float(cum_returns.iloc[-1] - 1) if len(cum_returns) > 0 else 0

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "avg_trade_return": avg_trade,
            "avg_hold_days": avg_hold,
            "profit_factor": profit_factor,
            "total_trades": len(self.trades),
        }


class BacktestEngine:
    """Event-driven backtesting engine."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(
        self,
        signals: list[SignalEvent],
        prices: dict[str, pd.DataFrame],
    ) -> BacktestResult:
        """Run backtest over historical prices using provided signals.

        Signal on day T is filled at day T+1 open (no look-ahead bias).
        Exits are checked on each day's Close price.

        Args:
            signals: List of SignalEvents sorted by timestamp
            prices: Dict of ticker -> OHLCV DataFrame

        Returns:
            BacktestResult with equity curve, trades, and daily returns
        """
        cfg = self.config
        cash = cfg.init_cash
        equity = cfg.init_cash

        # Sort signals by date
        signals = sorted(signals, key=lambda s: s.timestamp)

        # Get all trading dates sorted
        all_dates = set()
        for df in prices.values():
            all_dates.update(df.index)
        all_dates_sorted = sorted(all_dates)

        if not all_dates_sorted:
            return BacktestResult(
                equity_curve=pd.Series(dtype=float),
                trades=[],
                daily_returns=pd.Series(dtype=float),
                config=cfg,
            )

        # Build date → next trading date lookup for T+1 fills
        next_date: dict[object, object] = {}
        for i in range(len(all_dates_sorted) - 1):
            next_date[all_dates_sorted[i]] = all_dates_sorted[i + 1]

        # Track state
        open_positions: dict[str, dict] = {}
        completed_trades: list[Trade] = []
        equity_history: dict[object, float] = {}

        # Index signals by date — these will be FILLED on the next trading day
        pending_signals: dict[str, list[SignalEvent]] = {}
        for sig in signals:
            date_key = sig.timestamp.strftime("%Y-%m-%d")
            pending_signals.setdefault(date_key, []).append(sig)

        for dt in all_dates_sorted:
            date_key = pd.Timestamp(dt).strftime("%Y-%m-%d")

            # 1. Check exits on open positions using today's Close
            positions_to_close: list[str] = []
            for ticker, pos in open_positions.items():
                if ticker not in prices or dt not in prices[ticker].index:
                    continue

                close_price = prices[ticker].loc[dt, "Close"]
                if pd.isna(close_price):
                    continue
                current_price = float(close_price)
                entry_price = pos["entry_price"]
                direction = pos["direction"]

                # Calculate unrealized return
                if direction == "long":
                    unrealized = (current_price - entry_price) / entry_price
                else:
                    unrealized = (entry_price - current_price) / entry_price

                hold_days = (pd.Timestamp(dt) - pd.Timestamp(pos["entry_date"])).days
                params: TradeParams = pos["trade_params"]

                exit_reason = ""
                if unrealized <= -params.stop_loss_pct:
                    exit_reason = "stop_loss"
                elif params.take_profit_pct > 0 and unrealized >= params.take_profit_pct:
                    exit_reason = "take_profit"
                elif hold_days >= params.max_hold_days:
                    exit_reason = "max_hold"

                if exit_reason:
                    # Apply slippage on exit
                    if direction == "long":
                        fill_price = current_price * (1 - cfg.slippage_pct)
                    else:
                        fill_price = current_price * (1 + cfg.slippage_pct)

                    commission = pos["shares"] * cfg.commission_per_share

                    if direction == "long":
                        pnl = (fill_price - entry_price) * pos["shares"] - commission
                        # Return cash: sell proceeds
                        cash += fill_price * pos["shares"] - commission
                    else:
                        borrow_cost = (
                            entry_price * pos["shares"] * cfg.short_borrow_annual * hold_days / 365
                        )
                        pnl = (entry_price - fill_price) * pos["shares"] - commission - borrow_cost
                        # Return cash: margin + profit/loss
                        cash += pos["margin_cash"] + pnl

                    return_pct = pnl / (entry_price * pos["shares"])

                    completed_trades.append(
                        Trade(
                            ticker=ticker,
                            direction=direction,
                            strategy_id=pos["strategy_id"],
                            entry_date=pos["entry_date"],
                            entry_price=entry_price,
                            exit_date=dt,
                            exit_price=fill_price,
                            shares=pos["shares"],
                            return_pct=return_pct,
                            pnl=pnl,
                            hold_days=hold_days,
                            exit_reason=exit_reason,
                            sector=pos.get("sector", ""),
                        )
                    )
                    positions_to_close.append(ticker)

            for ticker in positions_to_close:
                del open_positions[ticker]

            # 2. Fill signals from PREVIOUS day (T+1 fill for T signal)
            # Look up what signals were generated yesterday
            prev_date_key = None
            idx = all_dates_sorted.index(dt) if dt in all_dates_sorted else -1
            if idx > 0:
                prev_date_key = pd.Timestamp(all_dates_sorted[idx - 1]).strftime("%Y-%m-%d")

            signals_to_fill = []
            if prev_date_key and prev_date_key in pending_signals:
                signals_to_fill = pending_signals[prev_date_key]

            for sig in signals_to_fill:
                if sig.ticker in open_positions:
                    continue  # Already have position

                if sig.ticker not in prices or dt not in prices[sig.ticker].index:
                    continue

                # Fill at today's Open (signal was from yesterday)
                open_price = prices[sig.ticker].loc[dt, "Open"]
                if pd.isna(open_price):
                    continue
                entry_price = float(open_price)

                # Position sizing based on current equity
                position_value = equity * cfg.max_position_pct

                # Apply slippage
                if sig.direction == Direction.LONG:
                    entry_price *= 1 + cfg.slippage_pct
                else:
                    entry_price *= 1 - cfg.slippage_pct

                shares = position_value / entry_price
                commission = shares * cfg.commission_per_share

                # Extract trade params
                trade_params = sig.metadata.get("trade_params")
                if not trade_params or not isinstance(trade_params, TradeParams):
                    trade_params = TradeParams(
                        entry_price=entry_price,
                        stop_loss_pct=0.07,
                        take_profit_pct=0.15,
                        max_hold_days=10,
                    )

                if sig.direction == Direction.LONG:
                    cost = shares * entry_price + commission
                    if cost > cash:
                        continue
                    cash -= cost
                    margin_cash = 0.0
                else:
                    # Short: set aside margin (entry_price * shares as collateral)
                    margin_cash = shares * entry_price + commission
                    if margin_cash > cash:
                        continue
                    cash -= margin_cash

                open_positions[sig.ticker] = {
                    "direction": sig.direction.value,
                    "entry_price": entry_price,
                    "entry_date": dt,
                    "shares": shares,
                    "strategy_id": sig.strategy_id,
                    "trade_params": trade_params,
                    "sector": sig.metadata.get("sector", ""),
                    "margin_cash": margin_cash,
                }

            # 3. Mark-to-market: cash + open positions
            position_value = 0.0
            for ticker, pos in open_positions.items():
                if ticker not in prices or dt not in prices[ticker].index:
                    continue
                close_val = prices[ticker].loc[dt, "Close"]
                if pd.isna(close_val):
                    continue
                current = float(close_val)

                if pos["direction"] == "long":
                    position_value += current * pos["shares"]
                else:
                    # Short MTM: margin_cash + unrealized P&L
                    unrealized_pnl = (pos["entry_price"] - current) * pos["shares"]
                    position_value += pos["margin_cash"] + unrealized_pnl

            equity = cash + position_value
            equity_history[dt] = equity

        # Build results
        equity_curve = pd.Series(equity_history, name="equity")
        equity_curve.index = pd.DatetimeIndex(equity_curve.index)
        daily_returns = equity_curve.pct_change()

        return BacktestResult(
            equity_curve=equity_curve,
            trades=completed_trades,
            daily_returns=daily_returns,
            config=cfg,
        )
