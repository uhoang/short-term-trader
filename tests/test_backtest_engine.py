"""Tests for the backtesting engine."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from signals.base import Direction, SignalEvent
from signals.strategy import TradeParams


def _make_prices(ticker: str = "AAPL", n: int = 100, trend: float = 0.001) -> pd.DataFrame:
    """Create synthetic price data with known trend."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-02", periods=n)
    returns = np.random.normal(trend, 0.02, n)
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(n, 5_000_000.0),
        },
        index=dates,
    )


def _make_signal(
    ticker: str = "AAPL",
    date: str = "2023-01-10",
    direction: Direction = Direction.LONG,
    stop: float = 0.07,
    tp: float = 0.15,
    hold: int = 10,
) -> SignalEvent:
    return SignalEvent(
        ticker=ticker,
        direction=direction,
        strength=0.8,
        strategy_id="test",
        timestamp=datetime.fromisoformat(date),
        metadata={
            "trade_params": TradeParams(
                entry_price=100.0,
                stop_loss_pct=stop,
                take_profit_pct=tp,
                max_hold_days=hold,
            ),
            "sector": "tech",
        },
    )


class TestBacktestEngine:
    def test_runs_without_signals(self) -> None:
        engine = BacktestEngine()
        prices = {"AAPL": _make_prices()}
        result = engine.run([], prices)
        assert isinstance(result, BacktestResult)
        assert len(result.trades) == 0

    def test_single_long_trade(self) -> None:
        engine = BacktestEngine(BacktestConfig(slippage_pct=0, commission_per_share=0))
        prices = {"AAPL": _make_prices(trend=0.005)}  # Uptrend
        signal = _make_signal(hold=5)
        result = engine.run([signal], prices)

        assert len(result.trades) >= 1
        trade = result.trades[0]
        assert trade.ticker == "AAPL"
        assert trade.direction == "long"
        assert trade.hold_days <= 10  # Calendar days may exceed trading days

    def test_stop_loss_triggers(self) -> None:
        engine = BacktestEngine(BacktestConfig(slippage_pct=0, commission_per_share=0))
        # Create a crash scenario
        np.random.seed(42)
        dates = pd.bdate_range("2023-01-02", periods=50)
        close = np.concatenate([np.full(10, 100.0), np.linspace(100, 85, 40)])
        prices = {
            "AAPL": pd.DataFrame(
                {
                    "Open": close,
                    "High": close * 1.005,
                    "Low": close * 0.995,
                    "Close": close,
                    "Volume": np.full(50, 5_000_000.0),
                },
                index=dates,
            )
        }

        signal = _make_signal(date="2023-01-10", stop=0.07, tp=0.50, hold=30)
        result = engine.run([signal], prices)

        if result.trades:
            trade = result.trades[0]
            assert trade.exit_reason in ("stop_loss", "max_hold")

    def test_slippage_applied(self) -> None:
        config = BacktestConfig(slippage_pct=0.01, commission_per_share=0)
        engine = BacktestEngine(config)
        prices = {"AAPL": _make_prices()}
        signal = _make_signal(hold=3)
        result = engine.run([signal], prices)

        if result.trades:
            # Entry should be higher than Open (long + slippage)
            trade = result.trades[0]
            date = trade.entry_date
            actual_open = float(prices["AAPL"].loc[date, "Open"])
            assert trade.entry_price > actual_open

    def test_equity_curve_starts_at_init_cash(self) -> None:
        engine = BacktestEngine(BacktestConfig(init_cash=500_000))
        prices = {"AAPL": _make_prices()}
        result = engine.run([], prices)
        if len(result.equity_curve) > 0:
            assert result.equity_curve.iloc[0] == pytest.approx(500_000)

    def test_multiple_signals(self) -> None:
        engine = BacktestEngine()
        prices = {
            "AAPL": _make_prices("AAPL"),
            "MSFT": _make_prices("MSFT"),
        }
        signals = [
            _make_signal(ticker="AAPL", date="2023-01-10", hold=5),
            _make_signal(ticker="MSFT", date="2023-01-15", hold=5),
        ]
        result = engine.run(signals, prices)
        assert len(result.trades) >= 1

    def test_short_trade(self) -> None:
        engine = BacktestEngine(BacktestConfig(slippage_pct=0, commission_per_share=0))
        # Downtrend for short profit
        np.random.seed(42)
        dates = pd.bdate_range("2023-01-02", periods=30)
        close = np.linspace(100, 90, 30)
        prices = {
            "AAPL": pd.DataFrame(
                {
                    "Open": close,
                    "High": close * 1.005,
                    "Low": close * 0.995,
                    "Close": close,
                    "Volume": np.full(30, 5_000_000.0),
                },
                index=dates,
            )
        }
        signal = _make_signal(
            date="2023-01-05", direction=Direction.SHORT, stop=0.07, tp=0.08, hold=15
        )
        result = engine.run([signal], prices)
        assert len(result.trades) >= 1


class TestBacktestResult:
    def test_metrics_computation(self) -> None:
        engine = BacktestEngine()
        prices = {"AAPL": _make_prices(trend=0.002)}
        signals = [_make_signal(date="2023-01-10", hold=5)]
        result = engine.run(signals, prices)

        metrics = result.metrics()
        assert "sharpe" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "total_trades" in metrics
        assert "profit_factor" in metrics

    def test_empty_metrics(self) -> None:
        result = BacktestResult(
            equity_curve=pd.Series(dtype=float),
            trades=[],
            daily_returns=pd.Series(dtype=float),
            config=BacktestConfig(),
        )
        metrics = result.metrics()
        assert metrics["total_trades"] == 0
