"""Tests for hyperparameter optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.optimizer import PARAM_SPACES, StrategyOptimizer
from signals.base import Direction, SignalEvent
from signals.strategy import TradeParams


def _make_prices(n: int = 500) -> dict[str, pd.DataFrame]:
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-02", periods=n)
    close = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
    return {
        "AAPL": pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": np.full(n, 5e6),
            },
            index=dates,
        )
    }


def _simple_scan(features: dict, params: dict) -> list[SignalEvent]:
    """Trivial scan: buy every N days (param-controlled)."""
    interval = int(params.get("interval", 20))
    signals = []
    for ticker, df in features.items():
        for i in range(0, len(df), interval):
            signals.append(
                SignalEvent(
                    ticker=ticker,
                    direction=Direction.LONG,
                    strength=0.7,
                    strategy_id="test",
                    timestamp=df.index[i].to_pydatetime(),
                    metadata={
                        "trade_params": TradeParams(
                            entry_price=float(df.iloc[i]["Close"]) if "Close" in df else 100,
                            stop_loss_pct=float(params.get("stop", 0.07)),
                            take_profit_pct=0.15,
                            max_hold_days=int(params.get("hold", 10)),
                        ),
                    },
                )
            )
    return signals


class TestStrategyOptimizer:
    def test_optimize_runs(self) -> None:
        optimizer = StrategyOptimizer()
        prices = _make_prices()
        features = {"AAPL": prices["AAPL"].copy()}

        result = optimizer.optimize(
            scan_fn=_simple_scan,
            prices=prices,
            features=features,
            param_space={
                "interval": {"low": 10, "high": 30, "step": 5},
                "stop": {"low": 0.04, "high": 0.10, "step": 0.02},
                "hold": {"low": 5, "high": 15, "step": 5},
            },
            n_trials=5,
        )

        assert "best_params" in result
        assert "best_sharpe" in result
        assert "n_trials" in result
        assert result["n_trials"] == 5

    def test_param_spaces_defined(self) -> None:
        assert "catalyst_capture" in PARAM_SPACES
        assert "volatility_breakout" in PARAM_SPACES
        assert "mean_reversion" in PARAM_SPACES
        assert "sector_momentum" in PARAM_SPACES
