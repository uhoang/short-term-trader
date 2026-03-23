"""Walk-forward validation framework for strategy parameter optimization."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
import structlog

from backtest.engine import BacktestConfig, BacktestEngine

logger = structlog.get_logger(__name__)


@dataclass
class WindowResult:
    """Results for a single walk-forward window."""

    window_id: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    best_params: dict[str, object]
    is_sharpe: float  # In-sample Sharpe
    oos_sharpe: float  # Out-of-sample Sharpe
    oos_max_dd: float
    oos_win_rate: float
    oos_trades: int
    regime: str  # "risk_on", "risk_off", "high_vix", "low_vix"


@dataclass
class WalkForwardConfig:
    """Walk-forward validation parameters."""

    is_months: int = 18  # In-sample window length
    oos_months: int = 6  # Out-of-sample window length
    step_months: int = 3  # Step size between windows


class WalkForward:
    """Rolling walk-forward optimization and validation."""

    def __init__(
        self,
        config: WalkForwardConfig | None = None,
        backtest_config: BacktestConfig | None = None,
    ) -> None:
        self.config = config or WalkForwardConfig()
        self.bt_config = backtest_config or BacktestConfig()

    def generate_windows(
        self, start_date: str, end_date: str
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate (is_start, is_end, oos_start, oos_end) tuples."""
        windows = []
        current = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        while True:
            is_start = current
            is_end = is_start + pd.DateOffset(months=self.config.is_months)
            oos_start = is_end
            oos_end = oos_start + pd.DateOffset(months=self.config.oos_months)

            if oos_end > end:
                break

            windows.append((is_start, is_end, oos_start, oos_end))
            current += pd.DateOffset(months=self.config.step_months)

        return windows

    def run(
        self,
        scan_fn: callable,
        prices: dict[str, pd.DataFrame],
        features: dict[str, pd.DataFrame],
        param_grid: dict[str, list] | None = None,
    ) -> list[WindowResult]:
        """Run walk-forward validation.

        Args:
            scan_fn: Function(features_dict, params) -> list[SignalEvent]
                     Takes features and optional params, returns signals.
            prices: Dict of ticker -> OHLCV DataFrame
            features: Dict of ticker -> features DataFrame
            param_grid: Optional parameter grid for optimization.
                        Dict of param_name -> list of values to try.

        Returns:
            List of WindowResult, one per walk-forward window.
        """
        # Determine date range from prices
        all_dates = set()
        for df in prices.values():
            all_dates.update(df.index)

        if not all_dates:
            return []

        start_date = min(all_dates).strftime("%Y-%m-%d")
        end_date = max(all_dates).strftime("%Y-%m-%d")
        windows = self.generate_windows(start_date, end_date)

        if not windows:
            logger.warning("no_walk_forward_windows", start=start_date, end=end_date)
            return []

        results: list[WindowResult] = []
        engine = BacktestEngine(self.bt_config)

        for i, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
            logger.info(
                "walk_forward_window",
                window=i + 1,
                total=len(windows),
                oos_period=f"{oos_start.date()} to {oos_end.date()}",
            )

            # Filter data to windows
            is_features = self._filter_dates(features, is_start, is_end)
            is_prices = self._filter_dates(prices, is_start, is_end)
            oos_features = self._filter_dates(features, oos_start, oos_end)
            oos_prices = self._filter_dates(prices, oos_start, oos_end)

            # Optimize on in-sample (or use default params)
            best_params: dict[str, object] = {}
            is_sharpe = 0.0

            if param_grid:
                best_params, is_sharpe = self._optimize(
                    scan_fn, is_prices, is_features, param_grid, engine
                )
            else:
                # Run with default params
                is_signals = scan_fn(is_features, {})
                is_result = engine.run(is_signals, is_prices)
                is_sharpe = is_result.metrics().get("sharpe", 0)

            # Evaluate on OOS with optimized params (never retroactive)
            oos_signals = scan_fn(oos_features, best_params)
            oos_result = engine.run(oos_signals, oos_prices)
            oos_metrics = oos_result.metrics()

            # Determine regime
            regime = self._classify_regime(oos_prices, oos_start, oos_end)

            results.append(
                WindowResult(
                    window_id=i + 1,
                    is_start=str(is_start.date()),
                    is_end=str(is_end.date()),
                    oos_start=str(oos_start.date()),
                    oos_end=str(oos_end.date()),
                    best_params=best_params,
                    is_sharpe=is_sharpe,
                    oos_sharpe=oos_metrics.get("sharpe", 0),
                    oos_max_dd=oos_metrics.get("max_drawdown", 0),
                    oos_win_rate=oos_metrics.get("win_rate", 0),
                    oos_trades=oos_metrics.get("total_trades", 0),
                    regime=regime,
                )
            )

        return results

    def _optimize(
        self,
        scan_fn: callable,
        prices: dict[str, pd.DataFrame],
        features: dict[str, pd.DataFrame],
        param_grid: dict[str, list],
        engine: BacktestEngine,
    ) -> tuple[dict[str, object], float]:
        """Grid search for best parameters on in-sample data."""
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        best_sharpe = -np.inf
        best_params: dict[str, object] = {}

        for combo in product(*values):
            params = dict(zip(keys, combo))
            signals = scan_fn(features, params)
            result = engine.run(signals, prices)
            sharpe = result.metrics().get("sharpe", 0)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params

        return best_params, float(best_sharpe)

    @staticmethod
    def _filter_dates(
        data: dict[str, pd.DataFrame],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        """Filter dict of DataFrames to date range."""
        return {
            ticker: df.loc[(df.index >= start) & (df.index < end)]
            for ticker, df in data.items()
            if not df.loc[(df.index >= start) & (df.index < end)].empty
        }

    @staticmethod
    def _classify_regime(
        prices: dict[str, pd.DataFrame],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> str:
        """Classify market regime based on average returns and volatility."""
        all_returns = []
        for df in prices.values():
            period = df.loc[(df.index >= start) & (df.index < end)]
            if len(period) > 1:
                all_returns.extend(period["Close"].pct_change().dropna().tolist())

        if not all_returns:
            return "unknown"

        avg_return = np.mean(all_returns)
        vol = np.std(all_returns) * np.sqrt(252)

        if vol > 0.30:
            return "high_vol"
        elif avg_return > 0.0005:
            return "risk_on"
        elif avg_return < -0.0005:
            return "risk_off"
        else:
            return "low_vol"
