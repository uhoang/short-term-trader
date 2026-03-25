"""Hyperparameter optimization using Optuna with walk-forward CV."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np
import structlog

from backtest.engine import BacktestConfig, BacktestEngine
from backtest.walkforward import WalkForward, WalkForwardConfig

logger = structlog.get_logger(__name__)

# Default search spaces per strategy
PARAM_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "catalyst_capture": {
        "event_score_min": {"low": 0.1, "high": 0.6, "step": 0.1},
        "atr_ratio_min": {"low": 1.0, "high": 2.5, "step": 0.25},
        "stop_loss_pct": {"low": 0.04, "high": 0.10, "step": 0.01},
        "take_profit_pct": {"low": 0.08, "high": 0.20, "step": 0.02},
        "max_hold_days": {"low": 5, "high": 15, "step": 1},
    },
    "volatility_breakout": {
        "atr_ratio_min": {"low": 1.0, "high": 2.0, "step": 0.1},
        "volume_spike_min": {"low": 1.2, "high": 2.0, "step": 0.1},
        "stop_loss_pct": {"low": 0.03, "high": 0.08, "step": 0.01},
        "max_hold_days": {"low": 5, "high": 15, "step": 1},
    },
    "mean_reversion": {
        "vwap_dev_threshold": {"low": -0.04, "high": -0.01, "step": 0.005},
        "rsi_threshold": {"low": 25.0, "high": 38.0, "step": 1.0},
        "stop_loss_pct": {"low": 0.05, "high": 0.12, "step": 0.01},
        "max_hold_days": {"low": 15, "high": 30, "step": 1},
    },
    "sector_momentum": {
        "rebalance_days": {"low": 10, "high": 25, "step": 5},
        "top_n": {"low": 2, "high": 5, "step": 1},
        "stop_loss_pct": {"low": 0.05, "high": 0.12, "step": 0.01},
    },
}


class StrategyOptimizer:
    """Bayesian optimization of strategy parameters via Optuna."""

    def __init__(
        self,
        wf_config: WalkForwardConfig | None = None,
        bt_config: BacktestConfig | None = None,
        max_dd_constraint: float = -0.12,
    ) -> None:
        self.wf_config = wf_config or WalkForwardConfig()
        self.bt_config = bt_config or BacktestConfig()
        self.max_dd_constraint = max_dd_constraint

    def optimize(
        self,
        scan_fn: callable,
        prices: dict[str, object],
        features: dict[str, object],
        param_space: dict[str, dict[str, Any]],
        n_trials: int = 50,
        seed: int = 42,
        progress_callback: callable | None = None,
    ) -> dict[str, Any]:
        """Run Optuna optimization with walk-forward CV.

        Args:
            scan_fn: Function(features, params) -> list[SignalEvent]
            prices: Dict of ticker -> OHLCV DataFrame
            features: Dict of ticker -> features DataFrame
            param_space: Dict of param_name -> {low, high, step}
            n_trials: Number of Optuna trials
            seed: Random seed for reproducibility
            progress_callback: Optional callback(trial_number, n_trials, best_value)
                called after each trial completes.

        Returns:
            Dict with best_params, best_sharpe, study_results.
        """
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        wf = WalkForward(config=self.wf_config, backtest_config=self.bt_config)

        def objective(trial: optuna.Trial) -> float:
            # Sample parameters
            params = {}
            for name, space in param_space.items():
                if isinstance(space.get("low"), float):
                    params[name] = trial.suggest_float(
                        name, space["low"], space["high"], step=space.get("step")
                    )
                else:
                    params[name] = trial.suggest_int(
                        name, space["low"], space["high"], step=space.get("step", 1)
                    )

            # Run walk-forward
            results = wf.run(scan_fn, prices, features, param_grid=None)
            if not results:
                return -10.0

            # Override: run scan_fn with sampled params for each window
            engine = BacktestEngine(self.bt_config)
            oos_sharpes = []
            oos_dds = []

            windows = wf.generate_windows(
                min(d.strftime("%Y-%m-%d") for df in prices.values() for d in df.index),
                max(d.strftime("%Y-%m-%d") for df in prices.values() for d in df.index),
            )

            for is_start, is_end, oos_start, oos_end in windows:
                oos_prices = wf._filter_dates(prices, oos_start, oos_end)
                oos_features = wf._filter_dates(features, oos_start, oos_end)

                signals = scan_fn(oos_features, params)
                result = engine.run(signals, oos_prices)
                metrics = result.metrics()
                oos_sharpes.append(metrics.get("sharpe", 0))
                oos_dds.append(metrics.get("max_drawdown", 0))

            if not oos_sharpes:
                return -10.0

            avg_sharpe = float(np.mean(oos_sharpes))
            worst_dd = float(min(oos_dds))

            # Penalize if max DD exceeds constraint
            if worst_dd < self.max_dd_constraint:
                avg_sharpe -= abs(worst_dd - self.max_dd_constraint) * 10

            return avg_sharpe

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        optuna_callbacks = []
        if progress_callback:

            def _optuna_cb(study: object, trial: object) -> None:
                progress_callback(trial.number + 1, n_trials, study.best_value)

            optuna_callbacks.append(_optuna_cb)

        study.optimize(
            objective, n_trials=n_trials, show_progress_bar=False, callbacks=optuna_callbacks
        )

        best = study.best_trial
        logger.info(
            "optimization_complete",
            best_sharpe=best.value,
            best_params=best.params,
            n_trials=len(study.trials),
        )

        return {
            "best_params": best.params,
            "best_sharpe": best.value,
            "n_trials": len(study.trials),
            "param_importances": self._get_importances(study),
        }

    @staticmethod
    def _get_importances(study: object) -> dict[str, float]:
        """Extract parameter importances from Optuna study."""
        try:
            import optuna

            importances = optuna.importance.get_param_importances(study)
            return dict(importances)
        except Exception:
            return {}

    @staticmethod
    def check_robustness(
        study: object,
        tolerance: float = 0.20,
    ) -> dict[str, bool]:
        """Check if performance is stable within ±tolerance of optimal params."""
        import optuna

        best = study.best_trial
        results: dict[str, bool] = {}

        for param_name, best_value in best.params.items():
            nearby_trials = [
                t
                for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
                and param_name in t.params
                and abs(t.params[param_name] - best_value) <= abs(best_value * tolerance)
            ]

            if len(nearby_trials) < 2:
                results[param_name] = True
                continue

            nearby_sharpes = [t.value for t in nearby_trials]
            variation = (max(nearby_sharpes) - min(nearby_sharpes)) / max(abs(best.value), 0.01)
            results[param_name] = variation < 0.30  # Stable if <30% variation

        return results

    @staticmethod
    def get_config_fields(config_cls: type) -> list[str]:
        """Get field names from a strategy config dataclass."""
        return [f.name for f in fields(config_cls)]
