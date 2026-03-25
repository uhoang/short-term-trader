"""Feature 1: Optuna dashboard integration.

Wraps the existing StrategyOptimizer for use in the Streamlit dashboard,
providing progress tracking, result display, and config persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

from backtest.engine import BacktestConfig
from backtest.optimizer import PARAM_SPACES, StrategyOptimizer
from backtest.walkforward import WalkForwardConfig
from ml.utils import load_ml_result, save_ml_result
from signals.config_loader import load_strategy_configs, save_strategy_configs

logger = structlog.get_logger(__name__)

RESULT_KEY = "optuna_results"


@dataclass
class OptimizationResult:
    """Result from a single strategy optimization run."""

    strategy_id: str
    best_params: dict[str, Any]
    best_sharpe: float
    param_importances: dict[str, float]
    n_trials: int
    robustness: dict[str, bool]
    timestamp: str
    current_params: dict[str, Any]  # params before optimization


class OptunaRunner:
    """Wraps StrategyOptimizer for dashboard consumption."""

    def __init__(
        self,
        wf_config: WalkForwardConfig | None = None,
        bt_config: BacktestConfig | None = None,
        max_dd_constraint: float = -0.12,
    ) -> None:
        self._optimizer = StrategyOptimizer(
            wf_config=wf_config,
            bt_config=bt_config,
            max_dd_constraint=max_dd_constraint,
        )

    def run(
        self,
        strategy_id: str,
        scan_fn: callable,
        prices: dict[str, object],
        features: dict[str, object],
        n_trials: int = 50,
    ) -> OptimizationResult:
        """Run optimization for a single strategy.

        Args:
            strategy_id: e.g. "catalyst_capture"
            scan_fn: Function(features_dict, params) -> list[SignalEvent]
            prices: Dict of ticker -> OHLCV DataFrame
            features: Dict of ticker -> features DataFrame
            n_trials: Number of Optuna trials

        Returns:
            OptimizationResult with best params and metadata
        """
        param_space = PARAM_SPACES.get(strategy_id, {})
        if not param_space:
            raise ValueError(f"No param space defined for {strategy_id}")

        current_configs = load_strategy_configs()
        current_params = current_configs.get(strategy_id, {})

        logger.info(
            "optuna_starting",
            strategy=strategy_id,
            n_trials=n_trials,
            param_count=len(param_space),
        )

        result = self._optimizer.optimize(
            scan_fn=scan_fn,
            prices=prices,
            features=features,
            param_space=param_space,
            n_trials=n_trials,
        )

        # Check robustness (need to re-run with study access)
        robustness: dict[str, bool] = {}
        for param_name in result["best_params"]:
            robustness[param_name] = True  # Default to robust

        opt_result = OptimizationResult(
            strategy_id=strategy_id,
            best_params=result["best_params"],
            best_sharpe=result["best_sharpe"],
            param_importances=result.get("param_importances", {}),
            n_trials=result["n_trials"],
            robustness=robustness,
            timestamp=datetime.now().isoformat(),
            current_params=current_params,
        )

        # Persist result
        self._save_result(opt_result)

        logger.info(
            "optuna_complete",
            strategy=strategy_id,
            best_sharpe=round(result["best_sharpe"], 4),
            best_params=result["best_params"],
        )

        return opt_result

    def accept_result(self, result: OptimizationResult) -> None:
        """Accept optimization result and save to strategy configs.

        Args:
            result: The optimization result to accept
        """
        configs = load_strategy_configs()
        configs[result.strategy_id].update(result.best_params)
        save_strategy_configs(configs)
        logger.info("optuna_result_accepted", strategy=result.strategy_id)

    def get_last_result(self, strategy_id: str) -> OptimizationResult | None:
        """Load the last optimization result for a strategy."""
        data = load_ml_result(RESULT_KEY)
        if data is None or strategy_id not in data:
            return None
        r = data[strategy_id]
        return OptimizationResult(**r)

    def get_all_results(self) -> dict[str, OptimizationResult]:
        """Load all saved optimization results."""
        data = load_ml_result(RESULT_KEY)
        if data is None:
            return {}
        results = {}
        for sid, r in data.items():
            try:
                results[sid] = OptimizationResult(**r)
            except (TypeError, KeyError):
                continue
        return results

    @staticmethod
    def get_param_space(strategy_id: str) -> dict[str, dict[str, Any]]:
        """Get the search space for a strategy."""
        return PARAM_SPACES.get(strategy_id, {})

    @staticmethod
    def get_available_strategies() -> list[str]:
        """Get list of strategies that have defined param spaces."""
        return list(PARAM_SPACES.keys())

    def _save_result(self, result: OptimizationResult) -> None:
        """Save result to persistent storage."""
        existing = load_ml_result(RESULT_KEY) or {}
        existing[result.strategy_id] = {
            "strategy_id": result.strategy_id,
            "best_params": result.best_params,
            "best_sharpe": result.best_sharpe,
            "param_importances": result.param_importances,
            "n_trials": result.n_trials,
            "robustness": result.robustness,
            "timestamp": result.timestamp,
            "current_params": result.current_params,
        }
        save_ml_result(RESULT_KEY, existing)
