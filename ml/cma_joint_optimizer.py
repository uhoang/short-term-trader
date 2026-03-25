"""Feature 3: CMA-ES joint optimization.

Encodes all ~37 parameters (strategy params + regime weights) as a single
vector and optimizes jointly using Covariance Matrix Adaptation Evolution Strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import structlog

from backtest.optimizer import PARAM_SPACES
from ml.utils import load_ml_result, save_ml_result
from signals.config_loader import load_strategy_configs, save_strategy_configs
from signals.regime import DEFAULT_REGIME_WEIGHTS, load_regime_weights, save_regime_weights

logger = structlog.get_logger(__name__)

RESULT_KEY = "cma_result"

STRATEGY_ORDER = sorted(PARAM_SPACES.keys())
REGIME_ORDER = sorted(DEFAULT_REGIME_WEIGHTS.keys())


@dataclass
class ParameterCodec:
    """Encodes/decodes between dict configs and flat numpy vector.

    Maintains deterministic ordering:
    1. Strategy params: sorted by strategy name, then sorted param name within each
    2. Regime weights: sorted by regime name, then sorted strategy name within each
    """

    param_spaces: dict[str, dict[str, dict[str, Any]]]
    regime_order: list[str]
    strategy_order: list[str]

    def __post_init__(self) -> None:
        # Build ordered list of (section, name, low, high, is_int)
        self._param_list: list[tuple[str, str, float, float, bool]] = []

        # Strategy params
        for sid in self.strategy_order:
            space = self.param_spaces.get(sid, {})
            for pname in sorted(space.keys()):
                s = space[pname]
                is_int = isinstance(s.get("low"), int) and isinstance(s.get("high"), int)
                self._param_list.append(
                    (
                        f"strategy:{sid}",
                        pname,
                        float(s["low"]),
                        float(s["high"]),
                        is_int,
                    )
                )

        # Regime weights
        for regime in self.regime_order:
            for sid in self.strategy_order:
                self._param_list.append(
                    (
                        f"regime:{regime}",
                        sid,
                        0.0,
                        2.0,
                        False,
                    )
                )

    @property
    def dimension(self) -> int:
        """Total number of parameters."""
        return len(self._param_list)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds."""
        lows = np.array([p[2] for p in self._param_list])
        highs = np.array([p[3] for p in self._param_list])
        return lows, highs

    @property
    def param_names(self) -> list[str]:
        """Human-readable parameter names."""
        return [f"{p[0]}.{p[1]}" for p in self._param_list]

    def encode(
        self,
        strategy_params: dict[str, dict[str, Any]],
        regime_weights: dict[str, dict[str, float]],
    ) -> np.ndarray:
        """Flatten configs to a vector."""
        vector = np.zeros(self.dimension)

        idx = 0
        # Strategy params
        for sid in self.strategy_order:
            space = self.param_spaces.get(sid, {})
            params = strategy_params.get(sid, {})
            for pname in sorted(space.keys()):
                s = space[pname]
                # Use current value or midpoint
                default = (s["low"] + s["high"]) / 2
                vector[idx] = float(params.get(pname, default))
                idx += 1

        # Regime weights
        for regime in self.regime_order:
            rw = regime_weights.get(regime, {})
            for sid in self.strategy_order:
                vector[idx] = float(rw.get(sid, 1.0))
                idx += 1

        return vector

    def decode(
        self, vector: np.ndarray
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, float]]]:
        """Reconstruct strategy_params and regime_weights from vector."""
        strategy_params: dict[str, dict[str, Any]] = {sid: {} for sid in self.strategy_order}
        regime_weights: dict[str, dict[str, float]] = {r: {} for r in self.regime_order}

        idx = 0
        # Strategy params
        for sid in self.strategy_order:
            space = self.param_spaces.get(sid, {})
            for pname in sorted(space.keys()):
                section, name, low, high, is_int = self._param_list[idx]
                value = float(np.clip(vector[idx], low, high))
                if is_int:
                    value = int(round(value))
                strategy_params[sid][pname] = value
                idx += 1

        # Regime weights
        for regime in self.regime_order:
            for sid in self.strategy_order:
                section, name, low, high, is_int = self._param_list[idx]
                regime_weights[regime][sid] = float(np.clip(vector[idx], low, high))
                idx += 1

        return strategy_params, regime_weights


class CMAJointOptimizer:
    """CMA-ES optimization over the full parameter space.

    Jointly optimizes all strategy parameters and regime weights using
    Covariance Matrix Adaptation Evolution Strategy, evaluated via
    walk-forward backtest fitness.
    """

    def __init__(
        self,
        codec: ParameterCodec | None = None,
        sigma0: float = 0.3,
        max_dd_constraint: float = -0.12,
    ) -> None:
        self.codec = codec or ParameterCodec(
            param_spaces=PARAM_SPACES,
            regime_order=REGIME_ORDER,
            strategy_order=STRATEGY_ORDER,
        )
        self.sigma0 = sigma0
        self.max_dd_constraint = max_dd_constraint
        self._best_fitness: float = -np.inf
        self._best_vector: np.ndarray | None = None
        self._history: list[float] = []

    def optimize(
        self,
        fitness_fn: callable,
        max_generations: int = 100,
        popsize: int | None = None,
        progress_callback: callable | None = None,
    ) -> dict[str, Any]:
        """Run CMA-ES optimization.

        Args:
            fitness_fn: Function(strategy_params, regime_weights) -> float
                Should return a fitness score (higher is better, e.g., OOS Sharpe)
            max_generations: Maximum number of generations
            popsize: Population size (None = CMA default)

        Returns:
            Dict with best_strategy_params, best_regime_weights, best_fitness, history
        """
        try:
            import cma
        except ImportError:
            raise ImportError("cma library required. Install with: pip install cma")

        # Initial vector from current configs
        current_strategy = load_strategy_configs()
        current_regime = load_regime_weights()
        x0 = self.codec.encode(current_strategy, current_regime)

        lows, highs = self.codec.bounds

        opts = {
            "maxiter": max_generations,
            "bounds": [lows.tolist(), highs.tolist()],
            "verb_disp": 0,
            "verb_log": 0,
            "seed": 42,
        }
        if popsize:
            opts["popsize"] = popsize

        def neg_fitness(x: np.ndarray) -> float:
            """CMA-ES minimizes, so negate our fitness."""
            strategy_params, regime_weights = self.codec.decode(x)
            try:
                score = fitness_fn(strategy_params, regime_weights)
            except Exception:
                logger.exception("cma_fitness_error")
                score = -10.0
            return -score

        es = cma.CMAEvolutionStrategy(x0.tolist(), self.sigma0, opts)

        generation = 0
        while not es.stop():
            solutions = es.ask()
            fitnesses = [neg_fitness(x) for x in solutions]
            es.tell(solutions, fitnesses)

            best_gen = -min(fitnesses)
            self._history.append(best_gen)

            if best_gen > self._best_fitness:
                self._best_fitness = best_gen
                self._best_vector = np.array(solutions[np.argmin(fitnesses)])

            generation += 1

            if progress_callback:
                progress_callback(generation, max_generations, self._best_fitness)

            if generation % 10 == 0:
                logger.info(
                    "cma_progress",
                    generation=generation,
                    best_fitness=round(self._best_fitness, 4),
                )

        # Decode best
        if self._best_vector is None:
            self._best_vector = x0

        best_strategy, best_regime = self.codec.decode(self._best_vector)

        result = {
            "best_strategy_params": best_strategy,
            "best_regime_weights": best_regime,
            "best_fitness": self._best_fitness,
            "generations": generation,
            "history": self._history,
            "dimension": self.codec.dimension,
            "timestamp": datetime.now().isoformat(),
        }

        save_ml_result(RESULT_KEY, result)

        logger.info(
            "cma_complete",
            best_fitness=round(self._best_fitness, 4),
            generations=generation,
            dimension=self.codec.dimension,
        )

        return result

    def accept_result(self, result: dict[str, Any]) -> None:
        """Accept CMA result and save to config files."""
        if "best_strategy_params" in result:
            configs = load_strategy_configs()
            for sid, params in result["best_strategy_params"].items():
                if sid in configs:
                    configs[sid].update(params)
            save_strategy_configs(configs)

        if "best_regime_weights" in result:
            save_regime_weights(result["best_regime_weights"])

        logger.info("cma_result_accepted")

    @staticmethod
    def get_last_result() -> dict[str, Any] | None:
        """Load the last CMA optimization result."""
        return load_ml_result(RESULT_KEY)

    @staticmethod
    def build_fitness_fn(
        scan_fn: callable,
        prices: dict[str, object],
        features: dict[str, object],
    ) -> callable:
        """Build a fitness function that runs walk-forward backtest.

        Args:
            scan_fn: Function(features, strategy_params, regime_weights) -> list[SignalEvent]
            prices: Dict of ticker -> OHLCV DataFrame
            features: Dict of ticker -> features DataFrame

        Returns:
            Callable(strategy_params, regime_weights) -> float (OOS Sharpe)
        """
        from backtest.engine import BacktestConfig, BacktestEngine
        from backtest.walkforward import WalkForward, WalkForwardConfig

        wf = WalkForward(config=WalkForwardConfig(), backtest_config=BacktestConfig())
        engine = BacktestEngine(BacktestConfig())

        # Pre-compute date range
        all_dates = set()
        for df in prices.values():
            all_dates.update(df.index)
        if not all_dates:
            return lambda sp, rw: -10.0

        start = min(all_dates).strftime("%Y-%m-%d")
        end = max(all_dates).strftime("%Y-%m-%d")
        windows = wf.generate_windows(start, end)

        if not windows:
            return lambda sp, rw: -10.0

        def fitness(strategy_params: dict, regime_weights: dict) -> float:
            oos_sharpes = []
            oos_dds = []

            for is_start, is_end, oos_start, oos_end in windows:
                oos_prices = wf._filter_dates(prices, oos_start, oos_end)
                oos_features = wf._filter_dates(features, oos_start, oos_end)

                try:
                    signals = scan_fn(oos_features, strategy_params, regime_weights)
                    result = engine.run(signals, oos_prices)
                    metrics = result.metrics()
                    oos_sharpes.append(metrics.get("sharpe", 0))
                    oos_dds.append(metrics.get("max_drawdown", 0))
                except Exception:
                    oos_sharpes.append(-1.0)
                    oos_dds.append(-0.20)

            if not oos_sharpes:
                return -10.0

            avg_sharpe = float(np.mean(oos_sharpes))
            worst_dd = float(min(oos_dds))

            if worst_dd < -0.12:
                avg_sharpe -= abs(worst_dd + 0.12) * 10

            return avg_sharpe

        return fitness
