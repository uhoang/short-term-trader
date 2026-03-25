"""Feature 2: Reinforcement learning for regime weight adjustment.

Trains a PPO agent to learn optimal regime weights based on market state.
Uses stable-baselines3 with a custom Gymnasium environment.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from ml.utils import ML_MODELS_DIR, ensure_dirs, load_ml_result, save_ml_result

logger = structlog.get_logger(__name__)

MODEL_PATH = ML_MODELS_DIR / "rl_regime_agent.zip"
RESULT_KEY = "rl_regime_result"

STRATEGY_ORDER = ["catalyst_capture", "mean_reversion", "sector_momentum", "volatility_breakout"]
REGIME_ORDER = ["high_vol", "low_vol", "normal"]
N_WEIGHTS = len(STRATEGY_ORDER) * len(REGIME_ORDER)  # 12
N_OBS_FEATURES = 6  # avg_hv20, momentum_breadth, sector_dispersion, vol_term, bb_width, rsi


def _check_deps() -> None:
    """Check for required dependencies."""
    try:
        import gymnasium  # noqa: F401
        import stable_baselines3  # noqa: F401
    except ImportError:
        raise ImportError(
            "stable-baselines3 and gymnasium required. "
            "Install with: pip install stable-baselines3 gymnasium"
        )


class TradingRegimeEnv:
    """Custom Gymnasium environment for learning regime weights.

    State: 6 market-level features (avg_hv20, momentum_breadth, etc.)
    Action: 12 continuous values [0, 2.0] — regime weights
    Reward: OOS Sharpe ratio from backtest over a forward window
    """

    def __init__(
        self,
        market_features: pd.DataFrame,
        prices: dict[str, pd.DataFrame],
        features: dict[str, pd.DataFrame],
        scan_fn: callable,
        window_days: int = 126,  # ~6 months of trading days
    ) -> None:
        """
        Args:
            market_features: DataFrame with daily market-level features
            prices: Dict of ticker -> OHLCV DataFrame
            features: Dict of ticker -> feature DataFrame
            scan_fn: Function(features, strategy_params, regime_weights) -> signals
            window_days: Forward evaluation window in trading days
        """
        _check_deps()
        import gymnasium as gym

        self.market_features = market_features
        self.prices = prices
        self.features = features
        self.scan_fn = scan_fn
        self.window_days = window_days

        # Gym spaces
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(N_OBS_FEATURES,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=0.0, high=2.0, shape=(N_WEIGHTS,), dtype=np.float32)

        self._dates = market_features.index.tolist()
        self._current_idx = 0
        self._max_idx = max(0, len(self._dates) - window_days - 1)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset to a random starting point."""
        rng = np.random.default_rng(seed)
        self._current_idx = rng.integers(0, max(1, self._max_idx))
        obs = self._get_observation()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply regime weights, run backtest, return Sharpe as reward.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Decode action to regime weights
        regime_weights = self._decode_action(action)

        # Define forward window
        start_idx = self._current_idx
        end_idx = min(start_idx + self.window_days, len(self._dates) - 1)

        start_date = self._dates[start_idx]
        end_date = self._dates[end_idx]

        # Run backtest on forward window
        reward = self._evaluate(regime_weights, start_date, end_date)

        # Advance
        self._current_idx = min(self._current_idx + self.window_days // 2, self._max_idx)
        terminated = self._current_idx >= self._max_idx
        obs = self._get_observation()

        return obs, reward, terminated, False, {"regime_weights": regime_weights}

    def _get_observation(self) -> np.ndarray:
        """Get current market state as observation vector."""
        if self._current_idx >= len(self._dates):
            return np.zeros(N_OBS_FEATURES, dtype=np.float32)

        row = self.market_features.iloc[self._current_idx]
        obs = np.array(
            [
                row.get("avg_hv20", 0.2),
                row.get("momentum_breadth", 0.5),
                row.get("sector_dispersion", 0.0),
                row.get("vol_term_structure", 1.0),
                row.get("avg_bb_width", 0.05),
                row.get("avg_rsi", 50.0) / 100.0,  # Normalize to ~[0, 1]
            ],
            dtype=np.float32,
        )
        return obs

    def _decode_action(self, action: np.ndarray) -> dict[str, dict[str, float]]:
        """Convert flat action vector to regime weights dict."""
        weights: dict[str, dict[str, float]] = {}
        idx = 0
        for regime in REGIME_ORDER:
            weights[regime] = {}
            for sid in STRATEGY_ORDER:
                weights[regime][sid] = float(np.clip(action[idx], 0.0, 2.0))
                idx += 1
        return weights

    def _evaluate(
        self,
        regime_weights: dict[str, dict[str, float]],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> float:
        """Run backtest with given weights and return Sharpe as reward."""
        from backtest.engine import BacktestConfig, BacktestEngine
        from backtest.walkforward import WalkForward
        from signals.config_loader import load_strategy_configs

        strategy_params = load_strategy_configs()

        # Filter data to window
        wf = WalkForward()
        window_prices = wf._filter_dates(self.prices, start_date, end_date)
        window_features = wf._filter_dates(self.features, start_date, end_date)

        if not window_prices:
            return -1.0

        try:
            signals = self.scan_fn(window_features, strategy_params, regime_weights)
            engine = BacktestEngine(BacktestConfig())
            result = engine.run(signals, window_prices)
            metrics = result.metrics()
            sharpe = metrics.get("sharpe", 0)

            # Penalize large drawdowns
            dd = metrics.get("max_drawdown", 0)
            if dd < -0.12:
                sharpe -= abs(dd + 0.12) * 5

            return float(sharpe)
        except Exception:
            return -1.0


class RLRegimeTrainer:
    """Trains and evaluates the RL agent for regime weight learning."""

    def __init__(self, model_path: Path | str = MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._model: Any = None
        self._trained = False

    def train(
        self,
        env: TradingRegimeEnv,
        total_timesteps: int = 10000,
        algorithm: str = "PPO",
    ) -> dict[str, Any]:
        """Train RL agent.

        Args:
            env: TradingRegimeEnv instance
            total_timesteps: Total training steps
            algorithm: 'PPO' or 'SAC'

        Returns:
            Dict with best_weights, training_curve, metrics
        """
        _check_deps()
        from stable_baselines3 import PPO, SAC

        # Wrap in proper Gym env
        wrapped_env = _GymWrapper(env)

        if algorithm.upper() == "SAC":
            self._model = SAC(
                "MlpPolicy",
                wrapped_env,
                verbose=0,
                seed=42,
                learning_rate=3e-4,
            )
        else:
            self._model = PPO(
                "MlpPolicy",
                wrapped_env,
                verbose=0,
                seed=42,
                n_steps=min(128, total_timesteps // 4),
                learning_rate=3e-4,
            )

        logger.info("rl_training_started", algorithm=algorithm, timesteps=total_timesteps)

        self._model.learn(total_timesteps=total_timesteps)
        self._trained = True

        # Extract best weights by running inference on current state
        obs, _ = wrapped_env.reset()
        action, _ = self._model.predict(obs, deterministic=True)
        best_weights = env._decode_action(action)

        # Save
        self.save()

        result = {
            "algorithm": algorithm,
            "total_timesteps": total_timesteps,
            "best_weights": best_weights,
            "timestamp": datetime.now().isoformat(),
        }

        save_ml_result(RESULT_KEY, result)

        logger.info("rl_training_complete", algorithm=algorithm)
        return result

    def predict(self, market_state: np.ndarray) -> dict[str, dict[str, float]]:
        """Given current market features, output regime weights.

        Args:
            market_state: Array of 6 market features

        Returns:
            Regime weights dict
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        obs = market_state.astype(np.float32)
        action, _ = self._model.predict(obs, deterministic=True)

        weights: dict[str, dict[str, float]] = {}
        idx = 0
        for regime in REGIME_ORDER:
            weights[regime] = {}
            for sid in STRATEGY_ORDER:
                weights[regime][sid] = float(np.clip(action[idx], 0.0, 2.0))
                idx += 1
        return weights

    def save(self, path: Path | str | None = None) -> None:
        """Save trained model."""
        ensure_dirs()
        path = Path(path) if path else self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            self._model.save(str(path))
            logger.info("rl_model_saved", path=str(path))

    def load(self, path: Path | str | None = None) -> None:
        """Load trained model."""
        _check_deps()
        from stable_baselines3 import PPO

        path = Path(path) if path else self.model_path
        if not path.exists():
            raise FileNotFoundError(f"No model at {path}")

        self._model = PPO.load(str(path))
        self._trained = True
        logger.info("rl_model_loaded", path=str(path))

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained."""
        return self._trained

    @staticmethod
    def get_last_result() -> dict[str, Any] | None:
        """Load the last training result."""
        return load_ml_result(RESULT_KEY)


class _GymWrapper:
    """Minimal Gymnasium-compatible wrapper for TradingRegimeEnv."""

    def __init__(self, env: TradingRegimeEnv) -> None:
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = {"render_modes": []}
        self.render_mode = None
        self.spec = None

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple:
        return self.env.reset(seed=seed)

    def step(self, action: np.ndarray) -> tuple:
        return self.env.step(action)

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
