"""Regime detection for adaptive strategy weighting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

REGIME_WEIGHTS_PATH = Path(__file__).parent.parent / "config" / "regime_weights.json"

# Default strategy weight multipliers by regime
DEFAULT_REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "high_vol": {
        "catalyst_capture": 0.8,
        "volatility_breakout": 1.2,
        "mean_reversion": 1.5,
        "sector_momentum": 0.0,
    },
    "low_vol": {
        "catalyst_capture": 1.3,
        "volatility_breakout": 0.5,
        "mean_reversion": 0.7,
        "sector_momentum": 1.2,
    },
    "normal": {
        "catalyst_capture": 1.0,
        "volatility_breakout": 1.0,
        "mean_reversion": 1.0,
        "sector_momentum": 1.0,
    },
}


def load_regime_weights(
    path: Path | str = REGIME_WEIGHTS_PATH,
) -> dict[str, dict[str, float]]:
    """Load regime weights from JSON, falling back to defaults."""
    path = Path(path)
    if not path.exists():
        return {r: dict(w) for r, w in DEFAULT_REGIME_WEIGHTS.items()}
    with open(path) as f:
        saved = json.load(f)
    # Merge with defaults
    merged: dict[str, dict[str, float]] = {}
    for regime, defaults in DEFAULT_REGIME_WEIGHTS.items():
        merged[regime] = {**defaults, **saved.get(regime, {})}
    return merged


def save_regime_weights(
    weights: dict[str, dict[str, float]],
    path: Path | str = REGIME_WEIGHTS_PATH,
) -> None:
    """Save regime weights to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(weights, f, indent=2)
    logger.info("regime_weights_saved", path=str(path))


# Active weights (loaded once, refreshed by load_regime_weights)
REGIME_WEIGHTS = load_regime_weights()


class RegimeDetector:
    """Detects market regime using HMM or VIX thresholds.

    Falls back to simple threshold-based detection if HMM is unavailable.
    """

    def __init__(
        self,
        high_vol_threshold: float = 0.30,
        low_vol_threshold: float = 0.12,
        use_hmm: bool = True,
    ) -> None:
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.use_hmm = use_hmm
        self._hmm_model: object | None = None
        self._fitted = False

    def fit(self, returns: pd.Series, volatility: pd.Series) -> None:
        """Train 2-state HMM on returns and volatility.

        Args:
            returns: Daily log returns
            volatility: Rolling volatility (e.g., HV20)
        """
        if not self.use_hmm:
            return

        try:
            from hmmlearn.hmm import GaussianHMM

            # Prepare features: returns + volatility
            features = pd.DataFrame({"returns": returns, "volatility": volatility}).dropna()

            if len(features) < 100:
                logger.warning("insufficient_data_for_hmm", rows=len(features))
                return

            x_data = features.values
            model = GaussianHMM(
                n_components=2,
                covariance_type="full",
                n_iter=200,
                random_state=42,
            )
            model.fit(x_data)
            self._hmm_model = model
            self._fitted = True
            logger.info("hmm_fitted", n_states=2, samples=len(x_data))

        except ImportError:
            logger.warning("hmmlearn_not_installed_using_threshold_fallback")
            self.use_hmm = False
        except Exception:
            logger.exception("hmm_fit_failed_using_threshold_fallback")
            self.use_hmm = False

    def predict(self, returns: pd.Series, volatility: pd.Series) -> pd.Series:
        """Classify each day as a regime.

        Returns Series with values: 'high_vol', 'low_vol', or 'normal'.
        """
        if self._fitted and self._hmm_model is not None:
            return self._predict_hmm(returns, volatility)
        return self._predict_threshold(volatility)

    def _predict_hmm(self, returns: pd.Series, volatility: pd.Series) -> pd.Series:
        """Predict regime using trained HMM."""
        features = pd.DataFrame({"returns": returns, "volatility": volatility}).dropna()
        x_data = features.values

        states = self._hmm_model.predict(x_data)

        # Map HMM states to regime labels based on volatility means
        state_vols = []
        for state in range(2):
            mask = states == state
            state_vols.append(volatility.iloc[: len(states)][mask].mean())

        high_vol_state = int(np.argmax(state_vols))
        low_vol_state = 1 - high_vol_state

        regime_map = {high_vol_state: "high_vol", low_vol_state: "low_vol"}
        regimes = pd.Series(
            [regime_map[s] for s in states],
            index=features.index,
            name="regime",
        )
        return regimes

    def _predict_threshold(self, volatility: pd.Series) -> pd.Series:
        """Simple threshold-based regime classification."""
        conditions = [
            volatility > self.high_vol_threshold,
            volatility < self.low_vol_threshold,
        ]
        choices = ["high_vol", "low_vol"]
        regimes = np.select(conditions, choices, default="normal")
        return pd.Series(regimes, index=volatility.index, name="regime")

    @staticmethod
    def get_strategy_weights(regime: str) -> dict[str, float]:
        """Get strategy weight multipliers for the given regime."""
        weights = load_regime_weights()
        return weights.get(regime, weights.get("normal", {})).copy()

    def get_current_regime(self, volatility: pd.Series) -> str:
        """Get the most recent regime classification."""
        if volatility.empty:
            return "normal"
        regimes = self._predict_threshold(volatility)
        return str(regimes.iloc[-1])
