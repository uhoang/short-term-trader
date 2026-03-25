"""Feature 6: Meta-learning regime detector.

Replaces simple HV20 threshold-based regime detection with a GMM/HMM
trained on multiple market features. Discovers 4-5 natural regimes
and maps each to optimal strategy weights.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from ml.utils import ML_MODELS_DIR, ensure_dirs, save_ml_result

logger = structlog.get_logger(__name__)

MODEL_PATH = ML_MODELS_DIR / "meta_regime.joblib"
RESULT_KEY = "meta_regime_weights"

STRATEGY_IDS = [
    "catalyst_capture",
    "volatility_breakout",
    "mean_reversion",
    "sector_momentum",
]

# Auto-label regimes based on cluster center characteristics
REGIME_LABELS = {
    "crisis": "High volatility, low momentum breadth — panic/crash",
    "high_vol_recovery": "Elevated vol with improving momentum — recovery phase",
    "bull_quiet": "Low vol, broad momentum — steady uptrend",
    "normal": "Average vol and momentum — typical conditions",
    "compression": "Very low vol, narrow BB — pre-breakout quiet",
}


class MetaRegimeDetector:
    """Multi-feature regime detection using Gaussian Mixture Model.

    Instead of just using HV20 thresholds, this detector uses 6 market-level
    features to discover natural regime clusters.

    Usage:
        detector = MetaRegimeDetector(n_regimes=4)
        market_features = detector.build_market_features(ticker_features)
        detector.fit(market_features)
        regimes = detector.predict(market_features)
        weights = detector.map_regimes_to_weights(regimes, backtest_trades, ticker_features)
    """

    def __init__(self, n_regimes: int = 4, method: str = "gmm") -> None:
        """
        Args:
            n_regimes: Number of regime clusters to discover
            method: 'gmm' (GaussianMixture) or 'hmm' (GaussianHMM)
        """
        self.n_regimes = n_regimes
        self.method = method
        self._model: Any = None
        self._scaler: Any = None
        self._fitted = False
        self._regime_names: dict[int, str] = {}
        self._regime_weights: dict[str, dict[str, float]] = {}
        self._feature_columns = [
            "avg_hv20",
            "momentum_breadth",
            "sector_dispersion",
            "vol_term_structure",
            "avg_bb_width",
            "avg_rsi",
        ]

    def build_market_features(self, features: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Aggregate per-ticker features into daily market-level features.

        Args:
            features: Dict of ticker -> features DataFrame

        Returns:
            DataFrame with one row per date, columns: avg_hv20, momentum_breadth,
            sector_dispersion, vol_term_structure, avg_bb_width, avg_rsi
        """
        # Collect daily values across all tickers
        daily_data: dict[str, dict[str, list[float]]] = {}

        for ticker, df in features.items():
            if df.empty:
                continue
            for dt in df.index:
                date_key = dt.strftime("%Y-%m-%d")
                if date_key not in daily_data:
                    daily_data[date_key] = {
                        "hv_20": [],
                        "vol_adj_mom_20": [],
                        "hv_5": [],
                        "hv_60": [],
                        "bb_width": [],
                        "rsi_14": [],
                    }

                row = df.loc[dt]
                for col in daily_data[date_key]:
                    if col in df.columns and not pd.isna(row.get(col)):
                        daily_data[date_key][col].append(float(row[col]))

        if not daily_data:
            return pd.DataFrame()

        # Aggregate to market-level
        rows = []
        for date_key in sorted(daily_data.keys()):
            d = daily_data[date_key]
            hv20s = d["hv_20"]
            moms = d["vol_adj_mom_20"]
            hv5s = d["hv_5"]
            hv60s = d["hv_60"]
            bbs = d["bb_width"]
            rsis = d["rsi_14"]

            if not hv20s:
                continue

            avg_hv20 = float(np.mean(hv20s))
            momentum_breadth = sum(1 for m in moms if m > 0) / len(moms) if moms else 0.5
            avg_hv5 = float(np.mean(hv5s)) if hv5s else avg_hv20
            avg_hv60 = float(np.mean(hv60s)) if hv60s else avg_hv20
            vol_term = avg_hv5 / avg_hv60 if avg_hv60 > 0 else 1.0

            rows.append(
                {
                    "date": date_key,
                    "avg_hv20": avg_hv20,
                    "momentum_breadth": momentum_breadth,
                    "sector_dispersion": float(np.std(hv20s)) if len(hv20s) > 1 else 0,
                    "vol_term_structure": vol_term,
                    "avg_bb_width": float(np.mean(bbs)) if bbs else 0.05,
                    "avg_rsi": float(np.mean(rsis)) if rsis else 50.0,
                }
            )

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df

    def fit(self, market_features: pd.DataFrame) -> dict[str, Any]:
        """Train GMM or HMM on market feature matrix.

        Args:
            market_features: DataFrame from build_market_features()

        Returns:
            Dict with cluster_centers, regime_names, n_samples
        """
        if market_features.empty or len(market_features) < 50:
            raise ValueError(f"Need at least 50 rows for fitting, got {len(market_features)}")

        X = market_features[self._feature_columns].dropna()
        if len(X) < 50:
            raise ValueError(f"After dropping NaN, only {len(X)} rows remain")

        try:
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        if self.method == "hmm":
            self._fit_hmm(X_scaled)
        else:
            self._fit_gmm(X_scaled)

        self._fitted = True

        # Auto-label regimes
        labels = self._model.predict(X_scaled)
        self._auto_label_regimes(X, labels)

        result = {
            "n_regimes": self.n_regimes,
            "n_samples": len(X),
            "regime_names": self._regime_names,
            "method": self.method,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("meta_regime_fitted", **result)
        return result

    def _fit_gmm(self, X_scaled: np.ndarray) -> None:
        """Fit Gaussian Mixture Model."""
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            raise ImportError("scikit-learn required")

        self._model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            n_init=5,
            random_state=42,
        )
        self._model.fit(X_scaled)

    def _fit_hmm(self, X_scaled: np.ndarray) -> None:
        """Fit Hidden Markov Model."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.warning("hmmlearn not available, falling back to GMM")
            self.method = "gmm"
            self._fit_gmm(X_scaled)
            return

        self._model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        self._model.fit(X_scaled)

    def _auto_label_regimes(self, X: pd.DataFrame, labels: np.ndarray) -> None:
        """Assign human-readable names to discovered regimes."""
        self._regime_names = {}
        cluster_stats = []

        for i in range(self.n_regimes):
            mask = labels == i
            if not mask.any():
                self._regime_names[i] = f"regime_{i}"
                continue

            cluster_data = X[mask]
            avg_hv20 = cluster_data["avg_hv20"].mean()
            avg_mom = cluster_data["momentum_breadth"].mean()
            avg_bb = cluster_data["avg_bb_width"].mean()
            cluster_stats.append((i, avg_hv20, avg_mom, avg_bb))

        # Sort by volatility to assign labels
        cluster_stats.sort(key=lambda x: x[1])

        label_pool = list(REGIME_LABELS.keys())
        for rank, (idx, hv, mom, bb) in enumerate(cluster_stats):
            if rank < len(label_pool):
                # Map by volatility rank
                if hv > 0.30:
                    name = "crisis"
                elif hv > 0.20 and mom > 0.5:
                    name = "high_vol_recovery"
                elif hv < 0.12 and bb < 0.03:
                    name = "compression"
                elif hv < 0.15 and mom > 0.6:
                    name = "bull_quiet"
                else:
                    name = "normal"
                self._regime_names[idx] = name
            else:
                self._regime_names[idx] = f"regime_{idx}"

    def predict(self, market_features: pd.DataFrame) -> pd.Series:
        """Return regime labels for each date.

        Args:
            market_features: DataFrame from build_market_features()

        Returns:
            Series with string regime labels indexed by date
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = market_features[self._feature_columns].dropna()
        X_scaled = self._scaler.transform(X)
        labels = self._model.predict(X_scaled)

        regime_names = pd.Series(
            [self._regime_names.get(lbl, f"regime_{lbl}") for lbl in labels],
            index=X.index,
            name="regime",
        )
        return regime_names

    def map_regimes_to_weights(
        self,
        regime_labels: pd.Series,
        backtest_trades: list[dict],
        features: dict[str, pd.DataFrame],
    ) -> dict[str, dict[str, float]]:
        """Compute optimal strategy weights per discovered regime.

        For each regime, compute per-strategy Sharpe from trades that
        occurred during that regime. Normalize to weights.

        Args:
            regime_labels: Series from predict()
            backtest_trades: List of trade dicts from backtest result
            features: Dict of ticker -> features DataFrame

        Returns:
            Dict of regime_name -> {strategy_id: weight}
        """
        # Map trades to regimes
        regime_trades: dict[str, dict[str, list[float]]] = {}
        for name in set(regime_labels.values):
            regime_trades[name] = {sid: [] for sid in STRATEGY_IDS}

        for trade in backtest_trades:
            entry_date = pd.Timestamp(trade["entry_date"][:10])
            strategy = trade.get("strategy", trade.get("strategy_id", "unknown"))
            strategy = strategy.split("+")[0] if "+" in strategy else strategy
            return_pct = trade.get("return_pct", 0)

            # Find regime on entry date
            if entry_date in regime_labels.index:
                regime = regime_labels.loc[entry_date]
            else:
                # Find closest date
                diffs = abs(regime_labels.index - entry_date)
                closest = regime_labels.index[diffs.argmin()]
                regime = regime_labels.loc[closest]

            if regime in regime_trades and strategy in regime_trades[regime]:
                regime_trades[regime][strategy].append(return_pct)

        # Compute weights: normalize per-strategy Sharpe within each regime
        weights: dict[str, dict[str, float]] = {}
        for regime_name, strategy_returns in regime_trades.items():
            regime_weights: dict[str, float] = {}
            for sid in STRATEGY_IDS:
                returns = strategy_returns.get(sid, [])
                if len(returns) < 3:
                    regime_weights[sid] = 1.0  # Default if insufficient data
                    continue

                avg = float(np.mean(returns))
                std = float(np.std(returns))
                sharpe = avg / std if std > 0 else 0

                # Convert Sharpe to weight: negative Sharpe → 0, positive → scaled
                if sharpe <= 0:
                    regime_weights[sid] = 0.0
                else:
                    regime_weights[sid] = min(round(sharpe / 0.5, 2), 2.0)

            weights[regime_name] = regime_weights

        self._regime_weights = weights
        return weights

    def get_regime_descriptions(self) -> dict[str, str]:
        """Return descriptions for each discovered regime."""
        descriptions = {}
        for idx, name in self._regime_names.items():
            descriptions[name] = REGIME_LABELS.get(name, f"Auto-discovered regime {idx}")
        return descriptions

    def get_current_regime(self, features: dict[str, pd.DataFrame]) -> str:
        """Get the current regime based on latest market features."""
        if not self._fitted:
            return "normal"

        from ml.utils import compute_market_state

        state = compute_market_state(features)
        row = pd.DataFrame([state])
        row.index = pd.DatetimeIndex([pd.Timestamp.now()])
        X = row[self._feature_columns]
        X_scaled = self._scaler.transform(X)
        label = self._model.predict(X_scaled)[0]
        return self._regime_names.get(label, "normal")

    def save(self, path: Path | str | None = None) -> None:
        """Save model, scaler, and regime metadata."""
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib required")

        ensure_dirs()
        path = Path(path) if path else MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "scaler": self._scaler,
                "regime_names": self._regime_names,
                "regime_weights": self._regime_weights,
                "n_regimes": self.n_regimes,
                "method": self.method,
                "feature_columns": self._feature_columns,
            },
            path,
        )

        # Also save weights as JSON for easy editing
        if self._regime_weights:
            save_ml_result(
                RESULT_KEY,
                {
                    "weights": self._regime_weights,
                    "regime_names": {str(k): v for k, v in self._regime_names.items()},
                    "descriptions": self.get_regime_descriptions(),
                    "timestamp": datetime.now().isoformat(),
                },
            )

        logger.info("meta_regime_saved", path=str(path))

    def load(self, path: Path | str | None = None) -> None:
        """Load model, scaler, and regime metadata."""
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib required")

        path = Path(path) if path else MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"No model at {path}")

        data = joblib.load(path)
        self._model = data["model"]
        self._scaler = data["scaler"]
        self._regime_names = data["regime_names"]
        self._regime_weights = data.get("regime_weights", {})
        self.n_regimes = data["n_regimes"]
        self.method = data["method"]
        self._feature_columns = data.get("feature_columns", self._feature_columns)
        self._fitted = True
        logger.info("meta_regime_loaded", path=str(path), n_regimes=self.n_regimes)

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._fitted
