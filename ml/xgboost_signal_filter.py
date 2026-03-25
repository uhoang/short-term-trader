"""Feature 4: XGBoost signal quality predictor.

Trains a binary classifier on historical backtest trades to predict
whether a signal will be profitable. Used as a filter on live signals.
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

MODEL_PATH = ML_MODELS_DIR / "signal_filter.joblib"
RESULT_KEY = "signal_filter_meta"

# Features to extract from the feature store
NUMERIC_FEATURES = [
    "hv_5",
    "hv_10",
    "hv_20",
    "hv_60",
    "atr_5",
    "atr_14",
    "atr_20",
    "bb_width",
    "bb_pct_b",
    "gk_vol",
    "vol_adj_mom_10",
    "vol_adj_mom_20",
    "vol_adj_mom_60",
    "vwap_dev",
    "rsi_14",
    "days_to_fomc",
    "event_score",
    "rsi_sector_rel",
    "sector_zscore",
]

# Categorical features (will be one-hot encoded)
CATEGORICAL_FEATURES = ["strategy_id", "sector", "direction"]


class SignalQualityPredictor:
    """XGBoost model to predict signal profitability.

    Usage:
        predictor = SignalQualityPredictor()
        X, y = predictor.build_training_data(backtest_result, feature_store)
        metrics = predictor.train(X, y)
        filtered = predictor.filter_signals(signals, features, threshold=0.5)
    """

    def __init__(self, model_path: Path | str = MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.model: Any = None
        self.feature_columns: list[str] = []
        self._trained = False

    def build_training_data(
        self,
        backtest_result: dict,
        feature_store: object,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Convert backtest trades + features into X, y for training.

        Args:
            backtest_result: Dict with 'trades' list
            feature_store: FeatureStore instance with load() method

        Returns:
            (X, y) where X is feature DataFrame and y is binary target
        """
        from ml.utils import load_backtest_trades_with_features

        backtest_path = Path(__file__).parent.parent / "warehouse" / "backtest_result.json"

        # Use utility to join trades with features
        df = load_backtest_trades_with_features(backtest_path, feature_store)
        if df.empty:
            return pd.DataFrame(), pd.Series(dtype=int)

        # Build feature matrix
        X = self._build_features(df)
        y = df["profitable"].astype(int)

        return X, y

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build feature matrix from raw trade+feature data."""
        parts = []

        # Numeric features
        for col in NUMERIC_FEATURES:
            if col in df.columns:
                parts.append(df[[col]].fillna(0))

        # One-hot encode categoricals
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
                parts.append(dummies)

        if not parts:
            return pd.DataFrame()

        X = pd.concat(parts, axis=1)
        X = X.fillna(0)
        return X

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        test_size: float = 0.2,
    ) -> dict[str, float]:
        """Train XGBoost classifier.

        Args:
            X: Feature matrix
            y: Binary target (1 = profitable)
            n_estimators: Number of boosting rounds
            max_depth: Max tree depth
            learning_rate: Learning rate
            test_size: Fraction for validation

        Returns:
            Dict of metrics: accuracy, precision, recall, auc, feature_importances
        """
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError(
                "xgboost is required for signal filtering. " "Install with: pip install xgboost"
            )

        try:
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )
            from sklearn.model_selection import train_test_split
        except ImportError:
            raise ImportError(
                "scikit-learn is required for signal filtering. "
                "Install with: pip install scikit-learn"
            )

        if len(X) < 20:
            raise ValueError(f"Need at least 20 trades for training, got {len(X)}")

        # Store feature columns for prediction
        self.feature_columns = list(X.columns)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if y.sum() > 1 else None
        )

        # Train
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="auc",
            random_state=42,
            use_label_encoder=False,
            verbosity=0,
        )
        self.model.fit(X_train, y_train)
        self._trained = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_test, y_proba)) if len(set(y_test)) > 1 else 0.0,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "positive_rate": float(y.mean()),
            "timestamp": datetime.now().isoformat(),
        }

        # Feature importances
        importances = dict(zip(self.feature_columns, self.model.feature_importances_.tolist()))
        # Sort by importance
        importances = dict(sorted(importances.items(), key=lambda x: -x[1]))
        metrics["feature_importances"] = importances

        # Save model and metadata
        self.save()
        save_ml_result(RESULT_KEY, {**metrics, "feature_columns": self.feature_columns})

        logger.info(
            "signal_filter_trained",
            accuracy=round(metrics["accuracy"], 3),
            auc=round(metrics["auc"], 3),
            n_features=len(self.feature_columns),
        )

        return metrics

    def predict_proba(self, feature_row: pd.DataFrame) -> np.ndarray:
        """Return probability of profitable outcome.

        Args:
            feature_row: DataFrame with same columns as training data

        Returns:
            Array of probabilities (one per row)
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        # Align columns
        aligned = feature_row.reindex(columns=self.feature_columns, fill_value=0)
        return self.model.predict_proba(aligned)[:, 1]

    def filter_signals(
        self,
        signals: list,
        features: dict[str, pd.DataFrame],
        regime: str = "normal",
        threshold: float = 0.5,
    ) -> list:
        """Filter signals, keeping only those above confidence threshold.

        Args:
            signals: List of SignalEvent objects
            features: Dict of ticker -> features DataFrame
            regime: Current market regime string
            threshold: Minimum probability to keep signal

        Returns:
            Filtered list of signals
        """
        if not self._trained:
            return signals  # Pass through if not trained

        kept = []
        for sig in signals:
            ticker = sig.ticker
            if ticker not in features or features[ticker].empty:
                kept.append(sig)  # Can't evaluate, keep by default
                continue

            # Build feature row
            last_row = features[ticker].iloc[[-1]].copy()
            row_dict = last_row.iloc[0].to_dict()
            row_dict["strategy_id"] = sig.strategy_id
            row_dict["sector"] = sig.metadata.get("sector", "")
            row_dict["direction"] = sig.direction.value
            row_df = pd.DataFrame([row_dict])
            X = self._build_features(row_df)

            try:
                proba = self.predict_proba(X)[0]
                if proba >= threshold:
                    kept.append(sig)
                else:
                    logger.debug(
                        "signal_filtered_by_ml",
                        ticker=ticker,
                        probability=round(proba, 3),
                        threshold=threshold,
                    )
            except Exception:
                kept.append(sig)  # On error, keep signal

        logger.info(
            "ml_filter_applied",
            input_signals=len(signals),
            output_signals=len(kept),
            filtered=len(signals) - len(kept),
        )
        return kept

    def save(self, path: Path | str | None = None) -> None:
        """Save model and feature columns."""
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib required. Install with: pip install joblib")

        ensure_dirs()
        path = Path(path) if path else self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_columns": self.feature_columns},
            path,
        )
        logger.info("signal_filter_saved", path=str(path))

    def load(self, path: Path | str | None = None) -> None:
        """Load model and feature columns."""
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib required. Install with: pip install joblib")

        path = Path(path) if path else self.model_path
        if not path.exists():
            raise FileNotFoundError(f"No model at {path}")

        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self._trained = True
        logger.info("signal_filter_loaded", path=str(path), n_features=len(self.feature_columns))

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained or loaded."""
        return self._trained

    @staticmethod
    def get_last_metrics() -> dict[str, Any] | None:
        """Load the last training metrics."""
        return load_ml_result(RESULT_KEY)
