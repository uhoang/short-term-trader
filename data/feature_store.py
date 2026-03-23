"""Feature store: consolidates all features into per-ticker DataFrames."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import structlog

from data.catalysts import compute_days_to_fomc, compute_event_score
from data.features.momentum import (
    compute_all_momentum_features,
    compute_sector_relative_rsi,
    compute_sector_zscore,
)
from data.features.volatility import compute_all_volatility_features
from data.warehouse import DataWarehouse
from scanner.universe import Universe

logger = structlog.get_logger(__name__)

FEATURES_DIR = Path(__file__).parent.parent / "warehouse" / "features"


class FeatureStore:
    """Builds and manages the consolidated feature store."""

    def __init__(
        self,
        features_dir: Path | str = FEATURES_DIR,
        warehouse: DataWarehouse | None = None,
    ) -> None:
        self.features_dir = Path(features_dir)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.warehouse = warehouse or DataWarehouse()
        self.universe = Universe()

    def _feature_path(self, ticker: str) -> Path:
        return self.features_dir / f"{ticker}.feather"

    def build(self, ticker: str) -> pd.DataFrame:
        """Compute all features for a single ticker and save as Feather.

        All signal features are shifted by 1 day to prevent look-ahead bias:
        features computed on day T are available for signals on day T+1.
        """
        ohlcv = self.warehouse.load(ticker)
        logger.info("building_features", ticker=ticker, rows=len(ohlcv))

        # Start with OHLCV
        features = ohlcv.copy()

        # Volatility features
        vol_features = compute_all_volatility_features(ohlcv)
        features = features.join(vol_features)

        # Momentum features (single-ticker)
        mom_features = compute_all_momentum_features(ohlcv)
        features = features.join(mom_features)

        # FOMC proximity
        fomc = compute_days_to_fomc(ohlcv.index)
        features = features.join(fomc)

        # Event score (requires sector)
        sector = self.universe.get_ticker_sector(ticker)
        if sector:
            features["event_score"] = features.apply(
                lambda row: compute_event_score(
                    days_to_earnings=None,  # Earnings dates added in build_all
                    days_to_fomc=row.get("days_to_fomc"),
                    sector=sector,
                ),
                axis=1,
            )
        else:
            features["event_score"] = 0.0

        # Shift signal features by 1 day (no look-ahead)
        # OHLCV columns are NOT shifted (they represent actual prices)
        ohlcv_cols = {"Open", "High", "Low", "Close", "Volume"}
        signal_cols = [c for c in features.columns if c not in ohlcv_cols]
        features[signal_cols] = features[signal_cols].shift(1)

        # Save
        features.to_feather(self._feature_path(ticker))
        logger.info("features_saved", ticker=ticker, columns=len(features.columns))
        return features

    def build_all(self) -> dict[str, int]:
        """Build features for all tickers in the universe.

        Also computes sector-relative features (RSI relative, z-score)
        which require cross-ticker data.

        Returns dict of ticker -> feature column count.
        """
        tickers = self.universe.get_unique_tickers()
        results: dict[str, int] = {}

        # Step 1: Build single-ticker features
        all_features: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                all_features[ticker] = self.build(ticker)
                results[ticker] = len(all_features[ticker].columns)
            except FileNotFoundError:
                logger.warning("no_warehouse_data", ticker=ticker)
                results[ticker] = 0
            except Exception:
                logger.exception("feature_build_failed", ticker=ticker)
                results[ticker] = 0

        # Step 2: Compute sector-relative features
        self._add_sector_relative_features(all_features)

        logger.info(
            "feature_build_complete",
            total=len(tickers),
            successful=sum(1 for v in results.values() if v > 0),
        )
        return results

    def _add_sector_relative_features(self, all_features: dict[str, pd.DataFrame]) -> None:
        """Add sector-relative RSI and z-score features to all tickers."""
        # Build sector map
        sector_map: dict[str, str] = {}
        for ticker in all_features:
            sector = self.universe.get_ticker_sector(ticker)
            if sector:
                sector_map[ticker] = sector

        # Sector-relative RSI
        # rsi_14 is already shifted(1) in each ticker's features,
        # so the relative RSI computed from them is also shifted. No extra shift.
        rsi_series = {t: df["rsi_14"] for t, df in all_features.items() if "rsi_14" in df.columns}
        if rsi_series:
            rel_rsi = compute_sector_relative_rsi(rsi_series, sector_map)
            for ticker, series in rel_rsi.items():
                all_features[ticker]["rsi_sector_rel"] = series
                all_features[ticker].to_feather(self._feature_path(ticker))

        # Sector z-score
        returns_dict = {t: df["Close"].pct_change() for t, df in all_features.items()}
        if returns_dict:
            zscores = compute_sector_zscore(returns_dict, sector_map)
            for ticker, series in zscores.items():
                all_features[ticker]["sector_zscore"] = series.shift(1)
                all_features[ticker].to_feather(self._feature_path(ticker))

    def load(self, ticker: str) -> pd.DataFrame:
        """Load features for a single ticker from Feather."""
        path = self._feature_path(ticker)
        if not path.exists():
            raise FileNotFoundError(f"No features for {ticker}. Run build first.")
        return pd.read_feather(path)

    def get_feature_summary(self) -> dict[str, object]:
        """Return metadata about the feature store."""
        files = list(self.features_dir.glob("*.feather"))
        if not files:
            return {"status": "empty", "tickers": 0}

        sample = pd.read_feather(files[0])
        return {
            "status": "ready",
            "tickers": len(files),
            "columns": list(sample.columns),
            "column_count": len(sample.columns),
            "date_range": {
                "start": str(sample.index.min()) if len(sample) > 0 else None,
                "end": str(sample.index.max()) if len(sample) > 0 else None,
            },
        }

    def check_nan_coverage(self, ticker: str, max_nan_pct: float = 0.02) -> dict[str, object]:
        """Check NaN coverage for a ticker's features.

        Returns dict with per-column NaN percentages and pass/fail status.
        """
        df = self.load(ticker)
        # Exclude warmup period (first 60 rows for lookback windows)
        df_check = df.iloc[60:]

        nan_pcts = df_check.isna().mean()
        failures = nan_pcts[nan_pcts > max_nan_pct]

        return {
            "ticker": ticker,
            "total_rows": len(df_check),
            "passed": len(failures) == 0,
            "max_nan_pct_allowed": max_nan_pct,
            "failures": {col: float(pct) for col, pct in failures.items()},
        }

    def save_metadata(self) -> None:
        """Save feature store metadata to JSON."""
        summary = self.get_feature_summary()
        summary["last_updated"] = datetime.now().isoformat()
        meta_path = self.features_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
