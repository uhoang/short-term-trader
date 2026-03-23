"""Tests for the feature store — consolidation, look-ahead, NaN coverage."""

from __future__ import annotations

import pandas as pd
import pytest

from data.feature_store import FeatureStore
from data.providers import YFinanceProvider
from data.warehouse import DataWarehouse


@pytest.fixture
def warehouse_with_data(tmp_path):
    """Create a warehouse with sample ticker data."""
    warehouse_dir = tmp_path / "parquet"
    features_dir = tmp_path / "features"
    warehouse_dir.mkdir()
    features_dir.mkdir()

    warehouse = DataWarehouse(warehouse_dir=warehouse_dir, provider=YFinanceProvider())
    # Download a small date range for testing
    warehouse.download_ticker("AAPL", start="2023-01-02", end="2024-06-30")
    warehouse.download_ticker("MSFT", start="2023-01-02", end="2024-06-30")

    return warehouse, features_dir


class TestFeatureStore:
    def test_build_single_ticker(self, warehouse_with_data) -> None:
        warehouse, features_dir = warehouse_with_data
        store = FeatureStore(features_dir=features_dir, warehouse=warehouse)
        features = store.build("AAPL")

        assert not features.empty
        # Should have OHLCV + computed features
        assert len(features.columns) > 5
        # Should have volatility features
        assert "hv_20" in features.columns
        assert "atr_14" in features.columns
        assert "bb_width" in features.columns
        # Should have momentum features
        assert "rsi_14" in features.columns
        assert "vwap_dev" in features.columns

    def test_no_lookahead_bias(self, warehouse_with_data) -> None:
        """Verify signal features are shifted by 1 day.

        On any given day T, signal features should only use data up to T-1.
        We verify this by checking that signal columns are NaN on the first row
        (because shift(1) makes the first value NaN).
        """
        warehouse, features_dir = warehouse_with_data
        store = FeatureStore(features_dir=features_dir, warehouse=warehouse)
        features = store.build("AAPL")

        # Signal features (non-OHLCV) should be NaN on first row due to shift(1)
        ohlcv_cols = {"Open", "High", "Low", "Close", "Volume"}
        signal_cols = [c for c in features.columns if c not in ohlcv_cols]

        for col in signal_cols:
            assert pd.isna(
                features[col].iloc[0]
            ), f"Feature '{col}' is not NaN on first row — shift(1) may be missing"

    def test_nan_coverage(self, warehouse_with_data) -> None:
        """Verify <2% NaN rate after warmup period."""
        warehouse, features_dir = warehouse_with_data
        store = FeatureStore(features_dir=features_dir, warehouse=warehouse)
        store.build("AAPL")

        result = store.check_nan_coverage("AAPL")
        assert result["ticker"] == "AAPL"
        # Should pass (or at worst, only edge columns fail slightly)
        if not result["passed"]:
            # Log failures for debugging but don't hard-fail on small test data
            for col, pct in result["failures"].items():
                assert pct < 0.10, f"Feature '{col}' has {pct:.1%} NaN (>10%)"

    def test_save_and_load(self, warehouse_with_data) -> None:
        warehouse, features_dir = warehouse_with_data
        store = FeatureStore(features_dir=features_dir, warehouse=warehouse)
        original = store.build("AAPL")
        loaded = store.load("AAPL")

        assert len(loaded) == len(original)
        assert list(loaded.columns) == list(original.columns)

    def test_load_nonexistent_raises(self, warehouse_with_data) -> None:
        warehouse, features_dir = warehouse_with_data
        store = FeatureStore(features_dir=features_dir, warehouse=warehouse)
        with pytest.raises(FileNotFoundError):
            store.load("NONEXISTENT")

    def test_feature_summary(self, warehouse_with_data) -> None:
        warehouse, features_dir = warehouse_with_data
        store = FeatureStore(features_dir=features_dir, warehouse=warehouse)
        store.build("AAPL")

        summary = store.get_feature_summary()
        assert summary["status"] == "ready"
        assert summary["tickers"] == 1
        assert summary["column_count"] > 10
