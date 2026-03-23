"""Tests for the data warehouse."""

from __future__ import annotations

import pandas as pd
import pytest

from data.providers import YFinanceProvider
from data.warehouse import DataWarehouse


@pytest.fixture
def warehouse(tmp_path):
    """Create a warehouse with a temp directory, always using yfinance."""
    return DataWarehouse(warehouse_dir=tmp_path, provider=YFinanceProvider())


class TestDataWarehouse:
    def test_download_single_ticker(self, warehouse):
        df = warehouse.download_ticker("AAPL", start="2024-01-02", end="2024-01-31")
        assert not df.empty
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_save_and_load_parquet(self, warehouse):
        warehouse.download_ticker("MSFT", start="2024-01-02", end="2024-01-31")
        loaded = warehouse.load("MSFT")
        assert not loaded.empty
        assert list(loaded.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_load_nonexistent_raises(self, warehouse):
        with pytest.raises(FileNotFoundError):
            warehouse.load("NONEXISTENT")

    def test_quality_checks(self, warehouse):
        warehouse.download_ticker("AAPL", start="2024-01-02", end="2024-03-31")
        checks = warehouse.check_quality("AAPL")
        assert checks["ticker"] == "AAPL"
        assert checks["rows"] > 0
        assert "gaps" in checks
        assert "zero_volume_days" in checks
        assert "stale_price_days" in checks

    def test_incremental_update(self, warehouse):
        # Download initial data
        warehouse.download_ticker("AAPL", start="2024-01-02", end="2024-01-15")
        initial = warehouse.load("AAPL")
        initial_len = len(initial)

        # Update should add more data (if available)
        # For test purposes, verify update doesn't crash
        warehouse.update()
        updated = warehouse.load("AAPL")
        assert len(updated) >= initial_len
