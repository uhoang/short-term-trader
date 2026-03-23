"""Tests for data providers."""

from __future__ import annotations

import pandas as pd
import pytest

from data.providers import YFinanceProvider, get_provider


class TestYFinanceProvider:
    @pytest.fixture
    def provider(self) -> YFinanceProvider:
        return YFinanceProvider()

    def test_fetches_ohlcv(self, provider: YFinanceProvider) -> None:
        df = provider.get_ohlcv("AAPL", start="2024-01-02", end="2024-01-10")
        assert not df.empty
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_returns_empty_for_invalid_ticker(self, provider: YFinanceProvider) -> None:
        df = provider.get_ohlcv("ZZZZZZNOTREAL", start="2024-01-02", end="2024-01-10")
        assert df.empty


class TestProviderFactory:
    def test_yfinance_when_preferred(self) -> None:
        """Explicitly requesting yfinance should return YFinanceProvider."""
        provider = get_provider(preferred="yfinance")
        assert isinstance(provider, YFinanceProvider)
