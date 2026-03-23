"""Data provider abstraction for OHLCV market data."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
import structlog
from dotenv import load_dotenv

load_dotenv()

logger = structlog.get_logger(__name__)


class DataProvider(ABC):
    """Base class for market data providers."""

    @abstractmethod
    def get_ohlcv(
        self,
        ticker: str,
        start: str | datetime,
        end: str | datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a ticker.

        Returns DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex (date only, no intraday).
        """
        ...


class YFinanceProvider(DataProvider):
    """Free-tier historical data via yfinance."""

    def get_ohlcv(
        self,
        ticker: str,
        start: str | datetime,
        end: str | datetime | None = None,
    ) -> pd.DataFrame:
        import yfinance as yf

        logger.info("fetching_ohlcv", provider="yfinance", ticker=ticker, start=str(start))
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

        if data.empty:
            logger.warning("no_data_returned", provider="yfinance", ticker=ticker)
            return pd.DataFrame()

        # yfinance may return MultiIndex columns for single ticker; flatten
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        expected = ["Open", "High", "Low", "Close", "Volume"]
        return data[expected]


class PolygonProvider(DataProvider):
    """Polygon.io REST API provider (requires API key)."""

    def __init__(self) -> None:
        self.api_key = os.getenv("POLYGON_API_KEY", "")
        if not self.api_key or self.api_key.startswith("your_"):
            logger.warning("polygon_api_key_not_configured")

    def get_ohlcv(
        self,
        ticker: str,
        start: str | datetime,
        end: str | datetime | None = None,
    ) -> pd.DataFrame:
        if not self.api_key or self.api_key.startswith("your_"):
            raise RuntimeError("Polygon API key not configured. Set POLYGON_API_KEY in .env")

        from polygon import RESTClient

        logger.info("fetching_ohlcv", provider="polygon", ticker=ticker, start=str(start))
        client = RESTClient(api_key=self.api_key)

        start_str = start if isinstance(start, str) else start.strftime("%Y-%m-%d")
        end_str = (
            end
            if isinstance(end, str)
            else (end.strftime("%Y-%m-%d") if end else datetime.now().strftime("%Y-%m-%d"))
        )

        aggs = list(
            client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start_str,
                to=end_str,
                limit=50000,
            )
        )

        if not aggs:
            logger.warning("no_data_returned", provider="polygon", ticker=ticker)
            return pd.DataFrame()

        rows = [
            {
                "Date": pd.Timestamp(a.timestamp, unit="ms"),
                "Open": a.open,
                "High": a.high,
                "Low": a.low,
                "Close": a.close,
                "Volume": a.volume,
            }
            for a in aggs
        ]
        df = pd.DataFrame(rows).set_index("Date")
        return df


def get_provider(preferred: str = "polygon") -> DataProvider:
    """Factory that returns the best available data provider.

    Falls back to yfinance if Polygon is not configured.
    """
    if preferred == "polygon":
        api_key = os.getenv("POLYGON_API_KEY", "")
        if api_key and not api_key.startswith("your_"):
            return PolygonProvider()
        logger.info("polygon_not_configured_falling_back_to_yfinance")

    return YFinanceProvider()
