"""Historical OHLCV data warehouse backed by Parquet files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import structlog

from data.providers import DataProvider, get_provider
from scanner.universe import Universe

logger = structlog.get_logger(__name__)

WAREHOUSE_DIR = Path(__file__).parent.parent / "warehouse" / "parquet"


class DataWarehouse:
    """Parquet-backed historical OHLCV store for the trading universe."""

    def __init__(
        self,
        warehouse_dir: Path | str = WAREHOUSE_DIR,
        provider: DataProvider | None = None,
    ) -> None:
        self.warehouse_dir = Path(warehouse_dir)
        self.warehouse_dir.mkdir(parents=True, exist_ok=True)
        self.provider = provider or get_provider()
        self.universe = Universe()

    def _ticker_path(self, ticker: str) -> Path:
        return self.warehouse_dir / f"{ticker}.parquet"

    def download_ticker(
        self,
        ticker: str,
        start: str = "2018-01-01",
        end: str | None = None,
    ) -> pd.DataFrame:
        """Download and save OHLCV data for a single ticker."""
        logger.info("downloading_ticker", ticker=ticker, start=start)
        df = self.provider.get_ohlcv(ticker, start=start, end=end)

        if df.empty:
            logger.warning("empty_download", ticker=ticker)
            return df

        df.to_parquet(self._ticker_path(ticker))
        logger.info("saved_parquet", ticker=ticker, rows=len(df))
        return df

    def download_all(self, start: str = "2018-01-01") -> dict[str, int]:
        """Download OHLCV for all universe tickers + sector ETFs.

        Returns dict of ticker -> row count.
        """
        tickers = self.universe.get_unique_tickers() + self.universe.get_all_etfs()
        results: dict[str, int] = {}

        for i, ticker in enumerate(tickers):
            logger.info("download_progress", current=i + 1, total=len(tickers), ticker=ticker)
            try:
                df = self.download_ticker(ticker, start=start)
                results[ticker] = len(df)
            except Exception:
                logger.exception("download_failed", ticker=ticker)
                results[ticker] = 0

        successful = sum(1 for v in results.values() if v > 0)
        logger.info(
            "download_complete",
            total=len(tickers),
            successful=successful,
            failed=len(tickers) - successful,
        )
        return results

    def update(self, force_refresh: bool = False) -> dict[str, int]:
        """Update warehouse with latest data.

        Args:
            force_refresh: If True, re-download full history for all tickers
                to fix dividend adjustment inconsistencies.

        Returns dict of ticker -> new rows appended.
        """
        tickers = self.universe.get_unique_tickers() + self.universe.get_all_etfs()
        results: dict[str, int] = {}
        full_start = self.universe._data["meta"].get("data_start_date", "2018-01-01")

        for ticker in tickers:
            path = self._ticker_path(ticker)

            if force_refresh or not path.exists():
                df = self.download_ticker(ticker, start=full_start)
                results[ticker] = len(df)
                continue

            existing = pd.read_parquet(path)
            existing = existing[~existing.index.duplicated(keep="last")]
            last_date = existing.index.max()
            start_next = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            if start_next >= datetime.now().strftime("%Y-%m-%d"):
                results[ticker] = 0
                continue

            # Re-download full history to ensure consistent dividend adjustment.
            # yfinance auto_adjust factors change over time, so appending new
            # rows to old adjusted data creates price discontinuities.
            full_data = self.provider.get_ohlcv(ticker, start=full_start)
            if full_data.empty:
                results[ticker] = 0
                continue

            full_data = full_data[~full_data.index.duplicated(keep="last")]
            full_data.sort_index(inplace=True)
            new_rows = len(full_data) - len(existing)
            full_data.to_parquet(path)
            results[ticker] = max(new_rows, 0)
            if new_rows > 0:
                logger.info("updated_ticker", ticker=ticker, new_rows=new_rows)

        return results

    def load(self, ticker: str) -> pd.DataFrame:
        """Load OHLCV data for a single ticker from Parquet."""
        path = self._ticker_path(ticker)
        if not path.exists():
            raise FileNotFoundError(f"No data for {ticker}. Run download first.")
        return pd.read_parquet(path)

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all tickers from warehouse."""
        result: dict[str, pd.DataFrame] = {}
        for path in sorted(self.warehouse_dir.glob("*.parquet")):
            ticker = path.stem
            result[ticker] = pd.read_parquet(path)
        return result

    def check_quality(self, ticker: str) -> dict[str, object]:
        """Run data quality checks on a ticker's OHLCV data.

        Returns dict with check results.
        """
        df = self.load(ticker)
        checks: dict[str, object] = {"ticker": ticker, "rows": len(df)}

        # Gap detection: find missing trading days
        if len(df) > 1:
            date_diffs = df.index.to_series().diff().dt.days
            # Gaps > 4 days (accounts for weekends + holidays)
            gaps = date_diffs[date_diffs > 4]
            checks["gaps"] = len(gaps)
            if len(gaps) > 0:
                checks["gap_dates"] = gaps.index.strftime("%Y-%m-%d").tolist()
        else:
            checks["gaps"] = 0

        # Stale price: unchanged close for 5+ consecutive days
        stale = (df["Close"].diff() == 0).rolling(5).sum() >= 5
        checks["stale_price_days"] = int(stale.sum())

        # Zero volume days
        zero_vol = (df["Volume"] == 0).sum()
        checks["zero_volume_days"] = int(zero_vol)

        # Date range
        checks["start_date"] = df.index.min().strftime("%Y-%m-%d")
        checks["end_date"] = df.index.max().strftime("%Y-%m-%d")

        return checks
