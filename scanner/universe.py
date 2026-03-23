"""Universe loader and sector mapping for the 60-stock trading universe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "universe.json"


class Universe:
    """Loads and queries the stock universe from config/universe.json."""

    def __init__(self, config_path: Path | str = CONFIG_PATH) -> None:
        self._config_path = Path(config_path)
        self._data = self._load()
        logger.info(
            "universe_loaded",
            total_tickers=len(self.get_tickers()),
            sectors=list(self._data["sectors"].keys()),
        )

    def _load(self) -> dict[str, Any]:
        with open(self._config_path) as f:
            data = json.load(f)
        self._validate(data)
        return data

    def _validate(self, data: dict[str, Any]) -> None:
        """Validate universe config structure and constraints."""
        assert "metadata" in data, "Missing 'metadata' in universe config"
        assert "sectors" in data, "Missing 'sectors' in universe config"

        all_tickers: list[str] = []
        for sector_name, sector_data in data["sectors"].items():
            assert "etf_benchmark" in sector_data, f"Missing ETF benchmark for {sector_name}"
            assert "tickers" in sector_data, f"Missing tickers for {sector_name}"
            assert len(sector_data["tickers"]) > 0, f"Empty ticker list for {sector_name}"
            all_tickers.extend(sector_data["tickers"])

        total = len(all_tickers)
        unique = len(set(all_tickers))
        if total != unique:
            dupes = [t for t in all_tickers if all_tickers.count(t) > 1]
            logger.warning("duplicate_tickers_in_universe", duplicates=list(set(dupes)))

    @property
    def metadata(self) -> dict[str, Any]:
        return self._data["metadata"]

    def get_sectors(self) -> list[str]:
        """Return list of sector names."""
        return list(self._data["sectors"].keys())

    def get_tickers(self, sector: str | None = None) -> list[str]:
        """Return tickers, optionally filtered by sector."""
        if sector:
            if sector not in self._data["sectors"]:
                raise ValueError(f"Unknown sector: {sector}")
            return list(self._data["sectors"][sector]["tickers"])

        all_tickers: list[str] = []
        for sector_data in self._data["sectors"].values():
            all_tickers.extend(sector_data["tickers"])
        return all_tickers

    def get_unique_tickers(self) -> list[str]:
        """Return deduplicated list of all tickers."""
        return sorted(set(self.get_tickers()))

    def get_sector_etf(self, sector: str) -> str:
        """Return the ETF benchmark ticker for a sector."""
        if sector not in self._data["sectors"]:
            raise ValueError(f"Unknown sector: {sector}")
        return self._data["sectors"][sector]["etf_benchmark"]

    def get_all_etfs(self) -> list[str]:
        """Return all sector ETF benchmark tickers."""
        return [s["etf_benchmark"] for s in self._data["sectors"].values()]

    def get_ticker_sector(self, ticker: str) -> str | None:
        """Return the sector for a given ticker, or None if not found."""
        for sector_name, sector_data in self._data["sectors"].items():
            if ticker in sector_data["tickers"]:
                return sector_name
        return None

    def add_ticker(self, ticker: str, sector: str) -> bool:
        """Add a ticker to a sector. Returns True if added, False if duplicate."""
        ticker = ticker.upper().strip()
        if not ticker:
            return False
        if sector not in self._data["sectors"]:
            return False
        if ticker in self._data["sectors"][sector]["tickers"]:
            return False
        # Remove from other sectors if present
        for s_data in self._data["sectors"].values():
            if ticker in s_data["tickers"]:
                s_data["tickers"].remove(ticker)
        self._data["sectors"][sector]["tickers"].append(ticker)
        return True

    def remove_ticker(self, ticker: str) -> bool:
        """Remove a ticker from the universe. Returns True if removed."""
        for sector_data in self._data["sectors"].values():
            if ticker in sector_data["tickers"]:
                sector_data["tickers"].remove(ticker)
                return True
        return False

    def add_sector(self, name: str, etf_benchmark: str) -> bool:
        """Add a new sector. Returns True if added, False if exists."""
        name = name.lower().strip().replace(" ", "_")
        if name in self._data["sectors"]:
            return False
        self._data["sectors"][name] = {
            "etf_benchmark": etf_benchmark.upper().strip(),
            "tickers": [],
        }
        return True

    def save(self, path: Path | str | None = None) -> None:
        """Save current universe to JSON."""
        path = Path(path) if path else self._config_path
        with open(path, "w") as f:
            json.dump(self._data, f, indent=2)
        logger.info(
            "universe_saved",
            total_tickers=len(self.get_tickers()),
            sectors=list(self._data["sectors"].keys()),
        )
