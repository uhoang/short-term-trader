"""Tests for the universe loader."""

from __future__ import annotations

import pytest

from scanner.universe import Universe


class TestUniverse:
    def test_loads_successfully(self, universe: Universe) -> None:
        assert universe is not None

    def test_has_four_sectors(self, universe: Universe) -> None:
        sectors = universe.get_sectors()
        assert len(sectors) == 4
        assert "semiconductors" in sectors
        assert "software" in sectors
        assert "cybersecurity" in sectors
        assert "energy" in sectors

    def test_has_tickers(self, universe: Universe) -> None:
        tickers = universe.get_tickers()
        assert len(tickers) >= 60

    def test_each_sector_has_tickers(self, universe: Universe) -> None:
        for sector in universe.get_sectors():
            tickers = universe.get_tickers(sector=sector)
            assert len(tickers) >= 1, f"{sector} has no tickers"

    def test_sector_etf_benchmarks(self, universe: Universe) -> None:
        assert universe.get_sector_etf("semiconductors") == "SMH"
        assert universe.get_sector_etf("software") == "IGV"
        assert universe.get_sector_etf("cybersecurity") == "CIBR"
        assert universe.get_sector_etf("energy") == "XLE"

    def test_get_all_etfs(self, universe: Universe) -> None:
        etfs = universe.get_all_etfs()
        assert set(etfs) == {"SMH", "IGV", "CIBR", "XLE"}

    def test_ticker_sector_lookup(self, universe: Universe) -> None:
        assert universe.get_ticker_sector("NVDA") == "semiconductors"
        assert universe.get_ticker_sector("MSFT") == "software"
        assert universe.get_ticker_sector("XOM") == "energy"
        assert universe.get_ticker_sector("CRWD") == "cybersecurity"
        assert universe.get_ticker_sector("FAKE") is None

    def test_unknown_sector_raises(self, universe: Universe) -> None:
        with pytest.raises(ValueError, match="Unknown sector"):
            universe.get_tickers(sector="biotech")

    def test_metadata_present(self, universe: Universe) -> None:
        meta = universe.metadata
        assert meta["min_avg_volume_30d"] == 2_000_000
        assert meta["max_weight_per_position"] == 0.05
        assert meta["max_sector_weight"] == 0.40

    def test_unique_tickers(self, universe: Universe) -> None:
        unique = universe.get_unique_tickers()
        assert len(unique) == len(set(universe.get_tickers()))
