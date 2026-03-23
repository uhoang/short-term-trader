"""Tests for scanner report generation."""

from __future__ import annotations

import json

from live.scanner_report import load_scan_report, save_scan_report


class TestScannerReport:
    def test_save_and_load(self, tmp_path) -> None:
        path = tmp_path / "report.json"
        evaluations = [
            {
                "strategy_id": "catalyst_capture",
                "ticker": "NVDA",
                "triggered": True,
                "conditions": [
                    {"name": "Event Score", "value": 0.8, "threshold": 0.3, "passed": True},
                ],
            },
            {
                "strategy_id": "catalyst_capture",
                "ticker": "AMD",
                "triggered": False,
                "conditions": [
                    {"name": "Event Score", "value": 0.1, "threshold": 0.3, "passed": False},
                ],
            },
        ]
        signals = [
            {"ticker": "NVDA", "direction": "long", "strength": 0.8, "strategy_id": "catalyst"},
        ]

        save_scan_report(evaluations, signals, path)
        loaded = load_scan_report(path)

        assert loaded["total_tickers_scanned"] == 2
        assert loaded["signals_generated"] == 1
        assert len(loaded["evaluations"]) == 2
        assert loaded["timestamp"] is not None

    def test_load_missing_file(self, tmp_path) -> None:
        path = tmp_path / "nonexistent.json"
        report = load_scan_report(path)
        assert report["evaluations"] == []
        assert report["signals"] == []

    def test_signal_serialization(self, tmp_path) -> None:
        path = tmp_path / "report.json"
        signals = [
            {"ticker": "MSFT", "direction": "long", "strength": 0.75, "strategy_id": "test"},
        ]
        save_scan_report([], signals, path)

        with open(path) as f:
            data = json.load(f)
        assert data["signals"][0]["ticker"] == "MSFT"
        assert data["signals"][0]["strength"] == 0.75
