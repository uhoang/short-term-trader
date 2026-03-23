"""Scanner report: captures per-ticker evaluation results across all strategies."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

REPORT_PATH = Path(__file__).parent.parent / "warehouse" / "scan_report.json"


def save_scan_report(
    evaluations: list[dict],
    signals: list[dict],
    path: Path | str = REPORT_PATH,
) -> None:
    """Save scan report to JSON.

    Args:
        evaluations: list of per-ticker strategy evaluation dicts
        signals: list of final filtered signal dicts
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert signals to serializable form
    clean_signals = []
    for sig in signals:
        clean_signals.append(
            {
                "ticker": sig.get("ticker", ""),
                "direction": sig.get("direction", ""),
                "strength": sig.get("strength", 0),
                "strategy_id": sig.get("strategy_id", ""),
            }
        )

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_tickers_scanned": len({e["ticker"] for e in evaluations}),
        "signals_generated": len(clean_signals),
        "signals": clean_signals,
        "evaluations": evaluations,
    }

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(
        "scan_report_saved",
        tickers=report["total_tickers_scanned"],
        signals=report["signals_generated"],
    )


def load_scan_report(path: Path | str = REPORT_PATH) -> dict:
    """Load the most recent scan report."""
    path = Path(path)
    if not path.exists():
        return {"timestamp": None, "evaluations": [], "signals": []}
    with open(path) as f:
        return json.load(f)
