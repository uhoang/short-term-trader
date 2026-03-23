"""Catalyst and event calendar pipeline for event-driven signals."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# FOMC meeting dates (2024-2026). Update quarterly.
FOMC_DATES: list[str] = [
    # 2024
    "2024-01-31",
    "2024-03-20",
    "2024-05-01",
    "2024-06-12",
    "2024-07-31",
    "2024-09-18",
    "2024-11-07",
    "2024-12-18",
    # 2025
    "2025-01-29",
    "2025-03-19",
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-17",
    # 2026
    "2026-01-28",
    "2026-03-18",
    "2026-04-29",
    "2026-06-17",
    "2026-07-29",
    "2026-09-16",
    "2026-11-04",
    "2026-12-16",
]


def get_fomc_dates() -> list[date]:
    """Return list of FOMC meeting dates."""
    return [datetime.strptime(d, "%Y-%m-%d").date() for d in FOMC_DATES]


def compute_days_to_fomc(dates: pd.DatetimeIndex) -> pd.Series:
    """Compute trading days until next FOMC meeting for each date.

    Returns Series with integer days-to-FOMC (0 on meeting day, negative after).
    """
    fomc = pd.to_datetime(FOMC_DATES)
    result = pd.Series(index=dates, dtype=float, name="days_to_fomc")

    for dt in dates:
        future = fomc[fomc >= dt]
        if len(future) > 0:
            result[dt] = (future[0] - dt).days
        else:
            result[dt] = np.nan

    return result


def get_earnings_dates(tickers: list[str]) -> dict[str, list[date]]:
    """Fetch upcoming earnings dates via yfinance.

    Returns dict of ticker -> list of earnings dates.
    Falls back gracefully if earnings data unavailable.
    """
    import yfinance as yf

    earnings: dict[str, list[date]] = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            cal = stock.get_earnings_dates(limit=8)
            if cal is not None and not cal.empty:
                dates = [d.date() for d in cal.index]
                earnings[ticker] = sorted(dates)
                logger.debug("earnings_fetched", ticker=ticker, count=len(dates))
            else:
                earnings[ticker] = []
        except Exception:
            logger.warning("earnings_fetch_failed", ticker=ticker)
            earnings[ticker] = []

    return earnings


def compute_days_to_earnings(
    dates: pd.DatetimeIndex,
    earnings_dates: list[date],
) -> pd.Series:
    """Compute days until next earnings date for each trading day.

    Returns Series of integer days (negative = days since last earnings).
    """
    result = pd.Series(index=dates, dtype=float, name="days_to_earnings")

    if not earnings_dates:
        return result

    earn_ts = pd.to_datetime(earnings_dates)

    for dt in dates:
        future = earn_ts[earn_ts >= dt]
        past = earn_ts[earn_ts < dt]

        if len(future) > 0:
            result[dt] = (future[0] - dt).days
        elif len(past) > 0:
            result[dt] = (dt - past[-1]).days * -1
        else:
            result[dt] = np.nan

    return result


# Sector relevance weights for different event types
SECTOR_EVENT_WEIGHTS: dict[str, dict[str, float]] = {
    "earnings": {
        "semiconductors": 1.0,
        "software": 1.0,
        "cybersecurity": 1.0,
        "energy": 1.0,
    },
    "fomc": {
        "semiconductors": 0.5,
        "software": 0.3,
        "cybersecurity": 0.3,
        "energy": 1.0,
    },
    "cyber_advisory": {
        "semiconductors": 0.5,
        "software": 0.8,
        "cybersecurity": 1.0,
        "energy": 0.2,
    },
    "export_control": {
        "semiconductors": 1.0,
        "software": 0.5,
        "cybersecurity": 0.3,
        "energy": 0.1,
    },
}


def compute_event_score(
    days_to_earnings: float | None,
    days_to_fomc: float | None,
    sector: str,
) -> float:
    """Compute composite event score (0-1) based on event proximity and sector.

    Score = sum(urgency_i * sector_relevance_i), capped at 1.0.
    Urgency decays as days increase: urgency = max(0, 1 - days/20).
    """
    score = 0.0

    # Earnings urgency
    if days_to_earnings is not None and not np.isnan(days_to_earnings):
        days = abs(days_to_earnings)
        if days <= 20:
            urgency = max(0.0, 1.0 - days / 20.0)
            weight = SECTOR_EVENT_WEIGHTS["earnings"].get(sector, 0.5)
            score += urgency * weight * 0.5  # Earnings contributes up to 0.5

    # FOMC urgency
    if days_to_fomc is not None and not np.isnan(days_to_fomc):
        days = abs(days_to_fomc)
        if days <= 10:
            urgency = max(0.0, 1.0 - days / 10.0)
            weight = SECTOR_EVENT_WEIGHTS["fomc"].get(sector, 0.3)
            score += urgency * weight * 0.3  # FOMC contributes up to 0.3

    return min(score, 1.0)
