"""Tests for catalyst and event calendar pipeline."""

from __future__ import annotations

from datetime import date

import pandas as pd

from data.catalysts import (
    compute_days_to_earnings,
    compute_days_to_fomc,
    compute_event_score,
    get_fomc_dates,
)


class TestFOMCCalendar:
    def test_fomc_dates_not_empty(self) -> None:
        dates = get_fomc_dates()
        assert len(dates) > 0

    def test_fomc_dates_are_sorted(self) -> None:
        dates = get_fomc_dates()
        assert dates == sorted(dates)

    def test_fomc_dates_are_date_objects(self) -> None:
        dates = get_fomc_dates()
        for d in dates:
            assert isinstance(d, date)

    def test_days_to_fomc(self) -> None:
        dates = pd.DatetimeIndex(["2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30"])
        result = compute_days_to_fomc(dates)
        assert result.iloc[0] == 2  # 2 days before Jan 29 FOMC
        assert result.iloc[1] == 1  # 1 day before
        assert result.iloc[2] == 0  # FOMC day
        # After FOMC, should point to next meeting
        assert result.iloc[3] > 0


class TestDaysToEarnings:
    def test_days_to_upcoming_earnings(self) -> None:
        dates = pd.DatetimeIndex(["2025-01-10", "2025-01-15", "2025-01-20"])
        earnings_dates = [date(2025, 1, 20), date(2025, 4, 20)]

        result = compute_days_to_earnings(dates, earnings_dates)
        assert result.iloc[0] == 10  # 10 days to Jan 20
        assert result.iloc[1] == 5  # 5 days to Jan 20
        assert result.iloc[2] == 0  # Earnings day

    def test_empty_earnings(self) -> None:
        dates = pd.DatetimeIndex(["2025-01-10"])
        result = compute_days_to_earnings(dates, [])
        assert pd.isna(result.iloc[0])

    def test_past_earnings_negative(self) -> None:
        dates = pd.DatetimeIndex(["2025-01-25"])
        earnings_dates = [date(2025, 1, 20)]

        result = compute_days_to_earnings(dates, earnings_dates)
        assert result.iloc[0] < 0  # Past earnings = negative


class TestEventScore:
    def test_score_range(self) -> None:
        score = compute_event_score(days_to_earnings=5, days_to_fomc=3, sector="energy")
        assert 0.0 <= score <= 1.0

    def test_no_events_zero_score(self) -> None:
        score = compute_event_score(days_to_earnings=None, days_to_fomc=None, sector="software")
        assert score == 0.0

    def test_imminent_earnings_high_score(self) -> None:
        score_near = compute_event_score(
            days_to_earnings=1, days_to_fomc=None, sector="semiconductors"
        )
        score_far = compute_event_score(
            days_to_earnings=19, days_to_fomc=None, sector="semiconductors"
        )
        assert score_near > score_far

    def test_fomc_more_important_for_energy(self) -> None:
        score_energy = compute_event_score(days_to_earnings=None, days_to_fomc=2, sector="energy")
        score_software = compute_event_score(
            days_to_earnings=None, days_to_fomc=2, sector="software"
        )
        assert score_energy > score_software

    def test_nan_handling(self) -> None:
        score = compute_event_score(
            days_to_earnings=float("nan"),
            days_to_fomc=float("nan"),
            sector="cybersecurity",
        )
        assert score == 0.0
