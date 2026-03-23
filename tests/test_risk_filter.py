"""Tests for risk filter module."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from signals.base import Direction, SignalEvent
from signals.risk_filter import PortfolioState, RiskFilter, RiskFilterConfig
from signals.strategy import TradeParams


def _signal(
    ticker: str = "NVDA",
    sector: str = "semiconductors",
) -> SignalEvent:
    return SignalEvent(
        ticker=ticker,
        direction=Direction.LONG,
        strength=0.7,
        strategy_id="test",
        timestamp=datetime(2024, 1, 15),
        metadata={
            "sector": sector,
            "trade_params": TradeParams(
                entry_price=150.0,
                stop_loss_pct=0.07,
                take_profit_pct=0.15,
                max_hold_days=10,
            ),
        },
    )


class TestDrawdownKillSwitch:
    def test_passes_in_normal_conditions(self) -> None:
        rf = RiskFilter()
        portfolio = PortfolioState(equity=1_000_000, peak_equity=1_000_000)
        result = rf.check(_signal(), portfolio)
        assert result.passed

    def test_triggers_on_deep_drawdown(self) -> None:
        rf = RiskFilter()
        portfolio = PortfolioState(equity=880_000, peak_equity=1_000_000)  # -12%
        result = rf.check(_signal(), portfolio)
        assert not result.passed
        assert "Kill switch" in result.reason

    def test_resumes_after_recovery(self) -> None:
        rf = RiskFilter()
        # Trigger kill switch
        portfolio_dd = PortfolioState(equity=880_000, peak_equity=1_000_000)
        rf.check(_signal(), portfolio_dd)

        # Still in drawdown but recovering (-4%)
        portfolio_recover = PortfolioState(equity=960_000, peak_equity=1_000_000)
        result = rf.check(_signal(), portfolio_recover)
        assert result.passed


class TestSectorConcentration:
    def test_passes_under_limit(self) -> None:
        rf = RiskFilter()
        portfolio = PortfolioState(
            equity=1_000_000,
            peak_equity=1_000_000,
            open_positions={
                "AMD": {"sector": "semiconductors", "value": 100_000},
            },
        )
        result = rf.check(_signal(sector="semiconductors"), portfolio)
        assert result.passed

    def test_rejects_over_sector_cap(self) -> None:
        rf = RiskFilter()
        portfolio = PortfolioState(
            equity=1_000_000,
            peak_equity=1_000_000,
            open_positions={
                "AMD": {"sector": "semiconductors", "value": 200_000},
                "AVGO": {"sector": "semiconductors", "value": 200_000},
            },
        )
        result = rf.check(_signal(sector="semiconductors"), portfolio)
        assert not result.passed
        assert "Sector" in result.reason


class TestPositionLimit:
    def test_rejects_at_max_positions(self) -> None:
        rf = RiskFilter(config=RiskFilterConfig(max_concurrent_positions=3))
        portfolio = PortfolioState(
            equity=1_000_000,
            peak_equity=1_000_000,
            open_positions={
                "A": {"sector": "s", "value": 50_000},
                "B": {"sector": "s", "value": 50_000},
                "C": {"sector": "s", "value": 50_000},
            },
        )
        result = rf.check(_signal(), portfolio)
        assert not result.passed
        assert "Max positions" in result.reason


class TestCorrelationCheck:
    def test_rejects_high_correlation(self) -> None:
        rf = RiskFilter()
        np.random.seed(42)
        dates = pd.bdate_range("2023-01-02", periods=30)
        base = np.random.normal(0, 0.02, 30)

        returns_history = {
            "NVDA": pd.Series(base, index=dates),
            "AMD": pd.Series(base * 1.01 + 0.001, index=dates),  # ~1.0 corr
        }
        portfolio = PortfolioState(
            equity=1_000_000,
            peak_equity=1_000_000,
            open_positions={"AMD": {"sector": "semiconductors", "value": 50_000}},
        )
        result = rf.check(_signal(ticker="NVDA"), portfolio, returns_history)
        assert not result.passed
        assert "Correlation" in result.reason

    def test_passes_low_correlation(self) -> None:
        rf = RiskFilter()
        np.random.seed(42)
        dates = pd.bdate_range("2023-01-02", periods=30)

        returns_history = {
            "NVDA": pd.Series(np.random.normal(0, 0.02, 30), index=dates),
            "XOM": pd.Series(np.random.normal(0, 0.02, 30), index=dates),  # Independent
        }
        portfolio = PortfolioState(
            equity=1_000_000,
            peak_equity=1_000_000,
            open_positions={"XOM": {"sector": "energy", "value": 50_000}},
        )
        result = rf.check(_signal(ticker="NVDA"), portfolio, returns_history)
        assert result.passed
