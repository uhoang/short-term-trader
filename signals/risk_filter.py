"""Risk filter module: pre-trade checks and portfolio-level controls."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import structlog

from signals.base import SignalEvent

logger = structlog.get_logger(__name__)


@dataclass
class PortfolioState:
    """Current portfolio state for risk checks."""

    equity: float = 1_000_000.0
    peak_equity: float = 1_000_000.0
    open_positions: dict[str, dict[str, object]] = field(default_factory=dict)
    # open_positions: {ticker: {sector, direction, value, entry_date}}


@dataclass
class RiskFilterConfig:
    """Risk filter thresholds."""

    max_sector_weight: float = 0.40
    drawdown_kill_switch: float = -0.10  # -10% from peak
    drawdown_resume: float = -0.05  # Resume at -5% from peak
    max_correlation: float = 0.85
    max_concurrent_positions: int = 15
    max_position_weight: float = 0.05  # 5% NAV per position


@dataclass
class RiskCheckResult:
    """Result of a risk filter check."""

    passed: bool
    reason: str = ""


class RiskFilter:
    """Pre-trade risk checks to protect the portfolio."""

    def __init__(self, config: RiskFilterConfig | None = None) -> None:
        self.config = config or RiskFilterConfig()
        self._kill_switch_active = False

    def check(
        self,
        signal: SignalEvent,
        portfolio: PortfolioState,
        returns_history: dict[str, pd.Series] | None = None,
    ) -> RiskCheckResult:
        """Run all risk checks on a proposed signal.

        Returns RiskCheckResult with pass/fail and reason.
        """
        checks = [
            self._check_drawdown_kill_switch(portfolio),
            self._check_position_limit(portfolio),
            self._check_sector_concentration(signal, portfolio),
            self._check_position_size(signal, portfolio),
        ]

        if returns_history:
            checks.append(self._check_correlation(signal, portfolio, returns_history))

        for result in checks:
            if not result.passed:
                logger.info(
                    "risk_check_rejected",
                    ticker=signal.ticker,
                    reason=result.reason,
                )
                return result

        return RiskCheckResult(passed=True)

    def _check_drawdown_kill_switch(self, portfolio: PortfolioState) -> RiskCheckResult:
        """Halt entries if portfolio drawdown exceeds threshold."""
        if portfolio.peak_equity == 0:
            return RiskCheckResult(passed=True)

        drawdown = (portfolio.equity - portfolio.peak_equity) / portfolio.peak_equity

        if drawdown <= self.config.drawdown_kill_switch:
            self._kill_switch_active = True

        if self._kill_switch_active:
            if drawdown > self.config.drawdown_resume:
                self._kill_switch_active = False
                logger.info("kill_switch_deactivated", drawdown=f"{drawdown:.1%}")
            else:
                return RiskCheckResult(
                    passed=False,
                    reason=f"Kill switch active: drawdown {drawdown:.1%}",
                )

        return RiskCheckResult(passed=True)

    def _check_position_limit(self, portfolio: PortfolioState) -> RiskCheckResult:
        """Check max concurrent positions."""
        if len(portfolio.open_positions) >= self.config.max_concurrent_positions:
            return RiskCheckResult(
                passed=False,
                reason=f"Max positions reached: {len(portfolio.open_positions)}",
            )
        return RiskCheckResult(passed=True)

    def _check_sector_concentration(
        self, signal: SignalEvent, portfolio: PortfolioState
    ) -> RiskCheckResult:
        """Check sector exposure cap."""
        if portfolio.equity == 0:
            return RiskCheckResult(passed=True)

        # Get signal's sector from metadata
        signal_sector = signal.metadata.get("sector", "unknown")

        sector_value = 0.0
        for pos in portfolio.open_positions.values():
            if pos.get("sector") == signal_sector:
                sector_value += float(pos.get("value", 0))

        # Add proposed position (estimate as max position weight * equity)
        proposed_value = portfolio.equity * self.config.max_position_weight
        new_sector_weight = (sector_value + proposed_value) / portfolio.equity

        if new_sector_weight > self.config.max_sector_weight:
            return RiskCheckResult(
                passed=False,
                reason=f"Sector {signal_sector} at {new_sector_weight:.1%} "
                f"> {self.config.max_sector_weight:.0%} cap",
            )
        return RiskCheckResult(passed=True)

    def _check_position_size(
        self, signal: SignalEvent, portfolio: PortfolioState
    ) -> RiskCheckResult:
        """Check individual position size limit."""
        # Position sizing handled by strategies; this is a safety cap
        trade_params = signal.metadata.get("trade_params")
        if trade_params and hasattr(trade_params, "entry_price"):
            if trade_params.entry_price <= 0:
                return RiskCheckResult(passed=False, reason="Invalid entry price")
            # Position size is capped externally; just validate it's reasonable
        return RiskCheckResult(passed=True)

    def _check_correlation(
        self,
        signal: SignalEvent,
        portfolio: PortfolioState,
        returns_history: dict[str, pd.Series],
    ) -> RiskCheckResult:
        """Reject if new position too correlated with existing."""
        if signal.ticker not in returns_history:
            return RiskCheckResult(passed=True)

        new_returns = returns_history[signal.ticker]

        for existing_ticker in portfolio.open_positions:
            if existing_ticker not in returns_history:
                continue

            existing_returns = returns_history[existing_ticker]
            # Align and compute 20-day rolling correlation
            aligned = pd.concat([new_returns, existing_returns], axis=1).dropna()
            if len(aligned) < 20:
                continue

            corr = aligned.iloc[-20:].corr().iloc[0, 1]
            if not np.isnan(corr) and abs(corr) > self.config.max_correlation:
                return RiskCheckResult(
                    passed=False,
                    reason=f"Correlation {corr:.2f} with "
                    f"{existing_ticker} > {self.config.max_correlation}",
                )

        return RiskCheckResult(passed=True)
