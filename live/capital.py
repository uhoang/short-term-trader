"""Capital ramp protocol and strategy rotation management."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RampLevel:
    """A capital ramp level with entry criteria."""

    allocation_pct: float  # Fraction of intended capital
    min_days: int  # Minimum days at prior level
    min_sharpe: float  # Required Sharpe to advance


# Default ramp schedule: 10% → 25% → 50% → 100%
DEFAULT_RAMP = [
    RampLevel(allocation_pct=0.10, min_days=0, min_sharpe=0.0),  # Start
    RampLevel(allocation_pct=0.25, min_days=7, min_sharpe=1.0),
    RampLevel(allocation_pct=0.50, min_days=30, min_sharpe=1.0),
    RampLevel(allocation_pct=1.00, min_days=60, min_sharpe=0.8),
]


class CapitalManager:
    """Manages capital allocation and strategy rotation."""

    def __init__(
        self,
        intended_capital: float = 1_000_000.0,
        ramp_schedule: list[RampLevel] | None = None,
        strategy_disable_sharpe: float = 0.4,
    ) -> None:
        self.intended_capital = intended_capital
        self.ramp_schedule = ramp_schedule or DEFAULT_RAMP
        self.strategy_disable_sharpe = strategy_disable_sharpe
        self._current_level = 0
        self._days_at_level = 0

    def get_allocation(self) -> float:
        """Get current capital allocation as a dollar amount."""
        level = self.ramp_schedule[self._current_level]
        return self.intended_capital * level.allocation_pct

    def get_allocation_pct(self) -> float:
        """Get current allocation as a percentage."""
        return self.ramp_schedule[self._current_level].allocation_pct

    def get_position_sizing_multiplier(self) -> float:
        """Get position sizing multiplier (0.5 for early levels, 1.0 at full)."""
        pct = self.get_allocation_pct()
        if pct < 0.50:
            return 0.5  # Conservative sizing during ramp
        return 1.0

    def should_ramp_up(self, live_sharpe: float, days_live: int) -> bool:
        """Check if we should advance to the next capital level."""
        if self._current_level >= len(self.ramp_schedule) - 1:
            return False  # Already at max

        next_level = self.ramp_schedule[self._current_level + 1]
        if days_live >= next_level.min_days and live_sharpe >= next_level.min_sharpe:
            return True
        return False

    def ramp_up(self) -> float:
        """Advance to the next capital level. Returns new allocation."""
        if self._current_level >= len(self.ramp_schedule) - 1:
            return self.get_allocation()

        self._current_level += 1
        self._days_at_level = 0
        new_alloc = self.get_allocation()
        logger.info(
            "capital_ramped_up",
            level=self._current_level,
            allocation=new_alloc,
            pct=f"{self.get_allocation_pct():.0%}",
        )
        return new_alloc

    def should_disable_strategy(self, strategy_id: str, rolling_sharpe_60d: float) -> bool:
        """Check if a strategy should be disabled due to poor performance."""
        if rolling_sharpe_60d < self.strategy_disable_sharpe:
            logger.warning(
                "strategy_underperforming",
                strategy=strategy_id,
                sharpe_60d=f"{rolling_sharpe_60d:.2f}",
                threshold=self.strategy_disable_sharpe,
            )
            return True
        return False

    def update_day(self) -> None:
        """Increment days counter at current level."""
        self._days_at_level += 1

    @property
    def current_level(self) -> int:
        return self._current_level

    @property
    def days_at_level(self) -> int:
        return self._days_at_level
