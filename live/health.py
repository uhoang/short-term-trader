"""Health monitoring for strategies and system components."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class HealthCheck:
    """Result of a single health check."""

    name: str
    passed: bool
    message: str = ""
    value: float | None = None


@dataclass
class HealthReport:
    """Aggregate health report."""

    checks: list[HealthCheck] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def summary(self) -> dict[str, object]:
        return {
            "all_passed": self.all_passed,
            "total_checks": len(self.checks),
            "failed": [c.name for c in self.checks if not c.passed],
            "timestamp": self.timestamp,
        }


class HealthMonitor:
    """Monitors strategy health and system status."""

    def __init__(self, sharpe_warning_threshold: float = 0.5) -> None:
        self.sharpe_warning = sharpe_warning_threshold

    def check_strategy_health(
        self,
        trades: list[dict],
        window_days: int = 30,
    ) -> dict[str, HealthCheck]:
        """Check rolling Sharpe for each strategy over recent trades."""
        results: dict[str, HealthCheck] = {}

        # Group trades by strategy
        by_strategy: dict[str, list[float]] = {}
        cutoff = datetime.now().timestamp() - window_days * 86400

        for trade in trades:
            exit_date = trade.get("exit_date", "")
            try:
                if isinstance(exit_date, str):
                    ts = datetime.fromisoformat(exit_date).timestamp()
                else:
                    ts = exit_date.timestamp()
                if ts < cutoff:
                    continue
            except (ValueError, AttributeError):
                continue

            strategy = trade.get("strategy_id", "unknown")
            ret = trade.get("return_pct", 0)
            by_strategy.setdefault(strategy, []).append(ret)

        for strategy, returns in by_strategy.items():
            if len(returns) < 3:
                results[strategy] = HealthCheck(
                    name=f"strategy_{strategy}",
                    passed=True,
                    message="Too few trades for assessment",
                )
                continue

            arr = np.array(returns)
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            sharpe = (mean / std * np.sqrt(20)) if std > 0 else 0  # ~20 trades/year

            passed = sharpe >= self.sharpe_warning
            results[strategy] = HealthCheck(
                name=f"strategy_{strategy}",
                passed=passed,
                message=f"30d Sharpe: {sharpe:.2f}"
                + ("" if passed else " (WARNING: below threshold)"),
                value=sharpe,
            )

            if not passed:
                logger.warning(
                    "strategy_health_warning",
                    strategy=strategy,
                    sharpe_30d=f"{sharpe:.2f}",
                )

        return results

    def check_data_freshness(self, last_update: datetime | str | None) -> HealthCheck:
        """Check if market data was updated today."""
        if last_update is None:
            return HealthCheck(
                name="data_freshness",
                passed=False,
                message="No data update timestamp found",
            )

        if isinstance(last_update, str):
            last_update = datetime.fromisoformat(last_update)

        age_hours = (datetime.now() - last_update).total_seconds() / 3600
        passed = age_hours < 24

        return HealthCheck(
            name="data_freshness",
            passed=passed,
            message=f"Last update: {age_hours:.1f}h ago",
            value=age_hours,
        )

    def check_position_count(self, open_positions: int, max_positions: int = 15) -> HealthCheck:
        """Check if position count is within limits."""
        utilization = open_positions / max_positions if max_positions > 0 else 0
        return HealthCheck(
            name="position_utilization",
            passed=open_positions <= max_positions,
            message=f"{open_positions}/{max_positions} positions ({utilization:.0%})",
            value=utilization,
        )

    def run_all_checks(
        self,
        trades: list[dict] | None = None,
        last_data_update: datetime | str | None = None,
        open_position_count: int = 0,
    ) -> HealthReport:
        """Run all health checks and return aggregate report."""
        report = HealthReport()

        if trades:
            strategy_checks = self.check_strategy_health(trades)
            report.checks.extend(strategy_checks.values())

        report.checks.append(self.check_data_freshness(last_data_update))
        report.checks.append(self.check_position_count(open_position_count))

        if not report.all_passed:
            logger.warning("health_check_failures", summary=report.summary())
        else:
            logger.info("health_check_passed", summary=report.summary())

        return report
