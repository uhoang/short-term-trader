"""Tests for capital management and strategy rotation."""

from __future__ import annotations

from live.capital import CapitalManager


class TestCapitalManager:
    def test_initial_allocation(self) -> None:
        cm = CapitalManager(intended_capital=1_000_000)
        assert cm.get_allocation() == 100_000  # 10%
        assert cm.get_allocation_pct() == 0.10

    def test_ramp_up_criteria(self) -> None:
        cm = CapitalManager(intended_capital=1_000_000)
        # Not enough days or Sharpe
        assert not cm.should_ramp_up(live_sharpe=0.5, days_live=3)
        # Meets criteria for level 2 (25%)
        assert cm.should_ramp_up(live_sharpe=1.2, days_live=7)

    def test_ramp_up_advances_level(self) -> None:
        cm = CapitalManager(intended_capital=1_000_000)
        cm.ramp_up()
        assert cm.current_level == 1
        assert cm.get_allocation_pct() == 0.25

    def test_full_ramp_schedule(self) -> None:
        cm = CapitalManager(intended_capital=1_000_000)
        # Level 0: 10%
        assert cm.get_allocation_pct() == 0.10
        cm.ramp_up()  # → 25%
        assert cm.get_allocation_pct() == 0.25
        cm.ramp_up()  # → 50%
        assert cm.get_allocation_pct() == 0.50
        cm.ramp_up()  # → 100%
        assert cm.get_allocation_pct() == 1.00

    def test_cannot_ramp_past_max(self) -> None:
        cm = CapitalManager(intended_capital=1_000_000)
        for _ in range(10):
            cm.ramp_up()
        assert cm.get_allocation_pct() == 1.00

    def test_position_sizing_multiplier(self) -> None:
        cm = CapitalManager(intended_capital=1_000_000)
        # At 10%: conservative sizing
        assert cm.get_position_sizing_multiplier() == 0.5
        cm.ramp_up()  # 25%: still conservative
        assert cm.get_position_sizing_multiplier() == 0.5
        cm.ramp_up()  # 50%: full sizing
        assert cm.get_position_sizing_multiplier() == 1.0

    def test_strategy_disable_threshold(self) -> None:
        cm = CapitalManager()
        assert cm.should_disable_strategy("catalyst_capture", rolling_sharpe_60d=0.2)
        assert not cm.should_disable_strategy("catalyst_capture", rolling_sharpe_60d=0.8)

    def test_update_day(self) -> None:
        cm = CapitalManager()
        assert cm.days_at_level == 0
        cm.update_day()
        cm.update_day()
        assert cm.days_at_level == 2


class TestHealthMonitor:
    def test_strategy_health_check(self) -> None:
        from live.health import HealthMonitor

        monitor = HealthMonitor()
        trades = [
            {"strategy_id": "catalyst", "return_pct": 0.05, "exit_date": "2026-03-20T10:00:00"},
            {"strategy_id": "catalyst", "return_pct": 0.03, "exit_date": "2026-03-19T10:00:00"},
            {"strategy_id": "catalyst", "return_pct": -0.02, "exit_date": "2026-03-18T10:00:00"},
        ]
        checks = monitor.check_strategy_health(trades)
        assert "catalyst" in checks

    def test_data_freshness_pass(self) -> None:
        from datetime import datetime

        from live.health import HealthMonitor

        monitor = HealthMonitor()
        check = monitor.check_data_freshness(datetime.now())
        assert check.passed

    def test_data_freshness_fail(self) -> None:
        from live.health import HealthMonitor

        monitor = HealthMonitor()
        check = monitor.check_data_freshness("2020-01-01T00:00:00")
        assert not check.passed

    def test_run_all_checks(self) -> None:
        from live.health import HealthMonitor

        monitor = HealthMonitor()
        report = monitor.run_all_checks(open_position_count=5)
        assert len(report.checks) >= 2

    def test_position_count_check(self) -> None:
        from live.health import HealthMonitor

        monitor = HealthMonitor()
        check = monitor.check_position_count(10, max_positions=15)
        assert check.passed
        check = monitor.check_position_count(16, max_positions=15)
        assert not check.passed
