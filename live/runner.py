"""Daily trading runner — orchestrates scan, signal, and order workflow."""

from __future__ import annotations

import argparse

import structlog

from data.feature_store import FeatureStore
from data.warehouse import DataWarehouse
from live.positions import PositionTracker
from live.scanner_report import save_scan_report
from scanner.universe import Universe
from signals.aggregator import SignalBus
from signals.base import SignalEvent
from signals.breakout import VolatilityBreakout
from signals.catalyst import CatalystCapture
from signals.config_loader import build_config, load_strategy_configs
from signals.mean_reversion import MeanReversion
from signals.momentum_pairs import SectorMomentumPairs
from signals.regime import RegimeDetector
from signals.risk_filter import RiskFilter
from utils.logging import setup_logging

logger = structlog.get_logger(__name__)


class DailyRunner:
    """Orchestrates the daily trading workflow."""

    def __init__(self) -> None:
        self.universe = Universe()
        # Use yfinance for bulk downloads — Polygon free tier rate-limits on bulk
        from data.providers import YFinanceProvider

        self.warehouse = DataWarehouse(provider=YFinanceProvider())
        self.feature_store = FeatureStore(warehouse=self.warehouse)
        self.tracker = PositionTracker()
        self.risk_filter = RiskFilter()
        self.regime_detector = RegimeDetector(use_hmm=False)

        # Load strategy configs from JSON (editable via dashboard)
        configs = load_strategy_configs()

        self.strategies = [
            CatalystCapture(config=build_config("catalyst_capture", configs["catalyst_capture"])),
            VolatilityBreakout(
                config=build_config("volatility_breakout", configs["volatility_breakout"])
            ),
            MeanReversion(config=build_config("mean_reversion", configs["mean_reversion"])),
        ]
        self.momentum = SectorMomentumPairs(
            config=build_config("sector_momentum", configs["sector_momentum"])
        )

    def bootstrap(self) -> None:
        """Download all data and build features from scratch."""
        logger.info("bootstrap_starting")
        logger.info("downloading_ohlcv_data")
        results = self.warehouse.download_all(start="2018-01-01")
        successful = sum(1 for v in results.values() if v > 0)
        logger.info("download_complete", successful=successful, total=len(results))

        logger.info("building_features")
        self.feature_store.build_all()
        self.feature_store.save_metadata()
        logger.info("bootstrap_complete")

    def update_data(self, force_refresh: bool = False) -> None:
        """Update warehouse and rebuild features.

        Args:
            force_refresh: If True, re-download all data from scratch to fix
                dividend adjustment inconsistencies.
        """
        logger.info("updating_data", force_refresh=force_refresh)
        updated = self.warehouse.update(force_refresh=force_refresh)
        new_rows = sum(v for v in updated.values() if v > 0)
        logger.info("warehouse_updated", tickers_with_new_data=new_rows)

        for ticker, rows in updated.items():
            if rows > 0:
                try:
                    self.feature_store.build(ticker)
                except Exception:
                    logger.exception("feature_rebuild_failed", ticker=ticker)

    def _load_features(self) -> dict[str, object]:
        """Load features for all tickers."""
        tickers = self.universe.get_unique_tickers()
        features: dict[str, object] = {}
        for ticker in tickers:
            try:
                features[ticker] = self.feature_store.load(ticker)
            except FileNotFoundError:
                continue
        return features

    def run_scan_detailed(self) -> tuple[list[SignalEvent], list[dict]]:
        """Run scan and return both filtered signals AND per-ticker evaluations.

        Returns:
            (filtered_signals, evaluations) where evaluations is a list of dicts
            with per-ticker, per-strategy condition breakdowns.
        """
        logger.info("starting_detailed_scan")
        features = self._load_features()

        if not features:
            logger.warning("no_features_available")
            print("\n*** No feature data found. ***\n" "Run: python -m live --mode bootstrap\n")
            return [], []

        # Detect regime
        sample_df = next(iter(features.values()))
        if "hv_20" in sample_df.columns:
            regime = self.regime_detector.get_current_regime(sample_df["hv_20"])
        else:
            regime = "normal"
        regime_weights = RegimeDetector.get_strategy_weights(regime)
        logger.info("regime_detected", regime=regime)

        all_signals: list[SignalEvent] = []
        all_evaluations: list[dict] = []

        # Evaluate single-ticker strategies
        for strategy in self.strategies:
            weight = regime_weights.get(strategy.strategy_id, 1.0)

            for ticker, feat_df in features.items():
                # Get detailed evaluation (rationale)
                evaluation = strategy.evaluate(feat_df, ticker)
                evaluation["regime_weight"] = weight
                evaluation["sector"] = self.universe.get_ticker_sector(ticker) or ""

                # Add key feature values for universe view
                if not feat_df.empty:
                    row = feat_df.iloc[-1]
                    close_val = float(row.get("Close", 0))
                    vwap_dev_raw = row.get("vwap_dev")
                    vwap_dev_val = float(vwap_dev_raw) if not _isnan(vwap_dev_raw) else None
                    if vwap_dev_val is not None and vwap_dev_val != -1:
                        vwap_price = round(close_val / (1 + vwap_dev_val), 2)
                    else:
                        vwap_price = None
                    evaluation["features"] = {
                        "close": round(close_val, 2),
                        "vwap": vwap_price,
                        "rsi_14": (
                            round(float(row.get("rsi_14", 0)), 1)
                            if not _isnan(row.get("rsi_14"))
                            else None
                        ),
                        "vwap_dev": (
                            round(float(row.get("vwap_dev", 0)), 4)
                            if not _isnan(row.get("vwap_dev"))
                            else None
                        ),
                        "atr_5": (
                            round(float(row.get("atr_5", 0)), 4)
                            if not _isnan(row.get("atr_5"))
                            else None
                        ),
                        "bb_width": (
                            round(float(row.get("bb_width", 0)), 4)
                            if not _isnan(row.get("bb_width"))
                            else None
                        ),
                        "event_score": (
                            round(float(row.get("event_score", 0)), 3)
                            if not _isnan(row.get("event_score"))
                            else None
                        ),
                        "hv_20": (
                            round(float(row.get("hv_20", 0)), 4)
                            if not _isnan(row.get("hv_20"))
                            else None
                        ),
                    }

                all_evaluations.append(evaluation)

                # Collect latest signal only (not full history)
                if weight > 0 and evaluation.get("triggered"):
                    sig = strategy.scan_latest(feat_df, ticker)
                    if sig:
                        sig.strength *= weight
                        sig.strength = min(sig.strength, 1.0)
                        all_signals.append(sig)

        # Evaluate sector momentum
        mom_weight = regime_weights.get("sector_momentum", 1.0)
        for sector in self.universe.get_sectors():
            sector_tickers = self.universe.get_tickers(sector=sector)
            sector_features = {t: features[t] for t in sector_tickers if t in features}
            if sector_features:
                # Detailed evaluation
                sector_evals = self.momentum.evaluate_sector(sector_features, sector)
                for ev in sector_evals:
                    ev["regime_weight"] = mom_weight
                all_evaluations.extend(sector_evals)

                # Collect only latest rebalance signals (not full history)
                if mom_weight > 0:
                    signals = self.momentum.scan_sector(sector_features, sector)
                    if signals:
                        # Take only signals from the most recent rebalance date
                        latest_ts = max(s.timestamp for s in signals)
                        latest_signals = [s for s in signals if s.timestamp == latest_ts]
                        for sig in latest_signals:
                            sig.strength *= mom_weight
                            sig.strength = min(sig.strength, 1.0)
                        all_signals.extend(latest_signals)

        logger.info("raw_signals_generated", count=len(all_signals))

        # Aggregate and resolve
        bus = SignalBus()
        bus.set_open_positions(set(self.tracker.get_open().keys()))
        bus.emit_batch(all_signals)
        resolved = bus.resolve()

        # Risk filter
        portfolio = self.tracker.get_portfolio_state()
        filtered: list[SignalEvent] = []
        for signal in resolved:
            result = self.risk_filter.check(signal, portfolio)
            if result.passed:
                filtered.append(signal)
            else:
                logger.info("signal_rejected", ticker=signal.ticker, reason=result.reason)

        # Save scan report for dashboard
        signal_dicts = [
            {
                "ticker": s.ticker,
                "direction": s.direction.value,
                "strength": round(s.strength, 3),
                "strategy_id": s.strategy_id,
            }
            for s in filtered
        ]
        save_scan_report(all_evaluations, signal_dicts)

        logger.info(
            "scan_complete",
            raw=len(all_signals),
            resolved=len(resolved),
            filtered=len(filtered),
            evaluations=len(all_evaluations),
        )
        return filtered, all_evaluations

    def run_scan(self) -> list[SignalEvent]:
        """Run scan and return filtered signals (backward-compatible)."""
        signals, _ = self.run_scan_detailed()
        return signals

    def run_paper(self) -> None:
        """Run scan and submit orders to Alpaca paper account."""
        from live.broker import AlpacaBroker

        signals = self.run_scan()
        if not signals:
            logger.info("no_signals_to_trade")
            return

        broker = AlpacaBroker(paper=True)
        account = broker.get_account()
        equity = account.get("equity", 0)

        submitted = 0
        for signal in signals:
            position_value = equity * 0.05
            order_id = broker.submit_order(signal, position_value)
            if order_id:
                price = broker.get_latest_price(signal.ticker)
                if not price:
                    trade_params = signal.metadata.get("trade_params")
                    price = trade_params.entry_price if trade_params else 0
                shares = position_value / price if price > 0 else 0
                self.tracker.open_position(signal, price, shares)
                submitted += 1

        self.tracker.save()
        logger.info("paper_trading_complete", submitted=submitted, total_signals=len(signals))

    def run_live(self) -> None:
        """Run scan and submit to Alpaca live account (with safety checks)."""
        from live.broker import AlpacaBroker
        from live.capital import CapitalManager

        signals = self.run_scan()
        if not signals:
            logger.info("no_signals_to_trade")
            return

        broker = AlpacaBroker(paper=False)
        account = broker.get_account()
        equity = account.get("equity", 0)

        cap_manager = CapitalManager(intended_capital=equity)
        sizing_mult = cap_manager.get_position_sizing_multiplier()

        for signal in signals:
            position_value = equity * 0.05 * sizing_mult
            order_id = broker.submit_bracket_order(signal, position_value)
            if order_id:
                trade_params = signal.metadata.get("trade_params")
                entry = trade_params.entry_price if trade_params else 0
                shares = position_value / entry if entry > 0 else 0
                self.tracker.open_position(signal, entry, shares)

        self.tracker.save()
        logger.info("live_trading_complete", orders=len(signals))

    def reconcile(self) -> dict[str, object]:
        """Compare internal tracker vs broker positions."""
        from live.broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)
        broker_positions = {p["ticker"]: p for p in broker.get_positions()}
        internal_positions = self.tracker.get_open()

        discrepancies = []
        for ticker in set(list(broker_positions.keys()) + list(internal_positions.keys())):
            in_broker = ticker in broker_positions
            in_internal = ticker in internal_positions
            if in_broker != in_internal:
                discrepancies.append(
                    {"ticker": ticker, "broker": in_broker, "internal": in_internal}
                )

        return {
            "discrepancies": discrepancies,
            "broker_count": len(broker_positions),
            "internal_count": len(internal_positions),
        }


def _isnan(val: object) -> bool:
    """Check if a value is NaN (handles None and non-numeric)."""
    import math

    if val is None:
        return True
    try:
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return True


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Daily trading runner")
    parser.add_argument(
        "--mode",
        choices=["bootstrap", "update", "scan-only", "backtest", "paper", "live"],
        default="scan-only",
        help="bootstrap | update | scan-only | backtest | paper | live",
    )
    parser.add_argument(
        "--start",
        default="2023-01-01",
        help="Backtest start date (default: 2023-01-01)",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")

    if args.mode == "backtest":
        from live.backtester import run_historical_backtest

        result = run_historical_backtest(start=args.start)
        if "error" not in result:
            m = result["metrics"]
            print(f"\n{'='*50}")
            print(f"  Backtest Results ({args.start} to present)")
            print(f"{'='*50}")
            print(f"  Sharpe Ratio:    {m.get('sharpe', 0):.2f}")
            print(f"  Total Return:    {m.get('total_return', 0):.2%}")
            print(f"  Max Drawdown:    {m.get('max_drawdown', 0):.2%}")
            print(f"  Win Rate:        {m.get('win_rate', 0):.0%}")
            print(f"  Total Trades:    {m.get('total_trades', 0)}")
            print(f"  Profit Factor:   {m.get('profit_factor', 0):.2f}")
            print(f"{'='*50}")
            print("  Results saved to warehouse/backtest_result.json")
            print("  View in dashboard: streamlit run dashboard/app.py\n")
        else:
            print(f"\nError: {result['error']}\n")
        return

    runner = DailyRunner()

    if args.mode == "bootstrap":
        runner.bootstrap()
    elif args.mode == "update":
        runner.update_data()
    elif args.mode == "scan-only":
        signals = runner.run_scan()
        for sig in signals:
            logger.info(
                "signal",
                ticker=sig.ticker,
                direction=sig.direction.value,
                strength=f"{sig.strength:.2f}",
                strategy=sig.strategy_id,
            )
        if not signals:
            logger.info("no_signals_generated")
    elif args.mode == "paper":
        runner.run_paper()
    elif args.mode == "live":
        runner.run_live()


if __name__ == "__main__":
    main()
