"""Run historical backtest using the signal engine and backtest engine."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import structlog

from backtest.engine import BacktestConfig, BacktestEngine
from data.feature_store import FeatureStore
from data.features.momentum import compute_rsi
from data.providers import YFinanceProvider
from data.warehouse import DataWarehouse
from scanner.universe import Universe
from signals.breakout import VolatilityBreakout
from signals.catalyst import CatalystCapture
from signals.config_loader import build_config, load_strategy_configs
from signals.mean_reversion import MeanReversion
from signals.momentum_pairs import SectorMomentumPairs
from signals.regime import RegimeDetector, load_regime_weights

logger = structlog.get_logger(__name__)

RESULT_PATH = Path(__file__).parent.parent / "warehouse" / "backtest_result.json"


def run_historical_backtest(
    start: str = "2023-01-01",
    end: str | None = None,
    sectors: list[str] | None = None,
    tickers: list[str] | None = None,
    result_path: Path | str = RESULT_PATH,
) -> dict:
    """Run a full historical backtest and save results.

    Args:
        start: Backtest start date.
        end: Backtest end date (None = present).
        sectors: Filter to these sectors only (None = all).
        tickers: Filter to these tickers only (None = all). Overrides sectors.
        result_path: Where to save the JSON result.
    """
    logger.info(
        "historical_backtest_starting",
        start=start,
        end=end,
        sectors=sectors,
        tickers=tickers,
    )

    universe = Universe()
    warehouse = DataWarehouse(provider=YFinanceProvider())
    feature_store = FeatureStore(warehouse=warehouse)

    # Load strategy configs
    configs = load_strategy_configs()
    strategies_non_mr = [
        CatalystCapture(config=build_config("catalyst_capture", configs["catalyst_capture"])),
        VolatilityBreakout(
            config=build_config("volatility_breakout", configs["volatility_breakout"])
        ),
    ]
    mean_rev = MeanReversion(config=build_config("mean_reversion", configs["mean_reversion"]))
    momentum = SectorMomentumPairs(
        config=build_config("sector_momentum", configs["sector_momentum"])
    )

    # Determine which tickers to include
    if tickers:
        # Explicit ticker list takes priority
        target_tickers = sorted(set(t.upper() for t in tickers))
    elif sectors:
        # Filter by selected sectors
        target_tickers = []
        for s in sectors:
            target_tickers.extend(universe.get_tickers(sector=s))
        target_tickers = sorted(set(target_tickers))
    else:
        target_tickers = universe.get_unique_tickers()
    features: dict[str, pd.DataFrame] = {}
    prices: dict[str, pd.DataFrame] = {}

    for ticker in target_tickers:
        try:
            feat_df = feature_store.load(ticker)
            if start:
                feat_df = feat_df[feat_df.index >= pd.Timestamp(start)]
            if end:
                feat_df = feat_df[feat_df.index <= pd.Timestamp(end)]
            if feat_df.empty:
                continue
            features[ticker] = feat_df
            prices[ticker] = feat_df[["Open", "High", "Low", "Close", "Volume"]]
        except FileNotFoundError:
            continue

    if not features:
        logger.warning("no_features_for_backtest")
        return {"error": "No feature data available. Run bootstrap first."}

    # Load sector ETF data for mean reversion sector health filter
    sector_etf_rsi: dict[str, pd.Series] = {}
    for sector in universe.get_sectors():
        etf = universe.get_sector_etf(sector)
        try:
            etf_df = warehouse.load(etf)
            if start:
                etf_df = etf_df[etf_df.index >= pd.Timestamp(start)]
            if end:
                etf_df = etf_df[etf_df.index <= pd.Timestamp(end)]
            if not etf_df.empty:
                sector_etf_rsi[sector] = compute_rsi(etf_df["Close"])
        except FileNotFoundError:
            pass

    logger.info("data_loaded", tickers=len(features))

    # Generate signals across entire history
    all_signals = []

    # Non-MR strategies: scan full history
    for strategy in strategies_non_mr:
        for ticker, feat_df in features.items():
            signals = strategy.scan(feat_df, ticker)
            all_signals.extend(signals)

    # Mean reversion: pass sector ETF RSI for health filter
    for ticker, feat_df in features.items():
        sector = universe.get_ticker_sector(ticker)
        etf_rsi = sector_etf_rsi.get(sector) if sector else None
        signals = mean_rev.scan(feat_df, ticker, sector_etf_rsi=etf_rsi)
        all_signals.extend(signals)

    # Sector momentum signals (only sectors with loaded tickers)
    scan_sectors = sectors if sectors else universe.get_sectors()
    for sector in scan_sectors:
        if sector not in universe.get_sectors():
            continue
        sector_tickers = universe.get_tickers(sector=sector)
        sector_features = {t: features[t] for t in sector_tickers if t in features}
        if sector_features:
            signals = momentum.scan_sector(sector_features, sector)
            all_signals.extend(signals)

    logger.info("signals_generated_raw", count=len(all_signals))

    # ── Apply regime weights to signals ──────────────────────────────────────
    # Detect regime for each signal date using a representative volatility series
    # (use SPY-like proxy: average HV20 across all loaded tickers)
    regime_detector = RegimeDetector()
    regime_weights = load_regime_weights()

    # Build a combined volatility series from all features
    hv_frames = []
    for feat_df in features.values():
        if "hv_20" in feat_df.columns:
            hv_frames.append(feat_df["hv_20"])
    if hv_frames:
        avg_hv = pd.concat(hv_frames, axis=1).mean(axis=1).dropna()
        regime_series = regime_detector.predict(
            returns=avg_hv.pct_change().fillna(0), volatility=avg_hv
        )
    else:
        regime_series = pd.Series(dtype=str)

    # Weight each signal by regime
    raw_count = len(all_signals)
    weighted_signals = []
    for sig in all_signals:
        sig_date = pd.Timestamp(sig.timestamp).normalize()
        # Find the regime on the signal date
        if not regime_series.empty and sig_date in regime_series.index:
            regime = regime_series.loc[sig_date]
        else:
            regime = "normal"

        strategy_weights = regime_weights.get(regime, regime_weights.get("normal", {}))
        weight = strategy_weights.get(sig.strategy_id, 1.0)

        if weight == 0:
            continue  # Strategy disabled in this regime

        # Scale signal strength by regime weight
        sig.strength = min(sig.strength * weight, 1.0)
        weighted_signals.append(sig)

    all_signals = weighted_signals
    logger.info(
        "signals_after_regime_weighting",
        count=len(all_signals),
        dropped=raw_count - len(all_signals),
    )

    if not all_signals:
        logger.warning("no_signals_after_regime_weighting")
        return {"error": "No signals survived regime weighting in the date range."}

    # Run backtest
    engine = BacktestEngine(BacktestConfig(init_cash=1_000_000))
    result = engine.run(all_signals, prices)
    metrics = result.metrics()

    logger.info(
        "backtest_complete",
        sharpe=f"{metrics.get('sharpe', 0):.2f}",
        total_return=f"{metrics.get('total_return', 0):.2%}",
        trades=metrics.get("total_trades", 0),
    )

    # Build per-strategy breakdown
    strategy_breakdown: dict[str, list[dict]] = {}
    for trade in result.trades:
        sid = trade.strategy_id
        if sid not in strategy_breakdown:
            strategy_breakdown[sid] = []
        strategy_breakdown[sid].append(
            {
                "ticker": trade.ticker,
                "direction": trade.direction,
                "entry_date": str(trade.entry_date),
                "exit_date": str(trade.exit_date),
                "entry_price": round(trade.entry_price, 2),
                "exit_price": round(trade.exit_price, 2),
                "return_pct": round(trade.return_pct, 4),
                "pnl": round(trade.pnl, 2),
                "hold_days": trade.hold_days,
                "exit_reason": trade.exit_reason,
                "sector": trade.sector,
            }
        )

    # Per-strategy metrics
    strategy_metrics = {}
    for sid, trades in strategy_breakdown.items():
        returns = [t["return_pct"] for t in trades]
        wins = [r for r in returns if r > 0]
        strategy_metrics[sid] = {
            "trades": len(trades),
            "avg_return": round(sum(returns) / len(returns), 4) if returns else 0,
            "win_rate": round(len(wins) / len(returns), 2) if returns else 0,
            "total_pnl": round(sum(t["pnl"] for t in trades), 2),
        }

    # Full equity curve (no sampling — keep all data points)
    equity_list = [
        {"date": str(dt), "equity": round(float(val), 2)} for dt, val in result.equity_curve.items()
    ]

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "period": {"start": start, "end": end or "present"},
        "filters": {
            "sectors": sectors,
            "tickers": tickers if tickers else None,
            "tickers_tested": len(features),
        },
        "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        "strategy_metrics": strategy_metrics,
        "trades": [t for trades in strategy_breakdown.values() for t in trades],
        "equity_curve": equity_list,
    }

    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("backtest_result_saved", path=str(path))
    return output
