"""Streamlit dashboard for signal analysis, backtesting, and strategy tuning."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Light theme styling ──────────────────────────────────────────────────────


def _apply_theme() -> None:
    """Inject CSS for light theme with larger tab names and light button text."""
    st.markdown(
        """
        <style>
        /* Larger tab labels */
        .stTabs [data-baseweb="tab"] {
            font-size: 1.05rem !important;
        }
        .stTabs [data-baseweb="tab"] p,
        .stTabs [data-baseweb="tab"] span,
        .stTabs [data-baseweb="tab-panel"],
        .stTabs button[role="tab"] {
            font-size: 1.05rem !important;
        }
        /* Light (white) text on buttons */
        .stButton > button {
            color: #ffffff !important;
        }
        .stDownloadButton > button {
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


WAREHOUSE_DIR = Path(__file__).parent.parent / "warehouse"
POSITIONS_FILE = WAREHOUSE_DIR / "positions.json"
SCAN_REPORT_FILE = WAREHOUSE_DIR / "scan_report.json"
BACKTEST_RESULT_FILE = WAREHOUSE_DIR / "backtest_result.json"
CONFIG_DIR = Path(__file__).parent.parent / "config"
STRATEGY_PARAMS_FILE = CONFIG_DIR / "strategy_params.json"

DEFAULT_STRATEGY_PARAMS = {
    "catalyst_capture": {
        "event_score_min": 0.3,
        "atr_ratio_min": 1.5,
        "iv_rank_multiplier": 1.3,
        "stop_loss_pct": 0.07,
        "take_profit_pct": 0.15,
        "max_hold_days": 10,
    },
    "volatility_breakout": {
        "atr_ratio_min": 1.3,
        "volume_spike_min": 1.5,
        "bb_width_lookback": 126,
        "stop_loss_pct": 0.06,
        "max_hold_days": 10,
    },
    "mean_reversion": {
        "vwap_dev_threshold": -0.02,
        "rsi_threshold": 32.0,
        "sector_etf_rsi_min": 45.0,
        "stop_loss_pct": 0.08,
        "max_hold_days": 22,
    },
    "sector_momentum": {
        "rebalance_days": 15,
        "top_n": 3,
        "bottom_n": 3,
        "stop_loss_pct": 0.08,
        "max_hold_days": 45,
    },
}


# ── Data loaders ─────────────────────────────────────────────────────────────


def load_positions() -> dict:
    if not POSITIONS_FILE.exists():
        return {"open_positions": {}, "closed_trades": [], "cash": 0}
    with open(POSITIONS_FILE) as f:
        return json.load(f)


def load_scan_report() -> dict:
    if not SCAN_REPORT_FILE.exists():
        return {"timestamp": None, "evaluations": [], "signals": []}
    with open(SCAN_REPORT_FILE) as f:
        return json.load(f)


def load_strategy_params() -> dict:
    from copy import deepcopy

    defaults = deepcopy(DEFAULT_STRATEGY_PARAMS)
    if not STRATEGY_PARAMS_FILE.exists():
        return defaults
    with open(STRATEGY_PARAMS_FILE) as f:
        saved = json.load(f)
    # Merge saved over defaults so missing keys get default values
    for strategy_id, default_vals in defaults.items():
        if strategy_id in saved:
            default_vals.update(saved[strategy_id])
        defaults[strategy_id] = default_vals
    return defaults


def save_strategy_params(params: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(STRATEGY_PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)


def load_backtest_result() -> dict | None:
    if not BACKTEST_RESULT_FILE.exists():
        return None
    with open(BACKTEST_RESULT_FILE) as f:
        return json.load(f)


# ── Shared actions ───────────────────────────────────────────────────────────


def _run_scan(force_refresh: bool = False) -> None:
    from live.runner import DailyRunner

    runner = DailyRunner()
    if force_refresh:
        with st.spinner("Re-downloading all data (fixing price adjustments)..."):
            runner.warehouse.download_all()
            runner.feature_store.build_all()
    else:
        with st.spinner("Downloading data for new/updated tickers..."):
            runner.update_data()
    with st.spinner("Scanning all tickers across 4 strategies..."):
        runner.run_scan()


def _run_backtest(
    start: str,
    end: str | None,
    sectors: list[str] | None = None,
    tickers: list[str] | None = None,
) -> None:
    from live.backtester import run_historical_backtest

    label = f"Running backtest from {start}"
    if sectors:
        label += f" ({', '.join(sectors)})"
    if tickers:
        label += f" ({len(tickers)} tickers)"
    with st.spinner(label + "..."):
        run_historical_backtest(
            start=start, end=end if end else None, sectors=sectors, tickers=tickers
        )


# ── Tab: Stock Universe ──────────────────────────────────────────────────────


def page_universe() -> None:
    col_hdr, col_btn, col_refresh = st.columns([4, 1, 1])
    with col_hdr:
        st.markdown("### Stock Universe")
        st.caption("Scan tickers for signals. View features, direction, and rationale.")
    with col_btn:
        if st.button("Run Scan", type="primary", key="universe_scan", use_container_width=True):
            _run_scan()
            st.rerun()
    with col_refresh:
        if st.button(
            "Force Refresh",
            type="primary",
            key="universe_refresh",
            use_container_width=True,
            help="Re-download all price data from scratch to fix dividend adjustment issues.",
        ):
            _run_scan(force_refresh=True)
            st.rerun()

    report = load_scan_report()
    evaluations = report.get("evaluations", [])
    signals = {s["ticker"] for s in report.get("signals", [])}

    if not evaluations:
        st.info("No scan data yet. Click **Run Scan** to generate signals.")
        return

    st.caption(f"Last scan: {report.get('timestamp', 'N/A')}")

    # Build per-ticker summary
    ticker_data: dict[str, dict] = {}
    for ev in evaluations:
        ticker = ev.get("ticker", "")
        if not ticker:
            continue
        if ticker not in ticker_data:
            ticker_data[ticker] = {
                "Ticker": ticker,
                "Sector": ev.get("sector", "").title(),
                "Close": None,
                "VWAP": None,
                "RSI": None,
                "VWAP Dev": None,
                "BB Width": None,
                "HV(20)": None,
                "Event Score": None,
                "Signals": [],
            }
        feats = ev.get("features", {})
        if feats and ticker_data[ticker]["Close"] is None:
            ticker_data[ticker]["Close"] = feats.get("close")
            ticker_data[ticker]["VWAP"] = feats.get("vwap")
            ticker_data[ticker]["RSI"] = feats.get("rsi_14")
            ticker_data[ticker]["VWAP Dev"] = feats.get("vwap_dev")
            ticker_data[ticker]["BB Width"] = feats.get("bb_width")
            ticker_data[ticker]["HV(20)"] = feats.get("hv_20")
            ticker_data[ticker]["Event Score"] = feats.get("event_score")
        if ev.get("triggered"):
            ticker_data[ticker]["Signals"].append(
                ev.get("strategy_id", "").replace("_", " ").title()
            )

    rows = []
    for td in sorted(ticker_data.values(), key=lambda x: x["Ticker"]):
        sig_list = td["Signals"]
        ticker = td["Ticker"]

        # Status
        if ticker in signals:
            status = "SIGNAL"
        elif sig_list:
            directions = set()
            for ev in evaluations:
                if ev.get("ticker") == ticker and ev.get("triggered"):
                    d = ev.get("direction", "none")
                    if d and d not in ("none", ""):
                        directions.add(d.title())
            if "Long" in directions and "Short" in directions:
                status = "Warning: conflicting long/short"
            else:
                status = "Warning: sector >40% cap or max 15 positions"
        else:
            status = ""

        # Direction
        rec_dirs = set()
        for ev in evaluations:
            if ev.get("ticker") == ticker and ev.get("triggered"):
                d = ev.get("direction", "none")
                if d and d not in ("none", ""):
                    rec_dirs.add(d.title())
        if "Long" in rec_dirs and "Short" in rec_dirs:
            recommendation = "Long/Short"
        elif "Long" in rec_dirs:
            recommendation = "Long"
        elif "Short" in rec_dirs:
            recommendation = "Short"
        else:
            recommendation = ""

        rows.append(
            {
                "Ticker": ticker,
                "Sector": td["Sector"],
                "Direction": recommendation,
                "Close": td["Close"],
                "VWAP": td["VWAP"],
                "RSI(14)": td["RSI"],
                "VWAP Dev": td["VWAP Dev"],
                "BB Width": td["BB Width"],
                "HV(20)": td["HV(20)"],
                "Event Score": td["Event Score"],
                "Status": status,
                "Strategies": ", ".join(sig_list) if sig_list else "",
            }
        )

    df = pd.DataFrame(rows)

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sectors = ["All"] + sorted(df["Sector"].unique().tolist())
        sel_sector = st.selectbox("Sector", sectors, key="u_sector")
    with col_f2:
        sel_dir = st.selectbox("Direction", ["All", "Long", "Short"], key="u_dir")
    with col_f3:
        show_signals = st.checkbox("Signals only", key="u_signals_only")

    if sel_sector != "All":
        df = df[df["Sector"] == sel_sector]
    if sel_dir != "All":
        df = df[df["Direction"].str.contains(sel_dir, na=False)]
    if show_signals:
        df = df[df["Direction"] != ""]

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Close": st.column_config.NumberColumn(format="$%.2f"),
            "VWAP": st.column_config.NumberColumn(format="$%.2f"),
            "RSI(14)": st.column_config.NumberColumn(format="%.1f"),
            "VWAP Dev": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    # Ticker detail
    st.markdown("---")
    st.markdown("#### Ticker Detail")
    ticker_list = sorted(ticker_data.keys())
    sel_ticker = st.selectbox("Select ticker", ticker_list, key="u_detail")

    if sel_ticker:
        ticker_evals = [e for e in evaluations if e.get("ticker") == sel_ticker]
        for ev in ticker_evals:
            strategy = ev.get("strategy_id", "unknown").replace("_", " ").title()
            triggered = ev.get("triggered", False)
            weight = ev.get("regime_weight", 1.0)
            direction = ev.get("direction", "none")

            if triggered:
                dir_label = direction.title() if direction != "none" else ""
                label = f"+ {strategy} | {dir_label}" if dir_label else f"+ {strategy}"
            else:
                label = f"- {strategy} | No Signal"
            if weight == 0:
                label += " | DISABLED"

            with st.expander(label, expanded=triggered):
                if triggered and direction not in ("none", ""):
                    st.markdown(f"**Recommendation: {direction.title()}**")

                conditions = ev.get("conditions", [])
                if conditions:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Condition": c.get("name", ""),
                                    "Value": c.get("value", ""),
                                    "Threshold": c.get("threshold", ""),
                                    "Op": c.get("operator", ""),
                                    "Pass": "Yes" if c.get("passed") else "No",
                                }
                                for c in conditions
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                if weight == 0:
                    st.error(
                        "**Regime: 0.0x (DISABLED)** -- Strategy disabled in current market regime."
                    )
                elif weight < 1.0:
                    st.warning(f"**Regime: {weight:.1f}x (REDUCED)** -- Strength at {weight:.0%}.")
                elif weight > 1.0:
                    st.success(f"**Regime: {weight:.1f}x (BOOSTED)** -- Strength at {weight:.0%}.")
                else:
                    st.caption("Regime: 1.0x (normal)")


# ── Tab: Signals ─────────────────────────────────────────────────────────────


def page_signals() -> None:
    col_hdr, col_btn = st.columns([4, 1])
    with col_hdr:
        st.markdown("### Actionable Signals")
        st.caption(
            "Signals that passed strategy conditions, conflict resolution, and risk filters."
        )
    with col_btn:
        if st.button("Run Scan", type="primary", key="signals_scan", use_container_width=True):
            _run_scan()
            st.rerun()

    report = load_scan_report()
    signals_list = report.get("signals", [])
    evaluations = report.get("evaluations", [])

    if not signals_list:
        total = report.get("total_tickers_scanned", 0)
        triggered = len([e for e in evaluations if e.get("triggered")])
        if total:
            st.info(
                f"No actionable signals from last scan. "
                f"Scanned {total} tickers, {triggered} met conditions but none passed filters."
            )
        else:
            st.info("No scan data. Click **Run Scan** above.")
        return

    st.caption(f"Scan: {report.get('timestamp', 'N/A')}")

    # Pipeline metrics
    total = report.get("total_tickers_scanned", 0)
    triggered = len([e for e in evaluations if e.get("triggered")])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scanned", total)
    c2.metric("Triggered", triggered)
    c3.metric("Signals", len(signals_list))
    c4.metric("Pass Rate", f"{len(signals_list)/triggered:.0%}" if triggered else "N/A")

    # Signals table
    sig_rows = []
    for sig in signals_list:
        ticker = sig.get("ticker", "")
        ev = next(
            (
                e
                for e in evaluations
                if e.get("ticker") == ticker
                and e.get("strategy_id") == sig.get("strategy_id")
                and e.get("triggered")
            ),
            None,
        )
        feats = ev.get("features", {}) if ev else {}
        sig_rows.append(
            {
                "Ticker": ticker,
                "Direction": sig.get("direction", "").title(),
                "Strategy": sig.get("strategy_id", "").replace("_", " ").title(),
                "Strength": sig.get("strength", 0),
                "Sector": (ev.get("sector", "") if ev else "").title(),
                "Close": feats.get("close"),
                "RSI": feats.get("rsi_14"),
                "VWAP Dev": feats.get("vwap_dev"),
            }
        )

    sig_df = pd.DataFrame(sig_rows)
    st.dataframe(
        sig_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Strength": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
            "Close": st.column_config.NumberColumn(format="$%.2f"),
            "RSI": st.column_config.NumberColumn(format="%.1f"),
            "VWAP Dev": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    # Strategy summary
    st.markdown("---")
    st.markdown("#### By Strategy")
    strategy_desc = {
        "Catalyst Capture": "Event-driven entries before earnings/catalysts with elevated vol",
        "Volatility Breakout": "BB squeeze breakouts confirmed by volume spike",
        "Mean Reversion": "Oversold bounces (low RSI + below VWAP) in healthy sectors",
        "Sector Momentum": "Long top / short bottom momentum-ranked stocks per sector",
    }
    for strat in sig_df["Strategy"].unique():
        s_df = sig_df[sig_df["Strategy"] == strat]
        longs = len(s_df[s_df["Direction"] == "Long"])
        shorts = len(s_df[s_df["Direction"] == "Short"])
        tickers_str = ", ".join(s_df["Ticker"].tolist())
        parts = []
        if longs:
            parts.append(f"{longs} Long")
        if shorts:
            parts.append(f"{shorts} Short")
        st.markdown(f"**{strat}** -- {strategy_desc.get(strat, '')}")
        st.write(f"{' | '.join(parts)} -- {tickers_str}")

    # Sector distribution
    if len(sig_df) > 1 and "Sector" in sig_df.columns:
        st.markdown("---")
        st.markdown("#### By Sector")
        st.bar_chart(sig_df["Sector"].value_counts())

    # Signal rationale
    st.markdown("---")
    st.markdown("#### Why These Signals Fired")
    for sig in signals_list:
        ticker = sig.get("ticker", "")
        raw_strategy = sig.get("strategy_id", "")
        strategy = raw_strategy.replace("_", " ").title()
        direction = sig.get("direction", "").title()
        strength = sig.get("strength", 0)

        ev = next(
            (
                e
                for e in evaluations
                if e.get("ticker") == ticker
                and e.get("strategy_id") == raw_strategy
                and e.get("triggered")
            ),
            None,
        )

        with st.expander(f"{direction} {ticker} via {strategy} (strength {strength:.0%})"):
            if not ev:
                st.write("Rationale not available.")
                continue

            conditions = ev.get("conditions", [])
            feats = ev.get("features", {})

            for c in conditions:
                name = c.get("name", "")
                value = c.get("value", "")
                threshold = c.get("threshold", "")
                op = c.get("operator", "")
                if op == ">=":
                    st.markdown(f"- {name} is **{value}** (above {threshold})")
                elif op == "<=":
                    st.markdown(f"- {name} is **{value}** (below {threshold})")
                elif op == "==" and value is True:
                    st.markdown(f"- {name} is **active**")
                elif op == "outside":
                    st.markdown(f"- {name} is **{value}** (outside normal range)")
                elif op == "rank":
                    st.markdown(f"- {name}: **{value}** ({threshold})")
                else:
                    st.markdown(f"- {name}: **{value}** {op} {threshold}")

            if feats:
                parts = []
                if feats.get("close"):
                    parts.append(f"Close: ${feats['close']:.2f}")
                if feats.get("rsi_14") is not None:
                    parts.append(f"RSI: {feats['rsi_14']:.1f}")
                if feats.get("vwap_dev") is not None:
                    parts.append(f"VWAP Dev: {feats['vwap_dev']:.4f}")
                if parts:
                    st.caption(" | ".join(parts))

            weight = ev.get("regime_weight", 1.0)
            if weight != 1.0:
                st.caption(f"Regime: {weight:.1f}x ({'boosted' if weight > 1 else 'reduced'})")


# ── Tab: Backtest ────────────────────────────────────────────────────────────


def page_backtest() -> None:
    st.markdown("### Historical Backtest")
    st.caption(
        "Test strategies against historical data with configurable date range and ticker filters."
    )

    from datetime import date

    from scanner.universe import Universe

    universe = Universe()

    # Controls — left half of the page
    ctrl_col, _ = st.columns([1, 1])
    with ctrl_col:
        col_start, col_end = st.columns(2)
        with col_start:
            bt_start = st.date_input(
                "Start", value=date(2023, 1, 1), min_value=date(2018, 1, 1), key="bt_start"
            )
        with col_end:
            bt_end = st.date_input("End", value=date.today(), key="bt_end")

        all_sectors = universe.get_sectors()
        sel_sectors = st.multiselect(
            "Sectors (empty = all)", all_sectors, default=[], key="bt_sectors"
        )

        avail = []
        if sel_sectors:
            for s in sel_sectors:
                avail.extend(universe.get_tickers(sector=s))
        else:
            avail = universe.get_unique_tickers()
        sel_tickers = st.multiselect(
            "Tickers (empty = all)", sorted(avail), default=[], key="bt_tickers"
        )

    scope = "all tickers"
    if sel_tickers:
        scope = f"{len(sel_tickers)} tickers"
    elif sel_sectors:
        scope = ", ".join(sel_sectors)

    btn_col, _ = st.columns([1, 1])
    with btn_col:
        run_bt = st.button(
            f"Run Backtest ({scope})", type="primary", key="run_bt", use_container_width=True
        )

    if run_bt:
        end_str = str(bt_end) if bt_end < date.today() else None
        _run_backtest(
            str(bt_start), end_str, sectors=sel_sectors or None, tickers=sel_tickers or None
        )
        st.rerun()

    result = load_backtest_result()
    if result is None:
        st.info("Configure parameters above and click **Run Backtest** to start.")
        return
    if "error" in result:
        st.error(result["error"])
        return

    # Header
    st.markdown("---")
    period = result.get("period", {})
    filters = result.get("filters", {})
    filter_desc = ""
    if filters.get("tickers"):
        filter_desc = f" | Tickers: {', '.join(filters['tickers'])}"
    elif filters.get("sectors"):
        filter_desc = f" | Sectors: {', '.join(filters['sectors'])}"
    tested = filters.get("tickers_tested", "?")
    st.caption(
        f"{period.get('start', '?')} to {period.get('end', '?')} | "
        f"{tested} tickers{filter_desc} | {result.get('timestamp', '')[:16]}"
    )

    # Metrics
    m = result.get("metrics", {})
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Sharpe", f"{m.get('sharpe', 0):.2f}")
    c2.metric("Return", f"{m.get('total_return', 0):.1%}")
    c3.metric("Max DD", f"{m.get('max_drawdown', 0):.1%}")
    c4.metric("Win Rate", f"{m.get('win_rate', 0):.0%}")
    c5.metric("Trades", m.get("total_trades", 0))
    c6.metric("Profit Factor", f"{m.get('profit_factor', 0):.2f}")

    # Equity curve
    eq_data = result.get("equity_curve", [])
    if eq_data:
        eq_df = pd.DataFrame(eq_data)
        eq_df["date"] = pd.to_datetime(eq_df["date"])
        st.line_chart(eq_df.set_index("date")["equity"])

    # Strategy breakdown
    sm = result.get("strategy_metrics", {})
    if sm:
        st.markdown("#### Strategy Breakdown")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Strategy": sid,
                        "Trades": v.get("trades", 0),
                        "Avg Return": f"{v.get('avg_return', 0):.2%}",
                        "Win Rate": f"{v.get('win_rate', 0):.0%}",
                        "Total P&L": f"${v.get('total_pnl', 0):,.2f}",
                    }
                    for sid, v in sm.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    # Trade log with filters
    trades = result.get("trades", [])
    if trades:
        st.markdown("#### Trade Log")
        df = pd.DataFrame(trades)

        col_ft, col_fs, col_fd = st.columns(3)
        with col_ft:
            log_tickers = sorted(df["ticker"].unique().tolist()) if "ticker" in df.columns else []
            sel_lt = st.multiselect("Ticker", log_tickers, default=[], key="bt_lt")
        with col_fs:
            log_strats = (
                sorted(df["strategy_id"].unique().tolist()) if "strategy_id" in df.columns else []
            )
            sel_ls = st.multiselect("Strategy", log_strats, default=[], key="bt_ls")
        with col_fd:
            log_dirs = (
                sorted(df["direction"].unique().tolist()) if "direction" in df.columns else []
            )
            sel_ld = st.selectbox("Direction", ["All"] + log_dirs, key="bt_ld")

        fdf = df.copy()
        if sel_lt:
            fdf = fdf[fdf["ticker"].isin(sel_lt)]
        if sel_ls:
            fdf = fdf[fdf["strategy_id"].isin(sel_ls)]
        if sel_ld != "All":
            fdf = fdf[fdf["direction"] == sel_ld]

        if len(fdf) < len(df) and "return_pct" in fdf.columns and len(fdf) > 0:
            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.metric("Filtered", len(fdf))
            fc2.metric("Avg Return", f"{fdf['return_pct'].mean():.2%}")
            fc3.metric("Win Rate", f"{(fdf['return_pct'] > 0).mean():.0%}")
            fc4.metric("P&L", f"${fdf['pnl'].sum():,.2f}")

        cols = [
            c
            for c in [
                "ticker",
                "direction",
                "strategy_id",
                "entry_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "return_pct",
                "pnl",
                "hold_days",
                "exit_reason",
                "sector",
            ]
            if c in fdf.columns
        ]
        st.dataframe(fdf[cols], use_container_width=True, hide_index=True)


# ── Tab: Settings ────────────────────────────────────────────────────────────

STRATEGY_LABELS = {
    "catalyst_capture": "Catalyst Capture",
    "volatility_breakout": "Volatility Breakout",
    "mean_reversion": "Mean Reversion",
    "sector_momentum": "Sector Momentum",
}

STRATEGY_DESCRIPTIONS = {
    "catalyst_capture": ("Buys before earnings/events when volatility is elevated. 2-week holds."),
    "volatility_breakout": (
        "BB squeeze breakouts with volume confirmation. Long and short, 1-2 weeks."
    ),
    "mean_reversion": (
        "Buys oversold stocks (low RSI + below VWAP) in healthy sectors. 1-month holds."
    ),
    "sector_momentum": (
        "Longs top momentum / shorts bottom momentum per sector. 15-day rebalance."
    ),
}

# (label, min, max, step, description)
PARAM_META = {
    "event_score_min": (
        "Min Event Score",
        0.0,
        1.0,
        0.05,
        "Minimum catalyst score (0-1). Lower = more trades, higher = only strong catalysts.",
    ),
    "atr_ratio_min": (
        "Min ATR Ratio",
        0.5,
        3.0,
        0.1,
        "ATR(5)/ATR(20). Higher = stricter vol expansion filter.",
    ),
    "iv_rank_multiplier": (
        "IV Rank Multiplier",
        0.5,
        3.0,
        0.1,
        "BB width vs 6-month median. 1.3 = 30% above median.",
    ),
    "stop_loss_pct": (
        "Stop Loss %",
        0.01,
        0.20,
        0.01,
        "Max loss before exit. Tighter = less risk but more whipsaws.",
    ),
    "take_profit_pct": (
        "Take Profit %",
        0.0,
        0.50,
        0.01,
        "Profit target. 0 = disabled. Lower = locks gains faster.",
    ),
    "max_hold_days": (
        "Max Hold Days",
        1,
        60,
        1,
        "Force exit after N days. Shorter = faster turnover.",
    ),
    "volume_spike_min": (
        "Min Volume Spike",
        1.0,
        3.0,
        0.1,
        "Volume vs 20d avg. 1.5 = 50% above average.",
    ),
    "bb_width_lookback": (
        "BB Lookback",
        20,
        252,
        1,
        "Period for BB width minimum. 126 = 6 months.",
    ),
    "vwap_dev_threshold": (
        "VWAP Dev Threshold",
        -0.10,
        0.0,
        0.005,
        "How far below VWAP. -0.02 = 2% below. More negative = stricter.",
    ),
    "rsi_threshold": (
        "RSI Threshold",
        10.0,
        50.0,
        1.0,
        "Buy below this RSI. Lower = only deeply oversold.",
    ),
    "sector_etf_rsi_min": (
        "Sector ETF RSI Floor",
        20.0,
        60.0,
        1.0,
        "Sector must be healthy (RSI above this). Higher = stricter.",
    ),
    "rebalance_days": (
        "Rebalance Days",
        5,
        30,
        1,
        "Re-rank interval. Shorter = responsive but higher costs.",
    ),
    "top_n": (
        "Long Top N",
        1,
        7,
        1,
        "Long top N per sector. More = diversified but dilutes edge.",
    ),
    "bottom_n": (
        "Short Bottom N",
        1,
        7,
        1,
        "Short bottom N per sector. More = more short exposure.",
    ),
}


def page_settings() -> None:
    st.markdown("### Strategy Settings")
    st.caption("Adjust parameters and regime weights. Changes apply on next scan or backtest.")

    params = load_strategy_params()
    updated = {}

    for strategy_id, label in STRATEGY_LABELS.items():
        st.markdown(f"#### {label}")
        desc = STRATEGY_DESCRIPTIONS.get(strategy_id)
        if desc:
            st.caption(desc)

        strategy_params = params.get(strategy_id, {})
        updated[strategy_id] = {}

        param_col, _ = st.columns([2, 1])
        with param_col:
            cols = st.columns(3)
            col_idx = 0
            for param_name, value in strategy_params.items():
                meta = PARAM_META.get(param_name)
                if meta is None:
                    updated[strategy_id][param_name] = value
                    continue
                display_name, min_val, max_val, step, help_text = meta
                with cols[col_idx % 3]:
                    if isinstance(value, int) and isinstance(step, int):
                        new_val = st.number_input(
                            display_name,
                            min_value=int(min_val),
                            max_value=int(max_val),
                            value=int(value),
                            step=int(step),
                            help=help_text,
                            key=f"{strategy_id}_{param_name}",
                        )
                    else:
                        new_val = st.number_input(
                            display_name,
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(value),
                            step=float(step),
                            format="%.4f" if abs(step) < 0.01 else "%.2f",
                            help=help_text,
                            key=f"{strategy_id}_{param_name}",
                        )
                    updated[strategy_id][param_name] = new_val
                col_idx += 1

    # Regime weights
    st.markdown("---")
    st.markdown("#### Regime Weights")
    st.caption(
        "Multiply signal strength by these weights based on market regime. "
        "0 = disabled, 1 = normal, >1 = boosted."
    )

    from signals.regime import load_regime_weights, save_regime_weights

    regime_weights = load_regime_weights()
    updated_regime: dict[str, dict[str, float]] = {}

    regime_desc = {
        "high_vol": "High Vol (HV20 > 30%): turbulent markets",
        "low_vol": "Low Vol (HV20 < 12%): calm markets",
        "normal": "Normal: baseline weights",
    }
    short_names = {
        "catalyst_capture": "Catalyst",
        "volatility_breakout": "Breakout",
        "mean_reversion": "Mean Rev",
        "sector_momentum": "Momentum",
    }

    for regime in ["high_vol", "low_vol", "normal"]:
        st.markdown(f"**{regime.replace('_', ' ').title()}** -- {regime_desc.get(regime, '')}")
        weights = regime_weights.get(regime, {})
        updated_regime[regime] = {}
        rw_col, _ = st.columns([2, 1])
        with rw_col:
            cols = st.columns(4)
            for i, (sid, w) in enumerate(weights.items()):
                with cols[i % 4]:
                    new_w = st.number_input(
                        short_names.get(sid, sid),
                        min_value=0.0,
                        max_value=3.0,
                        value=float(w),
                        step=0.1,
                        format="%.1f",
                        help=f"{sid} weight in {regime.replace('_', ' ')} regime",
                        key=f"rw_{regime}_{sid}",
                    )
                    updated_regime[regime][sid] = new_w

    # Action buttons
    st.markdown("---")
    btn_col, _ = st.columns([1, 1])
    with btn_col:
        col_save, col_scan, col_reset = st.columns(3)
        with col_save:
            if st.button("Save", type="primary", use_container_width=True):
                save_strategy_params(updated)
                save_regime_weights(updated_regime)
                st.success("Saved.")
        with col_scan:
            if st.button("Save & Scan", type="primary", use_container_width=True):
                save_strategy_params(updated)
                save_regime_weights(updated_regime)
                _run_scan()
                st.rerun()
        with col_reset:
            if st.button("Reset to Defaults", type="primary", use_container_width=True):
                from copy import deepcopy

                from signals.regime import DEFAULT_REGIME_WEIGHTS

                save_strategy_params(deepcopy(DEFAULT_STRATEGY_PARAMS))
                default_regime = {r: dict(w) for r, w in DEFAULT_REGIME_WEIGHTS.items()}
                save_regime_weights(default_regime)
                st.success("Reset to defaults.")
                st.rerun()


# ── Tab: Watchlist ───────────────────────────────────────────────────────────


def page_watchlist() -> None:
    st.markdown("### Watchlist Manager")
    st.caption("Add, remove, and organize tickers across sectors.")

    from scanner.universe import Universe

    universe = Universe()
    sectors = universe.get_sectors()

    c1, c2 = st.columns(2)
    c1.metric("Total Tickers", len(universe.get_unique_tickers()))
    c2.metric("Sectors", len(sectors))

    # Per-sector view
    for sector in sectors:
        tickers = universe.get_tickers(sector=sector)
        etf = universe.get_sector_etf(sector)
        with st.expander(
            f"{sector.replace('_', ' ').title()} ({len(tickers)}, ETF: {etf})",
            expanded=False,
        ):
            cols = st.columns(5)
            for i, t in enumerate(sorted(tickers)):
                with cols[i % 5]:
                    st.code(t)
            rm = st.selectbox("Remove", [""] + sorted(tickers), key=f"rm_{sector}")
            if rm and st.button(f"Remove {rm}", key=f"rm_btn_{sector}"):
                universe.remove_ticker(rm)
                universe.save()
                st.rerun()

    st.markdown("---")

    # Add ticker
    st.markdown("#### Add Ticker")
    col_t, col_s, col_b = st.columns([2, 2, 1])
    with col_t:
        new_t = st.text_input("Symbol", placeholder="TSLA", key="add_t").upper().strip()
    with col_s:
        tgt = st.selectbox("Sector", sectors, key="add_s")
    with col_b:
        st.write("")
        st.write("")
        if st.button("Add", type="primary", key="add_btn") and new_t:
            if universe.add_ticker(new_t, tgt):
                universe.save()
                st.success(f"Added {new_t} to {tgt}.")
                st.rerun()
            else:
                ex = universe.get_ticker_sector(new_t)
                st.warning(f"{new_t} already in {ex}." if ex else f"Could not add {new_t}.")

    st.markdown("---")

    # Add sector
    st.markdown("#### Add Sector")
    col_n, col_e, col_b2 = st.columns([2, 2, 1])
    with col_n:
        ns = st.text_input("Name", placeholder="healthcare", key="ns_name")
    with col_e:
        ne = st.text_input("ETF", placeholder="XLV", key="ns_etf")
    with col_b2:
        st.write("")
        st.write("")
        if st.button("Add Sector", key="ns_btn", type="primary") and ns and ne:
            if universe.add_sector(ns, ne):
                universe.save()
                st.success(f"Added {ns} with ETF {ne.upper()}.")
                st.rerun()
            else:
                st.warning(f"Sector '{ns}' already exists.")

    st.markdown("---")

    # Bulk add
    st.markdown("#### Bulk Add")
    bs = st.selectbox("Target sector", sectors, key="bulk_s")
    bt = st.text_area(
        "Tickers (comma or space separated)", placeholder="TSLA, META, GOOG", key="bulk_t"
    )
    if st.button("Add All", key="bulk_btn", type="primary") and bt:
        import re

        to_add = [t.strip().upper() for t in re.split(r"[,\s]+", bt) if t.strip()]
        added, skipped = [], []
        for t in to_add:
            (added if universe.add_ticker(t, bs) else skipped).append(t)
        if added:
            universe.save()
            st.success(f"Added {len(added)}: {', '.join(added)}")
        if skipped:
            st.warning(f"Skipped: {', '.join(skipped)}")
        if added:
            st.rerun()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Volatility Signals",
        page_icon="chart_with_upwards_trend",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Sidebar
    with st.sidebar:
        st.markdown(
            '<p style="font-size:2em; font-weight:bold; margin:0;">Volatility Signals</p>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(
            "**Quick Start**\n"
            "1. Go to **Watchlist** to manage tickers\n"
            "2. Go to **Stock Universe** and click **Run Scan**\n"
            "3. Check **Signals** for actionable trades\n"
            "4. Run **Backtest** to test strategies historically\n"
            "5. Tune **Settings** to adjust parameters"
        )
        st.markdown("---")
        report = load_scan_report()
        ts = report.get("timestamp")
        if ts:
            st.caption(f"Last scan: {ts[:16]}")
        bt = load_backtest_result()
        if bt and "error" not in bt:
            st.caption(f"Last backtest: {bt.get('timestamp', '')[:16]}")
            m = bt.get("metrics", {})
            st.caption(f"Sharpe: {m.get('sharpe', 0):.2f} | Trades: {m.get('total_trades', 0)}")

    # Apply light theme styling
    _apply_theme()

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Stock Universe",
            "Signals",
            "Backtest",
            "Settings",
            "Watchlist",
        ]
    )

    with tab1:
        page_universe()
    with tab2:
        page_signals()
    with tab3:
        page_backtest()
    with tab4:
        page_settings()
    with tab5:
        page_watchlist()


if __name__ == "__main__":
    main()
