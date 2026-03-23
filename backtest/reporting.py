"""Performance reporting and tearsheet generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from backtest.engine import BacktestResult

logger = structlog.get_logger(__name__)


def generate_tearsheet(result: BacktestResult, title: str = "Strategy") -> dict[str, object]:
    """Generate tearsheet data (figures require matplotlib at render time).

    Returns dict of chart data that can be rendered or saved.
    """
    metrics = result.metrics()
    equity = result.equity_curve
    returns = result.daily_returns.dropna()

    # Monthly returns
    if len(returns) > 0:
        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    else:
        monthly = pd.Series(dtype=float)

    # Rolling Sharpe (252-day)
    if len(returns) >= 252:
        rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
    else:
        rolling_sharpe = pd.Series(dtype=float)

    # Drawdown series
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak

    return {
        "title": title,
        "metrics": metrics,
        "equity_curve": equity,
        "monthly_returns": monthly,
        "rolling_sharpe": rolling_sharpe,
        "drawdown": drawdown,
    }


def render_tearsheet(tearsheet: dict[str, object], output_dir: Path | str) -> list[str]:
    """Render tearsheet to matplotlib figures and save as PNG files.

    Returns list of saved file paths.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    title = tearsheet["title"]

    # 1. Equity curve
    fig, ax = plt.subplots(figsize=(12, 5))
    equity = tearsheet["equity_curve"]
    if len(equity) > 0:
        ax.plot(equity.index, equity.values, linewidth=1)
        ax.set_title(f"{title} — Equity Curve")
        ax.set_ylabel("Portfolio Value ($)")
        ax.grid(True, alpha=0.3)
    path = str(output_dir / f"{title.lower().replace(' ', '_')}_equity.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # 2. Drawdown chart
    fig, ax = plt.subplots(figsize=(12, 4))
    dd = tearsheet["drawdown"]
    if len(dd) > 0:
        ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
        ax.set_title(f"{title} — Drawdown")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)
    path = str(output_dir / f"{title.lower().replace(' ', '_')}_drawdown.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # 3. Monthly returns heatmap
    monthly = tearsheet["monthly_returns"]
    if len(monthly) > 0:
        monthly_df = pd.DataFrame(
            {
                "year": monthly.index.year,
                "month": monthly.index.month,
                "return": monthly.values,
            }
        )
        pivot = monthly_df.pivot_table(
            values="return", index="year", columns="month", aggfunc="sum"
        )
        fig, ax = plt.subplots(figsize=(12, max(3, len(pivot) * 0.5)))
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(12))
        ax.set_xticklabels(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        )
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"{title} — Monthly Returns")
        fig.colorbar(im)
        path = str(output_dir / f"{title.lower().replace(' ', '_')}_monthly.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    logger.info("tearsheet_saved", title=title, files=len(saved))
    return saved


def comparison_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Generate a comparison table across multiple strategies."""
    rows = []
    for name, result in results.items():
        m = result.metrics()
        rows.append(
            {
                "Strategy": name,
                "Sharpe": round(m.get("sharpe", 0), 2),
                "Sortino": round(m.get("sortino", 0), 2),
                "Calmar": round(m.get("calmar", 0), 2),
                "Max DD": f"{m.get('max_drawdown', 0):.1%}",
                "Win Rate": f"{m.get('win_rate', 0):.0%}",
                "Avg Hold": f"{m.get('avg_hold_days', 0):.1f}d",
                "Trades": int(m.get("total_trades", 0)),
                "Profit Factor": round(m.get("profit_factor", 0), 2),
            }
        )
    return pd.DataFrame(rows).set_index("Strategy")


def save_trade_log(result: BacktestResult, path: Path | str) -> None:
    """Export trade log to CSV."""
    if not result.trades:
        return

    rows = [
        {
            "ticker": t.ticker,
            "direction": t.direction,
            "strategy": t.strategy_id,
            "entry_date": t.entry_date,
            "entry_price": round(t.entry_price, 2),
            "exit_date": t.exit_date,
            "exit_price": round(t.exit_price, 2),
            "return_pct": round(t.return_pct, 4),
            "pnl": round(t.pnl, 2),
            "hold_days": t.hold_days,
            "exit_reason": t.exit_reason,
            "sector": t.sector,
        }
        for t in result.trades
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("trade_log_saved", path=str(path), trades=len(rows))
