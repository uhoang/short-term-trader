"""Statistical analysis and bias detection for backtest validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from backtest.engine import BacktestResult


def bootstrap_test(
    trade_returns: list[float],
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap permutation test: shuffle trade dates and compare Sharpe.

    Returns dict with real_sharpe, p_value, and percentile.
    """
    if len(trade_returns) < 5:
        return {"real_sharpe": 0, "p_value": 1.0, "percentile": 0}

    rng = np.random.RandomState(seed)
    returns = np.array(trade_returns)
    real_sharpe = _sharpe_from_trades(returns)

    shuffled_sharpes = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(returns)
        shuffled_sharpes.append(_sharpe_from_trades(shuffled))

    shuffled_sharpes = np.array(shuffled_sharpes)
    percentile = (real_sharpe > shuffled_sharpes).sum() / n_permutations * 100
    p_value = 1 - percentile / 100

    return {
        "real_sharpe": float(real_sharpe),
        "p_value": float(p_value),
        "percentile": float(percentile),
    }


def ttest_mean_return(trade_returns: list[float]) -> dict[str, float]:
    """One-sample t-test: is mean trade return significantly > 0?

    Returns dict with t_stat, p_value, mean_return, and significant flag.
    """
    if len(trade_returns) < 3:
        return {"t_stat": 0, "p_value": 1.0, "mean_return": 0, "significant": False}

    t_stat, p_value = stats.ttest_1samp(trade_returns, 0)
    mean_return = np.mean(trade_returns)

    # One-sided test (we want mean > 0)
    p_one_sided = p_value / 2 if t_stat > 0 else 1 - p_value / 2

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_one_sided),
        "mean_return": float(mean_return),
        "significant": p_one_sided < 0.05 and mean_return > 0,
    }


def cost_sensitivity(
    result: BacktestResult,
    multipliers: list[float] | None = None,
) -> dict[str, float]:
    """Re-estimate Sharpe at different cost levels.

    Approximates impact by adjusting trade returns for higher costs.
    """
    if multipliers is None:
        multipliers = [1.0, 2.0, 5.0]

    baseline_cost_per_trade = (
        result.config.slippage_pct * 2  # Entry + exit slippage
        + result.config.commission_per_share * 2 / 100  # Approx as pct
    )

    trade_returns = [t.return_pct for t in result.trades]
    if not trade_returns:
        return {f"{m}x": 0.0 for m in multipliers}

    results = {}
    for mult in multipliers:
        extra_cost = baseline_cost_per_trade * (mult - 1)
        adjusted = [r - extra_cost for r in trade_returns]
        sharpe = _sharpe_from_trades(np.array(adjusted))
        results[f"{mult}x"] = float(sharpe)

    return results


def strategy_correlation(
    daily_returns: dict[str, pd.Series],
) -> pd.DataFrame:
    """Compute pairwise correlation between strategy daily returns."""
    df = pd.DataFrame(daily_returns)
    return df.corr()


def _sharpe_from_trades(returns: np.ndarray) -> float:
    """Compute annualized Sharpe from trade returns."""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    # Assume ~20 trades per year as rough annualization
    trades_per_year = min(len(returns), 20)
    mean_return = np.mean(returns) * trades_per_year
    vol = np.std(returns) * np.sqrt(trades_per_year)
    return float(mean_return / vol) if vol > 0 else 0.0
