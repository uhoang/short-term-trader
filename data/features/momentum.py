"""Momentum and mean reversion feature computations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.features.volatility import compute_historical_volatility


def compute_vol_adjusted_momentum(
    close: pd.Series,
    windows: tuple[int, ...] = (10, 20, 60),
) -> pd.DataFrame:
    """Compute volatility-adjusted momentum: N-day return / HV_N.

    Strong moves relative to recent volatility produce high scores.
    """
    hv = compute_historical_volatility(close, windows=windows)
    result = pd.DataFrame(index=close.index)

    for w in windows:
        n_day_return = close.pct_change(periods=w)
        hv_col = hv[f"hv_{w}"]
        # Avoid division by zero
        result[f"vol_adj_mom_{w}"] = n_day_return / hv_col.replace(0, np.nan)

    return result


def compute_vwap_deviation(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Compute deviation from volume-weighted average price.

    VWAP_dev = (close - 20d_VWAP) / 20d_VWAP
    """
    vwap = (close * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()
    return ((close - vwap) / vwap).rename("vwap_dev")


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing method.

    RSI = 100 - 100 / (1 + RS)
    RS = avg_gain / avg_loss (exponential moving average)
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename("rsi_14")


def compute_sector_relative_rsi(
    rsi_series: dict[str, pd.Series],
    sector_map: dict[str, str],
) -> dict[str, pd.Series]:
    """Compute sector-relative RSI: RSI - sector median RSI.

    Args:
        rsi_series: dict of ticker -> RSI Series
        sector_map: dict of ticker -> sector name

    Returns:
        dict of ticker -> relative RSI Series
    """
    # Group tickers by sector
    sector_tickers: dict[str, list[str]] = {}
    for ticker, sector in sector_map.items():
        sector_tickers.setdefault(sector, []).append(ticker)

    result: dict[str, pd.Series] = {}
    for sector, tickers in sector_tickers.items():
        available = [t for t in tickers if t in rsi_series]
        if not available:
            continue

        # Align all RSI series and compute rolling median
        sector_df = pd.DataFrame({t: rsi_series[t] for t in available})
        sector_median = sector_df.median(axis=1)

        for ticker in available:
            result[ticker] = (rsi_series[ticker] - sector_median).rename("rsi_sector_rel")

    return result


def compute_sector_zscore(
    returns_dict: dict[str, pd.Series],
    sector_map: dict[str, str],
    window: int = 60,
) -> dict[str, pd.Series]:
    """Compute rolling z-score of returns within sector peer group.

    For each stock: z = (return - sector_median) / sector_std
    """
    sector_tickers: dict[str, list[str]] = {}
    for ticker, sector in sector_map.items():
        sector_tickers.setdefault(sector, []).append(ticker)

    result: dict[str, pd.Series] = {}
    for sector, tickers in sector_tickers.items():
        available = [t for t in tickers if t in returns_dict]
        if len(available) < 2:
            continue

        sector_df = pd.DataFrame(
            {t: returns_dict[t].rolling(window=window).mean() for t in available}
        )
        sector_median = sector_df.median(axis=1)
        sector_std = sector_df.std(axis=1)

        for ticker in available:
            z = (sector_df[ticker] - sector_median) / sector_std.replace(0, np.nan)
            result[ticker] = z.rename("sector_zscore")

    return result


def compute_52w_proximity(close: pd.Series) -> pd.DataFrame:
    """Compute proximity to 52-week high and low with threshold flags.

    Returns columns: dist_52w_high, dist_52w_low, near_52w_high_2/5/10, near_52w_low_2/5/10
    """
    high_52w = close.rolling(window=252, min_periods=60).max()
    low_52w = close.rolling(window=252, min_periods=60).min()

    result = pd.DataFrame(index=close.index)
    result["dist_52w_high"] = (close - high_52w) / high_52w
    result["dist_52w_low"] = (close - low_52w) / low_52w

    # Binary flags: within X% of 52-week extreme
    for pct in (0.02, 0.05, 0.10):
        pct_label = int(pct * 100)
        result[f"near_52w_high_{pct_label}"] = (result["dist_52w_high"].abs() <= pct).astype(int)
        result[f"near_52w_low_{pct_label}"] = (result["dist_52w_low"] <= pct).astype(int)

    return result


def compute_all_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all single-ticker momentum/reversion features from OHLCV.

    Sector-relative features (RSI relative, z-score) require multi-ticker
    computation and are handled separately in the FeatureStore.
    """
    features = pd.DataFrame(index=df.index)

    # Vol-adjusted momentum
    mom = compute_vol_adjusted_momentum(df["Close"])
    features = features.join(mom)

    # VWAP deviation
    vwap = compute_vwap_deviation(df["Close"], df["Volume"])
    features = features.join(vwap)

    # RSI
    rsi = compute_rsi(df["Close"])
    features = features.join(rsi)

    # 52-week proximity
    prox = compute_52w_proximity(df["Close"])
    features = features.join(prox)

    return features
