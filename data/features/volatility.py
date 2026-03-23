"""Volatility feature computations: HV, ATR, Bollinger Bands, Garman-Klass."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(close: pd.Series) -> pd.Series:
    """Compute log returns from close prices."""
    return np.log(close / close.shift(1))


def compute_historical_volatility(
    close: pd.Series,
    windows: tuple[int, ...] = (5, 10, 20, 60),
) -> pd.DataFrame:
    """Compute annualized historical volatility for multiple windows.

    HV_N = sqrt(252) * std(log_returns, window=N)
    """
    log_ret = compute_log_returns(close)
    result = pd.DataFrame(index=close.index)

    for w in windows:
        result[f"hv_{w}"] = log_ret.rolling(window=w).std() * np.sqrt(252)

    return result


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    windows: tuple[int, ...] = (5, 14, 20),
) -> pd.DataFrame:
    """Compute Average True Range normalized by close price.

    TR = max(H-L, |H-prev_C|, |L-prev_C|)
    ATR_N = EMA(TR, N) / Close
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    result = pd.DataFrame(index=close.index)
    for w in windows:
        atr = tr.ewm(span=w, adjust=False).mean()
        result[f"atr_{w}"] = atr / close  # Normalized by price

    return result


def compute_bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Compute Bollinger Band width and %B position.

    BB_width = (upper - lower) / middle
    BB_pct_b = (close - lower) / (upper - lower)
    """
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()

    upper = sma + num_std * std
    lower = sma - num_std * std

    width = upper - lower
    result = pd.DataFrame(index=close.index)
    result["bb_width"] = width / sma
    result["bb_pct_b"] = (close - lower) / width

    return result


def compute_garman_klass(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Compute Garman-Klass volatility estimator (annualized).

    GK = sqrt(252 * rolling_mean(0.5*(log(H/L))^2 - (2*ln2-1)*(log(C/O))^2))
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)

    gk_daily = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    gk_vol = np.sqrt(252 * gk_daily.rolling(window=window).mean())

    return gk_vol.rename("gk_vol")


def compute_all_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all volatility features from an OHLCV DataFrame.

    Args:
        df: DataFrame with Open, High, Low, Close, Volume columns.

    Returns:
        DataFrame with all volatility feature columns.
    """
    features = pd.DataFrame(index=df.index)

    # Historical volatility
    hv = compute_historical_volatility(df["Close"])
    features = features.join(hv)

    # ATR (normalized)
    atr = compute_atr(df["High"], df["Low"], df["Close"])
    features = features.join(atr)

    # Bollinger Bands
    bb = compute_bollinger_bands(df["Close"])
    features = features.join(bb)

    # Garman-Klass
    gk = compute_garman_klass(df["Open"], df["High"], df["Low"], df["Close"])
    features = features.join(gk)

    return features
