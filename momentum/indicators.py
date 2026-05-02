"""Pure-function technical indicators used by the momentum engine.

Each function takes pandas Series / DataFrames and returns a pandas object —
no state, no side effects, easy to test and to reuse from a backtester.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def roc(series: pd.Series, n: int) -> pd.Series:
    """Rate of change over n bars (fractional, not %)."""
    return series / series.shift(n) - 1.0


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder's ADX, +DI, -DI. Returns (adx, plus_di, minus_di)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)

    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / n, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_ = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx_.fillna(0), plus_di.fillna(0), minus_di.fillna(0)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def donchian(high: pd.Series, low: pd.Series, n: int) -> tuple[pd.Series, pd.Series]:
    """Returns (upper, lower) bands over n bars (excludes current bar)."""
    upper = high.shift(1).rolling(n).max()
    lower = low.shift(1).rolling(n).min()
    return upper, lower


def slope(series: pd.Series, n: int) -> pd.Series:
    """Fractional change of value vs n bars ago."""
    return series / series.shift(n) - 1.0
