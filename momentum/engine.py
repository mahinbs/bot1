"""The momentum engine.

This is the foundational layer of the bot: BEFORE any strategy condition
can fire, every instrument is scored across multiple timeframes and
classified into a regime. Strategies are gated by this output.

Per timeframe the score combines four orthogonal momentum signals:
  - ROC (raw price displacement, ATR-normalised)
  - EMA alignment + slope (trend structure)
  - MACD histogram (acceleration)
  - Donchian position (where in the recent range we sit)

Each is squashed into [-1, 1], averaged, then ADX is used as a strength
gate. Per-timeframe scores are weighted (default H4 / H1 dominate over
M15) into a single number, and the regime classifier uses this score
plus alignment + ADX to label the state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from momentum.indicators import adx, atr, donchian, ema, macd, roc, slope
from momentum.regime import MomentumState, Regime


def _safe(x: float, default: float = 0.0) -> float:
    if x is None or np.isnan(x) or np.isinf(x):
        return default
    return float(x)


def _score_timeframe(df: pd.DataFrame) -> tuple[float, float]:
    """Return (signed_score in [-1, 1], strength in [0, 1]) for one timeframe."""
    if len(df) < 60:
        return 0.0, 0.0

    close, high, low = df["close"], df["high"], df["low"]

    # ATR-normalised displacement; tanh keeps it bounded.
    a = atr(high, low, close, 14)
    atr_pct = _safe(a.iloc[-1] / close.iloc[-1], 0.0)
    if atr_pct <= 0:
        return 0.0, 0.0

    # 1) Rate of change, normalised by typical move size.
    r20 = _safe(roc(close, 20).iloc[-1])
    roc_score = float(np.tanh(r20 / (atr_pct * 5)))

    # 2) EMA alignment + slope.
    ema_fast = ema(close, 21)
    ema_slow = ema(close, 55)
    align = 1.0 if ema_fast.iloc[-1] > ema_slow.iloc[-1] else -1.0
    slope_val = _safe(slope(ema_fast, 5).iloc[-1])
    slope_norm = float(np.tanh(slope_val / atr_pct))
    ema_score = align * abs(slope_norm)  # signed by alignment, magnitude by slope

    # 3) MACD histogram, normalised.
    _, _, hist = macd(close)
    hist_norm = _safe(hist.iloc[-1] / (a.iloc[-1] * 0.5))
    macd_score = float(np.tanh(hist_norm))

    # 4) Donchian position: where is close inside the n-bar range? Maps to [-1, 1].
    upper, lower = donchian(high, low, 20)
    rng = _safe(upper.iloc[-1] - lower.iloc[-1])
    if rng > 0:
        pos = (close.iloc[-1] - (upper.iloc[-1] + lower.iloc[-1]) / 2) / (rng / 2)
        donch_score = float(np.clip(pos, -1, 1))
    else:
        donch_score = 0.0

    raw = 0.30 * roc_score + 0.30 * ema_score + 0.20 * macd_score + 0.20 * donch_score
    score = float(np.clip(raw, -1, 1))

    # Strength: ADX maps 0..50 -> 0..1, capped.
    adx_, _, _ = adx(high, low, close, 14)
    strength = float(min(_safe(adx_.iloc[-1]) / 50.0, 1.0))

    return score, strength


class MomentumEngine:
    def __init__(
        self,
        timeframes: list[str],
        weights: dict[str, float],
        thresholds: dict,
    ):
        if not timeframes:
            raise ValueError("timeframes is empty")
        # Normalise weights to sum to 1 across the configured timeframes.
        total = sum(weights.get(tf, 0.0) for tf in timeframes)
        if total <= 0:
            self.weights = {tf: 1.0 / len(timeframes) for tf in timeframes}
        else:
            self.weights = {tf: weights.get(tf, 0.0) / total for tf in timeframes}
        self.timeframes = timeframes
        self.t_strong = float(thresholds.get("strong", 0.55))
        self.t_weak = float(thresholds.get("weak", 0.20))
        self.adx_trending = float(thresholds.get("adx_trending", 20.0)) / 50.0  # to [0,1]

    def compute(self, instrument: str, candles_by_tf: dict[str, pd.DataFrame]) -> MomentumState:
        per_score: dict[str, float] = {}
        per_strength: dict[str, float] = {}
        for tf in self.timeframes:
            df = candles_by_tf.get(tf)
            if df is None or df.empty:
                per_score[tf], per_strength[tf] = 0.0, 0.0
                continue
            s, st = _score_timeframe(df)
            per_score[tf] = s
            per_strength[tf] = st

        score = sum(per_score[tf] * self.weights[tf] for tf in self.timeframes)
        trend_strength = sum(per_strength[tf] * self.weights[tf] for tf in self.timeframes)

        signs = [np.sign(per_score[tf]) for tf in self.timeframes if abs(per_score[tf]) > 0.05]
        aligned = bool(signs) and len(set(signs)) == 1

        # Volatility from the slowest timeframe in config (last in list).
        slowest = self.timeframes[-1]
        slow_df = candles_by_tf.get(slowest)
        last_close = 0.0
        volatility = 0.0
        if slow_df is not None and not slow_df.empty:
            last_close = float(slow_df["close"].iloc[-1])
            a = atr(slow_df["high"], slow_df["low"], slow_df["close"], 14).iloc[-1]
            volatility = _safe(a / last_close)

        regime = self._classify(score, aligned, trend_strength)

        return MomentumState(
            instrument=instrument,
            score=float(score),
            per_tf_score=per_score,
            per_tf_strength=per_strength,
            trend_strength=float(trend_strength),
            volatility=float(volatility),
            aligned=aligned,
            regime=regime,
            last_close=last_close,
        )

    def _classify(self, score: float, aligned: bool, trend_strength: float) -> Regime:
        # No real trend → NEUTRAL no matter what the score says.
        if trend_strength < self.adx_trending:
            return Regime.NEUTRAL
        a = abs(score)
        if a < self.t_weak:
            return Regime.NEUTRAL
        # Strong only if score is high AND timeframes agree.
        if a >= self.t_strong and aligned:
            return Regime.STRONG_UP if score > 0 else Regime.STRONG_DOWN
        return Regime.UP if score > 0 else Regime.DOWN
