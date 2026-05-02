"""Donchian breakout, gated by the momentum engine.

The strategy itself is intentionally simple: a clean Donchian break on
the confirmation timeframe with an ATR-based stop. The engine handles
all the trend/regime reasoning upstream — this layer only adds the
trigger and the levels.
"""
from __future__ import annotations

import pandas as pd

from momentum.indicators import atr, donchian
from momentum.regime import MomentumState, Regime
from strategy.base import Signal, Strategy


class MomentumBreakout(Strategy):
    name = "momentum_breakout"

    def __init__(
        self,
        breakout_lookback: int = 20,
        confirm_timeframe: str = "H1",
        atr_stop_mult: float = 2.0,
        reward_risk: float = 2.0,
    ):
        self.lookback = breakout_lookback
        self.tf = confirm_timeframe
        self.atr_mult = atr_stop_mult
        self.rr = reward_risk

    def evaluate(
        self,
        instrument: str,
        momentum: MomentumState,
        candles_by_tf: dict[str, pd.DataFrame],
    ) -> Signal | None:
        # The main loop already guarantees regime is non-neutral and aligned,
        # but we re-check defensively so the strategy is safe to call standalone.
        if momentum.regime == Regime.NEUTRAL or not momentum.aligned:
            return None

        df = candles_by_tf.get(self.tf)
        if df is None or len(df) < self.lookback + 5:
            return None

        upper, lower = donchian(df["high"], df["low"], self.lookback)
        a = atr(df["high"], df["low"], df["close"], 14)
        last = df.iloc[-1]
        u = upper.iloc[-1]
        l = lower.iloc[-1]
        a_now = a.iloc[-1]
        if any(pd.isna(x) for x in (u, l, a_now)):
            return None

        side: str | None = None
        if momentum.regime.direction > 0 and last["close"] > u:
            side = "buy"
        elif momentum.regime.direction < 0 and last["close"] < l:
            side = "sell"
        if side is None:
            return None

        entry = float(last["close"])
        if side == "buy":
            stop = entry - self.atr_mult * float(a_now)
            target = entry + self.rr * (entry - stop)
        else:
            stop = entry + self.atr_mult * float(a_now)
            target = entry - self.rr * (stop - entry)

        return Signal(
            instrument=instrument,
            side=side,
            entry=entry,
            stop=stop,
            target=target,
            reason=f"{self.lookback}-bar Donchian {side} on {self.tf} | regime={momentum.regime.value}",
        )
