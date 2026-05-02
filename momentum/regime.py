"""Momentum regime types.

A regime is the bot's *understanding* of where the instrument is — not a
trade signal. Strategies consume regime; the regime layer never trades.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Regime(str, Enum):
    STRONG_UP = "STRONG_UP"
    UP = "UP"
    NEUTRAL = "NEUTRAL"
    DOWN = "DOWN"
    STRONG_DOWN = "STRONG_DOWN"

    @property
    def direction(self) -> int:
        return {
            Regime.STRONG_UP: 1, Regime.UP: 1,
            Regime.NEUTRAL: 0,
            Regime.DOWN: -1, Regime.STRONG_DOWN: -1,
        }[self]

    @property
    def is_strong(self) -> bool:
        return self in (Regime.STRONG_UP, Regime.STRONG_DOWN)


@dataclass
class MomentumState:
    instrument: str
    score: float                    # weighted [-1, 1]
    per_tf_score: dict[str, float]  # signed score per timeframe
    per_tf_strength: dict[str, float]  # ADX-derived [0, 1] per tf
    trend_strength: float           # weighted ADX strength [0, 1]
    volatility: float               # ATR / price on the slowest tf
    aligned: bool                   # all tfs agree on sign
    regime: Regime
    last_close: float = 0.0
    extras: dict = field(default_factory=dict)

    def __str__(self) -> str:
        per_tf = " ".join(f"{tf}:{s:+.2f}" for tf, s in self.per_tf_score.items())
        align = "✓" if self.aligned else "✗"
        return (
            f"{self.instrument} {self.regime.value:<11} "
            f"score={self.score:+.2f} strength={self.trend_strength:.2f} "
            f"align={align} [{per_tf}]"
        )
