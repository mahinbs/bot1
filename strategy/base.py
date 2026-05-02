"""Strategy interface.

A strategy receives the momentum state and the candles, and returns
a Signal or None. It does NOT decide whether momentum is good enough
to trade — that gating happens in the main loop, BEFORE evaluate() is
called. The strategy only adds its own conditions on top.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from momentum.regime import MomentumState


@dataclass
class Signal:
    instrument: str
    side: str           # 'buy' or 'sell'
    entry: float        # reference price (mid); execution uses live ask/bid
    stop: float
    target: float
    reason: str

    @property
    def stop_distance(self) -> float:
        return abs(self.entry - self.stop)


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def evaluate(
        self,
        instrument: str,
        momentum: MomentumState,
        candles_by_tf: dict[str, pd.DataFrame],
    ) -> Signal | None: ...
