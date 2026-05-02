"""Bar-by-bar backtester.

Same momentum engine, same strategy, same risk math as live — only the
execution side is replaced with a simulated fill model. Walks the
confirm timeframe and slices the higher timeframes up to each bar so
the engine sees only data that would have been available at decision
time (no look-ahead).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

from core.logger import get_logger
from momentum.engine import MomentumEngine
from momentum.regime import Regime
from risk.manager import RiskConfig, RiskManager
from strategy.base import Signal, Strategy

log = get_logger(__name__)


@dataclass
class BTTrade:
    instrument: str
    side: str
    entry_time: datetime
    entry: float
    stop: float
    target: float
    units: int
    exit_time: datetime | None = None
    exit: float | None = None
    pnl: float = 0.0
    reason_in: str = ""
    reason_out: str = ""


@dataclass
class BTResult:
    equity_curve: pd.Series
    trades: list[BTTrade] = field(default_factory=list)

    def stats(self) -> dict:
        eq = self.equity_curve
        if len(eq) < 2:
            return {"trades": 0}
        ret = eq.pct_change().dropna()
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = -sum(t.pnl for t in losses)
        peak = eq.cummax()
        dd = (eq - peak) / peak
        sharpe = float(ret.mean() / ret.std() * np.sqrt(252 * 24)) if ret.std() > 0 else 0.0
        return {
            "trades": len(self.trades),
            "win_rate": len(wins) / len(self.trades) if self.trades else 0.0,
            "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
            "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
            "max_drawdown": float(dd.min()),
            "sharpe_hourly_ann": sharpe,
            "final_equity": float(eq.iloc[-1]),
        }


def run_backtest(
    instrument: str,
    candles_by_tf: dict[str, pd.DataFrame],
    confirm_tf: str,
    engine: MomentumEngine,
    strategy: Strategy,
    risk_cfg: RiskConfig,
    starting_equity: float = 10_000.0,
    qhcf: float = 1.0,
    spread: float = 0.00002,
) -> BTResult:
    """Run a single-instrument backtest.

    candles_by_tf: must contain `confirm_tf` plus any tfs the engine needs.
    qhcf: quote-home conversion factor; constant approximation for backtest.
          For X/USD pairs with USD account use 1.0; for USD/JPY use ~1/price.
    """
    confirm = candles_by_tf[confirm_tf]
    if confirm.empty:
        return BTResult(equity_curve=pd.Series(dtype=float))

    rm = RiskManager(risk_cfg)
    equity = starting_equity
    equity_history: list[tuple[datetime, float]] = []
    trades: list[BTTrade] = []
    open_trade: BTTrade | None = None

    pricing = {
        instrument: {
            "tradeable": True,
            "qhcf_pos": qhcf,
            "qhcf_neg": qhcf,
            "bid": 0,
            "ask": 0,
            "mid": 0,
        }
    }

    for ts, bar in confirm.iterrows():
        # 1) Manage open trade against this bar's high/low (stop/target hit?).
        if open_trade is not None:
            hit_stop = bar["low"] <= open_trade.stop if open_trade.side == "buy" else bar["high"] >= open_trade.stop
            hit_target = bar["high"] >= open_trade.target if open_trade.side == "buy" else bar["low"] <= open_trade.target
            exit_price: float | None = None
            reason_out = ""
            # Conservative: if both touched in the same bar, assume stop hit first.
            if hit_stop:
                exit_price = open_trade.stop
                reason_out = "stop"
            elif hit_target:
                exit_price = open_trade.target
                reason_out = "target"
            if exit_price is not None:
                direction = 1 if open_trade.side == "buy" else -1
                pnl = direction * (exit_price - open_trade.entry) * abs(open_trade.units) * qhcf
                open_trade.exit_time = ts
                open_trade.exit = exit_price
                open_trade.pnl = pnl
                open_trade.reason_out = reason_out
                equity += pnl
                trades.append(open_trade)
                open_trade = None

        equity_history.append((ts, equity))
        rm.update_equity(equity, now=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts)
        if rm.halted or open_trade is not None:
            continue

        # 2) Slice all tfs up to (and including) this bar's timestamp.
        sliced: dict[str, pd.DataFrame] = {}
        for tf, df in candles_by_tf.items():
            sliced[tf] = df.loc[:ts]
            if sliced[tf].empty:
                break
        else:
            state = engine.compute(instrument, sliced)
            if state.regime != Regime.NEUTRAL and state.aligned:
                signal = strategy.evaluate(instrument, state, sliced)
                if signal is not None:
                    sized = rm.size(signal, equity, pricing)
                    if sized is not None:
                        # Fill at next bar's open (or this close + spread as fallback).
                        fill = float(bar["close"]) + (spread if signal.side == "buy" else -spread)
                        open_trade = BTTrade(
                            instrument=instrument,
                            side=signal.side,
                            entry_time=ts,
                            entry=fill,
                            stop=signal.stop,
                            target=signal.target,
                            units=sized.units,
                            reason_in=signal.reason,
                        )

    eq = pd.Series({t: v for t, v in equity_history}, name="equity")
    return BTResult(equity_curve=eq, trades=trades)


def load_history_from_broker(
    broker,
    instrument: str,
    timeframes: list[str],
    bars_per_tf: int = 1500,
) -> dict[str, pd.DataFrame]:
    """Pull recent history for every timeframe via the OANDA broker."""
    out = {}
    for tf in timeframes:
        out[tf] = broker.candles(instrument, tf, count=bars_per_tf)
    return out
