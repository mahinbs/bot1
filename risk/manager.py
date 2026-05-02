"""Risk manager.

Owns three concerns the strategy must not own:

1. Sizing — given account equity, the configured per-trade risk %, and
   the strategy's stop distance, returns an integer unit count expressed
   in the instrument's base currency. Uses OANDA's quote-home conversion
   factor so the math works for non-home-quoted pairs (e.g. USD/JPY).

2. Pre-trade gating — concurrent position cap, per-instrument cap,
   instrument tradeable check, daily loss halt.

3. Bookkeeping for the daily loss halt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime

from core.broker import Broker, OpenPosition
from core.logger import get_logger
from strategy.base import Signal

log = get_logger(__name__)


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.005
    max_concurrent_positions: int = 3
    one_position_per_instrument: bool = True
    daily_loss_halt: float = 0.03


@dataclass
class SizedOrder:
    units: int            # signed; negative = short
    risk_amount: float    # in account currency
    stop_distance: float
    notional: float       # units * entry, in base currency
    reason: str = ""


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self._day = date.today()
        self._day_start_equity: float | None = None
        self._halted = False

    # ------- daily PnL halt -------

    def update_equity(self, equity: float, now: datetime | date | None = None) -> None:
        today = now.date() if isinstance(now, datetime) else (now or date.today())
        if today != self._day or self._day_start_equity is None:
            self._day = today
            self._day_start_equity = equity
            self._halted = False
            return
        if self._day_start_equity > 0:
            dd = (self._day_start_equity - equity) / self._day_start_equity
            if dd >= self.cfg.daily_loss_halt and not self._halted:
                self._halted = True
                log.warning(
                    f"[red]DAILY LOSS HALT[/red] "
                    f"equity {equity:.2f} dd {dd:.2%} >= {self.cfg.daily_loss_halt:.2%}"
                )

    @property
    def halted(self) -> bool:
        return self._halted

    # ------- pre-trade gate -------

    def can_open(
        self,
        signal: Signal,
        open_positions: list[OpenPosition],
        pricing: dict[str, dict],
    ) -> tuple[bool, str]:
        if self._halted:
            return False, "daily loss halt active"
        active = [p for p in open_positions if p.units != 0]
        if len(active) >= self.cfg.max_concurrent_positions:
            return False, f"max concurrent {self.cfg.max_concurrent_positions} reached"
        if self.cfg.one_position_per_instrument:
            if any(p.instrument == signal.instrument and p.units != 0 for p in open_positions):
                return False, f"already in {signal.instrument}"
        p = pricing.get(signal.instrument)
        if not p or not p.get("tradeable"):
            return False, "instrument not tradeable"
        return True, "ok"

    # ------- sizing -------

    def size(
        self,
        signal: Signal,
        equity: float,
        pricing: dict[str, dict],
    ) -> SizedOrder | None:
        if signal.stop_distance <= 0:
            return None
        p = pricing.get(signal.instrument)
        if not p:
            return None

        # Pick the conversion factor that matches the side direction.
        conv = p["qhcf_pos"] if signal.side == "buy" else p["qhcf_neg"]
        if conv <= 0:
            return None

        risk_amount = equity * self.cfg.risk_per_trade
        # P&L per 1 unit when price moves by stop_distance, in account currency.
        pnl_per_unit = signal.stop_distance * conv
        if pnl_per_unit <= 0:
            return None

        raw_units = risk_amount / pnl_per_unit
        units = int(math.floor(raw_units))
        if units <= 0:
            return None
        if signal.side == "sell":
            units = -units

        notional = abs(units) * signal.entry
        return SizedOrder(
            units=units,
            risk_amount=risk_amount,
            stop_distance=signal.stop_distance,
            notional=notional,
            reason=f"risk={risk_amount:.2f} stop_dist={signal.stop_distance:.5f} conv={conv:.5f}",
        )
