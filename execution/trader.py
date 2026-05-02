"""Execution layer.

Submits sized orders to the broker with attached stop-loss and take-profit.
Splits live and paper modes so the same code path drives both.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.broker import Broker
from core.logger import get_logger
from risk.manager import SizedOrder
from strategy.base import Signal

log = get_logger(__name__)


@dataclass
class Trade:
    instrument: str
    side: str
    units: int
    entry: float
    stop: float
    target: float
    raw_response: dict | None = None


class Trader:
    def __init__(self, broker: Broker, paper: bool = False):
        self.broker = broker
        self.paper = paper

    def execute(self, signal: Signal, sized: SizedOrder) -> Trade | None:
        if self.paper:
            log.info(
                f"[yellow]PAPER[/yellow] {signal.side.upper():4s} "
                f"{signal.instrument} units={sized.units} "
                f"entry={signal.entry:.5f} stop={signal.stop:.5f} "
                f"target={signal.target:.5f} :: {signal.reason}"
            )
            return Trade(
                instrument=signal.instrument,
                side=signal.side,
                units=sized.units,
                entry=signal.entry,
                stop=signal.stop,
                target=signal.target,
            )

        log.info(
            f"[green]LIVE[/green] {signal.side.upper():4s} "
            f"{signal.instrument} units={sized.units} "
            f"entry≈{signal.entry:.5f} stop={signal.stop:.5f} "
            f"target={signal.target:.5f}"
        )
        try:
            resp = self.broker.market_order(
                instrument=signal.instrument,
                units=sized.units,
                stop_loss=signal.stop,
                take_profit=signal.target,
                client_tag=f"bot1-{signal.instrument}",
            )
        except Exception as e:
            log.error(f"order failed for {signal.instrument}: {e}")
            return None
        return Trade(
            instrument=signal.instrument,
            side=signal.side,
            units=sized.units,
            entry=signal.entry,
            stop=signal.stop,
            target=signal.target,
            raw_response=resp,
        )
