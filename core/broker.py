"""OANDA v20 REST wrapper.

Thin layer over oandapyV20 that returns clean pandas/dict structures
the rest of the bot can consume without touching the SDK directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from oandapyV20 import API
from oandapyV20.endpoints import accounts, instruments, orders, positions, pricing, trades
from oandapyV20.exceptions import V20Error

from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class AccountSummary:
    id: str
    balance: float
    nav: float
    unrealized_pl: float
    margin_available: float
    currency: str
    open_position_count: int


@dataclass
class OpenPosition:
    instrument: str
    long_units: float
    short_units: float
    unrealized_pl: float

    @property
    def units(self) -> float:
        return self.long_units + self.short_units  # short is negative

    @property
    def side(self) -> str:
        if self.long_units > 0:
            return "long"
        if self.short_units < 0:
            return "short"
        return "flat"


class Broker:
    GRANULARITY_MAP = {
        "M1": "M1", "M5": "M5", "M15": "M15", "M30": "M30",
        "H1": "H1", "H4": "H4", "D": "D", "W": "W",
    }

    def __init__(self, account_id: str, environment: str = "practice", token: str | None = None):
        token = token or os.environ.get("OANDA_API_TOKEN")
        if not token:
            raise RuntimeError("OANDA_API_TOKEN not set")
        if environment not in ("practice", "live"):
            raise ValueError("environment must be 'practice' or 'live'")
        self.account_id = account_id
        self.environment = environment
        self.api = API(access_token=token, environment=environment)

    # ---------- Account / positions ----------

    def account_summary(self) -> AccountSummary:
        r = accounts.AccountSummary(self.account_id)
        self.api.request(r)
        a = r.response["account"]
        return AccountSummary(
            id=a["id"],
            balance=float(a["balance"]),
            nav=float(a["NAV"]),
            unrealized_pl=float(a["unrealizedPL"]),
            margin_available=float(a["marginAvailable"]),
            currency=a["currency"],
            open_position_count=int(a["openPositionCount"]),
        )

    def open_positions(self) -> list[OpenPosition]:
        r = positions.OpenPositions(self.account_id)
        self.api.request(r)
        out = []
        for p in r.response.get("positions", []):
            out.append(OpenPosition(
                instrument=p["instrument"],
                long_units=float(p["long"]["units"]),
                short_units=float(p["short"]["units"]),
                unrealized_pl=float(p["unrealizedPL"]),
            ))
        return out

    # ---------- Market data ----------

    def candles(self, instrument: str, granularity: str, count: int = 250) -> pd.DataFrame:
        gran = self.GRANULARITY_MAP.get(granularity, granularity)
        params = {"granularity": gran, "count": count, "price": "M"}
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        self.api.request(r)
        rows = []
        for c in r.response.get("candles", []):
            if not c.get("complete"):
                continue
            rows.append({
                "time": c["time"],
                "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
                "volume": int(c.get("volume", 0)),
            })
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        return df

    def pricing(self, instruments_: Iterable[str]) -> dict[str, dict]:
        params = {"instruments": ",".join(instruments_)}
        r = pricing.PricingInfo(accountID=self.account_id, params=params)
        self.api.request(r)
        out = {}
        for p in r.response.get("prices", []):
            inst = p["instrument"]
            bid = float(p["bids"][0]["price"]) if p.get("bids") else None
            ask = float(p["asks"][0]["price"]) if p.get("asks") else None
            mid = (bid + ask) / 2 if (bid and ask) else None
            qhcf = p.get("quoteHomeConversionFactors", {})
            out[inst] = {
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "tradeable": p.get("tradeable", False),
                "qhcf_pos": float(qhcf.get("positiveUnits", 1.0)),
                "qhcf_neg": float(qhcf.get("negativeUnits", 1.0)),
            }
        return out

    # ---------- Orders / trades ----------

    def market_order(
        self,
        instrument: str,
        units: int,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        client_tag: str | None = None,
    ) -> dict:
        order: dict = {
            "order": {
                "instrument": instrument,
                "units": str(units),
                "type": "MARKET",
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        }
        if stop_loss is not None:
            order["order"]["stopLossOnFill"] = {"price": f"{stop_loss:.5f}"}
        if take_profit is not None:
            order["order"]["takeProfitOnFill"] = {"price": f"{take_profit:.5f}"}
        if client_tag:
            order["order"]["clientExtensions"] = {"id": client_tag, "tag": client_tag}
        r = orders.OrderCreate(self.account_id, data=order)
        try:
            self.api.request(r)
        except V20Error as e:
            log.error(f"Order rejected: {e}")
            raise
        return r.response

    def close_position(self, instrument: str) -> dict:
        # Close both sides; OANDA ignores sides with zero units.
        data = {"longUnits": "ALL", "shortUnits": "ALL"}
        r = positions.PositionClose(self.account_id, instrument=instrument, data=data)
        try:
            self.api.request(r)
        except V20Error as e:
            log.warning(f"Close position {instrument}: {e}")
            return {}
        return r.response

    def open_trades(self) -> list[dict]:
        r = trades.OpenTrades(self.account_id)
        self.api.request(r)
        return r.response.get("trades", [])
