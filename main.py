"""Bot entry point.

Subcommands:
  live      — trade a real (practice) OANDA account
  paper     — same code path but trades are logged, not sent
  backtest  — replay history through the same engine + strategy + risk

Event loop is intentionally MOMENTUM-FIRST:

  1.  Pull candles for every instrument across every configured timeframe.
  2.  Have the MomentumEngine compute a regime for each instrument.
  3.  Filter to instruments with a non-neutral, aligned regime — these are
      the ones the bot "understands". Everything else is dropped before
      the strategy is ever consulted.
  4.  Only then ask the strategy for a signal on the survivors.
  5.  Risk manager sizes / gates / halts; trader executes.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from backtest.runner import load_history_from_broker, run_backtest
from core.broker import Broker
from core.logger import get_logger
from execution.trader import Trader
from momentum.engine import MomentumEngine
from momentum.regime import Regime
from risk.manager import RiskConfig, RiskManager
from strategy.momentum_breakout import MomentumBreakout

log = get_logger("bot1")
console = Console()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_components(cfg: dict, paper: bool) -> tuple[Broker, MomentumEngine, MomentumBreakout, RiskManager, Trader]:
    broker = Broker(
        account_id=cfg["broker"]["account_id"],
        environment=cfg["broker"]["environment"],
    )
    engine = MomentumEngine(
        timeframes=cfg["momentum"]["timeframes"],
        weights=cfg["momentum"]["weights"],
        thresholds=cfg["momentum"]["thresholds"],
    )
    sp = cfg["strategy"]["params"]
    strategy = MomentumBreakout(
        breakout_lookback=int(sp["breakout_lookback"]),
        confirm_timeframe=sp["confirm_timeframe"],
        atr_stop_mult=float(cfg["risk"]["atr_stop_mult"]),
        reward_risk=float(cfg["risk"]["reward_risk"]),
    )
    risk = RiskManager(RiskConfig(
        risk_per_trade=float(cfg["risk"]["risk_per_trade"]),
        max_concurrent_positions=int(cfg["risk"]["max_concurrent_positions"]),
        one_position_per_instrument=bool(cfg["risk"]["one_position_per_instrument"]),
        daily_loss_halt=float(cfg["risk"]["daily_loss_halt"]),
    ))
    trader = Trader(broker, paper=paper)
    return broker, engine, strategy, risk, trader


def render_states(states):
    table = Table(title="Momentum understanding", show_lines=False)
    table.add_column("Instrument")
    table.add_column("Regime")
    table.add_column("Score", justify="right")
    table.add_column("Strength", justify="right")
    table.add_column("Aligned")
    table.add_column("Per-TF")
    for s in states:
        per_tf = " ".join(f"{tf}:{v:+.2f}" for tf, v in s.per_tf_score.items())
        colour = {
            Regime.STRONG_UP: "bold green", Regime.UP: "green",
            Regime.NEUTRAL: "white",
            Regime.DOWN: "red", Regime.STRONG_DOWN: "bold red",
        }[s.regime]
        table.add_row(
            s.instrument, f"[{colour}]{s.regime.value}[/{colour}]",
            f"{s.score:+.2f}", f"{s.trend_strength:.2f}",
            "✓" if s.aligned else "·", per_tf,
        )
    console.print(table)


def run_live(cfg: dict, paper: bool):
    broker, engine, strategy, risk, trader = build_components(cfg, paper=paper)
    instruments = cfg["instruments"]
    tfs = cfg["momentum"]["timeframes"]
    poll = int(cfg["loop"]["poll_seconds"])

    log.info(
        f"[bold cyan]bot1 starting[/bold cyan] mode={'PAPER' if paper else 'LIVE'} "
        f"instruments={instruments} tfs={tfs} poll={poll}s"
    )

    while True:
        try:
            account = broker.account_summary()
            risk.update_equity(account.nav)
            if risk.halted:
                log.warning("halted; sleeping")
                time.sleep(poll)
                continue

            # 1) Pull candles for everything.
            data: dict[str, dict] = {}
            for inst in instruments:
                data[inst] = {tf: broker.candles(inst, tf, count=cfg["momentum"]["lookback"]) for tf in tfs}

            # 2) Compute momentum.
            states = [engine.compute(inst, data[inst]) for inst in instruments]
            render_states(states)

            # 3) Filter — momentum-first gating.
            tradeable_states = [
                s for s in states
                if s.regime != Regime.NEUTRAL and s.aligned
            ]
            if not tradeable_states:
                log.info("no instrument with clear momentum this cycle")
                time.sleep(poll)
                continue

            # 4) Strategy + 5) risk + execution.
            pricing = broker.pricing(instruments)
            open_pos = broker.open_positions()

            for s in tradeable_states:
                signal = strategy.evaluate(s.instrument, s, data[s.instrument])
                if signal is None:
                    continue
                ok, why = risk.can_open(signal, open_pos, pricing)
                if not ok:
                    log.info(f"skip {signal.instrument}: {why}")
                    continue
                sized = risk.size(signal, account.nav, pricing)
                if sized is None or sized.units == 0:
                    log.info(f"skip {signal.instrument}: sizing returned 0 units")
                    continue
                trader.execute(signal, sized)

            time.sleep(poll)
        except KeyboardInterrupt:
            log.info("interrupted, exiting")
            return
        except Exception as e:
            log.exception(f"loop error: {e}")
            time.sleep(poll)


def run_bt(cfg: dict, instrument: str | None):
    broker, engine, strategy, _, _ = build_components(cfg, paper=True)
    instruments = [instrument] if instrument else cfg["instruments"]
    tfs = cfg["momentum"]["timeframes"]
    confirm = cfg["strategy"]["params"]["confirm_timeframe"]

    risk_cfg = RiskConfig(
        risk_per_trade=float(cfg["risk"]["risk_per_trade"]),
        max_concurrent_positions=1,             # single-instrument BT
        one_position_per_instrument=True,
        daily_loss_halt=float(cfg["risk"]["daily_loss_halt"]),
    )

    summary_table = Table(title="Backtest summary")
    for col in ("Inst", "Trades", "Win%", "PF", "Return", "MaxDD", "FinalEq"):
        summary_table.add_column(col, justify="right")

    for inst in instruments:
        log.info(f"loading history for {inst} …")
        history = load_history_from_broker(broker, inst, tfs, bars_per_tf=1500)
        if history[confirm].empty:
            log.warning(f"no history for {inst}")
            continue
        # USD/JPY style pairs need ~1/price conversion; everything X/USD ≈ 1.0.
        qhcf = 1.0
        if inst.endswith("_JPY") and not inst.startswith("USD_"):
            qhcf = 1.0 / float(history[confirm]["close"].iloc[-1])
        elif inst.startswith("USD_"):
            qhcf = 1.0 / float(history[confirm]["close"].iloc[-1])

        result = run_backtest(
            instrument=inst,
            candles_by_tf=history,
            confirm_tf=confirm,
            engine=engine,
            strategy=strategy,
            risk_cfg=risk_cfg,
            starting_equity=10_000.0,
            qhcf=qhcf,
        )
        st = result.stats()
        log.info(f"{inst} -> {st}")
        if st.get("trades", 0):
            summary_table.add_row(
                inst,
                str(st["trades"]),
                f"{st['win_rate']*100:.1f}",
                f"{st['profit_factor']:.2f}",
                f"{st['total_return']*100:.1f}%",
                f"{st['max_drawdown']*100:.1f}%",
                f"{st['final_equity']:.0f}",
            )
        else:
            summary_table.add_row(inst, "0", "-", "-", "-", "-", "-")

    console.print(summary_table)


def main():
    p = argparse.ArgumentParser(prog="bot1")
    p.add_argument("--config", default="config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("live", help="trade for real (uses configured environment)")
    sub.add_parser("paper", help="run the loop but log trades instead of sending")
    bt = sub.add_parser("backtest", help="replay history")
    bt.add_argument("--instrument", default=None)

    args = p.parse_args()
    load_dotenv(Path(args.config).parent / ".env")
    if not os.environ.get("OANDA_API_TOKEN"):
        log.error("OANDA_API_TOKEN missing — copy .env.example to .env and fill it in")
        sys.exit(1)

    cfg = load_config(args.config)
    if args.cmd == "live":
        run_live(cfg, paper=False)
    elif args.cmd == "paper":
        run_live(cfg, paper=True)
    elif args.cmd == "backtest":
        run_bt(cfg, args.instrument)


if __name__ == "__main__":
    main()
