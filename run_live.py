#!/usr/bin/env python3
"""Shadow paper-trading runner (§11.2 wk6).

Two modes:

* **replay shadow** (runnable now): replays a dataset through the *live stack*
  (``ExecutionManager`` -> ``PaperBroker`` with passive fills + Indian costs) and reports the
  intended orders, simulated fills, and markouts. Uses free real BSE data by default.
* **live shadow** (you wire the broker): :func:`live_loop` is the real-time skeleton --
  drive a per-name ``ScripState`` from your broker websocket, assemble the cross-section each
  completed minute, predict, build the book, and route to a ``PaperBroker`` fed by the live
  feed. No capital at risk; real signals. Fill in :func:`connect_broker_feed` with your
  broker SDK (Fyers / Angel One SmartAPI / Dhan) and it goes live.

    python run_live.py                 # replay shadow on free real BSE data
    python run_live.py --synthetic     # replay shadow on the synthetic panel
"""

from __future__ import annotations

import argparse
import warnings

from bsealpha.config import Config, load_config


# ----------------------------------------------------------------------- replay
def replay_shadow(cfg: Config, *, use_synthetic: bool, period: str) -> None:
    """Run the live stack over a replayed dataset and print the shadow session report."""
    import numpy as np

    from bsealpha.backtest import run_backtest
    from bsealpha.features import build_features, build_features_bars_only
    from bsealpha.labeling import (
        add_cross_sectional_targets,
        compute_weights,
        triple_barrier_labels,
    )
    from bsealpha.live import reconcile, run_paper_session
    from bsealpha.models import PooledEnsemble

    if use_synthetic:
        from bsealpha.data import generate_panel
        panel = generate_panel(cfg)
        feats, cols = build_features(panel.depth, panel.trades, panel.meta, cfg)
        meta = panel.meta
    else:
        from bsealpha.data import load_yfinance_panel
        print("Downloading FREE real BSE 1-minute bars via yfinance ...")
        grid, meta = load_yfinance_panel(period=period)
        print(f"  {grid.height:,} bars | {grid['scrip_code'].n_unique()} names | "
              f"{grid['date'].n_unique()} sessions (bars-only, no depth)\n")
        feats, cols = build_features_bars_only(grid, meta, cfg)

    labels = compute_weights(add_cross_sectional_targets(
        triple_barrier_labels(feats, cfg), cfg), cfg)
    ens = PooledEnsemble(cfg, cols).fit(labels)
    pred = ens.predict(labels)
    betas = {int(r["scrip_code"]): float(r["beta"])
             for r in meta.select(["scrip_code", "beta"]).iter_rows(named=True)}

    paper = run_paper_session(pred, cfg, betas=betas)
    bt = run_backtest(pred, cfg, betas=betas)
    rec = reconcile(bt, paper)

    line = "-" * 70
    print(line)
    print("SHADOW PAPER SESSION  (live stack, replayed; no capital at risk)")
    print(line)
    print(f"  Orders sent / fills                : {paper.n_orders} / {paper.n_fills}")
    print(f"  Maker fill ratio / taker fraction  : {paper.fill_ratio:.2f} / "
          f"{paper.taker_fraction:.2f}")
    print(f"  Paper net Sharpe                   : {paper.net_sharpe:+.2f}")
    print(f"  Turnover (x gross book)            : {paper.turnover_x:.1f}x")
    print(f"  Fees Rs / Impact Rs                : {paper.total_cost_rupees:,.0f} / "
          f"{paper.total_impact_rupees:,.0f}")
    print(f"  Markouts 1/5/15/30 min (bps)       : "
          + " / ".join(f"{paper.markout_bps.get(h, 0.0):.2f}" for h in (1, 5, 15, 30)))
    print()
    print("  vs backtest (the gap is what matters, §11.2):")
    print(f"    backtest net {rec.backtest_net_sharpe:+.2f}  ->  paper net "
          f"{rec.paper_net_sharpe:+.2f}   (gap {rec.sharpe_gap:+.2f})")
    print(line)
    print("This is a REPLAY. For a true live shadow, wire connect_broker_feed() and run")
    print("live_loop() -- the same code against your broker's websocket (no capital risk).")
    print(line)


# ------------------------------------------------------------------- live (wire)
def connect_broker_feed(cfg: Config):  # pragma: no cover - you implement this
    """Return an iterator of canonical events from your broker websocket.

    Implement against Fyers / Angel One SmartAPI / Dhan: subscribe to 5-level depth + trades
    for the screened universe, stamp each message with a LOCAL-RECEIPT ``ts_ns``, assert
    sequence continuity (gap -> flatten), and yield dicts like::

        {"kind": "book",  "scrip_code": int, "ts_ns": int, "minute": int,
         "bid_px": [...5], "bid_qty": [...5], "ask_px": [...5], "ask_qty": [...5]}
        {"kind": "trade", "scrip_code": int, "ts_ns": int, "minute": int,
         "price": float, "qty": float}

    Record every event to Parquet from day one (your own capture is irreplaceable, §8.4).
    """
    raise NotImplementedError(
        "Wire your broker websocket here (Fyers/Angel One SmartAPI/Dhan). "
        "See the docstring for the canonical event shape.")


def live_loop(cfg: Config, model, meta, betas) -> None:  # pragma: no cover - live only
    """Real-time shadow loop skeleton (§7.2, §8.3). Runs the SAME engine live as offline.

    Per event: update the per-name ``ScripState``. On each completed minute: collect the raw
    rows for all names, append to a trailing buffer, run the SHARED ``finalize_features`` on
    the buffer, predict on the last completed minute (t-1, so the cross-section is whole,
    §3.3), build the neutral book, and route to a ``PaperBroker`` fed by the live top-of-book.
    Swap ``PaperBroker`` for your ``BrokerAdapter`` to go from shadow to tiny-real.
    """
    from collections import deque

    import polars as pl

    from bsealpha.execution import ExecutionManager, PaperBroker
    from bsealpha.features import build_features_bars_only  # or full engine w/ depth
    from bsealpha.features.streaming import ScripState
    from bsealpha.market import SESSION_OPEN_MIN
    from bsealpha.portfolio import build_book

    broker = PaperBroker(maker_fill_prob=float(cfg.execution.maker_fill_prob))
    mgr = ExecutionManager(broker, cfg)
    states: dict[int, ScripState] = {}
    buffer: deque = deque(maxlen=90)          # trailing minutes of raw per-name rows
    cur_minute = None

    for ev in connect_broker_feed(cfg):
        sc = ev["scrip_code"]
        st = states.setdefault(sc, ScripState(cfg))
        if cur_minute is not None and ev["minute"] != cur_minute:
            # minute rolled over: close every name's minute, assemble the cross-section
            rows = []
            for s_code, s in states.items():
                row = s.close_minute()
                if row is not None:
                    rows.append({"scrip_code": s_code, "minute": cur_minute, **row})
                s.start_minute()
            if rows:
                buffer.append(pl.DataFrame(rows))
                grid = pl.concat(list(buffer), how="diagonal_relaxed")
                feats, cols = build_features_bars_only(grid, meta, cfg)   # SHARED code
                latest = feats.filter(pl.col("minute") == cur_minute)
                pred = model.predict(latest)
                market = {int(r["scrip_code"]): (r["mid"], r["mid"], r["mid"])
                          for r in latest.iter_rows(named=True)}
                targets = build_book(
                    pred["primary_score"].to_numpy(),
                    latest["scrip_code"].map_elements(lambda c: betas.get(int(c), 1.0),
                                                      return_dtype=pl.Float64).to_numpy(),
                    latest["sector"].to_numpy(),
                    latest["turnover"].to_numpy(), gross_target=float(cfg.portfolio.gross_target),
                    max_participation=float(cfg.portfolio.max_participation),
                    min_clip=float(cfg.portfolio.min_clip),
                    max_names=int(cfg.portfolio.max_names),
                    sector_cap=float(cfg.portfolio.sector_cap),
                    p_act=pred["p_act"].to_numpy())
                mod = SESSION_OPEN_MIN + cur_minute
                mgr.step(mod, mod * 60.0, {int(latest["scrip_code"][i]): float(targets[i])
                                           for i in range(len(targets))}, market,
                         feed_age_s=0.0)
        cur_minute = ev["minute"]
        if ev["kind"] == "book":
            st.on_book(ev["bid_px"], ev["bid_qty"], ev["ask_px"], ev["ask_qty"])
        else:
            st.on_trade(ev["price"], ev["qty"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="Replay the synthetic panel")
    ap.add_argument("--period", default="5d", help="yfinance period for free real data")
    ap.add_argument("--quiet-warnings", action="store_true")
    args = ap.parse_args()
    if args.quiet_warnings:
        warnings.simplefilter("ignore")

    overrides = {"residualize": {"method": "demean_sector"},
                 "model": {k: {"n_estimators": 60} for k in ("lambdarank", "regression", "meta")}}
    cfg = load_config(overrides=overrides)
    replay_shadow(cfg, use_synthetic=args.synthetic, period=args.period)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
