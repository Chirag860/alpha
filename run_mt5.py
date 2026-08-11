#!/usr/bin/env python3
"""Live (demo) trading loop against an MT5 stock-CFD account -- **Windows/VM only**.

Phase 1 mechanics: run the SAME cross-sectional engine live against an MT5 demo account.
Each completed minute it polls trailing M1 bars for the discovered universe, builds the
bars-only feature set, predicts, constructs the neutral book, and routes through
``ExecutionManager`` -> ``MT5BrokerAdapter`` (deploy.mode = demo, no real capital).

Prereqs (run on the VM, in order):
  1. python -m bsealpha.data.mt5_export --config config/mt5.yaml --discover-only   # check breadth
  2. python -m bsealpha.data.mt5_export --config config/mt5.yaml                    # export history
  3. python run_mt5.py --config config/mt5.yaml                                     # train + trade

The model here is trained IN-SAMPLE on the exported history at startup (a mechanics check,
like run_free.py). Proper OOF training + a serialized model artifact is Phase 2.

    python run_mt5.py --config config/mt5.yaml --dry-run   # build orders, don't send
"""

from __future__ import annotations

import os

# Cap polars parallelism BEFORE importing polars -- reduces memory and improves stability
# when running x64-emulated (e.g. an Apple-Silicon Windows VM). Override by pre-setting it.
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import polars as pl

from bsealpha import market
from bsealpha.config import load_config


def _load_cfg(config_path: str, *, dry_run: bool):
    import yaml
    overlay = yaml.safe_load(Path(config_path).read_text()) or {}
    if dry_run:
        overlay.setdefault("deploy", {})["mode"] = "dry_run"
    cfg = load_config(overrides=overlay)
    market.set_active_profile_from_config(cfg)
    return cfg


def _train(cfg):
    """Train the pooled ensemble in-sample on the exported panel; return (model, betas, cols)."""
    from bsealpha.data import load_mt5_panel
    from bsealpha.features import build_features_bars_only
    from bsealpha.labeling import (
        add_cross_sectional_targets, compute_weights, triple_barrier_labels)
    from bsealpha.models import PooledEnsemble

    grid, meta = load_mt5_panel(str(cfg.mt5.data_dir))
    print(f"Training on {grid.height:,} bars | {meta.height} names | "
          f"{grid['date'].n_unique()} sessions (in-sample, Phase 1)")
    feats, cols = build_features_bars_only(grid, meta, cfg)
    labels = compute_weights(add_cross_sectional_targets(
        triple_barrier_labels(feats, cfg), cfg), cfg)
    model = PooledEnsemble(cfg, cols).fit(labels)
    betas = {int(r["scrip_code"]): float(r["beta"])
             for r in meta.select(["scrip_code", "beta"]).iter_rows(named=True)}
    return model, betas, meta


def live_loop(cfg, mt5, adapter, model, betas, meta) -> None:  # pragma: no cover - VM only
    """Poll M1 bars each minute, predict, build the book, and route to the demo account."""
    from collections import deque

    from bsealpha.data.mt5_export import rates_to_grid
    from bsealpha.execution import ExecutionManager
    from bsealpha.features import build_features_bars_only
    from bsealpha.portfolio import build_book

    mgr = ExecutionManager(adapter, cfg)
    symbol_map = adapter.symbol_map
    tf = getattr(mt5, "TIMEFRAME_" + str(getattr(cfg.mt5, "timeframe", "M1")))
    n_tail = 120                                     # trailing minutes for feature windows
    sector_of = {int(r["scrip_code"]): r["sector"]
                 for r in meta.select(["scrip_code", "sector"]).iter_rows(named=True)}
    last_minute = None

    print("Live demo loop started. Ctrl-C to stop.")
    while True:
        # assemble the trailing grid across all names from fresh terminal bars
        frames = []
        for code, sym in symbol_map.items():
            rates = mt5.copy_rates_from_pos(sym, tf, 0, n_tail)
            g = rates_to_grid(rates, code, cfg)
            if g is not None:
                frames.append(g)
        if not frames:
            time.sleep(5.0)
            continue
        # rates_to_grid emits `date` as a string; cast to Date (as the batch paths do) so
        # date-based features (e.g. expiry_flag's weekday) work.
        grid = (pl.concat(frames, how="vertical_relaxed")
                .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
                .sort(["date", "minute", "scrip_code"]))
        cur = int(grid["minute"].max())
        if cur == last_minute:                       # no new completed minute yet
            time.sleep(2.0)
            continue
        last_minute = cur

        feats, cols = build_features_bars_only(grid, meta, cfg)
        latest = feats.filter(pl.col("minute") == cur)
        if latest.height == 0:
            continue
        pred = model.predict(latest)
        beta = np.array([betas.get(int(s), 1.0) for s in latest["scrip_code"]])
        sect = np.array([sector_of.get(int(s), "UNKNOWN") for s in latest["scrip_code"]])
        targets_arr = build_book(
            pred["primary_score"].to_numpy(), beta, sect, latest["turnover"].to_numpy(),
            gross_target=float(cfg.portfolio.gross_target),
            max_participation=float(cfg.portfolio.max_participation),
            min_clip=float(cfg.portfolio.min_clip),
            max_names=int(cfg.portfolio.max_names),
            sector_cap=float(cfg.portfolio.sector_cap),
            p_act=pred["p_act"].to_numpy())
        scrips = latest["scrip_code"].to_numpy()
        targets = {int(scrips[i]): float(targets_arr[i]) for i in range(len(scrips))}
        # market dict from the latest mid (adapter ignores it, but the manager needs mids)
        mkt = {int(r["scrip_code"]): (r["mid"], r["mid"], r["mid"])
               for r in latest.select(["scrip_code", "mid"]).iter_rows(named=True)}
        mod = market.session_open_min() + cur
        res = mgr.step(mod, time.time(), targets, mkt)
        print(f"[min {cur:3d}] phase={res.phase} orders={len(res.orders)} "
              f"rejects={adapter.reject_rate():.2%} halted={res.halted}")
        if res.halted:
            print("  RISK HALT:", "; ".join(res.reasons))
            break


def main() -> int:  # pragma: no cover - VM only
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/mt5.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute + log orders but never send (deploy.mode=dry_run)")
    ap.add_argument("--model", default=None,
                    help="Load a pre-trained artifact from train_model.py instead of training "
                         "here. Use this on the VM so it never does heavy training.")
    args = ap.parse_args()

    cfg = _load_cfg(args.config, dry_run=args.dry_run)
    from bsealpha.execution import MT5BrokerAdapter, connect_mt5

    if args.model:
        with open(args.model, "rb") as fh:
            art = pickle.load(fh)
        model = art["ensemble"]
        betas = {int(k): float(v) for k, v in art["betas"].items()}
        meta = pl.DataFrame(art["meta_rows"])
        print(f"Loaded model artifact {args.model} | {meta.height} names "
              f"(profile={art.get('profile')})")
    else:
        model, betas, meta = _train(cfg)
    mt5 = connect_mt5(cfg)
    try:
        symbol_map = {int(k): v for k, v in
                      json.loads((Path(str(cfg.mt5.data_dir)) / "symbol_map.json").read_text()).items()}
        adapter = MT5BrokerAdapter(mt5, symbol_map,
                                   magic=abs(hash(str(cfg.execution.algo_id))) % 2_000_000)
        live_loop(cfg, mt5, adapter, model, betas, meta)
    finally:
        mt5.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
