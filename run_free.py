#!/usr/bin/env python3
"""Run the pipeline on FREE real BSE data (yfinance) -- a zero-cost smoke test.

Downloads ~1 week of free 1-minute BSE bars, builds the **reduced** (bars-only) feature set
(no depth => no OFI/micro-price/book/flow), trains the model, and runs the backtest + paper
session. This proves the pipeline ingests and trades *real* BSE data end to end for free.

It is NOT research: ~1 week of history and no order-book depth. Read every caveat printed.
For real research you need a paid depth vendor (TrueData/GDFL) and years of history.

    python run_free.py                 # default liquid universe, last 5 sessions
    python run_free.py --period 7d --paper
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

from bsealpha.backtest import run_backtest
from bsealpha.config import load_config
from bsealpha.data import load_yfinance_panel
from bsealpha.features import build_features_bars_only
from bsealpha.labeling import (
    add_cross_sectional_targets,
    compute_weights,
    triple_barrier_labels,
)
from bsealpha.models import PooledEnsemble


def _f(x, n=2):
    return f"{x:.{n}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--period", default="5d", help="yfinance period (max ~7d for 1m data)")
    ap.add_argument("--interval", default="1m",
                    help="bar interval: 1m (<=7d), 5m/15m/60m (<=60d). Coarser = degraded.")
    ap.add_argument("--paper", action="store_true", help="Also run the paper session + reconcile")
    ap.add_argument("--quiet-warnings", action="store_true")
    args = ap.parse_args()
    if args.quiet_warnings:
        warnings.simplefilter("ignore")

    # reduced model config: fewer estimators (tiny data), demean-sector residualization
    cfg = load_config(overrides={
        "residualize": {"method": "demean_sector"},
        "model": {"lambdarank": {"min_data_in_leaf": 100, "n_estimators": 60},
                  "regression": {"min_data_in_leaf": 100, "n_estimators": 60},
                  "meta": {"min_data_in_leaf": 50, "n_estimators": 60}},
    })

    print(f"Downloading FREE real BSE bars via yfinance ({args.period} @ {args.interval}) ...")
    grid, meta = load_yfinance_panel(period=args.period, interval=args.interval)
    n_days = grid["date"].n_unique()
    n_names = grid["scrip_code"].n_unique()
    print(f"  loaded {grid.height:,} bars  |  {n_names} names  |  {n_days} sessions\n")

    feats, cols = build_features_bars_only(grid, meta, cfg)
    labels = compute_weights(add_cross_sectional_targets(
        triple_barrier_labels(feats, cfg), cfg), cfg)

    ens = PooledEnsemble(cfg, cols).fit(labels)
    pred = ens.predict(labels)                      # IN-SAMPLE (1 week can't support OOF)
    betas = {int(r["scrip_code"]): float(r["beta"])
             for r in meta.select(["scrip_code", "beta"]).iter_rows(named=True)}
    bt = run_backtest(pred, cfg, betas=betas)

    ic = float(np.corrcoef(pred["primary_score"].to_numpy(),
                           labels["y_voladj"].to_numpy())[0, 1])
    line = "-" * 70
    print(line)
    print(f"FREE REAL-DATA SMOKE TEST  (bars-only @ {args.interval}, "
          f"{n_names} names x {n_days} sessions, NOT research)")
    print(line)
    print(f"  Reduced features (no depth)        : {len(cols)}")
    print(f"  In-sample IC (primary)             : {_f(ic, 3)}")
    print(f"  Backtest gross / net Sharpe        : {_f(bt.gross_sharpe)} / {_f(bt.sharpe)}")
    print(f"  Turnover / n_trades                : {_f(bt.turnover_x, 1)}x / {bt.n_trades}")
    print(f"  Fees Rs / Impact Rs                : {bt.total_cost_rupees:,.0f} / "
          f"{bt.total_impact_rupees:,.0f}")
    print(f"  Effective breadth                  : {_f(bt.effective_breadth, 2)}")
    print(f"  Markouts 1/5/15/30 min (bps)       : "
          + " / ".join(_f(bt.markout_bps.get(h, 0.0)) for h in (1, 5, 15, 30)))

    if args.paper:
        from bsealpha.live import reconcile, run_paper_session
        ps = run_paper_session(pred, cfg, betas=betas)
        rec = reconcile(bt, ps)
        print()
        print("  PAPER (live-stack replay) net Sharpe:", _f(ps.net_sharpe))
        print("  Backtest->paper gap / haircut       :", _f(rec.sharpe_gap), "/", _f(rec.haircut))
        print("  Maker fill ratio                    :", _f(rec.fill_ratio))
    print(line)
    print("CAVEATS (read these):")
    print(f"  * NO order-book depth -> OFI / micro-price / book / signed-flow features are")
    print("    absent; this is a weak reduced model, not the microstructure strategy.")
    print(f"  * {args.interval} bars, {n_days} sessions, predictions IN-SAMPLE -> a mechanics")
    print("    check only. A trustworthy Sharpe needs YEARS of data (S5.1), and this cadence")
    print("    is coarser than the 1-min design (label/horizon semantics degrade).")
    print("  * For real research: paid depth vendor (TrueData/GDFL) + 3-5 years of data.")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
