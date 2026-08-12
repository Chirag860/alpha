#!/usr/bin/env python3
"""Out-of-sample validation of the strategy on an exported MT5 panel (run on the Mac).

Runs the honest validation harness on real MT5 stock-CFD data -- the numbers that tell you
whether the ranking signal has a *durable* edge, as opposed to the in-sample training score:

* purged / combinatorial-purged CV (no future leakage across the forward-looking labels),
* cross-sectional IC + meta-model OOS AUC,
* CPCV Sharpe distribution (median / 5th percentile) + PBO (overfitting probability),
* Deflated Sharpe (discounts for trials / sample length / return shape),
* effective breadth (how many *independent* bets you really have),
* and the out-of-fold **net** Sharpe after the US-CFD cost stack -- the bottom line.

    python3 validate_mt5.py --config config/mt5.yaml            # full run (with CPCV; slow)
    python3 validate_mt5.py --config config/mt5.yaml --fast     # skip the 66-fit CPCV sweep

Heavy compute -- run it where polars/LightGBM are stable (the Mac), not the emulated VM.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import yaml

from bsealpha import market
from bsealpha.config import load_config
from bsealpha.data import load_mt5_panel
from bsealpha.features import build_features_bars_only
from bsealpha.labeling import (
    add_cross_sectional_targets,
    compute_weights,
    triple_barrier_labels,
)
from bsealpha.validation import evaluate, oof_predicted_panel, perfect_foresight_ceiling


def _f(x, nd=3):
    return f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/mt5.yaml")
    ap.add_argument("--fast", action="store_true", help="skip the CPCV sweep (no PBO/dist)")
    ap.add_argument("--no-backtest", action="store_true", help="skip the net-cost backtest")
    args = ap.parse_args()

    overlay = yaml.safe_load(pathlib.Path(args.config).read_text()) or {}
    # keep the CV tractable on millions of rows: fewer boosting rounds than the deploy model.
    overlay.setdefault("model", {})
    for k in ("lambdarank", "regression", "meta"):
        overlay["model"].setdefault(k, {})
        overlay["model"][k].setdefault("n_estimators", 80)
        overlay["model"][k].setdefault("min_data_in_leaf", 200)
    cfg = load_config(overrides=overlay)
    market.set_active_profile_from_config(cfg)

    grid, meta = load_mt5_panel(str(cfg.mt5.data_dir))
    print(f"Panel: {grid.height:,} bars | {meta.height} names | "
          f"{grid['date'].n_unique()} sessions (profile={market.active_profile().name})")
    print("Building features + labels ...", flush=True)
    feats, cols = build_features_bars_only(grid, meta, cfg)
    labels = compute_weights(add_cross_sectional_targets(
        triple_barrier_labels(feats, cfg), cfg), cfg)

    ceiling = perfect_foresight_ceiling(labels, cfg)
    print(f"Running validation (CPCV={'off' if args.fast else 'on'}) ...", flush=True)
    rep = evaluate(labels, cols, cfg, run_cpcv=not args.fast)

    bt = None
    if not args.no_backtest:
        print("Running out-of-fold net-cost backtest ...", flush=True)
        betas = {int(r["scrip_code"]): float(r["beta"])
                 for r in meta.select(["scrip_code", "beta"]).iter_rows(named=True)}
        pred = oof_predicted_panel(labels, cols, cfg)
        from bsealpha.backtest import run_backtest
        bt = run_backtest(pred, cfg, betas=betas)

    line = "-" * 72
    print("\n" + line)
    print("MT5 US STOCK-CFD  —  OUT-OF-SAMPLE VALIDATION")
    print(line)
    print(f"Labeled rows (name x minute)        : {labels.height:,}")
    print(f"Features                            : {len(cols)}")
    print()
    print("PERFECT-FORESIGHT CEILING  (is the label design even tradable?)")
    print(f"  Gross / Net bps per trade         : {_f(ceiling.gross_bps_per_trade,2)} / "
          f"{_f(ceiling.net_bps_per_trade,2)}")
    print(f"  Net ceiling Sharpe                : {_f(ceiling.net_sharpe,2)}  "
          f"-> {'PASS' if ceiling.passes_gate else 'FAIL — labels/costs leave no room'}")
    print()
    print("VALIDATION  (out-of-fold — the honest numbers)")
    print(f"  Cross-sectional IC (primary)      : {_f(rep.ic)}     [real 0.01–0.04; >0.08 ⇒ leak]")
    print(f"  Meta-model OOS AUC                : {_f(rep.meta_auc)}     [real 0.52–0.58; >0.62 ⇒ leak]")
    print(f"  Effective breadth                 : {_f(rep.effective_breadth,2)}     "
          f"[gate ≥ {cfg.validation.breadth_gate}; ≈1 ⇒ index-timing, not stock-picking]")
    print(f"  OOF book Sharpe (gross)           : {_f(rep.sharpe_oos,2)}")
    print(f"  CPCV Sharpe median / 5th-pct      : {_f(rep.cpcv_sharpe_median,2)} / "
          f"{_f(rep.cpcv_sharpe_5pct,2)}" + ("     [--fast: skipped]" if args.fast else ""))
    print(f"  Deflated Sharpe (DSR)             : {_f(rep.dsr)}  (SR {_f(rep.dsr_sr,2)}, "
          f"N={rep.n_trials})  [gate > 0.95]")
    print(f"  PBO (prob. backtest overfitting)  : {_f(rep.pbo)}     [gate < 0.2]")
    if bt is not None:
        print()
        print("OUT-OF-FOLD BACKTEST  (US-CFD costs; the bottom line)")
        print(f"  Gross / Net Sharpe                : {_f(bt.gross_sharpe,2)} / {_f(bt.sharpe,2)}")
        print(f"  Sortino / MaxDD                   : {_f(bt.sortino,2)} / {_f(bt.max_drawdown,4)}")
        print(f"  Hit rate / n_trades               : {_f(bt.hit_rate,3)} / {bt.n_trades}")
        print(f"  Daily turnover (x gross)          : {_f(bt.turnover_x,2)}")
        print(f"  Fees $ / Impact $                 : {bt.total_cost_rupees:,.0f} / "
              f"{bt.total_impact_rupees:,.0f}")
        print(f"  Markouts (bps) 1/5/15/30 min      : "
              + " / ".join(_f(bt.markout_bps.get(h, 0.0), 2) for h in (1, 5, 15, 30)))
    print()
    notes = list(rep.tripwires)
    if bt is not None and bt.sharpe > cfg.validation.tripwire_sharpe:
        notes.append(f"NET Sharpe {bt.sharpe:.1f} > {cfg.validation.tripwire_sharpe} — too good; suspect a leak.")
    print("TRIPWIRES / NOTES:" if notes else "No tripwires fired.")
    for t in notes:
        print(f"  ! {t}")
    print(line)
    print("How to read it: DSR > 0.95 AND PBO < 0.2 AND breadth ≥ gate AND positive NET Sharpe")
    print("means a plausibly real edge. A strong in-sample story that deflates to ~0 here is the")
    print("expected, honest outcome for most first attempts — that's the harness doing its job.")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
