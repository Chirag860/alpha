#!/usr/bin/env python3
"""Per-feature cross-sectional IC table — the decisive, model-free test for H1.

For each *genuinely cross-sectional* feature, compute the mean per-(date,minute) Spearman rank
IC against the vol-adjusted residual target, plus a t-stat across cross-sections. This is the
correct metric (D2 fix: per-cross-section, not pooled) and it drops the 10 within-group-constant
features that can carry no ranking information (D3 fix). No model, no CV — just correlations.

Read: a feature with mean |IC| > ~0.01 AND |t| > 3 is a real cross-sectional signal worth
pursuing. If nothing clears that bar, the bars-only feature set has no cross-sectional edge and
H1 is confirmed. (With ~70k cross-sections, |t| is large even for economically-worthless IC, so
the |IC| > 0.01 magnitude bar is what matters, not significance alone.)

    python3 diagnose_ic.py --config config/mt5.yaml
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import polars as pl
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

# Constant within a (date, minute) cross-section => zero contribution to within-group ranking.
DEAD_XS = {"sin_tod", "cos_tod", "mins_to_close", "mins_to_flatten", "opening_flag",
           "squareoff_flag", "expiry_flag", "index_ret", "dispersion", "vix_proxy"}


def xs_ic(panel: pl.DataFrame, feat: str, target: str = "y_voladj", min_names: int = 10):
    """Mean per-(date,minute) Spearman IC of ``feat`` vs ``target``, with a t-stat across groups."""
    df = panel.select(["date", "minute", feat, target]).drop_nulls()
    per = (df.group_by(["date", "minute"])
           .agg(ic=pl.corr(feat, target, method="spearman"), n=pl.len())
           .filter(pl.col("n") >= min_names))
    ic = per["ic"].to_numpy()
    ic = ic[np.isfinite(ic)]                 # drop null AND NaN (degenerate/zero-variance groups)
    if len(ic) < 5:
        return None
    t = float(ic.mean() / (ic.std(ddof=1) + 1e-12) * np.sqrt(len(ic)))
    return float(ic.mean()), t, len(ic)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/mt5.yaml")
    args = ap.parse_args()

    cfg = load_config(overrides=yaml.safe_load(pathlib.Path(args.config).read_text()) or {})
    market.set_active_profile_from_config(cfg)
    grid, meta = load_mt5_panel(str(cfg.mt5.data_dir))
    print(f"Panel: {grid.height:,} rows | {meta.height} names | {grid['date'].n_unique()} sessions")
    print("Building features + labels ...", flush=True)
    feats, cols = build_features_bars_only(grid, meta, cfg)
    labels = add_cross_sectional_targets(triple_barrier_labels(feats, cfg), cfg)

    live = [c for c in cols if c not in DEAD_XS]
    print(f"Dropped {len(cols) - len(live)} within-group-constant features; "
          f"{len(live)} genuinely cross-sectional features remain.\n", flush=True)

    rows = []
    for c in live:
        r = xs_ic(labels, c)
        if r is not None:
            rows.append((c, *r))
    rows.sort(key=lambda x: -abs(x[1]))

    print(f"{'feature':26s} {'mean IC':>9s} {'t-stat':>8s} {'n_xs':>8s}")
    print("-" * 54)
    for c, ic, t, n in rows:
        flag = "  <-- SIGNAL" if abs(ic) > 0.01 and abs(t) > 3 else ""
        print(f"{c:26s} {ic:>9.4f} {t:>8.1f} {n:>8d}{flag}")

    best = max(rows, key=lambda x: abs(x[1]))
    hit = abs(best[1]) > 0.01 and abs(best[2]) > 3
    print("\n" + "=" * 70)
    print(f"VERDICT: strongest feature = {best[0]}  |IC|={abs(best[1]):.4f}  t={best[2]:.1f}")
    if hit:
        print("=> Some cross-sectional signal exists. Worth Tier-2 (horizon/target ablations).")
    else:
        print("=> NO feature clears |IC|>0.01 with |t|>3.  H1 CONFIRMED: the bars-only feature")
        print("   set carries no usable cross-sectional edge on this US large-cap universe.")
    print("=" * 70)
    market.set_active_profile("bse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
