#!/usr/bin/env python3
"""Train the pooled ensemble on an exported MT5 panel and save a portable model artifact.

Run this where heavy compute is stable (e.g. your Mac, natively). Then copy the artifact to
the VM and run the live loop against it:

    # 1. on the VM: export history (writes data/mt5/{grid,meta}.parquet, symbol_map.json)
    python -m bsealpha.data.mt5_export --config config/mt5.yaml
    # 2. copy data/mt5/grid.parquet + meta.parquet to the Mac's data/mt5/, then on the MAC:
    python3 train_model.py --config config/mt5.yaml --out data/mt5/model.pkl
    # 3. copy data/mt5/model.pkl back to the VM, then on the VM (market hours):
    python run_mt5.py --config config/mt5.yaml --model data/mt5/model.pkl

The artifact is a pickle of {ensemble, cols, betas, meta_rows, profile}. It contains only
numpy + LightGBM/sklearn models (no polars frames), so it loads cleanly on the other machine.
"""

from __future__ import annotations

import argparse
import pickle
import pathlib

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
from bsealpha.models import PooledEnsemble


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/mt5.yaml", help="MT5 overlay config path")
    ap.add_argument("--out", default="data/mt5/model.pkl", help="output artifact path")
    args = ap.parse_args()

    overlay = yaml.safe_load(pathlib.Path(args.config).read_text()) or {}
    cfg = load_config(overrides=overlay)
    market.set_active_profile_from_config(cfg)

    grid, meta = load_mt5_panel(str(cfg.mt5.data_dir))
    print(f"Training on {grid.height:,} bars | {meta.height} names | "
          f"{grid['date'].n_unique()} sessions  (profile={market.active_profile().name})")

    feats, cols = build_features_bars_only(grid, meta, cfg)
    labels = compute_weights(add_cross_sectional_targets(
        triple_barrier_labels(feats, cfg), cfg), cfg)
    ens = PooledEnsemble(cfg, cols).fit(labels)
    betas = {int(r["scrip_code"]): float(r["beta"])
             for r in meta.select(["scrip_code", "beta"]).iter_rows(named=True)}

    artifact = {
        "ensemble": ens,
        "cols": cols,
        "betas": betas,
        "meta_rows": meta.to_dicts(),          # plain dicts -> polars-version independent
        "profile": market.active_profile().name,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(artifact, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved model artifact -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print("Copy this file to the VM's data/mt5/ and run: "
          "python run_mt5.py --config config/mt5.yaml --model data/mt5/model.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
