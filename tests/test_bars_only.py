"""Tests for the reduced BARS-ONLY feature path (free-data mode), no network required.

Derives a bars-only common grid from the synthetic panel (dropping depth/trades), then runs
the reduced feature builder + labeling + model + backtest to prove the free-data path works
end to end. A live yfinance test is intentionally omitted (network-dependent).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from bsealpha.bars import common_minute_grid
from bsealpha.config import load_config
from bsealpha.data import generate_panel
from bsealpha.features import build_features_bars_only
from bsealpha.labeling import (
    add_cross_sectional_targets,
    compute_weights,
    triple_barrier_labels,
)
from bsealpha.models import PooledEnsemble


@pytest.fixture(scope="module")
def bars_and_meta():
    cfg = load_config(overrides={
        "synthetic": {"n_names": 14, "n_days": 8, "seed": 12},
        "model": {"lambdarank": {"min_data_in_leaf": 50, "n_estimators": 40},
                  "regression": {"min_data_in_leaf": 50, "n_estimators": 40},
                  "meta": {"min_data_in_leaf": 30, "n_estimators": 40}},
    })
    panel = generate_panel(cfg)
    # a bars-only grid: keep only OHLC/mid/turnover columns a free vendor would provide
    grid = common_minute_grid(panel.depth, panel.trades).select(
        ["scrip_code", "date", "minute", "session_min", "open", "high", "low",
         "close", "vwap", "turnover", "n_trades", "mid"]
    )
    meta = panel.meta.select(["scrip_code", "sector", "beta", "circuit_band_pct"])
    return cfg, grid, meta


def test_bars_only_features_reduced(bars_and_meta):
    cfg, grid, meta = bars_and_meta
    feats, cols = build_features_bars_only(grid, meta, cfg)
    # microstructure families must be ABSENT (no depth/trades)
    assert not any(c.startswith(("ofi_", "micro_minus", "imb_", "signed_vol", "spread_bps"))
                   for c in cols)
    # but residual momentum / vol / session / factor features are present
    assert any("resid_mom" in c for c in cols)
    assert any(c.startswith("rv_") for c in cols)
    assert "sin_tod" in cols and "dispersion" in cols
    # materially fewer features than the full depth-based set (~61); reduced ~25 on real data
    assert 12 <= len(cols) <= 40
    assert np.isfinite(feats.select(cols).to_numpy()).all()


def test_bars_only_pipeline_runs(bars_and_meta):
    cfg, grid, meta = bars_and_meta
    feats, cols = build_features_bars_only(grid, meta, cfg)
    labels = compute_weights(add_cross_sectional_targets(
        triple_barrier_labels(feats, cfg), cfg), cfg)
    ens = PooledEnsemble(cfg, cols).fit(labels)
    pred = ens.predict(labels)
    assert "primary_score" in pred.columns and "p_act" in pred.columns
    # in-sample the reduced model still has non-negative IC with the target
    ic = np.corrcoef(pred["primary_score"].to_numpy(), labels["y_voladj"].to_numpy())[0, 1]
    assert ic > 0.0


def test_bars_only_backtest_runs(bars_and_meta):
    cfg, grid, meta = bars_and_meta
    feats, cols = build_features_bars_only(grid, meta, cfg)
    labels = compute_weights(add_cross_sectional_targets(
        triple_barrier_labels(feats, cfg), cfg), cfg)
    pred = PooledEnsemble(cfg, cols).fit(labels).predict(labels)
    from bsealpha.backtest import run_backtest
    m = run_backtest(pred, cfg)
    assert m.daily_returns.size > 0 and np.isfinite(m.sharpe)
    assert set(m.markout_bps.keys()) == {1, 5, 15, 30}
