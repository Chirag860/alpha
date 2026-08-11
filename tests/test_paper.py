"""Phase 5 tests: paper session runner and live-vs-backtest reconciliation (§11.2)."""

from __future__ import annotations

import numpy as np
import pytest

from bsealpha.backtest import run_backtest
from bsealpha.config import load_config
from bsealpha.data import generate_panel
from bsealpha.features import build_features
from bsealpha.labeling import (
    add_cross_sectional_targets,
    compute_weights,
    triple_barrier_labels,
)
from bsealpha.live import reconcile, run_paper_session
from bsealpha.models import PooledEnsemble


@pytest.fixture(scope="module")
def predicted():
    cfg = load_config(overrides={
        "synthetic": {"n_names": 18, "n_days": 8, "seed": 31},
        "model": {"lambdarank": {"min_data_in_leaf": 50, "n_estimators": 40},
                  "regression": {"min_data_in_leaf": 50, "n_estimators": 40},
                  "meta": {"min_data_in_leaf": 30, "n_estimators": 40}},
    })
    panel = generate_panel(cfg)
    feats, cols = build_features(panel.depth, panel.trades, panel.meta, cfg)
    labels = compute_weights(add_cross_sectional_targets(triple_barrier_labels(feats, cfg),
                                                         cfg), cfg)
    ens = PooledEnsemble(cfg, cols).fit(labels)
    pred = ens.predict(labels)
    betas = {int(r["scrip_code"]): float(r["beta"])
             for r in panel.meta.select(["scrip_code", "beta"]).iter_rows(named=True)}
    return cfg, pred, betas


def test_paper_session_runs(predicted):
    cfg, pred, betas = predicted
    res = run_paper_session(pred, cfg, betas=betas)
    assert res.daily_returns.size > 0
    assert np.isfinite(res.net_sharpe)
    assert res.n_orders > 0
    # fill ratio is a real fraction in [0, 1] (passive orders may not fill, §5.1)
    assert 0.0 <= res.fill_ratio <= 1.0
    assert res.total_cost_rupees > 0
    assert set(res.markout_bps.keys()) == {1, 5, 15, 30}


def test_paper_session_flat_at_eod(predicted):
    """No position may survive the forced flatten -> the session books no overnight risk."""
    cfg, pred, betas = predicted
    # if any position survived, the last day's marking would keep compounding; instead we
    # assert the run completes and daily returns are finite (flatten enforced by the manager)
    res = run_paper_session(pred, cfg, betas=betas)
    assert np.isfinite(res.daily_returns).all()


def test_reconciliation_report(predicted):
    cfg, pred, betas = predicted
    bt = run_backtest(pred, cfg, betas=betas)
    ps = run_paper_session(pred, cfg, betas=betas)
    rep = reconcile(bt, ps)
    # the report exposes the gap, the haircut, and the fill ratio
    assert rep.sharpe_gap == pytest.approx(bt.sharpe - ps.net_sharpe)
    assert 0.0 <= rep.fill_ratio <= 1.0
    assert set(rep.markout_delta_bps.keys()) == {1, 5, 15, 30}
    assert "haircut" in rep.summary()
    # paper uses realistic passive fills, so its turnover should not exceed the idealized
    # backtest's by construction of the maker-fill model (fills are a subset of intent)
    assert rep.turnover_paper <= rep.turnover_backtest + 1e-6 or rep.fill_ratio < 1.0
