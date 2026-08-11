"""Tests for costs, fill simulator, portfolio construction, and the backtest engine."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from bsealpha.backtest import (
    CostParams,
    PassiveFillSimulator,
    PassiveOrder,
    leg_cost_rupees,
    markouts,
    round_trip_cost_bps,
    run_backtest,
    walk_the_book,
)
from bsealpha.config import load_config
from bsealpha.data import generate_panel
from bsealpha.features import build_features
from bsealpha.labeling import (
    add_cross_sectional_targets,
    compute_weights,
    triple_barrier_labels,
)
from bsealpha.models import PooledEnsemble
from bsealpha.portfolio import build_book, rebalance


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# ------------------------------------------------------------------ costs
def test_round_trip_cost_matches_report(cfg):
    """§0.1: a ₹10 lakh clip should cost ~4.18 bps round-trip."""
    p = CostParams.from_config(cfg)
    rt = round_trip_cost_bps(1_000_000, p)
    assert 4.0 < rt < 4.4


def test_stt_on_sell_only(cfg):
    p = CostParams.from_config(cfg)
    buy = leg_cost_rupees(1_000_000, +1, p)
    sell = leg_cost_rupees(1_000_000, -1, p)
    # sell leg carries the 2.5 bps STT; buy carries only 0.30 bps stamp -> sell dearer
    assert sell > buy
    assert (sell - buy) / 1_000_000 * 1e4 > 1.5


def test_brokerage_is_per_order(cfg):
    """Splitting one clip into 5 orders quintuples the flat brokerage (§6.2)."""
    p = CostParams.from_config(cfg)
    one = leg_cost_rupees(1_000_000, +1, p)
    five = 5 * leg_cost_rupees(200_000, +1, p)
    # five child orders pay 5x the ₹20 brokerage vs one
    assert five > one


# ------------------------------------------------------------------ fill sim
def test_walk_the_book_exhaustion():
    px = np.array([100.0, 100.05, 100.10])
    qty = np.array([10.0, 10.0, 10.0])
    avg, filled, exhausted = walk_the_book(px, qty, 25.0, side=1)
    assert filled == 25.0 and not exhausted
    assert 100.0 < avg < 100.10
    avg2, filled2, exhausted2 = walk_the_book(px, qty, 100.0, side=1)
    assert exhausted2 and filled2 == 30.0


def test_cancel_models_ordering():
    """Fill rate must be monotone in cancel-model optimism (§6.3)."""
    ratios = {}
    for model in ("optimistic", "proportional", "pessimistic"):
        sim = PassiveFillSimulator(cancel_model=model)
        o = PassiveOrder(side=1, price=100.0, qty=10.0)
        sim.place(o, displayed_depth=20.0)
        sim.depth_at = {100.0: 20.0}
        # a cancel removes 10 of the 20 ahead; then a sell trade of 6 arrives
        sim.on_book(1, {100.0: 10.0}, best_bid=100.0, best_ask=100.1)
        sim.on_trade(2, 100.0, 6.0, aggressor_side=-1, mid=100.05)
        ratios[model] = o.filled
    assert ratios["optimistic"] >= ratios["proportional"] >= ratios["pessimistic"]


def test_markouts_shape():
    fills = [dict(ts=0, side=1, price=100.0), dict(ts=5, side=-1, price=101.0)]
    mid_ts = np.arange(0, 40)
    mid_px = 100 + 0.01 * mid_ts
    mo = markouts(fills, mid_ts, mid_px, horizons_min=(1, 5, 15))
    assert set(mo.keys()) == {1, 5, 15}


# ------------------------------------------------------------------ portfolio
def test_build_book_is_neutral_and_capped():
    rng = np.random.default_rng(0)
    n = 30
    scores = rng.normal(0, 1, n)
    betas = rng.uniform(0.6, 1.4, n)
    sectors = np.array(["A", "B", "C"] * 10)
    adv = np.full(n, 1e8)
    w = build_book(scores, betas, sectors, adv, gross_target=5e6,
                   max_participation=0.03, min_clip=1e5, max_names=20, sector_cap=0.5)
    # market-neutral: net exposure ~ 0 relative to gross
    assert abs(w.sum()) < 0.05 * np.abs(w).sum() + 1.0
    # beta-neutral
    assert abs((w * betas).sum()) < 0.05 * np.abs(w).sum() + 1.0
    # participation cap respected
    assert (np.abs(w) <= 0.03 * adv + 1e-6).all()


def test_no_trade_band():
    current = np.array([1e6, 1e6, 0.0])
    target = np.array([1.05e6, 2e6, 3e5])
    trades = rebalance(current, target, band_frac=0.5, min_clip=5e5)
    assert trades[0] == 0.0        # 5% gap inside the 50% band
    assert trades[1] != 0.0        # 100% gap trades
    assert trades[2] == 0.0        # below min clip


# ------------------------------------------------------------------ engine
def test_backtest_runs_end_to_end(cfg):
    cfg = cfg.with_overrides({
        "synthetic": {"n_names": 20, "n_days": 8, "seed": 21},
        "model": {"lambdarank": {"min_data_in_leaf": 50, "n_estimators": 40},
                  "regression": {"min_data_in_leaf": 50, "n_estimators": 40},
                  "meta": {"min_data_in_leaf": 30, "n_estimators": 40}},
    })
    panel = generate_panel(cfg)
    feats, cols = build_features(panel.depth, panel.trades, panel.meta, cfg)
    labels = triple_barrier_labels(feats, cfg)
    labels = add_cross_sectional_targets(labels, cfg)
    labels = compute_weights(labels, cfg)
    ens = PooledEnsemble(cfg, cols).fit(labels)
    pred = ens.predict(labels)
    m = run_backtest(pred, cfg)
    assert m.daily_returns.size > 0
    assert np.isfinite(m.sharpe) and np.isfinite(m.gross_sharpe)
    assert m.n_trades > 0
    assert m.effective_breadth >= 0
    assert set(m.markout_bps.keys()) == {1, 5, 15, 30}
    # exact structural invariant: costs strictly reduce the mean daily return (gross > net)
    assert m.gross_daily_returns.mean() > m.daily_returns.mean()
    # net must be below the POSITIVE bug tripwire (§0.5); costs can make it negative
    assert m.sharpe < cfg.validation.tripwire_sharpe
    # costs are real and charged
    assert m.total_cost_rupees > 0 and m.total_impact_rupees > 0
