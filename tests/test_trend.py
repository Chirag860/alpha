"""Correctness tests for the trend + carry engine (run on synthetic data)."""

from __future__ import annotations

import numpy as np
import pytest

from bsealpha.trend import (
    TrendParams,
    backtest,
    ewma_vol,
    log_returns,
    run_trend_backtest,
    simple_returns,
    target_weights,
    to_matrices,
    tsmom_signal,
    validate_book,
)
from bsealpha.trend.synthetic import generate_daily_panel


@pytest.fixture(scope="module")
def trending():
    return generate_daily_panel(n_inst=24, n_days=1800, seed=1, frac_trending=0.8)


@pytest.fixture(scope="module")
def noise():
    return generate_daily_panel(n_inst=24, n_days=1800, seed=2, frac_trending=0.0)


# --------------------------------------------------------------- captures trend
def test_positive_sharpe_on_trending_data(trending):
    grid, meta = trending
    res, _ = run_trend_backtest(grid, meta, TrendParams())
    assert res.metrics["sharpe"] > 0.5           # a real trend edge must show up clearly
    assert res.metrics["ann_vol"] > 0.0


def test_no_free_sharpe_on_pure_noise(noise):
    """On driftless random walks the engine must NOT manufacture edge (net of costs)."""
    grid, meta = noise
    res, _ = run_trend_backtest(grid, meta, TrendParams())
    assert res.metrics["sharpe"] < 0.5           # no trend -> no (real) Sharpe


# --------------------------------------------------------------- vol targeting
def test_vol_target_is_approximately_hit(trending):
    grid, meta = trending
    res, _ = run_trend_backtest(grid, meta, TrendParams(target_ann_vol=0.15))
    assert 0.09 < res.metrics["ann_vol"] < 0.24  # realized vol near the 15% target


# --------------------------------------------------------------- costs bite
def test_costs_reduce_return(trending):
    """Same weights, higher per-side cost => strictly lower net return."""
    grid, meta = trending
    _, book = run_trend_backtest(grid, meta, TrendParams())
    w, ret = book["weights"], book["ret"]
    p = TrendParams(vol_target_overlay=False)          # deterministic: cost is the only variable
    lo = backtest(w, ret, p, cost_bps=np.zeros(w.shape[1]))
    hi = backtest(w, ret, p, cost_bps=np.full(w.shape[1], 50.0))
    assert hi.cost.sum() > lo.cost.sum()
    assert hi.metrics["ann_return"] < lo.metrics["ann_return"]


# --------------------------------------------------------------- no lookahead
def test_positions_are_lagged(trending):
    """Day-0 pnl must be zero and pnl[t] must use weights[t-1] (no same-day peeking)."""
    grid, meta = trending
    res, _ = run_trend_backtest(grid, meta, TrendParams())
    assert res.net[0] == 0.0 or abs(res.net[0]) < 1e-9 + abs(res.cost[0])
    # reconstruct gross pnl from lagged weights and confirm it matches
    w, ret = res.weights, res.ret
    recon = np.zeros(len(ret))
    recon[1:] = np.sum(w[:-1] * ret[1:], axis=1)
    assert np.allclose(recon, res.gross_pnl)


def test_independent_random_signal_gives_no_edge():
    """A signal drawn INDEPENDENTLY of returns must earn ~0 — the engine mines no free edge.

    (Robust to any real autocorrelation in the data, unlike a return-derived signal: an
    independent signal is uncorrelated with future returns by construction.)
    """
    grid, meta = generate_daily_panel(n_inst=12, n_days=1500, seed=5, frac_trending=0.0)
    _, sym, close = to_matrices(grid)
    ret = simple_returns(close)
    vol = ewma_vol(log_returns(close), 33)
    sig = np.random.default_rng(7).normal(0.0, 1.0, ret.shape)   # independent of returns
    w = target_weights(sig, vol, TrendParams())
    res = backtest(w, ret, TrendParams(cost_bps_per_side=0.0, vol_target_overlay=False))
    assert abs(res.metrics["sharpe"]) < 0.7       # no relationship -> no Sharpe (bar sampling noise)


# --------------------------------------------------------------- validation report
def test_validation_runs(trending):
    grid, meta = trending
    res, book = run_trend_backtest(grid, meta, TrendParams())
    v = validate_book(book["dates"], res, TrendParams(), n_trials=10)
    assert v.sharpe_oos != 0.0
    assert 0.0 <= v.dsr <= 1.0
    assert len(v.per_year) >= 3
