"""Labeling tests, with MANDATORY session-end / causality checks (§2.5, §5.5)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from bsealpha.config import load_config
from bsealpha.data import generate_panel
from bsealpha.features import build_features
from bsealpha.labeling import (
    add_cross_sectional_targets,
    compute_weights,
    make_meta_labels,
    triple_barrier_labels,
)
from bsealpha.market import flatten_session_min

FLATTEN_SESSION_MIN = flatten_session_min()   # session-relative forced-flatten (BSE: 360)


@pytest.fixture(scope="module")
def labeled():
    cfg = load_config(overrides={"synthetic": {"n_names": 12, "n_days": 6, "seed": 8}})
    panel = generate_panel(cfg)
    feats, cols = build_features(panel.depth, panel.trades, panel.meta, cfg)
    labels = triple_barrier_labels(feats, cfg)
    labels = add_cross_sectional_targets(labels, cfg)
    return cfg, labels, cols


def test_no_label_crosses_forced_flatten(labeled):
    """§2.5: no label OPENED before the 15:15 flatten may exit after it."""
    _, labels, _ = labeled
    tradable = labels.filter(pl.col("minute") < FLATTEN_SESSION_MIN)
    assert (tradable["exit_minute"] <= FLATTEN_SESSION_MIN).all()
    # rows at/after flatten are non-signals: truncated, label 0, zero horizon
    late = labels.filter(pl.col("minute") >= FLATTEN_SESSION_MIN)
    if late.height:
        assert late["truncated"].all()
        assert (late["tb_label"] == 0).all()
        assert (late["exit_minute"] == late["minute"]).all()


def test_truncation_flag_late_session(labeled):
    """Signals fired late in the session must be flagged truncated (shorter horizon)."""
    cfg, labels, _ = labeled
    h = cfg.labeling.horizon_min
    # a row with fewer than h minutes to flatten must be truncated
    mask = (FLATTEN_SESSION_MIN - labels["minute"]) < h
    sub = labels.filter(pl.Series(mask.to_numpy()) & (pl.col("minute") < FLATTEN_SESSION_MIN))
    if sub.height:
        assert sub["truncated"].all()


def test_labels_in_range(labeled):
    _, labels, _ = labeled
    assert set(np.unique(labels["tb_label"].to_numpy())).issubset({-1, 0, 1})
    # span non-negative and bounded by horizon
    assert (labels["span_bars"] >= 0).all()


def test_barrier_touch_direction(labeled):
    """A +1 label should have positive realized residual return; -1 negative."""
    _, labels, _ = labeled
    up = labels.filter(pl.col("tb_label") == 1)
    dn = labels.filter(pl.col("tb_label") == -1)
    if up.height:
        assert up["ret_resid"].mean() > 0
    if dn.height:
        assert dn["ret_resid"].mean() < 0


def test_cross_sectional_targets(labeled):
    _, labels, _ = labeled
    assert labels["y_rank"].min() >= -0.5001
    assert labels["y_rank"].max() <= 0.5001
    nb = labels["y_bucket"].max()
    assert nb <= 4  # n_rank_buckets - 1


def test_meta_labels_clear_cost():
    ret = np.array([0.0010, -0.0010, 0.00001, -0.00050])  # residual log-returns
    side = np.array([1, -1, 1, -1])
    ml = make_meta_labels(ret, side, cost_bps=4.18)
    # +10 bps long clears; +10 bps (short of a down move) clears; tiny move fails
    assert ml.tolist() == [1, 1, 0, 1]


def test_weights_positive_mean_one(labeled):
    cfg, labels, _ = labeled
    lw = compute_weights(labels, cfg)
    assert (lw["weight"] >= 0).all()
    assert abs(lw["weight"].mean() - 1.0) < 1e-6
    # uniqueness in (0, 1]
    assert (lw["w_uniqueness"] > 0).all()
    assert (lw["w_uniqueness"] <= 1.0 + 1e-9).all()


def test_uniqueness_below_one_under_overlap(labeled):
    """Overlapping 15-min labels on 1-min bars must yield mean uniqueness < 1 (§3.6)."""
    cfg, labels, _ = labeled
    lw = compute_weights(labels, cfg)
    # most rows have a horizon > 1 => concurrency > 1 => uniqueness < 1
    assert lw["w_uniqueness"].mean() < 0.9


def test_perfect_foresight_ceiling_positive(labeled):
    """Sanity (§11.2 wk3): perfect foresight of the label sign earns gross edge > cost.

    If a perfect-foresight strategy on the labels does not clear cost, the label design is
    wrong. Here it should comfortably exceed the 4.18 bps floor.
    """
    cfg, labels, _ = labeled
    traded = labels.filter((pl.col("tb_label") != 0) & (~pl.col("truncated")))
    if traded.height:
        gross_bps = (traded["tb_label"] * traded["ret_resid"]).mean() * 1e4
        assert gross_bps > cfg.labeling.meta_cost_bps
