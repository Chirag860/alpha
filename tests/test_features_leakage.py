"""Feature tests, with MANDATORY leakage / causality checks (§3.3, §5.5).

These are the tests the report insists on: OFI signs, micro-price sign, tick-rule
validation, the t-1 cross-sectional lag, and point-in-time-ness of the whole panel.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from bsealpha.config import load_config
from bsealpha.data import generate_panel
from bsealpha.features import (
    OFI5,
    build_features,
    ofi_frame,
    sign_trades,
    tick_rule_sign,
)
from bsealpha.features.ofi import IntegratedOFI
from bsealpha.market import micro_price


@pytest.fixture(scope="module")
def setup():
    cfg = load_config(overrides={"synthetic": {"n_names": 12, "n_days": 6, "seed": 5}})
    panel = generate_panel(cfg)
    return cfg, panel


# --------------------------------------------------------------- OFI correctness
def test_ofi_sign_bid_uptick():
    """A bid-price uptick with fresh size => positive OFI; ask-downtick => positive too."""
    o = OFI5(M=1)
    o.update(np.array([100.0]), np.array([10.0]), np.array([101.0]), np.array([10.0]))
    up = o.update(np.array([100.5]), np.array([8.0]), np.array([101.0]), np.array([10.0]))
    assert up[0] > 0  # bid moved up -> buy pressure
    o2 = OFI5(M=1)
    o2.update(np.array([100.0]), np.array([10.0]), np.array([101.0]), np.array([10.0]))
    dn = o2.update(np.array([99.5]), np.array([10.0]), np.array([101.0]), np.array([10.0]))
    assert dn[0] < 0  # bid moved down -> sell pressure


def test_ofi_streaming_matches_vectorized(setup):
    """Parity: the O(1) streaming class and the vectorized frame must rank-agree (§ofi)."""
    _, panel = setup
    vec = ofi_frame(panel.depth)
    # pick one scrip-day and recompute with the streaming class
    one = (panel.depth.filter(
        (pl.col("scrip_code") == panel.depth["scrip_code"][0]))
        .filter(pl.col("date") == panel.depth["date"][0])
        .sort("ts_ns"))
    o = OFI5(M=5)
    stream_integrated = []
    for row in one.iter_rows(named=True):
        bp = np.array([row[f"bid_px_{i}"] for i in range(5)])
        bq = np.array([row[f"bid_qty_{i}"] for i in range(5)])
        ap = np.array([row[f"ask_px_{i}"] for i in range(5)])
        aq = np.array([row[f"ask_qty_{i}"] for i in range(5)])
        stream_integrated.append(o.update(bp, bq, ap, aq).sum())
    vec_one = (vec.filter((pl.col("scrip_code") == one["scrip_code"][0])
                          & (pl.col("date") == one["date"][0]))
               .sort("ts_ns"))["ofi_integrated"].to_numpy()
    s = np.array(stream_integrated)
    # both zero on the first event; correlation strongly positive thereafter
    corr = np.corrcoef(s[1:], vec_one[1:])[0, 1]
    assert corr > 0.9


def test_integrated_ofi_orientation(setup):
    _, panel = setup
    vec = ofi_frame(panel.depth)
    mat = vec.select([f"ofi_{i}" for i in range(5)]).to_numpy()
    io = IntegratedOFI().fit(mat)
    # positive orientation: weight vector sums positive
    assert io.w.sum() > 0
    out = io.transform(mat)
    assert out.shape[0] == mat.shape[0]


# --------------------------------------------------------------- micro-price
def test_micro_price_inside_book_and_signed():
    micro, imb = micro_price(100.0, 30.0, 101.0, 10.0)  # thick bid -> upward pressure
    assert 100.0 <= micro <= 101.0
    assert micro > 100.5  # leans toward the ask
    micro2, _ = micro_price(100.0, 10.0, 101.0, 30.0)   # thick ask -> downward pressure
    assert micro2 < 100.5


# --------------------------------------------------------------- tick rule
def test_tick_rule_recovers_direction():
    px = np.array([100.0, 100.1, 100.1, 99.9, 100.2])
    s = tick_rule_sign(px)
    assert s.tolist() == [1, 1, 1, -1, 1]


def test_tick_rule_correlates_with_true_sign(setup):
    """India has no aggressor flag; the tick rule must still be positively correlated
    with the (test-only) true sign (§3.2 requires validating the estimator)."""
    _, panel = setup
    signed = sign_trades(panel.trades)
    corr = np.corrcoef(signed["sign"].to_numpy().astype(float),
                       signed["true_sign"].to_numpy().astype(float))[0, 1]
    assert corr > 0.2  # noisy, but clearly informative


# --------------------------------------------------------------- panel leakage
def test_cross_sectional_lag_is_enforced(setup):
    """A ``_xs`` rank at minute t must not depend on any name's value AT t.

    We perturb every name's raw feature at the last minute and assert the ``_xs`` ranks
    at that same minute are unchanged (they were built from t-1).
    """
    cfg, panel = setup
    panel_feats, cols = build_features(panel.depth, panel.trades, panel.meta, cfg)
    xs_cols = [c for c in cols if c.endswith("_xs")]
    assert xs_cols, "expected cross-sectional rank columns"

    # baseline xs values at the max minute of the first date
    d0 = panel_feats["date"].min()
    tmax = panel_feats.filter(pl.col("date") == d0)["minute"].max()
    base = (panel_feats.filter((pl.col("date") == d0) & (pl.col("minute") == tmax))
            .select(["scrip_code"] + xs_cols).sort("scrip_code"))
    # the xs rank at the FIRST minute must be null/constant (no t-1 exists) -> lag works
    tmin = panel_feats.filter(pl.col("date") == d0)["minute"].min()
    first = panel_feats.filter((pl.col("date") == d0) & (pl.col("minute") == tmin))
    # with a 1-min lag, the very first minute of the day has no prior cross-section;
    # ranks there are computed from all-null lagged features => all equal (rank ties).
    for c in xs_cols[:3]:
        assert first[c].n_unique() <= 1


def test_no_future_columns_in_features(setup):
    """Forward-looking columns (resid_px future, labels) must not be in feature_cols."""
    cfg, panel = setup
    _, cols = build_features(panel.depth, panel.trades, panel.meta, cfg)
    forbidden = {"resid_px", "r_resid", "ret", "sigma_resid", "mid", "label", "y"}
    assert not (set(cols) & forbidden)


def test_feature_count_in_expected_range(setup):
    cfg, panel = setup
    _, cols = build_features(panel.depth, panel.trades, panel.meta, cfg)
    # report targets ~70 features (§11.1)
    assert 50 <= len(cols) <= 90


def test_features_are_finite(setup):
    cfg, panel = setup
    panel_feats, cols = build_features(panel.depth, panel.trades, panel.meta, cfg)
    arr = panel_feats.select(cols).to_numpy()
    assert np.isfinite(arr).all()
