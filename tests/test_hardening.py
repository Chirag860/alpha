"""Phase 2 tests: trailing causal betas, lockbox, perfect-foresight ceiling, trial log."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from bsealpha.config import load_config
from bsealpha.data import generate_panel
from bsealpha.features import build_features
from bsealpha.features.index_factor import attach_factors, compute_returns
from bsealpha.features.residualize import fit_betas_trailing
from bsealpha.labeling import (
    add_cross_sectional_targets,
    compute_weights,
    triple_barrier_labels,
)
from bsealpha.validation import (
    Lockbox,
    TrialLog,
    date_split,
    perfect_foresight_ceiling,
)


@pytest.fixture(scope="module")
def labeled():
    cfg = load_config(overrides={"synthetic": {"n_names": 14, "n_days": 12, "seed": 5}})
    panel = generate_panel(cfg)
    feats, cols = build_features(panel.depth, panel.trades, panel.meta, cfg)
    labels = compute_weights(add_cross_sectional_targets(triple_barrier_labels(feats, cfg),
                                                         cfg), cfg)
    return cfg, labels, cols


# --------------------------------------------------------------- trailing betas
def test_trailing_betas_are_causal():
    """A beta at session t must use only sessions < t (no look-ahead, §5.2)."""
    cfg = load_config(overrides={"synthetic": {"n_names": 10, "n_days": 15, "seed": 2}})
    panel = generate_panel(cfg)
    from bsealpha.bars import common_minute_grid

    grid = common_minute_grid(panel.depth, panel.trades)
    grid = grid.join(panel.meta.select(["scrip_code", "sector"]), on="scrip_code")
    grid = attach_factors(compute_returns(grid))
    betas = fit_betas_trailing(grid, lookback_sessions=5)

    dates = sorted(betas["date"].unique().to_list())
    # first session per name has no prior history -> default beta 1.0, gamma 0.0
    first = betas.filter(pl.col("date") == dates[0])
    assert (first["beta"] == 1.0).all()
    assert (first["gamma"] == 0.0).all()
    # a later session generally has an estimated (non-default) beta
    later = betas.filter(pl.col("date") == dates[-1])
    assert later["beta"].n_unique() > 1 or later["beta"][0] != 1.0


def test_trailing_beta_ignores_future_shock():
    """Injecting a huge return on the LAST session must not change earlier-session betas."""
    cfg = load_config(overrides={"synthetic": {"n_names": 8, "n_days": 12, "seed": 9}})
    panel = generate_panel(cfg)
    from bsealpha.bars import common_minute_grid

    grid = common_minute_grid(panel.depth, panel.trades)
    grid = grid.join(panel.meta.select(["scrip_code", "sector"]), on="scrip_code")
    grid = attach_factors(compute_returns(grid))
    b0 = fit_betas_trailing(grid, lookback_sessions=6)

    last_date = grid["date"].max()
    shocked = grid.with_columns(
        pl.when(pl.col("date") == last_date).then(pl.col("ret") * 50.0)
        .otherwise(pl.col("ret")).alias("ret")
    )
    b1 = fit_betas_trailing(shocked, lookback_sessions=6)
    # betas on all-but-last session are identical (future shock can't leak backward)
    earlier = b0.filter(pl.col("date") < last_date).sort(["scrip_code", "date"])
    earlier1 = b1.filter(pl.col("date") < last_date).sort(["scrip_code", "date"])
    assert np.allclose(earlier["beta"].to_numpy(), earlier1["beta"].to_numpy())


# --------------------------------------------------------------- lockbox
def test_date_split_holds_out_recent():
    cfg = load_config(overrides={"synthetic": {"n_names": 6, "n_days": 10, "seed": 1}})
    panel = generate_panel(cfg)
    grid = panel.daily
    research, lockbox = date_split(grid, holdout_days=3)
    assert research["date"].max() < lockbox["date"].min()
    assert lockbox["date"].n_unique() == 3


def test_lockbox_is_single_use():
    df = pl.DataFrame({"date": [1, 2], "x": [0.1, 0.2]})
    lb = Lockbox(df, name="test")
    assert lb.is_sealed
    assert lb.n_rows() == 2                # metadata is safe
    opened = lb.open()
    assert opened.height == 2
    assert not lb.is_sealed
    with pytest.raises(RuntimeError):      # second peek is a hard error
        lb.open()


# --------------------------------------------------------------- ceiling
def test_perfect_foresight_ceiling(labeled):
    cfg, labels, _ = labeled
    c = perfect_foresight_ceiling(labels, cfg)
    # perfect knowledge of the side must earn positive gross edge per trade
    assert c.gross_bps_per_trade > 0
    # and net of cost it should clear the fee floor comfortably (label design viable)
    assert c.net_bps_per_trade > 0
    assert "ceiling" in c.summary()


# --------------------------------------------------------------- trial log DSR
def test_trial_log_drives_dsr_n(labeled, tmp_path):
    cfg, labels, cols = labeled
    from bsealpha.validation import evaluate

    db = tmp_path / "trials.sqlite"
    log = TrialLog(db)
    for _ in range(3):                     # pretend we've run 3 configs already
        log.log({"dummy": _}, 0.5)
    rep = evaluate(labels, cols, cfg, run_cpcv=False, trial_log=log)
    # N must include the prior logged trials + this run
    assert rep.n_trials >= 4
    assert log.count() >= 4
    log.close()
