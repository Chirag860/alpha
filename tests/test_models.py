"""Tests for models: GBM (LambdaRank/regression/meta), calibration, ensemble."""

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
    triple_barrier_labels,
)
from bsealpha.models import GBM, IsotonicCalibrator, PooledEnsemble, lightgbm_available
from bsealpha.models.gbm import make_group_array


@pytest.fixture(scope="module")
def labeled():
    cfg = load_config(overrides={
        "synthetic": {"n_names": 20, "n_days": 10, "seed": 11},
        "model": {"lambdarank": {"min_data_in_leaf": 50, "n_estimators": 60},
                  "regression": {"min_data_in_leaf": 50, "n_estimators": 60},
                  "meta": {"min_data_in_leaf": 30, "n_estimators": 60}},
    })
    panel = generate_panel(cfg)
    feats, cols = build_features(panel.depth, panel.trades, panel.meta, cfg)
    labels = triple_barrier_labels(feats, cfg)
    labels = add_cross_sectional_targets(labels, cfg)
    labels = compute_weights(labels, cfg)
    return cfg, labels, cols


def test_lightgbm_available():
    # Environment installed libomp; expect LightGBM to load. Fallback is still tested by
    # construction but we assert the intended primary path is active here.
    assert lightgbm_available()


def test_make_group_array():
    keys = pl.DataFrame({"date": [1, 1, 1, 2, 2], "minute": [0, 0, 1, 0, 0]})
    g = make_group_array(keys)
    assert g.tolist() == [2, 1, 2]
    assert g.sum() == 5


def test_gbm_regression_learns_signal(labeled):
    cfg, labels, cols = labeled
    X = labels.select(cols).to_numpy()
    y = labels["y_rank"].to_numpy()
    g = GBM("regression", cfg.model.regression.to_dict(), feature_cols=cols)
    g.fit(X, y)
    pred = g.predict(X)
    # in-sample correlation with the target should be clearly positive
    assert np.corrcoef(pred, y)[0, 1] > 0.1


def test_gbm_ranker_with_groups(labeled):
    cfg, labels, cols = labeled
    labels = labels.sort(["date", "minute", "scrip_code"])
    X = labels.select(cols).to_numpy()
    y = labels["y_bucket"].to_numpy()
    group = make_group_array(labels.select(["date", "minute"]))
    g = GBM("lambdarank", cfg.model.lambdarank.to_dict(), feature_cols=cols)
    g.fit(X, y, group=group)
    pred = g.predict(X)
    assert pred.shape[0] == labels.height
    assert np.isfinite(pred).all()


def test_monotone_constraint_applied(labeled):
    cfg, labels, cols = labeled
    g = GBM("regression", cfg.model.regression.to_dict(), feature_cols=cols,
            monotone_features=["ofi_5m_xs"])
    g.fit(labels.select(cols).to_numpy(), labels["y_rank"].to_numpy())
    assert g.model is not None  # trains without error with a constraint present


def test_isotonic_calibrator_monotone():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 2000)
    labels = (rng.uniform(0, 1, 2000) < scores).astype(int)  # calibrated-by-construction
    cal = IsotonicCalibrator().fit(scores, labels)
    p = cal.transform(np.array([0.1, 0.5, 0.9]))
    assert p[0] <= p[1] <= p[2]
    assert (p >= 0).all() and (p <= 1).all()


def test_tcn_embedder_optional(labeled):
    """Optional TCN trunk (§4.3): fit/transform, causal embedding of the right shape.

    Gated behind ``BSEALPHA_RUN_TCN=1`` because it needs PyTorch and some torch builds are
    unstable on very new Python versions. The trunk is off by default in the pipeline.
    """
    import os

    import pytest

    from bsealpha.models import torch_available

    if not os.environ.get("BSEALPHA_RUN_TCN"):
        pytest.skip("set BSEALPHA_RUN_TCN=1 to exercise the optional TCN trunk")
    if not torch_available():
        pytest.skip("PyTorch not installed; TCN trunk is optional")
    from bsealpha.models import TCNEmbedder

    cfg, labels, cols = labeled
    keep = labels["scrip_code"].unique().sort()[:3]
    days = labels["date"].unique().sort()[:3]
    labels = (labels.filter(pl.col("scrip_code").is_in(keep) & pl.col("date").is_in(days))
              .sort(["scrip_code", "date", "minute"]))
    X = labels.select(cols).to_numpy()
    y = labels["y_rank"].to_numpy()
    scrip = labels["scrip_code"].to_numpy()
    stock_idx = np.unique(scrip, return_inverse=True)[1]
    day_idx = np.unique(labels["date"].to_numpy().astype(str), return_inverse=True)[1]
    tcn_cfg = cfg.model.tcn.with_overrides({"seq_len": 8, "epochs": 1, "channels": 8,
                                            "embedding_dim": 8, "levels": 2})
    emb = TCNEmbedder(tcn_cfg, n_features=X.shape[1], n_stocks=int(stock_idx.max()) + 1)
    emb.fit(X, y, stock_idx, day_idx)
    out = emb.transform(X, stock_idx, day_idx)
    assert out.shape == (X.shape[0], 8)
    assert np.isfinite(out).all()


def test_pooled_ensemble_fit_predict(labeled):
    cfg, labels, cols = labeled
    ens = PooledEnsemble(cfg, cols).fit(labels)
    out = ens.predict(labels)
    assert "primary_score" in out.columns
    assert "p_act" in out.columns
    assert out["p_act"].min() >= 0.0 and out["p_act"].max() <= 1.0
    assert set(np.unique(out["primary_side"].to_numpy())).issubset({-1, 1})
    # primary score should have positive IC with the realized vol-adjusted residual return
    ic = np.corrcoef(out["primary_score"].to_numpy(), labels["y_voladj"].to_numpy())[0, 1]
    assert ic > 0.0
