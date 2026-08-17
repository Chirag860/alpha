"""Tests for the Phase-4 verdict harness (with vs without news), on a tiny synthetic panel.

The point of these is not to prove edge -- the stub extractor has none by design -- but to
prove the *harness* is sane: it runs the user's own validation twice, the join lands news on
real cells, and a no-signal feature does not manufacture a leak tripwire. That's the honest
null the whole method depends on.
"""

from __future__ import annotations

import pytest

from bsealpha.config import load_config
from bsealpha.text import (
    NEWS_FEATURES,
    NewsEvalResult,
    add_news_features,
    evaluate_news_feature,
)
from bsealpha.pipeline import build_labeled_panel
from bsealpha.validation.runner import ValidationReport


def test_new_tripwires_ignores_wobble_in_preexisting():
    # a meta-AUC tripwire present in BOTH runs, value slightly different, must NOT count as new
    base = ValidationReport(tripwires=["TRIPWIRE: meta-AUC 0.673 > 0.62 -- suspect leak (§11.3)."])
    aug = ValidationReport(tripwires=["TRIPWIRE: meta-AUC 0.672 > 0.62 -- suspect leak (§11.3)."])
    r = NewsEvalResult(base, aug, NEWS_FEATURES, 0.1)
    assert r.new_tripwires == []
    # a genuinely new category IS flagged
    aug2 = ValidationReport(tripwires=base.tripwires + ["TRIPWIRE: |IC| 0.11 > 0.08 -- suspect leak (§11.3)."])
    r2 = NewsEvalResult(base, aug2, NEWS_FEATURES, 0.1)
    assert len(r2.new_tripwires) == 1 and "|IC|" in r2.new_tripwires[0]


@pytest.fixture(scope="module")
def cfg():
    return load_config(overrides={"synthetic": {"n_names": 12, "n_days": 8, "seed": 1}})


def test_add_news_features_lands_on_real_cells(cfg):
    labels, feature_cols, _panel, _n = build_labeled_panel(cfg, seed=1)
    out, cols, coverage = add_news_features(labels, cfg, ann_seed=3)
    assert cols == NEWS_FEATURES
    assert out.height == labels.height                 # 1:1 join, panel preserved
    assert all(c in out.columns for c in NEWS_FEATURES)
    assert 0.0 < coverage <= 1.0                       # some cells carry news
    # no nulls introduced
    for c in NEWS_FEATURES:
        assert out[c].null_count() == 0


def test_evaluate_news_feature_runs_and_is_honest(cfg):
    # run_cpcv=False keeps it fast; the IC / tripwire comparison is what matters here
    res = evaluate_news_feature(cfg, seed=1, run_cpcv=False, ann_seed=3)
    assert res.news_cols == NEWS_FEATURES
    # the stub carries no signal, so it must not fabricate a leak tripwire
    assert res.new_tripwires == []
    # deltas are finite numbers the decision rule can read
    assert res.d_ic == pytest.approx(res.augmented.ic - res.baseline.ic)
    # report renders and states a verdict
    txt = res.report()
    assert "Phase 4" in txt and "verdict:" in txt
