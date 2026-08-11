"""Validation tests: purge/embargo, CPCV paths, DSR, PBO, effective breadth."""

from __future__ import annotations

import numpy as np
import pytest

from bsealpha.validation import (
    CombinatorialPurgedCV,
    PurgedDayGroupCV,
    deflated_sharpe,
    effective_breadth,
    expected_max_sharpe,
    pbo_cscv,
)
from bsealpha.validation.trials import TrialLog


def _dates(n_days: int, per_day: int) -> np.ndarray:
    return np.repeat(np.arange(n_days), per_day)


# --------------------------------------------------------------- purge/embargo
def test_purged_cv_no_day_overlap():
    """Train and test day-sets must be disjoint, and embargo must separate them (§5.2)."""
    dates = _dates(30, 5)
    cv = PurgedDayGroupCV(n_splits=6, embargo_days=2)
    for tr, te in cv.split(dates):
        train_days = set(dates[tr].tolist())
        test_days = set(dates[te].tolist())
        assert not (train_days & test_days)                # disjoint
        # embargo: no train day within 2 of any test day
        for td in test_days:
            for e in (-2, -1, 1, 2):
                assert (td + e) not in train_days or (td + e) in test_days


def test_cpcv_path_count_and_disjoint():
    cv = CombinatorialPurgedCV(n_groups=12, k_test=2, embargo_days=1)
    assert cv.n_paths == 11
    dates = _dates(60, 3)
    seen = 0
    for tr, te, combo in cv.split(dates):
        assert not (set(dates[tr].tolist()) & set(dates[te].tolist()))
        seen += 1
    assert seen == 66   # C(12, 2)


def test_cpcv_zero_label_span_intersection():
    """No test row's day may appear in the training set (the panel purge invariant)."""
    dates = _dates(48, 4)
    cv = CombinatorialPurgedCV(n_groups=12, k_test=2, embargo_days=1)
    for tr, te, combo in cv.split(dates):
        assert len(np.intersect1d(dates[tr], dates[te])) == 0


# --------------------------------------------------------------- DSR / PBO
def test_dsr_deflates_best_of_noise():
    """DSR must reject the best-of-500 pure-noise winner (§5.4 calibration, ~0.29)."""
    rng = np.random.default_rng(1)
    T, N = 2000, 500
    R = rng.normal(0, 1, (T, N))               # 500 pure-noise strategies
    sr_trials = R.mean(0) / R.std(0, ddof=1)    # per-period Sharpe of each
    best = int(np.argmax(sr_trials))
    dsr, sr, sr_star = deflated_sharpe(R[:, best], sr_trials)
    assert sr_star > 0                          # a positive expected-max hurdle exists
    assert dsr < 0.95                           # the noise winner is correctly rejected
    # deflation must matter: comparing against many trials lowers significance vs one
    dsr_one, _, _ = deflated_sharpe(R[:, best], np.array([sr_trials[best], 0.0]))
    assert dsr <= dsr_one + 1e-9


def test_expected_max_sharpe_grows_with_trials():
    rng = np.random.default_rng(2)
    small = expected_max_sharpe(rng.normal(0, 0.5, 100))
    large = expected_max_sharpe(rng.normal(0, 0.5, 5000))
    assert large > small > 0


def test_pbo_high_on_pure_noise():
    """PBO ~ 0.5 when all configs are pure noise (no real edge to find)."""
    rng = np.random.default_rng(3)
    mat = rng.normal(0, 1, (400, 10))
    pbo, logits = pbo_cscv(mat, S=10)
    assert 0.3 < pbo < 0.7


def test_pbo_low_with_one_real_strategy():
    """With one genuinely-better config, PBO should be low."""
    rng = np.random.default_rng(4)
    mat = rng.normal(0, 1, (400, 8))
    mat[:, 0] += 0.25                           # config 0 has a real edge
    pbo, _ = pbo_cscv(mat, S=10)
    assert pbo < 0.3


# --------------------------------------------------------------- breadth
def test_effective_breadth_market_factor_collapses():
    """A common market factor collapses breadth toward ~1 (§0.5 / §3.4 calibration)."""
    rng = np.random.default_rng(5)
    T, N = 500, 40
    factor = rng.normal(0, 1, (T, 1))
    idio = rng.normal(0, 1, (T, N))
    heavy = factor * 1.0 + idio                 # strong common factor
    light = factor * 0.2 + idio                 # mostly idiosyncratic
    b_heavy = effective_breadth(heavy)
    b_light = effective_breadth(light)
    assert b_heavy < b_light
    assert b_heavy < 5                          # index bet wearing many tickers
    assert b_light > 10                         # many independent bets


# --------------------------------------------------------------- trial log
def test_trial_log_counts(tmp_path):
    log = TrialLog(tmp_path / "trials.sqlite")
    log.log({"a": 1}, 1.2, "first")
    log.log({"a": 2}, 0.8, "second")
    assert log.count() == 2
    assert len(log.sharpes()) == 2
    log.close()
