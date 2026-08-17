"""Correctness tests for the Crash indicator + setup layer.

These lock the exact definitions from the research brief. The RSI test reproduces Wilder's own
worked example (StockCharts publishes the first value as 70.53); the CRSI tests pin the warm-up
boundary and the [0,100] bound; the rest guard the streak/percent-rank/HV conventions and the
screen wiring. If any of these break, the indicator has silently drifted from the book.
"""

from __future__ import annotations

import numpy as np
import pytest

from bsealpha.crash import (
    CrashParams,
    connors_rsi,
    crash_signals,
    hv_annualized,
    percent_rank,
    roc1,
    streak,
    wilder_rsi,
)

# Canonical StockCharts ChartSchool RSI(14) example closes.
WILDER_CLOSES = np.array([
    44.3389, 44.0902, 44.1497, 43.6124, 44.3278, 44.8264, 45.0955, 45.4245,
    45.8433, 46.0826, 45.8931, 46.0328, 45.6140, 46.2820, 46.2820, 46.0028,
    46.0328, 46.4116, 46.2222, 45.6439, 46.2122, 46.2521, 45.7137, 46.4515,
    45.7835, 45.3548, 44.0288, 44.1783, 44.2181, 44.5672, 43.4205, 42.6628, 43.1314,
])


# ----------------------------------------------------------------- Wilder RSI
def test_wilder_rsi_reproduces_stockcharts_worked_example():
    rsi = wilder_rsi(WILDER_CLOSES, 14)
    assert np.all(np.isnan(rsi[:14]))                 # warm-up: no value before index 14
    assert rsi[14] == pytest.approx(70.53, abs=0.01)  # published first RSI

def test_wilder_rsi_is_not_ewm():
    """Guard against the flagged ewm(adjust=False) regression: seeding with the first observation
    instead of the SMA-of-n gives a materially different (far lower) first value here."""
    rsi = wilder_rsi(WILDER_CLOSES, 14)
    assert rsi[14] > 65.0                              # ewm-seeded impl lands ~50, Wilder ~70.5

def test_rsi_bounded_and_flat_window_is_50():
    rng = np.random.default_rng(3)
    c = 10 * np.cumprod(1 + rng.normal(0, 0.02, 500))
    rsi = wilder_rsi(c, 3)
    v = rsi[~np.isnan(rsi)]
    assert v.min() >= 0.0 and v.max() <= 100.0
    assert wilder_rsi(np.full(20, 7.0), 3)[5] == 50.0  # perfectly flat -> neutral


# ----------------------------------------------------------------- streak
def test_streak_extends_and_hard_resets_on_unchanged():
    s = streak(np.array([1.0, 2.0, 3.0, 3.0, 4.0, 3.0, 2.0]))
    assert s.tolist() == [0.0, 1.0, 2.0, 0.0, 1.0, -1.0, -2.0]

def test_streak_flips_sign_immediately():
    s = streak(np.array([5.0, 6.0, 7.0, 6.0]))          # up, up, then down
    assert s.tolist() == [0.0, 1.0, 2.0, -1.0]


# ----------------------------------------------------------------- percent rank
def test_percent_rank_strict_and_excludes_today():
    # priors 1..5 (window 5), today = 3 -> two priors (1,2) strictly below -> 40.0
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 3.0])
    pr = percent_rank(vals, window=5)
    assert np.all(np.isnan(pr[:5]))
    assert pr[5] == pytest.approx(40.0)

def test_percent_rank_ties_do_not_count():
    vals = np.array([2.0, 2.0, 2.0, 2.0, 2.0])          # all equal
    pr = percent_rank(vals, window=4)
    assert pr[4] == pytest.approx(0.0)                   # strict < -> no prior counts


# ----------------------------------------------------------------- ConnorsRSI
def test_connors_rsi_bounds_and_warmup():
    rng = np.random.default_rng(7)
    c = 5 * np.cumprod(1 + rng.normal(0, 0.05, 1200))
    cr = connors_rsi(c)                                  # (3,2,100)
    valid = ~np.isnan(cr)
    assert int(np.argmax(valid)) == 101                 # first valid = 102nd close
    assert np.all(np.isnan(cr[:101]))
    assert np.nanmin(cr) >= 0.0 and np.nanmax(cr) <= 100.0

def test_connors_rsi_is_mean_of_three_components():
    from bsealpha.crash.indicators import wilder_rsi as w, streak as st, roc1 as rc
    rng = np.random.default_rng(11)
    c = 8 * np.cumprod(1 + rng.normal(0, 0.04, 300))
    cr = connors_rsi(c)
    expected = (w(c, 3) + w(st(c), 2) + percent_rank(rc(c), 100)) / 3.0
    ok = ~np.isnan(cr)
    assert np.allclose(cr[ok], expected[ok])


# ----------------------------------------------------------------- HV
def test_hv_recovers_known_sigma():
    rng = np.random.default_rng(1)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.05, 4000))  # ~5%/day -> ann ~0.05*sqrt(252)=0.794
    hv = hv_annualized(c, window=100)
    assert np.all(np.isnan(hv[:100]))
    assert np.nanmean(hv) == pytest.approx(0.794, abs=0.03)


# ----------------------------------------------------------------- screen wiring
def _series_with_final_setup(n=200, seed=7):
    """Build a >$5, high-vol (>100% ann) series ending in a clean up-run so CRSI is extreme.

    Returns are demeaned so the level does not random-walk into the floor, keeping realized vol
    high without the price collapsing; a six-bar +7% run is then appended onto the actual last
    close (not a fixed early value) to drive RSI(3), the streak, and the ROC percentile all high.
    """
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.085, n)                           # ~135% ann vol (survives run dilution)
    r -= r.mean()                                         # demean -> no net drift, stays > $5
    c = 100.0 * np.cumprod(1 + r)
    run = c[-1] * np.cumprod(np.full(6, 1.07))            # six +7% closes onto the real last level
    return np.concatenate([c, run])

def test_entry_setup_fires_on_overbought_highvol_liquid_name():
    c = _series_with_final_setup()
    vol = np.full_like(c, 2_000_000)                      # clears the 1M share screen
    sig = crash_signals(c, vol, CrashParams())
    assert sig.crsi[-1] >= 90.0
    assert sig.hv[-1] > 1.00
    assert sig.entry_setup[-1]
    assert sig.limit_price[-1] == pytest.approx(c[-1] * 1.03)

def test_illiquid_name_is_screened_out():
    c = _series_with_final_setup()
    vol = np.full_like(c, 100_000)                        # below the 1M floor
    sig = crash_signals(c, vol, CrashParams())
    assert not sig.tradeable[-1]
    assert not sig.entry_setup[-1]                        # same signal, killed by liquidity

def test_penny_name_is_screened_out():
    c = _series_with_final_setup() / 40.0                 # ~$2.5 prices (sub-$5)
    vol = np.full_like(c, 5_000_000)
    sig = crash_signals(c, vol, CrashParams())
    assert not sig.price_ok[-1]
    assert not sig.entry_setup[-1]

def test_exit_signal_on_loss_of_overbought():
    rng = np.random.default_rng(5)
    c = 20 * np.cumprod(1 + rng.normal(0, 0.05, 200))
    c[-5:] = c[-6] * np.cumprod(np.full(5, 0.95))         # five down closes -> CRSI collapses
    vol = np.full_like(c, 2_000_000)
    sig = crash_signals(np.maximum(c, 6.0), vol, CrashParams())
    assert sig.crsi[-1] < 30.0
    assert sig.exit_signal[-1]
