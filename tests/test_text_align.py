"""Tests for Phase-3 alignment: point-in-time correctness of the news->grid attachment.

The load-bearing properties: (1) no bar earlier than a filing's actionable instant ever
carries that filing (no look-ahead), (2) an after-close filing attaches to the NEXT
session, not the same day, (3) scores decay with age, (4) no-news cells are zero. All
offline.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from bsealpha.text import (
    NEWS_FEATURES,
    align_announcement_features,
    attach_scores,
    stub_extract_frame,
    synth_announcements,
    synthetic_grid,
)
from bsealpha.text.loaders import _ist_naive_to_fields


def _one_announcement(scrip: int, when: dt.datetime, *, direction=0.8, materiality=1.0,
                      surprise=0.5, novelty=0.5) -> pl.DataFrame:
    """A single scored announcement disclosed at the given IST wall-clock time."""
    disclosed_ns, _, _ = _ist_naive_to_fields(when)
    return pl.DataFrame(
        [{"scrip_code": scrip, "disclosed_ns": disclosed_ns, "direction": direction,
          "materiality": materiality, "surprise": surprise, "novelty": novelty}],
        schema={"scrip_code": pl.Int64, "disclosed_ns": pl.Int64, "direction": pl.Float64,
                "materiality": pl.Float64, "surprise": pl.Float64, "novelty": pl.Float64},
    )


def test_no_future_leak_in_session():
    day = dt.date(2024, 1, 2)  # Tuesday
    grid = synthetic_grid([500001], [day], step_min=1, n_minutes=140)
    # disclosed 11:00 IST => session_min 105; lag 1 => actionable at 11:01 (minute 106)
    anns = _one_announcement(500001, dt.datetime(2024, 1, 2, 11, 0, 0))
    out, cols = align_announcement_features(grid, anns, lag_minutes=1)
    assert cols == NEWS_FEATURES

    before = out.filter(pl.col("minute") <= 105)
    after = out.filter(pl.col("minute") >= 106)
    assert (before["has_news"] == 0.0).all()          # no bar at/before disclosure+lag
    assert (after["has_news"] == 1.0).all()
    # first actionable bar carries the signed score (age ~1 min, decay ~1)
    first = out.filter(pl.col("minute") == 106).row(0, named=True)
    assert first["news_dir"] == pytest.approx(0.8 * 1.0, abs=1e-3)


def test_after_close_informs_next_session():
    tue, wed = dt.date(2024, 1, 2), dt.date(2024, 1, 3)
    grid = synthetic_grid([500001], [tue, wed], step_min=5)
    anns = _one_announcement(500001, dt.datetime(2024, 1, 2, 18, 0, 0))  # after close
    out, _ = align_announcement_features(grid, anns, lag_minutes=1)

    assert (out.filter(pl.col("date") == tue)["has_news"] == 0.0).all()   # nothing same day
    assert (out.filter(pl.col("date") == wed)["has_news"] == 1.0).all()   # all of next day


def test_decay_is_monotone_after_event():
    day = dt.date(2024, 1, 2)
    grid = synthetic_grid([500001], [day], step_min=10, n_minutes=360)
    anns = _one_announcement(500001, dt.datetime(2024, 1, 2, 9, 20, 0))  # early, session_min 5
    out, _ = align_announcement_features(grid, anns, lag_minutes=1, half_life_min=120.0)
    active = out.filter(pl.col("has_news") == 1.0).sort("minute")["news_mat"].to_numpy()
    assert len(active) > 3
    assert (active[1:] <= active[:-1] + 1e-12).all()   # non-increasing in time


def test_no_news_cells_are_zero():
    day = dt.date(2024, 1, 2)
    grid = synthetic_grid([500001, 500002], [day], step_min=30)
    anns = _one_announcement(500001, dt.datetime(2024, 1, 2, 11, 0, 0))
    out, _ = align_announcement_features(grid, anns)
    # scrip 500002 never has a filing -> all news features zero
    other = out.filter(pl.col("scrip_code") == 500002)
    for c in NEWS_FEATURES:
        assert (other[c] == 0.0).all()


def test_max_age_cutoff_zeroes_stale_news():
    day = dt.date(2024, 1, 10)
    grid = synthetic_grid([500001], [day], step_min=30)
    # filing 5 days earlier, well beyond a 1-day cutoff
    anns = _one_announcement(500001, dt.datetime(2024, 1, 5, 11, 0, 0))
    out, _ = align_announcement_features(grid, anns, max_age_min=1440.0)
    assert (out["has_news"] == 0.0).all()


def test_grid_preserved_and_full_pipeline_runs():
    # end-to-end on the synthetic panel: synth anns -> stub scores -> align
    scrips = list(range(500001, 500006))
    dates = [dt.date(2024, 1, 2) + dt.timedelta(days=i) for i in range(10)]
    anns = synth_announcements(scrips, dates, seed=4)
    scored = attach_scores(anns, stub_extract_frame(anns))
    grid = synthetic_grid(scrips, dates, step_min=15)
    out, cols = align_announcement_features(grid, scored)
    assert out.height == grid.height                       # one row per grid cell, preserved
    assert set(["scrip_code", "date", "minute"]).issubset(out.columns)
    assert 0.0 < out["has_news"].mean() < 1.0              # sparse: some cells have news, most don't
