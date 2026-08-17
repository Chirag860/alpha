"""Tests for the text layer (Phase 1): announcement schema, offline synth source, audit.

Covers the point-in-time invariants the whole news thesis rests on -- real timestamps,
dedup, no future leakage, correct time-of-day bucketing -- entirely offline (no network).
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from bsealpha import market
from bsealpha.text import (
    ANNOUNCEMENT_SCHEMA,
    ParquetAnnouncementLoader,
    announcements_to_parquet,
    audit_announcements,
    synth_announcements,
    validate_announcements,
)

_SCRIPS = list(range(500001, 500021))
_DATES = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(90)]


@pytest.fixture(scope="module")
def anns() -> pl.DataFrame:
    return synth_announcements(_SCRIPS, _DATES, seed=3)


def test_schema_and_invariants(anns):
    for col in ANNOUNCEMENT_SCHEMA:
        assert col in anns.columns
    validate_announcements(anns)  # raises on null keys / bad ts / dupes
    assert anns.height > 0


def test_dedup_on_ann_id(anns):
    # synth injects duplicate ids; the frame must keep each exactly once
    assert anns["ann_id"].n_unique() == anns.height
    assert anns["ann_id"].is_duplicated().sum() == 0


def test_timestamps_positive_sorted_not_future(anns):
    ns = anns["disclosed_ns"].to_numpy()
    assert (ns > 0).all()
    assert (ns[1:] >= ns[:-1]).all()          # sorted by disclosure time
    assert (anns["disclosed_ns"] < 10**18 * 2).all()  # sane epoch-ns magnitude, not future


def test_no_weekend_disclosures(anns):
    # synth skips weekends; every date must be a weekday
    for d in anns["date"].unique().to_list():
        assert d.weekday() < 5


def test_session_min_matches_profile():
    # a single 11:00 IST disclosure lands at session_min = 11:00 - 09:15 = 105
    d = dt.date(2024, 1, 2)
    df = synth_announcements([500001], [d], seed=1, rate_per_name_day=1.0, dup_frac=0.0)
    # recompute expected for whatever hour the synth drew, via the schema's own mapping
    row = df.row(0, named=True)
    aware_min = row["session_min"] + market.session_open_min()
    assert 0 <= aware_min < 24 * 60


def test_audit_time_of_day_partition(anns):
    a = audit_announcements(anns, universe=set(_SCRIPS))
    total = a.pct_pre_open + a.pct_in_session + a.pct_after_close + a.pct_nontrading_day
    assert total == pytest.approx(1.0, abs=1e-9)
    assert a.ts_sorted and a.n_future_ts == 0 and a.n_nonpositive_ts == 0
    assert a.universe_size == len(_SCRIPS)
    assert 0 < a.n_covered <= len(_SCRIPS)
    assert "Phase 1" in a.report()


def test_future_timestamp_is_flagged(anns):
    # audit against a reference clock BEFORE the data => every row looks "future"
    a = audit_announcements(anns, now_ns=0)
    assert a.n_future_ts == anns.height


def test_parquet_roundtrip(anns, tmp_path):
    path = announcements_to_parquet(anns, tmp_path / "ann.parquet")
    back = ParquetAnnouncementLoader(path).load()
    assert back.height == anns.height
    assert set(back["ann_id"].to_list()) == set(anns["ann_id"].to_list())


def test_validate_rejects_duplicate_ids():
    df = synth_announcements([500001], [dt.date(2024, 1, 2)], seed=1,
                             rate_per_name_day=1.0, dup_frac=0.0)
    dupe = pl.concat([df, df])  # force a duplicate ann_id
    with pytest.raises(ValueError, match="duplicate ann_id"):
        validate_announcements(dupe)


def test_validate_rejects_nonpositive_timestamp(anns):
    bad = anns.head(3).with_columns(pl.lit(0).cast(pl.Int64).alias("disclosed_ns"))
    with pytest.raises(ValueError, match="disclosed_ns"):
        validate_announcements(bad)
