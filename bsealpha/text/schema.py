"""Canonical schema for the point-in-time announcement panel.

One row per disclosed filing. Keyed on ``scrip_code`` (an int) so it joins the price
panel directly, and renamed/ticker-changed names stay stable across time (§ data.schema).

The **disclosure instant** is the whole point of Phase 1, so it is stored two ways,
both derived from BSE's dissemination datetime:

* ``disclosed_ns`` -- absolute nanoseconds since the Unix epoch, **UTC**. This is the
  point-in-time key the Phase-3 as-of join uses: a filing may only inform a minute
  whose wall-clock is strictly *after* ``disclosed_ns``. Integer-ns (never a datetime
  object) for the same reason the event streams are integer-ns -- it removes the
  timezone/float ambiguity that silently manufactures look-ahead.
* ``date`` + ``session_min`` -- the IST trading-day projection (calendar date, and
  minutes since the session open; negative before open, > session length after close,
  NaN on non-trading days). Pure functions of ``disclosed_ns`` + the active market
  profile's timezone; carried for auditing *where in the day* news lands.

``fetched_ns`` records when we pulled the row -- provenance only, **never** an input
to any feature (using it would be look-ahead of the crudest kind).
"""

from __future__ import annotations

from typing import Final

import polars as pl

ANNOUNCEMENT_SCHEMA: Final[dict[str, pl.DataType]] = {
    "scrip_code": pl.Int64,      # the join key (BSE scrip code, stable across renames)
    "ann_id": pl.Utf8,           # BSE NEWSID -- the dedup key
    "disclosed_ns": pl.Int64,    # absolute UTC nanoseconds of BSE dissemination (point-in-time)
    "date": pl.Date,             # IST calendar date of disclosure
    "session_min": pl.Float64,   # minutes since 09:15 IST on `date` (NaN if non-trading day)
    "headline": pl.Utf8,         # NEWSSUB -- the subject line
    "body": pl.Utf8,             # inline text if any (often empty; real text is in the PDF, Phase 2)
    "category": pl.Utf8,         # CATEGORYNAME
    "subcategory": pl.Utf8,      # SUBCATNAME
    "source_url": pl.Utf8,       # attachment/PDF URL, for provenance and Phase-2 text pull
    "fetched_ns": pl.Int64,      # when WE fetched it -- provenance only, never a feature
}


def empty_announcements() -> pl.DataFrame:
    """Empty announcement frame with the canonical schema."""
    return pl.DataFrame(schema=ANNOUNCEMENT_SCHEMA)


def validate_announcements(df: pl.DataFrame) -> None:
    """Assert column presence, then the point-in-time invariants Phase 1 exists to guarantee.

    Presence (not exact dtype) is checked so downstream may carry extra columns, matching
    :func:`bsealpha.data.validate_frame`. Beyond presence we enforce the invariants that make
    the panel *safe to join*: a real disclosure stamp, no duplicate ids, no null keys.
    """
    missing = [c for c in ANNOUNCEMENT_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"announcements: missing required columns {missing}")
    if df.height == 0:
        return
    if df["scrip_code"].null_count():
        raise ValueError("announcements: null scrip_code (unjoinable row)")
    if df["disclosed_ns"].null_count() or (df["disclosed_ns"] <= 0).any():
        raise ValueError("announcements: non-positive/null disclosed_ns -- no trustworthy timestamp")
    dupes = df["ann_id"].is_duplicated().sum()
    if dupes:
        raise ValueError(f"announcements: {dupes} duplicate ann_id rows -- dedup before use")
