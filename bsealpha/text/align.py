"""Phase 3: align sparse announcement scores onto the ``(scrip_code, date, minute)`` grid.

The whole point is to attach news to bars **without leaking the future**. Rather than
special-casing in-session / pre-open / after-close / weekend filings, we do one thing: give
every grid cell its absolute wall-clock instant (IST ``09:15 + minute`` on ``date``, in UTC
ns) and every filing its *actionable* instant (``disclosed_ns`` plus a ``lag`` so a live
system has time to react), then a **backward as-of join** attaches, to each cell, the most
recent filing whose actionable instant is ``<= `` the cell's instant.

That single comparison is correct for all cases at once (this is why absolute-ns beats
per-case logic):

* in-session filing at 11:00 informs the 11:01+ bars of the same day;
* an after-close or weekend filing has no same-day cell at or after it, so it attaches to
  the **next session's** bars from the open -- exactly the ~69% of real BSE filings that
  land after close;
* nothing attaches to a bar earlier than the filing's actionable instant -- no look-ahead,
  enforced by the join, matching the §3.3 discipline the price features already follow.

Scores decay with age (news gets stale), and cells with no recent filing get zeros -- the
feature is **sparse and mostly zero**, active only around events. It is advisory only: it
becomes columns the validated model *may* use, and must still earn its place in
:mod:`bsealpha.validation`.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from .. import market

_SCORES = ("direction", "materiality", "surprise", "novelty")
# columns this module appends to the grid
NEWS_FEATURES = ["news_dir", "news_mat", "news_surprise", "news_novelty", "has_news"]

_NS_PER_MIN = 60_000_000_000


def _cell_ns_expr() -> pl.Expr:
    """Absolute UTC-ns instant of each grid cell from its IST ``(date, minute)`` wall-clock.

    ``minute`` is minutes since the session open (the grid convention, see features.engine),
    so the wall-clock is ``date @ (session_open + minute)`` in the active profile's tz.
    ``replace_time_zone`` interprets the naive stamp as IST; ``epoch("ns")`` yields UTC ns,
    directly comparable to the announcement panel's ``disclosed_ns``.
    """
    tz = market.active_profile().tz
    open_min = market.session_open_min()
    local = (pl.col("date").cast(pl.Datetime("ns"))
             + pl.duration(minutes=pl.col("minute") + open_min))
    return local.dt.replace_time_zone(tz).dt.epoch(time_unit="ns").alias("cell_ns")


def align_announcement_features(
    grid: pl.DataFrame,
    scored_anns: pl.DataFrame,
    *,
    lag_minutes: int = 1,
    half_life_min: float = 720.0,
    max_age_min: float = 2880.0,
    fill: float = 0.0,
) -> tuple[pl.DataFrame, list[str]]:
    """Attach decayed announcement features to ``grid``; return ``(grid, NEWS_FEATURES)``.

    Parameters
    ----------
    grid:
        Must carry ``scrip_code``, ``date``, ``minute`` (the cross-sectional panel).
    scored_anns:
        Announcement panel joined to its scores -- ``scrip_code``, ``disclosed_ns`` and the
        four score columns (``direction``/``materiality``/``surprise``/``novelty``).
    lag_minutes:
        Reaction lag added to each disclosure before it may inform a bar (§3.3 discipline).
    half_life_min:
        Wall-clock half-life of the decay; long enough that an after-close filing still
        carries weight at next open, short enough that intraday news fades within the day.
    max_age_min:
        Filings older than this (wall-clock) are treated as no-news, keeping the feature
        sparse. With the default half-life their decayed weight is already negligible.
    """
    grid = grid.with_columns(_cell_ns_expr()).sort(["scrip_code", "cell_ns"])

    lag_ns = int(lag_minutes) * _NS_PER_MIN
    right = (
        scored_anns
        .select(
            "scrip_code",
            (pl.col("disclosed_ns") + lag_ns).alias("lagged_ns"),
            pl.col("disclosed_ns").alias("anchor_ns"),
            *_SCORES,
        )
        .sort(["scrip_code", "lagged_ns"])
    )

    joined = grid.join_asof(
        right, left_on="cell_ns", right_on="lagged_ns", by="scrip_code",
        strategy="backward", allow_exact_matches=True,
    )

    age_min = (pl.col("cell_ns") - pl.col("anchor_ns")) / _NS_PER_MIN
    # valid = a filing matched AND it is not older than the cutoff
    valid = pl.col("anchor_ns").is_not_null() & (age_min <= float(max_age_min))
    decay = pl.when(valid).then((0.5 ** (age_min / float(half_life_min)))).otherwise(0.0)

    joined = joined.with_columns(
        (pl.col("direction").fill_null(0.0) * pl.col("materiality").fill_null(0.0) * decay).alias("news_dir"),
        (pl.col("materiality").fill_null(0.0) * decay).alias("news_mat"),
        (pl.col("surprise").fill_null(0.0) * decay).alias("news_surprise"),
        (pl.col("novelty").fill_null(0.0) * decay).alias("news_novelty"),
        pl.when(valid).then(1.0).otherwise(0.0).alias("has_news"),
    )

    out = joined.drop(["cell_ns", "lagged_ns", "anchor_ns", *_SCORES])
    out = out.with_columns([pl.col(c).fill_null(fill) for c in NEWS_FEATURES])
    return out.sort(["date", "minute", "scrip_code"]), list(NEWS_FEATURES)


def attach_scores(anns: pl.DataFrame, scores: pl.DataFrame) -> pl.DataFrame:
    """Join per-``ann_id`` scores back onto the announcement panel (keeps ``scrip_code``/``disclosed_ns``)."""
    return anns.join(scores, on="ann_id", how="inner")


def synthetic_grid(scrips: list[int], dates: list[dt.date], *, step_min: int = 5,
                   n_minutes: int | None = None) -> pl.DataFrame:
    """A bare ``(scrip_code, date, minute)`` grid for demos/tests (no price features).

    Covers every ``step_min``-th minute of each session for each scrip -- enough to exercise
    the alignment without building the full microstructure panel.
    """
    span = n_minutes if n_minutes is not None else market.session_len_min()
    minutes = list(range(0, int(span), int(step_min)))
    return pl.DataFrame(
        [{"scrip_code": s, "date": d, "minute": m}
         for d in dates for m in minutes for s in scrips],
        schema={"scrip_code": pl.Int64, "date": pl.Date, "minute": pl.Int64},
    ).sort(["date", "minute", "scrip_code"])
