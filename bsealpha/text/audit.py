"""Announcement-panel audit: the Phase-1 go/no-go on data quality.

Phase 1 succeeds only if the timestamps are trustworthy and coverage is real -- so before
any LLM or model touches this text, we quantify exactly that. The audit answers:

* **Timestamps sane?** every row has a positive disclosure stamp, disclosures sort in time,
  and none is dated in the future (a future stamp = a clock bug that would leak).
* **Where in the day does news land?** the pre-open / in-session / after-close split -- this
  drives how the Phase-3 as-of join must lag (an after-close filing informs *tomorrow*).
* **Coverage real?** how many universe names ever disclose, and the per-name frequency --
  a feature present on 3% of name-days behaves very differently from one on 30%.

Everything here is descriptive; it raises nothing (validation of invariants lives in
:func:`bsealpha.text.validate_announcements`). Its job is to let you *look* before building.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import polars as pl

from .. import market
from .schema import validate_announcements


@dataclass
class AnnouncementAudit:
    """Descriptive stats for an announcement panel (see :func:`audit_announcements`)."""

    n_rows: int = 0
    n_scrips: int = 0
    date_min: object = None
    date_max: object = None
    n_trading_days_seen: int = 0

    # timestamp sanity
    ts_sorted: bool = True
    n_future_ts: int = 0
    n_nonpositive_ts: int = 0

    # time-of-day split (fractions of rows)
    pct_pre_open: float = 0.0
    pct_in_session: float = 0.0
    pct_after_close: float = 0.0
    pct_nontrading_day: float = 0.0

    # coverage vs a universe (if supplied)
    universe_size: int = 0
    n_covered: int = 0
    pct_universe_covered: float = 0.0
    per_name_mean: float = 0.0
    per_name_median: float = 0.0
    per_name_max: int = 0
    n_missing_headline: int = 0

    top_categories: list[tuple[str, int]] = field(default_factory=list)

    def report(self) -> str:
        """Human-readable one-screen report."""
        L = [
            "=== Announcement panel audit (Phase 1) ===",
            f"rows={self.n_rows:,}  scrips={self.n_scrips:,}  "
            f"dates={self.date_min}..{self.date_max}  trading-days-seen={self.n_trading_days_seen}",
            "-- timestamp sanity --",
            f"  sorted-in-time: {self.ts_sorted}   future-stamped rows: {self.n_future_ts}   "
            f"non-positive stamps: {self.n_nonpositive_ts}",
            "-- time-of-day of disclosure --",
            f"  pre-open {self.pct_pre_open:5.1%}   in-session {self.pct_in_session:5.1%}   "
            f"after-close {self.pct_after_close:5.1%}   non-trading-day {self.pct_nontrading_day:5.1%}",
            "-- coverage --",
            f"  universe {self.n_covered}/{self.universe_size} covered "
            f"({self.pct_universe_covered:5.1%})" if self.universe_size
            else f"  distinct scrips disclosing: {self.n_scrips}",
            f"  per-name filings: mean={self.per_name_mean:.2f} median={self.per_name_median:.1f} "
            f"max={self.per_name_max}   missing-headline={self.n_missing_headline}",
            "-- top categories --",
        ]
        L += [f"  {c:<28} {n:,}" for c, n in self.top_categories]
        health = "OK" if (self.ts_sorted and self.n_future_ts == 0
                          and self.n_nonpositive_ts == 0) else "CHECK TIMESTAMPS"
        L.append(f"verdict: {health}")
        return "\n".join(L)


def audit_announcements(df: pl.DataFrame, *,
                        universe: set[int] | None = None,
                        now_ns: int | None = None,
                        top_k_categories: int = 8) -> AnnouncementAudit:
    """Compute an :class:`AnnouncementAudit` over a normalized announcement panel.

    ``universe`` (a set of scrip codes) turns on coverage-vs-universe stats; ``now_ns``
    (default: wall clock) is the reference used to flag future-dated disclosures.
    """
    validate_announcements(df)
    a = AnnouncementAudit()
    if df.height == 0:
        return a
    now_ns = int(time.time_ns() if now_ns is None else now_ns)
    df = df.sort(["disclosed_ns", "scrip_code"])

    a.n_rows = df.height
    a.n_scrips = df["scrip_code"].n_unique()
    a.date_min = df["date"].min()
    a.date_max = df["date"].max()
    a.n_trading_days_seen = df["date"].n_unique()

    ns = df["disclosed_ns"].to_numpy()
    a.ts_sorted = bool((ns[1:] >= ns[:-1]).all()) if len(ns) > 1 else True
    a.n_future_ts = int((df["disclosed_ns"] > now_ns).sum())
    a.n_nonpositive_ts = int((df["disclosed_ns"] <= 0).sum())

    # time-of-day split off session_min (NaN => non-trading day)
    open_min, close_min = market.session_open_min(), market.session_close_min()
    sm = df["session_min"]
    n = df.height
    a.pct_nontrading_day = float(sm.is_nan().sum()) / n
    a.pct_pre_open = float(((sm < 0) & sm.is_not_nan()).sum()) / n
    a.pct_in_session = float(((sm >= 0) & (sm <= (close_min - open_min))).sum()) / n
    a.pct_after_close = float((sm > (close_min - open_min)).sum()) / n

    a.n_missing_headline = int((df["headline"].str.len_chars() == 0).sum())

    per_name = df.group_by("scrip_code").len()
    a.per_name_mean = float(per_name["len"].mean())
    a.per_name_median = float(per_name["len"].median())
    a.per_name_max = int(per_name["len"].max())

    if universe:
        a.universe_size = len(universe)
        covered = set(df["scrip_code"].to_list()) & universe
        a.n_covered = len(covered)
        a.pct_universe_covered = a.n_covered / a.universe_size if a.universe_size else 0.0

    cats = (df.group_by("category").len().sort("len", descending=True).head(top_k_categories))
    a.top_categories = list(zip(cats["category"].to_list(), cats["len"].to_list()))
    return a
