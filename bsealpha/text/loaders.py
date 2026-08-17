"""Announcement loaders: live BSE fetch, Parquet round-trip, and an offline synth source.

Three sources, one canonical output (mirrors :mod:`bsealpha.data.loaders`):

* :class:`BseAnnouncementsClient` -- pulls the BSE corporate-announcements feed for a
  date range, caches the **raw JSON per day** to disk (an immutable point-in-time
  archive we can re-parse and audit), and normalizes to :data:`ANNOUNCEMENT_SCHEMA`.
* :class:`ParquetAnnouncementLoader` / :func:`announcements_to_parquet` -- persist and
  reload the normalized panel, exactly like :class:`~bsealpha.data.ParquetLoader`.
* :func:`synth_announcements` -- a deterministic offline generator so the whole text
  pipeline (and CI) runs with zero network, matching the synthetic-panel philosophy of
  the price layer.

The disclosure instant is taken from BSE's **dissemination** datetime (``DissemDT`` --
when the filing went public), falling back to ``NEWS_DT``. It is parsed as naive IST and
converted to an absolute UTC nanosecond stamp via the active market profile's timezone,
so repointing the venue (§ market profiles) carries the text layer with it.

Network note: the live client uses only the stdlib (no new dependency) and defensive
headers, because BSE rejects non-browser callers. It is not exercised in CI -- the
offline synth path and the audit are what tests cover.
"""

from __future__ import annotations

import datetime as dt
import json
import ssl
import time
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from .. import market
from .schema import ANNOUNCEMENT_SCHEMA, validate_announcements

# BSE's public announcements JSON endpoint. `strCat=-1`/`subcategory=-1` = all
# (sub)categories; `strType=C` = company announcements; dates are YYYYMMDD; `pageno` is
# 1-based and each page holds up to 50 rows; the day's total is in `Table1[0].ROWCNT`.
_BSE_ANN_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
_BSE_PAGE_SIZE = 50
_BSE_HEADERS = {
    # BSE 403s anything that does not look like a browser hitting bseindia.com.
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.bseindia.com/",
    "Origin": "https://www.bseindia.com",
}
# BSE datetime strings seen in the feed, most-specific first.
_DT_FORMATS = ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
               "%d %b %Y %H:%M:%S", "%Y-%m-%d %H:%M:%S")


@lru_cache(maxsize=1)
def _ssl_context() -> ssl.SSLContext:
    """A verifying TLS context backed by certifi's CA bundle if present.

    The stock macOS/framework Python ships without a usable system trust store, so
    ``urlopen`` fails cert verification against BSE even though the host is reachable.
    Prefer certifi's bundle; fall back to the default context (still verifying) elsewhere.
    Verification is never disabled -- a silent MITM downgrade is worse than a hard failure.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # certifi absent -> platform default (may work on properly-configured hosts)
        return ssl.create_default_context()


# --------------------------------------------------------------------- time helpers
def _ist_naive_to_fields(naive: dt.datetime) -> tuple[int, dt.date, float]:
    """Map a naive IST disclosure datetime to ``(disclosed_ns_utc, date_ist, session_min)``.

    ``session_min`` is minutes since the active profile's session open on the IST calendar
    day; it is negative for pre-open disclosures and exceeds the session length for
    after-close ones (both legitimate -- news does not respect market hours).
    """
    tz = ZoneInfo(market.active_profile().tz)
    aware = naive.replace(tzinfo=tz)
    disclosed_ns = int(aware.timestamp() * 1_000_000_000)
    date_ist = aware.date()
    session_min = (aware.hour * 60 + aware.minute + aware.second / 60.0
                   - market.session_open_min())
    return disclosed_ns, date_ist, float(session_min)


def _parse_dt(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    s = s.strip()
    for fmt in _DT_FORMATS:
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _normalize_row(row: dict, fetched_ns: int) -> dict | None:
    """Normalize one raw BSE row to the announcement schema, or ``None`` if unusable.

    Unusable = missing scrip code or an unparseable disclosure timestamp; such rows are
    dropped rather than admitted with a fabricated time (which would poison the as-of join).
    """
    raw_scrip = row.get("SCRIP_CD") or row.get("scrip_cd")
    disclosed = _parse_dt(row.get("DissemDT") or row.get("NEWS_DT") or row.get("News_submission_dt"))
    if raw_scrip in (None, "", 0) or disclosed is None:
        return None
    disclosed_ns, date_ist, session_min = _ist_naive_to_fields(disclosed)
    attach = (row.get("ATTACHMENTNAME") or "").strip()
    url = f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{attach}" if attach else ""
    news_id = str(row.get("NEWSID") or row.get("NEWS_ID") or f"{raw_scrip}-{disclosed_ns}")
    return {
        "scrip_code": int(raw_scrip),
        "ann_id": news_id,
        "disclosed_ns": disclosed_ns,
        "date": date_ist,
        "session_min": session_min,
        "headline": (row.get("NEWSSUB") or row.get("HEADLINE") or "").strip(),
        "body": (row.get("MORE") or "").strip(),
        "category": (row.get("CATEGORYNAME") or "").strip(),
        "subcategory": (row.get("SUBCATNAME") or "").strip(),
        "source_url": url,
        "fetched_ns": fetched_ns,
    }


def _frame_from_rows(rows: list[dict]) -> pl.DataFrame:
    """Build a deduplicated, time-sorted announcement frame from normalized rows."""
    if not rows:
        from .schema import empty_announcements
        return empty_announcements()
    df = pl.DataFrame(rows, schema=ANNOUNCEMENT_SCHEMA)
    return (df.unique(subset=["ann_id"], keep="first")
            .sort(["disclosed_ns", "scrip_code"]))


# --------------------------------------------------------------------- live BSE client
class BseAnnouncementsClient:
    """Fetch BSE corporate announcements for a date range, cached raw per day.

    Parameters
    ----------
    cache_dir:
        Directory for the immutable raw-JSON archive (one file per calendar day). Re-runs
        read the cache instead of re-hitting BSE, which keeps the fetch reproducible and
        gives an audit trail of exactly what the feed said on each day.
    pause_s:
        Politeness delay between HTTP calls.
    """

    def __init__(self, cache_dir: str | Path, *, pause_s: float = 0.4) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pause_s = float(pause_s)

    # -- raw layer (network) ------------------------------------------------
    def _raw_day(self, day: dt.date) -> list[dict]:
        """Return the raw BSE rows for one day, from cache or a paged live fetch."""
        cache = self.cache_dir / f"{day:%Y%m%d}.json"
        if cache.exists():
            return json.loads(cache.read_text())
        rows: list[dict] = []
        page = 1
        while True:  # pragma: no cover - network path, not exercised in CI
            payload = self._fetch_page(day, page)
            # empty days come back as the bare JSON string "No Record Found!", not an object
            if not isinstance(payload, dict):
                break
            table = payload.get("Table") or []
            rows.extend(table)
            total = (payload.get("Table1") or [{}])[0].get("ROWCNT")
            done = (total is not None and len(rows) >= int(total)) or len(table) < _BSE_PAGE_SIZE
            if not table or done:
                break
            page += 1
            time.sleep(self.pause_s)
        cache.write_text(json.dumps(rows))
        return rows

    def _fetch_page(self, day: dt.date, page: int):  # pragma: no cover - network
        params = (f"pageno={page}&strCat=-1&subcategory=-1&strPrevDate={day:%Y%m%d}"
                  f"&strToDate={day:%Y%m%d}&strSearch=P&strscrip=&strType=C")
        req = urllib.request.Request(f"{_BSE_ANN_URL}?{params}", headers=_BSE_HEADERS)
        with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # -- normalized layer ---------------------------------------------------
    def fetch(self, start: dt.date, end: dt.date, *,
              scrip_codes: set[int] | None = None) -> pl.DataFrame:
        """Fetch and normalize announcements for ``[start, end]`` (inclusive).

        ``scrip_codes`` optionally restricts to a universe (kept as ints, per §1.4). The
        result is deduped on ``ann_id`` and sorted by disclosure time.
        """
        fetched_ns = time.time_ns()
        rows: list[dict] = []
        day = start
        while day <= end:
            for raw in self._raw_day(day):
                norm = _normalize_row(raw, fetched_ns)
                if norm is None:
                    continue
                if scrip_codes is not None and norm["scrip_code"] not in scrip_codes:
                    continue
                rows.append(norm)
            day += dt.timedelta(days=1)
        df = _frame_from_rows(rows)
        validate_announcements(df)
        return df


# --------------------------------------------------------------------- parquet round-trip
def announcements_to_parquet(df: pl.DataFrame, path: str | Path) -> Path:
    """Persist a normalized announcement panel to Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_announcements(df)
    df.write_parquet(path)
    return path


class ParquetAnnouncementLoader:
    """Load a normalized announcement panel from a Parquet file (mirrors ParquetLoader)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> pl.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"expected {self.path}")
        df = pl.read_parquet(self.path)
        validate_announcements(df)
        return df.sort(["disclosed_ns", "scrip_code"])


# --------------------------------------------------------------------- offline synth source
_SYNTH_CATS = [
    ("Result", "Financial Results"),
    ("Board Meeting", "Board Meeting"),
    ("Company Update", "Award of Order / Receipt of Order"),
    ("Corp. Action", "Dividend"),
    ("AGM/EGM", "AGM"),
    ("Credit Rating", "Rating"),
]


def synth_announcements(scrip_codes: list[int], dates: list[dt.date], *,
                        seed: int = 0, rate_per_name_day: float = 0.12,
                        dup_frac: float = 0.03) -> pl.DataFrame:
    """Deterministic offline announcement panel for tests/CI (no network).

    Emits a **sparse**, Poisson-ish stream (most name-days have no filing, matching reality),
    with disclosures spread across pre-open, in-session, and after-close so the audit's
    time-of-day breakdown has something to measure. A small ``dup_frac`` re-emits some
    ``ann_id``s to exercise dedup. This is *plumbing* test data -- no signal is embedded.
    """
    rng = np.random.default_rng(seed)
    tz = ZoneInfo(market.active_profile().tz)
    rows: list[dict] = []
    fetched_ns = time.time_ns()
    for day in dates:
        if day.weekday() >= 5:  # skip weekends -- BSE does not disseminate company news then
            continue
        for scrip in scrip_codes:
            if rng.random() >= rate_per_name_day:
                continue
            # disclosure time: mostly in/around the session, some pre-open & after-close
            hour = int(rng.choice([8, 9, 11, 13, 15, 17, 19], p=[.08, .12, .2, .2, .2, .12, .08]))
            minute = int(rng.integers(0, 60))
            naive = dt.datetime(day.year, day.month, day.day, hour, minute, int(rng.integers(0, 60)))
            disclosed_ns, date_ist, session_min = _ist_naive_to_fields(naive.replace(tzinfo=None))
            cat, sub = _SYNTH_CATS[int(rng.integers(0, len(_SYNTH_CATS)))]
            ann_id = f"SYN-{scrip}-{day:%Y%m%d}-{hour:02d}{minute:02d}"
            rows.append({
                "scrip_code": int(scrip), "ann_id": ann_id, "disclosed_ns": disclosed_ns,
                "date": date_ist, "session_min": session_min,
                "headline": f"{cat}: {sub} for scrip {scrip}", "body": "",
                "category": cat, "subcategory": sub,
                "source_url": f"https://example.invalid/{ann_id}.pdf", "fetched_ns": fetched_ns,
            })
    # inject duplicate ids (kept once by _frame_from_rows) to exercise dedup
    if rows and dup_frac > 0:
        k = max(1, int(len(rows) * dup_frac))
        for r in list(rng.choice(rows, size=min(k, len(rows)), replace=False)):
            rows.append(dict(r))
    return _frame_from_rows(rows)
