"""Canonical panel data schemas.

Two clocks live side by side (§2.2):

* **Per-name event stream** (``DEPTH_SCHEMA`` + ``TRADE_SCHEMA``): 5-level depth
  snapshots and trade prints, in *local-receipt* order, one stream per scrip.
* **Common time grid** (``BAR_SCHEMA``): rupee/time bars aligned so cross-sectional
  operations (residualize, rank, portfolio) have a shared minute index.

Everything keys on ``scrip_code`` (an int), never on the ticker string, because BSE
scrip codes are stable across renames while symbols are not (§1.4).

Timestamps are integer **nanoseconds since the session-day epoch** for the event
streams, and integer **minute-of-day** on the common grid. Keeping them integers
avoids the float/゚datetime ambiguity that silently creates look-ahead.
"""

from __future__ import annotations

from typing import Final

import polars as pl

N_DEPTH_LEVELS: Final[int] = 5

# ---------------------------------------------------------------- event streams
# 5-level aggregated depth snapshot (what a retail broker feed actually gives, §3.1).
DEPTH_SCHEMA: Final[dict[str, pl.DataType]] = {
    "scrip_code": pl.Int64,
    "ts_ns": pl.Int64,           # local-receipt nanoseconds within the session day
    "date": pl.Date,
    "session_min": pl.Float64,   # minutes since 09:15 (can be fractional)
    **{f"bid_px_{i}": pl.Float64 for i in range(N_DEPTH_LEVELS)},
    **{f"bid_qty_{i}": pl.Float64 for i in range(N_DEPTH_LEVELS)},
    **{f"ask_px_{i}": pl.Float64 for i in range(N_DEPTH_LEVELS)},
    **{f"ask_qty_{i}": pl.Float64 for i in range(N_DEPTH_LEVELS)},
}

# Trade prints. India publishes NO aggressor flag (§3.2), so `sign` is *estimated*
# downstream by the tick rule; the generator stores the true sign only for tests.
TRADE_SCHEMA: Final[dict[str, pl.DataType]] = {
    "scrip_code": pl.Int64,
    "ts_ns": pl.Int64,
    "date": pl.Date,
    "session_min": pl.Float64,
    "price": pl.Float64,
    "qty": pl.Float64,
    "true_sign": pl.Int8,        # +1 buyer-initiated, -1 seller-initiated (test-only)
}

# ---------------------------------------------------------------- common grid
BAR_SCHEMA: Final[dict[str, pl.DataType]] = {
    "scrip_code": pl.Int64,
    "date": pl.Date,
    "minute": pl.Int64,          # minute-of-day on the common grid
    "session_min": pl.Float64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,         # mid-price close (label price base, §2.4)
    "vwap": pl.Float64,
    "turnover": pl.Float64,      # rupee turnover in the bar
    "n_trades": pl.Int64,
    "mid": pl.Float64,
    "micro": pl.Float64,
}

# ---------------------------------------------------------------- reference data
# EOD panel used by the point-in-time universe screen (§1.2). One row per
# (date, scrip_code). Surveillance flags are as-known on the *morning* of `date`.
DAILY_SCHEMA: Final[dict[str, pl.DataType]] = {
    "date": pl.Date,
    "scrip_code": pl.Int64,
    "symbol": pl.Utf8,
    "sector": pl.Utf8,
    "close": pl.Float64,
    "bse_turnover": pl.Float64,
    "bse_trades": pl.Int64,
    "median_spread_bps": pl.Float64,
    "series": pl.Utf8,           # A / B / T (T2T) / Z / M (SME) ...
    "asm_flag": pl.Boolean,
    "gsm_flag": pl.Boolean,
    "t2t_flag": pl.Boolean,
    "is_suspended": pl.Boolean,
    "circuit_band_pct": pl.Float64,
    "adj_factor": pl.Float64,    # cumulative corporate-action adjustment (§1.4)
}


def empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Return an empty :class:`polars.DataFrame` with the given schema."""
    return pl.DataFrame(schema=schema)


def validate_frame(df: pl.DataFrame, schema: dict[str, pl.DataType], *, name: str) -> None:
    """Assert that ``df`` contains every column in ``schema``.

    We check column *presence* (not exact dtype equality) so upstream code may carry
    extra feature columns, but a missing required column fails loudly rather than
    surfacing as a silent NaN later.
    """
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")
