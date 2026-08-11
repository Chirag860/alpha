"""Vendor -> canonical adapter (§10 data layer).

Turns a SEBI-authorized vendor's files (TrueData / GDFL / a generic CSV or Parquet) into
the canonical :mod:`bsealpha.data.schema` frames the whole pipeline consumes, so switching
from synthetic to real data touches only this module.

Mapping is config-driven via :class:`VendorSpec` (canonical column name -> vendor column
name) so you never hand-edit the pipeline. Presets are best-effort starting points --
**verify the exact column names and units against your subscription**, they differ by
vendor and plan.

Reality note (§3.1): most *retail* vendor products give 1-minute OHLCV bars and, separately,
5-level depth snapshots -- not a full trade tape. If you only have bars, use
:func:`minute_bars_to_grid` (microstructure features that need depth/trades are then
unavailable and the feature engine runs a reduced set); if you have depth + trades, map them
to ``DEPTH_SCHEMA`` / ``TRADE_SCHEMA`` and the full pipeline runs unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from .. import market
from .schema import DAILY_SCHEMA, N_DEPTH_LEVELS


@dataclass
class VendorSpec:
    """Column mapping and parsing rules from a vendor file to canonical columns.

    ``column_map`` maps canonical name -> vendor name. ``timestamp_col`` is parsed to a
    datetime with ``timestamp_fmt`` (or left as-is if already a datetime); the IST
    minute-of-day drives ``session_min`` / ``minute``.
    """

    column_map: dict[str, str] = field(default_factory=dict)
    timestamp_col: str = "timestamp"
    timestamp_fmt: str | None = "%Y-%m-%d %H:%M:%S"
    symbol_to_scrip: dict[str, int] = field(default_factory=dict)

    def rename(self, df: pl.DataFrame) -> pl.DataFrame:
        inv = {v: k for k, v in self.column_map.items() if v in df.columns}
        return df.rename(inv)


TRUEDATA_BARS_SPEC = VendorSpec(
    column_map={
        "symbol": "symbol", "open": "open", "high": "high", "low": "low",
        "close": "close", "turnover": "value", "n_trades": "trades", "volume": "volume",
    },
    timestamp_col="timestamp",
    timestamp_fmt="%Y-%m-%dT%H:%M:%S",
)
"""Best-effort TrueData 1-min bars preset. Verify column names against your feed."""


def _add_time_columns(df: pl.DataFrame, spec: VendorSpec) -> pl.DataFrame:
    ts = pl.col(spec.timestamp_col)
    if df.schema.get(spec.timestamp_col) == pl.Utf8 and spec.timestamp_fmt:
        ts = ts.str.to_datetime(spec.timestamp_fmt)
    df = df.with_columns(ts.alias("_ts"))
    return df.with_columns(
        pl.col("_ts").dt.date().alias("date"),
        (pl.col("_ts").dt.hour().cast(pl.Int32) * 60
         + pl.col("_ts").dt.minute().cast(pl.Int32)).alias("_mod"),
    ).with_columns(
        (pl.col("_mod") - market.session_open_min()).cast(pl.Float64).alias("session_min"),
        (pl.col("_mod") - market.session_open_min()).cast(pl.Int64).alias("minute"),
    )


def _resolve_scrip(df: pl.DataFrame, spec: VendorSpec) -> pl.DataFrame:
    if "scrip_code" in df.columns:
        return df
    if "symbol" in df.columns and spec.symbol_to_scrip:
        return df.with_columns(
            pl.col("symbol").replace_strict(spec.symbol_to_scrip, default=-1)
            .cast(pl.Int64).alias("scrip_code")
        )
    raise ValueError("vendor frame lacks scrip_code and no symbol_to_scrip map was given")


def minute_bars_to_grid(path: str | Path, spec: VendorSpec) -> pl.DataFrame:
    """Load vendor 1-min bars into a common-grid frame (§2.2).

    Produces ``scrip_code, date, minute, session_min, open/high/low/close, vwap, turnover,
    n_trades, mid, micro``. Without depth, ``mid = micro = close`` (a documented reduced
    fidelity; OFI/book features are unavailable).
    """
    df = pl.read_parquet(path) if str(path).endswith(".parquet") else pl.read_csv(path)
    df = spec.rename(df)
    df = _add_time_columns(df, spec)
    df = _resolve_scrip(df, spec)
    if "vwap" not in df.columns:
        df = df.with_columns(pl.col("close").alias("vwap"))
    if "n_trades" not in df.columns:
        df = df.with_columns(pl.lit(0, dtype=pl.Int64).alias("n_trades"))
    df = df.with_columns(
        pl.col("close").alias("mid"),
        pl.col("close").alias("micro"),
    )
    keep = ["scrip_code", "date", "minute", "session_min", "open", "high", "low",
            "close", "vwap", "turnover", "n_trades", "mid", "micro"]
    return df.select([c for c in keep if c in df.columns]).sort(["date", "minute", "scrip_code"])


def load_vendor_depth(path: str | Path, spec: VendorSpec,
                      m: int = N_DEPTH_LEVELS) -> pl.DataFrame:
    """Load vendor 5-level depth snapshots into ``DEPTH_SCHEMA`` shape.

    Expects ``column_map`` to cover ``bid_px_i/bid_qty_i/ask_px_i/ask_qty_i`` for
    ``i in [0, m)`` plus the timestamp and symbol/scrip columns.
    """
    df = pl.read_parquet(path) if str(path).endswith(".parquet") else pl.read_csv(path)
    df = spec.rename(df)
    df = _add_time_columns(df, spec)
    df = _resolve_scrip(df, spec)
    df = df.with_columns((pl.col("_ts").dt.timestamp("ns")).alias("ts_ns"))
    level_cols = []
    for i in range(m):
        level_cols += [f"bid_px_{i}", f"bid_qty_{i}", f"ask_px_{i}", f"ask_qty_{i}"]
    missing = [c for c in level_cols if c not in df.columns]
    if missing:
        raise ValueError(f"vendor depth missing level columns: {missing}")
    return df.select(["scrip_code", "ts_ns", "date", "session_min", *level_cols]).sort(
        ["date", "scrip_code", "ts_ns"]
    )


def load_vendor_daily(path: str | Path, spec: VendorSpec) -> pl.DataFrame:
    """Load a vendor / BSE-bhavcopy EOD file into ``DAILY_SCHEMA`` shape.

    Fills missing surveillance/flag columns with safe defaults so the universe screen can
    run; replace these with the real BSE circular flags in production (§1.3).
    """
    df = pl.read_parquet(path) if str(path).endswith(".parquet") else pl.read_csv(path)
    df = spec.rename(df)
    if "date" not in df.columns:
        df = _add_time_columns(df, spec).drop(["_ts", "_mod", "session_min", "minute"])
    df = _resolve_scrip(df, spec)
    defaults = {
        "sector": pl.lit("UNKNOWN"), "series": pl.lit("A"),
        "asm_flag": pl.lit(False), "gsm_flag": pl.lit(False), "t2t_flag": pl.lit(False),
        "is_suspended": pl.lit(False), "circuit_band_pct": pl.lit(20.0),
        "adj_factor": pl.lit(1.0), "median_spread_bps": pl.lit(5.0),
        "bse_trades": pl.lit(0, dtype=pl.Int64),
    }
    add = [expr.alias(c) for c, expr in defaults.items() if c not in df.columns]
    df = df.with_columns(add)
    keep = list(DAILY_SCHEMA.keys())
    have = [c for c in keep if c in df.columns]
    return df.select(have)
