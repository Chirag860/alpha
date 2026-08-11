"""Load an MT5-exported stock-CFD panel into the canonical ``(grid, meta)`` shape.

The Windows-only :mod:`bsealpha.data.mt5_export` script pulls M1 bars from a running MT5
terminal and writes them as Parquet in the **same schema** the yfinance free path produces
(:func:`bsealpha.data.free_data.load_yfinance_panel`). This loader reads that Parquet back --
it has **no** ``MetaTrader5`` dependency, so it runs anywhere (macOS research/training side).

Layout under ``data_dir``::

    <data_dir>/grid.parquet    # the 1-minute bar grid (GRID_COLUMNS below)
    <data_dir>/meta.parquet    # per-name static metadata (META_COLUMNS below)

``minute`` is session-relative and is computed by the exporter against the active market
profile's session open (US equities: 09:30 ET), exactly as the yfinance loader does for BSE.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# Canonical schema, mirroring free_data.load_yfinance_panel's output.
GRID_COLUMNS = ["scrip_code", "date", "minute", "session_min", "open", "high",
                "low", "close", "vwap", "turnover", "n_trades", "mid", "micro"]
META_COLUMNS = ["scrip_code", "symbol", "sector", "beta", "circuit_band_pct"]


def load_mt5_panel(data_dir: str | Path = "data/mt5") -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read an exported MT5 panel and return ``(grid, meta)`` in canonical shape.

    Parameters
    ----------
    data_dir
        Directory holding ``grid.parquet`` and ``meta.parquet`` (written by
        :mod:`bsealpha.data.mt5_export` on the Windows VM).

    Returns
    -------
    ``(grid, meta)`` polars frames matching :func:`load_yfinance_panel`'s contract, so every
    downstream stage (``build_features_bars_only`` -> labeling -> backtest/paper) is unchanged.

    Raises
    ------
    FileNotFoundError
        If either Parquet file is missing.
    ValueError
        If a required column is absent (schema drift from the exporter).
    """
    data_dir = Path(data_dir)
    grid_path = data_dir / "grid.parquet"
    meta_path = data_dir / "meta.parquet"
    if not grid_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"expected {grid_path} and {meta_path}. Run bsealpha/data/mt5_export.py on the "
            "MT5 (Windows) side first to export the panel.")

    grid = pl.read_parquet(grid_path)
    meta = pl.read_parquet(meta_path)
    _require(grid, GRID_COLUMNS, "grid")
    _require(meta, META_COLUMNS, "meta")

    # normalize dtypes to match the yfinance path (date as pl.Date, sorted key order)
    if grid.schema["date"] != pl.Date:
        grid = grid.with_columns(pl.col("date").cast(pl.Date))
    grid = grid.sort(["date", "minute", "scrip_code"])
    return grid, meta


def _require(df: pl.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"MT5 {name} frame is missing columns {missing}; "
                         f"got {df.columns}. Re-export with the current mt5_export.py.")
