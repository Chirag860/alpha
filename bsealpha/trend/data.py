"""Load a daily multi-asset panel and align it to dense ``[T, N]`` matrices.

Canonical Parquet layout under ``data_dir`` (written by the MT5 daily exporter):

    daily_grid.parquet : date, symbol, asset_class, open, high, low, close
    daily_meta.parquet : symbol, asset_class, spread_bps, contract_size,
                         swap_long, swap_short, currency

Markets keep different holidays, so closes are forward-filled per instrument (a non-trading
day carries the last price -> a 0 return that day, which is correct).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

GRID_COLS = ["date", "symbol", "asset_class", "open", "high", "low", "close"]
META_COLS = ["symbol", "asset_class", "spread_bps", "contract_size",
             "swap_long", "swap_short", "currency"]


def load_daily_panel(data_dir: str | Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    data_dir = Path(data_dir)
    grid = pl.read_parquet(data_dir / "daily_grid.parquet")
    meta = pl.read_parquet(data_dir / "daily_meta.parquet")
    if grid.schema["date"] != pl.Date:
        grid = grid.with_columns(pl.col("date").cast(pl.Date))
    return grid.sort(["date", "symbol"]), meta


def _ffill_bfill(a: np.ndarray) -> np.ndarray:
    """Forward-fill down each column, then back-fill any leading gaps."""
    a = a.astype(float).copy()
    T, N = a.shape
    for j in range(N):
        last = np.nan
        for i in range(T):
            if np.isnan(a[i, j]):
                a[i, j] = last
            else:
                last = a[i, j]
        # back-fill leading NaNs with the first observed value
        first = np.nan
        for i in range(T - 1, -1, -1):
            if np.isnan(a[i, j]):
                a[i, j] = first
            else:
                first = a[i, j]
    return a


def to_matrices(panel: pl.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return ``(dates, symbols, close[T, N])`` on a dense common date grid (ffill/bfill)."""
    wide = panel.pivot(values="close", index="date", on="symbol").sort("date")
    dates = wide["date"].to_numpy()
    symbols = [c for c in wide.columns if c != "date"]
    close = _ffill_bfill(wide.drop("date").to_numpy())
    return dates, symbols, close


def simple_returns(close: np.ndarray) -> np.ndarray:
    """Simple daily returns ``[T, N]`` with row 0 = 0."""
    close = np.asarray(close, dtype=float)
    r = np.zeros_like(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        r[1:] = close[1:] / close[:-1] - 1.0
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def meta_arrays(meta: pl.DataFrame, symbols: list[str]) -> dict[str, np.ndarray]:
    """Per-symbol arrays aligned to ``symbols`` order (missing -> sensible defaults)."""
    row = {r["symbol"]: r for r in meta.iter_rows(named=True)}
    def col(name, default):
        return np.array([float(row.get(s, {}).get(name, default) or default) for s in symbols])
    return {
        "spread_bps": col("spread_bps", 1.0),
        "contract_size": col("contract_size", 1.0),
        "swap_long": col("swap_long", 0.0),
        "swap_short": col("swap_short", 0.0),
        "asset_class": np.array([str(row.get(s, {}).get("asset_class", "NA")) for s in symbols]),
    }
