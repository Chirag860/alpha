"""load_mt5_panel tests: reads exported Parquet into the canonical (grid, meta) shape.

Runs on any platform (no MetaTrader5). We synthesize a tiny canonical export and assert the
loader returns frames matching load_yfinance_panel's contract, and that schema/file errors
surface clearly.
"""

from __future__ import annotations

import polars as pl
import pytest

from bsealpha.data import GRID_COLUMNS, META_COLUMNS, load_mt5_panel


def _write_export(tmp_path):
    grid = pl.DataFrame({
        "scrip_code": [101, 102, 101, 102],
        "date": ["2025-06-02", "2025-06-02", "2025-06-02", "2025-06-02"],
        "minute": [0, 0, 1, 1],
        "session_min": [0.0, 0.0, 1.0, 1.0],
        "open": [100.0, 200.0, 100.1, 200.2],
        "high": [100.2, 200.3, 100.3, 200.4],
        "low": [99.9, 199.8, 100.0, 200.0],
        "close": [100.1, 200.2, 100.2, 200.3],
        "vwap": [100.05, 200.1, 100.15, 200.25],
        "turnover": [1.0e6, 2.0e6, 1.1e6, 2.1e6],
        "n_trades": [0, 0, 0, 0],
        "mid": [100.1, 200.2, 100.2, 200.3],
        "micro": [100.1, 200.2, 100.2, 200.3],
    }).with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
    meta = pl.DataFrame({
        "scrip_code": [101, 102],
        "symbol": ["AAPL", "MSFT"],
        "sector": ["TECH", "TECH"],
        "beta": [1.0, 1.0],
        "circuit_band_pct": [20.0, 20.0],
    })
    grid.write_parquet(tmp_path / "grid.parquet")
    meta.write_parquet(tmp_path / "meta.parquet")


def test_load_returns_canonical_shape(tmp_path):
    _write_export(tmp_path)
    grid, meta = load_mt5_panel(tmp_path)
    assert set(GRID_COLUMNS).issubset(grid.columns)
    assert set(META_COLUMNS).issubset(meta.columns)
    assert grid.schema["date"] == pl.Date
    # sorted by (date, minute, scrip_code)
    assert grid["minute"].to_list() == [0, 0, 1, 1]
    assert grid["scrip_code"].to_list() == [101, 102, 101, 102]
    assert meta.height == 2


def test_missing_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mt5_panel(tmp_path)


def test_schema_drift_raises(tmp_path):
    _write_export(tmp_path)
    # corrupt the grid by dropping a required column
    pl.read_parquet(tmp_path / "grid.parquet").drop("mid").write_parquet(tmp_path / "grid.parquet")
    with pytest.raises(ValueError):
        load_mt5_panel(tmp_path)
