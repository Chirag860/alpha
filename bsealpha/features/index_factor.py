"""Index / factor features and cross-sectional dispersion (§3.2).

The market and sector factors are what make effective breadth << N (§0.5); they are
both the thing you neutralize (residualization) and useful state variables. Dispersion --
the cross-sectional spread of residual returns -- is a strong regime variable: high
dispersion means a stock-picking regime, which is when a cross-sectional model works.
"""

from __future__ import annotations

import polars as pl


def compute_returns(grid: pl.DataFrame) -> pl.DataFrame:
    """Add per-name log-return ``ret`` on the common grid (backward difference)."""
    grid = grid.sort(["scrip_code", "date", "minute"])
    return grid.with_columns(
        (pl.col("mid").log() - pl.col("mid").log().shift(1).over(["scrip_code", "date"]))
        .fill_null(0.0)
        .alias("ret")
    )


def attach_factors(grid: pl.DataFrame, *, sector_col: str = "sector") -> pl.DataFrame:
    """Attach market/sector factor returns, dispersion, and a VIX-like proxy.

    Requires ``ret`` (see :func:`compute_returns`) and a ``sector`` column.

    * ``index_ret`` -- equal-weight cross-sectional mean return per ``(date, minute)``,
    * ``sector_ret`` -- mean return within ``(date, minute, sector)``,
    * ``dispersion`` -- cross-sectional std of returns per ``(date, minute)``,
    * ``vix_proxy`` -- trailing 15-minute std of ``index_ret`` (a slow vol-regime state).
    """
    idx = (
        grid.group_by(["date", "minute"], maintain_order=True)
        .agg(index_ret=pl.col("ret").mean(), dispersion=pl.col("ret").std())
    )
    sec = (
        grid.group_by(["date", "minute", sector_col], maintain_order=True)
        .agg(sector_ret=pl.col("ret").mean())
    )
    out = (
        grid.join(idx, on=["date", "minute"], how="left")
        .join(sec, on=["date", "minute", sector_col], how="left")
        .with_columns(pl.col("dispersion").fill_null(0.0))
    )
    # VIX-like proxy: trailing market vol (per date, ordered by minute)
    vix = (
        idx.sort(["date", "minute"])
        .with_columns(
            pl.col("index_ret")
            .rolling_std(window_size=15, min_samples=3)
            .over("date")
            .alias("vix_proxy")
        )
        .select(["date", "minute", "vix_proxy"])
    )
    return out.join(vix, on=["date", "minute"], how="left").with_columns(
        pl.col("vix_proxy").fill_null(0.0)
    )
