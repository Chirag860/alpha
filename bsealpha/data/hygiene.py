"""Data-hygiene validators (§1.4, §2.0).

With a 300-name panel, survivorship, corporate actions, and feed gaps are first-order and
will silently inflate a backtest. These checks are meant to run on ingest, before any
research, and to *fail loudly* rather than surface as NaNs later:

* **Circuit-band violations** -- after corporate-action adjustment, no intraday bar return
  should exceed the name's circuit band. A violation is almost always an unhandled split,
  not alpha (§1.4).
* **Snapshot gaps** -- a dropped depth update corrupts the book; count gaps and drop those
  periods from training rather than interpolating (§2.0d).
* **Outliers** -- inspect every observation beyond ±5σ by hand for the first month;
  essentially all are data errors (§1.4).
* **Survivorship** -- count names that appear and later disappear (delisted/merged). If your
  universe has zero disappearances, it was screened on the currently-listed set (§1.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass
class HygieneReport:
    n_rows: int = 0
    n_names: int = 0
    n_days: int = 0
    circuit_violations: int = 0
    gap_periods: int = 0
    outliers_5sigma: int = 0
    disappeared_names: int = 0
    nan_rate: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def ok(self) -> bool:
        return self.circuit_violations == 0 and not self.warnings

    def summary(self) -> str:
        lines = [
            f"rows={self.n_rows:,} names={self.n_names} days={self.n_days}",
            f"circuit_violations={self.circuit_violations} (must be 0)",
            f"gap_periods={self.gap_periods} outliers_5sigma={self.outliers_5sigma}",
            f"disappeared_names={self.disappeared_names} nan_rate={self.nan_rate:.4f}",
        ]
        return " | ".join(lines)


def circuit_band_violations(bars: pl.DataFrame, band_by_scrip: dict[int, float], *,
                            ret_col: str = "adj_ret") -> pl.DataFrame:
    """Rows whose |return| exceeds the name's circuit band (as a fraction). §1.4."""
    if ret_col not in bars.columns:
        bars = bars.sort(["scrip_code", "date"]).with_columns(
            (pl.col("close").log() - pl.col("close").log().shift(1).over(["scrip_code", "date"]))
            .alias(ret_col)
        )
    band = bars["scrip_code"].map_elements(
        lambda c: band_by_scrip.get(int(c), 20.0) / 100.0, return_dtype=pl.Float64
    )
    return bars.with_columns(band.alias("_band")).filter(
        pl.col(ret_col).abs() > pl.col("_band")
    )


def snapshot_gap_report(depth: pl.DataFrame, *, expected_dt_ns: float,
                        tol: float = 3.0) -> int:
    """Count (scrip, date) periods with a snapshot gap > ``tol x`` the expected cadence."""
    d = depth.sort(["scrip_code", "date", "ts_ns"]).with_columns(
        pl.col("ts_ns").diff().over(["scrip_code", "date"]).alias("_dt")
    )
    gaps = d.filter(pl.col("_dt") > tol * expected_dt_ns)
    return int(gaps.height)


def outlier_report(bars: pl.DataFrame, *, ret_col: str = "adj_ret",
                   n_sigma: float = 5.0) -> pl.DataFrame:
    """Rows beyond ±``n_sigma`` of the per-name return distribution (§1.4)."""
    if ret_col not in bars.columns:
        return bars.head(0)
    stats = bars.group_by("scrip_code").agg(
        _mu=pl.col(ret_col).mean(), _sd=pl.col(ret_col).std()
    )
    return (bars.join(stats, on="scrip_code", how="left")
            .filter((pl.col(ret_col) - pl.col("_mu")).abs() > n_sigma * pl.col("_sd")))


def survivorship_report(daily: pl.DataFrame) -> int:
    """Count names present early in the sample but absent at the end (delisted/merged).

    Zero disappearances on a multi-year panel is itself a red flag: the universe was almost
    certainly screened on the currently-listed set (§1.4).
    """
    dates = daily.select(pl.col("date")).unique().sort("date")["date"]
    if dates.len() < 2:
        return 0
    first_half = daily.filter(pl.col("date") <= dates[dates.len() // 2])["scrip_code"].unique()
    last = daily.filter(pl.col("date") == dates[-1])["scrip_code"].unique()
    disappeared = set(first_half.to_list()) - set(last.to_list())
    return len(disappeared)


def run_hygiene(bars: pl.DataFrame, daily: pl.DataFrame,
                band_by_scrip: dict[int, float], *,
                depth: pl.DataFrame | None = None,
                expected_dt_ns: float | None = None) -> HygieneReport:
    """Run all validators and return an aggregated :class:`HygieneReport`."""
    rep = HygieneReport()
    rep.n_rows = bars.height
    rep.n_names = bars["scrip_code"].n_unique()
    rep.n_days = bars["date"].n_unique() if "date" in bars.columns else 0

    ret_col = "adj_ret" if "adj_ret" in bars.columns else "ret"
    rep.circuit_violations = circuit_band_violations(bars, band_by_scrip, ret_col=ret_col).height
    if rep.circuit_violations:
        rep.warnings.append(
            f"{rep.circuit_violations} bar returns exceed the circuit band -- likely "
            "unadjusted corporate actions (§1.4).")

    rep.outliers_5sigma = outlier_report(bars, ret_col=ret_col).height
    rep.disappeared_names = survivorship_report(daily)
    if rep.disappeared_names == 0 and rep.n_days > 60:
        rep.warnings.append(
            "zero names disappeared over a long sample -- suspect a survivorship-biased "
            "(currently-listed) universe (§1.4).")

    if depth is not None and expected_dt_ns:
        rep.gap_periods = snapshot_gap_report(depth, expected_dt_ns=expected_dt_ns)

    # NaN rate across numeric feature-ish columns
    num = bars.select([c for c, t in bars.schema.items()
                       if t in (pl.Float64, pl.Float32)])
    if num.width:
        total = num.height * num.width
        nulls = sum(num[c].null_count() for c in num.columns)
        rep.nan_rate = nulls / total if total else 0.0
    return rep
