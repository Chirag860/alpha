"""Point-in-time universe construction and clip caps (§1.2, §1.3).

Every exclusion here is a *tradability* constraint, not a cosmetic filter:

* **T2T / BE**  -> intraday netting prohibited; you cannot square off (§1.3).
* **ASM / GSM** -> 100% margin / call-auction / effectively untradeable.
* **narrow circuit band** (2/5/10%) -> the position can freeze at the band and trap
  you into a delivery trade at 20 bps STT (§0.4, §1.3).
* **Series Z / SME / suspended** -> not continuously tradable.

The screen is *strictly* backward-looking: it uses only rows dated **before** the
as-of date, keyed on ``scrip_code`` (stable across renames, §1.4). Screening on the
currently-listed set would inject survivorship bias -- the names that blew up are
exactly the ones the model most needs to have seen.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from ..config import Config


def build_universe(daily: pl.DataFrame, asof: dt.date, cfg: Config) -> pl.DataFrame:
    """Return the tradable universe as of ``asof``.

    Parameters
    ----------
    daily
        EOD reference panel (see :data:`~bsealpha.data.schema.DAILY_SCHEMA`).
    asof
        The morning of the trade date. Only ``date < asof`` rows are consulted.
    cfg
        Config; reads the ``universe:`` block.

    Returns
    -------
    polars.DataFrame
        One row per admitted ``scrip_code`` with the median liquidity stats and the
        per-name ``max_clip`` cap.
    """
    u = cfg.universe
    lookback = int(u.lookback_days)
    lo = asof - dt.timedelta(days=lookback * 2)          # calendar window (~lookback sessions)

    win = daily.filter((pl.col("date") < asof) & (pl.col("date") >= lo))
    if win.height == 0:
        return win.head(0)

    agg = (
        win.group_by("scrip_code")
        .agg(
            turnover=pl.col("bse_turnover").median(),
            trades=pl.col("bse_trades").median(),
            spread=pl.col("median_spread_bps").median(),
            price=pl.col("close").last(),
            band=pl.col("circuit_band_pct").min(),
            n_obs=pl.len(),
        )
    )

    latest_date = win.select(pl.col("date").max()).item()
    latest = daily.filter(pl.col("date") == latest_date).select(
        ["scrip_code", "symbol", "sector", "series",
         "asm_flag", "gsm_flag", "t2t_flag", "is_suspended"]
    )

    screened = (
        agg.join(latest, on="scrip_code", how="inner")
        .filter(
            (pl.col("n_obs") >= lookback)
            & (pl.col("turnover") >= u.min_bse_turnover)
            & (pl.col("trades") >= u.min_bse_trades)
            & (pl.col("spread") <= u.max_spread_bps)
            & (pl.col("price").is_between(u.min_price, u.max_price))
            & (pl.col("band") >= u.min_circuit_band_pct)
            & (~pl.col("t2t_flag"))
            & (~pl.col("asm_flag"))
            & (~pl.col("gsm_flag"))
            & (~pl.col("is_suspended"))
            & (pl.col("series").is_in(list(u.allowed_series)))
        )
        .with_columns(
            (pl.col("turnover") * u.participation_pct / 100.0).alias("max_clip"),
            pl.lit(asof).alias("asof"),
        )
        .sort("scrip_code")
    )
    return screened


def max_clip(turnover_median: float, participation_pct: float = 1.5) -> float:
    """Cap a clip at ``participation_pct`` % of the name's median BSE turnover (§1.2)."""
    return float(turnover_median) * participation_pct / 100.0


def rolling_universe(daily: pl.DataFrame, cfg: Config,
                     rebuild_every_days: int = 5) -> pl.DataFrame:
    """Rebuild the universe on a rolling cadence and stack the results.

    Emulates the weekly point-in-time rebuild (§1.2). Returns the union of per-asof
    universes, each tagged with its ``asof`` date, so downstream code can join a bar's
    trade date to the universe that was known that morning.
    """
    dates = daily.select(pl.col("date").unique().sort()).to_series().to_list()
    if not dates:
        return daily.head(0)
    lookback = int(cfg.universe.lookback_days)
    out: list[pl.DataFrame] = []
    for idx in range(lookback, len(dates), rebuild_every_days):
        asof = dates[idx]
        uni = build_universe(daily, asof, cfg)
        if uni.height:
            out.append(uni)
    if not out:
        return build_universe(daily, dates[-1], cfg).head(0)
    return pl.concat(out, how="vertical")
