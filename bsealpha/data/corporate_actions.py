"""Point-in-time corporate-action adjustment (§1.4).

Splits, bonuses, and consolidations must be applied with a point-in-time adjustment
factor. A 1:10 split shows up in raw data as a −90% one-minute return on the ex-date,
which would dominate any vol estimate and manufacture spectacular fake signals. The fix
is a **cumulative back-adjustment factor** that removes the ex-date jump.

Two correctness rules the report insists on:

* **Forward-only application (no look-ahead).** An adjustment factor is known only on the
  ex-date morning. For *live / point-in-time* feature construction on date ``T`` you must
  not apply an action whose ex-date is after ``T``. :func:`adjust_prices` therefore takes
  an optional ``asof`` and, in point-in-time mode, ignores future actions.
* **Returns, not levels.** Back-adjustment is uniform scaling, so it is correct for returns
  and vol. Absolute-price features (tick band, circuit reference) must use the *as-traded*
  price -- so we keep the raw columns and add ``adj_*`` columns rather than overwriting.

Action convention: ``price_ratio`` is the multiplier the price is scaled by *at* the ex-date
(1:10 split => 0.1; 1:1 bonus => 0.5; 10:1 consolidation => 10.0). ``qty_ratio`` is the
reciprocal (shares outstanding multiply). Reconcile these against the BSE corporate-action
file before use.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

CORP_ACTION_SCHEMA = {
    "scrip_code": pl.Int64,
    "ex_date": pl.Date,
    "action_type": pl.Utf8,     # split | bonus | consolidation | dividend
    "price_ratio": pl.Float64,  # price multiplier at ex-date (0.1 for 1:10 split)
    "qty_ratio": pl.Float64,    # share-count multiplier (10.0 for 1:10 split)
}


def cumulative_factor(ex_dates: np.ndarray, price_ratios: np.ndarray,
                      bar_dates: np.ndarray) -> np.ndarray:
    """Back-adjustment factor for each ``bar_date``: product of ``price_ratio`` for all
    actions strictly *after* that date.

    A bar before a 1:10 split gets factor 0.1 so its price is brought onto the post-split
    scale (removing the jump); a bar on/after the last action gets 1.0.
    """
    if len(ex_dates) == 0:
        return np.ones(len(bar_dates))
    order = np.argsort(ex_dates)
    ex_sorted = np.asarray(ex_dates)[order]
    r_sorted = np.asarray(price_ratios, float)[order]
    # suffix product: factor if bar_date < ex_sorted[k] for all k >= first future action
    suffix = np.concatenate([np.cumprod(r_sorted[::-1])[::-1], [1.0]])
    # for a bar date d, first index k with ex_sorted[k] > d ; factor = suffix[k]
    idx = np.searchsorted(ex_sorted, bar_dates, side="right")
    return suffix[idx]


def adjust_prices(bars: pl.DataFrame, actions: pl.DataFrame, *,
                  price_cols: tuple[str, ...] = ("open", "high", "low", "close", "mid", "vwap"),
                  qty_cols: tuple[str, ...] = ("turnover", "n_trades"),
                  asof: dt.date | None = None,
                  point_in_time: bool = False) -> pl.DataFrame:
    """Add ``adj_<col>`` columns to ``bars`` using ``actions`` (§1.4).

    Parameters
    ----------
    bars
        Frame with ``scrip_code``, ``date`` and the given price columns.
    actions
        Corporate actions (see :data:`CORP_ACTION_SCHEMA`).
    asof, point_in_time
        In point-in-time mode, actions with ``ex_date > asof`` are ignored (no look-ahead).
    """
    if point_in_time and asof is not None:
        actions = actions.filter(pl.col("ex_date") <= asof)

    if actions.height == 0:
        return bars.with_columns([pl.col(c).alias(f"adj_{c}") for c in price_cols
                                  if c in bars.columns])

    bars = bars.sort(["scrip_code", "date"])
    act_by_scrip = {
        int(sc[0] if isinstance(sc, tuple) else sc):
            (g["ex_date"].to_numpy(), g["price_ratio"].to_numpy())
        for sc, g in actions.group_by("scrip_code")
    }
    factors = np.ones(bars.height)
    codes = bars["scrip_code"].to_numpy()
    dates = bars["date"].to_numpy()
    # compute factor per scrip group (contiguous after sort)
    for sc in np.unique(codes):
        mask = codes == sc
        if int(sc) in act_by_scrip:
            ex, r = act_by_scrip[int(sc)]
            factors[mask] = cumulative_factor(ex, r, dates[mask])
    bars = bars.with_columns(pl.Series("_adj_factor", factors))

    price_exprs = [(pl.col(c) * pl.col("_adj_factor")).alias(f"adj_{c}")
                   for c in price_cols if c in bars.columns]
    # qty scales by 1/price_ratio (shares multiply as price divides); turnover is invariant
    qty_exprs = []
    for c in qty_cols:
        if c in bars.columns and c != "turnover":
            qty_exprs.append((pl.col(c) / pl.col("_adj_factor")).alias(f"adj_{c}"))
        elif c == "turnover" and c in bars.columns:
            qty_exprs.append(pl.col(c).alias("adj_turnover"))   # price*qty invariant
    return bars.with_columns(price_exprs + qty_exprs)


def adjusted_returns(bars: pl.DataFrame, price_col: str = "adj_close") -> pl.DataFrame:
    """Add an ``adj_ret`` log-return computed on the adjusted price, per (scrip, date)."""
    return bars.sort(["scrip_code", "date"]).with_columns(
        (pl.col(price_col).log() - pl.col(price_col).log().shift(1).over("scrip_code"))
        .alias("adj_ret")
    )
