"""Event and time bars, and the common cross-sectional minute grid (§2.2).

Two clocks, and the report is emphatic that you must be explicit about which one every
quantity lives on:

* **Per-name event bars** (``rupee_bars`` / ``volume_bars`` / ``tick_bars``) for
  feature and label construction. Rupee bars are primary: sample every ``X_i`` of BSE
  turnover, with ``X_i`` calibrated per name so median duration is a target seconds
  (§2.2). Event bars absorb the U-shaped intensity curve automatically.
* **Common 1-minute grid** (``common_minute_grid``) for every *cross-sectional*
  operation -- residualization, ranking, portfolio construction -- because those need
  a shared clock across names (§2.2, §2.3).

**Causality:** every bar's timestamp is the *local-receipt* time of the event that
closed it (``ts_ns`` of the closing trade), and the closing mid/micro is taken from the
last depth snapshot at-or-before that time (a strictly backward ``join_asof``). A live
system could have acted no earlier.
"""

from __future__ import annotations

import polars as pl

from ..config import Config
from ..market import micro_price, mid_price


def attach_mid_micro(depth: pl.DataFrame) -> pl.DataFrame:
    """Add ``mid`` and ``micro`` columns (L1) to a depth snapshot frame."""
    bq, aq = pl.col("bid_qty_0"), pl.col("ask_qty_0")
    imb = bq / (bq + aq)
    return depth.with_columns(
        (0.5 * (pl.col("bid_px_0") + pl.col("ask_px_0"))).alias("mid"),
        (pl.col("ask_px_0") * imb + pl.col("bid_px_0") * (1.0 - imb)).alias("micro"),
    )


def _minute_col(session_min: pl.Expr) -> pl.Expr:
    """Session-relative integer minute (0..374), the common cross-sectional clock."""
    return session_min.floor().cast(pl.Int64).alias("minute")


def rupee_bars(trades: pl.DataFrame, depth: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Per-name rupee (turnover) bars.

    ``X_i = median_daily_BSE_turnover_i / target_bars_per_day`` (§2.2). Bars reset each
    session (intraday only, §0.4). The closing mid/micro is joined from depth with a
    backward ``join_asof`` so no future book state leaks in.
    """
    target = int(cfg.bars.target_bars_per_day)
    tr = trades.sort(["scrip_code", "date", "ts_ns"]).with_columns(
        (pl.col("price") * pl.col("qty")).alias("val")
    )
    thresholds = (
        tr.group_by(["scrip_code", "date"]).agg(pl.col("val").sum().alias("day_turn"))
        .group_by("scrip_code").agg(pl.col("day_turn").median().alias("med_turn"))
        .with_columns((pl.col("med_turn") / target).clip(lower_bound=1.0).alias("threshold"))
        .select(["scrip_code", "threshold"])
    )
    tr = tr.join(thresholds, on="scrip_code", how="left")
    tr = tr.with_columns(
        pl.col("val").cum_sum().over(["scrip_code", "date"]).alias("cumval")
    ).with_columns(
        (pl.col("cumval") / pl.col("threshold")).floor().cast(pl.Int64).alias("bar_id")
    )
    return _aggregate_event_bars(tr, depth, group_extra=["bar_id"])


def volume_bars(trades: pl.DataFrame, depth: pl.DataFrame, threshold_qty: float) -> pl.DataFrame:
    """Per-name volume bars: sample every ``threshold_qty`` shares (§2.2)."""
    tr = trades.sort(["scrip_code", "date", "ts_ns"]).with_columns(
        (pl.col("price") * pl.col("qty")).alias("val"),
        pl.col("qty").cum_sum().over(["scrip_code", "date"]).alias("cumqty"),
    ).with_columns(
        (pl.col("cumqty") / threshold_qty).floor().cast(pl.Int64).alias("bar_id")
    )
    return _aggregate_event_bars(tr, depth, group_extra=["bar_id"])


def tick_bars(trades: pl.DataFrame, depth: pl.DataFrame, ticks_per_bar: int) -> pl.DataFrame:
    """Per-name tick bars: sample every ``ticks_per_bar`` trade prints (§2.2)."""
    tr = trades.sort(["scrip_code", "date", "ts_ns"]).with_columns(
        (pl.col("price") * pl.col("qty")).alias("val"),
        pl.int_range(pl.len()).over(["scrip_code", "date"]).alias("_i"),
    ).with_columns(
        (pl.col("_i") // ticks_per_bar).alias("bar_id")
    )
    return _aggregate_event_bars(tr, depth, group_extra=["bar_id"])


def _aggregate_event_bars(tr: pl.DataFrame, depth: pl.DataFrame,
                          group_extra: list[str]) -> pl.DataFrame:
    """Aggregate labelled trades into OHLC/VWAP bars and attach closing mid/micro."""
    keys = ["scrip_code", "date"] + group_extra
    bars = (
        tr.group_by(keys, maintain_order=True).agg(
            ts_ns=pl.col("ts_ns").last(),
            session_min=pl.col("session_min").last(),
            open=pl.col("price").first(),
            high=pl.col("price").max(),
            low=pl.col("price").min(),
            close_trade=pl.col("price").last(),
            vwap=(pl.col("val").sum() / pl.col("qty").sum()),
            turnover=pl.col("val").sum(),
            n_trades=pl.len(),
        )
        .with_columns(_minute_col(pl.col("session_min")))
        .sort(["scrip_code", "date", "ts_ns"])
    )
    dm = attach_mid_micro(depth).select(
        ["scrip_code", "date", "ts_ns", "mid", "micro"]
    ).sort(["scrip_code", "date", "ts_ns"])
    bars = bars.join_asof(dm, on="ts_ns", by=["scrip_code", "date"], strategy="backward")
    # bars before the first snapshot inherit no mid -> fall back to trade close
    bars = bars.with_columns(
        pl.col("mid").fill_null(pl.col("close_trade")),
        pl.col("micro").fill_null(pl.col("close_trade")),
    ).with_columns(pl.col("mid").alias("close"))
    return bars.drop(group_extra)


def common_minute_grid(depth: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:
    """Build the common 1-minute cross-sectional grid (§2.2).

    Per ``(scrip_code, date, minute)``: the closing mid/micro is the *last* snapshot in
    the minute; turnover / VWAP / trade-count come from that minute's trades. Minutes
    with a book but no trades carry zero turnover (a real, informative state in thin BSE
    names). This is the clock on which residualization and ranking happen.
    """
    dm = attach_mid_micro(depth).with_columns(_minute_col(pl.col("session_min")))
    grid_depth = (
        dm.sort(["scrip_code", "date", "ts_ns"])
        .group_by(["scrip_code", "date", "minute"], maintain_order=True)
        .agg(
            session_min=pl.col("session_min").last(),
            mid=pl.col("mid").last(),
            micro=pl.col("micro").last(),
        )
    )
    tr = trades.with_columns(
        (pl.col("price") * pl.col("qty")).alias("val"),
        _minute_col(pl.col("session_min")),
    )
    grid_tr = (
        tr.sort(["scrip_code", "date", "ts_ns"])
        .group_by(["scrip_code", "date", "minute"], maintain_order=True)
        .agg(
            open=pl.col("price").first(),
            high=pl.col("price").max(),
            low=pl.col("price").min(),
            vwap=(pl.col("val").sum() / pl.col("qty").sum()),
            turnover=pl.col("val").sum(),
            n_trades=pl.len(),
        )
    )
    grid = grid_depth.join(grid_tr, on=["scrip_code", "date", "minute"], how="left")
    grid = grid.with_columns(
        pl.col("turnover").fill_null(0.0),
        pl.col("n_trades").fill_null(0),
        pl.col("open").fill_null(pl.col("mid")),
        pl.col("high").fill_null(pl.col("mid")),
        pl.col("low").fill_null(pl.col("mid")),
        pl.col("vwap").fill_null(pl.col("mid")),
    ).with_columns(pl.col("mid").alias("close"))
    return grid.sort(["date", "minute", "scrip_code"])


def build_bars(depth: pl.DataFrame, trades: pl.DataFrame,
               cfg: Config) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Convenience: return ``(event_bars, common_grid)`` per the configured bar kind."""
    kind = cfg.bars.kind
    if kind == "rupee":
        event = rupee_bars(trades, depth, cfg)
    elif kind == "volume":
        event = volume_bars(trades, depth, threshold_qty=1000.0)
    elif kind == "tick":
        event = tick_bars(trades, depth, ticks_per_bar=20)
    elif kind == "time":
        event = common_minute_grid(depth, trades)
    else:  # pragma: no cover
        raise ValueError(f"unknown bar kind: {kind}")
    grid = common_minute_grid(depth, trades)
    return event, grid
