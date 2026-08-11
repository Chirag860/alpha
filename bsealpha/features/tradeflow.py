"""Trade-flow features with estimated trade signs (§3.2).

Unlike crypto venues, **India publishes no aggressor flag**, so trade signs must be
*estimated* and are noisy -- a genuine information loss. We use the tick rule (and, when
a synchronized quote is available, a Lee-Ready quote-midpoint test), and we validate the
estimator against L1 quote changes / the generator's ground-truth sign in tests (§3.2
insists on this validation).
"""

from __future__ import annotations

import numpy as np
import polars as pl


def tick_rule_sign(prices: np.ndarray) -> np.ndarray:
    """Classic tick rule: +1 on an uptick, -1 on a downtick, carry forward on a flat.

    Returns an int8 array the same length as ``prices``. The first print is signed +1 by
    convention (no prior price).
    """
    p = np.asarray(prices, float)
    diff = np.sign(np.diff(p, prepend=p[0]))
    sign = np.empty_like(diff)
    last = 1.0
    for i, d in enumerate(diff):
        if d != 0:
            last = d
        sign[i] = last
    return sign.astype(np.int8)


def lee_ready_sign(prices: np.ndarray, mids: np.ndarray) -> np.ndarray:
    """Lee-Ready: compare trade price to the prevailing mid; tie-break with the tick rule.

    ``mids`` must be the mid *known before* each trade (the engine supplies a backward
    ``join_asof`` mid). Above mid => buyer-initiated (+1), below => seller (-1), equal =>
    tick rule.
    """
    p = np.asarray(prices, float)
    mid = np.asarray(mids, float)
    sign = np.sign(p - mid)
    tr = tick_rule_sign(p).astype(float)
    sign = np.where(sign == 0, tr, sign)
    return sign.astype(np.int8)


def sign_trades(trades: pl.DataFrame, *, use_lee_ready_mid: bool = False) -> pl.DataFrame:
    """Add an estimated ``sign`` column to a trades frame, per ``(scrip_code, date)``.

    Uses the tick rule by default (only trade prices needed). ``true_sign`` is left
    untouched for test validation only -- it is never a model input.
    """
    trades = trades.sort(["scrip_code", "date", "ts_ns"])
    signed = (
        trades.group_by(["scrip_code", "date"], maintain_order=True)
        .agg(pl.col("ts_ns"), pl.col("price"))
        .with_columns(
            pl.col("price").map_elements(
                lambda s: list(tick_rule_sign(np.asarray(s))), return_dtype=pl.List(pl.Int8)
            ).alias("sign")
        )
        .explode(["ts_ns", "price", "sign"])
    )
    return trades.join(signed.select(["scrip_code", "date", "ts_ns", "sign"]),
                       on=["scrip_code", "date", "ts_ns"], how="left")


def tradeflow_minute_features(trades_signed: pl.DataFrame) -> pl.DataFrame:
    """Aggregate signed trades to per-``(scrip, date, minute)`` flow features.

    * ``signed_vol`` -- net signed rupee volume in the minute,
    * ``signed_vol_frac`` -- net / gross (a bounded flow-imbalance in ``[-1, 1]``),
    * ``large_print`` -- max single-trade rupee value / minute turnover,
    * ``trade_count`` -- number of prints,
    * ``vwap_minus_mid`` handled in the engine (needs the minute mid).
    """
    t = trades_signed.with_columns(
        (pl.col("price") * pl.col("qty")).alias("val"),
        (pl.col("session_min").floor().cast(pl.Int64)).alias("minute"),
    )
    return (
        t.group_by(["scrip_code", "date", "minute"], maintain_order=True)
        .agg(
            signed_vol=(pl.col("sign") * pl.col("val")).sum(),
            gross_vol=pl.col("val").sum(),
            large_print_val=pl.col("val").max(),
            trade_count=pl.len(),
            vwap_trade=(pl.col("val").sum() / pl.col("qty").sum()),
        )
        .with_columns(
            (pl.col("signed_vol") / pl.col("gross_vol").clip(lower_bound=1e-9)
             ).alias("signed_vol_frac"),
            (pl.col("large_print_val") / pl.col("gross_vol").clip(lower_bound=1e-9)
             ).alias("large_print"),
        )
    )
