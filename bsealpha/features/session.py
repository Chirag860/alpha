"""Session-structure features (§2.1, §3.2).

Among the most reliably useful features in this market: the Indian session has hard
structural boundaries (open auction, opening drift, midday lull, MIS square-off ramp,
forced flatten) and a far more pronounced intraday volume/vol curve than crypto. Session
time must be *relative*, and these features encode it.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .. import market

OPENING_WINDOW_MIN = 30               # first 30 minutes carry outsized vol/volume
SQUARE_OFF_WINDOW_MIN = 45            # square-off ramp: the last 45 min before close


def session_feature_expressions() -> list[pl.Expr]:
    """Session-time features from a ``minute`` (session-relative) column.

    * ``sess_min`` -- raw session minute,
    * ``sin_tod`` / ``cos_tod`` -- cyclical encoding of session time,
    * ``mins_to_close`` / ``mins_to_flatten`` -- time budget remaining,
    * ``opening_flag`` -- in the first 30 minutes,
    * ``squareoff_flag`` -- in the square-off ramp (last 45 min before close),
    * ``late_session`` -- past the forced-flatten deadline (should not open here).

    All boundaries come from the active market profile (:mod:`bsealpha.market`).
    """
    open_min = market.session_open_min()
    len_min = market.session_len_min()
    flatten = market.flatten_min()
    close_min = market.session_close_min()
    squareoff_start = close_min - SQUARE_OFF_WINDOW_MIN
    sess = pl.col("minute").cast(pl.Float64)                 # session-relative
    mod = open_min + sess                                    # minute-of-day
    frac = sess / len_min
    return [
        sess.alias("sess_min"),
        (2 * np.pi * frac).sin().alias("sin_tod"),
        (2 * np.pi * frac).cos().alias("cos_tod"),
        (close_min - mod).clip(lower_bound=0).alias("mins_to_close"),
        (flatten - mod).clip(lower_bound=0).alias("mins_to_flatten"),
        (sess < OPENING_WINDOW_MIN).cast(pl.Int8).alias("opening_flag"),
        (mod >= squareoff_start).cast(pl.Int8).alias("squareoff_flag"),
        (mod >= flatten).cast(pl.Int8).alias("late_session"),
    ]


def expiry_flag(date: pl.Expr) -> pl.Expr:
    """Weekly-expiry proxy: BSE Sensex weekly options expire on a fixed weekday (§4.5).

    Real deployments load the exchange expiry calendar; here we flag the configured
    weekday (Thursday=4 in polars ``weekday`` where Monday=1) as a stand-in so downstream
    code has the feature. Expiry-day intraday dynamics differ materially (§4.5).
    """
    return (date.dt.weekday() == 4).cast(pl.Int8).alias("expiry_flag")
