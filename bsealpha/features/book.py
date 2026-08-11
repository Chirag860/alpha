"""Book-shape and spread features (§3.2).

Large tick => imbalance is unusually informative and spreads are pinned at 1-2 ticks, so
spread *widening* is one of the few clean regime indicators at this frequency. Spread
must be normalized by the name's own tick regime before any cross-sectional comparison
(§3.2). All from a single snapshot; the engine adds temporal context.
"""

from __future__ import annotations

import polars as pl

from ..data.schema import N_DEPTH_LEVELS


def book_shape_expressions(m: int = N_DEPTH_LEVELS) -> list[pl.Expr]:
    """Book-shape features from a depth snapshot (assumes ``mid`` present).

    * ``spread_bps`` -- L1 spread relative to mid,
    * ``log_depth_bid`` / ``log_depth_ask`` -- total displayed depth (log),
    * ``depth_ratio`` -- BBO depth / deeper (L2-L{m}) depth (thin-touch indicator),
    * ``depth_slope`` -- how fast quantity grows away from the touch,
    * ``depth_imb_total`` -- signed total-depth imbalance in ``[-1, 1]``.
    """
    spread = pl.col("ask_px_0") - pl.col("bid_px_0")
    tot_bid = pl.sum_horizontal([pl.col(f"bid_qty_{i}") for i in range(m)])
    tot_ask = pl.sum_horizontal([pl.col(f"ask_qty_{i}") for i in range(m)])
    bbo_depth = pl.col("bid_qty_0") + pl.col("ask_qty_0")
    deep_depth = (tot_bid + tot_ask - bbo_depth)

    # slope: mean qty at deep levels minus qty at touch, per side, averaged
    deep_bid = pl.mean_horizontal([pl.col(f"bid_qty_{i}") for i in range(1, m)])
    deep_ask = pl.mean_horizontal([pl.col(f"ask_qty_{i}") for i in range(1, m)])

    return [
        (spread / pl.col("mid") * 1e4).alias("spread_bps"),
        (tot_bid + 1.0).log().alias("log_depth_bid"),
        (tot_ask + 1.0).log().alias("log_depth_ask"),
        (bbo_depth / (deep_depth + 1e-9)).alias("depth_ratio"),
        ((deep_bid + deep_ask) / (pl.col("bid_qty_0") + pl.col("ask_qty_0") + 1e-9)
         ).alias("depth_slope"),
        ((tot_bid - tot_ask) / (tot_bid + tot_ask)).alias("depth_imb_total"),
    ]
