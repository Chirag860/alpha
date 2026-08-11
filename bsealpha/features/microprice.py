"""Micro-price family features (§2.4, §3.2).

In a large-tick market like BSE, queue imbalance genuinely predicts which side the next
tick goes to, so ``micro - mid`` is a stronger, better-behaved input than in a small-tick
market. All quantities are bounded and near-stationary -- unusually well-behaved ML
inputs. Everything here is computed from a single snapshot (no look-ahead by
construction); temporal aggregation happens in the engine.
"""

from __future__ import annotations

import polars as pl

from ..data.schema import N_DEPTH_LEVELS


def microprice_expressions(m: int = N_DEPTH_LEVELS) -> list[pl.Expr]:
    """Return polars expressions for micro-price features from a depth snapshot.

    Assumes columns ``bid_px_i / bid_qty_i / ask_px_i / ask_qty_i`` and ``mid`` exist.
    Produces:

    * ``imb_l1`` .. ``imb_l{m}`` -- queue imbalance in ``[0, 1]`` per level,
    * ``micro`` (if not already present) and ``micro_minus_mid`` (bps),
    * ``wmid_minus_mid`` -- weighted-mid vs mid using cumulative depth,
    * ``imb_top`` -- L1 imbalance centered to ``[-0.5, 0.5]`` (signed pressure).
    """
    bq0, aq0 = pl.col("bid_qty_0"), pl.col("ask_qty_0")
    imb0 = bq0 / (bq0 + aq0)
    micro = pl.col("ask_px_0") * imb0 + pl.col("bid_px_0") * (1.0 - imb0)

    # cumulative-depth weighted mid across all levels
    tot_bid = pl.sum_horizontal([pl.col(f"bid_qty_{i}") for i in range(m)])
    tot_ask = pl.sum_horizontal([pl.col(f"ask_qty_{i}") for i in range(m)])
    imb_all = tot_bid / (tot_bid + tot_ask)
    wmid = pl.col("ask_px_0") * imb_all + pl.col("bid_px_0") * (1.0 - imb_all)

    exprs: list[pl.Expr] = [
        micro.alias("micro_calc"),
        ((micro - pl.col("mid")) / pl.col("mid") * 1e4).alias("micro_minus_mid"),
        ((wmid - pl.col("mid")) / pl.col("mid") * 1e4).alias("wmid_minus_mid"),
        (imb0 - 0.5).alias("imb_top"),
        imb_all.alias("imb_all"),
    ]
    for i in range(m):
        bq, aq = pl.col(f"bid_qty_{i}"), pl.col(f"ask_qty_{i}")
        exprs.append((bq / (bq + aq)).alias(f"imb_l{i + 1}"))
    return exprs
