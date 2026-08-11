"""Meta-labeling (§2.3d, §3.3).

The primary model decides the **side**; the meta-model decides **whether to act** -- i.e.
whether a trade taken on the primary's side would clear the round-trip cost floor
(4.18 bps, §0.1). This converts an awkward 3-class problem into a well-posed binary one
whose probability maps directly onto position size.

**The classic stacking leak (§3.3):** the primary predictions that define the meta-labels
and feed the meta-features must be **out-of-fold**. Using in-sample primary predictions
inflates meta AUC by 5-15 points. This module only builds labels/features from whatever
predictions it is *given*; the CV harness is responsible for making them OOF.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def make_meta_labels(realized_ret_resid: np.ndarray, side: np.ndarray,
                     cost_bps: float) -> np.ndarray:
    """Binary meta-labels: 1 if the sided trade cleared the cost, else 0.

    Parameters
    ----------
    realized_ret_resid
        Realized residual log-return to the label's exit (from the triple barrier).
    side
        Primary-model side per row (+1 long / -1 short), typically ``sign(score)``.
    cost_bps
        Round-trip cost floor in bps the trade must beat to be worth taking.
    """
    ret_bps = np.asarray(realized_ret_resid, float) * 1e4
    pnl_bps = np.asarray(side, float) * ret_bps
    return (pnl_bps > cost_bps).astype(np.int8)


def build_meta_frame(labels: pl.DataFrame, primary_score: np.ndarray,
                     cost_bps: float, meta_context_cols: list[str]) -> pl.DataFrame:
    """Assemble the meta-model training frame from OOF primary scores (§3.3).

    Meta-features = primary score, its confidence ``|score|``, plus execution-context
    columns (vol, spread, dispersion, session time, ...). The label is
    :func:`make_meta_labels` on the primary side.
    """
    side = np.sign(primary_score)
    side = np.where(side == 0, 1, side)
    meta_label = make_meta_labels(labels["ret_resid"].to_numpy(), side, cost_bps)
    out = labels.with_columns(
        pl.Series("primary_score", np.asarray(primary_score, float)),
        pl.Series("primary_conf", np.abs(np.asarray(primary_score, float))),
        pl.Series("primary_side", side.astype(np.int8)),
        pl.Series("meta_label", meta_label),
    )
    return out


META_CONTEXT_COLS = [
    "sigma_resid", "spread_bps", "dispersion", "vix_proxy",
    "mins_to_flatten", "squareoff_flag", "truncated",
]
"""Default execution-context columns fed to the meta-model alongside the primary score."""
