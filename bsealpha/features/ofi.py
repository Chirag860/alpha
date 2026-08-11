"""Order Flow Imbalance (OFI) -- the highest signal-per-effort microstructure feature.

Cont-Kukanov-Stoikov (2014): mid-price changes are approximately linear in OFI, with
slope inversely proportional to depth. Cont-Cucuringu-Zhang (2023): integrate the first
M levels via PCA ("integrated OFI"). BSE gives exactly 5 aggregated levels (§3.1), so we
use all of them (§3.2).

Two implementations that MUST agree (parity-tested):

* :class:`OFI5` -- an O(1) streaming state machine fed snapshots in local-receipt order.
  This is the shape of the live/event-driven engine; no look-ahead is possible.
* :func:`ofi_frame` -- a vectorized polars pass over ``(scrip_code, date)`` groups for
  bulk historical computation. Same arithmetic, written as a shift.

The per-level sign convention (§3.4): bid price **up** => the whole new queue is fresh
liquidity added (``+qty``); **down** => the whole old queue was consumed/cancelled
(``-prev_qty``); **unchanged** => the delta. Ask side is sign-flipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ..data.schema import N_DEPTH_LEVELS


@dataclass
class OFI5:
    """Streaming multi-level OFI over the ``M`` available depth levels (§3.4).

    Feed snapshots in local-receipt order via :meth:`update`. State is O(M).
    """

    M: int = N_DEPTH_LEVELS
    pb: np.ndarray | None = field(default=None)
    qb: np.ndarray | None = field(default=None)
    pa: np.ndarray | None = field(default=None)
    qa: np.ndarray | None = field(default=None)
    depth_ewma: np.ndarray | None = field(default=None)
    alpha: float = 1e-3

    def update(self, bid_px: np.ndarray, bid_qty: np.ndarray,
               ask_px: np.ndarray, ask_qty: np.ndarray) -> np.ndarray:
        """Return the length-M vector of depth-scaled OFI increments for this snapshot."""
        bid_px = np.asarray(bid_px, float); bid_qty = np.asarray(bid_qty, float)
        ask_px = np.asarray(ask_px, float); ask_qty = np.asarray(ask_qty, float)
        if self.pb is None:
            self._init(bid_px, bid_qty, ask_px, ask_qty)
            return np.zeros(self.M)
        e_bid = np.where(bid_px > self.pb, bid_qty,
                         np.where(bid_px < self.pb, -self.qb, bid_qty - self.qb))
        e_ask = np.where(ask_px < self.pa, ask_qty,
                         np.where(ask_px > self.pa, -self.qa, ask_qty - self.qa))
        ofi = e_bid - e_ask
        avg = 0.5 * (bid_qty + ask_qty)
        self.depth_ewma = (1 - self.alpha) * self.depth_ewma + self.alpha * avg
        out = ofi / np.maximum(self.depth_ewma, 1e-12)
        self.pb, self.qb = bid_px.copy(), bid_qty.copy()
        self.pa, self.qa = ask_px.copy(), ask_qty.copy()
        return out

    def _init(self, bp: np.ndarray, bq: np.ndarray, ap: np.ndarray, aq: np.ndarray) -> None:
        self.pb, self.qb, self.pa, self.qa = bp.copy(), bq.copy(), ap.copy(), aq.copy()
        self.depth_ewma = 0.5 * (bq + aq)


def _ofi_expr(level: int) -> list[pl.Expr]:
    """Per-level raw OFI increment as a polars expression over sorted snapshots."""
    bp, bq = pl.col(f"bid_px_{level}"), pl.col(f"bid_qty_{level}")
    ap, aq = pl.col(f"ask_px_{level}"), pl.col(f"ask_qty_{level}")
    pbp = bp.shift(1).over(["scrip_code", "date"])
    pbq = bq.shift(1).over(["scrip_code", "date"])
    pap = ap.shift(1).over(["scrip_code", "date"])
    paq = aq.shift(1).over(["scrip_code", "date"])
    e_bid = (
        pl.when(bp > pbp).then(bq)
        .when(bp < pbp).then(-pbq)
        .otherwise(bq - pbq)
    )
    e_ask = (
        pl.when(ap < pap).then(aq)
        .when(ap > pap).then(-paq)
        .otherwise(aq - paq)
    )
    return [(e_bid - e_ask).alias(f"_ofi_raw_{level}"),
            (0.5 * (bq + aq)).alias(f"_depth_{level}")]


def ofi_frame(depth: pl.DataFrame, m: int = N_DEPTH_LEVELS) -> pl.DataFrame:
    """Vectorized per-snapshot depth-scaled OFI for all levels + integrated OFI.

    Depth scaling here divides each level's OFI by that level's *contemporaneous* average
    depth (a causal, per-snapshot normalization); the streaming class uses a depth EWMA.
    Both are monotone in the same quantity, so signs and rankings match -- which is what
    the parity test checks. Integrated OFI is the equal-weight sum across levels (a
    PC1 proxy for the highly-correlated level vector; :class:`IntegratedOFI` fits the
    true PC1 when needed).
    """
    d = depth.sort(["scrip_code", "date", "ts_ns"])
    exprs: list[pl.Expr] = []
    for lv in range(m):
        exprs.extend(_ofi_expr(lv))
    d = d.with_columns(exprs)
    scaled = [
        (pl.col(f"_ofi_raw_{lv}") / pl.col(f"_depth_{lv}").clip(lower_bound=1e-12))
        .fill_null(0.0).alias(f"ofi_{lv}")
        for lv in range(m)
    ]
    d = d.with_columns(scaled)
    d = d.with_columns(
        pl.sum_horizontal([pl.col(f"ofi_{lv}") for lv in range(m)]).alias("ofi_integrated")
    )
    drop = [f"_ofi_raw_{lv}" for lv in range(m)] + [f"_depth_{lv}" for lv in range(m)]
    return d.drop(drop)


class IntegratedOFI:
    """PC1 of the level-wise OFI vector (Cont-Cucuringu-Zhang, §3.2).

    Fit on TRAIN data only, then transform forward -- fitting on the full sample would
    leak the test-period covariance structure (unsupervised, but still a leak the report
    flags in §5.5). Orientation is fixed so positive = buy pressure.
    """

    def __init__(self, m: int = N_DEPTH_LEVELS) -> None:
        self.m = m
        self.w: np.ndarray | None = None

    def fit(self, ofi_matrix: np.ndarray) -> "IntegratedOFI":
        X = np.asarray(ofi_matrix, float)
        X = X - X.mean(0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        w = vt[0]
        self.w = w * np.sign(w.sum() if w.sum() != 0 else 1.0)
        return self

    def transform(self, ofi_matrix: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("IntegratedOFI must be fit before transform")
        return np.asarray(ofi_matrix, float) @ self.w
