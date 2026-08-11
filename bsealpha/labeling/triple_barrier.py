"""Triple-barrier labeling on the RESIDUAL price path with a session-end vertical (§2.5).

Key departures from a textbook / single-instrument triple barrier, all from §2.3-§2.5:

* The barrier walks the **residual** cumulative log-price (index/sector stripped), not raw
  price -- otherwise the label encodes index direction and effective breadth collapses.
* The vertical barrier is ``min(horizon, forced-flatten)``. A signal firing at 15:10 does
  **not** have a 30-minute horizon; it is cut at 15:15 and flagged ``truncated`` so the
  model cannot learn to fire late and be credited a move it could never realize.
* Barrier widths scale with a **tod-normalized** point-in-time residual vol.

The forward vol-adjusted residual return becomes the modeling target (regression on its
cross-sectional rank, or buckets for LambdaRank, §4.2). The realized residual return and
the label spans feed meta-labeling (§meta) and sample weights (§weights).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .. import market
from ..config import Config


def _label_one_day(resid_px: np.ndarray, sigma: np.ndarray, minute: np.ndarray,
                   h_bars: int, u: float, l: float) -> tuple:
    """Label one ``(scrip, date)`` residual path. Minutes assumed 1-apart & ascending."""
    flatten_session_min = market.flatten_session_min()   # session-relative forced-flatten
    n = len(resid_px)
    label = np.zeros(n, np.int8)
    ret = np.zeros(n, np.float64)
    exit_off = np.zeros(n, np.int64)         # offset within the day of the exit bar
    trunc = np.zeros(n, bool)
    sqrt_h = np.sqrt(h_bars)
    for t in range(n):
        m = minute[t]
        if m >= flatten_session_min:          # too late to open a position
            exit_off[t] = t
            trunc[t] = True
            continue
        bars_to_flatten = int(flatten_session_min - m)
        t_end = min(t + h_bars, t + bars_to_flatten, n - 1)
        trunc[t] = (t + h_bars) > (t + bars_to_flatten)
        up = resid_px[t] + u * sigma[t] * sqrt_h
        dn = resid_px[t] - l * sigma[t] * sqrt_h
        hit = 0
        j = t + 1
        while j <= t_end:
            if resid_px[j] >= up:
                hit = 1
                break
            if resid_px[j] <= dn:
                hit = -1
                break
            j += 1
        j = min(j, t_end)
        label[t] = hit
        ret[t] = resid_px[j] - resid_px[t]
        exit_off[t] = j
    return label, ret, exit_off, trunc


def triple_barrier_labels(panel: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Compute residual-path triple-barrier labels for every grid row.

    Requires ``resid_px``, ``sigma_resid``, ``minute`` columns (from the feature engine).

    Returns a frame aligned to the panel with:

    * ``tb_label`` in ``{-1, 0, +1}`` (which barrier hit; 0 = vertical/timeout),
    * ``ret_resid`` -- realized residual log-return to the exit,
    * ``y_voladj`` -- ``ret_resid / (sigma_resid * sqrt(h))``, the modeling target base,
    * ``exit_minute`` -- session minute of the exit bar,
    * ``truncated`` -- vertical barrier was the forced flatten, not the horizon,
    * ``span_bars`` -- label duration in minutes (for uniqueness weights),
    * ``row_id`` -- stable global index for CV/uniqueness bookkeeping.
    """
    h = int(cfg.labeling.horizon_min)
    u, l = float(cfg.labeling.u), float(cfg.labeling.l)
    panel = panel.sort(["scrip_code", "date", "minute"]).with_row_index("row_id")

    labels = np.zeros(panel.height, np.int8)
    rets = np.zeros(panel.height, np.float64)
    exit_min = np.zeros(panel.height, np.float64)
    trunc = np.zeros(panel.height, bool)
    span = np.zeros(panel.height, np.int64)

    # iterate per (scrip, date); rows are contiguous & sorted after the sort above
    offsets = (
        panel.select(["scrip_code", "date"]).with_row_index("gid")
        .group_by(["scrip_code", "date"], maintain_order=True)
        .agg(start=pl.col("gid").min(), n=pl.len())
    ).sort("start")

    resid_all = panel["resid_px"].to_numpy()
    sigma_all = panel["sigma_resid"].to_numpy()
    minute_all = panel["minute"].to_numpy()

    for row in offsets.iter_rows(named=True):
        s, nn = int(row["start"]), int(row["n"])
        sl = slice(s, s + nn)
        lab, ret, exoff, tr = _label_one_day(
            resid_all[sl], sigma_all[sl], minute_all[sl], h, u, l
        )
        labels[sl] = lab
        rets[sl] = ret
        trunc[sl] = tr
        exit_min[sl] = minute_all[sl][exoff]
        span[sl] = exoff - np.arange(nn)

    out = panel.with_columns(
        pl.Series("tb_label", labels),
        pl.Series("ret_resid", rets),
        pl.Series("exit_minute", exit_min),
        pl.Series("truncated", trunc),
        pl.Series("span_bars", span),
    )
    denom = (pl.col("sigma_resid") * np.sqrt(h)).clip(lower_bound=1e-9)
    out = out.with_columns((pl.col("ret_resid") / denom).alias("y_voladj"))
    return out


def add_cross_sectional_targets(labels: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Add the cross-sectional modeling targets (§4.2).

    * ``y_rank`` -- within-``(date, minute)`` rank of ``y_voladj``, centered to
      ``[-0.5, 0.5]`` (regression target),
    * ``y_bucket`` -- non-negative integer quantile bucket of ``y_voladj`` within the
      cross-section (LambdaRank relevance label).
    """
    n_buckets = int(cfg.labeling.n_rank_buckets)
    n = pl.len().over(["date", "minute"])
    r = pl.col("y_voladj").rank("average").over(["date", "minute"])
    out = labels.with_columns(((r - 0.5) / n - 0.5).alias("y_rank"))
    # bucket: floor(rank_fraction * n_buckets), clamped to [0, n_buckets-1]
    out = out.with_columns(
        (((r - 1) / n * n_buckets).floor().clip(0, n_buckets - 1).cast(pl.Int32))
        .alias("y_bucket")
    )
    return out
