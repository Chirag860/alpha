"""Sample weights for overlapping, cross-sectionally-correlated labels (§3.6, §4.4).

Labels from overlapping horizons are not IID; with a 15-minute horizon on 1-minute bars
each label overlaps ~15 others. Standard bagging then massively oversamples redundant
information. We multiply up to five weights (López de Prado, plus a panel-specific one):

    w = uniqueness x return_attribution x time_decay x liquidity x xs_concurrency

* **avg uniqueness** -- inverse temporal concurrency within a name (AFML Ch. 4),
* **return attribution** -- weight by the magnitude of return earned over the label span,
* **time decay** -- exponential in sample age (half-life ~6 months, §4.5),
* **liquidity** -- an observation in a name where you can deploy a real clip is worth more
  than one capped at ₹1 lakh (§4.1),
* **cross-sectional concurrency** -- ``1/sqrt(n_active_names_at_t)``: two names at the same
  minute share the market factor and are not independent (§4.4).

All computed strictly from realized (past-and-present) label spans; nothing here peeks
forward.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..config import Config


def _uniqueness_and_attribution(n: int, starts: np.ndarray, spans: np.ndarray,
                                bar_ret: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized avg-uniqueness and return-attribution for one ``(scrip, date)`` group.

    ``starts`` are positions within the group, ``spans`` the label length in bars, and
    ``bar_ret`` the per-bar residual return. Uses a difference array for concurrency and
    prefix sums for O(1) range means (AFML Ch. 4).
    """
    ends = np.minimum(starts + spans, n - 1)
    diff = np.zeros(n + 1)
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends + 1, -1)
    c = np.cumsum(diff)[:n]
    c = np.maximum(c, 1.0)

    inv = 1.0 / c
    cum_inv = np.concatenate([[0.0], np.cumsum(inv)])
    length = (ends - starts + 1).astype(float)
    uniq = (cum_inv[ends + 1] - cum_inv[starts]) / length

    r_over_c = bar_ret / c
    cum_r = np.concatenate([[0.0], np.cumsum(r_over_c)])
    attr = np.abs(cum_r[ends + 1] - cum_r[starts])
    return uniq, attr


def compute_weights(labels: pl.DataFrame, cfg: Config,
                    turnover_by_scrip: dict[int, float] | None = None) -> pl.DataFrame:
    """Return ``labels`` with a ``weight`` column (mean-normalized to 1).

    Requires ``ret_resid``, ``span_bars``, ``date``, ``minute``, ``scrip_code``.
    ``turnover_by_scrip`` (optional) supplies each name's median BSE turnover for the
    liquidity weight; absent it, liquidity weight is uniform.
    """
    w = cfg.weights
    labels = labels.sort(["scrip_code", "date", "minute"]).with_row_index("__gid")

    uniq = np.ones(labels.height)
    attr = np.ones(labels.height)
    groups = (
        labels.group_by(["scrip_code", "date"], maintain_order=True)
        .agg(start=pl.col("__gid").min(), n=pl.len())
    ).sort("start")

    ret_all = labels["ret_resid"].to_numpy()
    span_all = labels["span_bars"].to_numpy()
    for row in groups.iter_rows(named=True):
        s, nn = int(row["start"]), int(row["n"])
        sl = slice(s, s + nn)
        starts = np.arange(nn)
        u_g, a_g = _uniqueness_and_attribution(nn, starts, span_all[sl], ret_all[sl])
        uniq[sl] = u_g
        attr[sl] = a_g

    labels = labels.with_columns(
        pl.Series("w_uniqueness", uniq),
        pl.Series("w_attribution", attr),
    )

    # time decay by sample age in days
    max_date = labels["date"].max()
    hl = float(w.time_decay_halflife_days)
    labels = labels.with_columns(
        (0.5 ** ((pl.lit(max_date) - pl.col("date")).dt.total_days() / hl)).alias("w_decay")
    )

    # cross-sectional concurrency: 1/sqrt(#names in that (date, minute))
    n_active = labels.group_by(["date", "minute"]).agg(n_active=pl.len())
    labels = labels.join(n_active, on=["date", "minute"], how="left").with_columns(
        (1.0 / pl.col("n_active").cast(pl.Float64).sqrt()).alias("w_xs")
    )

    # liquidity weight
    if turnover_by_scrip:
        cap = float(np.median(list(turnover_by_scrip.values())))
        liq = labels["scrip_code"].map_elements(
            lambda c: min(turnover_by_scrip.get(int(c), cap), cap) / cap,
            return_dtype=pl.Float64,
        )
        labels = labels.with_columns(liq.alias("w_liquidity"))
    else:
        labels = labels.with_columns(pl.lit(1.0).alias("w_liquidity"))

    weight = pl.lit(1.0)
    if w.use_uniqueness:
        weight = weight * pl.col("w_uniqueness")
    if w.use_return_attribution:
        # attribution can be ~0; add a small floor so a real-but-small move still counts
        weight = weight * (pl.col("w_attribution") + pl.col("w_attribution").mean() * 0.1)
    weight = weight * pl.col("w_decay")
    if w.use_liquidity:
        weight = weight * pl.col("w_liquidity")
    if w.use_xs_concurrency:
        weight = weight * pl.col("w_xs")

    labels = labels.with_columns(weight.alias("_w_raw"))
    mean_w = labels["_w_raw"].mean() or 1.0
    labels = labels.with_columns((pl.col("_w_raw") / mean_w).alias("weight")).drop(
        ["_w_raw", "__gid"]
    )
    return labels
