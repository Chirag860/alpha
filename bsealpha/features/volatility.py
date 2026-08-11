"""Volatility features (§2.5, §3.2).

Three ideas the report insists on:

* **Time-of-day normalization** (§2.5). Indian intraday vol is strongly U-shaped; without
  dividing it out, every model becomes an opening-auction detector. The profile is fit on
  TRAINING data only (:class:`TodVolProfile`).
* **Bipower variation** (Barndorff-Nielsen & Shephard): jump-robust vol; ``RV - BV``
  isolates the jump component, which has a distinct order-flow signature.
* **Realized vol at multiple windows** as a regime variable.

All rolling windows here are strictly backward (``.rolling`` over past bars), so the
features are point-in-time.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..market import tod_bin

_MU1 = np.sqrt(2.0 / np.pi)


def realized_vol_expr(ret_col: str, window: int, name: str) -> pl.Expr:
    """Rolling realized vol (std of log-returns) over ``window`` past bars, per name."""
    return (
        pl.col(ret_col).rolling_std(window_size=window, min_samples=max(2, window // 3))
        .over("scrip_code")
        .alias(name)
    )


def bipower_variation(log_ret: np.ndarray) -> float:
    """Jump-robust bipower variation of a log-return series."""
    r = np.abs(np.asarray(log_ret, float))
    if r.size < 2:
        return 0.0
    return float(_MU1 ** -2 * np.sum(r[1:] * r[:-1]))


def realized_variance(log_ret: np.ndarray) -> float:
    """Naive realized variance (sum of squared returns)."""
    r = np.asarray(log_ret, float)
    return float(np.sum(r ** 2))


def jump_component(log_ret: np.ndarray) -> float:
    """``max(RV - BV, 0)`` -- the jump part of quadratic variation."""
    return max(realized_variance(log_ret) - bipower_variation(log_ret), 0.0)


class TodVolProfile:
    """Time-of-day volatility multiplier, FIT ON TRAINING DATA ONLY (§2.5).

    ``multiplier[bin]`` = (vol of residual returns in that session-time bin) / (overall
    vol). Dividing residual returns by this profile removes the U-shape before any
    cross-sectional comparison, which is a prerequisite for the panel to be comparable
    across the day.
    """

    def __init__(self, n_bins: int = 25, clip: tuple[float, float] = (0.4, 3.0)) -> None:
        self.n_bins = n_bins
        self.clip = clip
        self.profile: np.ndarray | None = None

    def fit(self, resid_ret: np.ndarray, minute_of_day: np.ndarray) -> "TodVolProfile":
        resid_ret = np.asarray(resid_ret, float)
        b = tod_bin(np.asarray(minute_of_day, float), self.n_bins)
        overall = np.nanstd(resid_ret)
        overall = overall if overall > 1e-12 else 1.0
        prof = np.ones(self.n_bins)
        for k in range(self.n_bins):
            m = b == k
            if m.sum() > 20:
                prof[k] = np.nanstd(resid_ret[m]) / overall
        self.profile = np.clip(prof, *self.clip)
        return self

    def multiplier(self, minute_of_day: np.ndarray) -> np.ndarray:
        if self.profile is None:
            raise RuntimeError("TodVolProfile must be fit before use")
        b = tod_bin(np.asarray(minute_of_day, float), self.n_bins)
        return self.profile[b]
