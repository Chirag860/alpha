"""Residualization: strip the market and sector factors BEFORE labeling (§0.5, §2.3).

This is the single most consequential design decision in the report. Label raw returns
and you build an intraday index-timing model with effective breadth ~1 (§0.5); it will
look fine on a trending sample and will not survive. So we model and trade the
**residual**:

    r_resid[i,t] = r[i,t] - beta_i * index_ret[t] - gamma_i * sector_ret[t]

Two methods:

* ``demean_sector`` -- cross-sectional demeaning within sector each minute. Non-parametric,
  zero estimation error, leakage-free, captures most of the benefit. The robust default.
* ``regression`` -- per-name OLS betas on a **trailing** window, refit **within each CV
  fold** (§5.5) so test-period returns never touch the beta estimate. More faithful to
  §2.3; more moving parts.

The residual cumulative log-price path (``resid_px``) is what the triple barrier walks
(§2.5); ``sigma_resid`` (tod-normalized EWMA of residual returns) sets the barrier widths.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .. import market
from ..config import Config
from .volatility import TodVolProfile


def _ewm_halflife_alpha(halflife: float) -> float:
    return 1.0 - np.exp(np.log(0.5) / max(halflife, 1e-9))


def fit_betas(grid: pl.DataFrame, *, train_mask: np.ndarray | None = None) -> pl.DataFrame:
    """Fit per-name intraday ``(beta, gamma)`` by OLS of ``ret`` on index & sector returns.

    If ``train_mask`` is given (row-aligned to ``grid``), betas use only those rows -- this
    is how the CV harness refits per fold (§5.5). Returns one row per ``scrip_code``.
    """
    df = grid
    if train_mask is not None:
        df = grid.filter(pl.Series(train_mask))
    rows = []
    for sc, g in df.group_by("scrip_code", maintain_order=True):
        scrip = sc[0] if isinstance(sc, tuple) else sc
        y = g["ret"].to_numpy()
        x1 = g["index_ret"].to_numpy()
        x2 = g["sector_ret"].to_numpy()
        X = np.column_stack([x1, x2])
        mask = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
        if mask.sum() < 10:
            rows.append({"scrip_code": scrip, "beta": 1.0, "gamma": 0.0})
            continue
        coef, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        rows.append({"scrip_code": scrip, "beta": float(coef[0]), "gamma": float(coef[1])})
    return pl.DataFrame(rows)


def fit_betas_trailing(grid: pl.DataFrame, lookback_sessions: int = 60) -> pl.DataFrame:
    """Per-``(scrip_code, date)`` intraday betas from the PRIOR sessions only (§2.3, §5.2).

    This is the leak-free, point-in-time residualization the report says it would write:
    beta/gamma at session ``t`` are estimated from the trailing ``lookback_sessions``
    *before* ``t`` -- so the estimate never sees test-period returns and no per-fold refit
    is needed (the estimator is causal like any other feature, §5.5).

    Implementation: aggregate the 2-variable OLS sufficient statistics per session, roll
    them causally (shifted by one session), and solve the 2x2 normal equations per row.
    Returns ``[scrip_code, date, beta, gamma]``.
    """
    stats = (
        grid.group_by(["scrip_code", "date"], maintain_order=True)
        .agg(
            s11=(pl.col("index_ret") ** 2).sum(),
            s22=(pl.col("sector_ret") ** 2).sum(),
            s12=(pl.col("index_ret") * pl.col("sector_ret")).sum(),
            s1y=(pl.col("index_ret") * pl.col("ret")).sum(),
            s2y=(pl.col("sector_ret") * pl.col("ret")).sum(),
        )
        .sort(["scrip_code", "date"])
    )
    roll = lambda c: (pl.col(c).rolling_sum(lookback_sessions, min_samples=3)
                      .shift(1).over("scrip_code"))
    stats = stats.with_columns([roll(c).alias(f"r_{c}") for c in
                                ("s11", "s22", "s12", "s1y", "s2y")])
    m = stats.select(["r_s11", "r_s22", "r_s12", "r_s1y", "r_s2y"]).to_numpy()
    beta = np.ones(len(m))
    gamma = np.zeros(len(m))
    for i in range(len(m)):
        s11, s22, s12, s1y, s2y = m[i]
        if not np.isfinite([s11, s22, s12, s1y, s2y]).all():
            continue
        det = s11 * s22 - s12 * s12
        if abs(det) < 1e-18:
            continue
        beta[i] = (s22 * s1y - s12 * s2y) / det
        gamma[i] = (s11 * s2y - s12 * s1y) / det
    return stats.select(["scrip_code", "date"]).with_columns(
        pl.Series("beta", beta), pl.Series("gamma", gamma)
    )


def residualize(grid: pl.DataFrame, cfg: Config,
                betas: pl.DataFrame | None = None) -> pl.DataFrame:
    """Add ``r_resid`` (and helper columns) to a grid with factors attached.

    Method comes from ``cfg.residualize.method``. For ``regression`` you may pass a
    precomputed ``betas`` frame (from :func:`fit_betas` on the fold's training rows);
    otherwise betas are fit on the full frame (fine for a quick pass, but the CV harness
    should pass fold-specific betas).
    """
    method = cfg.residualize.method
    if method == "demean_sector":
        sec_mean = (
            grid.group_by(["date", "minute", "sector"], maintain_order=True)
            .agg(_sec_mean=pl.col("ret").mean())
        )
        out = grid.join(sec_mean, on=["date", "minute", "sector"], how="left")
        out = out.with_columns((pl.col("ret") - pl.col("_sec_mean")).alias("r_resid")).drop(
            "_sec_mean"
        )
    elif method == "regression":
        if betas is None:
            mode = getattr(cfg.residualize, "beta_mode", "trailing")
            if mode == "trailing":
                betas = fit_betas_trailing(grid, int(cfg.residualize.beta_lookback_sessions))
            else:
                betas = fit_betas(grid)   # global (leaky) -- opt-in only
        join_on = ["scrip_code", "date"] if "date" in betas.columns else ["scrip_code"]
        out = grid.join(betas, on=join_on, how="left").with_columns(
            pl.col("beta").fill_null(1.0), pl.col("gamma").fill_null(0.0)
        )
        out = out.with_columns(
            (pl.col("ret")
             - pl.col("beta") * pl.col("index_ret")
             - pl.col("gamma") * pl.col("sector_ret")).alias("r_resid")
        )
    else:  # pragma: no cover
        raise ValueError(f"unknown residualize method: {method}")

    # residual cumulative log-price path per (scrip, date) -- the triple barrier walks it
    out = out.sort(["scrip_code", "date", "minute"]).with_columns(
        pl.col("r_resid").cum_sum().over(["scrip_code", "date"]).alias("resid_px")
    )
    return out


def add_residual_vol(grid: pl.DataFrame, cfg: Config,
                     tod_profile: TodVolProfile | None = None) -> pl.DataFrame:
    """Add ``sigma_resid`` -- a point-in-time EWMA residual vol, tod-normalized (§2.5).

    The EWMA is strictly backward (shifted by one bar so the current return is excluded).
    If a fitted :class:`TodVolProfile` is supplied, the raw EWMA vol is divided by the
    session-time multiplier so barrier widths are comparable across the day.
    """
    hl = int(cfg.labeling.sigma_halflife_bars)
    alpha = _ewm_halflife_alpha(hl)
    out = grid.sort(["scrip_code", "date", "minute"]).with_columns(
        pl.col("r_resid")
        .pow(2)
        .shift(1)
        .ewm_mean(alpha=alpha, min_samples=5, adjust=False)
        .over("scrip_code")
        .sqrt()
        .alias("sigma_resid_raw")
    )
    # floor tiny/na vols to a robust cross-sectional level
    floor = out["sigma_resid_raw"].median() or 1e-4
    out = out.with_columns(
        pl.col("sigma_resid_raw").fill_null(floor).clip(lower_bound=floor * 0.25)
    )
    if tod_profile is not None:
        mod = (market.session_open_min() + out["minute"]).to_numpy().astype(float)
        mult = tod_profile.multiplier(mod)
        out = out.with_columns(
            (pl.col("sigma_resid_raw") / pl.Series(mult)).alias("sigma_resid")
        )
    else:
        out = out.with_columns(pl.col("sigma_resid_raw").alias("sigma_resid"))
    return out
