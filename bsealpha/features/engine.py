"""Feature engine: assemble the point-in-time cross-sectional feature panel.

One codebase, run over historical replay or (in a deployment) a live stream, so offline
and online features are identical (§8.3 parity). The modeling substrate is the **common
1-minute grid** (§2.2): each row is ``(scrip_code, date, minute)`` -- the natural unit for
a pooled cross-sectional model whose groups are ``(date, minute)`` cross-sections (§4.2).

Discipline enforced here:

* every microstructure feature comes from the *last snapshot in the minute* or a
  strictly-backward rolling window -- no future bar leaks in;
* returns/factors/residuals are computed on the grid, residualized (§2.3), and the
  residual vol is a shifted EWMA;
* cross-sectional ranks are lagged by one minute (§3.3);
* per-day resets prevent overnight leakage (§0.4).

Returns ``(panel, feature_cols)``: the panel carries features plus the label-support
columns (``resid_px``, ``sigma_resid``, ``mid``, ``sector``, ...); ``feature_cols`` is the
model's input list.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .. import market
from ..config import Config
from .book import book_shape_expressions
from .cross_sectional import cross_sectional_rank
from .index_factor import attach_factors, compute_returns
from .microprice import microprice_expressions
from .ofi import ofi_frame
from .residualize import add_residual_vol, fit_betas, residualize
from .session import expiry_flag, session_feature_expressions
from .tradeflow import sign_trades, tradeflow_minute_features
from .volatility import TodVolProfile

_MU1_INV2 = (np.sqrt(2.0 / np.pi)) ** -2

# per-name microstructure features that get cross-sectional ranks (§3.2)
_RANKED = [
    "ofi_1m", "ofi_5m", "ofi_30m",
    "micro_minus_mid", "imb_top", "depth_imb_total", "signed_vol_frac",
    "spread_rel", "rv_5", "jump_5", "resid_mom_5", "large_print",
]
# common / session / index features kept raw (a rank across names is meaningless)
_RAW_EXTRA = [
    "wmid_minus_mid", "imb_all", "depth_ratio", "depth_slope",
    "log_depth_bid", "log_depth_ask", "spread_bps", "vwap_minus_mid",
    "trade_count", "rv_15", "rv_ratio", "resid_mom_15",
    "circuit_dist_upper", "circuit_dist_lower",
    "sin_tod", "cos_tod", "mins_to_close", "mins_to_flatten",
    "opening_flag", "squareoff_flag", "expiry_flag",
    "index_ret", "sector_ret", "dispersion", "vix_proxy",
]


def _minute_snapshot(depth: pl.DataFrame) -> pl.DataFrame:
    """Last full 5-level snapshot per ``(scrip, date, minute)`` with mid attached."""
    from ..bars.event_bars import attach_mid_micro

    dm = attach_mid_micro(depth).with_columns(
        pl.col("session_min").floor().cast(pl.Int64).alias("minute")
    )
    level_cols = [c for c in dm.columns if c.startswith(("bid_px_", "bid_qty_",
                                                         "ask_px_", "ask_qty_"))]
    return (
        dm.sort(["scrip_code", "date", "ts_ns"])
        .group_by(["scrip_code", "date", "minute"], maintain_order=True)
        .agg([pl.col(c).last() for c in level_cols] + [pl.col("mid").last()])
    )


def _minute_ofi(depth: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Minute-aggregated integrated OFI with 1/5/30-minute backward rolling sums."""
    snap = ofi_frame(depth, m=int(cfg.features.ofi_levels)).with_columns(
        pl.col("session_min").floor().cast(pl.Int64).alias("minute")
    )
    per_min = (
        snap.group_by(["scrip_code", "date", "minute"], maintain_order=True)
        .agg(ofi_min=pl.col("ofi_integrated").sum())
        .sort(["scrip_code", "date", "minute"])
    )
    return per_min.with_columns(
        pl.col("ofi_min").alias("ofi_1m"),
        pl.col("ofi_min").rolling_sum(5, min_samples=1).over(["scrip_code", "date"]).alias("ofi_5m"),
        pl.col("ofi_min").rolling_sum(30, min_samples=1).over(["scrip_code", "date"]).alias("ofi_30m"),
    ).drop("ofi_min")


def _volatility_and_momentum(grid: pl.DataFrame) -> pl.DataFrame:
    """Rolling residual vol, bipower/jump, and trailing residual momentum (all backward)."""
    g = grid.sort(["scrip_code", "date", "minute"]).with_columns(
        pl.col("r_resid").abs().alias("_absr"),
        pl.col("r_resid").pow(2).alias("_r2"),
    )
    g = g.with_columns(
        (pl.col("_absr") * pl.col("_absr").shift(1).over(["scrip_code", "date"]))
        .alias("_bp_prod")
    )
    grp = ["scrip_code", "date"]
    g = g.with_columns(
        pl.col("_r2").rolling_sum(5, min_samples=2).over(grp).alias("rv_5"),
        pl.col("_r2").rolling_sum(15, min_samples=3).over(grp).alias("rv_15"),
        (_MU1_INV2 * pl.col("_bp_prod").rolling_sum(5, min_samples=2).over(grp)).alias("bpv_5"),
        pl.col("r_resid").rolling_sum(5, min_samples=1).over(grp).shift(1).over(grp).alias("resid_mom_5"),
        pl.col("r_resid").rolling_sum(15, min_samples=1).over(grp).shift(1).over(grp).alias("resid_mom_15"),
    )
    g = g.with_columns(
        (pl.col("rv_5") - pl.col("bpv_5")).clip(lower_bound=0.0).alias("jump_5"),
        (pl.col("rv_5") / (pl.col("rv_15") + 1e-12)).alias("rv_ratio"),
    )
    return g.drop(["_absr", "_r2", "_bp_prod", "bpv_5"])


def _circuit_distance(grid: pl.DataFrame, meta: pl.DataFrame) -> pl.DataFrame:
    """Headroom to the upper/lower circuit band as % of price (feature + constraint, §3.2)."""
    bands = meta.select(["scrip_code", "circuit_band_pct"])
    g = grid.join(bands, on="scrip_code", how="left").with_columns(
        pl.col("circuit_band_pct").fill_null(20.0)
    )
    day_open = (
        g.group_by(["scrip_code", "date"], maintain_order=True)
        .agg(day_open=pl.col("mid").first())
    )
    g = g.join(day_open, on=["scrip_code", "date"], how="left")
    return g.with_columns(
        ((pl.col("day_open") * (1 + pl.col("circuit_band_pct") / 100.0) - pl.col("mid"))
         / pl.col("mid") * 100.0).alias("circuit_dist_upper"),
        ((pl.col("mid") - pl.col("day_open") * (1 - pl.col("circuit_band_pct") / 100.0))
         / pl.col("mid") * 100.0).alias("circuit_dist_lower"),
    ).drop("day_open")


# raw per-name microstructure columns produced by `build_raw_grid` -- the parity surface
# between the batch engine and the streaming engine (§8.3). Everything downstream is a
# deterministic function of these, so matching them guarantees full-feature parity.
RAW_MICRO_COLS = [
    "mid", "micro", "ret", "micro_minus_mid", "wmid_minus_mid", "imb_top", "imb_all",
    "depth_imb_total", "spread_bps", "depth_ratio", "depth_slope", "log_depth_bid",
    "log_depth_ask", "spread_rel", "ofi_1m", "ofi_5m", "ofi_30m",
    "signed_vol_frac", "large_print", "trade_count", "vwap_minus_mid",
]


def build_raw_grid(depth: pl.DataFrame, trades: pl.DataFrame, meta: pl.DataFrame,
                   cfg: Config) -> pl.DataFrame:
    """Build the per-name minute grid of RAW microstructure features (no cross-section).

    This is the portion a live system computes from the event stream; it is reproduced
    event-by-event by :mod:`bsealpha.features.streaming`, and the parity test asserts the
    two agree (§8.3). None of these features touches other names, so the streaming path can
    compute them per scrip in local-receipt order with O(1) state.
    """
    from ..bars.event_bars import common_minute_grid

    grid = common_minute_grid(depth, trades)
    grid = grid.join(meta.select(["scrip_code", "sector"]), on="scrip_code", how="left")
    grid = compute_returns(grid)

    snap = _minute_snapshot(depth)
    snap = snap.with_columns(microprice_expressions(int(cfg.features.ofi_levels)))
    snap = snap.with_columns(book_shape_expressions(int(cfg.features.ofi_levels)))
    snap_feats = snap.select(
        ["scrip_code", "date", "minute", "micro_minus_mid", "wmid_minus_mid",
         "imb_top", "imb_all", "depth_imb_total", "spread_bps", "depth_ratio",
         "depth_slope", "log_depth_bid", "log_depth_ask"]
    )
    grid = grid.join(snap_feats, on=["scrip_code", "date", "minute"], how="left")

    grid = grid.sort(["scrip_code", "date", "minute"]).with_columns(
        (pl.col("spread_bps")
         / pl.col("spread_bps").rolling_mean(30, min_samples=3).over("scrip_code").clip(lower_bound=1e-6)
         ).alias("spread_rel")
    )
    grid = grid.join(_minute_ofi(depth, cfg), on=["scrip_code", "date", "minute"], how="left")

    signed = sign_trades(trades)
    tf = tradeflow_minute_features(signed)
    grid = grid.join(
        tf.select(["scrip_code", "date", "minute", "signed_vol_frac", "large_print",
                   "trade_count", "vwap_trade"]),
        on=["scrip_code", "date", "minute"], how="left",
    ).with_columns(
        ((pl.col("vwap_trade") - pl.col("mid")) / pl.col("mid") * 1e4)
        .fill_null(0.0).alias("vwap_minus_mid")
    )
    return grid.sort(["date", "minute", "scrip_code"])


def finalize_features(grid: pl.DataFrame, meta: pl.DataFrame, cfg: Config, *,
                      tod_profile: TodVolProfile | None = None,
                      betas: pl.DataFrame | None = None,
                      fit_tod: bool = True) -> tuple[pl.DataFrame, list[str]]:
    """Cross-sectional + residual + rank layer applied to a raw per-name grid.

    Deterministic function of the raw grid, so it is *shared* between the batch and
    streaming paths -- run it on whichever raw grid you have and the features are identical.
    """
    grid = attach_factors(grid)
    if cfg.residualize.method == "regression" and betas is None:
        betas = fit_betas(grid)
    grid = residualize(grid, cfg, betas=betas)

    if tod_profile is None and fit_tod:
        mod = (market.session_open_min() + grid["minute"]).to_numpy().astype(float)
        tod_profile = TodVolProfile(n_bins=int(cfg.features.tod_vol_bins)).fit(
            grid["r_resid"].to_numpy(), mod
        )
    grid = add_residual_vol(grid, cfg, tod_profile=tod_profile)

    grid = _volatility_and_momentum(grid)
    grid = grid.with_columns(session_feature_expressions())
    grid = grid.with_columns(expiry_flag(pl.col("date")))
    grid = _circuit_distance(grid, meta)

    # -- tidy: fill nulls in features, then cross-sectional ranks ---------
    raw_feats = _RANKED + _RAW_EXTRA
    grid = grid.with_columns([pl.col(c).fill_null(0.0).fill_nan(0.0) for c in raw_feats
                              if c in grid.columns])
    grid = cross_sectional_rank(
        grid, [c for c in _RANKED if c in grid.columns],
        lag_minutes=int(cfg.features.cross_sectional_lag_min),
    )

    rank_cols: list[str] = []
    for c in _RANKED:
        rank_cols += [f"{c}_xs", f"{c}_xsec"]
    feature_cols = [c for c in (raw_feats + rank_cols) if c in grid.columns]
    grid = grid.with_columns([pl.col(c).fill_null(0.0).fill_nan(0.0) for c in feature_cols])
    return grid, feature_cols


def build_features_bars_only(grid: pl.DataFrame, meta: pl.DataFrame, cfg: Config, *,
                             tod_profile: TodVolProfile | None = None,
                             betas: pl.DataFrame | None = None,
                             fit_tod: bool = True) -> tuple[pl.DataFrame, list[str]]:
    """Reduced feature set for BARS-ONLY data -- no depth, no trade tape (§3.1).

    Free vendors (yfinance / openchart) give 1-minute OHLCV only, so the microstructure
    families (OFI, micro-price, book shape, signed flow) are **unavailable**. What remains is
    still a coherent cross-sectional model: residualized returns, residual momentum/reversal,
    realized vol / bipower / jump, session structure, circuit distance, index/sector/
    dispersion, and the cross-sectional ranks of those. Expect materially weaker signal than
    the full depth-based model -- this is the honest ceiling of free data.

    ``grid`` must be a common-minute grid (``scrip_code, date, minute, mid, high, low,
    turnover, close``); ``meta`` needs ``sector`` and ``circuit_band_pct``.
    """
    grid = grid.join(meta.select(["scrip_code", "sector"]), on="scrip_code", how="left")
    grid = compute_returns(grid)
    return finalize_features(grid, meta, cfg, tod_profile=tod_profile, betas=betas,
                             fit_tod=fit_tod)


def build_features(depth: pl.DataFrame, trades: pl.DataFrame, meta: pl.DataFrame,
                   cfg: Config, *, tod_profile: TodVolProfile | None = None,
                   betas: pl.DataFrame | None = None,
                   fit_tod: bool = True) -> tuple[pl.DataFrame, list[str]]:
    """Build the full feature panel: ``finalize_features(build_raw_grid(...))``.

    The two-stage split (raw per-name grid -> cross-sectional finalize) is what lets the
    live/streaming path share the exact downstream code (§8.3). Signature and output are
    unchanged from before the refactor.
    """
    raw = build_raw_grid(depth, trades, meta, cfg)
    return finalize_features(raw, meta, cfg, tod_profile=tod_profile, betas=betas,
                             fit_tod=fit_tod)
