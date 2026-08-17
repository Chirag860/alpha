"""Per-symbol "Crash" setup screen (short mean-reversion on high-vol US equities).

Turns a single instrument's daily OHLCV into the per-bar boolean signals the cross-sectional
portfolio layer consumes. **No portfolio decisions here** — no slot accounting, no borrow, no
fills; just "is this name a valid short setup at the close of day t, and what limit would I post
for day t+1". Leak-free: every value at ``t`` uses only data through the close of ``t``.

Baseline screen is the *book's* rule (EdgeRater restatement), deliberately, so results can be
checked against its published per-trade stats before any variation is layered on:

    price > $5   AND   21-day avg SHARE volume >= 1,000,000   AND   HV(100) > 100%   AND   CRSI >= 90

Two documented deviations from the book are available as opt-in params, off by default:
``min_dollar_volume`` (a turnover overlay) and ``use_median_volume`` (the conservative,
spike-insensitive liquidity variant). The book uses *mean share* volume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .indicators import connors_rsi, hv_annualized


@dataclass(frozen=True)
class CrashParams:
    """Screen + signal parameters. Defaults reproduce the book's baseline (5%/30 is the book's
    primary cell; the user's spec is 3%/30, so ``limit_pct`` defaults to 0.03 with 0.05 available)."""
    crsi_entry: float = 90.0            # short when CRSI >= this
    crsi_exit: float = 30.0             # cover when CRSI < this
    hv_min: float = 1.00               # HV(100) must exceed this (decimal; 1.00 == 100%)
    limit_pct: float = 0.03            # short limit = close * (1 + limit_pct); book primary = 0.05
    min_price: float = 5.0             # close must exceed this
    min_avg_volume: float = 1_000_000  # trailing-avg share volume floor
    avg_vol_window: int = 21           # lookback for the volume average
    min_dollar_volume: float = 0.0     # opt-in turnover overlay (0 = off); NOT in the book
    use_median_volume: bool = False    # opt-in conservative liquidity variant (median vs mean)
    rsi_period: int = 3
    streak_period: int = 2
    rank_window: int = 100


def _trailing_avg(x: np.ndarray, window: int, *, median: bool) -> np.ndarray:
    """Causal trailing mean/median ``[T]`` over ``window`` bars through ``t`` inclusive; NaN until
    a full window exists. Leak-free (never uses ``x[t+1..]``)."""
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    out = np.full(T, np.nan)
    agg = np.median if median else np.mean
    for t in range(window - 1, T):
        seg = x[t - window + 1:t + 1]
        if np.isnan(seg).any():
            continue
        out[t] = float(agg(seg))
    return out


@dataclass(frozen=True)
class CrashSignals:
    """Per-bar signal frame for one symbol; all arrays ``[T]`` on the symbol's own date grid."""
    crsi: np.ndarray
    hv: np.ndarray
    avg_volume: np.ndarray
    price_ok: np.ndarray               # close > min_price
    liquidity_ok: np.ndarray           # avg volume (+ optional dollar-vol) screen passed
    hv_ok: np.ndarray                  # HV(100) > hv_min
    tradeable: np.ndarray              # price & liquidity & indicators-warm (universe membership)
    entry_setup: np.ndarray            # tradeable & hv_ok & CRSI >= entry  (short candidate @ t)
    exit_signal: np.ndarray            # CRSI < exit  (cover a held name; screen-independent)
    limit_price: np.ndarray            # close * (1 + limit_pct): the T+1 sell-short limit


def crash_signals(close: np.ndarray, volume: np.ndarray,
                  params: CrashParams = CrashParams()) -> CrashSignals:
    """Compute the per-symbol Crash signal frame from daily close + share volume.

    ``entry_setup[t]`` means "at the close of day t this is a valid short candidate"; the portfolio
    posts a day-limit sell-short at ``limit_price[t]`` for day ``t+1``. ``exit_signal[t]`` means
    "CRSI lost its overbought condition" and any held position covers on the following session per
    the backtest's exit convention. Both are computed only from information available at ``t``.
    """
    close = np.asarray(close, dtype=float)
    volume = np.asarray(volume, dtype=float)
    p = params

    crsi = connors_rsi(close, p.rsi_period, p.streak_period, p.rank_window)
    hv = hv_annualized(close, p.rank_window)
    avg_vol = _trailing_avg(volume, p.avg_vol_window, median=p.use_median_volume)

    price_ok = close > p.min_price
    liquidity_ok = avg_vol >= p.min_avg_volume
    if p.min_dollar_volume > 0.0:            # opt-in turnover overlay (mean $ turnover)
        avg_dollar = _trailing_avg(close * volume, p.avg_vol_window, median=p.use_median_volume)
        liquidity_ok = liquidity_ok & (avg_dollar >= p.min_dollar_volume)
    liquidity_ok = np.nan_to_num(liquidity_ok, nan=False).astype(bool)

    warm = ~np.isnan(crsi) & ~np.isnan(hv)   # indicators fully warmed up (>= 102 bars)
    tradeable = price_ok & liquidity_ok & warm

    hv_ok = np.nan_to_num(hv > p.hv_min, nan=False).astype(bool)
    entry_setup = tradeable & hv_ok & np.nan_to_num(crsi >= p.crsi_entry, nan=False).astype(bool)
    exit_signal = np.nan_to_num(crsi < p.crsi_exit, nan=False).astype(bool) & warm

    limit_price = close * (1.0 + p.limit_pct)

    return CrashSignals(
        crsi=crsi, hv=hv, avg_volume=avg_vol,
        price_ok=price_ok, liquidity_ok=liquidity_ok, hv_ok=hv_ok,
        tradeable=tradeable, entry_setup=entry_setup, exit_signal=exit_signal,
        limit_price=limit_price,
    )
