"""Trend + carry signals on a daily multi-asset panel.

All arrays are ``[T, N]`` (T daily observations, N instruments), aligned to a common date grid.

**No-lookahead convention (the one correctness property that matters):** every signal at row
``t`` is computed from information available at the *close of day t* (returns/closes up to and
including ``t``). The backtest then holds that position over day ``t+1`` and earns ``ret[t+1]``
— i.e. positions are lagged one day before being multiplied by realized returns. Nothing here
peeks at the future; the lag lives in :mod:`bsealpha.trend.backtest`.
"""

from __future__ import annotations

import numpy as np


def log_returns(close: np.ndarray) -> np.ndarray:
    """Daily log returns ``[T, N]`` with row 0 = 0. NaNs/inf -> 0."""
    close = np.asarray(close, dtype=float)
    r = np.zeros_like(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        r[1:] = np.log(close[1:] / close[:-1])
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def ewma_vol(ret: np.ndarray, halflife: float, *, floor: float = 1e-6) -> np.ndarray:
    """Causal EWMA daily volatility ``[T, N]`` using returns through row ``t`` (inclusive).

    ``vol[t]`` is known at the close of day ``t``, so using it to size the day-``t`` decision is
    leak-free. A small ``floor`` prevents divide-by-zero on dead/constant series.
    """
    ret = np.asarray(ret, dtype=float)
    T, N = ret.shape
    alpha = 1.0 - 0.5 ** (1.0 / float(halflife))
    var = np.zeros((T, N))
    v = ret[0] ** 2 if T else np.zeros(N)
    for t in range(T):
        v = alpha * ret[t] ** 2 + (1.0 - alpha) * v
        var[t] = v
    return np.maximum(np.sqrt(var), floor)


def tsmom_signal(close: np.ndarray, lookbacks: list[int], vol: np.ndarray,
                 *, cap: float = 2.0) -> np.ndarray:
    """Vol-scaled time-series-momentum signal ``[T, N]`` in roughly ``[-1, 1]`` per instrument.

    For each lookback ``L`` (trading days), take the trailing ``L``-day log return known at ``t``,
    risk-normalize it by ``vol * sqrt(L)`` (so a 1-month and a 12-month trend are comparable),
    squash with ``tanh``, and average across lookbacks. This is the standard risk-adjusted TSMOM
    of Moskowitz-Ooi-Pedersen, generalized to multiple horizons.
    """
    close = np.asarray(close, dtype=float)
    logc = np.log(np.maximum(close, 1e-12))
    T, N = close.shape
    sig = np.zeros((T, N))
    for L in lookbacks:
        L = int(L)
        m = np.zeros((T, N))
        m[L:] = logc[L:] - logc[:-L]                 # trailing L-day return, known at t
        z = m / (vol * np.sqrt(L) + 1e-12)           # risk-normalized trend strength
        sig += np.tanh(np.clip(z, -cap, cap))
    return sig / max(len(lookbacks), 1)


def carry_signal(swap_long: np.ndarray, swap_short: np.ndarray, price: np.ndarray,
                 contract_size: np.ndarray) -> np.ndarray:
    """Static per-instrument carry score ``[N]`` from broker swap (financing) rates.

    Carry favors the side that is *paid* to hold. We annualize the net financing edge as a yield
    and cross-sectionally z-score it so it is unit-agnostic across instruments/brokers. This is a
    LIVE tilt (swaps are a current attribute; there is no historical series in MT5 bars), so the
    backtest treats it as a small static overlay, not a validated historical signal.
    """
    sl = np.asarray(swap_long, dtype=float)
    ss = np.asarray(swap_short, dtype=float)
    px = np.maximum(np.asarray(price, dtype=float), 1e-9)
    cs = np.maximum(np.asarray(contract_size, dtype=float), 1e-9)
    # net daily financing edge for being long (points/ccy per lot) -> per-unit-notional yield
    edge = (sl - ss) / (px * cs)
    ann = edge * 252.0
    if np.nanstd(ann) < 1e-12:
        return np.zeros_like(ann)
    return np.nan_to_num((ann - np.nanmean(ann)) / (np.nanstd(ann) + 1e-12))
