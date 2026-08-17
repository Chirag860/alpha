"""ConnorsRSI + realized-volatility indicators for the "Crash" short strategy.

Reference: Larry Connors & Connors Research, *Buy the Fear, Sell the Greed* (2018),
strategy #2 "Crash"; public rule statement at StockCharts ChartSchool (ConnorsRSI) and
EdgeRater. Every definition here follows the canonical statement exactly — the places where
libraries commonly diverge are called out inline, because near a 90/30 threshold those
divergences flip signals.

All functions operate on a **single instrument's** 1-D close series (float ``[T]``), oldest
first. The portfolio/cross-section layer lives elsewhere; this file is pure, per-symbol,
leak-free indicator math (value at index ``t`` uses only closes through ``t``).

**The one bug to avoid (flagged by review):** ``pandas.Series.ewm(adjust=False)`` seeds the
average with the *first observation*, not Wilder's SMA-of-first-``n`` seed, and produces a
materially different RSI (e.g. 50.7 vs 70.5 on Wilder's own worked example). We implement
Wilder's smoothing explicitly. Do not "simplify" :func:`wilder_rsi` to ``ewm``.
"""

from __future__ import annotations

import numpy as np


def wilder_rsi(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI of a 1-D series, period ``n``. Returns ``[T]`` with NaN during warm-up.

    Seed = simple mean of the first ``n`` gains and losses (placed at index ``n``); thereafter
    ``avg = (avg*(n-1) + current) / n`` (Wilder smoothing, alpha = 1/n). The first valid RSI is
    at index ``n``. Used for both the price RSI (n=3) and the streak RSI (n=2) of ConnorsRSI.

    Flat-window convention: if the trailing average gain *and* loss are both 0 (a perfectly flat
    stretch — common on the integer streak series and in illiquid names), RSI is mathematically
    undefined; we return **50** (neutral), matching the most defensible vendor behaviour. Callers
    that care can detect these bars separately.
    """
    v = np.asarray(values, dtype=float)
    T = v.shape[0]
    out = np.full(T, np.nan)
    if T <= period:
        return out

    delta = np.diff(v)                       # length T-1, delta[i] corresponds to values[i+1]
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    # Seed with the SMA of the first `period` gains/losses -> placed at index `period`.
    avg_gain = gain[:period].mean()
    avg_loss = loss[:period].mean()
    out[period] = _rsi_from_avgs(avg_gain, avg_loss)

    for i in range(period, T - 1):           # delta index i -> writes RSI at values index i+1
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        out[i + 1] = _rsi_from_avgs(avg_gain, avg_loss)
    return out


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    total = avg_gain + avg_loss
    if total <= 0.0:                         # flat window: 0/0 -> neutral
        return 50.0
    return 100.0 * avg_gain / total          # == 100 - 100/(1+RS), avoids the 1/0 branch


def streak(close: np.ndarray) -> np.ndarray:
    """Consecutive up/down close streak ``[T]`` (int-valued float). +k = k up-closes in a row.

    Rules (StockCharts): an up close extends a positive streak or starts +1; a down close extends
    a negative streak or starts -1; an **unchanged close hard-resets the streak to 0** (it does
    not carry the previous streak forward). Index 0 = 0.
    """
    c = np.asarray(close, dtype=float)
    T = c.shape[0]
    s = np.zeros(T)
    for t in range(1, T):
        if c[t] > c[t - 1]:
            s[t] = s[t - 1] + 1 if s[t - 1] > 0 else 1.0
        elif c[t] < c[t - 1]:
            s[t] = s[t - 1] - 1 if s[t - 1] < 0 else -1.0
        else:
            s[t] = 0.0
    return s


def roc1(close: np.ndarray) -> np.ndarray:
    """1-day rate of change in percent ``[T]``; index 0 = NaN (no prior close)."""
    c = np.asarray(close, dtype=float)
    out = np.full(c.shape[0], np.nan)
    out[1:] = 100.0 * (c[1:] / c[:-1] - 1.0)
    return out


def percent_rank(values: np.ndarray, window: int = 100) -> np.ndarray:
    """PercentRank of ``values[t]`` within the ``window`` values *strictly before* t ``[T]``.

    ``pr[t] = 100 * count(values[t-window .. t-1] < values[t]) / window`` — strict ``<`` (ties do
    not count), comparison window excludes today, fixed denominator ``window``. Emits NaN until a
    full window of non-NaN priors exists (so on the 1-day ROC series, whose index 0 is NaN, the
    first value appears at index ``window + 1``).
    """
    v = np.asarray(values, dtype=float)
    T = v.shape[0]
    out = np.full(T, np.nan)
    for t in range(window, T):
        prior = v[t - window:t]
        x = v[t]
        if np.isnan(x) or np.isnan(prior).any():
            continue
        out[t] = 100.0 * np.count_nonzero(prior < x) / window
    return out


def connors_rsi(close: np.ndarray, rsi_period: int = 3, streak_period: int = 2,
                rank_window: int = 100) -> np.ndarray:
    """ConnorsRSI(3,2,100) ``[T]`` in [0,100] = mean of the three components.

    ``CRSI = ( RSI(close, 3) + RSI(streak, 2) + PercentRank(ROC(1), 100) ) / 3``. NaN until all
    three components are defined; the binding constraint is PercentRank, so the first valid value
    is at index ``rank_window + 1`` (i.e. the 102nd close for the default 100). Returns NaN, never
    a partial average.
    """
    close = np.asarray(close, dtype=float)
    rsi_c = wilder_rsi(close, rsi_period)
    rsi_s = wilder_rsi(streak(close), streak_period)
    pr = percent_rank(roc1(close), rank_window)
    return (rsi_c + rsi_s + pr) / 3.0        # NaN in any component -> NaN result


def hv_annualized(close: np.ndarray, window: int = 100, trading_days: int = 252) -> np.ndarray:
    """Annualized close-to-close historical volatility ``[T]`` (decimal; 1.00 == 100%).

    ``HV[t] = stdev( logret[t-window+1 .. t], ddof=1 ) * sqrt(252)`` over ``window`` daily log
    returns. Sample std (ddof=1) per the textbook convention; note Connors' AmiBroker research
    used population std (ddof=0), a ~0.5% relative difference at window=100 that only matters on
    the boundary. Emits NaN until ``window`` returns exist (index ``window``).
    """
    c = np.asarray(close, dtype=float)
    T = c.shape[0]
    out = np.full(T, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(c[1:] / c[:-1])           # length T-1, r[i] -> close index i+1
    ann = np.sqrt(float(trading_days))
    for t in range(window, T):
        seg = r[t - window:t]                # window returns ending at close index t
        if np.isnan(seg).any():
            continue
        out[t] = float(np.std(seg, ddof=1)) * ann
    return out
