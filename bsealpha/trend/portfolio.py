"""Position sizing: combine signals, size by inverse-vol, target portfolio volatility, cap.

The construction is deliberately parameter-light (no fitted weights to overfit):

1. combine trend + a static carry tilt, clip;
2. **naive risk parity** — position ∝ signal / vol, so every instrument contributes equal risk;
3. **analytic volatility target** — scale the book each day so its ex-ante daily vol (under a
   zero-cross-correlation approximation) equals ``target_ann_vol / sqrt(252)``;
4. cap per-instrument weight and gross leverage.

All inputs are known at day ``t`` (``signal`` and ``vol`` are causal), so ``weights[t]`` is
leak-free; the backtest lags it one day before earning returns.
"""

from __future__ import annotations

import numpy as np

from .config import TrendParams


def target_weights(signal: np.ndarray, vol: np.ndarray, params: TrendParams,
                   carry: np.ndarray | None = None) -> np.ndarray:
    """Signed notional weights ``[T, N]`` (fraction of NAV per instrument)."""
    comb = np.asarray(signal, dtype=float).copy()
    if carry is not None and params.carry_weight > 0:
        comb = comb + float(params.carry_weight) * np.asarray(carry, dtype=float)[None, :]
    comb = np.clip(comb, -float(params.signal_cap), float(params.signal_cap))

    # analytic vol target: w = daily_target * comb / (vol * ||comb||_2)  => ex-ante daily
    # portfolio vol (zero-corr approx) = daily_target.
    denom = np.sqrt(np.sum(comb ** 2, axis=1, keepdims=True))
    daily_target = float(params.target_ann_vol) / np.sqrt(252.0)
    w = daily_target * comb / (np.asarray(vol, dtype=float) * (denom + 1e-12))

    mw = float(params.max_weight_per_instrument)
    w = np.clip(w, -mw, mw)
    gross = np.sum(np.abs(w), axis=1, keepdims=True)
    w = w * np.minimum(1.0, float(params.max_gross_leverage) / np.maximum(gross, 1e-12))
    return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)


def apply_no_trade_band(weights: np.ndarray, band: float) -> np.ndarray:
    """Suppress small day-to-day weight changes to cut turnover (hysteresis).

    Only rebalance an instrument when its target moves more than ``band`` (in weight units)
    from the currently-held weight; otherwise carry yesterday's weight forward.
    """
    if band <= 0:
        return weights
    T, N = weights.shape
    held = np.zeros((T, N))
    cur = np.zeros(N)
    for t in range(T):
        move = np.abs(weights[t] - cur) > band
        cur = np.where(move, weights[t], cur)
        held[t] = cur
    return held
