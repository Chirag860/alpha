"""Cross-sectional book construction: score -> market/sector-neutral positions (§7.1).

The model output is a cross-sectional score. Turning it into a book means: neutralize the
market and sector factors (else effective breadth collapses, §0.5), cap participation at a
fraction of each name's BSE ADV (impact dominates fees in thin names, §6.4), cap sector
gross, drop clips below the ₹20/order brokerage floor, and beta-neutralize the residual.

A **no-trade band** (:func:`rebalance`) then holds the book rather than tracking the target
continuously -- under quadratic costs the optimal policy is a band around a moving target
(§7.1). Turnover is the enemy.
"""

from __future__ import annotations

import numpy as np


def build_book(scores: np.ndarray, betas: np.ndarray, sectors: np.ndarray,
               adv_bse: np.ndarray, *, gross_target: float,
               max_participation: float = 0.03, min_clip: float = 5e5,
               max_names: int = 60, sector_cap: float = 0.25,
               p_act: np.ndarray | None = None) -> np.ndarray:
    """Return target rupee positions: market- & sector-neutral, participation-capped (§7.1).

    Parameters
    ----------
    scores
        Cross-sectional score per name (higher = more attractive long).
    betas
        Intraday beta to the market factor, per name.
    sectors
        Sector label per name.
    adv_bse
        Median BSE daily turnover per name (the participation base, §6.4).
    gross_target
        Target gross book in rupees.
    p_act
        Optional meta-model P(act), used to gate/scale conviction.
    """
    scores = np.asarray(scores, float).copy()
    betas = np.asarray(betas, float)
    adv_bse = np.asarray(adv_bse, float)
    sectors = np.asarray(sectors)
    n = len(scores)
    if p_act is not None:
        scores = scores * np.asarray(p_act, float)

    s = scores - np.mean(scores)                       # market-neutral in score space
    for sec in np.unique(sectors):                      # sector-neutral
        m = sectors == sec
        s[m] -= s[m].mean()

    keep = np.argsort(-np.abs(s))[:max_names]           # trade only the strongest names
    w = np.zeros(n)
    w[keep] = s[keep]
    if np.abs(w).sum() == 0:
        return w
    w = w / np.abs(w).sum() * gross_target              # scale to gross target

    cap = max_participation * adv_bse                   # participation cap (§6.4)
    w = np.sign(w) * np.minimum(np.abs(w), cap)

    for sec in np.unique(sectors):                      # sector gross cap
        m = sectors == sec
        g = np.abs(w[m]).sum()
        if g > sector_cap * gross_target and g > 0:
            w[m] *= sector_cap * gross_target / g

    denom = float(betas @ betas)                        # residual beta-neutralize
    if denom > 1e-12:
        w -= betas * (w @ betas) / denom

    w[np.abs(w) < min_clip] = 0.0                       # brokerage floor (§6.2)
    return w


def rebalance(current: np.ndarray, target: np.ndarray, *, band_frac: float,
              min_clip: float) -> np.ndarray:
    """No-trade band: only move a name when the gap clears the band and the min clip (§7.1).

    The band is a **fraction of the (larger of) current/target position**, not a fixed bps
    of notional -- so a name is left alone until the target drifts by ``band_frac`` of the
    position. This is the quadratic-cost-optimal "band around a moving target" (§7.1) and
    the single biggest lever on turnover (grid-search it on *net* Sharpe, and count that in
    the trial budget). Returns the trade vector (0 where inside the band).
    """
    current = np.asarray(current, float)
    target = np.asarray(target, float)
    gap = target - current
    ref = np.maximum(np.maximum(np.abs(target), np.abs(current)), 1.0)
    inside_band = np.abs(gap) < band_frac * ref
    below_clip = np.abs(gap) < min_clip
    trade = np.where(inside_band | below_clip, 0.0, gap)
    return trade


def realized_beta(position_returns: np.ndarray, market_returns: np.ndarray) -> float:
    """Realized book beta = cov(book P&L, market) / var(market). Monitor it live (§7.3)."""
    b = np.asarray(position_returns, float)
    m = np.asarray(market_returns, float)
    if m.std() < 1e-12:
        return 0.0
    return float(np.cov(b, m)[0, 1] / np.var(m))
