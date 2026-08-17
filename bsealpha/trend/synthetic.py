"""Synthetic daily multi-asset panel with a **realistic correlation structure** — for tests/demo.

Real trend universes are not independent: equity indices move together, the energy complex
trends as one, and in a risk-off shock *everything* correlates and vol spikes. A synthetic that
ignores this (independent instruments) hands a trend follower a fantasy Sharpe and hides the two
things that actually decide real P&L -- diversification and crisis behaviour. So each instrument
here is::

    logret = beta_market * market_factor + beta_class * class_factor + idiosyncratic

giving within-class correlation ~0.5-0.8 and cross-class ~0.1-0.3, plus an occasional clustered
**crisis regime** (amplified, negatively-drifted market shocks) that pushes correlations toward 1.

Trends live in the *factors* (the whole class trends together) and in per-instrument idiosyncratic
drift, both scaled by ``frac_trending`` so a ``frac_trending=0`` panel is genuinely trendless
correlated noise. A correct trend engine still earns a clearly positive -- but now *realistic*,
not absurd -- Sharpe, and its diversification/vol-target machinery is actually exercised.
"""

from __future__ import annotations

import numpy as np
import polars as pl

_CLASSES = ["FX", "METAL", "INDEX", "ENERGY", "CRYPTO"]


def _regime_drift(n_days: int, rng: np.random.Generator, strength: float, dvol: float) -> np.ndarray:
    """Slowly sign-flipping drift (trend) — segments of 60-180 days, magnitude ``strength*dvol``."""
    drift = np.zeros(n_days)
    if strength <= 0:
        return drift
    i, sign = 0, rng.choice([-1.0, 1.0])
    while i < n_days:
        seg = int(rng.integers(60, 180))
        drift[i:i + seg] = sign * strength * dvol
        sign *= -1.0
        i += seg
    return drift


def _crisis_state(n_days: int, rng: np.random.Generator, *, p_enter: float = 0.010,
                  p_exit: float = 0.06, mult: float = 2.5) -> np.ndarray:
    """Two-state clustered vol regime: 1.0 (calm) or ``mult`` (crisis). Markov-persistent."""
    state = np.ones(n_days)
    s = 1.0
    for t in range(n_days):
        if s == 1.0 and rng.random() < p_enter:
            s = mult
        elif s > 1.0 and rng.random() < p_exit:
            s = 1.0
        state[t] = s
    return state


def generate_daily_panel(n_inst: int = 24, n_days: int = 1800, *, seed: int = 0,
                         frac_trending: float = 0.7, ann_vol: float = 0.15,
                         trend_strength: float = 0.15, crisis_intensity: float = 1.0):
    """Return ``(grid, meta)`` polars frames matching the canonical daily layout.

    ``crisis_intensity`` scales the risk-off drawdown depth (0 disables crises).
    """
    rng = np.random.default_rng(seed)
    dvol = ann_vol / np.sqrt(252.0)
    start = np.datetime64("2015-01-05")
    dates = np.array([start + np.timedelta64(i, "D") for i in range(n_days)])

    # -- common factors (correlation source) --------------------------------
    ft = float(np.clip(frac_trending, 0.0, 1.0))
    state = _crisis_state(n_days, rng)
    crisis = state > 1.0
    # market factor: mild trend (scaled by frac_trending), crisis-amplified & risk-off
    market = (_regime_drift(n_days, rng, 0.3 * trend_strength * ft, dvol)
              + rng.normal(0.0, dvol, n_days) * state)
    market[crisis] -= 0.20 * dvol * float(crisis_intensity)
    # per-class factor: the main correlated trend within a complex
    class_factor = {
        cls: (_regime_drift(n_days, rng, trend_strength * ft, dvol)
              + rng.normal(0.0, dvol, n_days))
        for cls in _CLASSES
    }

    grid_rows, meta_rows = [], []
    for k in range(n_inst):
        cls = _CLASSES[k % len(_CLASSES)]
        trending = k < int(ft * n_inst)
        beta_m = float(rng.uniform(0.3, 0.6))
        beta_c = float(rng.uniform(0.4, 0.7))
        idio_std = dvol * np.sqrt(max(1e-6, 1.0 - beta_m ** 2 - beta_c ** 2))
        idio_drift = _regime_drift(n_days, rng, trend_strength, dvol) if trending else 0.0

        logret = (beta_m * market + beta_c * class_factor[cls]
                  + idio_drift + rng.normal(0.0, idio_std, n_days))
        price = 100.0 * np.exp(np.cumsum(logret))
        for i in range(n_days):
            p = float(price[i])
            grid_rows.append({"date": str(dates[i]), "symbol": f"SYN{k:02d}", "asset_class": cls,
                              "open": p, "high": p * 1.003, "low": p * 0.997, "close": p})
        meta_rows.append({"symbol": f"SYN{k:02d}", "asset_class": cls,
                          "spread_bps": float(rng.uniform(0.5, 3.0)), "contract_size": 1.0,
                          "swap_long": float(rng.normal(0, 0.5)), "swap_short": float(rng.normal(0, 0.5)),
                          "currency": "USD"})
    grid = (pl.DataFrame(grid_rows)
            .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
            .sort(["date", "symbol"]))
    meta = pl.DataFrame(meta_rows)
    return grid, meta
