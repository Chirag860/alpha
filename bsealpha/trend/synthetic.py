"""Synthetic daily multi-asset panel with genuine, persistent trends — for tests/demo.

Each instrument is a geometric path with a slowly regime-switching drift (so time-series
momentum is actually present and a correct trend engine should earn a clearly positive Sharpe),
plus idiosyncratic noise. A configurable fraction of instruments are pure random walks (no
trend) so the risk-parity/diversification machinery is exercised too.
"""

from __future__ import annotations

import numpy as np
import polars as pl

_CLASSES = ["FX", "METAL", "INDEX", "ENERGY", "CRYPTO"]


def generate_daily_panel(n_inst: int = 24, n_days: int = 1800, *, seed: int = 0,
                         frac_trending: float = 0.7, ann_vol: float = 0.15,
                         trend_strength: float = 0.6):
    """Return ``(grid, meta)`` polars frames matching the canonical daily layout."""
    rng = np.random.default_rng(seed)
    dvol = ann_vol / np.sqrt(252.0)
    start = np.datetime64("2015-01-05")
    dates = np.array([start + np.timedelta64(i, "D") for i in range(n_days)])

    grid_rows = []
    meta_rows = []
    for k in range(n_inst):
        cls = _CLASSES[k % len(_CLASSES)]
        trending = k < int(frac_trending * n_inst)
        # regime-switching drift: flip sign every ~60-180 days, scaled by trend_strength
        drift = np.zeros(n_days)
        if trending:
            i = 0
            sign = rng.choice([-1.0, 1.0])
            while i < n_days:
                seg = int(rng.integers(60, 180))
                drift[i:i + seg] = sign * trend_strength * dvol
                sign *= -1.0
                i += seg
        shocks = rng.normal(0.0, dvol, n_days)
        logret = drift + shocks
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
