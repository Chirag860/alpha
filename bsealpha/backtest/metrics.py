"""Backtest performance metrics (§6, §7).

Sharpe, Sortino, max drawdown, turnover, hit rate, and a capacity estimate. All computed
on the **daily** P&L series (T = days, per §5.4), with effective breadth reported
alongside because raw Sharpe over a correlated cross-section is misleading (§0.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BacktestMetrics:
    sharpe: float = 0.0
    gross_sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    turnover_x: float = 0.0            # daily gross turnover / gross book
    hit_rate: float = 0.0
    n_trades: int = 0
    total_cost_rupees: float = 0.0
    total_impact_rupees: float = 0.0
    capacity_rupees: float = 0.0
    effective_breadth: float = 0.0
    markout_bps: dict = field(default_factory=dict)
    daily_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    gross_daily_returns: np.ndarray = field(default_factory=lambda: np.array([]))

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["daily_returns"] = list(map(float, self.daily_returns))
        d["gross_daily_returns"] = list(map(float, self.gross_daily_returns))
        return d


def sharpe(daily_returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(daily_returns, float)
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def sortino(daily_returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(daily_returns, float)
    downside = r[r < 0]
    dd = downside.std(ddof=1) if downside.size > 1 else 0.0
    if dd == 0:
        return 0.0
    return float(r.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(daily_returns: np.ndarray) -> float:
    """Max drawdown of the cumulative P&L curve (as a fraction of peak equity path)."""
    r = np.asarray(daily_returns, float)
    if r.size == 0:
        return 0.0
    equity = np.cumsum(r)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(-dd.min())


def hit_rate(trade_pnls: np.ndarray) -> float:
    p = np.asarray(trade_pnls, float)
    return float((p > 0).mean()) if p.size else 0.0


def capacity_estimate(adv_bse: np.ndarray, max_participation: float,
                      turns_per_day: float) -> float:
    """Sustainable gross book from the participation budget (§7.4).

    ``daily one-way budget = sum(participation * ADV)``; ``gross ≈ budget / turns_per_day``.
    """
    budget = float(np.sum(max_participation * np.asarray(adv_bse, float)))
    return budget / max(turns_per_day, 1e-9)
