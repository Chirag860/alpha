"""Backtest: Indian cost stack, queue fill sim, event-driven engine, metrics."""

from __future__ import annotations

from .costs import CostParams, impact_rupees, leg_cost_rupees, round_trip_cost_bps
from .engine import run_backtest
from .fill_sim import PassiveFillSimulator, PassiveOrder, markouts, walk_the_book
from .metrics import BacktestMetrics, max_drawdown, sharpe, sortino

__all__ = [
    "CostParams",
    "leg_cost_rupees",
    "impact_rupees",
    "round_trip_cost_bps",
    "run_backtest",
    "PassiveFillSimulator",
    "PassiveOrder",
    "walk_the_book",
    "markouts",
    "BacktestMetrics",
    "sharpe",
    "sortino",
    "max_drawdown",
]
