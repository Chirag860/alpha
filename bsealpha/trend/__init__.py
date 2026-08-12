"""Trend + carry: a rules-based, vol-targeted daily managed-futures system.

Signal (rules, no fitted predictor) -> inverse-vol risk parity -> volatility target -> costs ->
honest OOS validation. Designed for a diversified MT5 basket (FX / metals / indices / energy /
crypto) at a daily horizon, where a real, documented edge (time-series momentum) survives costs.
"""

from __future__ import annotations

from .backtest import BacktestResult, backtest, compute_metrics, finalize_weights
from .config import TrendParams, load_trend_params
from .data import load_daily_panel, meta_arrays, simple_returns, to_matrices
from .portfolio import apply_no_trade_band, target_weights
from .run import compute_book, run_trend_backtest
from .signals import carry_signal, ewma_vol, log_returns, tsmom_signal
from .validate import TrendValidation, format_report, validate_book

__all__ = [
    "TrendParams", "load_trend_params",
    "load_daily_panel", "to_matrices", "simple_returns", "meta_arrays",
    "log_returns", "ewma_vol", "tsmom_signal", "carry_signal",
    "target_weights", "apply_no_trade_band",
    "backtest", "compute_metrics", "finalize_weights", "BacktestResult",
    "compute_book", "run_trend_backtest",
    "validate_book", "format_report", "TrendValidation",
]
