"""Glue: panel -> signals -> weights -> backtest. One call runs the whole trend book."""

from __future__ import annotations

import numpy as np
import polars as pl

from .backtest import BacktestResult, backtest, finalize_weights
from .config import TrendParams
from .data import meta_arrays, simple_returns, to_matrices
from .portfolio import target_weights
from .signals import carry_signal, ewma_vol, log_returns, tsmom_signal


def compute_book(panel: pl.DataFrame, meta: pl.DataFrame, params: TrendParams,
                 *, include_carry: bool = False) -> dict:
    """Compute signals, vol, and target weights. Returns the intermediate arrays.

    ``include_carry`` adds the (live) carry tilt to sizing. It is OFF for backtests/validation
    because MT5 gives no historical swap series — carry is a live overlay, not a backtested edge.
    """
    dates, symbols, close = to_matrices(panel)
    # data-quality filter: drop instruments that trade too few days (dead/gappy/illiquid CFDs,
    # whose forward-filled flat stretches are pure noise + cost). Keep the rest.
    _r0 = simple_returns(close)
    active = (np.abs(_r0) > 1e-9).mean(axis=0)
    keep = active >= float(params.min_active_frac)
    if 5 <= int(keep.sum()) < len(symbols):
        close = close[:, keep]
        symbols = [s for s, k in zip(symbols, keep) if k]
    ret = simple_returns(close)
    lr = log_returns(close)
    vol = ewma_vol(lr, params.vol_halflife)
    sig = tsmom_signal(close, list(params.lookbacks), vol)
    ma = meta_arrays(meta, symbols)
    carry = None
    if include_carry and params.carry_weight > 0:
        carry = carry_signal(ma["swap_long"], ma["swap_short"], close[-1], ma["contract_size"])
    raw = target_weights(sig, vol, params, carry=carry)
    w = finalize_weights(raw, ret, params)          # overlay + caps + no-trade band (final book)
    return {"dates": dates, "symbols": symbols, "close": close, "ret": ret,
            "vol": vol, "signal": sig, "weights": w, "cost_bps": ma["spread_bps"],
            "asset_class": ma["asset_class"]}


def run_trend_backtest(panel: pl.DataFrame, meta: pl.DataFrame, params: TrendParams
                       ) -> tuple[BacktestResult, dict]:
    """Run the full trend backtest (carry excluded from pnl by design). Returns (result, book)."""
    book = compute_book(panel, meta, params, include_carry=False)
    res = backtest(book["weights"], book["ret"], params, cost_bps=book["cost_bps"])
    return res, book
