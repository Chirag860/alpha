"""Daily backtest engine + performance metrics for the trend/carry book.

Accounting (leak-free): the position decided at the close of day ``t`` (``weights[t]``) earns
day ``t+1``'s return, so portfolio pnl on day ``t`` uses ``weights[t-1]``. Costs are charged on
the change in weights at each rebalance; a static per-instrument carry (swap) return is added to
held positions. ``ret`` must be **simple** daily returns (``close[t]/close[t-1] - 1``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import TrendParams


def _ewma_std(x: np.ndarray, halflife: float) -> np.ndarray:
    alpha = 1.0 - 0.5 ** (1.0 / float(halflife))
    v = 0.0
    out = np.zeros_like(x)
    for t in range(len(x)):
        v = alpha * x[t] ** 2 + (1.0 - alpha) * v
        out[t] = np.sqrt(v)
    return out


def _ewma_smooth(x: np.ndarray, halflife: float) -> np.ndarray:
    """Causal EWMA of a series (used to de-churn the vol-target scale)."""
    alpha = 1.0 - 0.5 ** (1.0 / float(halflife))
    out = np.zeros_like(x)
    s = x[0] if len(x) else 0.0
    for t in range(len(x)):
        s = alpha * x[t] + (1.0 - alpha) * s
        out[t] = s
    return out


def finalize_weights(raw_w: np.ndarray, ret: np.ndarray, params: TrendParams) -> np.ndarray:
    """Turn raw signal-sized weights into the FINAL held book.

    Applies the realized-vol overlay (smoothed, so it doesn't churn daily), re-caps, and then
    the no-trade band **on the final weights** — so the band actually suppresses the turnover
    the book experiences (applying it before the overlay lets the overlay re-introduce churn).
    Used by both the backtest and the live path so they hold identical books.
    """
    raw_w = np.asarray(raw_w, dtype=float)
    ret = np.asarray(ret, dtype=float)
    T, N = raw_w.shape
    daily_target = float(params.target_ann_vol) / np.sqrt(252.0)
    if params.vol_target_overlay:
        base = np.zeros(T)
        base[1:] = np.sum(raw_w[:-1] * ret[1:], axis=1)
        trail = _ewma_std(base, params.vol_halflife)
        scale = np.clip(daily_target / np.maximum(trail, 1e-9), 0.2, 3.0)
        scale[: min(20, T)] = 1.0
        scale = _ewma_smooth(scale, 10.0)              # de-churn the rescaling
        w = raw_w * scale[:, None]
    else:
        w = raw_w.copy()
    mw = float(params.max_weight_per_instrument)
    w = np.clip(w, -mw, mw)
    gross = np.sum(np.abs(w), axis=1, keepdims=True)
    w = w * np.minimum(1.0, float(params.max_gross_leverage) / np.maximum(gross, 1e-12))
    from .portfolio import apply_no_trade_band
    w = apply_no_trade_band(w, float(params.no_trade_band))
    return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class BacktestResult:
    net: np.ndarray                       # daily net simple returns (fraction of NAV)
    gross_pnl: np.ndarray
    carry_pnl: np.ndarray
    cost: np.ndarray
    weights: np.ndarray                   # final (scaled, capped) weights [T, N]
    ret: np.ndarray
    metrics: dict = field(default_factory=dict)

    @property
    def equity(self) -> np.ndarray:
        return np.cumprod(1.0 + self.net)


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak)) if len(equity) else 0.0


def compute_metrics(net: np.ndarray, weights: np.ndarray) -> dict:
    net = np.asarray(net, dtype=float)
    ann_ret = float(net.mean() * 252.0)
    ann_vol = float(net.std(ddof=1) * np.sqrt(252.0)) if len(net) > 1 else 0.0
    downside = net[net < 0]
    dstd = float(downside.std(ddof=1) * np.sqrt(252.0)) if len(downside) > 1 else 0.0
    eq = np.cumprod(1.0 + net)
    mdd = _max_drawdown(eq)
    dw = np.zeros_like(weights)
    dw[1:] = np.abs(weights[1:] - weights[:-1])
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0.0,
        "sortino": ann_ret / dstd if dstd > 0 else 0.0,
        "max_drawdown": mdd,
        "calmar": ann_ret / mdd if mdd > 0 else 0.0,
        "hit_rate_daily": float((net > 0).mean()),
        "skew": float(((net - net.mean()) ** 3).mean() / (net.std() ** 3 + 1e-12)),
        "avg_gross_leverage": float(np.sum(np.abs(weights), axis=1).mean()),
        "avg_daily_turnover": float(np.sum(dw, axis=1).mean()),
        "total_return": float(eq[-1] - 1.0) if len(eq) else 0.0,
    }


def backtest(weights: np.ndarray, ret: np.ndarray, params: TrendParams, *,
             cost_bps: np.ndarray | None = None,
             carry_daily: np.ndarray | None = None) -> BacktestResult:
    """Pure daily pnl/cost accounting on FINAL held ``weights`` (see :func:`finalize_weights`).

    Positions are lagged one day (``weights[t-1]`` earns ``ret[t]``); ``ret`` are simple returns.
    """
    w = np.asarray(weights, dtype=float)
    ret = np.asarray(ret, dtype=float)
    T, N = w.shape

    # -- costs on rebalancing --------------------------------------------------------------
    cb = np.full(N, float(params.cost_bps_per_side)) if cost_bps is None else np.asarray(cost_bps, float)
    dw = np.zeros((T, N))
    dw[0] = np.abs(w[0])
    dw[1:] = np.abs(w[1:] - w[:-1])
    cost = np.sum(dw * cb[None, :] * 1e-4, axis=1)

    # -- pnl (lagged positions) ------------------------------------------------------------
    gross_pnl = np.zeros(T)
    gross_pnl[1:] = np.sum(w[:-1] * ret[1:], axis=1)
    carry_pnl = np.zeros(T)
    if carry_daily is not None:
        cd = np.asarray(carry_daily, dtype=float)
        carry_pnl[1:] = np.sum(w[:-1] * cd[None, :], axis=1)

    net = gross_pnl + carry_pnl - cost
    res = BacktestResult(net=net, gross_pnl=gross_pnl, carry_pnl=carry_pnl,
                         cost=cost, weights=w, ret=ret)
    res.metrics = compute_metrics(net, w)
    return res
