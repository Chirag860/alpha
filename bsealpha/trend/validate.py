"""Honest out-of-sample validation for the (parameter-light) trend book.

Because the system fits no predictive weights, the overfitting risk lives in *parameter
selection* (lookbacks, vol target, ...). So the validation reports: subperiod stability
(per-year Sharpe), a block-bootstrap Sharpe CI, a true out-of-sample split, and a Deflated
Sharpe that discounts for the number of configurations tried. Judge the book on these, never
on a single full-sample number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..validation.metrics import deflated_sharpe
from .backtest import BacktestResult
from .config import TrendParams


@dataclass
class TrendValidation:
    sharpe: float = 0.0
    ann_return: float = 0.0
    ann_vol: float = 0.0
    max_drawdown: float = 0.0
    sharpe_is: float = 0.0
    sharpe_oos: float = 0.0
    sharpe_ci: tuple[float, float] = (0.0, 0.0)
    dsr: float = 0.0
    n_trials: int = 0
    per_year: dict = field(default_factory=dict)


def _ann_sharpe(net: np.ndarray) -> float:
    if len(net) < 20 or net.std() == 0:
        return 0.0
    return float(net.mean() / net.std(ddof=1) * np.sqrt(252.0))


def _per_year_sharpe(dates: np.ndarray, net: np.ndarray) -> dict:
    years = np.array([str(d)[:4] for d in dates.astype("datetime64[D]").astype(str)])
    out = {}
    for y in sorted(np.unique(years)):
        r = net[years == y]
        if len(r) > 40:
            out[y] = round(_ann_sharpe(r), 2)
    return out


def _bootstrap_ci(net: np.ndarray, *, n: int = 2000, block: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    T = len(net)
    nb = int(np.ceil(T / block))
    sh = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, T, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        sh[i] = _ann_sharpe(net[idx])
    return float(np.percentile(sh, 5)), float(np.percentile(sh, 95))


def validate_book(dates: np.ndarray, res: BacktestResult, params: TrendParams,
                  *, n_trials: int = 1) -> TrendValidation:
    net = res.net
    v = TrendValidation()
    v.sharpe = res.metrics["sharpe"]
    v.ann_return = res.metrics["ann_return"]
    v.ann_vol = res.metrics["ann_vol"]
    v.max_drawdown = res.metrics["max_drawdown"]
    cut = int(len(net) * 0.7)
    v.sharpe_is = _ann_sharpe(net[:cut])
    v.sharpe_oos = _ann_sharpe(net[cut:])            # genuine holdout (no fitting either way)
    v.sharpe_ci = _bootstrap_ci(net)
    trials = list(np.random.default_rng(0).normal(0.0, float(params.dsr_trial_sr_std),
                                                  max(int(n_trials), 2)))
    v.dsr, _, _ = deflated_sharpe(net, np.asarray(trials))
    v.n_trials = len(trials)
    v.per_year = _per_year_sharpe(dates, net)
    return v


def format_report(res: BacktestResult, v: TrendValidation, book: dict) -> str:
    m = res.metrics
    L = "-" * 70
    lines = [
        L, "TREND + CARRY  —  OUT-OF-SAMPLE VALIDATION", L,
        f"Instruments / days                 : {book['weights'].shape[1]} / {book['weights'].shape[0]:,}",
        "",
        "PERFORMANCE (net of costs)",
        f"  Annualized return / vol          : {m['ann_return']:+.1%} / {m['ann_vol']:.1%}",
        f"  Sharpe / Sortino                 : {m['sharpe']:.2f} / {m['sortino']:.2f}",
        f"  Max drawdown / Calmar            : {m['max_drawdown']:.1%} / {m['calmar']:.2f}",
        f"  Skew (crisis-alpha if > 0)       : {m['skew']:+.2f}",
        f"  Avg gross leverage / turnover    : {m['avg_gross_leverage']:.2f}x / {m['avg_daily_turnover']:.2f}/day",
        "",
        "HONESTY CHECKS",
        f"  Sharpe in-sample -> out-of-sample: {v.sharpe_is:.2f} -> {v.sharpe_oos:.2f}   "
        "[OOS should hold up]",
        f"  Sharpe 90% bootstrap CI          : [{v.sharpe_ci[0]:.2f}, {v.sharpe_ci[1]:.2f}]",
        f"  Deflated Sharpe (N={v.n_trials})           : {v.dsr:.2f}   [want > 0.95]",
        f"  Per-year Sharpe                  : "
        + "  ".join(f"{y}:{s:+.1f}" for y, s in v.per_year.items()),
        L,
        "Read: a real trend edge shows a positive OOS Sharpe close to in-sample, a bootstrap CI",
        "clear of zero, positive skew, and stability across years — not one lucky number.",
        L,
    ]
    return "\n".join(lines)
