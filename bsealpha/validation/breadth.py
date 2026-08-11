"""Effective breadth -- the report's most important diagnostic (§0.5, §3.4).

Cross-sectional breadth in intraday equities is largely illusory: every Indian equity
loads on the market factor, so 30 positions are mostly one bet on the index plus a small
residual. The participation ratio of the eigenvalue spectrum of the position-return
covariance is the number of *independent* bets you actually make. Use it -- not N -- when
you annualize a Sharpe, and gate deployment at >= 3 (§11.1).

Calibration (§3.4): market loading 1.0x idio -> breadth ~1.2 (an index bet wearing
tickers); 0.2x -> ~11; pure idiosyncratic for N names -> ~N. Breadth ~1-2 means
residualization failed.
"""

from __future__ import annotations

import numpy as np


def effective_breadth(position_returns: np.ndarray) -> float:
    """Participation ratio of the eigenvalue spectrum of the position-return covariance.

    Parameters
    ----------
    position_returns
        ``(T, N)`` matrix of per-period P&L contributions by name.

    Returns
    -------
    float
        ``(sum lambda)^2 / sum(lambda^2)`` -- the number of independent bets.
    """
    X = np.asarray(position_returns, float)
    if X.ndim != 2 or X.shape[1] < 1:
        return 0.0
    X = X - X.mean(0, keepdims=True)
    if X.shape[0] < 2:
        return float(X.shape[1])
    C = np.cov(X, rowvar=False)
    w = np.linalg.eigvalsh(C)
    w = w[w > 1e-14]
    if w.size == 0:
        return 0.0
    return float(w.sum() ** 2 / (w ** 2).sum())


def breadth_from_panel(panel, ret_col: str = "pnl", name_col: str = "scrip_code",
                       time_cols: tuple[str, ...] = ("date", "minute")) -> float:
    """Compute effective breadth from a long panel of per-name P&L.

    Pivots ``panel`` to a ``(time, name)`` matrix of ``ret_col`` and calls
    :func:`effective_breadth`. Missing cells are treated as zero P&L (flat).
    """
    import polars as pl

    wide = (panel.select([*time_cols, name_col, ret_col])
            .pivot(values=ret_col, index=list(time_cols), on=name_col)
            .drop(list(time_cols))
            .fill_null(0.0))
    return effective_breadth(wide.to_numpy())
