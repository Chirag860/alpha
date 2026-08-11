"""Overfitting-defense statistics: Deflated Sharpe and PBO (§5.4).

Two errors the report calls "mechanical, catastrophic, and common":

* **T is the number of DAYS in the P&L series, not the number of trades** (§5.4). Using
  trades inflates significance by ~sqrt(trades/days). Our :func:`deflated_sharpe` takes a
  *daily* return series.
* **N (trial count) must include every configuration ever evaluated**, including abandoned
  ones. Log it automatically (:class:`~bsealpha.validation.trials.TrialLog`).

DSR corrects an observed Sharpe for trial count, non-normality, and sample length; PBO
(CSCV) estimates the probability the in-sample-best config underperforms out of sample.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized Sharpe of a per-period return series."""
    r = np.asarray(returns, float)
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def expected_max_sharpe(sr_trials: np.ndarray) -> float:
    """E[max SR] under the null that all ``N`` trials have true SR = 0 (per-period units)."""
    sr_trials = np.asarray(sr_trials, float)
    N = len(sr_trials)
    if N < 2:
        return 0.0
    v = np.var(sr_trials, ddof=1)
    return float(np.sqrt(v) * ((1 - EULER) * norm.ppf(1 - 1 / N)
                               + EULER * norm.ppf(1 - 1 / (N * np.e))))


def deflated_sharpe(daily_returns: np.ndarray, sr_trials: np.ndarray) -> tuple[float, float, float]:
    """Deflated Sharpe Ratio (Bailey & López de Prado, §5.4).

    Parameters
    ----------
    daily_returns
        **Daily** strategy P&L series -- one observation per trading session (NOT trades).
    sr_trials
        Per-day Sharpe of every configuration evaluated (the honest trial set).

    Returns
    -------
    (dsr, observed_sr, expected_max_sr)
        ``dsr`` = P(true SR > 0); gate at 0.95.
    """
    r = np.asarray(daily_returns, float)
    T = len(r)
    if T < 3 or r.std(ddof=1) == 0:
        return 0.0, 0.0, 0.0
    sr = r.mean() / r.std(ddof=1)
    g3 = float(pd.Series(r).skew())
    g4 = float(pd.Series(r).kurtosis()) + 3.0
    sr_star = expected_max_sharpe(sr_trials) if len(sr_trials) >= 2 else 0.0
    denom = np.sqrt(max(1.0 - g3 * sr + 0.25 * (g4 - 1.0) * sr ** 2, 1e-12))
    num = (sr - sr_star) * np.sqrt(T - 1)
    return float(norm.cdf(num / denom)), float(sr), float(sr_star)


def pbo_cscv(perf_matrix: np.ndarray, S: int = 16) -> tuple[float, np.ndarray]:
    """Probability of Backtest Overfitting via CSCV (§5.4).

    Parameters
    ----------
    perf_matrix
        ``(T, n_configs)`` matrix of per-period returns, one column per configuration.
    S
        Number of time blocks (even). All ``C(S, S/2)`` train/test partitions are formed;
        the in-sample-best config's out-of-sample rank is logit-transformed.

    Returns
    -------
    (pbo, logits)
        ``pbo`` = P(best-IS config ranks below median OOS); gate < 0.2.
    """
    M = np.asarray(perf_matrix, float)
    T, n = M.shape
    S = min(S, T)
    if S % 2 == 1:
        S -= 1
    if S < 2 or n < 2:
        return float("nan"), np.array([])
    blocks = np.array_split(np.arange(T), S)
    logits = []
    for tr in combinations(range(S), S // 2):
        te = [b for b in range(S) if b not in tr]
        tr_i = np.concatenate([blocks[b] for b in tr])
        te_i = np.concatenate([blocks[b] for b in te])
        sr_is = M[tr_i].mean(0) / (M[tr_i].std(0, ddof=1) + 1e-12)
        sr_oos = M[te_i].mean(0) / (M[te_i].std(0, ddof=1) + 1e-12)
        best = int(np.argmax(sr_is))
        rank = (np.argsort(np.argsort(sr_oos))[best] + 1) / (n + 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.asarray(logits)
    return float((logits <= 0).mean()), logits
