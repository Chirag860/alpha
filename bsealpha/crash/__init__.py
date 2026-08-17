"""Crash: short mean-reversion / crash-risk-premium capture on high-vol US equities.

Larry Connors' "Crash" strategy (*Buy the Fear, Sell the Greed*, 2018): each day, short liquid US
equities whose ConnorsRSI is extreme-overbought (>= 90) *and* whose 100-day realized vol exceeds
100%, via a limit-sell placed above the prior close; cover when CRSI < 30. No stop beyond the
signal exit.

This package is the honest-validation build: exact indicators (:mod:`.indicators`) with the
Wilder-seed RSI the review flagged, and the per-symbol setup screen (:mod:`.signals`). The
cross-sectional portfolio/backtest layer (slot accounting, realistic limit fills, borrow-cost
model, delisting-inclusive universe) is deliberately separate — it depends on the chosen data
source and is where the strategy actually lives or dies (see the research brief §6/§8).
"""

from __future__ import annotations

from .indicators import (
    connors_rsi,
    hv_annualized,
    percent_rank,
    roc1,
    streak,
    wilder_rsi,
)
from .signals import CrashParams, CrashSignals, crash_signals

__all__ = [
    "wilder_rsi", "streak", "roc1", "percent_rank", "connors_rsi", "hv_annualized",
    "CrashParams", "CrashSignals", "crash_signals",
]
