"""Purged, embargoed cross-validation for a panel (§5.2, §5.3).

Panel-specific rules the report is emphatic about:

* **Split by calendar day, always.** Every name shares the time index, so an
  observation-level split leaks across the whole cross-section instantly. Groups = days.
* **Purge on label spans.** Drop training rows whose label span reaches into the test
  window. With a 15-min horizon and forced flatten, labels never cross a session boundary
  (§5.2) -- a real simplification -- but *features* have long lookbacks, so we also
* **Embargo** ``embargo_days`` of days on each side of every test block.

:class:`CombinatorialPurgedCV` (CPCV) gives a *distribution* of out-of-sample paths rather
than a single walk-forward number (§5.3); report the 5th percentile.
"""

from __future__ import annotations

import datetime as dt
from itertools import combinations
from math import comb
from typing import Iterator

import numpy as np


def _day_index(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map each row to an integer day ordinal; return (row_day_ord, sorted_unique_days)."""
    uniq = np.array(sorted(set(dates)))
    order = {d: i for i, d in enumerate(uniq)}
    row_ord = np.array([order[d] for d in dates], dtype=np.int64)
    return row_ord, uniq


class PurgedDayGroupCV:
    """K-fold over contiguous day-blocks with purge + embargo (§5.2).

    ``split`` yields ``(train_idx, test_idx)`` row-index arrays. A training row is dropped
    if its day is within ``embargo_days`` of any test day (embargo), which -- because
    labels stay within a day -- also subsumes label-span purging here.
    """

    def __init__(self, n_splits: int = 6, embargo_days: int = 2) -> None:
        self.n_splits = n_splits
        self.embargo_days = embargo_days

    def split(self, dates: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        row_ord, uniq = _day_index(dates)
        n_days = len(uniq)
        folds = np.array_split(np.arange(n_days), min(self.n_splits, n_days))
        for f in folds:
            if len(f) == 0:
                continue
            test_days = set(f.tolist())
            lo, hi = f.min() - self.embargo_days, f.max() + self.embargo_days
            test_idx = np.flatnonzero(np.isin(row_ord, list(test_days)))
            embargoed = (row_ord >= lo) & (row_ord <= hi)
            train_idx = np.flatnonzero(~embargoed)
            if len(train_idx) and len(test_idx):
                yield train_idx, test_idx


class CombinatorialPurgedCV:
    """CPCV: N day-blocks, k test blocks per split, ``C(N,k)`` splits (§5.3).

    Reconstructs ``phi = C(N,k)*k/N`` distinct out-of-sample paths. Blocks are aligned to
    contiguous day ranges; align to calendar months in production so each block contains
    one F&O expiry (§5.3).
    """

    def __init__(self, n_groups: int = 12, k_test: int = 2, embargo_days: int = 2) -> None:
        self.n_groups = n_groups
        self.k_test = k_test
        self.embargo_days = embargo_days

    @property
    def n_paths(self) -> int:
        return comb(self.n_groups, self.k_test) * self.k_test // self.n_groups

    def split(self, dates: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray, tuple]]:
        row_ord, uniq = _day_index(dates)
        n_days = len(uniq)
        n_groups = min(self.n_groups, n_days)
        blocks = np.array_split(np.arange(n_days), n_groups)
        for combo in combinations(range(n_groups), min(self.k_test, n_groups)):
            test_days: list[int] = []
            keep = np.ones(len(row_ord), dtype=bool)
            for c in combo:
                g = blocks[c]
                test_days.extend(g.tolist())
                lo, hi = g.min() - self.embargo_days, g.max() + self.embargo_days
                keep &= ~((row_ord >= lo) & (row_ord <= hi))
            test_idx = np.flatnonzero(np.isin(row_ord, test_days))
            train_idx = np.flatnonzero(keep & ~np.isin(row_ord, test_days))
            if len(train_idx) and len(test_idx):
                yield train_idx, test_idx, combo
