"""Lockbox: the only overfitting defense that doesn't depend on your own honesty (§5.4).

Physically separate the most recent months of data before you begin. Do not look at it, do
not compute a single statistic on it. Touch it exactly once, at the very end; if it fails,
the project fails -- you do not get to iterate.

:func:`date_split` carves the panel by date. :class:`Lockbox` wraps the held-out slice and
**raises on any second access**, so an accidental re-peek is a hard error rather than a
silent invalidation of the whole project.
"""

from __future__ import annotations

import datetime as dt

import polars as pl


def date_split(panel: pl.DataFrame, holdout_days: int,
               date_col: str = "date") -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split ``panel`` into (research, lockbox) by the last ``holdout_days`` calendar days.

    Returns two frames; the lockbox is the most recent ``holdout_days`` distinct dates.
    """
    dates = panel.select(pl.col(date_col)).unique().sort(date_col)[date_col]
    if dates.len() <= holdout_days:
        raise ValueError(f"panel has {dates.len()} dates; cannot hold out {holdout_days}")
    cutoff = dates[dates.len() - holdout_days]
    research = panel.filter(pl.col(date_col) < cutoff)
    lockbox = panel.filter(pl.col(date_col) >= cutoff)
    return research, lockbox


class Lockbox:
    """A write-once / read-once wrapper around held-out data (§5.4).

    ``open()`` returns the data exactly once and marks it consumed; any further call raises.
    The point is to make an accidental second peek impossible to do silently.
    """

    def __init__(self, data: pl.DataFrame, name: str = "lockbox") -> None:
        self._data = data
        self.name = name
        self._opened = False
        self.opened_at: dt.datetime | None = None

    @property
    def is_sealed(self) -> bool:
        return not self._opened

    def n_rows(self) -> int:
        """Row count is metadata, not a statistic on the labels -- safe to read."""
        return self._data.height

    def open(self) -> pl.DataFrame:
        if self._opened:
            raise RuntimeError(
                f"{self.name} already opened at {self.opened_at}; the lockbox is single-use "
                "(§5.4). If it failed, the project fails -- you do not get to iterate.")
        self._opened = True
        self.opened_at = dt.datetime.now()
        return self._data
