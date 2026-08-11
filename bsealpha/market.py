"""Market conventions as swappable **profiles**: tick bands + session-time helpers.

Originally hard-coded to BSE cash equities. To let the same engine trade a different
venue (e.g. US stock CFDs on MT5), the venue-specific numbers now live in a
:class:`MarketProfile`, and every layer -- data generation, features, labeling, backtest,
live -- reads the **active** profile through the accessors below. Divergence here is a
classic source of live/backtest skew (§8.3), so there is exactly one source of truth.

The active profile defaults to :data:`BSE_PROFILE` (nothing changes for the BSE pipeline
or its tests). Select another with :func:`set_active_profile` / :func:`set_active_profile_from_config`
**once at process start**, before features are built.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarketProfile:
    """Venue-specific session boundaries (minutes-of-day) and tick-size bands.

    ``tick_bands`` is an ascending sequence of ``(price_below, tick)`` pairs; a price is
    assigned the tick of the first band whose ``price_below`` it is under, else
    ``default_tick``. An empty ``tick_bands`` means a flat ``default_tick`` (decimalized
    venues like US equities/CFDs).
    """

    name: str
    session_open_min: int          # minutes-of-day of the continuous-session open
    session_close_min: int         # minutes-of-day of the close
    flatten_min: int               # minutes-of-day the forced-unwind starts
    tick_bands: tuple[tuple[float, float], ...]
    default_tick: float
    tz: str

    @property
    def session_len_min(self) -> int:
        return self.session_close_min - self.session_open_min

    @property
    def flatten_session_min(self) -> int:
        """Forced-flatten deadline expressed as a session-relative minute."""
        return self.flatten_min - self.session_open_min


# -- BSE cash equities (IST). The historical default. -------------------------
BSE_PROFILE = MarketProfile(
    name="bse",
    session_open_min=9 * 60 + 15,        # 09:15
    session_close_min=15 * 60 + 30,      # 15:30
    flatten_min=15 * 60 + 15,            # 15:15 forced-unwind start
    tick_bands=((250.0, 0.01), (1000.0, 0.05), (5000.0, 0.10),
                (10000.0, 0.50), (20000.0, 1.00)),
    default_tick=5.00,
    tz="Asia/Kolkata",
)

# -- US equities / stock CFDs (ET). Regular trading hours, decimalized. -------
US_EQUITY_PROFILE = MarketProfile(
    name="us_equity",
    session_open_min=9 * 60 + 30,        # 09:30 ET
    session_close_min=16 * 60,           # 16:00 ET
    flatten_min=15 * 60 + 55,            # 15:55 ET forced-unwind start
    tick_bands=(),                       # decimalized -> flat penny tick
    default_tick=0.01,
    tz="America/New_York",
)

_PROFILES: dict[str, MarketProfile] = {
    BSE_PROFILE.name: BSE_PROFILE,
    US_EQUITY_PROFILE.name: US_EQUITY_PROFILE,
}

_ACTIVE: MarketProfile = BSE_PROFILE


# -- active-profile control ---------------------------------------------------
def active_profile() -> MarketProfile:
    return _ACTIVE


def register_profile(profile: MarketProfile) -> None:
    """Add/replace a named profile (e.g. a broker-specific session)."""
    _PROFILES[profile.name] = profile


def set_active_profile(profile: MarketProfile | str) -> MarketProfile:
    """Select the active market profile by object or registered name. Returns it."""
    global _ACTIVE
    if isinstance(profile, MarketProfile):
        _ACTIVE = profile
    else:
        try:
            _ACTIVE = _PROFILES[profile]
        except KeyError:
            raise ValueError(f"unknown market profile {profile!r}; "
                             f"known: {sorted(_PROFILES)}") from None
    return _ACTIVE


def set_active_profile_from_config(cfg) -> MarketProfile:
    """Set the active profile from ``cfg.market.profile`` (defaults to ``'bse'``)."""
    name = "bse"
    market_cfg = getattr(cfg, "market", None)
    if market_cfg is not None:
        name = str(getattr(market_cfg, "profile", "bse"))
    return set_active_profile(name)


# -- session accessors (read the active profile at call time) -----------------
def session_open_min() -> int:
    return _ACTIVE.session_open_min


def session_close_min() -> int:
    return _ACTIVE.session_close_min


def session_len_min() -> int:
    return _ACTIVE.session_len_min


def flatten_min() -> int:
    return _ACTIVE.flatten_min


def flatten_session_min() -> int:
    return _ACTIVE.flatten_session_min


# -- tick size ----------------------------------------------------------------
def tick_size(price: float | np.ndarray) -> float | np.ndarray:
    """Price-banded tick size for the active profile.

    BSE uses SEBI price bands; decimalized venues (empty ``tick_bands``) return a flat
    ``default_tick``. Verify bands against the live venue circular -- they change.
    """
    prof = _ACTIVE
    p = np.asarray(price, dtype=float)
    if not prof.tick_bands:
        if np.isscalar(price):
            return float(prof.default_tick)
        return np.full(p.shape, prof.default_tick, dtype=float)
    conds = [p < thr for thr, _ in prof.tick_bands]
    vals = [t for _, t in prof.tick_bands]
    ticks = np.select(conds, vals, default=prof.default_tick)
    if np.isscalar(price):
        return float(ticks)
    return ticks


def round_to_tick(price: float | np.ndarray) -> float | np.ndarray:
    """Round a price to its band's tick grid."""
    ts = tick_size(price)
    return np.round(np.asarray(price, dtype=float) / ts) * ts


def mid_price(bid_px: float | np.ndarray, ask_px: float | np.ndarray) -> float | np.ndarray:
    """Simple mid = (best bid + best ask) / 2. Label-price base (§2.4)."""
    return 0.5 * (np.asarray(bid_px, dtype=float) + np.asarray(ask_px, dtype=float))


def micro_price(bid_px: float | np.ndarray, bid_qty: float | np.ndarray,
                ask_px: float | np.ndarray, ask_qty: float | np.ndarray):
    """Volume-weighted micro-price with the CROSSED weights (§2.4).

    The **ask** price is weighted by the **bid** quantity: a thick bid (imbalance -> 1)
    means upward pressure, so the estimate leans toward the ask. Getting this backwards
    is a silent, common sign error. Returns ``(micro, imbalance)`` with imbalance in ``[0, 1]``.
    """
    bq = np.asarray(bid_qty, dtype=float)
    aq = np.asarray(ask_qty, dtype=float)
    imb = bq / np.maximum(bq + aq, 1e-12)
    micro = np.asarray(ask_px, dtype=float) * imb + np.asarray(bid_px, dtype=float) * (1.0 - imb)
    return micro, imb


# -- session-time helpers (all relative to the active profile) ----------------
def session_minute(minute_of_day: float | np.ndarray) -> float | np.ndarray:
    """0 at the open, ``session_len-1`` at the last minute. Outside => negative / >=len."""
    return np.asarray(minute_of_day, dtype=float) - _ACTIVE.session_open_min


def minutes_to_close(minute_of_day: float | np.ndarray) -> float | np.ndarray:
    """Minutes remaining until the close (clamped at 0)."""
    return np.maximum(_ACTIVE.session_close_min - np.asarray(minute_of_day, dtype=float), 0.0)


def minutes_to_flatten(minute_of_day: float | np.ndarray) -> float | np.ndarray:
    """Minutes remaining until the forced-flatten deadline (clamped at 0)."""
    return np.maximum(_ACTIVE.flatten_min - np.asarray(minute_of_day, dtype=float), 0.0)


def is_tradable_minute(minute_of_day: float | np.ndarray) -> np.ndarray:
    """True during the continuous session before the forced-flatten deadline."""
    m = np.asarray(minute_of_day, dtype=float)
    return (m >= _ACTIVE.session_open_min) & (m < _ACTIVE.flatten_min)


def tod_bin(minute_of_day: float | np.ndarray, n_bins: int) -> np.ndarray:
    """Discretize session time into ``n_bins`` for the time-of-day vol profile (§2.5)."""
    sm = session_minute(minute_of_day)
    b = (sm / _ACTIVE.session_len_min * n_bins).astype(int)
    return np.clip(b, 0, n_bins - 1)
