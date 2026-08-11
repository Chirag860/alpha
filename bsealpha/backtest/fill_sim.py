"""Fill modeling on 5-level snapshot data (§6.3).

You have aggregated 5-level snapshots, not order-by-order events, so the honest queue
model is coarse and we **bound** rather than pretend:

* **Taker** fills walk the visible levels (:func:`walk_the_book`); if the clip exhausts
  them, the remainder fills at the last visible level plus an impact penalty. In thin BSE
  names this is common and is itself a capacity signal.
* **Maker** fills use :class:`PassiveFillSimulator`: ``queue_ahead`` starts at the displayed
  depth, decrements by traded volume, and depth reductions unexplained by trades are
  attributed under three cancel models -- ``optimistic`` (all ahead), ``proportional``
  (default), ``pessimistic`` (all behind). **If a strategy is only profitable under the
  optimistic bound, it does not exist** (§6.3).

:func:`markouts` is the key maker diagnostic: side x (mid[t+h] - fill)/fill. Healthy
passive execution is mildly negative at 1 min (adverse selection paid) turning positive by
the holding horizon; monotonically negative means you are someone else's flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def walk_the_book(level_px: np.ndarray, level_qty: np.ndarray, qty: float,
                  side: int) -> tuple[float, float, bool]:
    """Fill ``qty`` shares as a taker against visible levels.

    ``side`` = +1 buy (consume asks), -1 sell (consume bids); ``level_px``/``level_qty``
    are that side's levels, best first. Returns ``(avg_price, filled_qty, exhausted)``.
    """
    remaining = float(qty)
    cost = 0.0
    filled = 0.0
    for px, q in zip(level_px, level_qty):
        take = min(remaining, float(q))
        cost += take * float(px)
        filled += take
        remaining -= take
        if remaining <= 1e-12:
            break
    exhausted = remaining > 1e-9
    if filled <= 0:
        return float(level_px[0]) if len(level_px) else 0.0, 0.0, True
    return cost / filled, filled, exhausted


@dataclass
class PassiveOrder:
    side: int
    price: float
    qty: float
    queue_ahead: float = np.nan
    filled: float = 0.0
    alive: bool = True


class PassiveFillSimulator:
    """Queue-aware maker fill simulator for aggregated snapshots (§6.3).

    ``cancel_model`` in ``{"optimistic", "proportional", "pessimistic"}``. Feed snapshots
    and trades in local-receipt order via :meth:`on_book` / :meth:`on_trade`.
    """

    def __init__(self, cancel_model: str = "proportional") -> None:
        self.cancel_model = cancel_model
        self.live: list[PassiveOrder] = []
        self.fills: list[dict] = []
        self.depth_at: dict[float, float] = {}

    def place(self, order: PassiveOrder, displayed_depth: float) -> None:
        order.queue_ahead = displayed_depth      # join the back of the visible queue
        self.live.append(order)

    def on_trade(self, ts: int, price: float, qty: float, aggressor_side: int,
                 mid: float) -> None:
        """A trade at ``price``. A resting BUY is filled by SELLER-initiated trades."""
        for o in self.live:
            if not o.alive or abs(o.price - price) > 1e-9:
                continue
            if aggressor_side == o.side:          # same-side aggressor doesn't fill us
                continue
            consumed = min(qty, o.queue_ahead)
            o.queue_ahead -= consumed
            residual = qty - consumed
            if residual > 0 and o.filled < o.qty:
                self._fill(ts, o, min(residual, o.qty - o.filled), mid)

    def on_book(self, ts: int, depth_at: dict[float, float], best_bid: float,
                best_ask: float) -> None:
        """Update queue positions from a new snapshot (trades handled separately)."""
        mid = 0.5 * (best_bid + best_ask)
        for o in self.live:
            if not o.alive:
                continue
            through = (best_ask <= o.price) if o.side > 0 else (best_bid >= o.price)
            if through and o.filled < o.qty:      # price swept our level -> full (bad) fill
                self._fill(ts, o, o.qty - o.filled, mid, taker=True)
                continue
            if o.price not in depth_at:            # our level fell out of the window
                continue
            new_d = depth_at[o.price]
            old_d = self.depth_at.get(o.price, new_d)
            delta = new_d - old_d
            if delta < 0:                          # cancels (removals not from our trades)
                cancel_vol = -delta
                if self.cancel_model == "optimistic":
                    o.queue_ahead -= cancel_vol
                elif self.cancel_model == "proportional":
                    frac = o.queue_ahead / max(old_d, 1e-12)
                    o.queue_ahead -= cancel_vol * min(frac, 1.0)
                # pessimistic: no change
                o.queue_ahead = max(o.queue_ahead, 0.0)
        self.depth_at = dict(depth_at)

    def _fill(self, ts: int, o: PassiveOrder, q: float, mid: float, taker: bool = False) -> None:
        o.filled += q
        self.fills.append(dict(ts=ts, side=o.side, price=o.price, qty=q, mid=mid,
                               taker=taker))
        if o.filled >= o.qty - 1e-12:
            o.alive = False

    @property
    def fill_ratio(self) -> float:
        placed = sum(o.qty for o in self.live)
        got = sum(o.filled for o in self.live)
        return float(got / placed) if placed > 0 else 0.0


def markouts(fills: list[dict], mid_ts: np.ndarray, mid_px: np.ndarray,
             horizons_min: tuple[float, ...] = (1, 5, 15, 30)) -> dict[int, float]:
    """Markout curve in bps: ``side * (mid[t+h] - fill_px) / fill_px`` averaged over fills.

    ``mid_ts`` are minute stamps aligned with ``mid_px``. Returns ``{h_min: bps}``.
    """
    if not fills:
        return {int(h): 0.0 for h in horizons_min}
    ts = np.array([f["ts"] for f in fills], float)
    side = np.array([f["side"] for f in fills], float)
    px = np.array([f["price"] for f in fills], float)
    out = {}
    for h in horizons_min:
        j = np.searchsorted(mid_ts, ts + h)
        j = np.clip(j, 0, len(mid_px) - 1)
        out[int(h)] = float(np.mean(side * (mid_px[j] - px) / px) * 1e4)
    return out
