"""Broker adapter interface + an in-process paper broker (§8, §10).

The strategy talks to the exchange only through a broker (SEBI: no direct exchange
connectivity, §8.1). :class:`BrokerAdapter` is the interface a real Kite/Dhan/Fyers client
implements; :class:`PaperBroker` is a deterministic in-process simulator that fills orders
against a supplied top-of-book, tracks positions, and can inject rejects/slippage -- enough
to drive and test the full order/position lifecycle without a live account.

A broker sandbox validates plumbing (order lifecycle, reconnects, rate limits, position
accounting) but tells you **nothing** about real fills or adverse selection (§10). The only
honest fill test is tiny real capital.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Order:
    scrip_code: int
    side: int                 # +1 buy, -1 sell
    qty: float                # shares
    order_type: str           # "LIMIT" | "MARKET"
    price: float | None       # limit price; None for MARKET
    algo_id: str              # exchange Algo-ID -- MUST be present on every order (§8.1)
    ts_ns: int
    order_id: str = ""
    status: str = "NEW"       # NEW | ACKED | FILLED | RESTING | REJECTED | CANCELLED
    filled_qty: float = 0.0


@dataclass
class Fill:
    order_id: str
    scrip_code: int
    side: int
    qty: float
    price: float
    ts_ns: int
    is_taker: bool


@dataclass
class Position:
    scrip_code: int
    qty: float = 0.0          # signed shares
    avg_price: float = 0.0

    def notional(self, mid: float) -> float:
        return self.qty * mid


@runtime_checkable
class BrokerAdapter(Protocol):
    """Interface a live broker client must implement (same shape as :class:`PaperBroker`)."""

    def place_order(self, order: Order) -> Order: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def positions(self) -> dict[int, Position]: ...
    def poll_fills(self) -> list[Fill]: ...
    def reject_rate(self) -> float: ...


class PaperBroker:
    """Deterministic in-process paper broker for lifecycle testing.

    Set the market with :meth:`update_market` before placing orders. MARKET orders fill
    immediately at the touch plus ``taker_slippage_bps``; marketable LIMIT orders fill as
    takers; non-marketable LIMITs rest and fill when the market later crosses them.
    """

    def __init__(self, *, reject_prob: float = 0.0, taker_slippage_bps: float = 1.0,
                 maker_fill_prob: float = 1.0, seed: int = 0) -> None:
        self.reject_prob = reject_prob
        self.taker_slippage = taker_slippage_bps * 1e-4
        self.maker_fill_prob = maker_fill_prob   # queue-position uncertainty bound (§6.3)
        self.rng = np.random.default_rng(seed)
        self.orders: dict[str, Order] = {}
        self._positions: dict[int, Position] = {}
        self.fills: list[Fill] = []
        self._pending_fills: list[Fill] = []
        self.market: dict[int, tuple[float, float, float]] = {}   # scrip -> (bid, ask, mid)
        self._resting: list[Order] = []
        self._next_id = 1
        self.order_count = 0
        self.reject_count = 0

    # -- market state -----------------------------------------------------
    def update_market(self, scrip_code: int, bid: float, ask: float,
                      mid: float | None = None) -> None:
        self.market[scrip_code] = (bid, ask, mid if mid is not None else 0.5 * (bid + ask))
        self._sweep_resting(scrip_code)

    # -- BrokerAdapter ----------------------------------------------------
    def place_order(self, order: Order) -> Order:
        self.order_count += 1
        order.order_id = f"O{self._next_id}"
        self._next_id += 1
        self.orders[order.order_id] = order
        if not order.algo_id:
            order.status = "REJECTED"          # SEBI: untagged algo orders are rejected (§8.1)
            self.reject_count += 1
            return order
        if self.rng.random() < self.reject_prob:
            order.status = "REJECTED"
            self.reject_count += 1
            return order
        order.status = "ACKED"
        self._match(order)
        return order

    def cancel_all_resting(self) -> int:
        """Cancel every resting order (cancel-replace at each decision, §6.4). Returns count."""
        n = 0
        for o in self._resting:
            if o.status == "RESTING":
                o.status = "CANCELLED"
                n += 1
        self._resting = []
        return n

    def cancel_order(self, order_id: str) -> bool:
        o = self.orders.get(order_id)
        if o and o.status in ("ACKED", "RESTING"):
            o.status = "CANCELLED"
            if o in self._resting:
                self._resting.remove(o)
            return True
        return False

    def positions(self) -> dict[int, Position]:
        return self._positions

    def poll_fills(self) -> list[Fill]:
        out = self._pending_fills
        self._pending_fills = []
        return out

    def reject_rate(self) -> float:
        return self.reject_count / max(self.order_count, 1)

    # -- matching ---------------------------------------------------------
    def _match(self, order: Order) -> None:
        if order.scrip_code not in self.market:
            order.status = "RESTING"
            self._resting.append(order)
            return
        bid, ask, mid = self.market[order.scrip_code]
        if order.order_type == "MARKET":
            px = ask * (1 + self.taker_slippage) if order.side > 0 else bid * (1 - self.taker_slippage)
            self._fill(order, order.qty, px, taker=True)
            return
        marketable = (order.side > 0 and order.price >= ask) or \
                     (order.side < 0 and order.price <= bid)
        if marketable:
            self._fill(order, order.qty, order.price, taker=True)
        else:
            order.status = "RESTING"
            self._resting.append(order)

    def fill_on_range(self, scrip_code: int, low: float, high: float) -> None:
        """Maker fill when a trade reaches a resting order's price (§6.3).

        A resting buy fills if the bar's traded ``low`` reaches its price; a resting sell if
        the ``high`` reaches its price. This is the honest bar-data maker-fill proxy: you are
        filled when the market comes to you -- which is often when you are wrong.
        """
        for o in list(self._resting):
            if o.scrip_code != scrip_code or o.status != "RESTING":
                continue
            touched = (o.side > 0 and low <= o.price) or (o.side < 0 and high >= o.price)
            # touched != filled: queue position is unobservable on aggregated data (§6.3),
            # so a touch fills only with probability `maker_fill_prob` (the proportional bound)
            if touched and self.rng.random() < self.maker_fill_prob:
                self._fill(o, o.qty - o.filled_qty, o.price, taker=False)
                self._resting.remove(o)

    def _sweep_resting(self, scrip_code: int) -> None:
        bid, ask, mid = self.market[scrip_code]
        still: list[Order] = []
        for o in self._resting:
            if o.scrip_code != scrip_code or o.status != "RESTING":
                continue
            crossed = (o.side > 0 and ask <= o.price) or (o.side < 0 and bid >= o.price)
            if crossed:
                self._fill(o, o.qty - o.filled_qty, o.price, taker=False)  # passive fill
            else:
                still.append(o)
        self._resting = [o for o in self._resting
                         if o.scrip_code != scrip_code or o in still]

    def _fill(self, order: Order, qty: float, price: float, taker: bool) -> None:
        pos = self._positions.setdefault(order.scrip_code, Position(order.scrip_code))
        signed = order.side * qty
        new_qty = pos.qty + signed
        if pos.qty == 0 or (pos.qty > 0) == (signed > 0):        # opening/adding
            denom = abs(pos.qty) + qty
            pos.avg_price = (abs(pos.qty) * pos.avg_price + qty * price) / denom if denom else price
        pos.qty = new_qty
        order.filled_qty += qty
        order.status = "FILLED" if order.filled_qty >= order.qty - 1e-9 else "PARTIAL"
        fill = Fill(order.order_id, order.scrip_code, order.side, qty, price, order.ts_ns, taker)
        self.fills.append(fill)
        self._pending_fills.append(fill)
