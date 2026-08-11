"""MT5 broker adapter tests -- run on any platform against a FAKE MetaTrader5 terminal.

The real ``MetaTrader5`` package is Windows-only; ``MT5BrokerAdapter`` takes the module as a
dependency, so here we inject a small fake that mimics ``order_send`` / ``positions_get`` /
``history_deals_get`` / ``symbol_info`` and the enums. This exercises the platform-independent
logic: lot rounding to ``volume_step``, symbol<->scrip mapping, signed position mapping,
async fills from deal history (with de-dup), reject accounting, and cancels.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bsealpha.config import load_config
from bsealpha.execution import ExecutionManager, MT5BrokerAdapter, Order


class FakeMT5:
    """Minimal in-memory MT5 terminal double (netting account)."""

    # -- enums the adapter reads --
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_PENDING = 5
    TRADE_ACTION_REMOVE = 2
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TYPE_BUY_LIMIT = 2
    ORDER_TYPE_SELL_LIMIT = 3
    ORDER_FILLING_IOC = 1
    ORDER_TIME_DAY = 0
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_PLACED = 10008
    TRADE_RETCODE_ERROR = 10004
    DEAL_TYPE_BUY = 0
    DEAL_TYPE_SELL = 1
    DEAL_ENTRY_IN = 0
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1

    def __init__(self, symbols: dict[str, dict]):
        # symbols: name -> {"bid","ask","digits","contract","step","min","max"}
        self._spec = symbols
        self._positions: dict[str, dict] = {}     # symbol -> {"qty_lots","price"}
        self._deals: list[SimpleNamespace] = []
        self._pending: list[SimpleNamespace] = []
        self._ticket = 1
        self._t = 1_700_000_000

    # -- symbol info --
    def symbol_info(self, symbol):
        s = self._spec.get(symbol)
        if s is None:
            return None
        return SimpleNamespace(trade_contract_size=s["contract"], volume_step=s["step"],
                               volume_min=s["min"], volume_max=s["max"], digits=s["digits"])

    def symbol_info_tick(self, symbol):
        s = self._spec[symbol]
        return SimpleNamespace(bid=s["bid"], ask=s["ask"])

    def symbol_select(self, symbol, enable=True):
        return symbol in self._spec

    # -- order routing --
    def order_send(self, req):
        self._t += 1
        action = req["action"]
        if action == self.TRADE_ACTION_REMOVE:
            before = len(self._pending)
            self._pending = [o for o in self._pending if o.ticket != req["order"]]
            code = self.TRADE_RETCODE_DONE if len(self._pending) < before else self.TRADE_RETCODE_ERROR
            return SimpleNamespace(retcode=code, order=req["order"])

        ticket = self._ticket
        self._ticket += 1
        symbol = req["symbol"]
        vol = req["volume"]
        if action == self.TRADE_ACTION_PENDING:
            self._pending.append(SimpleNamespace(
                ticket=ticket, symbol=symbol, magic=req["magic"],
                type=req["type"], volume=vol, price=req["price"]))
            return SimpleNamespace(retcode=self.TRADE_RETCODE_PLACED, order=ticket,
                                   volume=vol, price=req["price"])

        # market DEAL: fill immediately, update netting position, record a deal
        is_buy = req["type"] == self.ORDER_TYPE_BUY
        price = req["price"]
        pos = self._positions.setdefault(symbol, {"qty_lots": 0.0, "price": price})
        pos["qty_lots"] += vol if is_buy else -vol
        pos["price"] = price
        if abs(pos["qty_lots"]) < 1e-12:
            self._positions.pop(symbol, None)
        self._deals.append(SimpleNamespace(
            ticket=ticket, order=ticket, symbol=symbol,
            type=self.DEAL_TYPE_BUY if is_buy else self.DEAL_TYPE_SELL,
            entry=self.DEAL_ENTRY_IN, volume=vol, price=price, time=self._t))
        return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, order=ticket,
                               volume=vol, price=price)

    def positions_get(self):
        out = []
        for sym, p in self._positions.items():
            q = p["qty_lots"]
            out.append(SimpleNamespace(
                symbol=sym, volume=abs(q),
                type=self.POSITION_TYPE_BUY if q > 0 else self.POSITION_TYPE_SELL,
                price_open=p["price"]))
        return out

    def history_deals_get(self, *args):
        return list(self._deals)          # adapter de-dups by ticket

    def orders_get(self):
        return list(self._pending)


SPEC = {
    "AAPL": {"bid": 99.95, "ask": 100.05, "digits": 2,
             "contract": 1.0, "step": 0.01, "min": 0.01, "max": 100000.0},
    "MSFT": {"bid": 199.9, "ask": 200.1, "digits": 2,
             "contract": 1.0, "step": 0.10, "min": 0.10, "max": 100000.0},
}
SYMBOL_MAP = {101: "AAPL", 102: "MSFT"}


def _adapter():
    return MT5BrokerAdapter(FakeMT5(SPEC), SYMBOL_MAP, magic=40100)


def _order(scrip, side, qty, otype="MARKET", price=None):
    return Order(scrip_code=scrip, side=side, qty=qty, order_type=otype,
                 price=price, algo_id="MT5ALPHA-0001", ts_ns=0)


# --------------------------------------------------------------- routing + fills
def test_market_order_fills_and_maps_position():
    a = _adapter()
    a.place_order(_order(101, +1, 50.0))          # buy 50 AAPL shares
    pos = a.positions()
    assert 101 in pos and pos[101].qty == pytest.approx(50.0)
    fills = a.poll_fills()
    assert len(fills) == 1
    f = fills[0]
    assert f.scrip_code == 101 and f.side == 1 and f.qty == pytest.approx(50.0)
    assert f.price == pytest.approx(100.05)       # bought at the ask
    assert a.poll_fills() == []                    # de-dup: no repeat on second poll


def test_sell_maps_to_negative_position():
    a = _adapter()
    a.place_order(_order(102, -1, 30.0))
    assert a.positions()[102].qty == pytest.approx(-30.0)


def test_qty_rounds_to_lot_step():
    fake = FakeMT5(SPEC)
    a = MT5BrokerAdapter(fake, SYMBOL_MAP)
    a.place_order(_order(102, +1, 30.04))         # MSFT step=0.10 -> 30.0 lots
    assert fake._deals[-1].volume == pytest.approx(30.0)


def test_below_min_lot_is_dropped_not_sent():
    fake = FakeMT5(SPEC)
    a = MT5BrokerAdapter(fake, SYMBOL_MAP)
    o = a.place_order(_order(102, +1, 0.05))       # < MSFT min lot 0.10
    assert o.status == "TOO_SMALL"
    assert fake._deals == [] and a.positions() == {}


def test_unknown_scrip_is_rejected():
    a = _adapter()
    o = a.place_order(_order(999, +1, 10.0))       # not in SYMBOL_MAP
    assert o.status == "REJECTED"
    assert a.reject_rate() == pytest.approx(1.0)


def test_limit_order_rests_as_pending():
    fake = FakeMT5(SPEC)
    a = MT5BrokerAdapter(fake, SYMBOL_MAP)
    o = a.place_order(_order(101, +1, 20.0, otype="LIMIT", price=99.50))
    assert o.status == "RESTING"
    assert len(fake._pending) == 1
    assert a.positions() == {}                     # pending -> no position yet


def test_force_market_sends_limits_as_market():
    """force_market routes even a LIMIT order as a market DEAL -> immediate fill/position."""
    fake = FakeMT5(SPEC)
    a = MT5BrokerAdapter(fake, SYMBOL_MAP, force_market=True)
    a.place_order(_order(101, +1, 20.0, otype="LIMIT", price=99.50))
    assert fake._pending == []                     # no resting order
    assert a.positions()[101].qty == pytest.approx(20.0)   # filled at market instead


def test_cancel_all_resting_removes_our_pendings():
    fake = FakeMT5(SPEC)
    a = MT5BrokerAdapter(fake, SYMBOL_MAP)
    a.place_order(_order(101, +1, 20.0, otype="LIMIT", price=99.50))
    a.place_order(_order(102, -1, 20.0, otype="LIMIT", price=201.0))
    assert a.cancel_all_resting() == 2
    assert fake.orders_get() == []


# --------------------------------------------------------------- manager contract
def test_execution_manager_drives_mt5_adapter_in_demo_mode():
    """ExecutionManager (DEMO) must exercise the full adapter contract without error."""
    cfg = load_config(overrides={
        "market": {"profile": "us_equity"},
        "deploy": {"mode": "demo"},
        "execution": {"min_clip": 50.0, "algo_id": "MT5ALPHA-0001"},
    })
    fake = FakeMT5(SPEC)
    a = MT5BrokerAdapter(fake, SYMBOL_MAP)
    mgr = ExecutionManager(a, cfg)                 # DEMO must NOT raise (no live gate)
    market = {101: (99.95, 100.05, 100.0), 102: (199.9, 200.1, 200.0)}
    res = mgr.step(minute_of_day=600, now_s=0.0,
                   targets={101: 5000.0, 102: -6000.0}, market=market)
    assert not res.halted
    assert len(res.orders) == 2                    # both clips clear the $50 min
    assert a.order_count == 2 and a.reject_rate() == 0.0
    # normal phase routes passive LIMITs -> pending orders on the terminal
    assert len(fake._pending) == 2
