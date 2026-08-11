"""MetaTrader 5 broker adapter -- routes the strategy's orders to an MT5 (demo) account.

Implements the interface :class:`~bsealpha.execution.manager.ExecutionManager` actually
calls: ``place_order``, ``positions``, ``poll_fills``, ``reject_rate``, ``update_market``
(no-op -- prices come from the live terminal, not the caller), ``cancel_all_resting`` and
``cancel_order``. Fills are delivered asynchronously: ``place_order`` only submits; the
manager learns of fills through ``poll_fills`` (which reads new MT5 *deals*), matching the
same async contract the ``PaperBroker`` uses.

Two MT5 realities the rest of the system doesn't model, handled here:

* **Volume is in lots, not shares.** ``Order.qty`` is fractional "shares" (rupees/mid). We
  convert to lots via the symbol's ``trade_contract_size`` and round to ``volume_step``,
  clamping to ``[volume_min, volume_max]``. Clips that round below ``volume_min`` are dropped.
* **scrip_code (int) <-> MT5 symbol (str)** needs an explicit map (there are no scrip codes
  on MT5); it is supplied at construction (the exporter persists it).

The ``MetaTrader5`` package is Windows-only, so it is **injected** (``mt5=`` argument). In
production, :func:`connect_mt5` lazily imports and initializes the real terminal; in tests a
fake module with the same surface is passed, so all of this logic runs on any platform.
"""

from __future__ import annotations

from .broker import Fill, Order, Position


class MT5BrokerAdapter:
    """BrokerAdapter over a (demo) MetaTrader 5 terminal.

    Parameters
    ----------
    mt5
        A connected ``MetaTrader5`` module (or a compatible fake). Must expose the
        constants and functions used below (``order_send``, ``positions_get``,
        ``history_deals_get``, ``symbol_info``, ``symbol_info_tick``, ``symbol_select``,
        ``orders_get`` and the ``TRADE_ACTION_*`` / ``ORDER_TYPE_*`` / ``ORDER_FILLING_*`` /
        ``TRADE_RETCODE_*`` enums).
    symbol_map
        ``{scrip_code: mt5_symbol}``. The reverse map is derived for deal/position lookups.
    magic
        MT5 "magic number" stamped on every order (our Algo-ID analogue).
    deviation
        Max slippage in points for market orders.
    """

    def __init__(self, mt5, symbol_map: dict[int, str], *, magic: int = 40100,
                 deviation: int = 20) -> None:
        self.mt5 = mt5
        self.symbol_map = dict(symbol_map)
        self.rev_map = {v: k for k, v in self.symbol_map.items()}
        self.magic = int(magic)
        self.deviation = int(deviation)
        self._info_cache: dict[str, object] = {}
        self._seen_deals: set[int] = set()
        self._deal_cursor: int = 0                # last deal time seen (epoch seconds)
        self.order_count = 0
        self.reject_count = 0
        self.too_small_count = 0

    # -- market state (no-op: the terminal is the price source) -----------
    def update_market(self, scrip_code: int, bid: float, ask: float,
                      mid: float | None = None) -> None:
        return None

    # -- symbol / lot helpers ---------------------------------------------
    def _symbol_info(self, symbol: str):
        info = self._info_cache.get(symbol)
        if info is None:
            info = self.mt5.symbol_info(symbol)
            if info is None:
                raise ValueError(f"MT5 has no symbol {symbol!r} (not in Market Watch?)")
            self.mt5.symbol_select(symbol, True)
            self._info_cache[symbol] = info
        return info

    def _shares_to_lots(self, symbol: str, shares: float) -> float:
        """Convert fractional shares to an MT5 lot volume rounded to the symbol's step.

        Returns 0.0 if the clip rounds below ``volume_min`` (too small to place).
        """
        info = self._symbol_info(symbol)
        contract = float(getattr(info, "trade_contract_size", 1.0) or 1.0)
        step = float(getattr(info, "volume_step", 0.01) or 0.01)
        vmin = float(getattr(info, "volume_min", step) or step)
        vmax = float(getattr(info, "volume_max", 1e12) or 1e12)
        lots = abs(shares) / contract
        lots = round(round(lots / step) * step, 8)     # snap to step grid
        if lots < vmin:
            return 0.0
        return min(lots, vmax)

    def _contract_size(self, symbol: str) -> float:
        return float(getattr(self._symbol_info(symbol), "trade_contract_size", 1.0) or 1.0)

    # -- order routing ----------------------------------------------------
    def place_order(self, order: Order) -> Order:
        self.order_count += 1
        symbol = self.symbol_map.get(order.scrip_code)
        if symbol is None:
            order.status = "REJECTED"
            self.reject_count += 1
            return order

        lots = self._shares_to_lots(symbol, order.qty)
        if lots <= 0.0:                     # rounds below the broker's min lot -> skip
            order.status = "TOO_SMALL"
            self.too_small_count += 1
            return order

        mt5 = self.mt5
        info = self._symbol_info(symbol)
        digits = int(getattr(info, "digits", 2) or 2)
        tick = mt5.symbol_info_tick(symbol)
        is_buy = order.side > 0

        if order.order_type == "MARKET":
            price = float(tick.ask if is_buy else tick.bid)
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
                "price": round(price, digits),
                "deviation": self.deviation,
            }
        else:                               # LIMIT (passive)
            price = float(order.price if order.price is not None
                          else (tick.bid if is_buy else tick.ask))
            req = {
                "action": mt5.TRADE_ACTION_PENDING,
                "type": mt5.ORDER_TYPE_BUY_LIMIT if is_buy else mt5.ORDER_TYPE_SELL_LIMIT,
                "price": round(price, digits),
            }
        req.update({
            "symbol": symbol,
            "volume": lots,
            "magic": self.magic,
            "type_time": getattr(mt5, "ORDER_TIME_DAY", 0),
            "type_filling": self._filling_mode(),
            "comment": (order.algo_id or "bsealpha")[:31],
        })

        result = mt5.order_send(req)
        retcode = getattr(result, "retcode", None)
        ok = retcode in (getattr(mt5, "TRADE_RETCODE_DONE", 10009),
                         getattr(mt5, "TRADE_RETCODE_PLACED", 10008))
        order.order_id = str(getattr(result, "order", "") or "")
        if not ok:
            order.status = "REJECTED"
            self.reject_count += 1
            return order
        # DONE => filled market order; PLACED => resting pending order
        order.status = "RESTING" if order.order_type != "MARKET" else "ACKED"
        return order

    def _filling_mode(self):
        mt5 = self.mt5
        return getattr(mt5, "ORDER_FILLING_IOC",
                       getattr(mt5, "ORDER_FILLING_FOK", 0))

    # -- fills (async, from the deal history) -----------------------------
    def poll_fills(self) -> list[Fill]:
        """Return MT5 deals executed since the last poll, as canonical ``Fill``s."""
        mt5 = self.mt5
        deals = mt5.history_deals_get(self._deal_cursor, self._now()) or ()
        out: list[Fill] = []
        for d in deals:
            ticket = int(getattr(d, "ticket", 0))
            if ticket in self._seen_deals:
                continue
            symbol = getattr(d, "symbol", "")
            sc = self.rev_map.get(symbol)
            if sc is None:
                continue
            # skip balance/credit and other non-trade deals
            dtype = getattr(d, "type", None)
            if dtype not in (getattr(mt5, "DEAL_TYPE_BUY", 0),
                             getattr(mt5, "DEAL_TYPE_SELL", 1)):
                continue
            self._seen_deals.add(ticket)
            side = 1 if dtype == getattr(mt5, "DEAL_TYPE_BUY", 0) else -1
            shares = float(getattr(d, "volume", 0.0)) * self._contract_size(symbol)
            out.append(Fill(
                order_id=str(getattr(d, "order", "") or ""),
                scrip_code=int(sc), side=side, qty=shares,
                price=float(getattr(d, "price", 0.0)),
                ts_ns=int(getattr(d, "time", 0)) * 1_000_000_000,
                # MT5 deals don't expose maker/taker; the manager ignores this field and
                # market orders dominate the demo path, so default to taker.
                is_taker=True,
            ))
        self._deal_cursor = self._now()
        return out

    def _now(self) -> int:
        import time
        return int(time.time())

    # -- positions --------------------------------------------------------
    def positions(self) -> dict[int, Position]:
        mt5 = self.mt5
        out: dict[int, Position] = {}
        for p in (mt5.positions_get() or ()):
            symbol = getattr(p, "symbol", "")
            sc = self.rev_map.get(symbol)
            if sc is None:
                continue
            contract = self._contract_size(symbol)
            signed = 1 if getattr(p, "type", 0) == getattr(mt5, "POSITION_TYPE_BUY", 0) else -1
            shares = signed * float(getattr(p, "volume", 0.0)) * contract
            pos = out.setdefault(int(sc), Position(int(sc)))
            # a netting account has one position per symbol; sum defensively otherwise
            pos.qty += shares
            pos.avg_price = float(getattr(p, "price_open", 0.0))
        return out

    def reject_rate(self) -> float:
        return self.reject_count / max(self.order_count, 1)

    # -- cancels ----------------------------------------------------------
    def cancel_all_resting(self) -> int:
        """Remove every pending order stamped with our magic. Returns the count removed."""
        mt5 = self.mt5
        n = 0
        for o in (mt5.orders_get() or ()):
            if getattr(o, "magic", None) != self.magic:
                continue
            if self._remove_pending(int(getattr(o, "ticket", 0))):
                n += 1
        return n

    def cancel_order(self, order_id: str) -> bool:
        try:
            ticket = int(order_id)
        except (TypeError, ValueError):
            return False
        return self._remove_pending(ticket)

    def _remove_pending(self, ticket: int) -> bool:
        mt5 = self.mt5
        result = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": ticket})
        return getattr(result, "retcode", None) == getattr(mt5, "TRADE_RETCODE_DONE", 10009)


def connect_mt5(cfg):  # pragma: no cover - requires the Windows MT5 terminal
    """Lazily import ``MetaTrader5``, initialize the terminal, and log into the demo account.

    Reads ``cfg.mt5`` (``server``, ``login``, ``password``, optional ``terminal_path``).
    Returns the connected module. Raises ``RuntimeError`` on failure. VM-only.
    """
    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "MetaTrader5 is Windows-only and not installed here. Run this on the "
            "Windows VM/VPS with the MT5 terminal.") from exc

    m = cfg.mt5
    path = str(getattr(m, "terminal_path", "") or "")
    login = int(getattr(m, "login", 0) or 0)
    password = str(getattr(m, "password", "") or "")
    server = str(getattr(m, "server", "") or "")

    # Pass credentials INTO initialize() (one call) -- more reliable than initialize()
    # followed by a separate login(), and works whether or not the terminal is already
    # logged in. Only pass account args when a login is configured.
    init_kwargs: dict = {}
    if path:
        init_kwargs["path"] = path
    if login:
        init_kwargs.update(login=login, password=password, server=server)

    if not mt5.initialize(**init_kwargs):
        err = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(
            f"mt5.initialize failed: {err}. Checklist: (1) the MT5 terminal is installed "
            f"and 'Allow algorithmic trading' is enabled (Tools -> Options -> Expert "
            f"Advisors); (2) mt5.login/password/server in config/mt5.yaml are exactly right "
            f"(server={server!r}); (3) try logging into the account in the terminal first "
            f"(File -> Login to Trade Account) and confirm it connects.")
    return mt5
