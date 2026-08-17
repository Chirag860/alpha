# region imports
from AlgorithmImports import *
import numpy as np
# endregion

# =============================================================================
# Connors "Crash" — short mean-reversion / crash-risk-premium on high-vol US equities
# QuantConnect / LEAN research port. Companion to bsealpha/crash/ (the indicator math here is
# a scalar-"latest" copy of the verified functions in bsealpha/crash/indicators.py).
#
# WHY QUANTCONNECT: its US Equity Security Master is survivorship-bias-free and includes
# delisted tickers with delisting events — the #3 kill risk (survivorship) handled for free.
#
# WHAT QC DOES *NOT* DO FOR FREE, and is modeled here explicitly:
#   * Borrow cost. QC's default charges NO stock-loan fee. Borrow is the #1 kill risk (the
#     HV>100% screen IS a hard-to-borrow screen). We deduct it daily — see BORROW_ANNUAL /
#     borrow_rate_for(). RUN THE SCENARIO BANDS: 0.00, 0.15, 0.50, 1.00.
#   * Shorts pay dividends: QC's default DOES handle this. Good.
#
# HONESTY LEVERS STILL OPTIMISTIC (documented, tackle as a second pass — see README):
#   * Fills: LEAN fills a sell-limit when the daily HIGH >= limit ("touched"), not "traded
#     THROUGH". Real fills are adverse-selected. Treat fill-heavy results with suspicion.
#   * No per-name shortable/locate data on the free tier: we assume every signal is shortable
#     (reality: 10-25% are not, non-randomly the best ones). AVAILABILITY_HAIRCUT approximates it.
#
# FIRST CHECKPOINT (proves the port is faithful BEFORE trusting any net number):
#   Set BORROW_ANNUAL=0.0 and AVAILABILITY_HAIRCUT=0.0 and compare QC's reported
#   Win Rate / Average Win / Average Loss / Profit-Loss Ratio against EdgeRater's published
#   gross stats (~70% win, ~+14% / -15%, PF ~1.9). If those don't roughly match, the port is
#   wrong — debug that before interpreting Sharpe.
# =============================================================================


class ConnorsCrashStrategy(QCAlgorithm):

    # --------- strategy params (defaults = book baseline; user's spec = 3% limit) -----------
    CRSI_ENTRY = 90.0          # short when ConnorsRSI >= this
    CRSI_EXIT = 30.0           # cover when ConnorsRSI < this
    HV_MIN = 1.00              # HV(100) must exceed this (decimal; 1.00 == 100%)
    LIMIT_PCT = 0.03           # sell-short limit = close * (1 + LIMIT_PCT); book primary = 0.05
    MIN_PRICE = 5.0
    MIN_AVG_VOLUME = 1_000_000 # 21-day avg SHARE volume floor (the BOOK screen — shares, not $)
    AVG_VOL_WINDOW = 21
    RANK_WINDOW = 100          # ConnorsRSI percent-rank window and HV window
    WARMUP_BARS = 102          # closes needed before the first valid CRSI

    TARGET_POSITIONS = 40      # NOTE: not from the book (Alvarez suggests ~10). A research knob.
    TARGET_WEIGHT = 0.025      # 2.5% of equity per name (short => negative notional)
    ALLOW_DUPLICATES = False   # book-silent; duplicates are a martingale. Keep off.

    # --------- honesty knobs — RUN THE BANDS ------------------------------------------------
    BORROW_ANNUAL = 0.50       # flat annual borrow fee. Sweep: 0.00 / 0.15 / 0.50 / 1.00.
    USE_HV_BORROW_PROXY = False # if True, borrow scales with HV (crude proxy for the real fee)
    AVAILABILITY_HAIRCUT = 0.15 # fraction of setups randomly dropped as "no borrow / no locate"
    COVER_SLIPPAGE_BPS = 20.0  # extra slippage charged on the cover (market order)

    def Initialize(self):
        self.SetStartDate(2016, 1, 1)   # warm-up + covers 2018 vol, 2020 crash, 2021 SQUEEZE, 2022-25
        self.SetEndDate(2017, 12, 31)   # TEMP diagnostic cap — revert to 2025-12-31 once fills confirmed
        self.SetCash(100_000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)

        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.MinimumTimeInUniverse = timedelta(days=1)
        self.AddUniverse(self.CoarseSelection)

        # Day-order semantics WITHOUT TimeInForce.Day: at daily resolution a Day order expires
        # before the next daily bar can fill it (bar T arrives after T's close; a Day order dies
        # at end of that day; bar T+1 — the one that would fill it — arrives the following day).
        # So we leave orders GTC (default) and cancel yesterday's unfilled limits at the top of
        # each OnData, AFTER they've had their one-bar fill chance. This actually fills.

        self.symbol_data: dict[Symbol, SymbolData] = {}
        self._rng = np.random.default_rng(7)  # seeded availability haircut -> reproducible
        self._diag_day = 0        # funnel-diagnostic day counter
        self._orders_total = 0    # cumulative sell-short orders placed

        self.SetWarmUp(0)  # we backfill per-symbol via History in OnSecuritiesChanged instead
        self.Settings.FreePortfolioValuePercentage = 0.05

    # ---------------------------------------------------------------- universe (the book screen)
    def CoarseSelection(self, coarse):
        # HasFundamentalData excludes most ETFs/ETNs/leveraged products (brief §3.2). Price > $5.
        # Single-day volume >= 0.9M is a cheap prefilter; the true 21-day avg is enforced in-algo.
        # NOTE: deliberately NOT capped by dollar volume — that would cut the small-cap high-vol
        # tail this strategy targets.
        return [c.Symbol for c in coarse
                if c.HasFundamentalData and c.Price > self.MIN_PRICE and c.Volume >= 900_000]

    def OnSecuritiesChanged(self, changes):
        for sec in changes.AddedSecurities:
            sym = sec.Symbol
            if sym in self.symbol_data:
                continue
            sd = SymbolData(sym)
            # backfill so indicators are warm quickly (survivorship-free history)
            hist = self.History(sym, self.WARMUP_BARS + self.AVG_VOL_WINDOW, Resolution.Daily)
            if not hist.empty and "close" in hist.columns:
                for close, vol in zip(hist["close"].values, hist["volume"].values):
                    sd.update(float(close), float(vol))
            self.symbol_data[sym] = sd

        for sec in changes.RemovedSecurities:
            sym = sec.Symbol
            self.symbol_data.pop(sym, None)
            # FORCED LIQUIDATION ON STRUCTURAL REMOVAL (delist/halt/universe drop) — the ONLY
            # non-signal exit. The screen (HV/price/vol) must NEVER act as a hidden exit; only
            # CRSI<30 and structural removal close a position.
            if self.Portfolio[sym].Invested:
                self.Liquidate(sym, tag="universe-removal")

    # ---------------------------------------------------------------- daily engine
    def OnData(self, data: Slice):
        if self.IsWarmingUp:
            return

        # 0) cancel yesterday's unfilled sell-limits — they had their one-bar fill chance against
        #    the bar we just processed; carrying them forward would be a stale GTC (not the book).
        self.Transactions.CancelOpenOrders()

        # 1) update indicators with today's close for every subscribed name that printed
        for sym, sd in self.symbol_data.items():
            bar = data.Bars.get(sym)
            if bar is not None:
                sd.update(float(bar.Close), float(bar.Volume))

        # 2) charge borrow on every open short (daily accrual on marked-to-market value)
        self._charge_borrow()

        # 3) COVER FIRST: held names whose CRSI has fallen below the exit threshold
        for sym in list(self.symbol_data.keys()):
            if not self.Portfolio[sym].Invested:
                continue
            sd = self.symbol_data[sym]
            crsi = sd.connors_rsi()
            if crsi is not None and crsi < self.CRSI_EXIT:
                self.Liquidate(sym, tag=f"CRSI<{self.CRSI_EXIT:.0f} ({crsi:.1f})")

        # 4) rank new setups and OPEN shorts into available slots
        setups = []
        for sym, sd in self.symbol_data.items():
            if self.Portfolio[sym].Invested and not self.ALLOW_DUPLICATES:
                continue
            sig = sd.evaluate(self)
            if sig is not None:
                setups.append(sig)  # (crsi, symbol, limit_price)
        setups.sort(key=lambda s: s[0], reverse=True)  # highest CRSI (most extreme) first

        slots = self._open_slots()
        equity = self.Portfolio.TotalPortfolioValue
        placed = 0
        for crsi, sym, limit_price in setups:
            if slots <= 0:
                break
            # availability haircut: some setups are simply not shortable in reality
            if self.AVAILABILITY_HAIRCUT > 0 and self._rng.random() < self.AVAILABILITY_HAIRCUT:
                continue
            shares = int((self.TARGET_WEIGHT * equity) / limit_price)  # size on the LIMIT price
            if shares <= 0:
                continue
            self.LimitOrder(sym, -shares, round(limit_price, 2),
                            tag=f"short CRSI={crsi:.1f}")
            slots -= 1  # counts against the cap immediately (working order occupies a slot)
            placed += 1
        self._orders_total += placed

        # ---- monthly funnel diagnostic: where do candidates vanish? -------------------------
        self._diag_day += 1
        if self._diag_day % 21 == 0:
            n_uni = len(self.symbol_data)
            n_warm = sum(1 for sd in self.symbol_data.values()
                         if len(sd.closes) >= self.WARMUP_BARS)
            held = sum(1 for kv in self.Portfolio.Values if kv.Invested and kv.IsShort)
            self.Debug(f"{self.Time.date()} | universe={n_uni} warm={n_warm} "
                       f"setups={len(setups)} placed={placed} held={held} "
                       f"orders_total={self._orders_total} equity={equity:,.0f}")

    # ---------------------------------------------------------------- helpers
    def _open_slots(self) -> int:
        held = sum(1 for kv in self.Portfolio.Values if kv.Invested and kv.IsShort)
        working = len({t.Symbol for t in self.Transactions.GetOpenOrders()})
        return self.TARGET_POSITIONS - held - working  # the working-order subtraction is essential

    def _charge_borrow(self):
        total = 0.0
        for sym in self.symbol_data:
            h = self.Portfolio[sym]
            if h.Invested and h.IsShort:
                rate = self.borrow_rate_for(sym)
                total += abs(h.HoldingsValue) * rate / 360.0  # daily accrual, 360-day basis
        if total > 0:
            self.Portfolio.CashBook["USD"].AddAmount(-total)

    def borrow_rate_for(self, sym) -> float:
        if not self.USE_HV_BORROW_PROXY:
            return self.BORROW_ANNUAL
        sd = self.symbol_data.get(sym)
        hv = sd.hv() if sd else None
        if hv is None:
            return self.BORROW_ANNUAL
        # crude proxy: fee rises with realized vol; clamp to [15%, 300%]. Replace with real
        # borrow data (S3/Ortex) before trusting the net number.
        return float(np.clip(0.10 + 0.50 * (hv - 1.0), 0.15, 3.0))

    def OnOrderEvent(self, order_event):
        # approximate cover slippage as an extra cash cost on filled buy-to-cover market orders
        if order_event.Status == OrderStatus.Filled and self.COVER_SLIPPAGE_BPS > 0:
            order = self.Transactions.GetOrderById(order_event.OrderId)
            if order is not None and order.Type == OrderType.Market and order_event.FillQuantity > 0:
                notional = abs(order_event.FillQuantity) * order_event.FillPrice
                self.Portfolio.CashBook["USD"].AddAmount(-notional * self.COVER_SLIPPAGE_BPS / 1e4)

    def OnEndOfAlgorithm(self):
        self.Log(f"TOTAL sell-short orders placed over the run = {self._orders_total}")
        self.Log(f"Borrow band = {self.BORROW_ANNUAL:.0%} | availability haircut = "
                 f"{self.AVAILABILITY_HAIRCUT:.0%} | limit = {self.LIMIT_PCT:.0%} above close")
        self.Log("Compare QC's Win Rate / Avg Win / Avg Loss / PL-ratio vs EdgeRater "
                 "(~70% / +14% / -15% / ~1.9) at BORROW=0 to confirm the port is faithful.")


class SymbolData:
    """Rolling per-symbol close/volume history + scalar 'latest' ConnorsRSI / HV / avg-volume.

    The indicator math mirrors bsealpha/crash/indicators.py exactly (Wilder SMA-seed RSI — NOT
    ewm; strict percent-rank excluding today; hard streak reset on unchanged close; HV = sample
    std of 100 log returns * sqrt(252))."""
    MAXLEN = 140

    def __init__(self, symbol):
        self.symbol = symbol
        self.closes: list[float] = []
        self.volumes: list[float] = []

    def update(self, close: float, volume: float):
        self.closes.append(close)
        self.volumes.append(volume)
        if len(self.closes) > self.MAXLEN:
            self.closes = self.closes[-self.MAXLEN:]
            self.volumes = self.volumes[-self.MAXLEN:]

    # -- screen inputs -------------------------------------------------------
    def avg_volume(self):
        if len(self.volumes) < ConnorsCrashStrategy.AVG_VOL_WINDOW:
            return None
        return float(np.mean(self.volumes[-ConnorsCrashStrategy.AVG_VOL_WINDOW:]))

    def hv(self):
        c = np.asarray(self.closes, dtype=float)
        w = ConnorsCrashStrategy.RANK_WINDOW
        if c.shape[0] < w + 1:
            return None
        r = np.log(c[-(w + 1):][1:] / c[-(w + 1):][:-1])
        return float(np.std(r, ddof=1) * np.sqrt(252.0))

    def connors_rsi(self):
        c = np.asarray(self.closes, dtype=float)
        if c.shape[0] < ConnorsCrashStrategy.WARMUP_BARS:
            return None
        rsi_c = _wilder_rsi_last(c, 3)
        rsi_s = _wilder_rsi_last(_streak(c), 2)
        pr = _percent_rank_last(_roc1(c), ConnorsCrashStrategy.RANK_WINDOW)
        if rsi_c is None or rsi_s is None or pr is None:
            return None
        return (rsi_c + rsi_s + pr) / 3.0

    # -- setup decision ------------------------------------------------------
    def evaluate(self, algo: "ConnorsCrashStrategy"):
        """Return (crsi, symbol, limit_price) if this is a valid short setup at today's close."""
        if len(self.closes) < ConnorsCrashStrategy.WARMUP_BARS:
            return None  # not warmed up (empty/short history) -> no signal, no index error
        if self.closes[-1] <= algo.MIN_PRICE:
            return None
        av = self.avg_volume()
        if av is None or av < algo.MIN_AVG_VOLUME:
            return None
        hv = self.hv()
        if hv is None or hv <= algo.HV_MIN:
            return None
        crsi = self.connors_rsi()
        if crsi is None or crsi < algo.CRSI_ENTRY:
            return None
        return (crsi, self.symbol, self.closes[-1] * (1.0 + algo.LIMIT_PCT))


# ---- scalar 'latest' indicator helpers (exact copies of the verified bsealpha/crash math) ----
def _wilder_rsi_last(v: np.ndarray, period: int):
    v = np.asarray(v, dtype=float)
    if v.shape[0] <= period:
        return None
    delta = np.diff(v)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = gain[:period].mean()
    al = loss[:period].mean()
    for i in range(period, v.shape[0] - 1):
        ag = (ag * (period - 1) + gain[i]) / period
        al = (al * (period - 1) + loss[i]) / period
    total = ag + al
    return 50.0 if total <= 0.0 else 100.0 * ag / total


def _streak(close: np.ndarray) -> np.ndarray:
    c = np.asarray(close, dtype=float)
    s = np.zeros(c.shape[0])
    for t in range(1, c.shape[0]):
        if c[t] > c[t - 1]:
            s[t] = s[t - 1] + 1 if s[t - 1] > 0 else 1.0
        elif c[t] < c[t - 1]:
            s[t] = s[t - 1] - 1 if s[t - 1] < 0 else -1.0
        else:
            s[t] = 0.0
    return s


def _roc1(close: np.ndarray) -> np.ndarray:
    c = np.asarray(close, dtype=float)
    out = np.full(c.shape[0], np.nan)
    out[1:] = 100.0 * (c[1:] / c[:-1] - 1.0)
    return out


def _percent_rank_last(values: np.ndarray, window: int):
    v = np.asarray(values, dtype=float)
    if v.shape[0] < window + 1:
        return None
    prior = v[-(window + 1):-1]
    x = v[-1]
    if np.isnan(x) or np.isnan(prior).any():
        return None
    return 100.0 * np.count_nonzero(prior < x) / window
