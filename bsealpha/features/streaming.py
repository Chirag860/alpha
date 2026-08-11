"""Event-driven streaming feature engine (§7.2, §8.3).

The #1 production bug in this domain is training/serving skew: features computed offline
from files differ subtly from those computed online from the live stream. The defense is
to compute the per-name microstructure features with an **event-driven state machine** that
processes depth/trade events in strict local-receipt order with O(1) state, and to run the
*same* cross-sectional/residual code (:func:`~bsealpha.features.engine.finalize_features`)
on its output as on the batch output.

:class:`ScripState` is that state machine for one name; :class:`StreamingFeatureEngine`
drives it across the panel and emits a per-name minute grid whose columns match
:func:`~bsealpha.features.engine.build_raw_grid` (verified by the parity test to 1e-6).

Per-name state (all O(1) / O(window)):
* OFI accumulator (contemporaneous depth scaling, matching the batch definition),
* last-in-minute depth snapshot for book/micro features,
* tick-rule sign carried across the day for signed flow,
* trailing deques for the 5/30-minute OFI sums and the 30-minute spread-relative window.

Day boundaries reset intraday state (OFI history, prev-minute mid, tick sign); the
spread-relative window persists across days, mirroring the batch ``rolling_mean.over(name)``.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import polars as pl

from ..config import Config
from ..data.schema import N_DEPTH_LEVELS


class ScripState:
    """O(1) streaming feature state for a single scrip (§8.3)."""

    def __init__(self, cfg: Config, m: int = N_DEPTH_LEVELS) -> None:
        self.cfg = cfg
        self.m = m
        self.spread_hist: deque[float] = deque(maxlen=30)   # persists across days
        self.new_day()

    # -- lifecycle --------------------------------------------------------
    def new_day(self) -> None:
        self._prev = None                 # previous snapshot arrays (bp, bq, ap, aq)
        self.ofi_hist: deque[float] = deque(maxlen=30)
        self.prev_minute_mid: float | None = None
        self.tick_last: float | None = None
        self.tick_sign_last: int = 1
        self.start_minute()

    def start_minute(self) -> None:
        self._last_snap = None
        self._ofi_min_sum = 0.0
        self._t_open = self._t_high = self._t_low = None
        self._val_sum = 0.0
        self._qty_sum = 0.0
        self._signed_val = 0.0
        self._large_val = 0.0
        self._count = 0

    # -- event handlers ---------------------------------------------------
    def on_book(self, bp: np.ndarray, bq: np.ndarray,
                ap: np.ndarray, aq: np.ndarray) -> None:
        """Consume a depth snapshot: accumulate OFI, remember it as last-in-minute."""
        if self._prev is None:
            ofi = 0.0
        else:
            pbp, pbq, pap, paq = self._prev
            e_bid = np.where(bp > pbp, bq, np.where(bp < pbp, -pbq, bq - pbq))
            e_ask = np.where(ap < pap, aq, np.where(ap > pap, -paq, aq - paq))
            scaled = (e_bid - e_ask) / np.maximum(0.5 * (bq + aq), 1e-12)
            ofi = float(scaled.sum())
        self._ofi_min_sum += ofi
        self._prev = (bp.copy(), bq.copy(), ap.copy(), aq.copy())
        self._last_snap = (bp, bq, ap, aq)

    def on_trade(self, price: float, qty: float) -> None:
        """Consume a trade: tick-rule sign + flow/OHLC accumulation."""
        if self.tick_last is None:
            sign = 1
        else:
            d = price - self.tick_last
            sign = 1 if d > 0 else (-1 if d < 0 else self.tick_sign_last)
        self.tick_sign_last = sign
        self.tick_last = price
        val = price * qty
        if self._t_open is None:
            self._t_open = self._t_high = self._t_low = price
        else:
            self._t_high = max(self._t_high, price)
            self._t_low = min(self._t_low, price)
        self._val_sum += val
        self._qty_sum += qty
        self._signed_val += sign * val
        self._large_val = max(self._large_val, val)
        self._count += 1

    # -- minute close -----------------------------------------------------
    def close_minute(self) -> dict | None:
        """Emit the minute's raw feature row, or ``None`` if the minute had no snapshot."""
        if self._last_snap is None:
            return None
        bp, bq, ap, aq = self._last_snap
        mid = 0.5 * (bp[0] + ap[0])
        imb0 = bq[0] / (bq[0] + aq[0])
        micro = ap[0] * imb0 + bp[0] * (1.0 - imb0)
        tot_bid = float(bq.sum())
        tot_ask = float(aq.sum())
        imb_all = tot_bid / (tot_bid + tot_ask)
        wmid = ap[0] * imb_all + bp[0] * (1.0 - imb_all)
        bbo = bq[0] + aq[0]
        deep = tot_bid + tot_ask - bbo
        deep_bid = float(bq[1:].mean())
        deep_ask = float(aq[1:].mean())

        ret = 0.0 if self.prev_minute_mid is None else float(np.log(mid) - np.log(self.prev_minute_mid))

        self.ofi_hist.append(self._ofi_min_sum)
        ofi_5m = float(sum(list(self.ofi_hist)[-5:]))
        ofi_30m = float(sum(self.ofi_hist))

        spread_bps = (ap[0] - bp[0]) / mid * 1e4
        self.spread_hist.append(float(spread_bps))
        spread_rel = (float(spread_bps) / max(float(np.mean(self.spread_hist)), 1e-6)
                      if len(self.spread_hist) >= 3 else None)

        if self._count > 0:
            gross = max(self._val_sum, 1e-9)
            signed_vol_frac = self._signed_val / gross
            large_print = self._large_val / gross
            vwap_trade = self._val_sum / self._qty_sum
            vwap_minus_mid = (vwap_trade - mid) / mid * 1e4
            trade_count = self._count
        else:
            signed_vol_frac = None
            large_print = None
            vwap_minus_mid = 0.0
            trade_count = None

        self.prev_minute_mid = mid
        return {
            "mid": float(mid), "micro": float(micro), "ret": ret,
            "micro_minus_mid": float((micro - mid) / mid * 1e4),
            "wmid_minus_mid": float((wmid - mid) / mid * 1e4),
            "imb_top": float(imb0 - 0.5), "imb_all": float(imb_all),
            "depth_imb_total": float((tot_bid - tot_ask) / (tot_bid + tot_ask)),
            "spread_bps": float(spread_bps),
            "depth_ratio": float(bbo / (deep + 1e-9)),
            "depth_slope": float((deep_bid + deep_ask) / (bbo + 1e-9)),
            "log_depth_bid": float(np.log(tot_bid + 1.0)),
            "log_depth_ask": float(np.log(tot_ask + 1.0)),
            "spread_rel": spread_rel,
            "ofi_1m": float(self._ofi_min_sum), "ofi_5m": ofi_5m, "ofi_30m": ofi_30m,
            "signed_vol_frac": signed_vol_frac, "large_print": large_print,
            "trade_count": trade_count, "vwap_minus_mid": float(vwap_minus_mid),
        }


class StreamingFeatureEngine:
    """Drive :class:`ScripState` across the panel to produce the raw per-name minute grid.

    Replaying the same events a live feed would deliver -- so historical mode is *just*
    replay of the same code (§8.3), and there is no separate 'training' feature path to
    drift from production.
    """

    def __init__(self, cfg: Config, m: int = N_DEPTH_LEVELS) -> None:
        self.cfg = cfg
        self.m = m

    def run(self, depth: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:
        """Return a per-``(scrip_code, date, minute)`` raw feature grid (streaming path)."""
        m = self.m
        bid_px = [f"bid_px_{i}" for i in range(m)]
        bid_qty = [f"bid_qty_{i}" for i in range(m)]
        ask_px = [f"ask_px_{i}" for i in range(m)]
        ask_qty = [f"ask_qty_{i}" for i in range(m)]

        depth = depth.with_columns(pl.col("session_min").floor().cast(pl.Int64).alias("minute"))
        trades = trades.with_columns(pl.col("session_min").floor().cast(pl.Int64).alias("minute"))

        rows: list[dict] = []
        for scrip in depth["scrip_code"].unique().sort().to_list():
            state = ScripState(self.cfg, m)
            d_s = depth.filter(pl.col("scrip_code") == scrip)
            t_s = trades.filter(pl.col("scrip_code") == scrip)
            for date in d_s["date"].unique().sort().to_list():
                state.new_day()
                dd = d_s.filter(pl.col("date") == date).sort("ts_ns")
                tt = t_s.filter(pl.col("date") == date).sort("ts_ns")
                self._run_day(state, scrip, date, dd, tt, rows,
                              bid_px, bid_qty, ask_px, ask_qty)
        return pl.DataFrame(rows).sort(["date", "minute", "scrip_code"])

    def _run_day(self, state, scrip, date, dd, tt, rows,
                 bid_px, bid_qty, ask_px, ask_qty) -> None:
        # numpy event arrays for the day
        d_ts = dd["ts_ns"].to_numpy(); d_min = dd["minute"].to_numpy()
        d_bp = dd.select(bid_px).to_numpy(); d_bq = dd.select(bid_qty).to_numpy()
        d_ap = dd.select(ask_px).to_numpy(); d_aq = dd.select(ask_qty).to_numpy()
        t_ts = tt["ts_ns"].to_numpy(); t_min = tt["minute"].to_numpy()
        t_px = tt["price"].to_numpy(); t_qty = tt["qty"].to_numpy()

        i = j = 0
        cur_minute = None
        while i < len(d_ts) or j < len(t_ts):
            take_book = j >= len(t_ts) or (i < len(d_ts) and d_ts[i] <= t_ts[j])
            ev_min = int(d_min[i]) if take_book else int(t_min[j])
            if cur_minute is not None and ev_min != cur_minute:
                row = state.close_minute()
                if row is not None:
                    rows.append({"scrip_code": scrip, "date": date,
                                 "minute": cur_minute, **row})
                state.start_minute()
            cur_minute = ev_min
            if take_book:
                state.on_book(d_bp[i], d_bq[i], d_ap[i], d_aq[i])
                i += 1
            else:
                state.on_trade(float(t_px[j]), float(t_qty[j]))
                j += 1
        row = state.close_minute()
        if row is not None and cur_minute is not None:
            rows.append({"scrip_code": scrip, "date": date, "minute": cur_minute, **row})
