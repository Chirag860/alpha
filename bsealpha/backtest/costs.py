"""The Indian intraday cost stack (§0.1) and square-root market impact (§6.4).

Faithful to the report's arithmetic, and to the ways India differs from crypto:

* **STT 0.025% on the SELL leg only** (2.5 bps). A long and a short round trip cost the
  same total, but the charge is booked on the sell (§6.2).
* **No maker/taker fee distinction** -- BSE transaction charges are flat both ways
  (0.00375% per side); passivity saves only the spread (§0.3).
* **Stamp duty 0.003% on the BUY leg only** (0.30 bps).
* **Brokerage is ₹20 per ORDER, flat, not per fill** -- so splitting one clip into five
  child orders quintuples brokerage (§6.2). Modeled per order sent.
* **GST 18%** on (transaction + SEBI + brokerage).
* **Impact via the square-root law against BSE ADV** (§6.4) -- in thin BSE names this
  dominates fees, the opposite of the crypto conclusion.

Config ``costs.bse_txn_bps`` / ``sebi_bps`` are stated as round-trip totals (§0.1 table);
here they are split per side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config


@dataclass
class CostParams:
    stt_bps_sell: float
    txn_bps_per_side: float
    stamp_bps_buy: float
    sebi_bps_per_side: float
    gst_rate: float
    brokerage_per_order: float
    otr_penalty_per_order: float
    impact_Y: float

    @classmethod
    def from_config(cls, cfg: Config) -> "CostParams":
        c = cfg.costs
        return cls(
            stt_bps_sell=float(c.stt_bps_sell),
            txn_bps_per_side=float(c.bse_txn_bps) / 2.0,
            stamp_bps_buy=float(c.stamp_bps_buy),
            sebi_bps_per_side=float(c.sebi_bps) / 2.0,
            gst_rate=float(c.gst_rate),
            brokerage_per_order=float(c.brokerage_per_order),
            otr_penalty_per_order=float(c.otr_penalty_per_order),
            impact_Y=float(c.impact_Y),
        )


def leg_cost_rupees(notional: float, side: int, p: CostParams) -> float:
    """Total cost in rupees for one order leg (§0.1).

    ``side`` = +1 buy, -1 sell. ``notional`` is the (positive) rupee value of the leg.
    """
    notional = abs(float(notional))
    txn = p.txn_bps_per_side * 1e-4 * notional
    sebi = p.sebi_bps_per_side * 1e-4 * notional
    stt = p.stt_bps_sell * 1e-4 * notional if side < 0 else 0.0
    stamp = p.stamp_bps_buy * 1e-4 * notional if side > 0 else 0.0
    brokerage = p.brokerage_per_order
    gst = p.gst_rate * (txn + sebi + brokerage)
    return txn + sebi + stt + stamp + brokerage + gst + p.otr_penalty_per_order


def impact_rupees(notional: float, adv_bse: float, sigma_daily: float,
                  p: CostParams) -> float:
    """Square-root-law temporary impact in rupees (§6.4).

    ``impact_fraction = Y * sigma_daily * sqrt(Q / ADV_bse)``; cost = fraction * Q. Impact
    is charged against **BSE** ADV, since BSE liquidity is what absorbs the order.
    """
    notional = abs(float(notional))
    adv_bse = max(float(adv_bse), 1.0)
    frac = p.impact_Y * float(sigma_daily) * np.sqrt(notional / adv_bse)
    return float(frac * notional)


def round_trip_cost_bps(clip_rupees: float, p: CostParams,
                        include_two_orders: bool = True) -> float:
    """Round-trip explicit cost in bps for a given clip size (reproduces §0.1's table).

    Useful as a design check: at ₹10 lakh this returns ~4.18 bps.
    """
    buy = leg_cost_rupees(clip_rupees, +1, p)
    sell = leg_cost_rupees(clip_rupees, -1, p)
    return (buy + sell) / clip_rupees * 1e4
