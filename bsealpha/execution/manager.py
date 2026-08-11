"""Execution manager: target book -> orders, through every live gate (§7, §8).

One ``step`` per decision cycle. Order of operations is deliberate and matches the report's
production checklist:

1. ingest market + fills, reconcile local vs broker positions;
2. **risk gates first** -- any kill-switch breach flattens immediately and halts (§7.3);
3. **forced-flatten** window overrides targets to zero and escalates LIMIT->MARKET (§7.3);
4. compute trades vs current book, skip sub-min-clip churn (§6.2);
5. **rate-limit** to ≤10 OPS via the token bucket (§8.1);
6. **tag every order with the Algo-ID** and route to the broker (§8.1).

Drivable step-by-step against :class:`~bsealpha.execution.broker.PaperBroker` so the whole
lifecycle is testable without a live account.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from .broker import BrokerAdapter, Order
from .rate_limit import TokenBucket
from .risk import RiskLimits, RiskMonitor
from .safety import DeploymentGuard, DeploymentMode
from .scheduler import ForcedFlattenSchedule


@dataclass
class StepResult:
    orders: list[Order] = field(default_factory=list)
    phase: str = "normal"
    halted: bool = False
    reasons: list[str] = field(default_factory=list)
    n_rate_limited: int = 0


class ExecutionManager:
    """Drive a target rupee book to the broker through all live gates."""

    def __init__(self, broker: BrokerAdapter, cfg: Config,
                 token_bucket: TokenBucket | None = None,
                 risk_monitor: RiskMonitor | None = None,
                 schedule: ForcedFlattenSchedule | None = None,
                 guard: DeploymentGuard | None = None) -> None:
        self.broker = broker
        self.cfg = cfg
        self.algo_id = str(cfg.execution.algo_id)
        self.min_clip = float(cfg.execution.min_clip)
        self.bucket = token_bucket or TokenBucket(float(cfg.execution.max_ops))
        self.risk = risk_monitor or RiskMonitor(RiskLimits.from_config(cfg))
        self.schedule = schedule or ForcedFlattenSchedule.from_config(cfg)
        # deployment safety: LIVE fails fast unless explicitly confirmed (§7.3, §8.1)
        self.guard = guard or DeploymentGuard.from_config(cfg)
        if self.guard.mode == DeploymentMode.LIVE:
            self.guard.assert_can_route_real_orders()
        self._expected: dict[int, float] = {}     # our view of shares, for mismatch checks

    def _send(self, order: Order) -> None:
        """Route an order unless we are in DRY_RUN (compute-and-log only, no real send)."""
        if self.guard.mode == DeploymentMode.DRY_RUN:
            return
        self.broker.place_order(order)

    # ------------------------------------------------------------------
    def _current_rupees(self, market: dict[int, tuple]) -> dict[int, float]:
        out: dict[int, float] = {}
        for sc, pos in self.broker.positions().items():
            mid = market.get(sc, (0, 0, pos.avg_price))[2]
            out[sc] = pos.qty * mid
        return out

    def _position_mismatch(self, market: dict[int, tuple]) -> float:
        mism = 0.0
        broker_pos = self.broker.positions()
        scrips = set(self._expected) | set(broker_pos)
        for sc in scrips:
            exp = self._expected.get(sc, 0.0)
            act = broker_pos.get(sc).qty if sc in broker_pos else 0.0
            mid = market.get(sc, (0, 0, 1.0))[2]
            mism += abs(exp - act) * mid
        return mism

    def _ingest_fills(self) -> None:
        for f in self.broker.poll_fills():
            self._expected[f.scrip_code] = self._expected.get(f.scrip_code, 0.0) + f.side * f.qty

    # ------------------------------------------------------------------
    def step(self, minute_of_day: float, now_s: float,
             targets: dict[int, float], market: dict[int, tuple], *,
             feed_age_s: float = 0.0, clock_drift_ms: float = 0.0,
             snapshot_gap: bool = False) -> StepResult:
        """Advance one decision cycle. ``targets`` and ``market`` are keyed by scrip_code;
        ``market[sc] = (bid, ask, mid)``. Returns the orders sent and the step state."""
        res = StepResult()
        for sc, (bid, ask, mid) in market.items():
            self.broker.update_market(sc, bid, ask, mid)
        self._ingest_fills()

        # 2) risk gates first
        breaches = self.risk.check(
            feed_age_s=feed_age_s, reject_rate=self.broker.reject_rate(),
            clock_drift_ms=clock_drift_ms, snapshot_gap=snapshot_gap,
            position_mismatch=self._position_mismatch(market),
        )
        current = self._current_rupees(market)
        if self.risk.halted:
            res.halted = True
            res.reasons = self.risk.reasons or breaches
            res.orders = self._flatten(current, market, now_s, force_market=True)
            self._ingest_fills()
            return res

        # 3) forced-flatten window
        res.phase = self.schedule.phase(minute_of_day)
        flattening = res.phase != "normal"
        if flattening:
            targets = {sc: 0.0 for sc in current}    # unwind everything
        order_type = self.schedule.order_type(minute_of_day)

        # 4-6) compute trades, min-clip filter, rate-limit, tag Algo-ID, route
        res.orders = self._route(targets, current, market, now_s, order_type,
                                 flattening=flattening, result=res)
        self._ingest_fills()
        return res

    def _route(self, targets, current, market, now_s, order_type, *, flattening,
               result: StepResult) -> list[Order]:
        # cancel-replace: retire stale resting orders before quoting afresh (§6.4)
        if hasattr(self.broker, "cancel_all_resting"):
            self.broker.cancel_all_resting()
        orders: list[Order] = []
        scrips = set(targets) | set(current)
        # trade the largest gaps first so the OPS budget goes to what matters
        gaps = []
        for sc in scrips:
            tgt = targets.get(sc, 0.0)
            cur = current.get(sc, 0.0)
            gap = tgt - cur
            closing = (tgt == 0.0 and cur != 0.0)
            if abs(gap) < self.min_clip and not closing:
                continue
            if sc not in market:
                continue
            gaps.append((abs(gap), sc, gap, tgt))
        for _, sc, gap, tgt in sorted(gaps, reverse=True):
            if not self.bucket.take(now_s):          # SEBI ≤10 OPS ceiling (§8.1)
                result.n_rate_limited += 1
                continue
            bid, ask, mid = market[sc]
            side = 1 if gap > 0 else -1
            shares = abs(gap) / mid
            if order_type == "MARKET":
                price = None
            else:                                    # passive: join our side of the book
                price = bid if side > 0 else ask
            order = Order(scrip_code=sc, side=side, qty=shares, order_type=order_type,
                          price=price, algo_id=self.algo_id, ts_ns=int(now_s * 1e9))
            self._send(order)
            orders.append(order)
        return orders

    def _flatten(self, current, market, now_s, *, force_market: bool) -> list[Order]:
        """Emergency flatten: cross immediately, never optimize execution in a risk event."""
        orders: list[Order] = []
        for sc, cur in current.items():
            if cur == 0.0 or sc not in market:
                continue
            bid, ask, mid = market[sc]
            side = -1 if cur > 0 else 1
            order = Order(scrip_code=sc, side=side, qty=abs(cur) / mid,
                          order_type="MARKET", price=None, algo_id=self.algo_id,
                          ts_ns=int(now_s * 1e9))
            self._send(order)
            orders.append(order)
        return orders
