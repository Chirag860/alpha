"""Phase 4 tests: paper broker, token bucket, kill switches, forced flatten, manager."""

from __future__ import annotations

import pytest

from bsealpha.config import load_config
from bsealpha.execution import (
    ExecutionManager,
    ForcedFlattenSchedule,
    Order,
    PaperBroker,
    RiskLimits,
    RiskMonitor,
    TokenBucket,
)


@pytest.fixture
def cfg():
    return load_config()


# --------------------------------------------------------------- token bucket
def test_token_bucket_caps_ops():
    tb = TokenBucket(rate_per_sec=10, capacity=10)
    allowed = sum(tb.take(now_s=0.0) for _ in range(15))   # 15 orders in the same instant
    assert allowed == 10                                    # only 10 pass at t=0
    assert tb.rejected == 5
    # after 1 second, 10 more tokens refill
    assert sum(tb.take(now_s=1.0) for _ in range(10)) == 10


# --------------------------------------------------------------- paper broker
def test_paper_broker_market_fill_and_position():
    b = PaperBroker(taker_slippage_bps=1.0)
    b.update_market(500001, bid=100.0, ask=100.1)
    o = Order(500001, side=1, qty=10, order_type="MARKET", price=None,
              algo_id="A1", ts_ns=0)
    b.place_order(o)
    assert o.status == "FILLED"
    assert b.positions()[500001].qty == 10
    fills = b.poll_fills()
    assert len(fills) == 1 and fills[0].is_taker
    assert fills[0].price > 100.1                           # taker pays slippage above ask


def test_paper_broker_resting_limit_fills_on_cross():
    b = PaperBroker()
    b.update_market(1, bid=100.0, ask=100.1)
    o = Order(1, side=1, qty=5, order_type="LIMIT", price=100.0, algo_id="A1", ts_ns=0)
    b.place_order(o)
    assert o.status == "RESTING"                            # passive buy at the bid rests
    b.update_market(1, bid=99.9, ask=100.0)                 # ask crosses our 100.0 -> fill
    assert o.status == "FILLED"
    assert not b.poll_fills()[-1].is_taker                  # filled passively (maker)


def test_untagged_order_is_rejected():
    """SEBI: every algo order must carry an Algo-ID; untagged orders are rejected (§8.1)."""
    b = PaperBroker()
    b.update_market(1, 100.0, 100.1)
    o = Order(1, 1, 10, "MARKET", None, algo_id="", ts_ns=0)
    b.place_order(o)
    assert o.status == "REJECTED"
    assert b.positions().get(1) is None


# --------------------------------------------------------------- kill switches
def test_risk_monitor_latches_and_needs_manual_reset():
    rm = RiskMonitor(RiskLimits(max_feed_staleness_s=5.0))
    assert rm.check(feed_age_s=1.0, reject_rate=0.0) == []
    breaches = rm.check(feed_age_s=9.0, reject_rate=0.0)
    assert breaches and rm.halted
    # stays halted even if the condition clears (no auto-restart, §7.3)
    rm.check(feed_age_s=0.0, reject_rate=0.0)
    assert rm.halted
    with pytest.raises(ValueError):
        rm.reset(acknowledged_by="")
    rm.reset(acknowledged_by="ops-oncall post-mortem")
    assert not rm.halted


def test_reject_rate_kill_switch():
    rm = RiskMonitor(RiskLimits(max_reject_rate=0.02))
    assert rm.check(feed_age_s=0, reject_rate=0.10)         # 10% > 2%
    assert rm.halted


# --------------------------------------------------------------- scheduler
def test_forced_flatten_phases():
    s = ForcedFlattenSchedule(flatten_start_min=915, escalate_min=925, hard_flat_min=928)
    assert s.phase(900) == "normal" and s.can_open_new(900)
    assert s.phase(916) == "flatten" and not s.can_open_new(916)
    assert s.order_type(916) == "LIMIT"
    assert s.phase(926) == "escalate" and s.order_type(926) == "MARKET"
    assert s.phase(929) == "hard"


# --------------------------------------------------------------- manager e2e
def _market(scrips, px=100.0):
    return {sc: (px - 0.05, px + 0.05, px) for sc in scrips}


def _open_via_manager(mgr, b, scrip, target, minute=700):
    """Open a position through the manager: post passive, cross to fill, ingest the fill."""
    mgr.step(minute, 0.0, targets={scrip: target}, market=_market([scrip]))
    # cross the resting passive order so it fills (a counterparty hits our quote)
    if target > 0:
        b.update_market(scrip, 99.90, 99.95)      # ask crosses our resting buy at 99.95
    else:
        b.update_market(scrip, 100.05, 100.10)    # bid crosses our resting sell at 100.05


def test_manager_routes_targets_with_algo_id(cfg):
    b = PaperBroker()
    mgr = ExecutionManager(b, cfg)
    scrips = [1, 2, 3]
    targets = {1: 8e5, 2: -7e5, 3: 6e5}      # all above the 5-lakh min clip
    res = mgr.step(minute_of_day=700, now_s=0.0, targets=targets, market=_market(scrips))
    assert not res.halted and res.phase == "normal"
    assert len(res.orders) == 3
    assert all(o.algo_id == cfg.execution.algo_id for o in res.orders)   # tagged (§8.1)
    # passive maker orders rest at the touch (maker-first, §6.4); sides are correct
    by_scrip = {o.scrip_code: o for o in res.orders}
    assert by_scrip[1].side == 1 and by_scrip[2].side == -1
    assert all(o.order_type == "LIMIT" for o in res.orders)


def test_manager_skips_sub_min_clip(cfg):
    b = PaperBroker()
    mgr = ExecutionManager(b, cfg)
    res = mgr.step(700, 0.0, targets={1: 1e5}, market=_market([1]))   # below 5-lakh floor
    assert len(res.orders) == 0                                       # churn skipped (§6.2)


def test_manager_rate_limits(cfg):
    b = PaperBroker()
    mgr = ExecutionManager(b, cfg, token_bucket=TokenBucket(10, 10))
    scrips = list(range(1, 16))                       # 15 names, all tradable
    targets = {sc: 8e5 for sc in scrips}
    res = mgr.step(700, 0.0, targets=targets, market=_market(scrips))
    assert len(res.orders) == 10                      # capped at 10 OPS
    assert res.n_rate_limited == 5


def test_manager_forced_flatten(cfg):
    b = PaperBroker()
    mgr = ExecutionManager(b, cfg)
    _open_via_manager(mgr, b, 1, 8e5)
    assert b.positions()[1].qty > 0
    # after 15:25 the schedule escalates -> MARKET flatten regardless of targets
    res = mgr.step(926, 5.0, targets={1: 8e5}, market=_market([1]))
    assert res.phase == "escalate"
    assert res.orders and res.orders[0].order_type == "MARKET"
    assert abs(b.positions()[1].qty) < 1e-6           # flat


def test_manager_kill_switch_flattens(cfg):
    b = PaperBroker()
    mgr = ExecutionManager(b, cfg)
    _open_via_manager(mgr, b, 1, 8e5)
    assert b.positions()[1].qty > 0
    # a feed-staleness breach must flatten immediately and halt
    res = mgr.step(701, 1.0, targets={1: 8e5}, market=_market([1]), feed_age_s=99.0)
    assert res.halted and any("feed_staleness" in r for r in res.reasons)
    assert abs(b.positions()[1].qty) < 1e-6
