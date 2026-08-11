"""Deployment-safety tests: real orders are impossible without explicit confirmation."""

from __future__ import annotations

import pytest

from bsealpha.config import load_config
from bsealpha.execution import (
    DeploymentGuard,
    DeploymentMode,
    ExecutionManager,
    Order,
    PaperBroker,
)
from bsealpha.execution.safety import LIVE_CONFIRM_ENV, LIVE_CONFIRM_VALUE


def test_default_mode_is_paper():
    cfg = load_config()
    guard = DeploymentGuard.from_config(cfg)
    assert guard.mode == DeploymentMode.PAPER
    assert not guard.routes_real_orders()


def test_paper_and_dry_run_block_real_orders():
    for mode in (DeploymentMode.PAPER, DeploymentMode.DRY_RUN):
        with pytest.raises(PermissionError):
            DeploymentGuard(mode).assert_can_route_real_orders()


def test_live_requires_env_confirmation(monkeypatch):
    guard = DeploymentGuard(DeploymentMode.LIVE)
    monkeypatch.delenv(LIVE_CONFIRM_ENV, raising=False)
    with pytest.raises(PermissionError):
        guard.assert_can_route_real_orders()
    monkeypatch.setenv(LIVE_CONFIRM_ENV, "wrong")
    with pytest.raises(PermissionError):
        guard.assert_can_route_real_orders()
    monkeypatch.setenv(LIVE_CONFIRM_ENV, LIVE_CONFIRM_VALUE)
    guard.assert_can_route_real_orders()          # now allowed


def test_manager_live_without_confirmation_fails_fast(monkeypatch):
    monkeypatch.delenv(LIVE_CONFIRM_ENV, raising=False)
    cfg = load_config(overrides={"deploy": {"mode": "live"}})
    with pytest.raises(PermissionError):
        ExecutionManager(PaperBroker(), cfg)      # construction is blocked in LIVE unconfirmed


def test_dry_run_computes_but_does_not_send():
    cfg = load_config(overrides={"deploy": {"mode": "dry_run"}})
    b = PaperBroker()
    mgr = ExecutionManager(b, cfg)
    market = {1: (99.95, 100.05, 100.0)}
    res = mgr.step(700, 0.0, targets={1: 8e5}, market=market)
    assert len(res.orders) == 1                   # order was computed
    assert b.order_count == 0                     # ...but NOT sent to the broker


def test_preflight_gates():
    guard = DeploymentGuard(DeploymentMode.LIVE)
    r = guard.preflight(model_loaded=True, universe_screened=True, algo_id_present=True,
                        kill_switches_armed=True, clock_synced=True, lockbox_passed=False)
    assert not r.passed
    assert "research lockbox passed on first touch" in r.failures
    r2 = guard.preflight(model_loaded=True, universe_screened=True, algo_id_present=True,
                         kill_switches_armed=True, clock_synced=True, lockbox_passed=True)
    assert r2.passed
