"""Deployment safety guard -- makes accidental real-money trading impossible (§7.3, §8.1).

This system has **never touched a live broker** and must not route real orders until a human
has cleared every gate below. To make "oops, it was pointed at prod" impossible, real order
routing is blocked unless BOTH are true:

1. the deployment mode is explicitly ``LIVE``, and
2. the environment variable ``BSEALPHA_LIVE_CONFIRM`` equals the exact acknowledgement
   string -- a deliberate speed-bump so live trading is always a conscious act.

``PAPER`` (default) and ``DRY_RUN`` never route real orders. ``DEMO`` routes to a broker
**demo/paper account** (no real capital), so it is exempt from the live-confirmation gate but
still sends orders. A preflight checklist verifies the operational gates before live is even
permitted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

LIVE_CONFIRM_ENV = "BSEALPHA_LIVE_CONFIRM"
LIVE_CONFIRM_VALUE = "I_UNDERSTAND_THE_RISK"


class DeploymentMode(str, Enum):
    PAPER = "paper"        # PaperBroker only; no real orders ever
    DRY_RUN = "dry_run"    # compute + log intended orders against a live feed; do not send
    DEMO = "demo"          # route to a broker DEMO account (no real capital; no live gate)
    LIVE = "live"          # real orders -- gated behind explicit confirmation


@dataclass
class PreflightResult:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        head = "PREFLIGHT PASSED" if self.passed else "PREFLIGHT FAILED"
        rows = [f"  [{'x' if v else ' '}] {k}" for k, v in self.checks.items()]
        return "\n".join([head, *rows])


class DeploymentGuard:
    """Gate that must be satisfied before any real order leaves the process."""

    def __init__(self, mode: DeploymentMode = DeploymentMode.PAPER) -> None:
        self.mode = DeploymentMode(mode)

    @classmethod
    def from_config(cls, cfg) -> "DeploymentGuard":
        mode = getattr(getattr(cfg, "deploy", None), "mode", "paper")
        return cls(DeploymentMode(mode))

    def routes_real_orders(self) -> bool:
        return self.mode == DeploymentMode.LIVE

    def assert_can_route_real_orders(self) -> None:
        """Raise unless mode is LIVE *and* the operator has set the confirmation env var."""
        if self.mode != DeploymentMode.LIVE:
            raise PermissionError(
                f"deployment mode is {self.mode.value!r}; real orders are blocked. "
                "PAPER/DRY_RUN never route to a broker (§7.3).")
        if os.environ.get(LIVE_CONFIRM_ENV) != LIVE_CONFIRM_VALUE:
            raise PermissionError(
                f"LIVE mode requires {LIVE_CONFIRM_ENV}={LIVE_CONFIRM_VALUE!r} in the "
                "environment. Set it only after preflight passes and you accept the risk.")

    def preflight(self, *, model_loaded: bool, universe_screened: bool,
                  algo_id_present: bool, kill_switches_armed: bool,
                  clock_synced: bool, lockbox_passed: bool) -> PreflightResult:
        """Verify the operational go-live gates. All must be True to permit LIVE (§8, §11.1)."""
        checks = {
            "model trained on REAL data & loaded": model_loaded,
            "point-in-time universe screened today": universe_screened,
            "Algo-ID present on order path (SEBI)": algo_id_present,
            "kill switches armed (feed/gap/reject/pos)": kill_switches_armed,
            "clock synced (<5ms drift vs NTP)": clock_synced,
            "research lockbox passed on first touch": lockbox_passed,
        }
        failures = [k for k, v in checks.items() if not v]
        return PreflightResult(passed=not failures, checks=checks, failures=failures)
