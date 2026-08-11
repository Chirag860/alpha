"""Infrastructure kill switches and risk gates (§7.3, §8.4).

Kill switches on **infrastructure, not just P&L**: a websocket gap, feed staleness, a
clock-drift breach, an order-reject-rate breach, or a local-vs-broker position mismatch
must flatten the book immediately, alert, and **not auto-restart** (§7.3, §8.4). Once
tripped, the monitor stays halted until a human restarts with a written post-mortem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config


@dataclass
class RiskLimits:
    max_feed_staleness_s: float = 5.0
    max_reject_rate: float = 0.02
    max_clock_drift_ms: float = 50.0
    max_position_mismatch: float = 1.0e5      # rupees

    @classmethod
    def from_config(cls, cfg: Config) -> "RiskLimits":
        r = cfg.risk
        return cls(
            max_feed_staleness_s=float(r.max_feed_staleness_s),
            max_reject_rate=float(r.max_reject_rate),
            max_clock_drift_ms=float(r.max_clock_drift_ms),
            max_position_mismatch=float(r.max_position_mismatch),
        )


@dataclass
class RiskMonitor:
    """Latches to HALTED on the first breach; refuses to clear without an explicit reset."""

    limits: RiskLimits
    halted: bool = False
    reasons: list[str] = field(default_factory=list)

    def check(self, *, feed_age_s: float, reject_rate: float, clock_drift_ms: float = 0.0,
              snapshot_gap: bool = False, position_mismatch: float = 0.0) -> list[str]:
        """Evaluate all kill-switch conditions; latch and return any breaches (§7.3)."""
        breaches: list[str] = []
        if feed_age_s > self.limits.max_feed_staleness_s:
            breaches.append(f"feed_staleness {feed_age_s:.1f}s")
        if reject_rate > self.limits.max_reject_rate:
            breaches.append(f"reject_rate {reject_rate:.3f}")
        if abs(clock_drift_ms) > self.limits.max_clock_drift_ms:
            breaches.append(f"clock_drift {clock_drift_ms:.0f}ms")
        if snapshot_gap:
            breaches.append("snapshot_gap")
        if position_mismatch > self.limits.max_position_mismatch:
            breaches.append(f"position_mismatch Rs{position_mismatch:,.0f}")
        if breaches:
            self.halted = True
            self.reasons = breaches
        return breaches

    def reset(self, *, acknowledged_by: str) -> None:
        """Manual restart only -- never auto-restart after a kill (§7.3)."""
        if not acknowledged_by:
            raise ValueError("a kill-switch reset requires a written acknowledgement (§7.3)")
        self.halted = False
        self.reasons = []
