"""Forced-flatten schedule (§0.4, §7.3).

The strategy is structurally required to be flat by 15:30 -- carrying overnight retroactively
converts the buy leg into a 20 bps delivery trade (§0.4). So the schedule escalates: begin
unwinding passively at 15:15, switch to market orders at 15:25, and be guaranteed flat by
15:28. Never let an intraday position become a delivery position.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config

# phases, in order of urgency
NORMAL = "normal"
FLATTEN = "flatten"       # 15:15+ : unwind passively (LIMIT)
ESCALATE = "escalate"     # 15:25+ : cross the spread (MARKET)
HARD = "hard"             # 15:28+ : must be flat


@dataclass
class ForcedFlattenSchedule:
    flatten_start_min: int = 915     # 15:15 (minute-of-day)
    escalate_min: int = 925          # 15:25
    hard_flat_min: int = 928         # 15:28

    @classmethod
    def from_config(cls, cfg: Config) -> "ForcedFlattenSchedule":
        e = cfg.execution
        return cls(int(e.flatten_start_min), int(e.escalate_min), int(e.hard_flat_min))

    def phase(self, minute_of_day: float) -> str:
        if minute_of_day >= self.hard_flat_min:
            return HARD
        if minute_of_day >= self.escalate_min:
            return ESCALATE
        if minute_of_day >= self.flatten_start_min:
            return FLATTEN
        return NORMAL

    def is_flattening(self, minute_of_day: float) -> bool:
        return self.phase(minute_of_day) != NORMAL

    def can_open_new(self, minute_of_day: float) -> bool:
        """No new positions once the forced-flatten window starts."""
        return self.phase(minute_of_day) == NORMAL

    def order_type(self, minute_of_day: float) -> str:
        """LIMIT while unwinding passively; MARKET once escalating."""
        return "MARKET" if self.phase(minute_of_day) in (ESCALATE, HARD) else "LIMIT"
