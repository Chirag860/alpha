"""Order-rate limiting: the SEBI ≤10 orders-per-second ceiling (§8.1).

Individual traders whose algos stay **under 10 OPS** are treated as regular API users and
skip separate strategy registration. Order splitting is doubly penalized in India -- once by
the ₹20/order brokerage, once by this budget -- so the manager must throttle in code with a
hard token bucket, especially during forced-flatten bursts and rebalance storms (§8.1).
"""

from __future__ import annotations


class TokenBucket:
    """Classic token bucket. ``rate`` tokens refill per second, capped at ``capacity``."""

    def __init__(self, rate_per_sec: float, capacity: float | None = None) -> None:
        self.rate = float(rate_per_sec)
        self.capacity = float(capacity if capacity is not None else rate_per_sec)
        self.tokens = self.capacity
        self._last: float | None = None
        self.rejected = 0

    def _refill(self, now_s: float) -> None:
        if self._last is not None:
            self.tokens = min(self.capacity, self.tokens + (now_s - self._last) * self.rate)
        self._last = now_s

    def take(self, now_s: float, n: int = 1) -> bool:
        """Try to consume ``n`` tokens at time ``now_s``; return whether allowed."""
        self._refill(now_s)
        if self.tokens >= n:
            self.tokens -= n
            return True
        self.rejected += 1
        return False

    def available(self, now_s: float) -> float:
        self._refill(now_s)
        return self.tokens
