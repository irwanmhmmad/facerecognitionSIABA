from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock
from time import monotonic


class FixedWindowRateLimiter:
    """Small in-process limiter suitable for the service's single-worker deployment."""

    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = monotonic()
        cutoff = now - 60.0
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(now)
            return True
