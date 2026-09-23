"""Shared 40 RPM limiter. All agents must call limiter.wait() before each NIM request."""
import time
import threading
from collections import deque


class RateLimiter:
    def __init__(self, rpm: int = 40):
        self.rpm = rpm
        self.min_interval = 60.0 / rpm  # 1.5s for 40 RPM
        self._lock = threading.Lock()
        self._calls: deque[float] = deque()
        self._last: float = 0.0

    def wait(self) -> float:
        """Block until next request is allowed. Returns seconds waited."""
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                # drop calls older than 60s
                while self._calls and now - self._calls[0] > 60:
                    self._calls.popleft()

                # sliding-window check
                if len(self._calls) >= self.rpm:
                    sleep_for = 60 - (now - self._calls[0]) + 0.05
                else:
                    # min-interval check (spread 40 calls evenly)
                    gap = now - self._last
                    sleep_for = self.min_interval - gap if self._last else 0

                if sleep_for <= 0:
                    self._calls.append(now)
                    self._last = now
                    return waited
                waited += sleep_for
            time.sleep(sleep_for)


# Global singleton: one budget for the whole agency
limiter = RateLimiter(rpm=40)
