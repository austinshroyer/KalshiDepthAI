"""Adaptive token bucket, denominated in Kalshi's tokens.

Kalshi 429s carry no Retry-After header and no penalty, so the limiter cuts its
rate on a 429 and creeps back to the configured rate on success.

Default cost is 10 tokens. The public docs do not list the heavier read endpoints
(the authoritative GET /account/endpoint_costs needs a key), so ENDPOINT_COSTS was
measured on 2026-10-30 at a Basic budget of 200 tokens/s: these paths sustained
about 5 requests/s while default-cost paths sustained 10+ without a 429.
"""
from __future__ import annotations

import asyncio
import contextvars
import time

# Set for slow sweeps, so they can never take the budget of the fast polls.
background_work: contextvars.ContextVar[bool] = contextvars.ContextVar("background_work", default=False)


DEFAULT_COST = 10
ENDPOINT_COSTS = {"markets": 40, "markets/trades": 40, "events": 40, "markets/candlesticks": 40,
                  "historical/markets": 40}


def endpoint_cost(path: str) -> int:
    return ENDPOINT_COSTS.get(path.strip("/"), DEFAULT_COST)


class AdaptiveRateLimiter:
    def __init__(self, tokens_per_second: float, burst_seconds: float = 2.0, floor_fraction: float = 0.25,
                 background_share: float = 0.5):
        self.max_rate = tokens_per_second
        self.rate = tokens_per_second
        self.floor = tokens_per_second * floor_fraction
        self.burst_seconds = burst_seconds
        self._tokens = tokens_per_second * burst_seconds
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()
        self.throttled = 0
        self.background_share = background_share
        self._background_lock = asyncio.Lock()
        self._next_background = 0.0

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self.rate * self.burst_seconds, self._tokens + (now - self._updated) * self.rate)
        self._updated = now

    async def acquire(self, cost: int = DEFAULT_COST) -> None:
        if background_work.get():
            async with self._background_lock:  # paces sweeps to their share of the current rate
                wait = self._next_background - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                self._next_background = time.monotonic() + cost / (self.rate * self.background_share)
        async with self._lock:  # FIFO: waiters queue behind the current holder
            self._refill()
            if self._tokens < cost:
                await asyncio.sleep((cost - self._tokens) / self.rate)
                self._refill()
            self._tokens -= cost

    def on_throttled(self) -> None:
        self.throttled += 1
        self.rate = max(self.floor, self.rate * 0.7)

    def on_success(self) -> None:
        self.rate = min(self.max_rate, self.rate * 1.05)
