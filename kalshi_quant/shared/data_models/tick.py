"""The event type every engine consumes, live or historical."""
from __future__ import annotations

from dataclasses import dataclass

ACTIVE = 2


@dataclass(slots=True)
class Tick:
    ticker: str
    ts_ms: int
    yes_bid: float | None
    yes_ask: float | None
    last: float | None = None
    yes_bid_size: float | None = None
    yes_ask_size: float | None = None
    volume: float | None = None        # cumulative contracts
    volume_delta: float | None = None
    open_interest: float | None = None
    status: int = ACTIVE
    # Displayed depth as resting bids, best first: ((price, size), ...). None when unknown.
    yes_levels: tuple[tuple[float, float], ...] | None = None
    no_levels: tuple[tuple[float, float], ...] | None = None
    is_candle: bool = False

    @property
    def probability(self) -> float | None:
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2
        return self.last


@dataclass(frozen=True, slots=True)
class MarketInfo:
    """What was knowable about a market before trading it. Deliberately no result fields.

    ``scheduled_close_ms`` is the close time first seen, because the exchange rewrites
    close_time when a market ends early.
    """
    ticker: str
    event_ticker: str | None
    series_ticker: str | None
    category: str | None
    title: str | None
    event_title: str | None
    open_ms: int | None
    scheduled_close_ms: int | None
    market_type: str | None = "binary"
    floor_strike: float | None = None
    cap_strike: float | None = None
    fee_type: str | None = "quadratic"
    fee_multiplier: float = 1.0
    tick_ranges: tuple[tuple[float, float, float], ...] = ()  # (start, end, step)

    @property
    def duration_seconds(self) -> float | None:
        if self.open_ms is None or self.scheduled_close_ms is None:
            return None
        return (self.scheduled_close_ms - self.open_ms) / 1000
