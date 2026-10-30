"""Normalization from Kalshi API payloads to platform records.

Stored prices are integers in 1/10,000 dollar (``_e4``) and contract counts in
1/100 contract (``_e2``), because the API sends fixed-point strings with those
precisions. Integers are exact and much smaller on disk than REAL. Views and the
Python accessors expose decimal probabilities in [0, 1].
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from kalshi_quant.shared.utilities.timeutil import parse_ts_ms

STATUS_CODES = {
    "unknown": 0, "initialized": 1, "active": 2, "inactive": 3, "closed": 4,
    "determined": 5, "disputed": 6, "amended": 7, "finalized": 8,
}
STATUS_NAMES = {v: k for k, v in STATUS_CODES.items()}
SOURCE_REST_MARKET, SOURCE_WS_TICKER, SOURCE_REST_BOOK, SOURCE_WS_BOOK = 1, 2, 3, 4
SOURCE_NAMES = {1: "rest_market", 2: "ws_ticker", 3: "rest_orderbook", 4: "ws_orderbook"}


def to_e4(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(round(float(value) * 10_000))


def to_e2(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(round(float(value) * 100))


def normalize_quote(bid_e4: int | None, ask_e4: int | None) -> tuple[int | None, int | None]:
    """An empty side arrives as 0 or 1.0000. Only prices strictly inside (0, 1) are quotes."""
    bid = bid_e4 if bid_e4 and 0 < bid_e4 < 10_000 else None
    ask = ask_e4 if ask_e4 and 0 < ask_e4 < 10_000 else None
    return bid, ask


def implied_probability_e4(bid: int | None, ask: int | None, last: int | None) -> int | None:
    """Mid of a two-sided quote, else the last trade, else unknown."""
    if bid is not None and ask is not None:
        return (bid + ask) // 2
    return last if last else None


@dataclass(slots=True)
class Quote:
    yes_bid: int | None
    yes_ask: int | None
    last: int | None = None
    yes_bid_size: int | None = None
    yes_ask_size: int | None = None

    @property
    def no_bid(self) -> int | None:
        return None if self.yes_ask is None else 10_000 - self.yes_ask

    @property
    def no_ask(self) -> int | None:
        return None if self.yes_bid is None else 10_000 - self.yes_bid

    @property
    def yes_prob(self) -> int | None:
        return implied_probability_e4(self.yes_bid, self.yes_ask, self.last)


@dataclass(slots=True)
class NormalizedMarket:
    ticker: str
    event_ticker: str
    series_ticker: str | None
    title: str | None
    subtitle: str | None
    yes_sub_title: str | None
    no_sub_title: str | None
    rules_primary: str | None
    rules_secondary: str | None
    market_type: str | None
    status: str
    created_ms: int | None
    open_ms: int | None
    close_ms: int | None
    expected_expiration_ms: int | None
    latest_expiration_ms: int | None
    expiration_ms: int | None
    updated_ms: int | None
    can_close_early: bool | None
    strike_type: str | None
    floor_strike: float | None
    cap_strike: float | None
    price_ranges: str | None
    notional_e4: int | None
    exchange_index: int | None
    mve_collection: str | None
    quote: Quote
    volume: int | None
    volume_24h: int | None
    open_interest: int | None
    result: str | None
    settlement_value_e4: int | None
    settlement_ms: int | None
    expiration_value: str | None
    raw: dict[str, Any]

    @property
    def duration_s(self) -> int | None:
        if self.open_ms is None or self.close_ms is None:
            return None
        return (self.close_ms - self.open_ms) // 1000


def series_from_event_ticker(event_ticker: str | None) -> str | None:
    return event_ticker.split("-", 1)[0] if event_ticker else None


def normalize_market(m: dict[str, Any]) -> NormalizedMarket:
    bid, ask = normalize_quote(to_e4(m.get("yes_bid_dollars")), to_e4(m.get("yes_ask_dollars")))
    event_ticker = m.get("event_ticker") or ""
    ranges = m.get("price_ranges")
    return NormalizedMarket(
        ticker=m["ticker"],
        event_ticker=event_ticker,
        series_ticker=m.get("series_ticker") or series_from_event_ticker(event_ticker),
        title=m.get("title"),
        subtitle=m.get("subtitle"),
        yes_sub_title=m.get("yes_sub_title"),
        no_sub_title=m.get("no_sub_title"),
        rules_primary=m.get("rules_primary"),
        rules_secondary=m.get("rules_secondary"),
        market_type=m.get("market_type"),
        status=m.get("status") or "unknown",
        created_ms=parse_ts_ms(m.get("created_time")),
        open_ms=parse_ts_ms(m.get("open_time")),
        close_ms=parse_ts_ms(m.get("close_time")),
        expected_expiration_ms=parse_ts_ms(m.get("expected_expiration_time")),
        latest_expiration_ms=parse_ts_ms(m.get("latest_expiration_time")),
        expiration_ms=parse_ts_ms(m.get("expiration_time")),
        updated_ms=parse_ts_ms(m.get("updated_time")),
        can_close_early=m.get("can_close_early"),
        strike_type=m.get("strike_type"),
        floor_strike=m.get("floor_strike"),
        cap_strike=m.get("cap_strike"),
        price_ranges=json.dumps(ranges, separators=(",", ":")) if ranges else None,
        notional_e4=to_e4(m.get("notional_value_dollars")),
        exchange_index=m.get("exchange_index"),
        mve_collection=m.get("mve_collection_ticker"),
        quote=Quote(bid, ask, to_e4(m.get("last_price_dollars")) or None,
                    to_e2(m.get("yes_bid_size_fp")), to_e2(m.get("yes_ask_size_fp"))),
        volume=to_e2(m.get("volume_fp")),
        volume_24h=to_e2(m.get("volume_24h_fp")),
        open_interest=to_e2(m.get("open_interest_fp")),
        result=m.get("result") or None,
        settlement_value_e4=to_e4(m.get("settlement_value_dollars")),
        settlement_ms=parse_ts_ms(m.get("settlement_ts")),
        expiration_value=m.get("expiration_value") or None,
        raw=m,
    )


@dataclass(slots=True)
class BookSnapshot:
    """Resting bids per side, best first. A YES ask is a NO bid at (1 - price)."""
    yes: list[tuple[int, int]]
    no: list[tuple[int, int]]

    @property
    def yes_bid(self) -> int | None:
        return self.yes[0][0] if self.yes else None

    @property
    def yes_ask(self) -> int | None:
        return 10_000 - self.no[0][0] if self.no else None

    def clipped(self, levels: int) -> "BookSnapshot":
        return BookSnapshot(self.yes[:levels], self.no[:levels])


def normalize_orderbook(payload: dict[str, Any]) -> BookSnapshot:
    """Parse ``orderbook_fp``. Levels arrive ascending by price, so the best bid is last."""
    book = payload.get("orderbook_fp") or payload
    side = lambda key: sorted(((to_e4(p), to_e2(c)) for p, c in book.get(key) or []), reverse=True)  # noqa: E731
    return BookSnapshot(side("yes_dollars"), side("no_dollars"))
