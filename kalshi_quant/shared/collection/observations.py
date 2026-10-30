"""Turns quotes into observation rows.

An unchanged quote is skipped until the heartbeat interval passes, which is what
keeps storage viable. Volume deltas survive restarts because state is reloaded
from ``market_latest``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from kalshi_quant.shared.data_models.market import STATUS_CODES, Quote
from kalshi_quant.shared.database.engine import Database


@dataclass(slots=True)
class LastState:
    ts_ms: int
    yes_bid: int | None
    yes_ask: int | None
    last: int | None
    volume: int | None
    oi: int | None
    status: int
    yes_bid_size: int | None = None
    yes_ask_size: int | None = None


class ObservationTracker:
    def __init__(self, db: Database, heartbeat_seconds: float):
        self.heartbeat_ms = int(heartbeat_seconds * 1000)
        self.state: dict[int, LastState] = {}
        self.volume_regressions = 0
        t = db.t["market_latest"]
        with db.connect() as conn:
            for row in conn.execute(select(t)).mappings():
                self.state[row["market_id"]] = LastState(
                    row["ts_ms"], row["yes_bid"], row["yes_ask"], row["last_price"],
                    row["volume"], row["open_interest"], row["status"] or 0)

    def consider(
        self, market_id: int, ts_ms: int, recv_ms: int, quote: Quote, volume: int | None,
        open_interest: int | None, status: str | int, source: int, volume_24h: int | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Return (observation row or None, market_latest row or None)."""
        code = status if isinstance(status, int) else STATUS_CODES.get(status, 0)
        previous = self.state.get(market_id)
        if previous is not None:
            if ts_ms <= previous.ts_ms and source != 2:
                return None, None  # stale REST response overtaken by a newer read
            unchanged = (
                quote.yes_bid == previous.yes_bid and quote.yes_ask == previous.yes_ask
                and quote.last == previous.last and volume == previous.volume
                and open_interest == previous.oi and code == previous.status
                and previous.yes_bid_size in (None, quote.yes_bid_size)  # None: unknown right after a restart
                and previous.yes_ask_size in (None, quote.yes_ask_size)
            )
            if unchanged and ts_ms - previous.ts_ms < self.heartbeat_ms:
                return None, None
        delta = None
        if previous is not None and previous.volume is not None and volume is not None:
            if volume < previous.volume:  # cached /markets reads can step back; cumulative volume cannot
                self.volume_regressions += 1
                volume = previous.volume
            delta = volume - previous.volume
        self.state[market_id] = LastState(ts_ms, quote.yes_bid, quote.yes_ask, quote.last, volume, open_interest,
                                          code, quote.yes_bid_size, quote.yes_ask_size)
        prob = quote.yes_prob
        row = {
            "market_id": market_id, "ts_ms": ts_ms, "source": source, "recv_ms": recv_ms,
            "yes_bid": quote.yes_bid, "yes_ask": quote.yes_ask, "no_bid": quote.no_bid, "no_ask": quote.no_ask,
            "yes_prob": prob, "last_price": quote.last, "yes_bid_size": quote.yes_bid_size,
            "yes_ask_size": quote.yes_ask_size, "volume": volume, "volume_delta": delta,
            "open_interest": open_interest, "status": code,
        }
        latest = {
            "market_id": market_id, "ts_ms": ts_ms, "yes_bid": quote.yes_bid, "yes_ask": quote.yes_ask,
            "last_price": quote.last, "yes_prob": prob, "volume": volume, "volume_24h": volume_24h,
            "open_interest": open_interest, "status": code,
        }
        return row, latest
