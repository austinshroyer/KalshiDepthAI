"""15-minute pipeline.

Markets are listed up to two days ahead with their open_time, so the active set
is computed from a schedule. Quotes come from the order book endpoint, which is live and
costs 10 tokens for up to 100 markets, so every market can be read about once per second.
``/markets`` is cached for about 10 seconds and supplies only status, volume and results.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from sqlalchemy import text

from kalshi_quant.shared.collection.classification import MarketClass
from kalshi_quant.shared.collection.pipeline import Pipeline
from kalshi_quant.shared.config import Env
from kalshi_quant.shared.data_models.market import NormalizedMarket
from kalshi_quant.shared.database.writer import read_checkpoint
from kalshi_quant.shared.utilities.timeutil import now_ms

RESOLUTION_WAIT_MS = 30 * 60_000
PRE_OPEN_MS = 2_000


@dataclass(slots=True)
class Scheduled:
    open_ms: int
    close_ms: int
    finished: bool = False


class FifteenMinutePipeline(Pipeline):
    env = Env.FIFTEEN_MINUTE
    market_class = MarketClass.FIFTEEN_MINUTE

    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.schedule: dict[str, Scheduled] = {}
        self.quotes_only_from_books = True
        self._load_schedule()

    def _load_schedule(self) -> None:
        """After a restart, keep polling markets that closed while we were down."""
        with self.db.connect() as conn:
            rows = conn.execute(text("""
                SELECT ticker, open_ms, close_ms FROM markets
                WHERE close_ms > :since AND status NOT IN ('finalized') AND open_ms IS NOT NULL"""),
                {"since": now_ms() - RESOLUTION_WAIT_MS}).all()
        self.schedule = {r[0]: Scheduled(r[1], r[2]) for r in rows}

    def tasks(self) -> list[tuple[str, Callable[[], Awaitable[None]], float]]:
        c = self.cc
        jobs: list[tuple[str, Callable[[], Awaitable[None]], float]] = [
            ("series", self.refresh_series, c.series_refresh_seconds),
            ("discovery", self.discover, c.discovery_seconds),
            ("active_poll", self.poll_active, c.active_poll_seconds),
            ("orderbooks", self.book_poll, c.orderbook_seconds),
            ("history_walk", self.history_walk, 5),
            ("stubs", self.resolve_stubs, 60),
            ("rollups", self.housekeeping, self.config.maintenance.rollup_seconds),
            ("maintenance", self.maintenance, self.config.maintenance.growth_log_seconds),
        ]
        if c.collect_trades:
            jobs.append(("trades", self.poll_trades, c.trades_poll_seconds))
        return jobs

    def _note(self, markets: list[NormalizedMarket]) -> None:
        for market in markets:
            if market.open_ms is None or market.close_ms is None:
                continue
            entry = self.schedule.get(market.ticker)
            if entry is None:
                self.schedule[market.ticker] = entry = Scheduled(market.open_ms, market.close_ms)
            entry.open_ms, entry.close_ms = market.open_ms, market.close_ms  # closes can move earlier
            entry.finished = market.status == "finalized"

    async def discover(self) -> None:
        for series in sorted(self.classifier.fifteen_series):
            if self.stop.is_set():
                return
            if self.owns(self.classifier.classify(series)):
                async for page, _ in self.client.events(series_ticker=series, status="open"):
                    self.store_events(page)
                for status in ("open", "unopened"):
                    async for page, _ in self.client.markets(series_ticker=series, status=status):
                        self._note(await self.ingest_markets(page))

    def active_tickers(self) -> list[str]:
        now = now_ms()
        return [t for t, s in self.schedule.items()
                if not s.finished and s.open_ms - PRE_OPEN_MS <= now <= s.close_ms + RESOLUTION_WAIT_MS]

    def tradable_tickers(self) -> set[str]:
        now = now_ms()
        return {t for t, s in self.schedule.items() if s.open_ms - PRE_OPEN_MS <= now <= s.close_ms}

    async def poll_active(self) -> None:
        tickers = self.active_tickers()
        if not tickers:
            return
        self._note(await self.ingest_markets(await self.client.markets_by_ticker(tickers)))
        finished_before = now_ms() - 2 * RESOLUTION_WAIT_MS
        self.schedule = {t: s for t, s in self.schedule.items() if s.close_ms > finished_before}

    async def book_poll(self) -> None:
        tickers = self.tradable_tickers()
        if self.stream:
            await self.stream.set_book_tickers(tickers)
            if self.ws_healthy():
                return
        await self.poll_books(sorted(tickers))

    async def history_walk(self) -> None:
        """Record resolved markets series by series; one page per run, resumable."""
        if self.writer.paused_tables:  # the disk guard is engaged: history can wait
            return
        position = read_checkpoint(self.db, "history_walk", {"index": 0, "historical": False, "cursor": None})
        series_list = sorted(s for s in self.classifier.fifteen_series if self.owns(self.classifier.classify(s)))
        if position["index"] >= len(series_list):
            return
        series = series_list[position["index"]]
        if position["historical"]:
            pages = self.client.historical_markets(cursor=position["cursor"], series_ticker=series)
        else:
            pages = self.client.markets(cursor=position["cursor"], series_ticker=series, status="settled")
        async for page, next_cursor in pages:
            await self.ingest_markets(page, observe=False, keep_raw=False)
            if next_cursor:
                position["cursor"] = next_cursor
            elif position["historical"]:
                position = {"index": position["index"] + 1, "historical": False, "cursor": None}
            else:
                position = {"index": position["index"], "historical": True, "cursor": None}
            self.writer.checkpoint("history_walk", position)
            break
