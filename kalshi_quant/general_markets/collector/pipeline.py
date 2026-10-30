"""General-market pipeline: every eligible market except 15-minute and perpetual.

Coverage tiers, sized from the live universe (about 170k open markets):
  full sweep    every open market, slowly (372 MB per pass; quotes up to ~15s old)
  hot sweep     the busiest markets by 24h volume, live quotes from order books
  websocket     every market in real time, when credentials exist
  archive walk  resolved markets, one page at a time, resumable
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from sqlalchemy import text

from kalshi_quant.shared.collection.classification import MarketClass
from kalshi_quant.shared.collection.pipeline import Pipeline, log
from kalshi_quant.shared.config import Env
from kalshi_quant.shared.database.writer import read_checkpoint
from kalshi_quant.shared.utilities.timeutil import now_ms

ARCHIVE_STAGES = ("settled_markets", "historical_markets", "settled_events")


class GeneralPipeline(Pipeline):
    env = Env.GENERAL
    market_class = MarketClass.GENERAL

    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.hot: list[str] = []
        self.mve_params = {} if self.cc.include_multivariate else {"mve_filter": "exclude"}
        self._load_hot()

    def _load_hot(self) -> None:
        with self.db.connect() as conn:
            self.hot = list(conn.execute(text("""
                SELECT m.ticker FROM market_latest l JOIN markets m USING (market_id)
                WHERE l.status = 2 AND l.volume_24h > 0 ORDER BY l.volume_24h DESC LIMIT :n"""),
                {"n": self.cc.hot_market_count}).scalars())
        self.book_quoted = set(self.hot)

    def tasks(self) -> list[tuple[str, Callable[[], Awaitable[None]], float]]:
        c = self.cc
        jobs: list[tuple[str, Callable[[], Awaitable[None]], float]] = [
            ("series", self.refresh_series, c.series_refresh_seconds),
            ("full_sweep", self.full_sweep, c.full_sweep_seconds),
            ("hot_sweep", self.hot_sweep, c.hot_sweep_seconds),
            ("metadata_sweep", self.metadata_sweep, c.metadata_sweep_seconds),
            ("settlement_sweep", self.settlement_sweep, c.settlement_sweep_seconds),
            ("event_sweep", self.event_sweep, c.event_sweep_seconds),
            ("archive_walk", self.archive_walk, c.archive_sweep_seconds),
            ("stubs", self.resolve_stubs, 60),
            ("rollups", self.housekeeping, self.config.maintenance.rollup_seconds),
            ("maintenance", self.maintenance, self.config.maintenance.growth_log_seconds),
        ]
        if c.collect_trades:
            jobs.append(("trades", self.poll_trades, c.trades_poll_seconds))
        return jobs

    async def full_sweep(self) -> None:
        """Resumes mid-way: the cursor is committed with the rows of the page before it."""
        cursor = read_checkpoint(self.db, "full_sweep_cursor")
        volumes: dict[str, int] = {}
        async for page, next_cursor in self.client.markets(cursor=cursor, status="open", **self.mve_params):
            markets = await self.ingest_markets(page)
            volumes.update({m.ticker: m.volume_24h or 0 for m in markets})
            self.writer.checkpoint("full_sweep_cursor", next_cursor)
            await self.writer.maybe_flush()
            if self.stop.is_set():
                return
        ranked = sorted((t for t, v in volumes.items() if v > 0), key=volumes.get, reverse=True)
        if cursor:  # a resumed sweep saw only part of the universe: keep the earlier ranking too
            ranked += [t for t in self.hot if t not in volumes]
        if ranked:
            self.hot = ranked[: self.cc.hot_market_count]
            self.book_quoted = set(self.hot)

    async def hot_sweep(self) -> None:
        depth = set(self.hot[: self.cc.orderbook_market_count])
        await self.poll_books(sorted(depth) if self.ws_healthy() else self.hot, depth)

    async def metadata_sweep(self) -> None:
        since = read_checkpoint(self.db, "metadata_since")
        started = now_ms() // 1000
        if since is None:  # the full sweep covers the first pass
            self.writer.checkpoint("metadata_since", started)
            return
        async for page, _ in self.client.markets(min_updated_ts=since - 60, mve_filter="exclude"):
            await self.ingest_markets(page)
        self.writer.checkpoint("metadata_since", started)

    async def settlement_sweep(self) -> None:
        since = read_checkpoint(self.db, "settled_since", now_ms() // 1000 - 3600)
        started = now_ms() // 1000
        async for page, _ in self.client.markets(status="settled", min_settled_ts=since - 300, **self.mve_params):
            await self.ingest_markets(page)
        self.writer.checkpoint("settled_since", started)
        closed_since = read_checkpoint(self.db, "closed_since", now_ms() // 1000 - 86_400)
        async for page, _ in self.client.markets(status="closed", min_close_ts=closed_since - 300, **self.mve_params):
            await self.ingest_markets(page)
        self.writer.checkpoint("closed_since", started)

    async def event_sweep(self) -> None:
        async for page, _ in self.client.events(status="open"):
            self.store_events(page)
            if self.stop.is_set():
                return

    async def archive_walk(self) -> None:
        """One page per run: walks resolved markets and events, then idles when finished."""
        if self.writer.paused_tables:  # the disk guard is engaged: history can wait
            return
        position = read_checkpoint(self.db, "archive_walk", {"stage": 0, "cursor": None})
        if position["stage"] >= len(ARCHIVE_STAGES):
            return
        stage, cursor = ARCHIVE_STAGES[position["stage"]], position["cursor"]
        if stage == "settled_markets":
            pages = self.client.markets(cursor=cursor, status="settled", **self.mve_params)
        elif stage == "historical_markets":
            pages = self.client.historical_markets(cursor=cursor, **self.mve_params)
        else:
            pages = self.client.events(cursor=cursor, status="settled")
        async for page, next_cursor in pages:
            if stage == "settled_events":
                self.store_events(page)
            else:
                await self.ingest_markets(page, observe=False, keep_raw=False)
            if next_cursor:
                self.writer.checkpoint("archive_walk", {"stage": position["stage"], "cursor": next_cursor})
            else:
                log.info("archive_stage_done", stage=stage)
                self.writer.checkpoint("archive_walk", {"stage": position["stage"] + 1, "cursor": None})
            break
