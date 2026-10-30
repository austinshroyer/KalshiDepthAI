"""Machinery common to both collection pipelines.

Each pipeline owns its client budget, database, writer and websocket. Nothing
here is shared between running pipelines, so one can run without the other.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from sqlalchemy import select, text

from kalshi_quant.shared.api.auth import Credentials
from kalshi_quant.shared.api.client import KalshiApiError, KalshiClient
from kalshi_quant.shared.api.rate_limit import AdaptiveRateLimiter, background_work
from kalshi_quant.shared.api.websocket import KalshiStream
from kalshi_quant.shared.collection.classification import Classification, Classifier, MarketClass
from kalshi_quant.shared.collection.observations import ObservationTracker
from kalshi_quant.shared.collection.registry import MarketRegistry, compress_raw
from kalshi_quant.shared.collection.rollups import run_rollups
from kalshi_quant.shared.config import PROJECT_ROOT, Env, EnvConfig
from kalshi_quant.shared.data_models.market import (
    SOURCE_REST_BOOK, SOURCE_REST_MARKET, SOURCE_WS_BOOK, SOURCE_WS_TICKER,
    BookSnapshot, NormalizedMarket, Quote, normalize_market, normalize_orderbook,
    normalize_quote, series_from_event_ticker, to_e2, to_e4,
)
from kalshi_quant.shared.database.engine import Database
from kalshi_quant.shared.database.maintenance import log_growth, run_archive
from kalshi_quant.shared.database.writer import BatchWriter, read_checkpoint
from kalshi_quant.shared.utilities.disk import free_gb
from kalshi_quant.shared.utilities.logging import get_logger
from kalshi_quant.shared.utilities.timeutil import now_ms, parse_ts_ms

log = get_logger(__name__)
HIGH_VOLUME_TABLES = {"observations", "trades", "orderbook_snapshots"}
TRADE_OVERLAP_MS = 5_000  # re-read this much of the feed; duplicates are ignored by trade_id
WS_QUIET_MS = 15_000
MAX_TRADE_CATCHUP_MS = 30 * 60_000
SHUTDOWN_GRACE_SECONDS = 5


class Pipeline(ABC):
    env: Env
    market_class: MarketClass
    background_jobs = frozenset({"series", "discovery", "full_sweep", "metadata_sweep", "settlement_sweep",
                                 "event_sweep", "archive_walk", "history_walk", "stubs"})
    delayed_jobs = frozenset({"series"})  # run() already did the first refresh

    def __init__(self, config: EnvConfig, db: Database | None = None, client: KalshiClient | None = None,
                 credentials: Credentials | None = None):
        self.config = config
        self.cc = config.collection
        self.credentials = credentials if credentials is not None else Credentials.from_environment()
        self.db = db or Database.for_config(config)
        limiter = AdaptiveRateLimiter(config.tokens_per_second)
        self.client = client or KalshiClient(config.shared.api, limiter, self.credentials)
        self.classifier = Classifier(self.cc.include_multivariate)
        self.registry = MarketRegistry(self.db)
        self.tracker = ObservationTracker(self.db, self.cc.observation_heartbeat_seconds)
        self.writer = BatchWriter(self.db, self.cc.flush_seconds, self.cc.flush_rows)
        self.stop = asyncio.Event()
        self.stream: KalshiStream | None = None
        self.stats: dict[str, int] = {"markets_seen": 0, "observations": 0, "trades": 0, "books": 0}
        self.event_cadence: dict[str, str | None] = {}
        self.book_hashes: dict[int, int] = {}
        self.last_ws_ms = 0
        self.unknown_series_checked: set[str] = set()
        self.job_seconds: dict[str, float] = {}
        # /markets is cached for about 10 seconds; the order book endpoints are live. Tickers
        # here take their quotes from books, and /markets supplies only slower fields.
        self.book_quoted: set[str] = set()
        self.quotes_only_from_books = False
        self.market_cache: dict[str, NormalizedMarket] = {}
        self.job_runs: dict[str, int] = {}
        self.name = config.env.value

    # Pipeline-specific behaviour
    @abstractmethod
    def tasks(self) -> list[tuple[str, Callable[[], Awaitable[None]], float]]:
        """(name, coroutine, seconds between runs) for each periodic job."""

    def owns(self, classification: Classification) -> bool:
        return classification.market_class is self.market_class

    def ws_channels(self) -> tuple[str, ...]:
        return ("ticker", "trade", "market_lifecycle_v2")

    # Lifecycle
    async def run(self, duration_seconds: float | None = None) -> None:
        self._record_restart_gap()
        await self.refresh_series(full=True)
        await self.writer.flush()
        jobs = [asyncio.create_task(self.writer.run(self.stop), name=f"{self.name}:writer")]
        for name, job, interval in self.tasks():
            jobs.append(asyncio.create_task(self._periodic(name, job, interval), name=f"{self.name}:{name}"))
        if self.cc.use_websocket and self.credentials:
            self.stream = KalshiStream(self.config.shared.api.ws_url, self.credentials, self.on_ws_message,
                                       self.record_gap_stream, self.ws_channels(), self.cc.ws_orderbook_interval_ms)
            jobs.append(asyncio.create_task(self.stream.run(self.stop), name=f"{self.name}:ws"))
        else:
            log.warning("websocket_disabled", pipeline=self.name,
                        reason="no credentials" if not self.credentials else "disabled in config")
        log.info("pipeline_started", pipeline=self.name, tasks=[j.get_name() for j in jobs])
        if duration_seconds:
            asyncio.get_running_loop().call_later(duration_seconds, self.stop.set)
        try:
            await self.stop.wait()
        finally:
            _, unfinished = await asyncio.wait(jobs, timeout=SHUTDOWN_GRACE_SECONDS)
            for job in unfinished:
                job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            await self.writer.flush()
            self._heartbeat("pipeline", "stopped")
            await self.client.close()
            log.info("pipeline_stopped", pipeline=self.name, **self.stats, written=dict(self.writer.total_written))

    def request_stop(self) -> None:
        self.stop.set()

    async def _periodic(self, name: str, job: Callable[[], Awaitable[None]], interval: float) -> None:
        loop = asyncio.get_running_loop()
        if name in self.background_jobs:
            background_work.set(True)  # a task-local context: other jobs are unaffected
        if name in self.delayed_jobs:
            await self._sleep(interval)
        while not self.stop.is_set():
            started = loop.time()
            try:
                await job()
                await self.writer.maybe_flush()
                self.job_seconds[name] = round(loop.time() - started, 3)
                self.job_runs[name] = self.job_runs.get(name, 0) + 1
                self._heartbeat(name, f"ok {self.job_seconds[name]}s")
            except asyncio.CancelledError:
                raise
            except (KalshiApiError, OSError, ValueError, KeyError, TimeoutError) as exc:
                # One job failing must never stop collection; the next cycle retries.
                log.error("job_failed", pipeline=self.name, job=name, error=repr(exc)[:300])
                self._heartbeat(name, f"error: {exc!r}"[:200])
            await self._sleep(max(0.05, interval - (loop.time() - started)))

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self.stop.wait(), seconds)
        except TimeoutError:
            pass

    def _heartbeat(self, stream: str, detail: str) -> None:
        statement = self.db.upsert("heartbeats")
        statement = statement.on_conflict_do_update(
            index_elements=["stream"], set_={"ts_ms": statement.excluded.ts_ms, "detail": statement.excluded.detail})
        with self.db.begin() as conn:
            conn.execute(statement, [{"stream": f"{self.name}:{stream}", "ts_ms": now_ms(), "detail": detail}])

    def _record_restart_gap(self) -> None:
        """The time since the previous run's last write is a gap in prospective data."""
        table = self.db.t["heartbeats"]
        with self.db.connect() as conn:
            last = conn.execute(select(table.c.ts_ms).order_by(table.c.ts_ms.desc()).limit(1)).scalar()
        if last:
            self.record_gap_stream(last, now_ms(), "collector not running")

    def record_gap_stream(self, start_ms: int, end_ms: int, reason: str) -> None:
        with self.db.begin() as conn:
            conn.execute(self.db.t["collection_gaps"].insert().values(
                stream=self.name, start_ms=start_ms, end_ms=end_ms, reason=reason))

    # Series and events
    async def refresh_series(self, full: bool = False) -> None:
        since = None if full else read_checkpoint(self.db, "series_since")
        started = now_ms() // 1000
        series = await self.client.series_list(min_updated_ts=since)
        self.classifier.update_series(series)
        await asyncio.to_thread(self.registry.upsert_series, self.relevant_series(series))
        self.writer.checkpoint("series_since", started - 60)
        fee_changes = await self.client.series_fee_changes()
        if fee_changes:
            await asyncio.to_thread(self._store_fee_changes, fee_changes)
        log.info("series_refreshed", pipeline=self.name, count=len(series),
                 fifteen=len(self.classifier.fifteen_series))

    def relevant_series(self, series: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [s for s in series if self.owns(self.classifier.classify(s["ticker"]))]

    def _store_fee_changes(self, changes: list[dict[str, Any]]) -> None:
        rows = [{"id": c["id"], "series_ticker": c["series_ticker"], "fee_type": c.get("fee_type"),
                 "fee_multiplier": c.get("fee_multiplier"), "scheduled_ms": parse_ts_ms(c["scheduled_ts"])}
                for c in changes]
        with self.db.begin() as conn:
            conn.execute(self.db.upsert("fee_changes").on_conflict_do_nothing(index_elements=["id"]), rows)

    async def _resolve_unknown_series(self, tickers: set[str]) -> None:
        for ticker in tickers - self.unknown_series_checked:
            self.unknown_series_checked.add(ticker)
            series = await self.client.series(ticker)
            if series:
                self.classifier.update_series([series])
                if self.owns(self.classifier.classify(ticker)):
                    await asyncio.to_thread(self.registry.upsert_series, [series])

    # Market ingestion
    async def ingest_markets(self, payloads: list[dict[str, Any]], ts_ms: int | None = None,
                             observe: bool = True, keep_raw: bool = True) -> list[NormalizedMarket]:
        """Classify, keep only this pipeline's markets, store metadata, quotes and resolutions."""
        recv = now_ms()
        ts_ms = ts_ms or recv
        markets = [normalize_market(p) for p in payloads]
        await self._resolve_unknown_series({m.series_ticker for m in markets
                                            if m.series_ticker and not self.classifier.knows(m.series_ticker)})
        mine: list[tuple[NormalizedMarket, Classification]] = []
        audit: list[dict[str, Any]] = []
        for market in markets:
            classification = self.classifier.classify(
                market.series_ticker, market.market_type, market.mve_collection, market.duration_s,
                self.event_cadence.get(market.event_ticker))
            if not self.owns(classification):
                continue
            mine.append((market, classification))
            if classification.anomaly:
                audit.append({"ticker": market.ticker, "series_ticker": market.series_ticker,
                              "anomaly": classification.anomaly, "ts_ms": recv})
        raw = await asyncio.to_thread(self.registry.upsert_markets, mine)
        if keep_raw:
            self.writer.add("raw_responses", raw)
        self.writer.add("classification_audit", audit)
        rows: list[dict[str, Any]] = []
        for market, _ in mine:
            market_id = self.registry.ids[market.ticker]
            self.market_cache[market.ticker] = market
            from_books = self.quotes_only_from_books or market.ticker in self.book_quoted
            if observe and not from_books and market.status != "initialized":  # listed ahead: nothing to record yet
                row, latest = self.tracker.consider(
                    market_id, ts_ms, recv, market.quote, market.volume, market.open_interest, market.status,
                    SOURCE_REST_MARKET, market.volume_24h)
                if row:
                    rows.append(row)
                    self.writer.set_latest(latest)
            self.record_resolution(market_id, market)
        self.writer.add("observations", rows)
        self.stats["observations"] += len(rows)
        self.stats["markets_seen"] += len(mine)
        return [m for m, _ in mine]

    def record_resolution(self, market_id: int, m: NormalizedMarket) -> None:
        if m.status not in ("determined", "disputed", "amended", "finalized") and not m.result:
            return
        winning = losing = None
        if m.result in ("yes", "no"):
            winning, losing = m.result, "no" if m.result == "yes" else "yes"
        settled = m.settlement_value_e4
        if settled is None and m.result in ("yes", "no"):
            settled = 10_000 if m.result == "yes" else 0
        self.writer.add("resolutions", [{
            "market_id": market_id, "result": m.result, "winning_outcome": winning, "losing_outcome": losing,
            "settlement_value_e4": settled, "settlement_ms": m.settlement_ms, "expiration_value": m.expiration_value,
            "final_status": m.status, "resolved_successfully": bool(m.status == "finalized" and m.result),
            "recorded_ms": now_ms(),
        }])

    # Order books
    def store_book(self, ticker: str, book: BookSnapshot, ts_ms: int, source: int) -> None:
        market_id = self.registry.ids.get(ticker)
        if market_id is None:
            return
        clipped = book.clipped(self.cc.orderbook_levels)
        digest = hash((tuple(clipped.yes), tuple(clipped.no)))
        if self.book_hashes.get(market_id) == digest:
            return
        self.book_hashes[market_id] = digest
        self.writer.add("orderbook_snapshots", [{
            "market_id": market_id, "ts_ms": ts_ms, "source": source, "yes_bid": book.yes_bid, "yes_ask": book.yes_ask,
            "yes_depth": sum(c for _, c in clipped.yes), "no_depth": sum(c for _, c in clipped.no),
            "levels": json.dumps({"y": clipped.yes, "n": clipped.no}, separators=(",", ":")),
        }])
        self.stats["books"] += 1

    def observe_book(self, ticker: str, book: BookSnapshot, ts_ms: int, source: int) -> None:
        """Record a live quote from a book. Volume and status come from the latest /markets read."""
        market_id = self.registry.ids.get(ticker)
        market = self.market_cache.get(ticker)
        if market_id is None or market is None:
            return
        quote = Quote(book.yes_bid, book.yes_ask, market.quote.last, book.yes[0][1] if book.yes else None,
                      book.no[0][1] if book.no else None)
        row, latest = self.tracker.consider(market_id, ts_ms, ts_ms, quote, market.volume, market.open_interest,
                                            market.status, source, market.volume_24h)
        if row:
            self.writer.add("observations", [row])
            self.writer.set_latest(latest)
            self.stats["observations"] += 1

    async def poll_books(self, tickers: list[str], depth_tickers: "set[str] | None" = None) -> None:
        """One request per 100 markets at the default token cost, which makes this the cheap live feed."""
        if not tickers:
            return
        for payload in await self.client.orderbooks(tickers):
            ticker = payload.get("ticker", "")
            book, ts = normalize_orderbook(payload), now_ms()
            self.observe_book(ticker, book, ts, SOURCE_REST_BOOK)
            if depth_tickers is None or ticker in depth_tickers:
                self.store_book(ticker, book, ts, SOURCE_REST_BOOK)

    # Trades
    def _trade_market_id(self, ticker: str) -> int | None:
        """Route a trade to this pipeline. Metadata may arrive later, via a stub row."""
        series = series_from_event_ticker(ticker)
        if not self.classifier.knows(series):
            return None  # never guess: unknown series are resolved during market ingestion
        if not self.owns(self.classifier.classify(series)):
            return None
        if ticker in self.registry.ids:
            return self.registry.ids[ticker]
        event = ticker.rsplit("-", 1)[0] if ticker.count("-") >= 2 else ticker
        return self.registry.ensure_stub(ticker, event)

    def trade_row(self, t: dict[str, Any], source: int) -> dict[str, Any] | None:
        ticker = t.get("ticker") or t.get("market_ticker")
        market_id = self._trade_market_id(ticker)
        if market_id is None:
            return None
        side = t.get("taker_outcome_side") or t.get("taker_side")
        ts = parse_ts_ms(t["created_time"]) if t.get("created_time") else t.get("ts_ms")
        return {"trade_id": uuid.UUID(t["trade_id"]).bytes, "market_id": market_id, "ts_ms": ts,
                "yes_price": to_e4(t["yes_price_dollars"]), "count": to_e2(t["count_fp"]),
                "taker_side": 1 if side == "yes" else 0 if side == "no" else None,
                "is_block": t.get("is_block_trade"), "source": source}

    async def poll_trades(self) -> None:
        """Walk the global trade feed newest-first back to the last committed position.

        The checkpoint moves only after the walk reaches it, so an interrupted walk
        rereads the same span instead of skipping older trades.
        """
        since_ms = read_checkpoint(self.db, "trades_since_ms")
        started = now_ms()
        if since_ms is None:
            since_ms = started - 60_000
        elif started - since_ms > MAX_TRADE_CATCHUP_MS:
            # Walking a long outage would take hours. `backfill` recovers it per market.
            self.record_gap_stream(since_ms, started - MAX_TRADE_CATCHUP_MS, "trade feed outage beyond catch-up window")
            since_ms = started - MAX_TRADE_CATCHUP_MS
        min_ts = max(0, since_ms - TRADE_OVERLAP_MS) // 1000
        async for page, _ in self.client.trades(min_ts=min_ts):
            rows = [row for row in (self.trade_row(t, 1) for t in page) if row]
            self.writer.add("trades", rows)
            self.stats["trades"] += len(rows)
            await self.writer.maybe_flush()
            if self.stop.is_set():
                return
        self.writer.checkpoint("trades_since_ms", started)

    # WebSocket
    async def on_ws_message(self, message: dict[str, Any]) -> None:
        kind, msg = message.get("type"), message.get("msg", {})
        self.last_ws_ms = now_ms()
        if kind == "ticker":
            ticker = msg["market_ticker"]
            market_id = self.registry.ids.get(ticker)
            if market_id is None:
                return
            bid, ask = normalize_quote(to_e4(msg.get("yes_bid_dollars")), to_e4(msg.get("yes_ask_dollars")))
            quote = Quote(bid, ask, to_e4(msg.get("price_dollars")) or None,
                          to_e2(msg.get("yes_bid_size_fp")), to_e2(msg.get("yes_ask_size_fp")))
            previous = self.tracker.state.get(market_id)
            row, latest = self.tracker.consider(
                market_id, msg.get("ts_ms") or now_ms(), now_ms(), quote, to_e2(msg.get("volume_fp")),
                to_e2(msg.get("open_interest_fp")), previous.status if previous else 2, SOURCE_WS_TICKER)
            if row:
                self.writer.add("observations", [row])
                self.writer.set_latest(latest)
                self.stats["observations"] += 1
        elif kind == "trade":
            row = self.trade_row(msg, 2)
            if row:
                self.writer.add("trades", [row])
                self.stats["trades"] += 1
        elif kind in ("orderbook_snapshot", "orderbook_delta"):
            ticker = msg["market_ticker"]
            if self.stream and ticker in self.stream.books:
                book = self.stream.books[ticker].snapshot()
                self.store_book(ticker, book, msg.get("ts_ms") or now_ms(), SOURCE_WS_BOOK)
        elif kind == "market_lifecycle_v2":
            await self.on_lifecycle(msg)
        await self.writer.maybe_flush()

    async def on_lifecycle(self, msg: dict[str, Any]) -> None:
        """Lifecycle events are triggers: REST remains the source of record for the market."""
        ticker = msg.get("market_ticker")
        if ticker and msg.get("event_type") in ("created", "determined", "settled", "activated", "deactivated",
                                                "close_date_updated"):
            payload = await self.client.market(ticker)
            if payload:
                await self.ingest_markets([payload])

    def ws_healthy(self) -> bool:
        return self.stream is not None and self.stream.connected.is_set() and now_ms() - self.last_ws_ms < WS_QUIET_MS

    async def resolve_stubs(self) -> None:
        """Fetch metadata for markets first seen in the trade feed."""
        with self.db.connect() as conn:
            tickers = list(conn.execute(text("SELECT ticker FROM markets WHERE is_stub LIMIT 400")).scalars())
        if tickers:
            await self.ingest_markets(await self.client.markets_by_ticker(tickers), observe=False)

    # Housekeeping
    async def guard_disk(self) -> None:
        directory = self.db.path.parent if self.db.path else PROJECT_ROOT
        free = free_gb(Path(directory))
        limit = self.config.shared.storage.min_free_disk_gb
        if free < limit and not self.writer.paused_tables:
            log.error("disk_low_pausing_writes", free_gb=round(free, 2), limit_gb=limit)
            self.writer.paused_tables = set(HIGH_VOLUME_TABLES)
            self.record_gap_stream(now_ms(), now_ms(), f"disk guard: {free:.1f} GB free")
        elif free >= limit * 1.2 and self.writer.paused_tables:
            log.warning("disk_recovered_resuming_writes", free_gb=round(free, 2))
            self.writer.paused_tables = set()

    async def housekeeping(self) -> None:
        await self.writer.flush()
        await asyncio.to_thread(run_rollups, self.db, self.env, self.cc.interval_minutes)

    async def maintenance(self) -> None:
        await self.guard_disk()
        await asyncio.to_thread(log_growth, self.db)
        await asyncio.to_thread(run_archive, self.db, self.config)

    def store_events(self, events: list[dict[str, Any]]) -> None:
        for e in events:
            self.event_cadence[e["event_ticker"]] = (e.get("product_metadata") or {}).get("cadence")
        ts = now_ms()
        self.registry.upsert_events(events)
        self.writer.add("raw_responses", [compress_raw("event", e["event_ticker"], e, ts) for e in events])
