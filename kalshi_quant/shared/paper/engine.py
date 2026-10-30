"""Paper trading against live Kalshi data. No code path can place a real order: the
client is GET-only and fills come from ``simulate_fill``.

Quotes come from the live order book endpoint. ``/markets`` is cached for about 10 seconds,
so it only supplies status, volume and settlement: filling against it would mean filling
against prices that no longer exist.

Each strategy gets a session with its own virtual portfolio. Every fill, closed trade and
portfolio state change is committed straight away, so a restart resumes where it stopped.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select, update

from kalshi_quant.shared.api.auth import Credentials
from kalshi_quant.shared.api.client import KalshiClient
from kalshi_quant.shared.api.rate_limit import AdaptiveRateLimiter
from kalshi_quant.shared.api.websocket import KalshiStream
from kalshi_quant.shared.collection.classification import Classifier, MarketClass
from kalshi_quant.shared.config import Env, EnvConfig
from kalshi_quant.shared.data_models.market import (
    STATUS_CODES, BookSnapshot, NormalizedMarket, normalize_market, normalize_orderbook, normalize_quote,
    series_from_event_ticker, to_e2, to_e4,
)
from kalshi_quant.shared.data_models.tick import MarketInfo, Tick
from kalshi_quant.shared.database.engine import Database
from kalshi_quant.shared.execution.fees import FeeSchedule
from kalshi_quant.shared.execution.fills import Fill
from kalshi_quant.shared.execution.portfolio import ClosedTrade, Portfolio
from kalshi_quant.shared.strategy.base import Strategy
from kalshi_quant.shared.strategy.runner import StrategyRunner
from kalshi_quant.shared.utilities.logging import get_logger
from kalshi_quant.shared.utilities.timeutil import now_ms

log = get_logger(__name__)
ENV_CLASS = {Env.GENERAL: MarketClass.GENERAL, Env.FIFTEEN_MINUTE: MarketClass.FIFTEEN_MINUTE}


def market_info(m: NormalizedMarket, category: str | None, multiplier: float, fee_type: str | None) -> MarketInfo:
    ranges = tuple((float(r["start"]), float(r["end"]), float(r["step"])) for r in json.loads(m.price_ranges or "[]"))
    return MarketInfo(m.ticker, m.event_ticker, m.series_ticker, category, m.title, None, m.open_ms, m.close_ms,
                      m.market_type, m.floor_strike, m.cap_strike, fee_type, multiplier, ranges)


def book_levels(side: list[tuple[int, int]]) -> tuple[tuple[float, float], ...]:
    return tuple((p / 10_000, s / 100) for p, s in side)


def tick_from_book(m: NormalizedMarket, book: BookSnapshot, ts_ms: int) -> Tick:
    bid, ask = book.yes_bid, book.yes_ask
    return Tick(m.ticker, ts_ms, None if bid is None else bid / 10_000, None if ask is None else ask / 10_000,
                None if m.quote.last is None else m.quote.last / 10_000,
                book.yes[0][1] / 100 if book.yes else None, book.no[0][1] / 100 if book.no else None,
                None if m.volume is None else m.volume / 100, None,
                None if m.open_interest is None else m.open_interest / 100, STATUS_CODES.get(m.status, 0),
                book_levels(book.yes), book_levels(book.no))


@dataclass
class Session:
    session_id: int
    strategy: Strategy
    runner: StrategyRunner
    last_equity_ms: int = 0
    stopped: bool = field(default=False)


class PaperEngine:
    def __init__(self, config: EnvConfig, strategies: list[tuple[Strategy, Path]], capital: float | None = None,
                 db: Database | None = None, client: KalshiClient | None = None,
                 credentials: Credentials | None = None, resume: bool = False):
        self.config = config
        self.env = config.env
        self.pc = config.paper
        self.db = db or Database.for_config(config)
        self.credentials = credentials if credentials is not None else Credentials.from_environment()
        self.client = client or KalshiClient(config.shared.api, AdaptiveRateLimiter(self.pc.tokens_per_second), self.credentials or None)
        self.classifier = Classifier()
        self.fees = FeeSchedule(rounding=config.execution.fee_rounding_dollars)
        self.stop = asyncio.Event()
        self.tracked: dict[str, MarketInfo] = {}
        self.books: dict[str, BookSnapshot] = {}
        self.market_state: dict[str, NormalizedMarket] = {}
        self.last_ts: dict[str, int] = {}
        self.sessions = [self._open_session(s, p, capital or config.backtest.starting_capital, resume) for s, p in strategies]

    # Sessions and persistence
    def _open_session(self, strategy: Strategy, path: Path, capital: float, resume: bool) -> Session:
        t = self.db.t["paper_sessions"]
        latency = self.config.backtest.latency_ms
        previous = None
        if resume:
            with self.db.connect() as conn:
                previous = conn.execute(select(t).where(t.c.strategy_name == strategy.name)
                                        .order_by(t.c.session_id.desc()).limit(1)).mappings().first()
        state = None
        if previous:
            session_id = previous["session_id"]
            s = self.db.t["paper_state"]
            with self.db.begin() as conn:
                row = conn.execute(select(s.c.state).where(s.c.session_id == session_id)).first()
                state = json.loads(row[0]) if row else None
                conn.execute(update(t).where(t.c.session_id == session_id).values(status="running", pid=os.getpid()))
        else:
            with self.db.begin() as conn:
                session_id = conn.execute(t.insert().values(
                    strategy_name=strategy.name, strategy_path=str(path), strategy_hash=strategy.source_hash(),
                    params=json.dumps(strategy.to_spec(), default=str), starting_capital=capital, status="running",
                    pid=os.getpid(), started_ms=now_ms(), heartbeat_ms=now_ms())).inserted_primary_key[0]
        portfolio = Portfolio.from_state(state["portfolio"]) if state else Portfolio(capital)
        session = Session(session_id, strategy, None)  # type: ignore[arg-type]
        session.runner = StrategyRunner(
            strategy, portfolio, self.config.execution, self.fees, latency,
            int(self.config.backtest.max_quote_age_seconds * 1000),
            on_fill=lambda fill: self._record_fill(session, fill), on_close=lambda trade: self._record_close(session, trade))
        if state:
            session.runner.entries = state["entries"]
        log.info("paper_session", session=session_id, strategy=strategy.name, resumed=bool(state), cash=round(portfolio.cash, 2))
        return session

    def _record_fill(self, session: Session, fill: Fill) -> None:
        order = fill.order
        with self.db.begin() as conn:
            conn.execute(self.db.t["paper_orders"].insert().values(
                session_id=session.session_id, ticker=order.ticker, side=order.side, action=order.action,
                requested=order.quantity, filled=fill.filled, avg_price=fill.avg_price or None, fee=fill.fee,
                slippage=fill.slippage, status=fill.status, reason=fill.reason or order.reason,
                decision_ms=order.decision_ms, execution_ms=fill.ts_ms))
        self._save_state(session)
        log.info("paper_fill", session=session.session_id, ticker=order.ticker, side=order.side, action=order.action,
                 filled=fill.filled, price=round(fill.avg_price, 4), status=fill.status, reason=fill.reason)

    def _record_close(self, session: Session, trade: ClosedTrade) -> None:
        with self.db.begin() as conn:
            conn.execute(self.db.t["paper_trades"].insert().values(
                session_id=session.session_id, ticker=trade.ticker, side=trade.side, quantity=trade.quantity,
                entry_price=trade.entry_price, exit_price=trade.exit_price, entry_ms=trade.entry_ms,
                exit_ms=trade.exit_ms, pnl=trade.pnl, fees=trade.fees, exit_reason=trade.exit_reason))
        self._save_state(session)

    def _save_state(self, session: Session) -> None:
        state = {"portfolio": session.runner.portfolio.to_state(), "entries": session.runner.entries}
        statement = self.db.upsert("paper_state")
        statement = statement.on_conflict_do_update(
            index_elements=["session_id"], set_={"state": statement.excluded.state, "updated_ms": statement.excluded.updated_ms})
        with self.db.begin() as conn:
            conn.execute(statement, [{"session_id": session.session_id, "state": json.dumps(state), "updated_ms": now_ms()}])

    def _housekeeping(self) -> None:
        """Heartbeats, equity samples and stop requests from `paper stop`."""
        ts = now_ms()
        t = self.db.t["paper_sessions"]
        with self.db.begin() as conn:
            for session in self.sessions:
                status = conn.execute(select(t.c.status).where(t.c.session_id == session.session_id)).scalar()
                if status == "stop_requested":
                    session.stopped = True
                conn.execute(update(t).where(t.c.session_id == session.session_id).values(heartbeat_ms=ts))
                if ts - session.last_equity_ms >= self.pc.equity_log_seconds * 1000:
                    p = session.runner.portfolio
                    session.last_equity_ms = ts
                    conn.execute(self.db.t["paper_equity"].insert().values(
                        session_id=session.session_id, ts_ms=ts, cash=p.cash, market_value=p.market_value(),
                        realized_pnl=p.realized_pnl, unrealized_pnl=p.unrealized_pnl()))
        if all(s.stopped for s in self.sessions):
            self.stop.set()

    def _close_sessions(self) -> None:
        t = self.db.t["paper_sessions"]
        with self.db.begin() as conn:
            conn.execute(update(t).where(t.c.session_id.in_([s.session_id for s in self.sessions]))
                         .values(status="stopped", stopped_ms=now_ms()))
        for session in self.sessions:
            self._save_state(session)

    # Market data
    def _owned(self, m: NormalizedMarket) -> bool:
        return self.classifier.classify(m.series_ticker, m.market_type, m.mve_collection, m.duration_s).market_class \
            is ENV_CLASS[self.env]

    def _series_meta(self, series_ticker: str | None) -> tuple[str | None, float, str | None]:
        s = self.classifier.series.get(series_ticker or "") or {}
        return s.get("category"), s.get("fee_multiplier", 1.0), s.get("fee_type")

    def _target_series(self) -> list[str] | None:
        """The series worth listing, or None when only a full sweep can answer."""
        names: set[str] = set()
        for session in self.sessions:
            universe = session.strategy.universe
            if universe.series:
                names |= set(universe.series)
            elif universe.categories:
                names |= {t for t, s in self.classifier.series.items() if s.get("category") in universe.categories}
            elif self.env is Env.FIFTEEN_MINUTE:
                names |= set(self.classifier.fifteen_series)
            else:
                return None
        if self.env is Env.FIFTEEN_MINUTE:
            names &= self.classifier.fifteen_series
        return sorted(names)

    async def discover(self) -> None:
        self.classifier.update_series(await self.client.series_list())
        found: list[NormalizedMarket] = []
        targets = self._target_series()
        if targets is None or len(targets) > 200:
            async for page, _ in self.client.markets(status="open", mve_filter="exclude"):
                found += [normalize_market(p) for p in page]
        else:
            for series in targets:
                async for page, _ in self.client.markets(status="open", series_ticker=series):
                    found += [normalize_market(p) for p in page]
        owned = [m for m in found if self._owned(m)]
        owned.sort(key=lambda m: m.volume_24h or 0, reverse=True)
        chosen: dict[str, MarketInfo] = {}
        for market in owned:
            info = market_info(market, *self._series_meta(market.series_ticker))
            wanted = False
            for session in self.sessions:
                session.runner.register(info)
                wanted |= session.runner._is_selected(info.ticker)
            if wanted and len(chosen) < self.pc.max_tracked_markets:
                chosen[info.ticker] = info
        held = {t for s in self.sessions for t in s.runner.portfolio.positions}
        self.tracked = {**{t: i for t, i in self.tracked.items() if t in held}, **chosen}
        log.info("paper_discovery", listed=len(found), owned=len(owned), tracked=len(self.tracked))

    def dispatch(self, tick: Tick) -> None:
        if tick.ts_ms <= self.last_ts.get(tick.ticker, 0):
            return
        self.last_ts[tick.ticker] = tick.ts_ms
        for session in self.sessions:
            if not session.stopped:
                session.runner.on_tick(tick)

    async def poll_markets(self) -> None:
        """Status, volume and settlement. Slow fields only: this endpoint is cached."""
        ts = now_ms()
        for payload in await self.client.markets_by_ticker(list(self.tracked)):
            market = normalize_market(payload)
            self.market_state[market.ticker] = market
            if market.status == "finalized" and market.result:
                value = market.settlement_value_e4
                if value is None:
                    value = 10_000 if market.result == "yes" else 0
                for session in self.sessions:
                    if market.ticker in session.runner.infos:
                        session.runner.settle(market.ticker, value / 10_000, ts)
                self.tracked.pop(market.ticker, None)

    async def poll_books(self) -> None:
        """The live feed: each book read becomes a tick for every strategy."""
        held = [t for s in self.sessions for t in (*s.runner.portfolio.positions, *s.runner.pending)]
        tickers = list(dict.fromkeys([*held, *self.tracked]))[: self.pc.max_book_markets]
        if not tickers:
            return
        for payload in await self.client.orderbooks(tickers):
            ticker = payload["ticker"]
            book = self.books[ticker] = normalize_orderbook(payload)
            market = self.market_state.get(ticker)
            if market is not None:
                self.dispatch(tick_from_book(market, book, now_ms()))

    async def on_stream(self, message: dict[str, Any]) -> None:
        if message.get("type") != "ticker":
            return
        msg = message["msg"]
        ticker = msg["market_ticker"]
        if ticker not in self.tracked:
            return
        bid, ask = normalize_quote(to_e4(msg.get("yes_bid_dollars")), to_e4(msg.get("yes_ask_dollars")))
        book = self.books.get(ticker)
        volume, bid_size, ask_size = to_e2(msg.get("volume_fp")), to_e2(msg.get("yes_bid_size_fp")), to_e2(msg.get("yes_ask_size_fp"))
        self.dispatch(Tick(ticker, msg.get("ts_ms") or now_ms(), None if bid is None else bid / 10_000,
                           None if ask is None else ask / 10_000, (to_e4(msg.get("price_dollars")) or 0) / 10_000 or None,
                           None if bid_size is None else bid_size / 100, None if ask_size is None else ask_size / 100,
                           None if volume is None else volume / 100, None, None, STATUS_CODES["active"],
                           book_levels(book.yes) if book else None, book_levels(book.no) if book else None))

    async def _loop(self, name: str, job: Any, interval: float) -> None:
        while not self.stop.is_set():
            try:
                await job()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a bad cycle must not end a long session
                log.error("paper_job_failed", job=name, error=repr(exc)[:300])
            try:
                await asyncio.wait_for(self.stop.wait(), interval)
            except TimeoutError:
                pass

    async def _housekeep_job(self) -> None:
        await asyncio.to_thread(self._housekeeping)

    async def run(self, duration_seconds: float | None = None) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.stop.set)
            except (NotImplementedError, RuntimeError):
                pass
        if duration_seconds:
            loop.call_later(duration_seconds, self.stop.set)
        await self.discover()
        await self.poll_markets()
        jobs = [("discover", self.discover, self.pc.discovery_seconds), ("markets", self.poll_markets, self.pc.market_seconds),
                ("housekeeping", self._housekeep_job, 2)]
        tasks = [asyncio.create_task(self._loop(n, j, i)) for n, j, i in jobs]
        if self.pc.use_websocket and self.credentials:
            stream = KalshiStream(self.config.shared.api.ws_url, self.credentials, self.on_stream,
                                  lambda *_: None, ("ticker",))
            tasks.append(asyncio.create_task(stream.run(self.stop)))
        tasks.append(asyncio.create_task(self._loop("books", self.poll_books, self.pc.poll_seconds)))
        await self.stop.wait()
        await asyncio.wait(tasks, timeout=5)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._close_sessions()
        await self.client.close()
