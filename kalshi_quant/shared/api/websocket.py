"""Authenticated Kalshi WebSocket stream with reconnect and order book sequencing.

Kalshi requires a signed handshake even for public channels. The stream is
therefore optional: pipelines run on REST alone when no credentials exist.
"""
from __future__ import annotations

import asyncio
import json
import ssl
from collections.abc import Awaitable, Callable
from typing import Any

import certifi
import websockets
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from kalshi_quant.shared.api.auth import Credentials
from kalshi_quant.shared.data_models.market import BookSnapshot, to_e2, to_e4
from kalshi_quant.shared.utilities.logging import get_logger
from kalshi_quant.shared.utilities.timeutil import now_ms

log = get_logger(__name__)
Handler = Callable[[dict[str, Any]], Awaitable[None]]
GapHandler = Callable[[int, int, str], None]


class LocalBook:
    """Order book built from orderbook_snapshot plus orderbook_delta messages."""

    def __init__(self) -> None:
        self.yes: dict[int, int] = {}
        self.no: dict[int, int] = {}

    def apply_snapshot(self, msg: dict[str, Any]) -> None:
        self.yes = {to_e4(p): to_e2(c) for p, c in msg.get("yes_dollars_fp") or []}
        self.no = {to_e4(p): to_e2(c) for p, c in msg.get("no_dollars_fp") or []}

    def apply_delta(self, msg: dict[str, Any]) -> None:
        side = self.yes if msg["side"] == "yes" else self.no
        price = to_e4(msg["price_dollars"])
        size = side.get(price, 0) + to_e2(msg["delta_fp"])
        if size > 0:
            side[price] = size
        else:
            side.pop(price, None)

    def snapshot(self) -> BookSnapshot:
        return BookSnapshot(sorted(self.yes.items(), reverse=True), sorted(self.no.items(), reverse=True))


class KalshiStream:
    """One connection. ``ticker``, ``trade`` and lifecycle cover all markets; books need tickers."""

    def __init__(
        self, url: str, credentials: Credentials, on_message: Handler, on_gap: GapHandler,
        channels: tuple[str, ...] = ("ticker", "trade", "market_lifecycle_v2"),
        book_interval_ms: int = 1000, ssl_context: "ssl.SSLContext | None" = None,
    ):
        self.url = url
        self.credentials = credentials
        self.on_message = on_message
        self.on_gap = on_gap
        self.channels = channels
        self.book_interval_ms = book_interval_ms
        self.ssl_context = ssl_context or (ssl.create_default_context(cafile=certifi.where())
                                           if url.startswith("wss") else None)
        self.books: dict[str, LocalBook] = {}
        self.book_tickers: set[str] = set()
        self._socket: Any = None
        self._book_sid: int | None = None
        self._seq: dict[int, int] = {}
        self._next_id = 1
        self.reconnects = 0
        self.connected = asyncio.Event()

    def _command(self, cmd: str, params: dict[str, Any]) -> str:
        self._next_id += 1
        return json.dumps({"id": self._next_id, "cmd": cmd, "params": params})

    async def set_book_tickers(self, tickers: set[str]) -> None:
        """Keep the order book subscription equal to ``tickers``."""
        added, removed = tickers - self.book_tickers, self.book_tickers - tickers
        self.book_tickers = set(tickers)
        for ticker in removed:
            self.books.pop(ticker, None)
        if self._socket is None:
            return
        if self._book_sid is None:
            if tickers:
                await self._socket.send(self._book_subscribe(sorted(tickers)))
            return
        if added:
            await self._socket.send(self._command("update_subscription", {
                "sid": self._book_sid, "action": "add_markets", "market_tickers": sorted(added)}))
        if removed:
            await self._socket.send(self._command("update_subscription", {
                "sid": self._book_sid, "action": "delete_markets", "market_tickers": sorted(removed)}))

    def _book_subscribe(self, tickers: list[str]) -> str:
        return self._command("subscribe", {"channels": ["orderbook_delta"], "market_tickers": tickers,
                                           "update_interval_ms": self.book_interval_ms})

    async def run(self, stop: asyncio.Event) -> None:
        delay = 1.0
        disconnected_at: int | None = None
        while not stop.is_set():
            headers = self.credentials.headers("GET", self.url)
            try:
                async with websockets.connect(self.url, additional_headers=headers, ssl=self.ssl_context,
                                              max_size=2**24, ping_interval=20, ping_timeout=30) as socket:
                    self._socket = socket
                    self._book_sid = None
                    self._seq.clear()
                    if disconnected_at is not None:
                        self.on_gap(disconnected_at, now_ms(), "websocket disconnected")
                        disconnected_at = None
                    await socket.send(self._command("subscribe", {"channels": list(self.channels)}))
                    if self.book_tickers:
                        await socket.send(self._book_subscribe(sorted(self.book_tickers)))
                    self.connected.set()
                    log.info("ws_connected", channels=self.channels)
                    delay = 1.0
                    await self._consume(socket, stop)
            except (ConnectionClosed, InvalidHandshake, OSError, TimeoutError) as exc:
                log.warning("ws_disconnected", error=repr(exc)[:200], retry_in=delay)
            finally:
                self._socket = None
                self.connected.clear()
            if stop.is_set():
                return
            disconnected_at = disconnected_at or now_ms()
            self.reconnects += 1
            try:
                await asyncio.wait_for(stop.wait(), delay)
            except TimeoutError:
                pass
            delay = min(delay * 2, 60)

    async def _consume(self, socket: Any, stop: asyncio.Event) -> None:
        stopper = asyncio.create_task(stop.wait())
        try:
            while True:
                reader = asyncio.create_task(socket.recv())
                done, _ = await asyncio.wait({reader, stopper}, return_when=asyncio.FIRST_COMPLETED)
                if stopper in done:
                    reader.cancel()
                    return
                message = json.loads(reader.result())
                await self._handle(socket, message)
        finally:
            stopper.cancel()

    async def _handle(self, socket: Any, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "subscribed":
            if message["msg"]["channel"] == "orderbook_delta":
                self._book_sid = message["msg"]["sid"]
            return
        if kind == "error":
            log.error("ws_error", code=message.get("msg", {}).get("code"), detail=message.get("msg"))
            return
        sid, seq = message.get("sid"), message.get("seq")
        if kind in ("orderbook_snapshot", "orderbook_delta"):
            ticker = message["msg"]["market_ticker"]
            expected = self._seq.get(sid)
            if kind == "orderbook_delta" and expected is not None and seq != expected + 1:
                # Deltas after a gap cannot be trusted; ask for a fresh snapshot.
                log.warning("ws_seq_gap", sid=sid, expected=expected + 1, got=seq)
                self.on_gap(now_ms(), now_ms(), f"orderbook seq gap on sid {sid}")
                self._seq[sid] = seq
                await socket.send(self._command("update_subscription", {
                    "sid": sid, "action": "get_snapshot", "market_tickers": sorted(self.book_tickers)}))
                return
            if seq is not None:
                self._seq[sid] = seq
            book = self.books.setdefault(ticker, LocalBook())
            if kind == "orderbook_snapshot":
                book.apply_snapshot(message["msg"])
            else:
                book.apply_delta(message["msg"])
        await self.on_message(message)
