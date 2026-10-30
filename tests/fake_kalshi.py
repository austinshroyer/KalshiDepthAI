"""In-memory Kalshi for tests: REST via httpx.MockTransport, WebSocket via a local server."""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any
from urllib.parse import parse_qs

import httpx
import websockets

from kalshi_quant.shared.utilities.timeutil import ms_to_iso, now_ms

M15 = 900_000


def series(ticker: str, frequency: str, category: str = "Crypto", fee_type: str = "quadratic", mult: float = 1) -> dict:
    return {"ticker": ticker, "title": ticker, "frequency": frequency, "category": category, "tags": ["tag"],
            "fee_type": fee_type, "fee_multiplier": mult}


def market(ticker: str, event: str, open_ms: int, close_ms: int, status: str = "active", bid: str = "0.4000",
           ask: str = "0.4200", volume: str = "100.00", result: str = "", **extra: Any) -> dict:
    return {"ticker": ticker, "event_ticker": event, "market_type": "binary", "title": ticker, "status": status,
            "created_time": ms_to_iso(open_ms - 60_000), "open_time": ms_to_iso(open_ms), "close_time": ms_to_iso(close_ms),
            "yes_bid_dollars": bid, "yes_ask_dollars": ask, "no_bid_dollars": "0.5800", "no_ask_dollars": "0.6000",
            "yes_bid_size_fp": "10.00", "yes_ask_size_fp": "12.00", "last_price_dollars": "0.4100", "volume_fp": volume,
            "volume_24h_fp": volume, "open_interest_fp": "50.00", "result": result, "can_close_early": False,
            "notional_value_dollars": "1.0000", "rules_primary": "rules", **extra}


class FakeKalshi:
    def __init__(self) -> None:
        self.series: list[dict] = []
        self.markets: dict[str, dict] = {}
        self.events: list[dict] = []
        self.trades: list[dict] = []
        self.books: dict[str, dict] = {}
        self.calls: list[str] = []
        self.fail_next: list[int] = []  # status codes to return before succeeding
        self.page_size = 1000
        self.cutoff = "2020-01-01T00:00:00Z"
        self.candles: dict[str, list[dict]] = {}

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def add_trade(self, ticker: str, price: str = "0.4100", count: str = "5.00", ts_ms: int | None = None) -> str:
        trade_id = str(uuid.uuid4())
        self.trades.insert(0, {"trade_id": trade_id, "ticker": ticker, "yes_price_dollars": price,
                               "no_price_dollars": f"{1 - float(price):.4f}", "count_fp": count,
                               "taker_outcome_side": "yes", "created_time": ms_to_iso(ts_ms or now_ms()),
                               "is_block_trade": False})
        return trade_id

    def _page(self, items: list[dict], key: str, query: dict[str, str]) -> httpx.Response:
        start = int(query.get("cursor", 0))
        limit = min(int(query.get("limit", self.page_size)), self.page_size)
        chunk = items[start:start + limit]
        cursor = str(start + limit) if start + limit < len(items) else ""
        return httpx.Response(200, json={key: chunk, "cursor": cursor})

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.split("/trade-api/v2/")[-1]
        self.calls.append(path)
        assert request.method == "GET", "the platform must only ever issue GET"
        if self.fail_next:
            return httpx.Response(self.fail_next.pop(0), json={"error": "injected"})
        query = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
        multi = parse_qs(request.url.query.decode())
        if path == "series":
            return httpx.Response(200, json={"series": self.series})
        if path == "series/fee_changes":
            return httpx.Response(200, json={"series_fee_change_arr": []})
        if path.startswith("series/"):
            found = [s for s in self.series if s["ticker"] == path.split("/")[1]]
            return httpx.Response(200, json={"series": found[0]}) if found else httpx.Response(404, json={})
        if path == "events":
            return self._page([e for e in self.events if query.get("series_ticker") in (None, e["series_ticker"])], "events", query)
        if path == "markets":
            items = list(self.markets.values())
            if "tickers" in query:
                wanted = set(query["tickers"].split(","))
                items = [m for m in items if m["ticker"] in wanted]
            if "series_ticker" in query:
                items = [m for m in items if m["event_ticker"].split("-")[0] == query["series_ticker"]]
            status = query.get("status")
            mapping = {"open": ("active",), "unopened": ("initialized",), "settled": ("finalized",),
                       "closed": ("closed", "determined")}
            if status:
                items = [m for m in items if m["status"] in mapping[status]]
            if query.get("mve_filter") == "exclude":
                items = [m for m in items if not m.get("mve_collection_ticker")]
            return self._page(items, "markets", query)
        if path == "markets/trades":
            items = self.trades
            if "ticker" in query:
                items = [t for t in items if t["ticker"] == query["ticker"]]
            return self._page(items, "trades", query)
        if path == "markets/orderbooks":
            return httpx.Response(200, json={"orderbooks": [
                {"ticker": t, "orderbook_fp": self.books[t]} for t in multi.get("tickers", []) if t in self.books]})
        if path == "markets/candlesticks":
            return httpx.Response(200, json={"markets": [
                {"market_ticker": t, "candlesticks": self.candles.get(t, [])} for t in query["market_tickers"].split(",")]})
        match = re.fullmatch(r"markets/([^/]+)", path)
        if match:
            found = self.markets.get(match[1])
            return httpx.Response(200, json={"market": found}) if found else httpx.Response(404, json={})
        if path == "historical/cutoff":
            return httpx.Response(200, json={"market_settled_ts": self.cutoff, "trades_created_ts": self.cutoff})
        if path == "historical/markets":
            return httpx.Response(200, json={"markets": [], "cursor": ""})
        return httpx.Response(404, json={"error": path})


class FakeStream:
    """Local WebSocket server speaking the documented subscribe/ack/snapshot protocol."""

    def __init__(self) -> None:
        self.received: list[dict] = []
        self.headers: list[dict] = []
        self.connections: list[Any] = []
        self.server: Any = None
        self.port = 0

    async def __aenter__(self) -> "FakeStream":
        self.server = await websockets.serve(self._serve, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *_: object) -> None:
        self.server.close()
        await self.server.wait_closed()

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/trade-api/ws/v2"

    async def _serve(self, socket: Any) -> None:
        self.connections.append(socket)
        self.headers.append(dict(socket.request.headers))
        try:
            async for raw in socket:
                command = json.loads(raw)
                self.received.append(command)
                if command["cmd"] == "subscribe":
                    for index, channel in enumerate(command["params"]["channels"]):
                        await socket.send(json.dumps({"id": command["id"], "type": "subscribed",
                                                      "msg": {"channel": channel, "sid": 100 + index}}))
        except websockets.ConnectionClosed:
            pass

    async def send(self, message: dict) -> None:
        await self.connections[-1].send(json.dumps(message))

    async def wait_for(self, predicate: Any, timeout: float = 3) -> None:
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(0.01)
