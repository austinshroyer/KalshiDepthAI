"""Read-only async client for the Kalshi Trade API.

This platform never places orders: ``_get`` is the only request path and the
class exposes no way to issue POST, PUT or DELETE (enforced by tests).
"""
from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Iterable
from typing import Any

import httpx

from kalshi_quant.shared.api.auth import Credentials
from kalshi_quant.shared.api.rate_limit import AdaptiveRateLimiter, endpoint_cost
from kalshi_quant.shared.config import ApiConfig
from kalshi_quant.shared.utilities.logging import get_logger

log = get_logger(__name__)
RETRYABLE = {429, 500, 502, 503, 504}
MAX_ORDERBOOK_TICKERS = 100
MAX_CANDLE_TICKERS = 100
MAX_TICKERS_PER_MARKET_QUERY = 200  # keeps the URL at a safe length; not a documented limit


class KalshiApiError(RuntimeError):
    def __init__(self, status: int, url: str, body: str):
        super().__init__(f"HTTP {status} for {url}: {body[:200]}")
        self.status = status


def chunked(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


class KalshiClient:
    def __init__(
        self,
        config: ApiConfig,
        limiter: AdaptiveRateLimiter,
        credentials: Credentials | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.config = config
        self.limiter = limiter
        self.credentials = credentials
        self._http = httpx.AsyncClient(
            base_url=config.rest_base_url.rstrip("/") + "/",
            timeout=config.timeout_seconds,
            transport=transport,
            headers={"User-Agent": "kalshi-quant-research/1.0"},
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        self.requests = 0
        self.errors = 0

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "KalshiClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def _get(self, path: str, params: dict[str, Any] | None = None, allow_404: bool = False) -> Any:
        """GET with retries. Network failures and 429/5xx back off exponentially with jitter."""
        url = path.lstrip("/")
        params = {k: v for k, v in (params or {}).items() if v is not None}
        delay = 0.5
        for attempt in range(self.config.max_retries + 1):
            await self.limiter.acquire(endpoint_cost(url))
            self.requests += 1
            headers = self.credentials.headers("GET", str(self._http.base_url.join(url))) if self.credentials else None
            try:
                response = await self._http.get(url, params=params, headers=headers)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                failure: str = repr(exc)
                status = None
            else:
                status = response.status_code
                if status == 200:
                    self.limiter.on_success()
                    return response.json()
                if status == 404 and allow_404:
                    return None
                if status not in RETRYABLE:
                    self.errors += 1
                    raise KalshiApiError(status, url, response.text)
                failure = f"HTTP {status}"
                if status == 429:
                    self.limiter.on_throttled()
            self.errors += 1
            if attempt == self.config.max_retries:
                raise KalshiApiError(status or 0, url, failure)
            sleep_for = delay * (1 + random.random() / 2)
            log.warning("request_retry", path=url, failure=failure, attempt=attempt + 1, sleep=round(sleep_for, 2))
            await asyncio.sleep(sleep_for)
            delay = min(delay * 2, 30)
        raise AssertionError("unreachable")

    async def paginate(
        self, path: str, key: str, params: dict[str, Any] | None = None, cursor: str | None = None
    ) -> AsyncIterator[tuple[list[dict[str, Any]], str | None]]:
        """Yield (items, next_cursor) per page, so callers can checkpoint the cursor."""
        params = dict(params or {})
        while True:
            if cursor:
                params["cursor"] = cursor
            page = await self._get(path, params)
            cursor = page.get("cursor") or None
            yield page.get(key) or [], cursor
            if not cursor:
                return

    # Exchange
    async def exchange_status(self) -> dict[str, Any]:
        return await self._get("exchange/status")

    async def exchange_schedule(self) -> dict[str, Any]:
        return await self._get("exchange/schedule")

    async def historical_cutoff(self) -> dict[str, Any]:
        return await self._get("historical/cutoff")

    async def series_fee_changes(self) -> list[dict[str, Any]]:
        return (await self._get("series/fee_changes", {"show_historical": "true"})).get("series_fee_change_arr") or []

    # Series and events
    async def series_list(self, min_updated_ts: int | None = None) -> list[dict[str, Any]]:
        return (await self._get("series", {"min_updated_ts": min_updated_ts})).get("series") or []

    async def series(self, ticker: str) -> dict[str, Any] | None:
        data = await self._get(f"series/{ticker}", allow_404=True)
        return data["series"] if data else None

    async def event(self, event_ticker: str) -> dict[str, Any] | None:
        data = await self._get(f"events/{event_ticker}", {"with_nested_markets": "false"}, allow_404=True)
        return data.get("event") if data else None

    def events(self, cursor: str | None = None, **params: Any) -> AsyncIterator[tuple[list[dict], str | None]]:
        return self.paginate("events", "events", {"limit": 200, **params}, cursor)

    # Markets
    def markets(self, cursor: str | None = None, **params: Any) -> AsyncIterator[tuple[list[dict], str | None]]:
        return self.paginate("markets", "markets", {"limit": 1000, **params}, cursor)

    def historical_markets(self, cursor: str | None = None, **params: Any) -> AsyncIterator[tuple[list[dict], str | None]]:
        return self.paginate("historical/markets", "markets", {"limit": 1000, **params}, cursor)

    async def market(self, ticker: str) -> dict[str, Any] | None:
        data = await self._get(f"markets/{ticker}", allow_404=True)
        return data["market"] if data else None

    async def markets_by_ticker(self, tickers: list[str]) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for batch in chunked(tickers, MAX_TICKERS_PER_MARKET_QUERY):
            found += (await self._get("markets", {"tickers": ",".join(batch), "limit": 1000})).get("markets") or []
        return found

    async def orderbooks(self, tickers: list[str]) -> list[dict[str, Any]]:
        books: list[dict[str, Any]] = []
        for batch in chunked(tickers, MAX_ORDERBOOK_TICKERS):
            books += (await self._get("markets/orderbooks", {"tickers": batch})).get("orderbooks") or []
        return books

    # Trades and candles
    def trades(self, cursor: str | None = None, **params: Any) -> AsyncIterator[tuple[list[dict], str | None]]:
        return self.paginate("markets/trades", "trades", {"limit": 1000, **params}, cursor)

    def historical_trades(self, cursor: str | None = None, **params: Any) -> AsyncIterator[tuple[list[dict], str | None]]:
        return self.paginate("historical/trades", "trades", {"limit": 1000, **params}, cursor)

    async def candlesticks(self, tickers: list[str], start_ts: int, end_ts: int, period: int = 1) -> dict[str, list[dict]]:
        """Batch candles for live-tier markets, keyed by ticker."""
        data = await self._get("markets/candlesticks", {
            "market_tickers": ",".join(tickers), "start_ts": start_ts, "end_ts": end_ts, "period_interval": period,
        })
        return {m.get("market_ticker") or m.get("ticker"): m.get("candlesticks") or [] for m in data.get("markets") or []}

    async def historical_candlesticks(self, ticker: str, start_ts: int, end_ts: int, period: int = 1) -> list[dict]:
        data = await self._get(f"historical/markets/{ticker}/candlesticks", {
            "start_ts": start_ts, "end_ts": end_ts, "period_interval": period,
        }, allow_404=True)
        return (data or {}).get("candlesticks") or []
