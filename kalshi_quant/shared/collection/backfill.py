"""Backfill of exchange history: candlesticks and executed trades.

Kalshi offers no historical order books. What this recovers is 1-minute or 1-hour
OHLC of bid, ask and trade price, plus every trade. Those stay in their own tables
and are never written as observations, which are prospective snapshots only.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import text

from kalshi_quant.shared.api.client import MAX_CANDLE_TICKERS, KalshiClient, chunked
from kalshi_quant.shared.collection.rollups import run_rollups
from kalshi_quant.shared.config import EnvConfig
from kalshi_quant.shared.data_models.market import to_e2, to_e4
from kalshi_quant.shared.database.engine import Database
from kalshi_quant.shared.database.writer import BatchWriter
from kalshi_quant.shared.utilities.logging import get_logger
from kalshi_quant.shared.utilities.timeutil import now_ms, parse_ts_ms

log = get_logger(__name__)
MAX_CANDLES_PER_CALL = 4_000


def candle_row(market_id: int, period: int, c: dict[str, Any], tier: str) -> dict[str, Any]:
    bid, ask, price = c.get("yes_bid") or {}, c.get("yes_ask") or {}, c.get("price") or {}
    row: dict[str, Any] = {"market_id": market_id, "period_minutes": period, "end_ts_ms": c["end_period_ts"] * 1000,
                           "volume": to_e2(c.get("volume_fp")), "open_interest": to_e2(c.get("open_interest_fp")), "tier": tier}
    for prefix, source in (("yes_bid", bid), ("yes_ask", ask), ("price", price)):
        for part in ("open", "high", "low", "close"):
            row[f"{prefix}_{part}"] = to_e4(source.get(f"{part}_dollars"))
    row["price_mean"] = to_e4(price.get("mean_dollars"))
    return row


def _targets(db: Database, kind: str, since_ms: int | None, until_ms: int | None, series: str | None,
             limit: int | None, min_volume_e2: int) -> list[Any]:
    table = "candlesticks" if kind == "candles" else "trades"
    filters = ["m.open_ms IS NOT NULL", "m.close_ms < :now", "m.status IN ('finalized', 'determined', 'closed', 'amended')"]
    filters.append(f"NOT EXISTS (SELECT 1 FROM {table} x WHERE x.market_id = m.market_id"
                   + (" AND x.source = 3)" if kind == "trades" else ")"))
    params: dict[str, Any] = {"now": now_ms()}
    if since_ms:
        filters.append("m.close_ms >= :since")
        params["since"] = since_ms
    if until_ms:
        filters.append("m.close_ms < :until")
        params["until"] = until_ms
    if series:
        filters.append("m.series_ticker = :series")
        params["series"] = series
    sql = f"SELECT m.market_id, m.ticker, m.open_ms, m.close_ms FROM markets m WHERE {' AND '.join(filters)} ORDER BY m.close_ms DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with db.connect() as conn:
        return conn.execute(text(sql), params).mappings().all()


async def backfill_candles(client: KalshiClient, db: Database, config: EnvConfig, period: int, since_ms: int | None,
                           until_ms: int | None = None, series: str | None = None, limit: int | None = None) -> int:
    targets = _targets(db, "candles", since_ms, until_ms, series, limit, 0)
    cutoff_ms = parse_ts_ms((await client.historical_cutoff())["market_settled_ts"])
    writer = BatchWriter(db, flush_rows=5000)
    written = 0
    live = [t for t in targets if t["close_ms"] >= cutoff_ms]
    old = [t for t in targets if t["close_ms"] < cutoff_ms]
    log.info("backfill_candles", live=len(live), historical=len(old), period=period)
    for market in old:
        rows = await _historical_market_candles(client, market, period)
        writer.add("candlesticks", rows)
        written += len(rows)
        await writer.maybe_flush()
    for group in _group_by_window(live, period):
        by_ticker = {t["ticker"]: t for t in group}
        start = min(t["open_ms"] for t in group) // 1000 - 60
        end = max(t["close_ms"] for t in group) // 1000 + 60
        fetched = await client.candlesticks(list(by_ticker), start, end, period)
        for ticker, candles in fetched.items():
            if ticker in by_ticker:
                rows = [candle_row(by_ticker[ticker]["market_id"], period, c, "live") for c in candles]
                writer.add("candlesticks", rows)
                written += len(rows)
        await writer.maybe_flush()
    await writer.flush()
    run_rollups(db, config.env, config.collection.interval_minutes, limit=len(targets) + 1)
    return written


def _group_by_window(targets: list[Any], period: int) -> list[list[Any]]:
    """Batch markets with similar windows so one call never exceeds the candle limit."""
    max_window_ms = MAX_CANDLES_PER_CALL // MAX_CANDLE_TICKERS * period * 60_000
    groups: list[list[Any]] = []
    ordered = sorted(targets, key=lambda t: t["open_ms"])
    current: list[Any] = []
    for target in ordered:
        window_start = current[0]["open_ms"] if current else target["open_ms"]
        if current and (len(current) == MAX_CANDLE_TICKERS or target["close_ms"] - window_start > max_window_ms):
            groups.append(current)
            current = []
        current.append(target)
    if current:
        groups.append(current)
    return groups


async def _historical_market_candles(client: KalshiClient, market: Any, period: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    step = MAX_CANDLES_PER_CALL * period * 60
    start, end = market["open_ms"] // 1000 - 60, market["close_ms"] // 1000 + 60
    while start < end:
        candles = await client.historical_candlesticks(market["ticker"], start, min(start + step, end), period)
        rows += [candle_row(market["market_id"], period, c, "historical") for c in candles]
        start += step
    return rows


async def backfill_trades(client: KalshiClient, db: Database, since_ms: int | None, until_ms: int | None = None,
                          series: str | None = None, limit: int | None = None) -> int:
    targets = _targets(db, "trades", since_ms, until_ms, series, limit, 0)
    cutoff_ms = parse_ts_ms((await client.historical_cutoff())["trades_created_ts"])
    writer = BatchWriter(db, flush_rows=5000)
    written = 0
    for market in targets:
        feeds = []
        if market["close_ms"] >= cutoff_ms:
            feeds.append(client.trades(ticker=market["ticker"]))
        if market["open_ms"] < cutoff_ms:
            feeds.append(client.historical_trades(ticker=market["ticker"]))
        for feed in feeds:
            async for page, _ in feed:
                rows = [{"trade_id": uuid.UUID(t["trade_id"]).bytes, "market_id": market["market_id"],
                         "ts_ms": parse_ts_ms(t["created_time"]), "yes_price": to_e4(t["yes_price_dollars"]),
                         "count": to_e2(t["count_fp"]),
                         "taker_side": {"yes": 1, "no": 0}.get(t.get("taker_outcome_side") or t.get("taker_side")),
                         "is_block": t.get("is_block_trade"), "source": 3} for t in page]
                writer.add("trades", rows)
                written += len(rows)
                await writer.maybe_flush()
    await writer.flush()
    return written
