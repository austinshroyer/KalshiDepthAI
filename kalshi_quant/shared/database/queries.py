"""Market queries by category, event, ticker, date, duration, resolution, probability,
volume and initial projected winner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import text

from kalshi_quant.shared.config import Env
from kalshi_quant.shared.database.engine import Database


@dataclass
class MarketQuery:
    category: str | None = None
    series: str | None = None
    event: str | None = None
    ticker: str | None = None
    status: str | None = None
    open_after_ms: int | None = None
    open_before_ms: int | None = None
    close_after_ms: int | None = None
    close_before_ms: int | None = None
    min_duration_s: int | None = None
    max_duration_s: int | None = None
    result: str | None = None
    min_probability: float | None = None
    max_probability: float | None = None
    min_volume: float | None = None
    initial_winner: str | None = None
    limit: int = 100


def find_markets(db: Database, q: MarketQuery) -> pd.DataFrame:
    conditions = ["1=1"]
    params: dict[str, Any] = {"limit": q.limit}

    def add(value: Any, sql: str, name: str, convert: Any = lambda v: v) -> None:
        if value is not None:
            conditions.append(sql)
            params[name] = convert(value)

    add(q.category, "v.category = :category", "category")
    add(q.series, "v.series_ticker = :series", "series")
    add(q.event, "v.event_ticker = :event", "event")
    add(q.ticker, "v.ticker = :ticker", "ticker")
    add(q.status, "v.status = :status", "status")
    add(q.open_after_ms, "v.open_ms >= :open_after", "open_after")
    add(q.open_before_ms, "v.open_ms < :open_before", "open_before")
    add(q.close_after_ms, "v.close_ms >= :close_after", "close_after")
    add(q.close_before_ms, "v.close_ms < :close_before", "close_before")
    add(q.min_duration_s, "v.duration_s >= :min_duration", "min_duration")
    add(q.max_duration_s, "v.duration_s <= :max_duration", "max_duration")
    add(q.result, "v.result = :result", "result")
    add(q.min_probability, "v.current_yes_prob >= :min_prob", "min_prob")
    add(q.max_probability, "v.current_yes_prob <= :max_prob", "max_prob")
    add(q.min_volume, "v.cumulative_volume >= :min_volume", "min_volume")
    columns = "v.ticker, v.category, v.status, v.title, v.current_yes_prob, v.cumulative_volume, v.result"
    join = ""
    if db.env is Env.GENERAL:
        columns += ", s.initial_projected_winner, s.initial_winner_prob / 10000.0 AS initial_winner_prob, s.initial_source, s.initial_reliable"
        join = "LEFT JOIN market_summary s ON s.market_id = v.market_id"
        add(q.initial_winner, "s.initial_projected_winner = :initial_winner", "initial_winner")
    sql = f"SELECT {columns} FROM market_overview v {join} WHERE {' AND '.join(conditions)} ORDER BY v.close_ms DESC LIMIT :limit"
    with db.connect() as conn:
        return pd.read_sql_query(text(sql), conn, params=params)


def observations_for(db: Database, ticker: str, start_ms: int | None = None, end_ms: int | None = None, limit: int = 1000) -> pd.DataFrame:
    sql = """SELECT * FROM observations_decimal WHERE ticker = :t
             AND (:a IS NULL OR ts_ms >= :a) AND (:b IS NULL OR ts_ms < :b) ORDER BY ts_ms DESC LIMIT :n"""
    with db.connect() as conn:
        return pd.read_sql_query(text(sql), conn, params={"t": ticker, "a": start_ms, "b": end_ms, "n": limit})
