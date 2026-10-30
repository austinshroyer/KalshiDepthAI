"""Derived per-market summaries, recomputed idempotently from stored data.

Initial probability uses the earliest reliable evidence from three sources:
collected observations, exchange candles and exchange trades. An observation only
counts when it is a two-sided quote with a sane spread, because Kalshi's opening
book is often a meaningless 0.01/0.99. If the first evidence arrives long after
the open, the row is kept but marked unreliable.
"""
from __future__ import annotations

import bisect
from typing import Any

from sqlalchemy import select, text

from kalshi_quant.shared.config import Env
from kalshi_quant.shared.data_models.market import STATUS_CODES
from kalshi_quant.shared.database.engine import Database
from kalshi_quant.shared.utilities.timeutil import now_ms

MAX_RELIABLE_SPREAD_E4 = 2000
RELIABLE_DELAY_S = {Env.GENERAL: 3600, Env.FIFTEEN_MINUTE: 60}
BATCH = 3000


def _candidates(db: Database, limit: int) -> list[Any]:
    sql = """
        SELECT m.market_id, m.open_ms, m.close_ms, l.ts_ms AS latest_ts, l.yes_prob AS latest_prob,
               l.volume AS latest_volume, l.status AS latest_status
        FROM markets m
        JOIN market_latest l ON l.market_id = m.market_id
        LEFT JOIN market_summary s ON s.market_id = m.market_id
        WHERE s.market_id IS NULL OR s.updated_ms < l.ts_ms OR s.initial_reliable IS NOT 1
        ORDER BY s.updated_ms IS NOT NULL, s.updated_ms
        LIMIT :limit"""
    with db.connect() as conn:
        return conn.execute(text(sql), {"limit": limit}).mappings().all()


def _first_observation(conn: Any, market_id: int) -> tuple[int, int] | None:
    return conn.execute(text("""
        SELECT ts_ms, (yes_bid + yes_ask) / 2 FROM observations
        WHERE market_id = :m AND yes_bid IS NOT NULL AND yes_ask IS NOT NULL
          AND yes_ask - yes_bid <= :spread AND status = :active
        ORDER BY ts_ms LIMIT 1"""), {"m": market_id, "spread": MAX_RELIABLE_SPREAD_E4,
                                     "active": STATUS_CODES["active"]}).first()


def _first_candle(conn: Any, market_id: int) -> tuple[int, int] | None:
    return conn.execute(text("""
        SELECT end_ts_ms, (yes_bid_close + yes_ask_close) / 2 FROM candlesticks
        WHERE market_id = :m AND yes_bid_close > 0 AND yes_ask_close < 10000
          AND yes_ask_close - yes_bid_close <= :spread
        ORDER BY end_ts_ms LIMIT 1"""), {"m": market_id, "spread": MAX_RELIABLE_SPREAD_E4}).first()


def _first_trade(conn: Any, market_id: int) -> tuple[int, int] | None:
    return conn.execute(text("""
        SELECT ts_ms, yes_price FROM trades WHERE market_id = :m ORDER BY ts_ms LIMIT 1"""),
                        {"m": market_id}).first()


def earliest_evidence(conn: Any, market_id: int) -> tuple[int, int, str] | None:
    found = [(row[0], row[1], source) for source, row in (
        ("observation", _first_observation(conn, market_id)),
        ("candle", _first_candle(conn, market_id)),
        ("trade", _first_trade(conn, market_id))) if row]
    return min(found, key=lambda item: item[0]) if found else None


def projected(prob: int | None) -> tuple[str | None, int | None]:
    if prob is None:
        return None, None
    if prob == 5000:
        return "tie", 5000
    return ("yes", prob) if prob > 5000 else ("no", 10_000 - prob)


def run_rollups(db: Database, env: Env, interval_minutes: int = 1, limit: int = BATCH) -> int:
    candidates = _candidates(db, limit)
    if not candidates:
        return 0
    ts = now_ms()
    closed = {STATUS_CODES[s] for s in ("closed", "determined", "disputed", "amended", "finalized")}
    rows = []
    with db.connect() as conn:
        for c in candidates:
            evidence = earliest_evidence(conn, c["market_id"])
            row: dict[str, Any] = {
                "market_id": c["market_id"], "initial_ts_ms": None, "initial_yes_prob": None, "initial_source": None,
                "initial_delay_s": None, "initial_reliable": False, "total_volume": c["latest_volume"],
                "trade_count": conn.execute(text("SELECT count(*) FROM trades WHERE market_id = :m"),
                                            {"m": c["market_id"]}).scalar(),
                "updated_ms": ts, "final_ts_ms": None, "final_yes_prob": None,
            }
            if evidence:
                first_ts, prob, source = evidence
                delay = None if c["open_ms"] is None else max(0, (first_ts - c["open_ms"]) // 1000)
                row.update(initial_ts_ms=first_ts, initial_yes_prob=prob, initial_source=source, initial_delay_s=delay,
                           initial_reliable=delay is not None and delay <= RELIABLE_DELAY_S[env])
            if c["latest_status"] in closed and c["close_ms"]:
                final = conn.execute(text("""
                    SELECT ts_ms, yes_prob FROM observations
                    WHERE market_id = :m AND ts_ms < :close AND yes_prob IS NOT NULL
                    ORDER BY ts_ms DESC LIMIT 1"""), {"m": c["market_id"], "close": c["close_ms"]}).first()
                if final:
                    row.update(final_ts_ms=final[0], final_yes_prob=final[1])
            if env is Env.GENERAL:
                winner, winner_prob = projected(row["initial_yes_prob"])
                current, current_prob = projected(c["latest_prob"])
                row.update(initial_projected_winner=winner, initial_winner_prob=winner_prob,
                           current_projected_winner=current, current_winner_prob=current_prob)
            rows.append(row)
    with db.begin() as conn:
        statement = db.upsert("market_summary")
        statement = statement.on_conflict_do_update(
            index_elements=["market_id"], set_={k: statement.excluded[k] for k in rows[0] if k != "market_id"})
        conn.execute(statement, rows)
    if env is Env.FIFTEEN_MINUTE:
        fifteen_minute_intervals(db, [c for c in candidates if c["latest_status"] in closed], interval_minutes)
    return len(rows)


def fifteen_minute_intervals(db: Database, markets: list[Any], interval_minutes: int) -> None:
    """Carry the last observation at or before each grid point. Never looks past the point."""
    step_ms = interval_minutes * 60_000
    obs = db.t["observations"]
    grid_rows: list[dict[str, Any]] = []
    with db.connect() as conn:
        for market in markets:
            if not market["open_ms"] or not market["close_ms"]:
                continue
            rows = conn.execute(select(obs.c.ts_ms, obs.c.yes_prob, obs.c.yes_bid, obs.c.yes_ask, obs.c.volume)
                                .where(obs.c.market_id == market["market_id"], obs.c.ts_ms <= market["close_ms"])
                                .order_by(obs.c.ts_ms)).all()
            if not rows:
                continue
            times = [r[0] for r in rows]
            for grid_ms in range(market["open_ms"], market["close_ms"] + 1, step_ms):
                index = bisect.bisect_right(times, grid_ms) - 1
                if index >= 0:
                    r = rows[index]
                    grid_rows.append({"market_id": market["market_id"], "offset_seconds": (grid_ms - market["open_ms"]) // 1000,
                                      "ts_ms": r[0], "yes_prob": r[1], "yes_bid": r[2], "yes_ask": r[3], "volume": r[4]})
    if grid_rows:
        with db.begin() as conn:
            statement = db.upsert("market_interval_probs")
            statement = statement.on_conflict_do_update(
                index_elements=["market_id", "offset_seconds"],
                set_={c: statement.excluded[c] for c in ("ts_ms", "yes_prob", "yes_bid", "yes_ask", "volume")})
            conn.execute(statement, grid_rows)
