"""Batched writer. One per pipeline, so each database has a single write path.

Rows accumulate in memory and flush in one transaction, together with the
checkpoints that describe them. A crash can therefore lose at most the unflushed
window, and a restart resumes from a checkpoint that matches what was committed.
"""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sqlalchemy.exc import OperationalError

from kalshi_quant.shared.database.engine import Database
from kalshi_quant.shared.utilities.logging import get_logger
from kalshi_quant.shared.utilities.timeutil import now_ms

log = get_logger(__name__)
CHUNK = 2000


@dataclass(frozen=True)
class Upsert:
    """How a buffered table is written: ignore conflicts, or update the given columns."""
    keys: tuple[str, ...]
    update: tuple[str, ...] = ()


POLICIES: dict[str, Upsert] = {
    "observations": Upsert(("market_id", "ts_ms", "source")),
    "trades": Upsert(("trade_id",)),
    "orderbook_snapshots": Upsert(("market_id", "ts_ms", "source")),
    "candlesticks": Upsert(("market_id", "period_minutes", "end_ts_ms"), (
        "yes_bid_open", "yes_bid_high", "yes_bid_low", "yes_bid_close",
        "yes_ask_open", "yes_ask_high", "yes_ask_low", "yes_ask_close",
        "price_open", "price_high", "price_low", "price_close", "price_mean",
        "volume", "open_interest", "tier")),
    "market_latest": Upsert(("market_id",), (
        "ts_ms", "yes_bid", "yes_ask", "last_price", "yes_prob", "volume", "volume_24h",
        "open_interest", "status")),
    "resolutions": Upsert(("market_id",), (
        "result", "winning_outcome", "losing_outcome", "settlement_value_e4", "settlement_ms",
        "expiration_value", "final_status", "resolved_successfully", "recorded_ms")),
    "classification_audit": Upsert(("ticker",), ("anomaly", "ts_ms")),
    "raw_responses": Upsert(("kind", "key", "payload_hash")),
}
ORDER = ["observations", "trades", "orderbook_snapshots", "candlesticks", "resolutions",
         "classification_audit", "raw_responses", "market_latest"]


class BatchWriter:
    def __init__(self, db: Database, flush_seconds: float = 2, flush_rows: int = 5000):
        self.db = db
        self.flush_seconds = flush_seconds
        self.flush_rows = flush_rows
        self.buffers: dict[str, dict[Any, dict[str, Any]] | list[dict[str, Any]]] = defaultdict(list)
        self.latest: dict[int, dict[str, Any]] = {}
        self.checkpoints: dict[str, str] = {}
        self.pending = 0
        self.total_written: dict[str, int] = defaultdict(int)
        self.paused_tables: set[str] = set()
        self._flush_lock = asyncio.Lock()

    def add(self, table: str, rows: list[dict[str, Any]]) -> None:
        if table in self.paused_tables or not rows:
            return
        self.buffers[table].extend(rows)  # type: ignore[union-attr]
        self.pending += len(rows)

    def set_latest(self, row: dict[str, Any]) -> None:
        self.latest[row["market_id"]] = row

    def checkpoint(self, name: str, value: Any) -> None:
        self.checkpoints[name] = json.dumps(value)

    async def maybe_flush(self) -> None:
        if self.pending >= self.flush_rows:
            await self.flush()

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), self.flush_seconds)
            except TimeoutError:
                pass
            await self.flush()

    async def flush(self) -> None:
        async with self._flush_lock:
            if not self.pending and not self.checkpoints and not self.latest:
                return
            buffers, latest, checkpoints = self.buffers, self.latest, self.checkpoints
            self.buffers, self.latest, self.checkpoints = defaultdict(list), {}, {}
            self.pending = 0
            if latest:
                buffers["market_latest"] = list(latest.values())
            for attempt in range(6):
                try:
                    await asyncio.to_thread(self._write, buffers, checkpoints)
                    return
                except OperationalError as exc:  # another process holding the write lock
                    log.warning("flush_retry", attempt=attempt, error=str(exc)[:120])
                    await asyncio.sleep(0.5 * 2**attempt)
            raise RuntimeError("database write failed after retries")

    def _write(self, buffers: dict[str, list[dict[str, Any]]], checkpoints: dict[str, str]) -> None:
        with self.db.begin() as conn:
            for table in ORDER:
                rows = buffers.get(table)
                if not rows:
                    continue
                policy = POLICIES[table]
                statement = self.db.upsert(table)
                if policy.update:
                    statement = statement.on_conflict_do_update(
                        index_elements=list(policy.keys),
                        set_={c: statement.excluded[c] for c in policy.update})
                else:
                    statement = statement.on_conflict_do_nothing(index_elements=list(policy.keys))
                for start in range(0, len(rows), CHUNK):
                    conn.execute(statement, rows[start:start + CHUNK])
                self.total_written[table] += len(rows)
            if checkpoints:
                write_checkpoints(self.db, conn, checkpoints)


def write_checkpoints(db: Database, conn: Any, checkpoints: dict[str, str]) -> None:
    statement = db.upsert("checkpoints")
    statement = statement.on_conflict_do_update(
        index_elements=["name"], set_={"value": statement.excluded.value, "updated_ms": statement.excluded.updated_ms})
    ts = now_ms()
    conn.execute(statement, [{"name": k, "value": v, "updated_ms": ts} for k, v in checkpoints.items()])


def read_checkpoint(db: Database, name: str, default: Any = None) -> Any:
    from sqlalchemy import select
    table = db.t["checkpoints"]
    with db.connect() as conn:
        row = conn.execute(select(table.c.value).where(table.c.name == name)).first()
    return json.loads(row[0]) if row else default
