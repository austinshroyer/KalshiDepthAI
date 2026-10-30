"""Stats, growth tracking, integrity checks, backups and Parquet archiving."""
from __future__ import annotations

import gzip
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from sqlalchemy import inspect, text

from kalshi_quant.shared.config import PROJECT_ROOT, EnvConfig
from kalshi_quant.shared.database.engine import Database
from kalshi_quant.shared.database.schema import TIME_SERIES_TABLES
from kalshi_quant.shared.utilities.logging import get_logger
from kalshi_quant.shared.utilities.timeutil import ms_to_dt, now_ms

log = get_logger(__name__)
DAY_MS = 86_400_000
STAT_TABLES = ("series", "events", "markets", "observations", "trades", "orderbook_snapshots", "candlesticks",
               "resolutions", "market_summary", "raw_responses")


def row_counts(db: Database, tables: tuple[str, ...] = STAT_TABLES) -> dict[str, int]:
    with db.connect() as conn:
        return {t: conn.execute(text(f'SELECT count(*) FROM "{t}"')).scalar() for t in tables}


def file_bytes(db: Database) -> int:
    if not db.path or not db.path.exists():
        return 0
    return sum(p.stat().st_size for p in db.path.parent.glob(db.path.name + "*"))


def log_growth(db: Database) -> None:
    with db.begin() as conn:
        conn.execute(db.t["db_growth"].insert().values(
            ts_ms=now_ms(), file_bytes=file_bytes(db), row_counts=json.dumps(row_counts(db))))


def growth_report(db: Database) -> list[dict[str, Any]]:
    with db.connect() as conn:
        rows = conn.execute(text("SELECT ts_ms, file_bytes, row_counts FROM db_growth ORDER BY ts_ms")).all()
    return [{"ts_ms": r[0], "file_bytes": r[1], "rows": json.loads(r[2])} for r in rows]


def database_stats(db: Database) -> dict[str, Any]:
    stats: dict[str, Any] = {"environment": db.env.value, "path": str(db.path), "file_bytes": file_bytes(db),
                             "rows": row_counts(db)}
    with db.connect() as conn:
        stats["observation_span"] = tuple(conn.execute(text("SELECT min(ts_ms), max(ts_ms) FROM observations")).first())
        stats["gaps"] = conn.execute(text("SELECT count(*) FROM collection_gaps")).scalar()
        stats["archived_rows"] = conn.execute(text("SELECT coalesce(sum(row_count), 0) FROM archives")).scalar()
        stats["heartbeats"] = [tuple(r) for r in conn.execute(text("SELECT stream, ts_ms, detail FROM heartbeats ORDER BY stream"))]
    return stats


def integrity_check(db: Database) -> dict[str, Any]:
    with db.connect() as conn:
        quick = [r[0] for r in conn.execute(text("PRAGMA quick_check"))]
        fk = conn.execute(text("PRAGMA foreign_key_check")).fetchmany(20)
    head = inspect(db.engine).has_table("alembic_version")
    return {"quick_check": quick, "foreign_key_violations": [tuple(r) for r in fk], "migrated": head,
            "ok": quick == ["ok"] and not fk and head}


def backup(db: Database, destination: Path | None = None, compress: bool = True) -> Path:
    """Consistent online backup through SQLite's backup API; safe while collectors run."""
    assert db.path, "backup requires SQLite"
    stamp = ms_to_dt(now_ms()).strftime("%Y%m%dT%H%M%SZ")
    directory = destination or db.path.parent / "backups"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{db.path.stem}-{stamp}.sqlite"
    source_conn = sqlite3.connect(db.path)
    target_conn = sqlite3.connect(target)
    with target_conn:
        source_conn.backup(target_conn, pages=20_000, sleep=0.01)
    source_conn.close()
    target_conn.close()
    if not compress:
        return target
    gz = target.with_suffix(".sqlite.gz")
    with target.open("rb") as raw, gzip.open(gz, "wb", compresslevel=6) as out:
        shutil.copyfileobj(raw, out, 1 << 20)
    target.unlink()
    return gz


def archive_table_day(db: Database, table: str, day_start_ms: int, directory: Path) -> int:
    """Move one UTC day of a table to zstd Parquet. Rows are deleted only after a read-back."""
    ts_column = TIME_SERIES_TABLES[table]
    day_end = day_start_ms + DAY_MS
    params = {"a": day_start_ms, "b": day_end}
    with db.connect() as conn:
        frame = pd.read_sql_query(text(f"SELECT * FROM {table} WHERE {ts_column} >= :a AND {ts_column} < :b"), conn, params=params)
    if frame.empty:
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{table}-{ms_to_dt(day_start_ms):%Y-%m-%d}.parquet"
    temp = path.with_suffix(".tmp")
    duck = duckdb.connect()
    duck.register("frame", frame)
    duck.execute(f"COPY frame TO '{temp}' (FORMAT parquet, COMPRESSION zstd)")
    written = duck.execute(f"SELECT count(*) FROM read_parquet('{temp}')").fetchone()[0]
    duck.close()
    if written != len(frame):
        temp.unlink()
        raise RuntimeError(f"archive read-back mismatch for {path}: {written} != {len(frame)}")
    temp.replace(path)
    with db.begin() as conn:
        conn.execute(db.t["archives"].insert().values(
            table_name=table, min_ts_ms=int(frame[ts_column].min()), max_ts_ms=int(frame[ts_column].max()),
            path=str(path), row_count=len(frame), created_ms=now_ms()))
        conn.execute(text(f"DELETE FROM {table} WHERE {ts_column} >= :a AND {ts_column} < :b"), params)
    log.info("archived", table=table, day=str(ms_to_dt(day_start_ms).date()), rows=len(frame), path=str(path))
    return len(frame)


def run_archive(db: Database, config: EnvConfig, max_days: int = 1) -> int:
    cutoff = (now_ms() - config.maintenance.archive_after_days * DAY_MS) // DAY_MS * DAY_MS
    directory = PROJECT_ROOT / config.maintenance.archive_directory
    moved = 0
    for table, column in TIME_SERIES_TABLES.items():
        for _ in range(max_days):
            oldest = db.scalar(f"SELECT min({column}) FROM {table}")
            if oldest is None or oldest >= cutoff:
                break
            moved += archive_table_day(db, table, oldest // DAY_MS * DAY_MS, directory)
    return moved


def archive_paths(db: Database, table: str, start_ms: int | None, end_ms: int | None) -> list[str]:
    with db.connect() as conn:
        return [r[0] for r in conn.execute(text("""
            SELECT path FROM archives WHERE table_name = :t
              AND (:a IS NULL OR max_ts_ms >= :a) AND (:b IS NULL OR min_ts_ms <= :b) ORDER BY min_ts_ms"""),
            {"t": table, "a": start_ms, "b": end_ms})]
