"""Database handle: one per environment, guarded so data cannot cross over."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import Connection, Engine, MetaData, create_engine, event, insert, inspect, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from kalshi_quant.shared.config import PROJECT_ROOT, Env, EnvConfig
from kalshi_quant.shared.database.schema import build_metadata


class EnvironmentMismatch(RuntimeError):
    """Raised when a process opens the other environment's database."""


def resolve_url(url: str) -> str:
    """Anchor relative SQLite paths at the project root."""
    prefix = "sqlite:///"
    if url.startswith(prefix) and not url[len(prefix):].startswith("/"):
        return prefix + str(PROJECT_ROOT / url[len(prefix):])
    return url


def _configure_sqlite(dbapi_conn: Any, _: Any) -> None:
    cursor = dbapi_conn.cursor()
    for pragma in ("journal_mode=WAL", "synchronous=NORMAL", "foreign_keys=ON", "busy_timeout=30000",
                   "temp_store=MEMORY", "mmap_size=268435456", "cache_size=-64000", "wal_autocheckpoint=4000"):
        cursor.execute(f"PRAGMA {pragma}")
    cursor.close()


class Database:
    def __init__(self, env: "Env | str", url: str, migrate: bool = True):
        self.env = Env.parse(env)
        self.url = resolve_url(url)
        self.is_sqlite = self.url.startswith("sqlite")
        if self.is_sqlite:
            self.path: Path | None = Path(self.url.removeprefix("sqlite:///"))
            self.path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.path = None
        self.engine: Engine = create_engine(self.url, future=True)
        if self.is_sqlite:
            event.listen(self.engine, "connect", _configure_sqlite)
        self.metadata: MetaData = build_metadata(self.env)
        self.t = self.metadata.tables
        self._guard_environment()  # before migrating: never touch the other environment's file
        if migrate:
            self.upgrade()
            self._guard_environment()

    @classmethod
    def for_config(cls, config: EnvConfig, migrate: bool = True) -> "Database":
        return cls(config.env, config.database_url, migrate)

    def upgrade(self) -> None:
        command.upgrade(alembic_config(self.env, self.url), "head")

    def _guard_environment(self) -> None:
        meta = self.t["meta"]
        if not inspect(self.engine).has_table("meta"):
            return
        with self.engine.begin() as conn:
            row = conn.execute(select(meta.c.value).where(meta.c.key == "environment")).first()
            if row is None:
                conn.execute(insert(meta).values(key="environment", value=self.env.value))
            elif row[0] != self.env.value:
                raise EnvironmentMismatch(f"{self.url} belongs to {row[0]!r}, not {self.env.value!r}")

    def connect(self) -> Connection:
        return self.engine.connect()

    def begin(self):  # noqa: ANN201
        return self.engine.begin()

    def upsert(self, table_name: str):  # noqa: ANN201
        """Dialect-aware INSERT supporting on_conflict_*: SQLite today, PostgreSQL later."""
        table = self.t[table_name]
        return sqlite_insert(table) if self.is_sqlite else pg_insert(table)

    def scalar(self, sql: str, **params: Any) -> Any:
        with self.connect() as conn:
            return conn.execute(text(sql), params).scalar()

    def dispose(self) -> None:
        self.engine.dispose()


def alembic_config(env: Env, url: str) -> AlembicConfig:
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    cfg.set_main_option("version_locations", str(PROJECT_ROOT / "migrations" / env.value))
    cfg.set_main_option("path_separator", "os")
    cfg.attributes["environment"] = env.value
    cfg.attributes["url"] = url
    return cfg
