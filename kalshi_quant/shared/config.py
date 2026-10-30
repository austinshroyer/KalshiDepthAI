"""Typed configuration for the two environments.

YAML files in ``configuration/`` hold every tunable. Credentials come only from
the process environment (or a git-ignored ``.env``).
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configuration"


class Env(str, Enum):
    GENERAL = "general"
    FIFTEEN_MINUTE = "fifteen_minute"

    @classmethod
    def parse(cls, value: "str | Env") -> "Env":
        if isinstance(value, Env):
            return value
        aliases = {"general": cls.GENERAL, "15m": cls.FIFTEEN_MINUTE, "fifteen_minute": cls.FIFTEEN_MINUTE}
        try:
            return aliases[value.lower()]
        except KeyError:
            raise ValueError(f"unknown environment {value!r}; use 'general' or '15m'") from None


class ApiConfig(BaseModel):
    rest_base_url: str
    ws_url: str
    timeout_seconds: float = 30
    max_retries: int = 8


class LoggingConfig(BaseModel):
    level: str = "INFO"
    directory: str = "logs"


class StorageConfig(BaseModel):
    data_directory: str = "data"
    min_free_disk_gb: float = 5.0


class SharedConfig(BaseModel):
    api: ApiConfig
    logging: LoggingConfig = LoggingConfig()
    storage: StorageConfig = StorageConfig()


class CollectionConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    collect_trades: bool = True
    trades_poll_seconds: float = 3
    orderbook_seconds: float = 60
    orderbook_levels: int = 10
    series_refresh_seconds: float = 600
    observation_heartbeat_seconds: float = 900
    use_websocket: bool = True
    ws_orderbook_interval_ms: int = 1000
    flush_seconds: float = 2
    flush_rows: int = 5000
    # general
    include_multivariate: bool = False
    full_sweep_seconds: float = 1800
    hot_sweep_seconds: float = 30
    hot_market_count: int = 3000
    metadata_sweep_seconds: float = 300
    settlement_sweep_seconds: float = 300
    event_sweep_seconds: float = 900
    orderbook_market_count: int = 300
    archive_sweep_seconds: float = 5
    # fifteen minute
    active_poll_seconds: float = 1.0
    discovery_seconds: float = 300
    settlement_poll_seconds: float = 3
    interval_minutes: int = 1


class MaintenanceConfig(BaseModel):
    archive_after_days: int = 30
    archive_directory: str = "data/archive"
    growth_log_seconds: float = 3600
    rollup_seconds: float = 300


class BacktestConfig(BaseModel):
    starting_capital: float = 10_000
    latency_ms: int = 1000
    max_quote_age_seconds: float = 900


class ExecutionConfig(BaseModel):
    max_book_participation: float = 0.5
    max_candle_volume_participation: float = 0.10
    slippage_ticks: int = 0
    candle_slippage_ticks: int = 1
    fee_rounding_dollars: float = 0.0001
    contract_step: float = 1


class PaperConfig(BaseModel):
    poll_seconds: float = 10     # live order books, the source of quotes
    market_seconds: float = 15   # /markets for status, volume and settlement (cached upstream)
    discovery_seconds: float = 300
    max_tracked_markets: int = 500
    max_book_markets: int = 500
    tokens_per_second: float = 50
    equity_log_seconds: float = 60
    use_websocket: bool = True


class EnvConfig(BaseModel):
    environment: Env
    database_url: str
    strategy_directory: str
    tokens_per_second: float = 100
    collection: CollectionConfig = CollectionConfig()
    maintenance: MaintenanceConfig = MaintenanceConfig()
    backtest: BacktestConfig = BacktestConfig()
    execution: ExecutionConfig = ExecutionConfig()
    paper: PaperConfig = PaperConfig()
    shared: SharedConfig

    @property
    def env(self) -> Env:
        return self.environment

    def strategy_dir(self) -> Path:
        return PROJECT_ROOT / self.strategy_directory


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def load_config(env: "str | Env", config_dir: Path | None = None) -> EnvConfig:
    env = Env.parse(env)
    root = config_dir or CONFIG_DIR
    shared = SharedConfig(**_read_yaml(root / "shared.yaml"))
    return EnvConfig(**_read_yaml(root / f"{env.value}.yaml"), shared=shared)


def load_dotenv(path: Path | None = None) -> None:
    """Load simple KEY=VALUE lines without overriding the real environment."""
    path = path or PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
