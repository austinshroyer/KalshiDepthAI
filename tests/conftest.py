from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_kalshi import FakeKalshi  # noqa: E402

from kalshi_quant.shared.api.client import KalshiClient  # noqa: E402
from kalshi_quant.shared.api.rate_limit import AdaptiveRateLimiter  # noqa: E402
from kalshi_quant.shared.config import load_config  # noqa: E402
from kalshi_quant.shared.database.engine import Database  # noqa: E402


@pytest.fixture
def fake() -> FakeKalshi:
    return FakeKalshi()


def make_config(env: str, tmp_path: Path):
    config = load_config(env)
    config.database_url = f"sqlite:///{tmp_path}/{config.env.value}.sqlite"
    config.collection.use_websocket = False
    config.maintenance.archive_directory = str(tmp_path / "archive")
    config.shared.api.max_retries = 3
    return config


@pytest.fixture
def general_config(tmp_path):
    return make_config("general", tmp_path)


@pytest.fixture
def fifteen_config(tmp_path):
    return make_config("15m", tmp_path)


@pytest.fixture
def general_db(general_config):
    db = Database.for_config(general_config)
    yield db
    db.dispose()


@pytest.fixture
def fifteen_db(fifteen_config):
    db = Database.for_config(fifteen_config)
    yield db
    db.dispose()


def client_for(fake: FakeKalshi, config) -> KalshiClient:
    return KalshiClient(config.shared.api, AdaptiveRateLimiter(1_000_000), None, fake.transport())
