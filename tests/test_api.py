import asyncio
import base64
import inspect
from pathlib import Path

import pytest
from conftest import client_for
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from kalshi_quant.shared.api.auth import Credentials
from kalshi_quant.shared.api.client import KalshiApiError, KalshiClient
from kalshi_quant.shared.api.rate_limit import AdaptiveRateLimiter, background_work, endpoint_cost
from kalshi_quant.shared.config import PROJECT_ROOT


async def test_retries_429_and_5xx_then_succeeds(fake, general_config):
    fake.fail_next = [429, 503]
    async with client_for(fake, general_config) as client:
        assert (await client.series_list()) == []
        assert client.errors == 2


async def test_gives_up_after_max_retries(fake, general_config):
    general_config.shared.api.max_retries = 1
    fake.fail_next = [500, 500, 500]
    async with client_for(fake, general_config) as client:
        with pytest.raises(KalshiApiError):
            await client.series_list()


async def test_non_retryable_error_is_immediate(fake, general_config):
    fake.fail_next = [400]
    async with client_for(fake, general_config) as client:
        with pytest.raises(KalshiApiError):
            await client.series_list()
        assert len(fake.calls) == 1


async def test_pagination_walks_all_pages(fake, general_config):
    from fake_kalshi import market
    fake.page_size = 2
    for i in range(5):
        fake.markets[f"A-{i}"] = market(f"A-{i}", "A", 0, 1)
    async with client_for(fake, general_config) as client:
        pages = [p async for p, _ in client.markets()]
    assert [len(p) for p in pages] == [2, 2, 1]


def test_client_has_no_write_methods():
    """Paper trading must stay separate from real execution."""
    source = (PROJECT_ROOT / "kalshi_quant/shared/api/client.py").read_text()
    for verb in (".post(", ".put(", ".delete(", ".patch(", ".request("):
        assert verb not in source
    names = {name for name, _ in inspect.getmembers(KalshiClient)}
    assert not any("order" in n and "orderbook" not in n for n in names)


def test_no_code_references_order_endpoints():
    for path in (PROJECT_ROOT / "kalshi_quant").rglob("*.py"):
        text = path.read_text()
        assert "portfolio/orders" not in text, path


def test_ed25519_signature_verifies_and_excludes_query():
    key = ed25519.Ed25519PrivateKey.generate()
    headers = Credentials("id", key).headers("GET", "https://x.test/trade-api/v2/markets?limit=5")
    message = (headers["KALSHI-ACCESS-TIMESTAMP"] + "GET/trade-api/v2/markets").encode()
    key.public_key().verify(base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]), message)


def test_rsa_pss_signature_verifies():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    headers = Credentials("id", key).headers("GET", "wss://x.test/trade-api/ws/v2")
    message = (headers["KALSHI-ACCESS-TIMESTAMP"] + "GET/trade-api/ws/v2").encode()
    key.public_key().verify(base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]), message,
                            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                            hashes.SHA256())


def test_no_credentials_in_repository_config():
    for path in Path(PROJECT_ROOT / "configuration").glob("*.yaml"):
        assert "PRIVATE KEY" not in path.read_text()


def test_endpoint_costs():
    assert endpoint_cost("markets") == 40
    assert endpoint_cost("/markets/trades") == 40
    assert endpoint_cost("markets/orderbooks") == 10
    assert endpoint_cost("series/KXBTC15M") == 10


async def test_limiter_paces_by_token_cost():
    limiter = AdaptiveRateLimiter(1000, burst_seconds=0.01)
    start = asyncio.get_running_loop().time()
    for _ in range(5):
        await limiter.acquire(40)
    assert asyncio.get_running_loop().time() - start >= 0.12


async def test_limiter_backs_off_and_recovers():
    limiter = AdaptiveRateLimiter(100)
    limiter.on_throttled()
    assert limiter.rate == 70
    for _ in range(100):
        limiter.on_success()
    assert limiter.rate == 100


async def test_background_work_is_held_to_its_share():
    limiter = AdaptiveRateLimiter(1000, burst_seconds=10, background_share=0.5)

    async def sweep():
        background_work.set(True)
        for _ in range(3):
            await limiter.acquire(50)

    start = asyncio.get_running_loop().time()
    await asyncio.create_task(sweep())
    assert asyncio.get_running_loop().time() - start >= 0.19  # two gaps of 100 tokens at 500/s
