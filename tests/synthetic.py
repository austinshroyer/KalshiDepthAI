"""Deterministic synthetic history written straight into an environment database."""
from __future__ import annotations

import random

from kalshi_quant.shared.config import Env
from kalshi_quant.shared.database.engine import Database

DAY = 86_400_000
BASE = 1_780_000_000_000 // DAY * DAY


def seed(db: Database, days: int = 60, markets_per_day: int = 4, drift: float = 0.0, seed_value: int = 7,
         duration_ms: int = 3_600_000, step_ms: int = 60_000, series: str = "SERIES", category: str = "Crypto") -> list[str]:
    """Random-walk markets. Outcomes follow the final price, so a market ends where it trended."""
    rng = random.Random(seed_value)
    t = db.t
    frequency = "fifteen_min" if db.env is Env.FIFTEEN_MINUTE else "daily"
    with db.begin() as conn:
        category_id = conn.execute(t["categories"].insert().values(name=category)).inserted_primary_key[0]
        conn.execute(t["series"].insert().values(series_ticker=series, title=series, category_id=category_id, frequency=frequency,
                                                 fee_type="quadratic", fee_multiplier=1.0, first_seen_ms=BASE, updated_ms=BASE))
    tickers = []
    observations, resolutions = [], []
    with db.begin() as conn:
        for day in range(days):
            for k in range(markets_per_day):
                ticker, event = f"{series}-D{day:03d}-{k}", f"{series}-D{day:03d}"
                open_ms = BASE + day * DAY + k * 6 * 3_600_000
                close_ms = open_ms + duration_ms
                conn.execute(db.upsert("events").on_conflict_do_nothing(index_elements=["event_ticker"]),
                             [{"event_ticker": event, "series_ticker": series, "title": event, "is_stub": False}])
                market_id = conn.execute(t["markets"].insert().values(
                    ticker=ticker, event_ticker=event, series_ticker=series, title=ticker, market_type="binary", status="finalized",
                    open_ms=open_ms, close_ms=close_ms, first_seen_close_ms=close_ms, first_seen_ms=open_ms - 1000,
                    last_seen_ms=close_ms, duration_s=duration_ms // 1000)).inserted_primary_key[0]
                price, volume = 5000, 0
                for ts in range(open_ms, close_ms, step_ms):
                    price = max(300, min(9700, price + rng.randint(-250, 250) + int(drift * 10_000)))
                    delta = rng.randint(50, 500) * 100
                    volume += delta
                    bid, ask = price - 100, price + 100
                    observations.append({
                        "market_id": market_id, "ts_ms": ts, "source": 1, "recv_ms": ts, "yes_bid": bid, "yes_ask": ask,
                        "no_bid": 10_000 - ask, "no_ask": 10_000 - bid, "yes_prob": price, "last_price": price,
                        "yes_bid_size": 50_000, "yes_ask_size": 50_000, "volume": volume, "volume_delta": delta,
                        "open_interest": volume, "status": 2})
                won = price > 5000
                resolutions.append({"market_id": market_id, "result": "yes" if won else "no", "winning_outcome": "yes" if won else "no",
                                    "losing_outcome": "no" if won else "yes", "settlement_value_e4": 10_000 if won else 0,
                                    "settlement_ms": close_ms + 5000, "final_status": "finalized", "resolved_successfully": True,
                                    "recorded_ms": close_ms + 5000})
                tickers.append(ticker)
        conn.execute(t["observations"].insert(), observations)
        conn.execute(t["resolutions"].insert(), resolutions)
    return tickers
