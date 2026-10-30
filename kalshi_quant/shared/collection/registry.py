"""Maps tickers to market ids and persists metadata only when it changes."""
from __future__ import annotations

import hashlib
import json
import zlib
from typing import Any

from sqlalchemy import select

from kalshi_quant.shared.collection.classification import Classification
from kalshi_quant.shared.data_models.market import NormalizedMarket
from kalshi_quant.shared.database.engine import Database
from kalshi_quant.shared.utilities.timeutil import now_ms, parse_ts_ms

META_FIELDS = (
    "event_ticker", "series_ticker", "title", "subtitle", "yes_sub_title", "no_sub_title", "rules_primary",
    "rules_secondary", "market_type", "status", "created_ms", "open_ms", "close_ms", "expected_expiration_ms",
    "latest_expiration_ms", "expiration_ms", "can_close_early", "strike_type", "floor_strike", "cap_strike",
    "price_ranges", "notional_e4", "exchange_index",
)
UPDATE_FIELDS = (*META_FIELDS, "duration_s", "mve_collection", "classification_reason", "is_stub", "last_seen_ms")


def compress_raw(kind: str, key: str, payload: dict[str, Any], ts_ms: int) -> dict[str, Any]:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {"kind": kind, "key": key, "ts_ms": ts_ms, "payload_hash": hashlib.sha1(text).hexdigest()[:16],
            "payload": zlib.compress(text, 6)}


class MarketRegistry:
    def __init__(self, db: Database):
        self.db = db
        self.ids: dict[str, int] = {}
        self.hashes: dict[str, int] = {}
        self.known_events: set[str] = set()
        self.known_series: set[str] = set()
        self.category_ids: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        t = self.db.t
        with self.db.connect() as conn:
            self.ids = dict(conn.execute(select(t["markets"].c.ticker, t["markets"].c.market_id)).all())
            self.known_events = set(conn.execute(select(t["events"].c.event_ticker)).scalars())
            self.known_series = set(conn.execute(select(t["series"].c.series_ticker)).scalars())
            self.category_ids = dict(conn.execute(select(t["categories"].c.name, t["categories"].c.category_id)).all())

    def category_id(self, conn: Any, name: str | None) -> int | None:
        if not name:
            return None
        if name not in self.category_ids:
            statement = self.db.upsert("categories").values(name=name).on_conflict_do_nothing(index_elements=["name"])
            conn.execute(statement)
            table = self.db.t["categories"]
            self.category_ids[name] = conn.execute(select(table.c.category_id).where(table.c.name == name)).scalar_one()
        return self.category_ids[name]

    def upsert_series(self, series_list: list[dict[str, Any]]) -> None:
        ts = now_ms()
        with self.db.begin() as conn:
            rows = [{
                "series_ticker": s["ticker"], "title": s.get("title"),
                "category_id": self.category_id(conn, s.get("category")),
                "subcategory": ",".join(s.get("tags") or []) or None,
                "frequency": s.get("frequency"), "fee_type": s.get("fee_type"),
                "fee_multiplier": s.get("fee_multiplier"), "first_seen_ms": ts,
                "updated_ms": parse_ts_ms(s.get("last_updated_ts")) or ts,
            } for s in series_list]
            if not rows:
                return
            statement = self.db.upsert("series")
            update = ("title", "category_id", "subcategory", "frequency", "fee_type", "fee_multiplier", "updated_ms")
            statement = statement.on_conflict_do_update(
                index_elements=["series_ticker"], set_={c: statement.excluded[c] for c in update})
            for start in range(0, len(rows), 2000):
                conn.execute(statement, rows[start:start + 2000])
        self.known_series |= {r["series_ticker"] for r in rows}

    def upsert_events(self, events: list[dict[str, Any]]) -> None:
        rows = [{
            "event_ticker": e["event_ticker"],
            "series_ticker": e.get("series_ticker") if e.get("series_ticker") in self.known_series else None,
            "title": e.get("title"), "sub_title": e.get("sub_title"),
            "strike_ms": parse_ts_ms(e.get("strike_date")), "strike_period": e.get("strike_period") or None,
            "mutually_exclusive": e.get("mutually_exclusive"),
            "collateral_return_type": e.get("collateral_return_type") or None,
            "cadence": (e.get("product_metadata") or {}).get("cadence"),
            "fee_type_override": e.get("fee_type_override"),
            "fee_multiplier_override": e.get("fee_multiplier_override"),
            "exchange_index": e.get("exchange_index"), "is_stub": False,
            "updated_ms": parse_ts_ms(e.get("last_updated_ts")) or now_ms(),
        } for e in events]
        if not rows:
            return
        with self.db.begin() as conn:
            statement = self.db.upsert("events")
            update = [c for c in rows[0] if c != "event_ticker"]
            statement = statement.on_conflict_do_update(
                index_elements=["event_ticker"], set_={c: statement.excluded[c] for c in update})
            for start in range(0, len(rows), 1000):
                conn.execute(statement, rows[start:start + 1000])
        self.known_events |= {r["event_ticker"] for r in rows}

    def upsert_markets(self, markets: list[tuple[NormalizedMarket, Classification]]) -> list[dict[str, Any]]:
        """Persist new or changed markets. Returns raw-response rows to buffer."""
        ts = now_ms()
        changed: list[tuple[NormalizedMarket, Classification]] = []
        for market, classification in markets:
            digest = hash(tuple(getattr(market, f) for f in META_FIELDS))
            if self.hashes.get(market.ticker) != digest:
                self.hashes[market.ticker] = digest
                changed.append((market, classification))
        if not changed:
            return []
        rows = []
        for market, classification in changed:
            row = {f: getattr(market, f) for f in META_FIELDS}
            if row["series_ticker"] not in self.known_series:
                row["series_ticker"] = None
            row.update(duration_s=market.duration_s, mve_collection=market.mve_collection,
                       classification_reason=classification.reason, is_stub=False,
                       first_seen_close_ms=market.close_ms, first_seen_ms=ts, last_seen_ms=ts, ticker=market.ticker)
            rows.append(row)
        stub_events = {r["event_ticker"]: r["series_ticker"] for r in rows
                       if r["event_ticker"] and r["event_ticker"] not in self.known_events}
        table = self.db.t["markets"]
        with self.db.begin() as conn:
            if stub_events:
                conn.execute(self.db.upsert("events").on_conflict_do_nothing(index_elements=["event_ticker"]),
                             [{"event_ticker": e, "series_ticker": s, "is_stub": True} for e, s in stub_events.items()])
            statement = self.db.upsert("markets")
            statement = statement.on_conflict_do_update(
                index_elements=["ticker"], set_={c: statement.excluded[c] for c in UPDATE_FIELDS})
            for start in range(0, len(rows), 1000):
                conn.execute(statement, rows[start:start + 1000])
            new = [r["ticker"] for r in rows if r["ticker"] not in self.ids]
            for start in range(0, len(new), 900):
                chunk = new[start:start + 900]
                self.ids.update(conn.execute(select(table.c.ticker, table.c.market_id)
                                             .where(table.c.ticker.in_(chunk))).all())
        self.known_events |= set(stub_events)
        return [compress_raw("market", m.ticker, m.raw, ts) for m, _ in changed]

    def ensure_stub(self, ticker: str, event_ticker: str | None) -> int:
        """Register a market seen in the trade feed before its metadata arrives."""
        if ticker in self.ids:
            return self.ids[ticker]
        ts = now_ms()
        table = self.db.t["markets"]
        with self.db.begin() as conn:
            if event_ticker and event_ticker not in self.known_events:
                conn.execute(self.db.upsert("events").on_conflict_do_nothing(index_elements=["event_ticker"]),
                             [{"event_ticker": event_ticker, "is_stub": True}])
                self.known_events.add(event_ticker)
            statement = self.db.upsert("markets").on_conflict_do_nothing(index_elements=["ticker"])
            conn.execute(statement, [{"ticker": ticker, "event_ticker": event_ticker, "status": "unknown",
                                      "is_stub": True, "first_seen_ms": ts, "last_seen_ms": ts}])
            self.ids[ticker] = conn.execute(select(table.c.market_id).where(table.c.ticker == ticker)).scalar_one()
        return self.ids[ticker]
