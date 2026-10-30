"""Historical dataset: ticks in time order plus market facts and settlements.

Candles are stamped at their end time, the first moment their close was knowable. Outcome
data stays in ``settlements`` and only reaches a portfolio through the engine's settlement
step, at the market's real settlement time.

Provenance per market: collected observations when the market has them, otherwise exchange
candles. Candles have no displayed sizes, so fills against them are capped by candle volume.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sqlalchemy import text

from kalshi_quant.shared.data_models.tick import MarketInfo
from kalshi_quant.shared.database.engine import Database
from kalshi_quant.shared.database.maintenance import archive_paths

MIN_OBSERVATIONS = 5
NUMERIC = ("yes_bid", "yes_ask", "last", "yes_bid_size", "yes_ask_size", "volume", "volume_delta", "open_interest")


@dataclass
class DatasetSpec:
    environment: str
    start_ms: int | None = None
    end_ms: int | None = None
    categories: list[str] = field(default_factory=list)
    series: list[str] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    source: str = "auto"          # auto, observations, candles
    candle_minutes: int | None = None
    max_markets: int | None = None


@dataclass
class Settlement:
    ticker: str
    ts_ms: int
    yes_value: float
    timing_estimated: bool


@dataclass
class Dataset:
    spec: DatasetSpec
    infos: dict[str, MarketInfo]
    settlements: dict[str, Settlement]
    tickers: list[str]
    ticker_index: np.ndarray
    ts_ms: np.ndarray
    status: np.ndarray
    is_candle: np.ndarray
    values: dict[str, np.ndarray]
    levels: dict[int, tuple[tuple, tuple]]  # row index -> (yes_levels, no_levels)
    warnings: dict[str, int] = field(default_factory=dict)
    categories: dict[str, str | None] = field(default_factory=dict)
    gaps_ms: list[tuple[int, int]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ts_ms)

    def dataset_id(self) -> str:
        digest = hashlib.sha256()
        for array in (self.ticker_index, self.ts_ms, self.status, self.is_candle, *self.values.values()):
            digest.update(array.tobytes())
        digest.update(json.dumps(self.tickers).encode())
        digest.update(json.dumps(sorted((s.ticker, s.ts_ms, s.yes_value) for s in self.settlements.values())).encode())
        return digest.hexdigest()[:20]

    def time_range(self) -> tuple[int, int]:
        return int(self.ts_ms[0]), int(self.ts_ms[-1])

    def slice(self, start_ms: int, end_ms: int) -> "Dataset":
        """Rows in [start, end), for markets that finish before ``end``.

        A market still open at ``end`` is left out, so a fold can never settle a position
        using an outcome that lies outside the fold.
        """
        keep_tickers = {t for t, s in self.settlements.items() if s.ts_ms < end_ms}
        allowed = np.array([t in keep_tickers for t in self.tickers], dtype=bool)
        mask = (self.ts_ms >= start_ms) & (self.ts_ms < end_ms) & allowed[self.ticker_index]
        indices = np.nonzero(mask)[0]
        positions = {int(old): new for new, old in enumerate(indices)} if self.levels else {}
        return Dataset(self.spec, self.infos, {t: s for t, s in self.settlements.items() if t in keep_tickers},
                       self.tickers, self.ticker_index[mask], self.ts_ms[mask], self.status[mask], self.is_candle[mask],
                       {k: v[mask] for k, v in self.values.items()},
                       {positions[i]: v for i, v in self.levels.items() if i in positions}, dict(self.warnings),
                       self.categories, self.gaps_ms)

    def restrict(self, tickers: "set[str] | frozenset[str]") -> "Dataset":
        """The same data limited to these markets. Indices stay valid against ``self.tickers``."""
        allowed = np.array([t in tickers for t in self.tickers], dtype=bool)
        mask = allowed[self.ticker_index] if len(self) else np.empty(0, dtype=bool)
        indices = np.nonzero(mask)[0]
        positions = {int(old): new for new, old in enumerate(indices)} if self.levels else {}
        return Dataset(self.spec, {t: i for t, i in self.infos.items() if t in tickers},
                       {t: s for t, s in self.settlements.items() if t in tickers}, self.tickers,
                       self.ticker_index[mask], self.ts_ms[mask], self.status[mask], self.is_candle[mask],
                       {k: v[mask] for k, v in self.values.items()},
                       {positions[i]: v for i, v in self.levels.items() if i in positions}, dict(self.warnings),
                       self.categories, self.gaps_ms)

    def save(self, path: Path) -> None:
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> "Dataset":
        with path.open("rb") as fh:
            return pickle.load(fh)


def _market_rows(db: Database, spec: DatasetSpec) -> list[Any]:
    filters = ["m.open_ms IS NOT NULL"]
    params: dict[str, Any] = {}
    if spec.start_ms:
        filters.append("m.close_ms >= :start")
        params["start"] = spec.start_ms
    if spec.end_ms:
        filters.append("m.open_ms < :end")
        params["end"] = spec.end_ms
    for column, values, prefix in (("c.name", spec.categories, "c"), ("m.series_ticker", spec.series, "s"), ("m.ticker", spec.tickers, "t")):
        if values:
            names = [f"{prefix}{i}" for i in range(len(values))]
            filters.append(f"{column} IN ({', '.join(':' + n for n in names)})")
            params.update(zip(names, values))
    sql = f"""
        SELECT m.market_id, m.ticker, m.event_ticker, m.series_ticker, m.title, e.title AS event_title, c.name AS category,
               m.open_ms, m.close_ms, m.first_seen_close_ms, m.first_seen_ms, m.market_type, m.floor_strike, m.cap_strike,
               m.price_ranges, s.fee_type, s.fee_multiplier, e.fee_multiplier_override,
               r.result, r.settlement_value_e4, r.settlement_ms, r.resolved_successfully
        FROM markets m
        LEFT JOIN events e ON e.event_ticker = m.event_ticker
        LEFT JOIN series s ON s.series_ticker = m.series_ticker
        LEFT JOIN categories c ON c.category_id = s.category_id
        LEFT JOIN resolutions r ON r.market_id = m.market_id
        WHERE {' AND '.join(filters)} ORDER BY m.ticker"""
    if spec.max_markets:
        sql += f" LIMIT {int(spec.max_markets)}"
    with db.connect() as conn:
        return conn.execute(text(sql), params).mappings().all()


def _time_clause(column: str, spec: DatasetSpec) -> tuple[str, dict[str, Any]]:
    clause, params = "", {}
    if spec.start_ms:
        clause += f" AND {column} >= :start"
        params["start"] = spec.start_ms
    if spec.end_ms:
        clause += f" AND {column} < :end"
        params["end"] = spec.end_ms
    return clause, params


def _read_table(db: Database, table: str, columns: str, spec: DatasetSpec, ts_column: str = "ts_ms") -> pd.DataFrame:
    clause, params = _time_clause(ts_column, spec)
    with db.connect() as conn:
        frame = pd.read_sql_query(text(f"SELECT {columns} FROM {table} WHERE 1=1 {clause}"), conn, params=params)
    paths = archive_paths(db, table, spec.start_ms, spec.end_ms)
    if paths:
        listed = ", ".join(f"'{p}'" for p in paths)
        condition = "".join(f" AND {ts_column} {op} {value}" for op, value in
                            ((">=", spec.start_ms), ("<", spec.end_ms)) if value)
        duck = duckdb.connect()
        old = duck.execute(f"SELECT {columns} FROM read_parquet([{listed}]) WHERE 1=1 {condition}").df()
        duck.close()
        frame = pd.concat([old, frame], ignore_index=True) if not old.empty else frame
    return frame


def _candle_ticks(frame: pd.DataFrame) -> pd.DataFrame:
    """One tick per candle, at its end. Cumulative volume is rebuilt from candle volumes."""
    frame = frame.sort_values(["market_id", "end_ts_ms"])
    out = pd.DataFrame({
        "market_id": frame["market_id"], "ts_ms": frame["end_ts_ms"],
        "yes_bid": frame["yes_bid_close"], "yes_ask": frame["yes_ask_close"],
        "last_price": frame["price_close"], "volume_delta": frame["volume"],
        "volume": frame.groupby("market_id")["volume"].cumsum(), "open_interest": frame["open_interest"],
    })
    out.loc[(out["yes_bid"] <= 0), "yes_bid"] = np.nan
    out.loc[(out["yes_ask"] >= 10_000) | (out["yes_ask"] <= 0), "yes_ask"] = np.nan
    out["yes_bid_size"] = np.nan
    out["yes_ask_size"] = np.nan
    out["status"] = 2
    out["is_candle"] = True
    return out


def load_dataset(db: Database, spec: DatasetSpec) -> Dataset:
    markets = _market_rows(db, spec)
    warnings: dict[str, int] = {}
    ids = {m["market_id"]: m for m in markets}
    frames: list[pd.DataFrame] = []
    with_observations: set[int] = set()
    book_rows: dict[int, Any] = {}
    if spec.source in ("auto", "observations"):
        obs = _read_table(db, "observations", "market_id, ts_ms, yes_bid, yes_ask, last_price, yes_bid_size, yes_ask_size, "
                                              "volume, volume_delta, open_interest, status", spec)
        obs = obs[obs["market_id"].isin(ids)]
        counts = obs.groupby("market_id")["ts_ms"].transform("count")
        obs = obs[counts >= MIN_OBSERVATIONS]
        if not obs.empty:
            obs = obs.sort_values(["market_id", "ts_ms"]).drop_duplicates(["market_id", "ts_ms"], keep="last")
            obs = obs.assign(is_candle=False)
            with_observations = set(obs["market_id"].unique().tolist())
            frames.append(obs)
            books = _read_table(db, "orderbook_snapshots", "market_id, ts_ms, levels", spec)
            book_rows = {mid: g.sort_values("ts_ms") for mid, g in books[books["market_id"].isin(with_observations)].groupby("market_id")}
    if spec.source in ("auto", "candles"):
        candles = _read_table(db, "candlesticks", "*", spec, "end_ts_ms")
        candles = candles[candles["market_id"].isin(ids) & ~candles["market_id"].isin(with_observations)]
        if spec.candle_minutes:
            candles = candles[candles["period_minutes"] == spec.candle_minutes]
        elif not candles.empty:  # finest period per market
            finest = candles.groupby("market_id")["period_minutes"].transform("min")
            candles = candles[candles["period_minutes"] == finest]
        if not candles.empty:
            ticks = _candle_ticks(candles)
            closes = ticks["market_id"].map({k: m["close_ms"] for k, m in ids.items()})
            ticks.loc[ticks["ts_ms"] >= closes, "status"] = 4  # a candle ending at the close is not tradable
            frames.append(ticks)
    if frames:
        data = pd.concat(frames, ignore_index=True).sort_values(["ts_ms", "market_id"], kind="stable").reset_index(drop=True)
    else:
        data = pd.DataFrame(columns=["market_id", "ts_ms", "yes_bid", "yes_ask", "last_price", "yes_bid_size", "yes_ask_size",
                                     "volume", "volume_delta", "open_interest", "status", "is_candle"])
    present = set(data["market_id"].unique().tolist())
    tickers: list[str] = []
    index_of: dict[int, int] = {}
    infos: dict[str, MarketInfo] = {}
    settlements: dict[str, Settlement] = {}
    categories: dict[str, str | None] = {}
    warnings["markets_without_data"] = len(ids) - len(present)
    for market_id, m in ids.items():
        if market_id not in present:
            continue
        index_of[market_id] = len(tickers)
        tickers.append(m["ticker"])
        ranges = tuple((float(r["start"]), float(r["end"]), float(r["step"])) for r in json.loads(m["price_ranges"] or "[]"))
        multiplier = m["fee_multiplier_override"] if m["fee_multiplier_override"] is not None else m["fee_multiplier"]
        infos[m["ticker"]] = MarketInfo(m["ticker"], m["event_ticker"], m["series_ticker"], m["category"], m["title"],
                                        m["event_title"], m["open_ms"], m["first_seen_close_ms"] or m["close_ms"],
                                        m["market_type"], m["floor_strike"], m["cap_strike"], m["fee_type"],
                                        1.0 if multiplier is None else multiplier, ranges)
        categories[m["ticker"]] = m["category"]
        if m["first_seen_ms"] > (m["close_ms"] or 0):
            warnings["markets_first_seen_after_close"] = warnings.get("markets_first_seen_after_close", 0) + 1
        if m["close_ms"] != m["first_seen_close_ms"]:
            warnings["markets_close_time_moved"] = warnings.get("markets_close_time_moved", 0) + 1
        if m["result"] and m["settlement_value_e4"] is not None:
            estimated = m["settlement_ms"] is None
            settle_ms = m["settlement_ms"] or m["close_ms"]
            settlements[m["ticker"]] = Settlement(m["ticker"], settle_ms, m["settlement_value_e4"] / 10_000, estimated)
            if estimated:
                warnings["settlement_time_estimated"] = warnings.get("settlement_time_estimated", 0) + 1
        else:
            warnings["markets_unresolved"] = warnings.get("markets_unresolved", 0) + 1
    size = len(data)
    def column(name: str, scale: float) -> np.ndarray:
        return (pd.to_numeric(data[name], errors="coerce").to_numpy(dtype=np.float64) / scale) if size else np.empty(0)
    values = {"yes_bid": column("yes_bid", 10_000), "yes_ask": column("yes_ask", 10_000), "last": column("last_price", 10_000),
              "yes_bid_size": column("yes_bid_size", 100), "yes_ask_size": column("yes_ask_size", 100),
              "volume": column("volume", 100), "volume_delta": column("volume_delta", 100),
              "open_interest": column("open_interest", 100)}
    market_ids = data["market_id"].to_numpy(dtype=np.int64) if size else np.empty(0, dtype=np.int64)
    ts = data["ts_ms"].to_numpy(dtype=np.int64) if size else np.empty(0, dtype=np.int64)
    is_candle = data["is_candle"].fillna(False).to_numpy(dtype=bool) if size else np.empty(0, dtype=bool)
    warnings["candle_only_markets"] = len({m for m in present if m not in with_observations})
    levels: dict[int, tuple[tuple, tuple]] = {}
    for market_id, group in book_rows.items():
        rows = np.nonzero(market_ids == market_id)[0]
        book_ts = group["ts_ms"].to_numpy()
        # The snapshot used at a row is the latest one at or before that row: never a later one.
        positions = np.searchsorted(book_ts, ts[rows], side="right") - 1
        raw = group["levels"].tolist()
        parsed: dict[int, tuple[tuple, tuple]] = {}
        for row, position in zip(rows, positions):
            if position >= 0:
                if position not in parsed:
                    book = json.loads(raw[position])
                    parsed[position] = (tuple((p / 10_000, s / 100) for p, s in book["y"]),
                                        tuple((p / 10_000, s / 100) for p, s in book["n"]))
                levels[int(row)] = parsed[position]
    with db.connect() as conn:
        gaps = [tuple(r) for r in conn.execute(text("SELECT start_ms, end_ms FROM collection_gaps WHERE end_ms > start_ms"))]
    return Dataset(spec, infos, settlements, tickers,
                   np.array([index_of[m] for m in market_ids], dtype=np.int32) if size else np.empty(0, dtype=np.int32),
                   ts, data["status"].fillna(0).to_numpy(dtype=np.int16) if size else np.empty(0, dtype=np.int16),
                   is_candle, values, levels, warnings, categories, gaps)


def spec_to_json(spec: DatasetSpec) -> str:
    return json.dumps(asdict(spec), sort_keys=True)
