"""Relational schema, defined once in SQLAlchemy Core.

Separated by purpose: metadata (categories, series, events, markets), time series
(observations, trades, orderbook_snapshots, candlesticks), outcomes (resolutions,
market_summary), and operations (checkpoints, gaps, paper, backtest, optimization).

Provenance is explicit. ``observations`` and ``orderbook_snapshots`` are prospective
snapshots captured by this application. ``trades`` are exchange trade records.
``candlesticks`` are exchange history. The three never mix.

Each environment gets its own metadata object and its own database file. Tables
that only make sense in one environment are added by ``build_metadata``.
"""
from __future__ import annotations

from sqlalchemy import (
    BigInteger, Boolean, Column, Float, ForeignKey, Index, Integer, LargeBinary,
    MetaData, SmallInteger, String, Table, Text, UniqueConstraint,
)

from kalshi_quant.shared.config import Env

Ms = BigInteger
AutoId = BigInteger().with_variant(Integer, "sqlite")


def _market_fk() -> ForeignKey:
    return ForeignKey("markets.market_id")


def build_metadata(env: "Env | str") -> MetaData:
    env = Env.parse(env)
    md = MetaData()

    Table("meta", md,
          Column("key", String, primary_key=True),
          Column("value", Text, nullable=False))

    Table("categories", md,
          Column("category_id", AutoId, primary_key=True, autoincrement=True),
          Column("name", String, nullable=False, unique=True))

    Table("series", md,
          Column("series_ticker", String, primary_key=True),
          Column("title", Text),
          Column("category_id", ForeignKey("categories.category_id"), index=True),
          Column("subcategory", Text),  # Kalshi tags; the API has no subcategory field
          Column("frequency", String, index=True),
          Column("fee_type", String),
          Column("fee_multiplier", Float),
          Column("first_seen_ms", Ms, nullable=False),
          Column("updated_ms", Ms, nullable=False))

    Table("fee_changes", md,
          Column("id", String, primary_key=True),
          Column("series_ticker", String, nullable=False, index=True),
          Column("fee_type", String),
          Column("fee_multiplier", Float),
          Column("scheduled_ms", Ms, nullable=False))

    Table("events", md,
          Column("event_ticker", String, primary_key=True),
          Column("series_ticker", ForeignKey("series.series_ticker"), index=True),
          Column("title", Text),
          Column("sub_title", Text),
          Column("strike_ms", Ms),
          Column("strike_period", String),
          Column("mutually_exclusive", Boolean),
          Column("collateral_return_type", String),
          Column("cadence", String),
          Column("fee_type_override", String),
          Column("fee_multiplier_override", Float),
          Column("exchange_index", SmallInteger),
          Column("is_stub", Boolean, nullable=False, default=False),
          Column("updated_ms", Ms))

    Table("markets", md,
          Column("market_id", AutoId, primary_key=True, autoincrement=True),
          Column("ticker", String, nullable=False, unique=True),
          Column("event_ticker", ForeignKey("events.event_ticker"), index=True),
          Column("series_ticker", ForeignKey("series.series_ticker"), index=True),
          Column("title", Text),
          Column("subtitle", Text),
          Column("yes_sub_title", Text),
          Column("no_sub_title", Text),
          Column("rules_primary", Text),
          Column("rules_secondary", Text),
          Column("market_type", String),
          Column("status", String, index=True),
          Column("created_ms", Ms),
          Column("open_ms", Ms, index=True),
          Column("close_ms", Ms, index=True),
          # close_ms as first seen. The API rewrites close_time when a market closes
          # early, so only this value is safe to show a strategy at decision time.
          Column("first_seen_close_ms", Ms),
          Column("expected_expiration_ms", Ms),
          Column("latest_expiration_ms", Ms),
          Column("expiration_ms", Ms),
          Column("duration_s", Integer, index=True),
          Column("can_close_early", Boolean),
          Column("strike_type", String),
          Column("floor_strike", Float),
          Column("cap_strike", Float),
          Column("price_ranges", Text),
          Column("notional_e4", Integer),
          Column("exchange_index", SmallInteger),
          Column("mve_collection", String),
          Column("classification_reason", String),
          Column("is_stub", Boolean, nullable=False, default=False),
          Column("first_seen_ms", Ms, nullable=False),
          Column("last_seen_ms", Ms, nullable=False))

    # Latest state per market: lets a restarted collector keep computing volume
    # deltas and suppressing unchanged rows without scanning observations.
    Table("market_latest", md,
          Column("market_id", _market_fk(), primary_key=True),
          Column("ts_ms", Ms, nullable=False),
          Column("yes_bid", Integer), Column("yes_ask", Integer), Column("last_price", Integer),
          Column("yes_prob", Integer),
          Column("volume", BigInteger), Column("volume_24h", BigInteger), Column("open_interest", BigInteger),
          Column("status", SmallInteger))

    Table("observations", md,
          Column("market_id", _market_fk(), primary_key=True),
          Column("ts_ms", Ms, primary_key=True),
          Column("source", SmallInteger, primary_key=True),
          Column("recv_ms", Ms),
          Column("yes_bid", Integer), Column("yes_ask", Integer),
          Column("no_bid", Integer), Column("no_ask", Integer),
          Column("yes_prob", Integer),  # no_prob is 10000 - yes_prob
          Column("last_price", Integer),
          Column("yes_bid_size", BigInteger), Column("yes_ask_size", BigInteger),
          Column("volume", BigInteger), Column("volume_delta", BigInteger),
          Column("open_interest", BigInteger),
          Column("status", SmallInteger),
          Index("ix_observations_ts", "ts_ms"),
          sqlite_with_rowid=False)

    Table("trades", md,
          Column("trade_id", LargeBinary(16), primary_key=True),
          Column("market_id", _market_fk(), nullable=False),
          Column("ts_ms", Ms, nullable=False),
          Column("yes_price", Integer, nullable=False),
          Column("count", BigInteger, nullable=False),
          Column("taker_side", SmallInteger),  # 1 yes, 0 no
          Column("is_block", Boolean),
          Column("source", SmallInteger),  # 1 live REST, 2 websocket, 3 historical REST
          Index("ix_trades_market_ts", "market_id", "ts_ms"),
          Index("ix_trades_ts", "ts_ms"),
          sqlite_with_rowid=False)

    Table("orderbook_snapshots", md,
          Column("market_id", _market_fk(), primary_key=True),
          Column("ts_ms", Ms, primary_key=True),
          Column("source", SmallInteger, primary_key=True),
          Column("yes_bid", Integer), Column("yes_ask", Integer),
          Column("yes_depth", BigInteger), Column("no_depth", BigInteger),  # over stored levels
          Column("levels", Text, nullable=False),  # {"y":[[price,size],...],"n":[...]} best first
          Index("ix_orderbook_ts", "ts_ms"),
          sqlite_with_rowid=False)

    Table("candlesticks", md,
          Column("market_id", _market_fk(), primary_key=True),
          Column("period_minutes", SmallInteger, primary_key=True),
          Column("end_ts_ms", Ms, primary_key=True),
          Column("yes_bid_open", Integer), Column("yes_bid_high", Integer),
          Column("yes_bid_low", Integer), Column("yes_bid_close", Integer),
          Column("yes_ask_open", Integer), Column("yes_ask_high", Integer),
          Column("yes_ask_low", Integer), Column("yes_ask_close", Integer),
          Column("price_open", Integer), Column("price_high", Integer),
          Column("price_low", Integer), Column("price_close", Integer), Column("price_mean", Integer),
          Column("volume", BigInteger), Column("open_interest", BigInteger),
          Column("tier", String),  # live or historical endpoint
          Index("ix_candles_end", "end_ts_ms"),
          sqlite_with_rowid=False)

    Table("resolutions", md,
          Column("market_id", _market_fk(), primary_key=True),
          Column("result", String),
          Column("winning_outcome", String),
          Column("losing_outcome", String),
          Column("settlement_value_e4", Integer),
          Column("settlement_ms", Ms, index=True),
          Column("expiration_value", String),
          Column("final_status", String),
          Column("resolved_successfully", Boolean, nullable=False),
          Column("recorded_ms", Ms, nullable=False))

    summary_columns = [
        Column("market_id", _market_fk(), primary_key=True),
        Column("initial_ts_ms", Ms),
        Column("initial_yes_prob", Integer),
        Column("initial_source", String),  # observation, candle, trade
        Column("initial_delay_s", Integer),  # seconds after open_ms
        Column("initial_reliable", Boolean),
        Column("final_ts_ms", Ms),
        Column("final_yes_prob", Integer),
        Column("total_volume", BigInteger),
        Column("trade_count", Integer),
        Column("updated_ms", Ms, nullable=False),
    ]
    if env is Env.GENERAL:
        summary_columns += [
            Column("initial_projected_winner", String, index=True),
            Column("initial_winner_prob", Integer, index=True),
            Column("current_projected_winner", String),
            Column("current_winner_prob", Integer),
        ]
    Table("market_summary", md, *summary_columns)

    if env is Env.FIFTEEN_MINUTE:
        # Probability on a fixed grid through each market's life, for shape analysis.
        Table("market_interval_probs", md,
              Column("market_id", _market_fk(), primary_key=True),
              Column("offset_seconds", Integer, primary_key=True),
              Column("ts_ms", Ms),
              Column("yes_prob", Integer),
              Column("yes_bid", Integer), Column("yes_ask", Integer),
              Column("volume", BigInteger),
              sqlite_with_rowid=False)

    Table("raw_responses", md,
          Column("raw_id", AutoId, primary_key=True, autoincrement=True),
          Column("kind", String, nullable=False),  # market, event, series
          Column("key", String, nullable=False),
          Column("ts_ms", Ms, nullable=False),
          Column("payload_hash", String, nullable=False),
          Column("payload", LargeBinary, nullable=False),  # zlib-compressed JSON
          UniqueConstraint("kind", "key", "payload_hash", name="uq_raw_version"))

    Table("classification_audit", md,
          Column("ticker", String, primary_key=True),
          Column("series_ticker", String),
          Column("anomaly", Text, nullable=False),
          Column("ts_ms", Ms, nullable=False))

    Table("checkpoints", md,
          Column("name", String, primary_key=True),
          Column("value", Text, nullable=False),
          Column("updated_ms", Ms, nullable=False))

    Table("collection_gaps", md,
          Column("gap_id", AutoId, primary_key=True, autoincrement=True),
          Column("stream", String, nullable=False),
          Column("start_ms", Ms, nullable=False, index=True),
          Column("end_ms", Ms, nullable=False),
          Column("reason", Text))

    Table("heartbeats", md,
          Column("stream", String, primary_key=True),
          Column("ts_ms", Ms, nullable=False),
          Column("detail", Text))

    Table("db_growth", md,
          Column("ts_ms", Ms, primary_key=True),
          Column("file_bytes", BigInteger, nullable=False),
          Column("row_counts", Text, nullable=False))

    Table("archives", md,
          Column("archive_id", AutoId, primary_key=True, autoincrement=True),
          Column("table_name", String, nullable=False),
          Column("min_ts_ms", Ms, nullable=False),
          Column("max_ts_ms", Ms, nullable=False),
          Column("path", Text, nullable=False),
          Column("row_count", BigInteger, nullable=False),
          Column("created_ms", Ms, nullable=False))

    # Paper trading: every simulated state change is reconstructable from fills.
    Table("paper_sessions", md,
          Column("session_id", AutoId, primary_key=True, autoincrement=True),
          Column("strategy_name", String, nullable=False, index=True),
          Column("strategy_path", Text, nullable=False),
          Column("strategy_hash", String, nullable=False),
          Column("params", Text, nullable=False),
          Column("starting_capital", Float, nullable=False),
          Column("status", String, nullable=False),  # running, stop_requested, stopped, crashed
          Column("pid", Integer),
          Column("started_ms", Ms, nullable=False),
          Column("heartbeat_ms", Ms),
          Column("stopped_ms", Ms))

    Table("paper_orders", md,
          Column("order_id", AutoId, primary_key=True, autoincrement=True),
          Column("session_id", ForeignKey("paper_sessions.session_id"), nullable=False, index=True),
          Column("ticker", String, nullable=False, index=True),
          Column("side", String, nullable=False),
          Column("action", String, nullable=False),
          Column("requested", Float, nullable=False),
          Column("filled", Float, nullable=False),
          Column("avg_price", Float),
          Column("fee", Float, nullable=False),
          Column("slippage", Float, nullable=False),
          Column("status", String, nullable=False),  # filled, partial, rejected
          Column("reason", String),
          Column("decision_ms", Ms, nullable=False),
          Column("execution_ms", Ms, nullable=False))

    Table("paper_trades", md,  # one row per closed round trip
          Column("trade_id", AutoId, primary_key=True, autoincrement=True),
          Column("session_id", ForeignKey("paper_sessions.session_id"), nullable=False, index=True),
          Column("ticker", String, nullable=False, index=True),
          Column("side", String, nullable=False),
          Column("quantity", Float, nullable=False),
          Column("entry_price", Float, nullable=False),
          Column("exit_price", Float, nullable=False),
          Column("entry_ms", Ms, nullable=False),
          Column("exit_ms", Ms, nullable=False),
          Column("pnl", Float, nullable=False),
          Column("fees", Float, nullable=False),
          Column("exit_reason", String, nullable=False))

    Table("paper_equity", md,
          Column("session_id", ForeignKey("paper_sessions.session_id"), primary_key=True),
          Column("ts_ms", Ms, primary_key=True),
          Column("cash", Float, nullable=False),
          Column("market_value", Float, nullable=False),
          Column("realized_pnl", Float, nullable=False),
          Column("unrealized_pnl", Float, nullable=False),
          sqlite_with_rowid=False)

    Table("paper_state", md,  # open positions, replaced on each change, for restart
          Column("session_id", ForeignKey("paper_sessions.session_id"), primary_key=True),
          Column("state", Text, nullable=False),
          Column("updated_ms", Ms, nullable=False))

    Table("backtest_runs", md,
          Column("run_id", String, primary_key=True),
          Column("created_ms", Ms, nullable=False, index=True),
          Column("strategy_name", String, nullable=False, index=True),
          Column("strategy_hash", String, nullable=False),
          Column("strategy_source", Text, nullable=False),
          Column("params", Text, nullable=False),
          Column("dataset_id", String, nullable=False),
          Column("dataset_spec", Text, nullable=False),
          Column("config", Text, nullable=False),  # capital, execution, fees, dates, seed
          Column("seed", Integer),
          Column("git_commit", String),
          Column("metrics", Text, nullable=False),
          Column("trades_hash", String, nullable=False))

    Table("backtest_trades", md,
          Column("run_id", ForeignKey("backtest_runs.run_id"), primary_key=True),
          Column("seq", Integer, primary_key=True),
          Column("ticker", String, nullable=False),
          Column("side", String, nullable=False),
          Column("quantity", Float, nullable=False),
          Column("entry_price", Float, nullable=False),
          Column("exit_price", Float, nullable=False),
          Column("entry_ms", Ms, nullable=False),
          Column("exit_ms", Ms, nullable=False),
          Column("pnl", Float, nullable=False),
          Column("fees", Float, nullable=False),
          Column("slippage", Float, nullable=False),
          Column("exit_reason", String, nullable=False),
          Column("category", String),
          sqlite_with_rowid=False)

    Table("optimization_runs", md,
          Column("opt_id", String, primary_key=True),
          Column("created_ms", Ms, nullable=False),
          Column("strategy_name", String, nullable=False),
          Column("strategy_hash", String, nullable=False),
          Column("config", Text, nullable=False),
          Column("status", String, nullable=False),
          Column("report", Text))

    Table("optimization_trials", md,
          Column("opt_id", ForeignKey("optimization_runs.opt_id"), primary_key=True),
          Column("fold", String, primary_key=True),
          Column("param_hash", String, primary_key=True),
          Column("params", Text, nullable=False),
          Column("metrics", Text, nullable=False),
          Column("objective", Float),
          Column("daily_returns", Text),
          sqlite_with_rowid=False)
    Table("evolution_runs", md,
          Column("run_id", String, primary_key=True),
          Column("created_ms", Ms, nullable=False),
          Column("status", String, nullable=False),
          Column("dataset_id", String, nullable=False),
          Column("config", Text, nullable=False),
          Column("report", Text))

    Table("evolution_strategies", md,  # every saved strategy, one row per (population, cycle, rank)
          Column("run_id", ForeignKey("evolution_runs.run_id"), primary_key=True),
          Column("population", String, primary_key=True),
          Column("stage", String, primary_key=True),  # cycle_001 ... final
          Column("rank", Integer, primary_key=True),
          Column("strategy_id", String, nullable=False, index=True),
          Column("population_type", String, nullable=False),
          Column("genome", Text, nullable=False),
          Column("origin", String, nullable=False),
          Column("parents", Text, nullable=False),
          Column("components", Text, nullable=False),
          Column("fitness", Float),
          Column("metrics", Text, nullable=False),
          sqlite_with_rowid=False)
    return md


VIEWS = {
    "observations_decimal": """
        CREATE VIEW observations_decimal AS
        SELECT m.ticker, o.ts_ms, o.source,
               o.yes_prob / 10000.0 AS yes_prob, (10000 - o.yes_prob) / 10000.0 AS no_prob,
               o.yes_bid / 10000.0 AS yes_bid, o.yes_ask / 10000.0 AS yes_ask,
               o.no_bid / 10000.0 AS no_bid, o.no_ask / 10000.0 AS no_ask,
               o.yes_ask / 10000.0 - o.yes_bid / 10000.0 AS spread,
               o.last_price / 10000.0 AS last_price,
               o.yes_bid_size / 100.0 AS yes_bid_size, o.yes_ask_size / 100.0 AS yes_ask_size,
               o.volume / 100.0 AS cumulative_volume, o.volume_delta / 100.0 AS incremental_volume,
               o.open_interest / 100.0 AS open_interest, o.status
        FROM observations o JOIN markets m USING (market_id)""",
    "trades_decimal": """
        CREATE VIEW trades_decimal AS
        SELECT m.ticker, t.ts_ms, t.yes_price / 10000.0 AS yes_price,
               (10000 - t.yes_price) / 10000.0 AS no_price, t.count / 100.0 AS contracts,
               CASE t.taker_side WHEN 1 THEN 'yes' WHEN 0 THEN 'no' END AS taker_side,
               t.is_block, t.source
        FROM trades t JOIN markets m USING (market_id)""",
    "market_overview": """
        CREATE VIEW market_overview AS
        SELECT m.market_id, m.ticker, m.event_ticker, e.title AS event_title, m.series_ticker,
               c.name AS category, s.subcategory, s.frequency, m.title, m.status,
               m.created_ms, m.open_ms, m.close_ms, m.expiration_ms, m.duration_s,
               r.result, r.winning_outcome, r.losing_outcome,
               r.settlement_value_e4 / 10000.0 AS final_settlement_price, r.settlement_ms,
               r.resolved_successfully,
               l.yes_prob / 10000.0 AS current_yes_prob, (10000 - l.yes_prob) / 10000.0 AS current_no_prob,
               l.volume / 100.0 AS cumulative_volume
        FROM markets m
        LEFT JOIN events e USING (event_ticker)
        LEFT JOIN series s ON s.series_ticker = m.series_ticker
        LEFT JOIN categories c ON c.category_id = s.category_id
        LEFT JOIN resolutions r ON r.market_id = m.market_id
        LEFT JOIN market_latest l ON l.market_id = m.market_id""",
}

TIME_SERIES_TABLES = {"observations": "ts_ms", "orderbook_snapshots": "ts_ms", "trades": "ts_ms"}


def view_statements() -> list[str]:
    return list(VIEWS.values())
