import asyncio
import uuid

import pytest
from conftest import client_for
from cryptography.hazmat.primitives.asymmetric import ed25519
from fake_kalshi import FakeStream, M15, market, series
from sqlalchemy import text

from kalshi_quant.fifteen_minute.collector.pipeline import FifteenMinutePipeline
from kalshi_quant.general_markets.collector.pipeline import GeneralPipeline
from kalshi_quant.shared.api.auth import Credentials
from kalshi_quant.shared.api.websocket import KalshiStream, LocalBook
from kalshi_quant.shared.collection.backfill import backfill_candles
from kalshi_quant.shared.collection.rollups import run_rollups
from kalshi_quant.shared.collection.validation import validate, validate_separation
from kalshi_quant.shared.config import Env
from kalshi_quant.shared.database.engine import Database, EnvironmentMismatch
from kalshi_quant.shared.database.maintenance import archive_table_day, backup, database_stats, integrity_check, log_growth, run_archive
from kalshi_quant.shared.database.writer import read_checkpoint
from kalshi_quant.shared.utilities.timeutil import now_ms


def rows(db, sql, **params):
    with db.connect() as conn:
        return conn.execute(text(sql), params).all()


def universe(fake):
    now = now_ms()
    fake.series = [series("KXBTC15M", "fifteen_min"), series("KXFED", "monthly", "Economics"),
                   series("KXGBPUSD15MTEST", "fifteen_min"), series("KXNBADRAFTTOP15", "annual", "Sports")]
    fake.events = [{"event_ticker": "KXBTC15M-E1", "series_ticker": "KXBTC15M", "title": "BTC Up",
                    "product_metadata": {"cadence": "fifteen_min"}},
                   {"event_ticker": "KXFED-E", "series_ticker": "KXFED", "title": "Fed"}]
    fake.markets = {
        "KXBTC15M-E1-00": market("KXBTC15M-E1-00", "KXBTC15M-E1", now - 60_000, now + M15 - 60_000),
        "KXBTC15M-E2-00": market("KXBTC15M-E2-00", "KXBTC15M-E2", now + M15, now + 2 * M15, status="initialized", bid="0", ask="0"),
        "KXFED-E-T1": market("KXFED-E-T1", "KXFED-E", now - 10**7, now + 10**8),
        "KXNBADRAFTTOP15-E-A": market("KXNBADRAFTTOP15-E-A", "KXNBADRAFTTOP15-E", now - 10**7, now + 10**8),
        "KXMVECROSSCATEGORY-S-X": market("KXMVECROSSCATEGORY-S-X", "KXMVECROSSCATEGORY-S", now - 1000, now + 10**8,
                                         mve_collection_ticker="KXMVECROSSCATEGORY-R"),
        "KXGBPUSD15MTEST-E-1": market("KXGBPUSD15MTEST-E-1", "KXGBPUSD15MTEST-E", now, now + M15),
    }
    for entry in fake.markets.values():
        entry["volume_24h_fp"] = "0.00"  # not in the hot set unless a test says so
    fake.books = {"KXBTC15M-E1-00": {"yes_dollars": [["0.4000", "10.00"]], "no_dollars": [["0.5800", "8.00"]]}}


def general(fake, config, db):
    return GeneralPipeline(config, db=db, client=client_for(fake, config), credentials=False)


def fifteen(fake, config, db):
    return FifteenMinutePipeline(config, db=db, client=client_for(fake, config), credentials=False)


async def sweep_both(fake, gc, gdb, fc, fdb):
    g, f = general(fake, gc, gdb), fifteen(fake, fc, fdb)
    for p in (g, f):
        await p.refresh_series(full=True)
    await g.full_sweep()
    await f.discover()
    await f.book_poll()
    await g.writer.flush()
    await f.writer.flush()
    return g, f


async def test_markets_land_only_in_their_environment(fake, general_config, general_db, fifteen_config, fifteen_db):
    universe(fake)
    await sweep_both(fake, general_config, general_db, fifteen_config, fifteen_db)
    assert {r[0] for r in rows(general_db, "SELECT ticker FROM markets")} == {"KXFED-E-T1", "KXNBADRAFTTOP15-E-A"}
    assert {r[0] for r in rows(fifteen_db, "SELECT ticker FROM markets")} == {"KXBTC15M-E1-00", "KXBTC15M-E2-00"}
    assert validate_separation(general_db, fifteen_db).passed


async def test_unopened_market_is_scheduled_without_observations(fake, general_config, general_db, fifteen_config, fifteen_db):
    universe(fake)
    _, f = await sweep_both(fake, general_config, general_db, fifteen_config, fifteen_db)
    assert "KXBTC15M-E2-00" in f.schedule and "KXBTC15M-E2-00" not in f.active_tickers()
    assert rows(fifteen_db, """SELECT count(*) FROM observations o JOIN markets m USING (market_id)
                               WHERE m.ticker = 'KXBTC15M-E2-00'""")[0][0] == 0


async def test_trades_routed_by_classification_and_deduplicated(fake, general_config, general_db, fifteen_config, fifteen_db):
    universe(fake)
    g, f = await sweep_both(fake, general_config, general_db, fifteen_config, fifteen_db)
    fake.add_trade("KXBTC15M-E1-00")
    fake.add_trade("KXFED-E-T1")
    fake.add_trade("KXMVECROSSCATEGORY-S-X")
    for _ in range(2):  # a repeat read must not duplicate
        await g.poll_trades()
        await f.poll_trades()
    await g.writer.flush()
    await f.writer.flush()
    assert rows(general_db, "SELECT count(*) FROM trades")[0][0] == 1
    assert rows(fifteen_db, "SELECT count(*) FROM trades")[0][0] == 1


async def test_trade_for_unknown_market_makes_stub_then_fills(fake, general_config, general_db):
    universe(fake)
    g = general(fake, general_config, general_db)
    await g.refresh_series(full=True)
    fake.add_trade("KXFED-E-T1")
    await g.poll_trades()
    await g.writer.flush()
    assert rows(general_db, "SELECT is_stub FROM markets WHERE ticker='KXFED-E-T1'") == [(1,)]
    await g.full_sweep()
    await g.writer.flush()
    assert rows(general_db, "SELECT is_stub, title FROM markets WHERE ticker='KXFED-E-T1'") == [(0, "KXFED-E-T1")]


async def test_unchanged_quote_is_suppressed_until_heartbeat(fake, general_config, general_db):
    universe(fake)
    general_config.collection.observation_heartbeat_seconds = 0.05
    g = general(fake, general_config, general_db)
    await g.refresh_series(full=True)
    await g.full_sweep()
    await asyncio.sleep(0.005)
    await g.full_sweep()
    await g.writer.flush()
    first = rows(general_db, "SELECT count(*) FROM observations")[0][0]
    await asyncio.sleep(0.06)
    await g.full_sweep()
    await g.writer.flush()
    assert first == 2 and rows(general_db, "SELECT count(*) FROM observations")[0][0] == 4


async def test_volume_delta_survives_restart(fake, fifteen_config, fifteen_db):
    universe(fake)
    f = fifteen(fake, fifteen_config, fifteen_db)
    await f.refresh_series(full=True)
    await f.discover()
    await f.book_poll()
    await f.writer.flush()
    fake.markets["KXBTC15M-E1-00"]["volume_fp"] = "130.00"
    restarted = fifteen(fake, fifteen_config, fifteen_db)  # new process, state from the database
    assert "KXBTC15M-E1-00" in restarted.schedule
    await restarted.refresh_series(full=True)
    await restarted.poll_active()
    await asyncio.sleep(0.002)
    await restarted.book_poll()
    await restarted.writer.flush()
    assert rows(fifteen_db, "SELECT volume, volume_delta FROM observations ORDER BY ts_ms")[-1] == (13000, 3000)


async def test_interrupted_full_sweep_resumes_from_cursor(fake, general_config, general_db):
    universe(fake)
    fake.page_size = 1
    g = general(fake, general_config, general_db)
    await g.refresh_series(full=True)

    original = g.ingest_markets
    seen = []

    async def stop_after_first(page, **kw):
        seen.append(page[0]["ticker"])
        result = await original(page, **kw)
        g.stop.set()
        return result

    g.ingest_markets = stop_after_first
    await g.full_sweep()
    await g.writer.flush()
    cursor = read_checkpoint(general_db, "full_sweep_cursor")
    assert cursor == "1"
    resumed = general(fake, general_config, general_db)
    await resumed.refresh_series(full=True)
    pages_before = len(fake.calls)
    await resumed.full_sweep()
    assert fake.calls[pages_before:].count("markets") == 3  # 4 open non-combo markets, 1 already read


async def test_hard_crash_keeps_committed_data(fake, general_config, general_db):
    """Unflushed rows may be lost; everything committed stays, and the checkpoint matches."""
    universe(fake)
    g = general(fake, general_config, general_db)
    await g.refresh_series(full=True)
    await g.full_sweep()
    await g.writer.flush()
    committed = rows(general_db, "SELECT count(*) FROM observations")[0][0]
    fake.markets["KXFED-E-T1"]["yes_bid_dollars"] = "0.4100"
    await g.full_sweep()  # buffered, never flushed: the process dies here
    del g
    restarted = general(fake, general_config, general_db)
    assert rows(general_db, "SELECT count(*) FROM observations")[0][0] == committed
    await restarted.refresh_series(full=True)
    await restarted.full_sweep()
    await restarted.writer.flush()
    assert rows(general_db, "SELECT count(*) FROM observations")[0][0] == committed + 1
    assert integrity_check(general_db)["ok"]


async def test_restart_records_gap(fake, general_config, general_db):
    universe(fake)
    g = general(fake, general_config, general_db)
    g._heartbeat("x", "ok")
    await asyncio.sleep(0.01)
    general(fake, general_config, general_db)._record_restart_gap()
    assert rows(general_db, "SELECT reason FROM collection_gaps") == [("collector not running",)]


async def test_resolution_recorded_and_validated(fake, fifteen_config, fifteen_db):
    universe(fake)
    f = fifteen(fake, fifteen_config, fifteen_db)
    await f.refresh_series(full=True)
    await f.discover()
    fake.markets["KXBTC15M-E1-00"].update(status="finalized", result="yes", settlement_value_dollars="1.0000",
                                          settlement_ts="2026-10-30T21:30:03.612926Z", expiration_value="76707.32",
                                          yes_bid_dollars="0.0000", yes_ask_dollars="1.0000")
    await f.poll_active()
    await f.writer.flush()
    assert rows(fifteen_db, "SELECT winning_outcome, losing_outcome, settlement_value_e4, resolved_successfully FROM resolutions") \
        == [("yes", "no", 10_000, 1)]
    assert f.schedule["KXBTC15M-E1-00"].finished
    assert not [c for c in validate(fifteen_db) if not c.passed and not c.warning]


async def test_full_validation_passes_on_clean_data(fake, general_config, general_db, fifteen_config, fifteen_db):
    universe(fake)
    await sweep_both(fake, general_config, general_db, fifteen_config, fifteen_db)
    for db in (general_db, fifteen_db):
        assert not [c for c in validate(db) if not c.passed and not c.warning]


async def test_validation_detects_corruption(fake, general_config, general_db):
    universe(fake)
    g = general(fake, general_config, general_db)
    await g.refresh_series(full=True)
    await g.full_sweep()
    await g.writer.flush()
    with general_db.begin() as conn:
        conn.execute(text("UPDATE observations SET yes_prob = 12000, no_bid = 1"))
    failed = {c.name for c in validate(general_db) if not c.passed}
    assert {"prob_range", "prob_no_is_complement"} <= failed


async def test_rollup_general_uses_earliest_reliable_evidence(fake, general_config, general_db):
    universe(fake)
    g = general(fake, general_config, general_db)
    await g.refresh_series(full=True)
    fake.markets["KXFED-E-T1"].update(yes_bid_dollars="0.0100", yes_ask_dollars="0.9900")  # unusable opening book
    await g.full_sweep()
    await g.writer.flush()
    await asyncio.sleep(0.01)
    fake.markets["KXFED-E-T1"].update(yes_bid_dollars="0.7000", yes_ask_dollars="0.7200", last_price_dollars="0.7100")
    await g.full_sweep()
    await g.writer.flush()
    run_rollups(general_db, Env.GENERAL)
    row = rows(general_db, """SELECT s.initial_yes_prob, s.initial_source, s.initial_projected_winner, s.initial_winner_prob,
                                     s.current_projected_winner FROM market_summary s JOIN markets m USING (market_id)
                              WHERE m.ticker = 'KXFED-E-T1'""")[0]
    assert row == (7100, "observation", "yes", 7100, "yes")


async def test_rollup_prefers_earlier_candle_after_backfill(fake, general_config, general_db):
    universe(fake)
    g = general(fake, general_config, general_db)
    await g.refresh_series(full=True)
    await g.full_sweep()
    await g.writer.flush()
    market_id = g.registry.ids["KXFED-E-T1"]
    open_ms = rows(general_db, "SELECT open_ms FROM markets WHERE market_id=:m", m=market_id)[0][0]
    with general_db.begin() as conn:
        conn.execute(general_db.t["candlesticks"].insert().values(
            market_id=market_id, period_minutes=1, end_ts_ms=open_ms + 60_000, yes_bid_close=2000, yes_ask_close=2200))
    run_rollups(general_db, Env.GENERAL)
    assert rows(general_db, "SELECT initial_yes_prob, initial_source, initial_projected_winner, initial_delay_s FROM market_summary WHERE market_id=:m",
                m=market_id) == [(2100, "candle", "no", 60)]


async def test_fifteen_minute_interval_grid_never_looks_ahead(fake, fifteen_config, fifteen_db):
    universe(fake)
    f = fifteen(fake, fifteen_config, fifteen_db)
    f.quotes_only_from_books = False  # feeds quotes directly to control timestamps
    await f.refresh_series(full=True)
    open_ms = 1_790_000_000_000
    fake.markets = {"KXBTC15M-E1-00": market("KXBTC15M-E1-00", "KXBTC15M-E1", open_ms, open_ms + M15, status="finalized", result="no")}
    await f.ingest_markets(list(fake.markets.values()), ts_ms=open_ms + 30_000)
    fake.markets["KXBTC15M-E1-00"].update(yes_bid_dollars="0.1000", yes_ask_dollars="0.1200", last_price_dollars="0.1100")
    await f.ingest_markets(list(fake.markets.values()), ts_ms=open_ms + 90_000)
    await f.writer.flush()
    run_rollups(fifteen_db, Env.FIFTEEN_MINUTE)
    grid = dict(rows(fifteen_db, "SELECT offset_seconds, yes_prob FROM market_interval_probs"))
    assert 0 not in grid and grid[60] == 4100 and grid[120] == 1100 and grid[900] == 1100


async def test_candle_backfill_stays_out_of_observations(fake, fifteen_config, fifteen_db):
    universe(fake)
    f = fifteen(fake, fifteen_config, fifteen_db)
    await f.refresh_series(full=True)
    open_ms = now_ms() - 3 * M15
    fake.markets = {"KXBTC15M-E1-00": market("KXBTC15M-E1-00", "KXBTC15M-E1", open_ms, open_ms + M15, status="finalized", result="yes")}
    await f.ingest_markets(list(fake.markets.values()), observe=False)
    await f.writer.flush()
    fake.candles["KXBTC15M-E1-00"] = [{
        "end_period_ts": (open_ms + 60_000) // 1000, "volume_fp": "5.00", "open_interest_fp": "1.00",
        "yes_bid": {"open_dollars": "0.4", "high_dollars": "0.5", "low_dollars": "0.3", "close_dollars": "0.45"},
        "yes_ask": {"open_dollars": "0.5", "high_dollars": "0.6", "low_dollars": "0.4", "close_dollars": "0.47"},
        "price": {"close_dollars": "0.46"}}]
    written = await backfill_candles(f.client, fifteen_db, fifteen_config, 1, None)
    assert written == 1
    assert rows(fifteen_db, "SELECT yes_bid_close, yes_ask_close, tier FROM candlesticks") == [(4500, 4700, "live")]
    assert rows(fifteen_db, "SELECT count(*) FROM observations")[0][0] == 0
    assert await backfill_candles(f.client, fifteen_db, fifteen_config, 1, None) == 0  # idempotent


def test_databases_refuse_the_wrong_environment(tmp_path):
    url = f"sqlite:///{tmp_path}/g.sqlite"
    Database("general", url).dispose()
    with pytest.raises(EnvironmentMismatch):
        Database("15m", url)


def test_schema_matches_migrations(general_db, fifteen_db):
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    for db in (general_db, fifteen_db):
        with db.connect() as conn:
            assert compare_metadata(MigrationContext.configure(conn), db.metadata) == []


def test_environment_specific_tables(general_db, fifteen_db):
    assert "market_interval_probs" in fifteen_db.t and "market_interval_probs" not in general_db.t
    assert "initial_projected_winner" in general_db.t["market_summary"].c
    assert "initial_projected_winner" not in fifteen_db.t["market_summary"].c


async def seeded(fake, config, db):
    universe(fake)
    g = general(fake, config, db)
    await g.refresh_series(full=True)
    await g.full_sweep()
    await g.writer.flush()


async def test_backup_is_restorable(fake, general_config, general_db, tmp_path):
    import gzip
    import shutil
    await seeded(fake, general_config, general_db)
    path = backup(general_db, tmp_path / "b")
    restored = tmp_path / "restored.sqlite"
    with gzip.open(path) as src, restored.open("wb") as out:
        shutil.copyfileobj(src, out)
    copy = Database("general", f"sqlite:///{restored}", migrate=False)
    assert copy.scalar("SELECT count(*) FROM markets") == general_db.scalar("SELECT count(*) FROM markets")
    copy.dispose()


async def test_archive_moves_rows_to_parquet_after_readback(fake, general_config, general_db, tmp_path):
    import duckdb
    await seeded(fake, general_config, general_db)
    day = now_ms() // 86_400_000 * 86_400_000
    before = general_db.scalar("SELECT count(*) FROM observations")
    moved = archive_table_day(general_db, "observations", day, tmp_path / "a")
    assert moved == before and general_db.scalar("SELECT count(*) FROM observations") == 0
    path = rows(general_db, "SELECT path FROM archives")[0][0]
    assert duckdb.sql(f"SELECT count(*) FROM read_parquet('{path}')").fetchall()[0][0] == before


async def test_run_archive_leaves_recent_data(fake, general_config, general_db):
    await seeded(fake, general_config, general_db)
    assert run_archive(general_db, general_config) == 0


async def test_stats_and_growth(fake, general_config, general_db):
    await seeded(fake, general_config, general_db)
    log_growth(general_db)
    stats = database_stats(general_db)
    assert stats["rows"]["markets"] == 2 and stats["file_bytes"] > 0


async def test_disk_guard_pauses_high_volume_writes(fake, general_config, general_db):
    await seeded(fake, general_config, general_db)
    g = general(fake, general_config, general_db)
    general_config.shared.storage.min_free_disk_gb = 10**9
    await g.guard_disk()
    assert "observations" in g.writer.paused_tables
    g.writer.add("observations", [{"market_id": 1}])
    assert g.writer.pending == 0
    general_config.shared.storage.min_free_disk_gb = 0.001
    await g.guard_disk()
    assert not g.writer.paused_tables


def test_local_book_applies_deltas():
    book = LocalBook()
    book.apply_snapshot({"yes_dollars_fp": [["0.4000", "10.00"]], "no_dollars_fp": [["0.5500", "5.00"]]})
    book.apply_delta({"side": "yes", "price_dollars": "0.4100", "delta_fp": "3.00"})
    book.apply_delta({"side": "no", "price_dollars": "0.5500", "delta_fp": "-5.00"})
    snap = book.snapshot()
    assert snap.yes_bid == 4100 and snap.no == [] and snap.yes_ask is None


def credentials():
    return Credentials("key-id", ed25519.Ed25519PrivateKey.generate())


async def run_stream(server, handler, gaps, channels=("ticker", "trade")):
    stop = asyncio.Event()
    stream = KalshiStream(server.url, credentials(), handler, lambda a, b, r: gaps.append(r), channels)
    task = asyncio.create_task(stream.run(stop))
    await stream.connected.wait()
    return stream, stop, task


async def test_stream_handshake_is_signed_and_subscribes():
    async with FakeStream() as server:
        got, gaps = [], []

        async def handler(m):
            got.append(m)

        stream, stop, task = await run_stream(server, handler, gaps)
        await server.wait_for(lambda: server.received)
        assert server.received[0]["cmd"] == "subscribe" and server.received[0]["params"]["channels"] == ["ticker", "trade"]
        assert server.headers[0]["kalshi-access-key"] == "key-id" and "kalshi-access-signature" in server.headers[0]
        await server.send({"type": "ticker", "sid": 1, "msg": {"market_ticker": "A"}})
        await server.wait_for(lambda: got)
        stop.set()
        await task


async def test_stream_reconnects_and_records_gap():
    async with FakeStream() as server:
        gaps = []

        async def handler(_):
            pass

        stream, stop, task = await run_stream(server, handler, gaps)
        await server.wait_for(lambda: server.connections)
        await server.connections[-1].close()
        await server.wait_for(lambda: len(server.connections) == 2, timeout=5)
        await server.wait_for(lambda: gaps)
        assert gaps == ["websocket disconnected"] and stream.reconnects == 1
        stop.set()
        await task


async def test_stream_sequence_gap_requests_snapshot_and_skips_delta():
    async with FakeStream() as server:
        got, gaps = [], []

        async def handler(m):
            got.append(m)

        stream, stop, task = await run_stream(server, handler, gaps, ("ticker",))
        await stream.set_book_tickers({"A"})
        await server.wait_for(lambda: len(server.received) == 2)
        snapshot = {"market_ticker": "A", "market_id": "x", "clock": 1, "yes_dollars_fp": [["0.4000", "10.00"]]}
        await server.send({"type": "orderbook_snapshot", "sid": 7, "seq": 1, "msg": snapshot})
        delta = {"market_ticker": "A", "market_id": "x", "clock": 2, "price_dollars": "0.4000", "delta_fp": "5.00", "side": "yes"}
        await server.send({"type": "orderbook_delta", "sid": 7, "seq": 2, "msg": delta})
        await server.wait_for(lambda: len(got) == 2)
        assert stream.books["A"].yes == {4000: 1500}
        await server.send({"type": "orderbook_delta", "sid": 7, "seq": 9, "msg": delta})
        await server.wait_for(lambda: any(c["cmd"] == "update_subscription" for c in server.received))
        assert server.received[-1]["params"]["action"] == "get_snapshot"
        assert stream.books["A"].yes == {4000: 1500} and gaps
        stop.set()
        await task


async def test_ws_messages_reach_the_right_tables(fake, fifteen_config, fifteen_db):
    universe(fake)
    f = fifteen(fake, fifteen_config, fifteen_db)
    await f.refresh_series(full=True)
    await f.discover()
    await f.book_poll()
    await f.on_ws_message({"type": "ticker", "msg": {
        "market_ticker": "KXBTC15M-E1-00", "price_dollars": "0.5000", "yes_bid_dollars": "0.4900", "yes_ask_dollars": "0.5100",
        "volume_fp": "200.00", "open_interest_fp": "60.00", "yes_bid_size_fp": "1.00", "yes_ask_size_fp": "2.00", "ts_ms": now_ms() + 5}})
    await f.on_ws_message({"type": "ticker", "msg": {"market_ticker": "KXFED-E-T1", "ts_ms": now_ms()}})  # not ours
    await f.on_ws_message({"type": "trade", "msg": {
        "trade_id": str(uuid.uuid4()), "market_ticker": "KXBTC15M-E1-00", "yes_price_dollars": "0.5000", "count_fp": "3.00",
        "taker_outcome_side": "no", "ts_ms": now_ms(), "is_block_trade": False}})
    await f.writer.flush()
    assert rows(fifteen_db, "SELECT yes_bid, source, volume_delta FROM observations ORDER BY ts_ms DESC LIMIT 1") == [(4900, 2, 10000)]
    assert rows(fifteen_db, "SELECT yes_price, taker_side, source FROM trades") == [(5000, 0, 2)]


async def test_lifecycle_event_triggers_rest_ingest(fake, fifteen_config, fifteen_db):
    universe(fake)
    f = fifteen(fake, fifteen_config, fifteen_db)
    await f.refresh_series(full=True)
    await f.on_ws_message({"type": "market_lifecycle_v2", "msg": {"market_ticker": "KXBTC15M-E1-00", "event_type": "created"}})
    assert "KXBTC15M-E1-00" in f.registry.ids


async def test_fifteen_minute_quotes_come_from_books_not_cached_markets(fake, fifteen_config, fifteen_db):
    """/markets lags by about 10 seconds; a row must carry the live book's quote."""
    universe(fake)
    f = fifteen(fake, fifteen_config, fifteen_db)
    await f.refresh_series(full=True)
    await f.discover()
    await f.poll_active()
    await f.writer.flush()
    assert rows(fifteen_db, "SELECT count(*) FROM observations")[0][0] == 0
    await f.book_poll()
    await f.writer.flush()
    assert rows(fifteen_db, "SELECT yes_bid, yes_ask, source FROM observations") == [(4000, 4200, 3)]


async def test_hot_markets_get_live_quotes_and_depth_only_for_the_top(fake, general_config, general_db):
    universe(fake)
    general_config.collection.orderbook_market_count = 1
    fake.books = {"KXFED-E-T1": {"yes_dollars": [["0.3000", "5.00"]], "no_dollars": [["0.6000", "5.00"]]},
                  "KXNBADRAFTTOP15-E-A": {"yes_dollars": [["0.2000", "5.00"]], "no_dollars": [["0.7000", "5.00"]]}}
    fake.markets["KXFED-E-T1"]["volume_24h_fp"] = "100.00"
    fake.markets["KXNBADRAFTTOP15-E-A"]["volume_24h_fp"] = "90.00"
    g = general(fake, general_config, general_db)
    await g.refresh_series(full=True)
    await g.full_sweep()
    assert g.hot[0] == "KXFED-E-T1" and g.book_quoted == set(g.hot)
    await g.hot_sweep()
    await g.writer.flush()
    assert rows(general_db, """SELECT m.ticker, o.yes_bid FROM observations o JOIN markets m USING (market_id)
                               WHERE o.source = 3 ORDER BY m.ticker""") == [("KXFED-E-T1", 3000), ("KXNBADRAFTTOP15-E-A", 2000)]
    assert rows(general_db, "SELECT count(*) FROM orderbook_snapshots")[0][0] == 1
    await asyncio.sleep(0.002)
    await g.full_sweep()  # the cached quote must not overwrite the live one
    await g.writer.flush()
    assert rows(general_db, "SELECT count(*) FROM observations WHERE source = 1 AND market_id = :m", m=g.registry.ids["KXFED-E-T1"])[0][0] == 1


async def test_cumulative_volume_never_steps_back_on_a_cached_read(fake, general_config, general_db):
    universe(fake)
    g = general(fake, general_config, general_db)
    await g.refresh_series(full=True)
    await g.full_sweep()
    fake.markets["KXFED-E-T1"].update(volume_fp="90.00", yes_bid_dollars="0.3900")
    await asyncio.sleep(0.002)
    await g.full_sweep()
    await g.writer.flush()
    assert rows(general_db, """SELECT o.volume, o.volume_delta FROM observations o JOIN markets m USING (market_id)
                               WHERE m.ticker = 'KXFED-E-T1' ORDER BY ts_ms""") == [(10000, None), (10000, 0)]
    assert g.tracker.volume_regressions == 1


async def test_resumed_sweep_keeps_existing_hot_ranking(fake, general_config, general_db):
    universe(fake)
    fake.page_size = 1
    fake.markets["KXNBADRAFTTOP15-E-A"]["volume_24h_fp"] = "90.00"
    g = general(fake, general_config, general_db)
    await g.refresh_series(full=True)
    g.hot = ["ALREADY-HOT"]
    g.writer.checkpoint("full_sweep_cursor", "2")
    await g.writer.flush()
    await g.full_sweep()
    assert "ALREADY-HOT" in g.hot
