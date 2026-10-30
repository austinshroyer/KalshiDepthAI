import asyncio
import json
from pathlib import Path

import pytest
from conftest import client_for
from fake_kalshi import M15, market, series
from sqlalchemy import text

from kalshi_quant.shared.paper import commands
from kalshi_quant.shared.paper.engine import PaperEngine
from kalshi_quant.shared.strategy.base import YES, Risk, Signal, Strategy, Universe
from kalshi_quant.shared.utilities.timeutil import now_ms


class Buy(Strategy):
    name = "buy_all"
    risk = Risk(position_fraction=0.05, take_profit=0.20, max_market_exposure=0.5)

    def should_enter(self, view):
        return Signal(YES) if view.yes_ask is not None else None


class Other(Buy):
    name = "other"
    risk = Risk(position_fraction=0.10, max_market_exposure=0.5)


def setup(fake):
    now = now_ms()
    fake.series = [series("KXBTC15M", "fifteen_min"), series("KXFED", "monthly")]
    fake.markets = {
        "KXBTC15M-E-1": market("KXBTC15M-E-1", "KXBTC15M-E", now - 60_000, now + M15, status="active"),
        "KXFED-E-1": market("KXFED-E-1", "KXFED-E", now - 60_000, now + 10**8, status="active"),
    }
    fake.books = {"KXBTC15M-E-1": {"yes_dollars": [["0.4000", "100.00"]], "no_dollars": [["0.5800", "100.00"]]}}


def engine(fake, config, db, strategies=None, resume=False, capital=100):
    config.backtest.latency_ms = 0
    return PaperEngine(config, strategies or [(Buy(), Path("b.py"))], capital, db, client_for(fake, config), False, resume)


async def cycle(e, gap=0.002):
    await asyncio.sleep(gap)
    await e.poll_markets()
    await e.poll_books()


async def test_only_this_environments_markets_are_tracked(fake, fifteen_config, fifteen_db, general_config, general_db):
    setup(fake)
    f = engine(fake, fifteen_config, fifteen_db)
    await f.discover()
    g = engine(fake, general_config, general_db)
    await g.discover()
    assert list(f.tracked) == ["KXBTC15M-E-1"] and list(g.tracked) == ["KXFED-E-1"]


async def test_simulated_entry_uses_book_and_persists(fake, fifteen_config, fifteen_db):
    setup(fake)
    e = engine(fake, fifteen_config, fifteen_db)
    await e.discover()
    await cycle(e)
    await cycle(e)
    position = e.sessions[0].runner.portfolio.positions["KXBTC15M-E-1"]
    assert position.avg_price == pytest.approx(0.42) and position.side == "yes"
    assert [p["ticker"] for p in commands.positions(fifteen_db)] == ["KXBTC15M-E-1"]
    assert fifteen_db.scalar("SELECT count(*) FROM paper_orders WHERE status = 'filled'") == 1


async def test_settlement_closes_position(fake, fifteen_config, fifteen_db):
    setup(fake)
    e = engine(fake, fifteen_config, fifteen_db)
    await e.discover()
    await cycle(e)
    await cycle(e)
    fake.markets["KXBTC15M-E-1"].update(status="finalized", result="yes", settlement_value_dollars="1.0000")
    await cycle(e)
    trade = commands.trades(fifteen_db)[0]
    assert trade["exit_reason"] == "settlement" and trade["exit_price"] == 1 and trade["pnl"] > 0
    assert not commands.positions(fifteen_db)


async def test_determined_is_not_settled(fake, fifteen_config, fifteen_db):
    setup(fake)
    e = engine(fake, fifteen_config, fifteen_db)
    await e.discover()
    await cycle(e)
    await cycle(e)
    fake.markets["KXBTC15M-E-1"].update(status="determined", result="yes")  # may still be disputed
    await cycle(e)
    assert commands.positions(fifteen_db)


async def test_strategies_have_independent_portfolios(fake, fifteen_config, fifteen_db):
    setup(fake)
    e = engine(fake, fifteen_config, fifteen_db, [(Buy(), Path("b.py")), (Other(), Path("o.py"))])
    await e.discover()
    await cycle(e)
    await cycle(e)
    first, second = [s.runner.portfolio for s in e.sessions]
    assert first.positions["KXBTC15M-E-1"].quantity != second.positions["KXBTC15M-E-1"].quantity
    assert first is not second and {r["strategy"] for r in commands.performance(fifteen_db)} == {"buy_all", "other"}


async def test_restart_resumes_portfolio(fake, fifteen_config, fifteen_db):
    setup(fake)
    e = engine(fake, fifteen_config, fifteen_db)
    await e.discover()
    await cycle(e)
    await cycle(e)
    cash = e.sessions[0].runner.portfolio.cash
    resumed = engine(fake, fifteen_config, fifteen_db, resume=True)
    assert resumed.sessions[0].session_id == e.sessions[0].session_id
    assert resumed.sessions[0].runner.portfolio.cash == cash
    await resumed.discover()
    await cycle(resumed)
    await cycle(resumed)
    assert resumed.sessions[0].runner.portfolio.positions["KXBTC15M-E-1"].entries == 1  # no double entry


async def test_stop_request_ends_engine(fake, fifteen_config, fifteen_db):
    setup(fake)
    e = engine(fake, fifteen_config, fifteen_db)
    fifteen_config.paper.poll_seconds = 0.01
    task = asyncio.create_task(e.run(10))
    await asyncio.sleep(0.2)
    commands.request_stop(fifteen_db, e.sessions[0].session_id)
    await asyncio.wait_for(task, 8)
    assert fifteen_db.scalar("SELECT status FROM paper_sessions") == "stopped"


async def test_stale_tick_ignored(fake, fifteen_config, fifteen_db):
    from kalshi_quant.shared.data_models.tick import Tick
    setup(fake)
    e = engine(fake, fifteen_config, fifteen_db)
    e.dispatch(Tick("X", 100, 0.4, 0.5))
    e.dispatch(Tick("X", 50, 0.4, 0.5))
    assert e.last_ts["X"] == 100


async def test_fill_uses_live_book_not_cached_market_quote(fake, fifteen_config, fifteen_db):
    setup(fake)
    fake.markets["KXBTC15M-E-1"].update(yes_bid_dollars="0.1000", yes_ask_dollars="0.1200")  # stale cache
    e = engine(fake, fifteen_config, fifteen_db)
    await e.discover()
    await cycle(e)
    await cycle(e)
    assert e.sessions[0].runner.portfolio.positions["KXBTC15M-E-1"].avg_price == pytest.approx(0.42)
