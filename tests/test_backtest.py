import json
import random

import numpy as np
import pytest
import synthetic

from kalshi_quant.shared.backtest.dataset import DatasetSpec, load_dataset
from kalshi_quant.shared.backtest.engine import BacktestSettings, replay_run, run_backtest, save_run, trades_hash
from kalshi_quant.shared.backtest.metrics import compute_metrics, daily_returns, drawdown
from kalshi_quant.shared.config import ExecutionConfig, load_config
from kalshi_quant.shared.database.maintenance import archive_table_day
from kalshi_quant.shared.execution.portfolio import ClosedTrade
from kalshi_quant.shared.strategy.base import YES, NO, Param, Risk, Signal, Strategy
from sqlalchemy import text


class Buyer(Strategy):
    name = "buyer"
    parameters = {"threshold": Param(0.45, 0.3, 0.6, 0.05)}
    risk = Risk(position_fraction=0.05, max_market_exposure=0.2, max_portfolio_exposure=1.0)
    history_seconds = 600

    def should_enter(self, view):
        return Signal(YES) if view.probability is not None and view.probability < self.p.threshold else None


class Cheater(Strategy):
    """Tries every route it can find to the outcome, and records what it could reach."""
    name = "cheater"
    risk = Risk(position_fraction=0.01)
    seen_after: list = []
    leaks: list = []

    def should_enter(self, view):
        Cheater.seen_after.append(view.now_ms)
        for attribute in ("result", "settlement_value", "winning_outcome", "close_ms"):
            if hasattr(view.info, attribute) or hasattr(view.tick, attribute) or hasattr(view, attribute):
                Cheater.leaks.append(attribute)
        return None


def settings(capital=10_000, latency=1000, **execution):
    return BacktestSettings(capital, latency, 900, ExecutionConfig(**execution))


def dataset(db, **kw):
    return load_dataset(db, DatasetSpec(environment=db.env.value, **kw))


def test_replay_is_chronological_and_dataset_ordered(general_db):
    synthetic.seed(general_db, days=5)
    data = dataset(general_db)
    assert len(data) == 5 * 4 * 60 and (np.diff(data.ts_ms) >= 0).all()


def test_strategy_cannot_reach_outcomes(general_db):
    synthetic.seed(general_db, days=3)
    Cheater.seen_after, Cheater.leaks = [], []
    run_backtest(Cheater(), dataset(general_db), settings())
    assert not Cheater.leaks and Cheater.seen_after == sorted(Cheater.seen_after)


def test_changing_outcomes_cannot_change_decisions(general_db, fifteen_db):
    """The definitive look-ahead test: flip every outcome; entries must be identical."""
    synthetic.seed(general_db, days=10)
    original = run_backtest(Buyer(), dataset(general_db), settings())
    with general_db.begin() as conn:
        conn.execute(text("""UPDATE resolutions SET settlement_value_e4 = 10000 - settlement_value_e4,
                             result = CASE result WHEN 'yes' THEN 'no' ELSE 'yes' END"""))
    flipped = run_backtest(Buyer(), dataset(general_db), settings())
    entries = lambda r: [(t.ticker, t.entry_ms, t.entry_price, t.quantity) for t in r.trades]  # noqa: E731
    first, second = entries(original), entries(flipped)
    assert {x[:3] for x in first} == {x[:3] for x in second}
    assert original.metrics["total_pnl"] != flipped.metrics["total_pnl"]


def test_changing_the_future_cannot_change_the_past(general_db):
    """Rewrite all data after a point; everything decided before it must be unchanged."""
    tickers = synthetic.seed(general_db, days=10)
    cut = synthetic.BASE + 5 * synthetic.DAY
    before = run_backtest(Buyer(), dataset(general_db), settings())
    with general_db.begin() as conn:
        conn.execute(text("UPDATE observations SET yes_bid = 100, yes_ask = 300, yes_prob = 200, no_bid = 9700, no_ask = 9900 WHERE ts_ms >= :c"), {"c": cut})
    after = run_backtest(Buyer(), dataset(general_db), settings())
    early = lambda r: [(t.ticker, t.entry_ms, t.exit_ms, round(t.pnl, 6)) for t in r.trades if t.exit_ms < cut]  # noqa: E731
    assert early(before) == early(after) and early(before)


def test_settlement_applied_only_at_real_settlement_time(general_db):
    synthetic.seed(general_db, days=4)
    data = dataset(general_db)
    result = run_backtest(Buyer(), data, settings())
    settled = [t for t in result.trades if t.exit_reason == "settlement"]
    assert settled and all(t.exit_ms == data.settlements[t.ticker].ts_ms for t in settled)


def test_fill_uses_quote_after_latency(general_db):
    synthetic.seed(general_db, days=2)
    data = dataset(general_db)
    t = run_backtest(Buyer(), data, settings(latency=30_000)).trades[0]
    index = data.tickers.index(t.ticker)
    rows = np.nonzero(data.ticker_index == index)[0]
    fill_row = rows[data.ts_ms[rows] == t.entry_ms][0]
    assert t.entry_price == pytest.approx(data.values["yes_ask"][fill_row])
    first_signal = rows[data.values["yes_bid"][rows] + 0.01 < 0.45][0]
    assert t.entry_ms >= data.ts_ms[first_signal] + 30_000


def test_cash_reconciles(general_db):
    synthetic.seed(general_db, days=10)
    result = run_backtest(Buyer(), dataset(general_db), settings())
    assert result.metrics["final_equity"] == pytest.approx(10_000 + sum(t.pnl for t in result.trades))


def test_fees_reduce_pnl(general_db):
    synthetic.seed(general_db, days=10)
    data = dataset(general_db)
    with_fees = run_backtest(Buyer(), data, settings())
    free = settings()
    free.fee_coefficient = 0
    no_fees = run_backtest(Buyer(), data, free)
    assert with_fees.metrics["fees"] > 0 and no_fees.metrics["fees"] == 0
    assert with_fees.metrics["total_pnl"] < no_fees.metrics["total_pnl"]


def test_slippage_setting_hurts(general_db):
    synthetic.seed(general_db, days=10)
    data = dataset(general_db)
    assert run_backtest(Buyer(), data, settings(slippage_ticks=2)).metrics["total_pnl"] < run_backtest(Buyer(), data, settings()).metrics["total_pnl"]


def test_identical_runs_are_identical(general_db):
    synthetic.seed(general_db, days=10)
    data = dataset(general_db)
    assert trades_hash(run_backtest(Buyer(), data, settings()).trades) == trades_hash(run_backtest(Buyer(), data, settings()).trades)


def test_saved_run_replays_exactly(general_db, tmp_path):
    synthetic.seed(general_db, days=10)
    data = dataset(general_db)
    strategy = Buyer({"threshold": 0.40})
    result = run_backtest(strategy, data, settings())
    run_id = save_run(general_db, strategy, data, settings(), result)
    replayed = replay_run(general_db, run_id, tmp_path)
    assert replayed["trades_match"] and replayed["dataset_matches"]
    row = general_db.scalar("SELECT count(*) FROM backtest_trades WHERE run_id = :r", r=run_id)
    assert row == len(result.trades)


def test_replay_detects_changed_data(general_db, tmp_path):
    synthetic.seed(general_db, days=5)
    data = dataset(general_db)
    run_id = save_run(general_db, Buyer(), data, settings(), run_backtest(Buyer(), data, settings()))
    with general_db.begin() as conn:
        conn.execute(text("UPDATE observations SET yes_bid = yes_bid - 10 WHERE ts_ms % 7 = 0"))
    assert not replay_run(general_db, run_id, tmp_path)["dataset_matches"]


def test_archived_rows_replay_like_live_rows(general_db, tmp_path):
    synthetic.seed(general_db, days=4)
    before = run_backtest(Buyer(), dataset(general_db), settings())
    archive_table_day(general_db, "observations", synthetic.BASE, tmp_path / "a")
    data = dataset(general_db)
    after = run_backtest(Buyer(), data, settings())
    assert len(data) == 4 * 4 * 60 and trades_hash(before.trades) == trades_hash(after.trades)


def test_candle_fallback_stamps_period_end_and_uses_volume_caps(fifteen_db):
    t = fifteen_db.t
    with fifteen_db.begin() as conn:
        conn.execute(t["categories"].insert().values(name="Crypto"))
        conn.execute(t["series"].insert().values(series_ticker="S", frequency="fifteen_min", category_id=1, fee_multiplier=1.0, first_seen_ms=1, updated_ms=1))
        conn.execute(t["events"].insert().values(event_ticker="S-E", series_ticker="S", is_stub=False))
        open_ms = 1_780_000_000_000
        market_id = conn.execute(t["markets"].insert().values(
            ticker="S-E-1", event_ticker="S-E", series_ticker="S", open_ms=open_ms, close_ms=open_ms + 900_000, first_seen_close_ms=open_ms + 900_000,
            first_seen_ms=open_ms - 1, last_seen_ms=open_ms, status="finalized")).inserted_primary_key[0]
        for minute in range(1, 11):
            conn.execute(t["candlesticks"].insert().values(
                market_id=market_id, period_minutes=1, end_ts_ms=open_ms + minute * 60_000, yes_bid_close=4000, yes_ask_close=4200,
                price_close=4100, volume=10_000, open_interest=100, tier="live"))
        conn.execute(t["resolutions"].insert().values(market_id=market_id, result="yes", settlement_value_e4=10_000,
                                                      settlement_ms=open_ms + 901_000, resolved_successfully=True, recorded_ms=1))
    data = dataset(fifteen_db)
    assert data.is_candle.all() and data.ts_ms[0] == open_ms + 60_000
    assert data.warnings["candle_only_markets"] == 1
    assert data.values["volume"][-1] == 1000 and np.isnan(data.values["yes_bid_size"]).all()
    result = run_backtest(Buyer({"threshold": 0.5}), data, settings(capital=1000))
    trade = result.trades[0]
    assert trade.entry_price == pytest.approx(0.43) and trade.quantity == 10  # 1 tick worse, 10% of volume


def test_orderbook_at_a_row_is_never_from_the_future(general_db):
    tickers = synthetic.seed(general_db, days=1)
    market_id = general_db.scalar("SELECT market_id FROM markets WHERE ticker = :t", t=tickers[0])
    open_ms = general_db.scalar("SELECT open_ms FROM markets WHERE market_id = :m", m=market_id)
    with general_db.begin() as conn:
        conn.execute(general_db.t["orderbook_snapshots"].insert().values(
            market_id=market_id, ts_ms=open_ms + 300_000, source=3, levels=json.dumps({"y": [[4000, 100]], "n": [[5000, 100]]})))
    data = dataset(general_db)
    rows = np.nonzero(data.ticker_index == data.tickers.index(tickers[0]))[0]
    with_book = [int(data.ts_ms[r]) for r in rows if int(r) in data.levels]
    assert with_book and min(with_book) >= open_ms + 300_000


def test_unresolved_position_is_marked_and_flagged(general_db):
    synthetic.seed(general_db, days=2)
    with general_db.begin() as conn:
        conn.execute(text("DELETE FROM resolutions"))
    data = dataset(general_db)
    result = run_backtest(Buyer(), data, settings())
    assert result.metrics["data_quality"]["unsettled_positions_at_end"] > 0
    assert data.warnings["markets_unresolved"] == 8
    assert all(t.exit_reason in ("end_of_data",) for t in result.trades)


def test_thin_market_reports_liquidity_rejections(general_db):
    synthetic.seed(general_db, days=3)
    with general_db.begin() as conn:
        conn.execute(text("UPDATE observations SET yes_ask_size = 0, yes_bid_size = 0"))
    metrics = run_backtest(Buyer(), dataset(general_db), settings()).metrics
    assert metrics["data_quality"]["rejected_insufficient_liquidity"] > 0 and metrics["trades"] == 0


def test_filters_by_category_series_and_date(general_db):
    synthetic.seed(general_db, days=4)
    assert not len(dataset(general_db, categories=["Politics"]))
    assert len(dataset(general_db, series=["SERIES"])) == 4 * 4 * 60
    limited = dataset(general_db, start_ms=synthetic.BASE + synthetic.DAY, end_ms=synthetic.BASE + 2 * synthetic.DAY)
    assert limited.ts_ms.min() >= synthetic.BASE + synthetic.DAY and limited.ts_ms.max() < synthetic.BASE + 2 * synthetic.DAY


def closed(pnl, ticker="T", side="yes", entry=0, exit=1000):
    return ClosedTrade(ticker, side, 10, 0.5, 0.6, entry, exit, pnl, 0.1, 0.0, "settlement")


def test_metrics_values():
    trades = [closed(10), closed(20), closed(-5), closed(-5), closed(-5), closed(10)]
    ts = np.array([0, 86_400_000, 2 * 86_400_000], dtype=np.int64)
    m = compute_metrics(trades, 1000, ts, np.array([1000.0, 1100.0, 1025.0]), np.array([0.5, 0.5, 0.5]), {"T": "Crypto"}, 3.0)
    assert m["trades"] == 6 and m["win_rate"] == 0.5 and m["profit_factor"] == pytest.approx(40 / 15)
    assert m["average_win"] == pytest.approx(40 / 3) and m["average_loss"] == -5
    assert m["longest_win_streak"] == 2 and m["longest_loss_streak"] == 3
    assert m["max_drawdown"] == pytest.approx(75 / 1100) and m["total_return"] == pytest.approx(0.025)
    assert m["estimated_slippage"] == 3.0 and m["by_category"]["Crypto"]["trades"] == 6


def test_sharpe_withheld_on_short_samples():
    ts = np.arange(10) * 86_400_000
    m = compute_metrics([], 1000, ts, 1000 + np.arange(10.0), np.zeros(10), {}, 0)
    assert m["sharpe"] is None and "days" in m["risk_ratio_note"]


def test_sharpe_computed_on_long_samples():
    rng = random.Random(1)
    equity = [1000.0]
    for _ in range(99):
        equity.append(equity[-1] * (1 + rng.gauss(0.001, 0.01)))
    m = compute_metrics([], 1000, np.arange(100) * 86_400_000, np.array(equity), np.zeros(100), {}, 0)
    assert m["sharpe"] is not None and m["sortino"] is not None


def test_profit_factor_undefined_without_losses():
    m = compute_metrics([closed(5)], 1000, np.array([0]), np.array([1005.0]), np.zeros(1), {}, 0)
    assert m["profit_factor"] is None and m["profit_factor_note"]


def test_drawdown_helper():
    assert drawdown(np.array([100.0, 120, 60, 130])) == pytest.approx(0.5)


def test_daily_returns_use_end_of_day_equity():
    ts = np.array([0, 1000, 86_400_000, 86_401_000], dtype=np.int64)
    out = daily_returns(ts, np.array([100.0, 110, 110, 121]))
    assert out == pytest.approx([0.1, 0.1])


def test_mirrored_positions_sum_to_fees():
    """YES and NO on the same market cannot both win: gross P&L nets to zero."""
    from kalshi_quant.shared.execution.fees import FeeSchedule
    from kalshi_quant.shared.execution.fills import Fill, Order
    from kalshi_quant.shared.execution.portfolio import Portfolio
    p = Portfolio(100)
    q = Portfolio(100)
    p.apply_fill(Fill(Order("T", "yes", "buy", 10, 0, 0), 10, 0.40, 0, 0, "filled", "", 0))
    q.apply_fill(Fill(Order("T", "no", "buy", 10, 0, 0), 10, 0.60, 0, 0, "filled", "", 0))
    assert p.settle("T", 1.0, 1).pnl + q.settle("T", 1.0, 1).pnl == pytest.approx(0)


def test_candle_gaps_are_not_stale_and_close_candle_is_untradable(fifteen_db):
    t = fifteen_db.t
    open_ms = 1_780_000_000_000
    with fifteen_db.begin() as conn:
        conn.execute(t["events"].insert().values(event_ticker="S-E", is_stub=False))
        market_id = conn.execute(t["markets"].insert().values(
            ticker="S-E-1", event_ticker="S-E", open_ms=open_ms, close_ms=open_ms + 300_000, first_seen_close_ms=open_ms + 300_000,
            first_seen_ms=open_ms - 1, last_seen_ms=open_ms, status="finalized")).inserted_primary_key[0]
        for minute in range(1, 6):
            conn.execute(t["candlesticks"].insert().values(
                market_id=market_id, period_minutes=1, end_ts_ms=open_ms + minute * 60_000, yes_bid_close=4000, yes_ask_close=4200,
                price_close=4100, volume=100_000, open_interest=100, tier="live"))
    data = dataset(fifteen_db)
    assert data.status[-1] == 4 and (data.status[:-1] == 2).all()
    short_age = BacktestSettings(1000, 1000, 10, ExecutionConfig())  # the 15-minute default
    assert run_backtest(Buyer({"threshold": 0.5}), data, short_age).metrics["data_quality"]["rejected_execution"] == {}
