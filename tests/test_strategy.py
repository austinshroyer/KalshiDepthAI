import pytest

from kalshi_quant.shared.config import ExecutionConfig, load_config
from kalshi_quant.shared.data_models.tick import MarketInfo, Tick
from kalshi_quant.shared.execution.fees import FeeSchedule
from kalshi_quant.shared.execution.portfolio import Portfolio
from kalshi_quant.shared.strategy.base import MarketView, Param, Risk, Signal, Strategy, Universe
from kalshi_quant.shared.strategy.loader import load_strategy
from kalshi_quant.shared.strategy.runner import StrategyRunner

INFO = MarketInfo("T", "E", "S", "Crypto", "t", "e", 0, 900_000)


def tick(ts, bid=0.40, ask=0.42, volume=100, status=2):
    return Tick("T", ts, bid, ask, volume=volume, status=status, yes_bid_size=1000, yes_ask_size=1000)


class Always(Strategy):
    name = "always"
    parameters = {"side": Param("yes")}
    risk = Risk(position_fraction=0.10, max_market_exposure=1.0, max_portfolio_exposure=1.0)

    def should_enter(self, view):
        return Signal(self.p.side)


def runner(strategy=None, capital=1000, latency=100, **risk):
    strategy = strategy or Always(risk=risk)
    r = StrategyRunner(strategy, Portfolio(capital), ExecutionConfig(contract_step=1), FeeSchedule(), latency)
    r.register(INFO)
    return r


def test_view_only_contains_past_ticks():
    v = MarketView(INFO, 100)
    v.update(tick(1000))
    v.update(tick(2000, bid=0.50, ask=0.52))
    assert v.probability == pytest.approx(0.51)
    assert [t.ts_ms for t in v.history()] == [1000, 2000]


def test_view_rejects_out_of_order_ticks():
    v = MarketView(INFO, 100)
    v.update(tick(2000))
    with pytest.raises(ValueError):
        v.update(tick(1000))


def test_probability_ago_and_change():
    v = MarketView(INFO, 300)
    for ts, bid in ((0, 0.30), (60_000, 0.35), (120_000, 0.40), (180_000, 0.45)):
        v.update(tick(ts, bid, bid + 0.02))
    assert v.probability_ago(60) == pytest.approx(0.41)
    assert v.probability_change(180) == pytest.approx(0.15)
    assert v.probability_change(9000) is None  # history does not reach that far


def test_history_window_keeps_a_base_tick():
    v = MarketView(INFO, 10)
    for ts in range(0, 100_000, 1000):
        v.update(tick(ts))
    assert v.probability_ago(10) is not None and len(v.history()) <= 12


def test_market_info_has_no_outcome_fields():
    forbidden = {"result", "settlement_value", "expiration_value", "winning_outcome", "close_ms", "settlement_ms"}
    assert forbidden.isdisjoint(MarketInfo.__dataclass_fields__)


def test_decision_never_fills_on_the_triggering_tick():
    r = runner()
    r.on_tick(tick(1000))
    assert "T" in r.pending and not r.portfolio.positions
    r.on_tick(tick(1050))  # before latency elapsed
    assert not r.portfolio.positions
    r.on_tick(tick(1100, bid=0.44, ask=0.46))
    assert r.portfolio.positions["T"].avg_price == pytest.approx(0.46)  # the later quote, not the signal's


def test_stop_loss_exits_at_executable_bid():
    r = runner(stop_loss=0.10)
    r.on_tick(tick(0))
    r.on_tick(tick(100))
    r.on_tick(tick(200, bid=0.36, ask=0.38))
    assert "T" in r.pending
    r.on_tick(tick(300, bid=0.35, ask=0.37))
    trade = r.portfolio.closed[0]
    assert trade.exit_reason == "stop_loss" and trade.exit_price == pytest.approx(0.35)


def test_take_profit():
    r = runner(take_profit=0.20)
    r.on_tick(tick(0))
    r.on_tick(tick(100))
    r.on_tick(tick(200, bid=0.51, ask=0.53))
    r.on_tick(tick(300, bid=0.51, ask=0.53))
    assert r.portfolio.closed[0].exit_reason == "take_profit"


def test_trailing_stop_follows_peak():
    r = runner(trailing_stop=0.10)
    r.on_tick(tick(0))
    r.on_tick(tick(100))
    r.on_tick(tick(200, bid=0.70, ask=0.72))
    r.on_tick(tick(300, bid=0.62, ask=0.64))
    r.on_tick(tick(400, bid=0.62, ask=0.64))
    assert r.portfolio.closed[0].exit_reason == "trailing_stop"


def test_max_holding_and_exit_before_close():
    r = runner(max_holding_seconds=1)
    r.on_tick(tick(0))
    r.on_tick(tick(100))
    r.on_tick(tick(1200))
    r.on_tick(tick(1300))
    assert r.portfolio.closed[0].exit_reason == "max_holding"
    r = runner(exit_before_close_seconds=30)
    r.on_tick(tick(0))
    r.on_tick(tick(100))
    r.on_tick(tick(871_000))
    r.on_tick(tick(871_100))
    assert r.portfolio.closed[0].exit_reason == "before_close"


def test_settlement_pays_and_blocks_further_trading():
    r = runner()
    r.on_tick(tick(0))
    r.on_tick(tick(100))
    r.settle("T", 1.0, 900_000)
    assert r.portfolio.closed[0].exit_reason == "settlement"
    r.on_tick(tick(900_100))
    assert not r.pending and not r.portfolio.positions


def test_max_entries_per_market():
    r = runner(stop_loss=0.05)
    r.on_tick(tick(0))
    r.on_tick(tick(100))
    r.on_tick(tick(200, bid=0.30, ask=0.32))
    r.on_tick(tick(300, bid=0.30, ask=0.32))
    assert not r.portfolio.positions
    r.on_tick(tick(400, bid=0.30, ask=0.32))
    assert not r.pending


def test_capital_constraint_limits_size():
    r = runner(capital=10, position_fraction=1.0)
    r.on_tick(tick(0))
    r.on_tick(tick(100))
    assert r.portfolio.cash >= 0 and r.portfolio.positions["T"].quantity <= 23


def test_market_exposure_limit():
    r = runner(capital=1000, position_fraction=1.0, max_market_exposure=0.05)
    r.on_tick(tick(0))
    assert r.pending["T"].quantity * 0.42 <= 50


def test_no_position_when_market_not_active():
    r = runner()
    r.on_tick(tick(0, status=4))
    assert not r.pending


def test_failed_exit_retries_next_tick():
    r = runner(stop_loss=0.10)
    r.on_tick(tick(0))
    r.on_tick(tick(100))
    r.on_tick(tick(200, bid=0.30, ask=0.32))
    r.on_tick(Tick("T", 300, None, None, volume=100))
    assert r.counters.failed_exits == 1 and r.portfolio.positions
    r.on_tick(tick(400, bid=0.30, ask=0.32))
    r.on_tick(tick(500, bid=0.30, ask=0.32))
    assert not r.portfolio.positions


def test_both_sides_supported():
    r = runner(Always({"side": "no"}))
    r.on_tick(tick(0))
    r.on_tick(tick(100))
    assert r.portfolio.positions["T"].side == "no" and r.portfolio.positions["T"].avg_price == pytest.approx(0.60)


def test_universe_selection():
    class Picky(Always):
        universe = Universe(categories=["Politics"])

    r = runner(Picky())
    r.on_tick(tick(0))
    assert not r.pending and "T" not in r.views


def test_unknown_parameter_rejected():
    with pytest.raises(ValueError):
        Always({"nope": 1})


def test_parameters_can_override_risk():
    assert Always({"stop_loss": 0.2}).risk.stop_loss == 0.2


@pytest.mark.parametrize("env,name", [("general", "example.py"), ("general", "longshot_fade.py"),
                                      ("15m", "example.py"), ("15m", "late_favorite.py")])
def test_example_strategies_load_and_describe_themselves(env, name):
    strategy, _ = load_strategy(load_config(env), name)
    spec = strategy.to_spec()
    assert spec["name"] and spec["risk"] and strategy.source_hash()


def test_strategy_directories_are_separate():
    assert load_strategy(load_config("15m"), "example.py")[0].name != load_strategy(load_config("general"), "example.py")[0].name


def test_brief_example_conditions():
    strategy, _ = load_strategy(load_config("general"), "example.py")
    view = MarketView(MarketInfo("T", "E", "S", "c", "t", "e", 0, 10**9), 3600)
    view.update(Tick("T", 0, 0.29, 0.31, volume=20_000))
    view.update(Tick("T", 180_000, 0.36, 0.38, volume=20_000))
    assert strategy.should_enter(view).side == "yes"
    view = MarketView(MarketInfo("T", "E", "S", "c", "t", "e", 0, 10**9), 3600)
    view.update(Tick("T", 0, 0.29, 0.31, volume=5_000))
    view.update(Tick("T", 180_000, 0.36, 0.38, volume=5_000))
    assert strategy.should_enter(view) is None
