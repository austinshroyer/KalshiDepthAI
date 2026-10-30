import pytest

from kalshi_quant.shared.config import ExecutionConfig
from kalshi_quant.shared.data_models.tick import MarketInfo, Tick
from kalshi_quant.shared.execution.fees import FeeSchedule
from kalshi_quant.shared.execution.fills import Order, simulate_fill, touch
from kalshi_quant.shared.execution.portfolio import Portfolio

INFO = MarketInfo("T", "E", "S", "Crypto", "t", "e", 0, 900_000, tick_ranges=((0, 0.1, 0.001), (0.1, 0.9, 0.01), (0.9, 1, 0.001)))
FEES = FeeSchedule()
CFG = ExecutionConfig()


def order(side="yes", action="buy", qty=10, limit=None, decision=None):
    return Order("T", side, action, qty, 0, 0, limit, decision)


def book_tick(**kw):
    return Tick("T", 1000, 0.40, 0.42, yes_levels=((0.40, 100), (0.39, 100)), no_levels=((0.58, 20), (0.57, 100)), **kw)


CENT_FEES = FeeSchedule(rounding=0.01)


def test_fee_matches_the_published_example():
    assert FEES.order_fee([(0.60, 100)]) == pytest.approx(1.68)  # kalshi.com: 100 contracts at 60 cents


def test_fee_rounds_up_to_a_centicent():
    assert FEES.order_fee([(0.50, 1)]) == pytest.approx(0.0175)
    assert FEES.order_fee([(0.01, 1)]) == pytest.approx(0.0007)


def test_fcm_accounts_round_to_a_cent():
    assert CENT_FEES.order_fee([(0.50, 1)]) == pytest.approx(0.02)
    assert CENT_FEES.order_fee([(0.01, 1)]) == pytest.approx(0.01)


def test_fee_multiplier_and_free_series():
    assert FEES.order_fee([(0.50, 100)], 0.5) == pytest.approx(0.875)
    assert FEES.order_fee([(0.50, 100)], 0) == 0


def test_fee_rounds_once_per_order_not_per_level():
    assert CENT_FEES.order_fee([(0.5, 1), (0.5, 1), (0.5, 1), (0.5, 1)]) == pytest.approx(0.07)


def test_touch_prices_each_side():
    t = Tick("T", 0, 0.40, 0.42)
    assert touch(t, "yes", "buy") == 0.42 and touch(t, "yes", "sell") == 0.40
    assert touch(t, "no", "buy") == pytest.approx(0.60) and touch(t, "no", "sell") == pytest.approx(0.58)


def test_buy_yes_consumes_no_bids_with_participation_cap():
    fill = simulate_fill(order(qty=15), book_tick(), INFO, CFG, FEES)
    assert fill.status == "filled"  # 10 at 0.42 (half of 20) then 5 at 0.43
    assert fill.avg_price == pytest.approx((10 * 0.42 + 5 * 0.43) / 15)


def test_partial_fill_when_book_is_thin():
    fill = simulate_fill(order(qty=1000), book_tick(), INFO, CFG, FEES)
    assert fill.status == "partial" and fill.filled == 60


def test_sell_no_consumes_no_bids():
    fill = simulate_fill(order("no", "sell", 5), book_tick(), INFO, CFG, FEES)
    assert fill.avg_price == pytest.approx(0.58)


def test_buy_no_consumes_yes_bids():
    fill = simulate_fill(order("no", "buy", 5), book_tick(), INFO, CFG, FEES)
    assert fill.avg_price == pytest.approx(0.60)


def test_limit_stops_the_walk():
    fill = simulate_fill(order(qty=50, limit=0.42), book_tick(), INFO, CFG, FEES)
    assert fill.status == "partial" and fill.filled == 10


def test_limit_rejected_when_never_reached():
    fill = simulate_fill(order(limit=0.30), book_tick(), INFO, CFG, FEES)
    assert fill.status == "rejected" and fill.reason == "limit_not_reached"


def test_top_of_book_only_when_no_depth():
    fill = simulate_fill(order(qty=100), Tick("T", 0, 0.40, 0.42, yes_ask_size=30), INFO, CFG, FEES)
    assert fill.filled == 15 and fill.status == "partial"


def test_unknown_size_is_rejected_not_assumed():
    fill = simulate_fill(order(), Tick("T", 0, 0.40, 0.42), INFO, CFG, FEES)
    assert fill.reason == "insufficient_liquidity"


def test_candle_fill_capped_by_volume_with_extra_slippage():
    tick = Tick("T", 0, 0.40, 0.42, volume_delta=100, is_candle=True)
    fill = simulate_fill(order(qty=50), tick, INFO, CFG, FEES)
    assert fill.filled == 10 and fill.avg_price == pytest.approx(0.43)


def test_candle_with_no_volume_is_rejected():
    tick = Tick("T", 0, 0.40, 0.42, volume_delta=0, is_candle=True)
    assert simulate_fill(order(), tick, INFO, CFG, FEES).status == "rejected"


def test_slippage_uses_market_tick_grid():
    config = ExecutionConfig(slippage_ticks=1)
    tick = Tick("T", 0, 0.03, 0.05, yes_ask_size=100)
    assert simulate_fill(order(qty=1), tick, INFO, config, FEES).avg_price == pytest.approx(0.051)


def test_inactive_market_rejected():
    assert simulate_fill(order(), book_tick(status=4), INFO, CFG, FEES).reason == "market_not_active"


def test_stale_quote_rejected():
    fill = simulate_fill(order(), book_tick(), INFO, CFG, FEES, max_quote_age_ms=100, quote_ts_ms=0)
    assert fill.reason == "stale_quote"


def test_no_quote_rejected():
    assert simulate_fill(order(), Tick("T", 0, 0.40, None), INFO, CFG, FEES).reason == "no_quote"


def test_slippage_measured_against_decision_price():
    fill = simulate_fill(order(qty=10, decision=0.41), book_tick(), INFO, CFG, FEES)
    assert fill.slippage == pytest.approx(0.1)


def fill_of(side="yes", action="buy", qty=10, price=0.40, fee=0.17, ts=1000, reason=""):
    from kalshi_quant.shared.execution.fills import Fill
    return Fill(Order("T", side, action, qty, 0, 0, reason=reason), qty, price, fee, 0.0, "filled", "", ts)


def test_winning_yes_settlement_pnl():
    p = Portfolio(100)
    p.apply_fill(fill_of())
    assert p.cash == pytest.approx(100 - 4 - 0.17)
    trade = p.settle("T", 1.0, 2000)
    assert p.cash == pytest.approx(105.83) and trade.pnl == pytest.approx(5.83) and not p.positions


def test_losing_no_settlement_pnl():
    p = Portfolio(100)
    p.apply_fill(fill_of("no", price=0.60, fee=0.17))
    trade = p.settle("T", 1.0, 2000)
    assert p.cash == pytest.approx(100 - 6 - 0.17) and trade.pnl == pytest.approx(-6.17)


def test_scalar_settlement():
    p = Portfolio(100)
    p.apply_fill(fill_of("no", price=0.60, fee=0))
    assert p.settle("T", 0.25, 2000).pnl == pytest.approx(1.5)


def test_partial_exit_allocates_entry_fees():
    p = Portfolio(100)
    p.apply_fill(fill_of(fee=1.0))
    trade = p.apply_fill(fill_of(action="sell", qty=5, price=0.50, fee=0.10, reason="take_profit"))
    assert trade.pnl == pytest.approx(5 * 0.10 - 0.10 - 0.50) and p.positions["T"].quantity == 5
    assert p.positions["T"].entry_fees == pytest.approx(0.5)


def test_realized_pnl_reconciles_with_cash():
    p = Portfolio(100)
    p.apply_fill(fill_of())
    p.apply_fill(fill_of(action="sell", price=0.55, fee=0.17, reason="x"))
    assert p.cash - 100 == pytest.approx(p.realized_pnl)


def test_state_round_trip():
    p = Portfolio(100)
    p.apply_fill(fill_of())
    restored = Portfolio.from_state(p.to_state())
    assert restored.cash == p.cash and restored.positions["T"].avg_price == 0.40
