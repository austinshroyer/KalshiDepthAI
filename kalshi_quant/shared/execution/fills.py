"""Fill simulation shared by paper trading and backtesting.

Orders are immediate-or-cancel takers. Only a fraction of displayed size is assumed
available, because other traders compete for it, and no hidden liquidity is assumed.
Resting (maker) orders are not simulated: without queue position data any maker fill
estimate would be invented.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from kalshi_quant.shared.config import ExecutionConfig
from kalshi_quant.shared.data_models.tick import ACTIVE, MarketInfo, Tick
from kalshi_quant.shared.execution.fees import FeeSchedule

DEFAULT_TICK = 0.01
BUY, SELL = "buy", "sell"
YES, NO = "yes", "no"


@dataclass(slots=True)
class Order:
    ticker: str
    side: str
    action: str
    quantity: float
    decision_ms: int
    execute_at_ms: int
    limit_price: float | None = None
    decision_price: float | None = None  # touch when decided, for slippage
    reason: str = ""


@dataclass(slots=True)
class Fill:
    order: Order
    filled: float
    avg_price: float
    fee: float
    slippage: float  # dollars worse than decision_price
    status: str      # filled, partial, rejected
    reason: str
    ts_ms: int


def tick_size(info: MarketInfo, price: float) -> float:
    for start, end, step in info.tick_ranges:
        if start <= price <= end:
            return step
    return DEFAULT_TICK


def touch(tick: Tick, side: str, action: str) -> float | None:
    """Best executable price in the order's own side terms."""
    if action == BUY:
        if side == YES:
            return tick.yes_ask
        return None if tick.yes_bid is None else 1 - tick.yes_bid
    if side == YES:
        return tick.yes_bid
    return None if tick.yes_ask is None else 1 - tick.yes_ask


def available_levels(tick: Tick, side: str, action: str, config: ExecutionConfig) -> tuple[list[tuple[float, float]], str]:
    """Executable (price, size) levels, best first, and how they were sourced."""
    best = touch(tick, side, action)
    if best is None:
        return [], "no_quote"
    # A YES buy lifts NO bids at 1 - price, and so on: each trade consumes one side's bids.
    consume_yes_bids = (action == SELL and side == YES) or (action == BUY and side == NO)
    book = tick.yes_levels if consume_yes_bids else tick.no_levels
    own_side_price = (action == SELL)
    if book is not None:
        levels = [(p if own_side_price else 1 - p, s * config.max_book_participation) for p, s in book]
        return levels, "book"
    size = tick.yes_bid_size if consume_yes_bids else tick.yes_ask_size
    if size is not None:
        return [(best, size * config.max_book_participation)], "top_of_book"
    if tick.is_candle and tick.volume_delta:
        return [(best, tick.volume_delta * config.max_candle_volume_participation)], "candle_volume"
    return [], "no_size"


def simulate_fill(order: Order, tick: Tick, info: MarketInfo, config: ExecutionConfig, fees: FeeSchedule,
                  max_quote_age_ms: int | None = None, quote_ts_ms: int | None = None) -> Fill:
    def rejected(reason: str) -> Fill:
        return Fill(order, 0.0, 0.0, 0.0, 0.0, "rejected", reason, tick.ts_ms)

    if tick.status != ACTIVE:
        return rejected("market_not_active")
    if max_quote_age_ms is not None and quote_ts_ms is not None and tick.ts_ms - quote_ts_ms > max_quote_age_ms:
        return rejected("stale_quote")
    levels, source = available_levels(tick, order.side, order.action, config)
    if not levels:
        return rejected("insufficient_liquidity" if source != "no_quote" else "no_quote")
    slippage_ticks = config.candle_slippage_ticks if source == "candle_volume" else config.slippage_ticks
    sign = 1 if order.action == BUY else -1
    remaining = order.quantity
    fills: list[tuple[float, float]] = []
    step = max(config.contract_step, 1e-9)
    for price, size in levels:
        price += sign * slippage_ticks * tick_size(info, price)
        if not 0 < price < 1:
            continue
        if order.limit_price is not None and sign * (price - order.limit_price) > 1e-9:
            break
        take = min(remaining, size)
        take = math.floor(take / step + 1e-9) * step
        if take <= 0:
            continue
        fills.append((price, take))
        remaining -= take
        if remaining <= 1e-9:
            break
    if not fills:
        return rejected("limit_not_reached" if order.limit_price is not None else "insufficient_liquidity")
    filled = sum(s for _, s in fills)
    avg = sum(p * s for p, s in fills) / filled
    reference = order.decision_price if order.decision_price is not None else fills[0][0]
    return Fill(order, filled, avg, fees.order_fee(fills, info.fee_multiplier), sign * (avg - reference) * filled,
                "filled" if remaining <= 1e-9 else "partial", "", tick.ts_ms)
