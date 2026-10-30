"""The decision loop. Paper trading and backtesting run this exact code.

Order of events on a tick: (1) the view advances, (2) orders whose latency elapsed fill
against this tick, (3) exits are evaluated, (4) entries are evaluated. A decision made
on tick t can only fill on a later tick at or after t + latency, so a strategy can never
trade on the same quote that triggered it.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from kalshi_quant.shared.config import ExecutionConfig
from kalshi_quant.shared.data_models.tick import ACTIVE, MarketInfo, Tick
from kalshi_quant.shared.execution.fees import FeeSchedule
from kalshi_quant.shared.execution.fills import BUY, SELL, Fill, Order, simulate_fill, touch
from kalshi_quant.shared.execution.portfolio import ClosedTrade, Portfolio, Position
from kalshi_quant.shared.strategy.base import MarketView, Signal, Strategy


@dataclass
class RunCounters:
    signals: int = 0
    orders: int = 0
    partial_fills: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    capital_blocked: int = 0
    exposure_blocked: int = 0
    failed_exits: int = 0

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1


class StrategyRunner:
    def __init__(self, strategy: Strategy, portfolio: Portfolio, execution: ExecutionConfig, fees: FeeSchedule,
                 latency_ms: int, max_quote_age_ms: int | None = None,
                 on_fill: Callable[[Fill], None] | None = None,
                 on_close: Callable[[ClosedTrade], None] | None = None):
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution = execution
        self.fees = fees
        self.latency_ms = latency_ms
        self.max_quote_age_ms = max_quote_age_ms
        self.on_fill = on_fill or (lambda fill: None)
        self.on_close = on_close or (lambda trade: None)
        self.views: dict[str, MarketView] = {}
        self.infos: dict[str, MarketInfo] = {}
        self.selected: dict[str, bool] = {}
        self.pending: dict[str, Order] = {}
        self.entries: dict[str, int] = {}
        self.finished: set[str] = set()
        self.counters = RunCounters()

    def register(self, info: MarketInfo) -> None:
        self.infos[info.ticker] = info

    def _is_selected(self, ticker: str) -> bool:
        cached = self.selected.get(ticker)
        if cached is None:
            info = self.infos.get(ticker)
            cached = bool(info and self.strategy.universe.matches(info) and self.strategy.select_market(info))
            self.selected[ticker] = cached
        return cached

    def on_tick(self, tick: Tick) -> None:
        ticker = tick.ticker
        if ticker in self.finished or not self._is_selected(ticker):
            return
        view = self.views.get(ticker)
        if view is None:
            view = self.views[ticker] = MarketView(self.infos[ticker], self.strategy.history_seconds)
        previous_ts = view.tick.ts_ms if view.tick else None
        view.update(tick)
        order = self.pending.get(ticker)
        if order is not None and tick.ts_ms >= order.execute_at_ms:
            del self.pending[ticker]
            self._execute(order, tick, previous_ts)
        position = self.portfolio.positions.get(ticker)
        if position is not None:
            self._mark(position, tick)
        if ticker in self.pending or tick.status != ACTIVE:
            return
        if position is not None:
            self._evaluate_exit(view, position)
        elif self.entries.get(ticker, 0) < self.strategy.risk.max_entries_per_market:
            self._evaluate_entry(view)

    def _mark(self, position: Position, tick: Tick) -> None:
        price = touch(tick, position.side, SELL)
        if price is not None:
            position.mark_price = price
            position.peak_price = max(position.peak_price, price)

    def _execute(self, order: Order, tick: Tick, quote_ts_ms: int | None) -> None:
        # A gap between candles is their period, not a stale market; volume caps cover them.
        max_age = None if tick.is_candle else self.max_quote_age_ms
        fill = simulate_fill(order, tick, self.infos[order.ticker], self.execution, self.fees, max_age, quote_ts_ms)
        if fill.status == "rejected":
            self.counters.reject(fill.reason)
            if order.action == SELL:
                self.counters.failed_exits += 1
            self.on_fill(fill)
            return
        if order.action == BUY:
            cost = fill.filled * fill.avg_price + fill.fee
            if cost > self.portfolio.cash:  # the book moved against us during latency
                self.counters.capital_blocked += 1
                fill.status, fill.filled, fill.fee, fill.reason = "rejected", 0.0, 0.0, "insufficient_capital"
                fill.slippage = 0.0
                self.on_fill(fill)
                return
            self.entries[order.ticker] = self.entries.get(order.ticker, 0) + 1
        if fill.status == "partial":
            self.counters.partial_fills += 1
        closed = self.portfolio.apply_fill(fill)
        self.on_fill(fill)
        if closed:
            self.on_close(closed)

    def _exit_reason(self, view: MarketView, position: Position) -> str | None:
        risk = self.strategy.risk
        price = position.mark_price
        if price is not None:
            if risk.stop_loss is not None and price <= position.avg_price * (1 - risk.stop_loss):
                return "stop_loss"
            if risk.take_profit is not None and price >= position.avg_price * (1 + risk.take_profit):
                return "take_profit"
            if risk.trailing_stop is not None and price <= position.peak_price * (1 - risk.trailing_stop):
                return "trailing_stop"
        if risk.max_holding_seconds is not None and view.now_ms - position.entry_ms >= risk.max_holding_seconds * 1000:
            return "max_holding"
        to_close = view.seconds_to_close
        if risk.exit_before_close_seconds is not None and to_close is not None and to_close <= risk.exit_before_close_seconds:
            return "before_close"
        return self.strategy.should_exit(view, position)

    def _evaluate_exit(self, view: MarketView, position: Position) -> None:
        reason = self._exit_reason(view, position)
        if reason:
            self.pending[view.tick.ticker] = Order(
                view.tick.ticker, position.side, SELL, position.quantity, view.now_ms, view.now_ms + self.latency_ms,
                None, touch(view.tick, position.side, SELL), reason)

    def _evaluate_entry(self, view: MarketView) -> None:
        signal = self.strategy.should_enter(view)
        if signal is None:
            return
        self.counters.signals += 1
        risk = self.strategy.risk
        if risk.max_spread_at_entry is not None and (view.spread is None or view.spread > risk.max_spread_at_entry):
            self.counters.reject("spread_too_wide")
            return
        price = touch(view.tick, signal.side, BUY)
        if price is None:
            self.counters.reject("no_quote")
            return
        quantity = self._size(view, signal, price)
        if quantity <= 0:
            return
        self.counters.orders += 1
        self.pending[view.tick.ticker] = Order(view.tick.ticker, signal.side, BUY, quantity, view.now_ms,
                                               view.now_ms + self.latency_ms, signal.limit_price, price, signal.tag)

    def _size(self, view: MarketView, signal: Signal, price: float) -> float:
        """Apply sizing, then capital and exposure limits, in whole contract steps."""
        risk = self.strategy.risk
        equity = self.portfolio.equity()
        wanted = signal.quantity if signal.quantity is not None else equity * risk.position_fraction / price
        if risk.max_contracts is not None:
            wanted = min(wanted, risk.max_contracts)
        market_room = equity * risk.max_market_exposure - self.portfolio.exposure(view.tick.ticker)
        portfolio_room = equity * risk.max_portfolio_exposure - self.portfolio.exposure()
        if min(market_room, portfolio_room) <= 0:
            self.counters.exposure_blocked += 1
            return 0.0
        fee_rate = self.fees.taker_coefficient * self.infos[view.tick.ticker].fee_multiplier * price * (1 - price)
        limit = min(market_room / price, portfolio_room / price,
                    max(self.portfolio.cash - self.fees.rounding, 0) / (price + fee_rate))
        if limit < wanted:
            self.counters.capital_blocked += 1
        step = self.execution.contract_step
        return math.floor(min(wanted, limit) / step + 1e-9) * step

    def settle(self, ticker: str, yes_value: float, ts_ms: int) -> None:
        """Called only once the market's real settlement time is reached."""
        self.finished.add(ticker)
        self.pending.pop(ticker, None)
        closed = self.portfolio.settle(ticker, yes_value, ts_ms)
        if closed:
            self.on_close(closed)
