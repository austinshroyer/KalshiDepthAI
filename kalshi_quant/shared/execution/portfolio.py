"""Virtual portfolio. One per strategy, never shared."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from kalshi_quant.shared.execution.fills import BUY, YES, Fill


@dataclass(slots=True)
class Position:
    ticker: str
    side: str
    quantity: float
    avg_price: float       # per contract, before fees
    entry_ms: int
    entry_fees: float = 0.0
    entry_slippage: float = 0.0
    peak_price: float = 0.0  # best bid seen since entry, for trailing stops
    mark_price: float | None = None
    entries: int = 1

    @property
    def cost(self) -> float:
        return self.quantity * self.avg_price


@dataclass(slots=True)
class ClosedTrade:
    ticker: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_ms: int
    exit_ms: int
    pnl: float    # after all fees on this quantity
    fees: float
    slippage: float
    exit_reason: str


@dataclass
class Portfolio:
    starting_capital: float
    cash: float = -1.0
    positions: dict[str, Position] = field(default_factory=dict)
    closed: list[ClosedTrade] = field(default_factory=list)
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    slippage: float = 0.0

    def __post_init__(self) -> None:
        if self.cash < 0:
            self.cash = self.starting_capital

    def apply_fill(self, fill: Fill) -> ClosedTrade | None:
        order = fill.order
        self.fees_paid += fill.fee
        self.slippage += fill.slippage
        if order.action == BUY:
            self.cash -= fill.filled * fill.avg_price + fill.fee
            position = self.positions.get(order.ticker)
            if position is None:
                self.positions[order.ticker] = Position(order.ticker, order.side, fill.filled, fill.avg_price, fill.ts_ms,
                                                        fill.fee, fill.slippage, fill.avg_price)
            else:
                total = position.quantity + fill.filled
                position.avg_price = (position.cost + fill.filled * fill.avg_price) / total
                position.quantity = total
                position.entry_fees += fill.fee
                position.entry_slippage += fill.slippage
            return None
        self.cash += fill.filled * fill.avg_price - fill.fee
        return self._close(order.ticker, fill.filled, fill.avg_price, fill.ts_ms, fill.fee, fill.slippage, order.reason)

    def settle(self, ticker: str, yes_value: float, ts_ms: int) -> ClosedTrade | None:
        """Pay out at settlement: ``yes_value`` per YES contract, the complement per NO."""
        position = self.positions.get(ticker)
        if position is None:
            return None
        payout = yes_value if position.side == YES else 1 - yes_value
        self.cash += position.quantity * payout
        return self._close(ticker, position.quantity, payout, ts_ms, 0.0, 0.0, "settlement")

    def _close(self, ticker: str, quantity: float, price: float, ts_ms: int, fee: float, slippage: float,
               reason: str) -> ClosedTrade:
        position = self.positions[ticker]
        fraction = quantity / position.quantity
        entry_fees = position.entry_fees * fraction
        entry_slippage = position.entry_slippage * fraction
        pnl = quantity * (price - position.avg_price) - fee - entry_fees
        trade = ClosedTrade(ticker, position.side, quantity, position.avg_price, price, position.entry_ms, ts_ms, pnl,
                            fee + entry_fees, slippage + entry_slippage, reason)
        self.closed.append(trade)
        self.realized_pnl += pnl
        position.quantity -= quantity
        position.entry_fees -= entry_fees
        position.entry_slippage -= entry_slippage
        if position.quantity <= 1e-9:
            del self.positions[ticker]
        return trade

    def market_value(self) -> float:
        return sum(p.quantity * (p.mark_price if p.mark_price is not None else p.avg_price)
                   for p in self.positions.values())

    def equity(self) -> float:
        return self.cash + self.market_value()

    def exposure(self, ticker: str | None = None) -> float:
        if ticker is not None:
            position = self.positions.get(ticker)
            return position.cost if position else 0.0
        return sum(p.cost for p in self.positions.values())

    def unrealized_pnl(self) -> float:
        return sum(p.quantity * ((p.mark_price if p.mark_price is not None else p.avg_price) - p.avg_price) - p.entry_fees
                   for p in self.positions.values())

    def to_state(self) -> dict[str, Any]:
        return {"starting_capital": self.starting_capital, "cash": self.cash, "realized_pnl": self.realized_pnl,
                "fees_paid": self.fees_paid, "slippage": self.slippage,
                "positions": {t: asdict(p) for t, p in self.positions.items()}}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "Portfolio":
        portfolio = cls(state["starting_capital"], state["cash"])
        portfolio.realized_pnl = state["realized_pnl"]
        portfolio.fees_paid = state["fees_paid"]
        portfolio.slippage = state["slippage"]
        portfolio.positions = {t: Position(**p) for t, p in state["positions"].items()}
        return portfolio
