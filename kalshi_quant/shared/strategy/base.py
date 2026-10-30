"""Strategy interface.

A strategy is ordinary Python. It sees a ``MarketView``, which holds only ticks already
delivered to it, plus ``MarketInfo``, which has no outcome fields. It therefore cannot
reach future data: nothing it holds refers to any.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any

from kalshi_quant.shared.data_models.tick import MarketInfo, Tick

YES, NO = "yes", "no"


@dataclass(frozen=True)
class Param:
    """A tunable value. min/max/step define its default optimization range."""
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    description: str = ""


@dataclass
class Risk:
    """Exit and sizing rules, applied by the engine against executable prices."""
    position_fraction: float = 0.05       # of current equity, per entry
    max_contracts: float | None = None
    stop_loss: float | None = None        # fraction of entry price, e.g. 0.15
    take_profit: float | None = None      # fraction of entry price, e.g. 0.30
    trailing_stop: float | None = None    # fraction below the best bid since entry
    max_holding_seconds: float | None = None
    exit_before_close_seconds: float | None = None
    max_market_exposure: float = 0.10     # cost basis in one market, fraction of equity
    max_portfolio_exposure: float = 0.80
    max_entries_per_market: int = 1
    max_spread_at_entry: float | None = None


@dataclass
class Universe:
    """Market selection. Empty lists mean no restriction."""
    categories: list[str] = field(default_factory=list)
    series: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    exclude_series: list[str] = field(default_factory=list)

    def matches(self, info: MarketInfo) -> bool:
        return ((not self.categories or info.category in self.categories)
                and (not self.series or info.series_ticker in self.series)
                and (not self.events or info.event_ticker in self.events)
                and (not self.tickers or info.ticker in self.tickers)
                and info.series_ticker not in self.exclude_series)


@dataclass(slots=True)
class Signal:
    side: str
    quantity: float | None = None        # contracts; default sizing from Risk when None
    limit_price: float | None = None     # worst acceptable price for this side
    tag: str = ""


class MarketView:
    """Rolling, past-only view of one market."""

    def __init__(self, info: MarketInfo, history_seconds: float):
        self.info = info
        self._window_ms = int(history_seconds * 1000)
        self._ticks: deque[Tick] = deque()
        self.tick: Tick | None = None
        self.first_seen_ms: int | None = None

    def update(self, tick: Tick) -> None:
        if self.tick is not None and tick.ts_ms < self.tick.ts_ms:
            raise ValueError(f"out-of-order tick for {tick.ticker}: {tick.ts_ms} < {self.tick.ts_ms}")
        self.tick = tick
        self.first_seen_ms = self.first_seen_ms or tick.ts_ms
        self._ticks.append(tick)
        cutoff = tick.ts_ms - self._window_ms
        while len(self._ticks) > 1 and self._ticks[1].ts_ms <= cutoff:
            self._ticks.popleft()

    now_ms = property(lambda self: self.tick.ts_ms)
    yes_bid = property(lambda self: self.tick.yes_bid)
    yes_ask = property(lambda self: self.tick.yes_ask)
    no_bid = property(lambda self: None if self.tick.yes_ask is None else 1 - self.tick.yes_ask)
    no_ask = property(lambda self: None if self.tick.yes_bid is None else 1 - self.tick.yes_bid)
    last = property(lambda self: self.tick.last)
    volume = property(lambda self: self.tick.volume)
    open_interest = property(lambda self: self.tick.open_interest)
    probability = property(lambda self: self.tick.probability)

    @property
    def no_probability(self) -> float | None:
        p = self.probability
        return None if p is None else 1 - p

    @property
    def spread(self) -> float | None:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return self.yes_ask - self.yes_bid

    @property
    def seconds_since_open(self) -> float | None:
        return None if self.info.open_ms is None else (self.now_ms - self.info.open_ms) / 1000

    @property
    def seconds_to_close(self) -> float | None:
        """Time to the scheduled close. Early closes are not knowable in advance."""
        return None if self.info.scheduled_close_ms is None else (self.info.scheduled_close_ms - self.now_ms) / 1000

    def probability_ago(self, seconds: float) -> float | None:
        """Latest probability at or before ``now - seconds``, if history reaches that far."""
        target = self.now_ms - seconds * 1000
        found = None
        for tick in self._ticks:
            if tick.ts_ms > target:
                break
            if tick.probability is not None:
                found = tick.probability
        return found

    def probability_change(self, seconds: float) -> float | None:
        then, now = self.probability_ago(seconds), self.probability
        return None if then is None or now is None else now - then

    def volume_since(self, seconds: float) -> float | None:
        target = self.now_ms - seconds * 1000
        base = None
        for tick in self._ticks:
            if tick.ts_ms > target:
                break
            base = tick.volume
        return None if base is None or self.volume is None else self.volume - base

    def history(self, seconds: float | None = None) -> list[Tick]:
        if seconds is None:
            return list(self._ticks)
        cutoff = self.now_ms - seconds * 1000
        return [t for t in self._ticks if t.ts_ms >= cutoff]


class Strategy:
    """Subclass and implement ``should_enter``. See the example strategies."""
    name: str = ""
    environments: tuple[str, ...] = ("general", "fifteen_minute")
    parameters: dict[str, Param] = {}
    risk: Risk = Risk()
    universe: Universe = Universe()
    history_seconds: float = 900

    def __init__(self, params: dict[str, Any] | None = None, risk: dict[str, Any] | None = None,
                 universe: dict[str, Any] | None = None):
        self.name = self.name or type(self).__name__
        self.rng = random.Random(0)  # engines reseed this; use it for any randomness
        values = {k: p.default for k, p in self.parameters.items()}
        unknown = set(params or {}) - set(values) - set(Risk.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown parameters for {self.name}: {sorted(unknown)}")
        risk_values = {**asdict(type(self).risk), **(risk or {})}
        for key, value in (params or {}).items():
            if key in values:
                values[key] = value
            else:  # risk settings are tunable by the optimizer like any parameter
                risk_values[key] = value
        self.p = SimpleNamespace(**values)
        self.risk = Risk(**risk_values)
        self.universe = Universe(**{**asdict(type(self).universe), **(universe or {})})

    def select_market(self, info: MarketInfo) -> bool:
        """Extra selection beyond ``universe``. Runs once per market, before any data."""
        return True

    def should_enter(self, view: MarketView) -> Signal | None:
        raise NotImplementedError

    def should_exit(self, view: MarketView, position: Any) -> str | None:
        """Return a reason to exit early. Stops and take-profit are handled by Risk."""
        return None

    def objective(self, metrics: dict[str, Any]) -> float | None:
        """Optional custom objective for the optimizer (``--objective custom``)."""
        return None

    def to_spec(self) -> dict[str, Any]:
        """Standard description shared by paper trading, backtests and the optimizer."""
        return {"name": self.name, "class": type(self).__name__, "params": vars(self.p), "risk": asdict(self.risk),
                "universe": asdict(self.universe), "history_seconds": self.history_seconds}

    def source(self) -> str:
        return inspect.getsource(inspect.getmodule(type(self)))

    def source_hash(self) -> str:
        return hashlib.sha256(self.source().encode()).hexdigest()[:16]


def params_hash(params: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:16]
