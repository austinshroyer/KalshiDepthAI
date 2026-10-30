"""Strategy genomes. A strategy is a set of named components, so crossover can exchange
individual indicators, conditions, thresholds, operators, sizing and risk rules instead of
splicing whole strategies. ``GenomeStrategy`` runs through the ordinary StrategyRunner: it sees
only the past-only MarketView, so evolved logic has the same look-ahead protection as hand-written.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from kalshi_quant.shared.strategy.base import NO, YES, MarketView, Param, Risk, Signal, Strategy

# name: (lowest threshold, highest threshold, uses a lookback)
INDICATORS: dict[str, tuple[float, float, bool]] = {
    "probability": (0.02, 0.98, False),
    "favorite_probability": (0.50, 0.99, False),
    "spread": (0.001, 0.15, False),
    "probability_change": (-0.20, 0.20, True),
    "abs_probability_change": (0.0, 0.25, True),
    "volatility": (0.0, 0.10, True),
    "volume_share": (0.0, 0.50, True),       # share of all volume traded within the lookback
    "log_volume": (0.0, 16.0, False),
    "elapsed_fraction": (0.0, 1.0, False),   # of the market's scheduled life
    "book_imbalance": (-1.0, 1.0, False),    # (bid size - ask size) / total at the touch
}
SIDES = ("yes", "no", "favorite", "underdog", "momentum", "contrarian")
OPERATORS = (">", "<")
LOGIC = ("all", "any")


@dataclass(frozen=True)
class GeneSpace:
    """Environment scale: a 15-minute market cannot use a six-hour lookback."""
    lookbacks: tuple[int, ...]
    holding_seconds: tuple[int, ...]
    max_conditions: int = 4


def compute_indicator(view: MarketView, name: str, lookback: int) -> float | None:
    if name == "probability":
        return view.probability
    if name == "favorite_probability":
        p = view.probability
        return None if p is None else max(p, 1 - p)
    if name == "spread":
        return view.spread
    if name == "probability_change":
        return view.probability_change(lookback)
    if name == "abs_probability_change":
        change = view.probability_change(lookback)
        return None if change is None else abs(change)
    if name == "volatility":
        values = [t.probability for t in view.history(lookback) if t.probability is not None]
        if len(values) < 3:
            return None
        mean = sum(values) / len(values)
        return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    if name == "volume_share":
        recent, total = view.volume_since(lookback), view.volume
        return None if recent is None or not total else recent / total
    if name == "log_volume":
        return None if view.volume is None else math.log1p(view.volume)
    if name == "elapsed_fraction":
        since, duration = view.seconds_since_open, view.info.duration_seconds
        return None if since is None or not duration else since / duration
    if name == "book_imbalance":
        bid, ask = view.tick.yes_bid_size, view.tick.yes_ask_size
        return None if bid is None or ask is None or bid + ask == 0 else (bid - ask) / (bid + ask)
    raise KeyError(name)


@dataclass(frozen=True)
class Condition:
    indicator: str
    operator: str
    threshold: float
    lookback: int = 0

    def test(self, view: MarketView) -> bool:
        value = compute_indicator(view, self.indicator, self.lookback)
        if value is None:
            return False
        return value > self.threshold if self.operator == ">" else value < self.threshold


@dataclass(frozen=True)
class Genome:
    entry: tuple[Condition, ...]
    entry_logic: str
    exits: tuple[Condition, ...]
    exit_logic: str
    side: str
    position_fraction: float
    stop_loss: float | None
    take_profit: float | None
    trailing_stop: float | None
    max_holding_seconds: int | None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Genome":
        data = json.loads(text)
        data["entry"] = tuple(Condition(**c) for c in data["entry"])
        data["exits"] = tuple(Condition(**c) for c in data["exits"])
        return cls(**data)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]

    def features(self) -> list[str]:
        """Labels for asking which strategy types succeed where."""
        return [f"side:{self.side}", *sorted({f"indicator:{c.indicator}" for c in self.entry})]


@dataclass
class Individual:
    """A genome plus how it came to exist."""
    genome: Genome
    origin: str = "random"          # random, mutation, crossover, or a hybrid mode
    parents: tuple[str, ...] = ()
    components: dict[str, str] = field(default_factory=dict)  # component -> parent it came from

    @property
    def id(self) -> str:
        return self.genome.fingerprint()


class GenomeStrategy(Strategy):
    """Runs a genome. Stored as a parameter, so saved runs and replay work as for any strategy."""
    name = "evolved"
    parameters = {"genome": Param("")}

    def __init__(self, params: dict[str, Any] | None = None, risk: dict[str, Any] | None = None,
                 universe: dict[str, Any] | None = None):
        super().__init__(params, risk, universe)
        self.genome = Genome.from_json(self.p.genome)
        g = self.genome
        self.risk = Risk(position_fraction=g.position_fraction, stop_loss=g.stop_loss, take_profit=g.take_profit,
                         trailing_stop=g.trailing_stop, max_holding_seconds=g.max_holding_seconds,
                         max_market_exposure=max(g.position_fraction * 2, 0.05), max_portfolio_exposure=0.8)
        self.history_seconds = max([c.lookback for c in (*g.entry, *g.exits)] + [60]) * 1.2

    @classmethod
    def build(cls, genome: Genome) -> "GenomeStrategy":
        return cls({"genome": genome.to_json()})

    def _side(self, view: MarketView) -> str | None:
        side = self.genome.side
        if side in (YES, NO):
            return side
        p = view.probability
        if p is None:
            return None
        if side == "favorite":
            return YES if p >= 0.5 else NO
        if side == "underdog":
            return NO if p >= 0.5 else YES
        lookback = next((c.lookback for c in self.genome.entry if c.lookback), 300)
        change = view.probability_change(lookback)
        if not change:
            return None
        rising = change > 0
        return (YES if rising else NO) if side == "momentum" else (NO if rising else YES)

    def should_enter(self, view: MarketView) -> Signal | None:
        g = self.genome
        combine = all if g.entry_logic == "all" else any
        if not g.entry or not combine(c.test(view) for c in g.entry):
            return None
        side = self._side(view)
        return Signal(side) if side else None

    def should_exit(self, view: MarketView, position: Any) -> str | None:
        g = self.genome
        combine = all if g.exit_logic == "all" else any
        return "genome_exit" if g.exits and combine(c.test(view) for c in g.exits) else None


def random_condition(rng: random.Random, space: GeneSpace) -> Condition:
    name = rng.choice(list(INDICATORS))
    low, high, uses_lookback = INDICATORS[name]
    return Condition(name, rng.choice(OPERATORS), round(rng.uniform(low, high), 4),
                     rng.choice(space.lookbacks) if uses_lookback else 0)


def random_genome(rng: random.Random, space: GeneSpace) -> Genome:
    optional = lambda chance, low, high: round(rng.uniform(low, high), 3) if rng.random() < chance else None  # noqa: E731
    return Genome(
        tuple(random_condition(rng, space) for _ in range(rng.randint(1, space.max_conditions))), rng.choice(LOGIC),
        tuple(random_condition(rng, space) for _ in range(rng.randint(0, 2))), rng.choice(LOGIC), rng.choice(SIDES),
        round(rng.uniform(0.01, 0.10), 3), optional(0.5, 0.05, 0.5), optional(0.5, 0.05, 1.0), optional(0.2, 0.05, 0.4),
        rng.choice(space.holding_seconds) if rng.random() < 0.4 else None)


def mutate(genome: Genome, rng: random.Random, space: GeneSpace, strength: float = 0.15) -> Genome:
    """Change one component, so each child differs from its parent in a traceable way."""
    def nudge(condition: Condition) -> Condition:
        low, high, uses_lookback = INDICATORS[condition.indicator]
        roll = rng.random()
        if roll < 0.6:
            threshold = min(high, max(low, condition.threshold + rng.gauss(0, strength * (high - low))))
            return replace(condition, threshold=round(threshold, 4))
        if roll < 0.8:
            return replace(condition, operator="<" if condition.operator == ">" else ">")
        return replace(condition, lookback=rng.choice(space.lookbacks)) if uses_lookback else random_condition(rng, space)

    def edit(conditions: tuple[Condition, ...], minimum: int) -> tuple[Condition, ...]:
        items = list(conditions)
        roll = rng.random()
        if roll < 0.2 and len(items) < space.max_conditions:
            items.append(random_condition(rng, space))
        elif roll < 0.4 and len(items) > minimum:
            items.pop(rng.randrange(len(items)))
        elif items:
            index = rng.randrange(len(items))
            items[index] = nudge(items[index])
        return tuple(items)

    def scale(value: float | None, low: float, high: float) -> float | None:
        if value is None:
            return round(rng.uniform(low, high), 3)
        return None if rng.random() < 0.2 else round(min(high, max(low, value * rng.uniform(0.7, 1.3))), 3)

    choice = rng.choice(("entry", "entry", "entry", "exits", "logic", "side", "size", "stop", "take", "trail", "hold"))
    if choice == "entry":
        return replace(genome, entry=edit(genome.entry, 1))
    if choice == "exits":
        return replace(genome, exits=edit(genome.exits, 0))
    if choice == "logic":
        return replace(genome, entry_logic="any" if genome.entry_logic == "all" else "all")
    if choice == "side":
        return replace(genome, side=rng.choice(SIDES))
    if choice == "size":
        return replace(genome, position_fraction=round(min(0.15, max(0.005, genome.position_fraction * rng.uniform(0.7, 1.3))), 3))
    if choice == "stop":
        return replace(genome, stop_loss=scale(genome.stop_loss, 0.05, 0.5))
    if choice == "take":
        return replace(genome, take_profit=scale(genome.take_profit, 0.05, 1.0))
    if choice == "trail":
        return replace(genome, trailing_stop=scale(genome.trailing_stop, 0.05, 0.4))
    return replace(genome, max_holding_seconds=None if rng.random() < 0.3 else rng.choice(space.holding_seconds))


def crossover(primary: Individual, secondary: Individual, rng: random.Random, space: GeneSpace,
              origin: str = "crossover", primary_bias: float = 0.5) -> Individual:
    """Build a child one component at a time, and record which parent supplied each."""
    a, b = primary.genome, secondary.genome
    sources: dict[str, str] = {}

    def pick(component: str, x: Any, y: Any) -> Any:
        take_first = rng.random() < primary_bias
        sources[component] = primary.id if take_first else secondary.id
        return x if take_first else y

    def merge(component: str, xs: tuple[Condition, ...], ys: tuple[Condition, ...], minimum: int) -> tuple[Condition, ...]:
        """Conditions are inherited singly. Shared indicators can also blend their thresholds."""
        chosen: list[Condition] = []
        by_indicator = {c.indicator: c for c in ys}
        used = []
        for condition in xs:
            match = by_indicator.get(condition.indicator)
            if match is not None and match.operator == condition.operator and rng.random() < 0.5:
                weight = rng.random()
                condition = replace(condition, threshold=round(weight * condition.threshold + (1 - weight) * match.threshold, 4))
                used.append("blend")
            if rng.random() < primary_bias:
                chosen.append(condition)
                used.append(primary.id)
        for condition in ys:
            if rng.random() < 1 - primary_bias:
                chosen.append(condition)
                used.append(secondary.id)
        unique = list(dict.fromkeys(chosen))[: space.max_conditions]
        if len(unique) < minimum:
            unique = list((xs or ys)[:minimum])
            used.append(primary.id)
        sources[component] = ",".join(dict.fromkeys(used)) or "none"
        return tuple(unique)

    child = Genome(
        merge("entry_conditions", a.entry, b.entry, 1), pick("entry_logic", a.entry_logic, b.entry_logic),
        merge("exit_conditions", a.exits, b.exits, 0), pick("exit_logic", a.exit_logic, b.exit_logic),
        pick("side", a.side, b.side), pick("position_sizing", a.position_fraction, b.position_fraction),
        pick("stop_loss", a.stop_loss, b.stop_loss), pick("take_profit", a.take_profit, b.take_profit),
        pick("trailing_stop", a.trailing_stop, b.trailing_stop), pick("max_holding", a.max_holding_seconds, b.max_holding_seconds))
    return Individual(child, origin, (primary.id, secondary.id), sources)
