"""Parameter spaces and search methods."""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Dimension:
    name: str
    values: tuple[Any, ...]


def parse_space(spec: dict[str, dict[str, Any]]) -> list[Dimension]:
    """{name: {min, max, step}} or {name: {values: [...]}}."""
    dimensions = []
    for name, entry in spec.items():
        if "values" in entry:
            values = tuple(entry["values"])
        else:
            low, high, step = entry["min"], entry["max"], entry["step"]
            count = int(round((high - low) / step)) + 1
            is_int = all(float(x).is_integer() for x in (low, high, step))
            values = tuple(int(low + i * step) if is_int else round(low + i * step, 10) for i in range(count))
        if not values:
            raise ValueError(f"empty range for {name}")
        dimensions.append(Dimension(name, values))
    return dimensions


def space_size(space: list[Dimension]) -> int:
    size = 1
    for dimension in space:
        size *= len(dimension.values)
    return size


def grid(space: list[Dimension]) -> list[dict[str, Any]]:
    names = [d.name for d in space]
    return [dict(zip(names, combo)) for combo in itertools.product(*(d.values for d in space))]


def random_sample(space: list[Dimension], count: int, rng: random.Random) -> list[dict[str, Any]]:
    seen: set[tuple] = set()
    samples = []
    for _ in range(min(count, space_size(space)) * 20):
        params = {d.name: rng.choice(d.values) for d in space}
        key = tuple(params.values())
        if key not in seen:
            seen.add(key)
            samples.append(params)
            if len(samples) == count:
                break
    return samples


def adaptive_proposals(space: list[Dimension], elites: list[dict[str, Any]], count: int, rng: random.Random,
                       seen: set[tuple]) -> list[dict[str, Any]]:
    """Cross-entropy style: step near the best points so far, with occasional random jumps."""
    proposals = []
    for _ in range(count * 30):
        if len(proposals) == count:
            break
        base = rng.choice(elites) if elites and rng.random() > 0.2 else {d.name: rng.choice(d.values) for d in space}
        params = {}
        for dimension in space:
            index = dimension.values.index(base[dimension.name])
            index = min(len(dimension.values) - 1, max(0, index + rng.choice((-2, -1, 0, 0, 1, 2))))
            params[dimension.name] = dimension.values[index]
        key = tuple(params.values())
        if key not in seen:
            seen.add(key)
            proposals.append(params)
    return proposals


def neighbors(space: list[Dimension], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Every point one step away along a single dimension."""
    points = []
    for dimension in space:
        index = dimension.values.index(params[dimension.name])
        for other in (index - 1, index + 1):
            if 0 <= other < len(dimension.values):
                points.append({**params, dimension.name: dimension.values[other]})
    return points
