"""Evolution settings: defaults, profiles and per-population overrides."""
from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kalshi_quant.shared.config import CONFIG_DIR, Env
from kalshi_quant.shared.evolution.fitness import FitnessConfig
from kalshi_quant.shared.evolution.genome import GeneSpace


@dataclass
class PopulationSettings:
    cycles: int
    epochs: int
    population_size: int
    saved_per_cycle: int
    final_epochs: int
    final_saved: int
    elite_fraction: float
    tournament_size: int
    crossover_rate: float
    mutation_rate: float


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


class EvolutionSettings:
    def __init__(self, env: Env, profile: str | None = None, path: Path | None = None, extra: dict[str, Any] | None = None):
        raw = yaml.safe_load((path or CONFIG_DIR / "evolution.yaml").read_text())
        data = deepcopy(raw)
        if profile:
            if profile not in raw["profiles"]:
                raise ValueError(f"unknown profile {profile!r}; choose from {sorted(raw['profiles'])}")
            _merge(data, raw["profiles"][profile])
        if extra:
            _merge(data, extra)
        self.env, self.profile, self.data = env, profile, data
        self.hybrid = data["hybrid"]
        self.validation = data["validation"]
        self.eligibility = data["eligibility"]
        self.credibility = data["credibility"]
        self.cross = data["cross_category"]
        space = data["gene_space"][env.value]
        self.gene_space = GeneSpace(tuple(space["lookbacks"]), tuple(space["holding_seconds"]))

    def population(self, name: str) -> PopulationSettings:
        """Defaults, then this population's override for this environment."""
        values = {**self.data["defaults"], **((self.data.get("overrides") or {}).get(self.env.value, {}).get(name) or {})}
        return PopulationSettings(**values)

    def fitness(self, population_type: str) -> FitnessConfig:
        return FitnessConfig(**self.data["fitness"][population_type])


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "uncategorized"
