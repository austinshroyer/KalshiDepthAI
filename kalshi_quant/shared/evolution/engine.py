"""Category-specific, cross-category and hybrid evolution.

Data discipline: time splits are shared by every population. Search uses train. Saving and
finalist selection use validation. Test is read only after all three population types have
finished, once per finalist, so nothing that evolved ever saw it.

Stages for each population: N independent cycles, each saving its best strategies; crossover
across that archive; a final cycle. Hybrids are then bred from the finalists of the others.
"""
from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select

from kalshi_quant.shared.backtest.dataset import Dataset
from kalshi_quant.shared.backtest.engine import BacktestSettings
from kalshi_quant.shared.config import PROJECT_ROOT, Env, EnvConfig
from kalshi_quant.shared.database.engine import Database
from kalshi_quant.shared.evolution.evaluation import Evaluator, FoldSpec
from kalshi_quant.shared.evolution.fitness import FAILED, FitnessConfig, fitness, generalization_score
from kalshi_quant.shared.evolution.genome import Genome, Individual, crossover, mutate, random_genome
from kalshi_quant.shared.evolution.report import build_report, render_markdown
from kalshi_quant.shared.evolution.settings import EvolutionSettings, PopulationSettings, slug
from kalshi_quant.shared.optimization.statistics import benjamini_hochberg_threshold
from kalshi_quant.shared.utilities.logging import get_logger
from kalshi_quant.shared.utilities.timeutil import now_ms

log = get_logger(__name__)
CATEGORY, CROSS, HYBRID = "category", "cross_category", "hybrid"
FITNESS_KEY = {CATEGORY: "category", CROSS: "cross_category", HYBRID: "hybrid"}
HYBRID_MODES = ("same_category", "cross_category", "category_to_cross", "cross_to_category")
ENV_DIRECTORY = {Env.GENERAL: "general_markets", Env.FIFTEEN_MINUTE: "fifteen_minute"}


@dataclass
class Population:
    name: str               # the Kalshi category, or cross_category, or hybrid
    kind: str
    tickers: frozenset[str]
    categories: list[str]   # categories it trains on
    finalists: list[Individual] = field(default_factory=list)

    @property
    def scope(self) -> str:
        return slug(self.name)


def plan_categories(dataset: Dataset, eligibility: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Group resolved markets by Kalshi category. Thin categories are reported, not forced."""
    groups: dict[str, list[str]] = {}
    for ticker in dataset.settlements:
        groups.setdefault(dataset.categories.get(ticker) or "uncategorized", []).append(ticker)
    index = {t: i for i, t in enumerate(dataset.tickers)}
    counts = np.bincount(dataset.ticker_index, minlength=len(dataset.tickers)) if len(dataset) else np.zeros(len(dataset.tickers))
    eligible, insufficient = {}, {}
    for category, tickers in sorted(groups.items()):
        rows = int(sum(counts[index[t]] for t in tickers))
        days = _span_days(dataset, tickers, index)
        reasons = []
        if len(tickers) < eligibility["min_markets"]:
            reasons.append(f"{len(tickers)} resolved markets, need {eligibility['min_markets']}")
        if rows < eligibility["min_rows"]:
            reasons.append(f"{rows} rows, need {eligibility['min_rows']}")
        if days < eligibility["min_days"]:
            reasons.append(f"{days:.1f} days of history, need {eligibility['min_days']}")
        if reasons:
            insufficient[category] = "; ".join(reasons)
        else:
            eligible[category] = sorted(tickers)
    return eligible, insufficient


def _span_days(dataset: Dataset, tickers: list[str], index: dict[str, int]) -> float:
    ids = np.array([index[t] for t in tickers])
    times = dataset.ts_ms[np.isin(dataset.ticker_index, ids)]
    return float(times.max() - times.min()) / 86_400_000 if len(times) else 0.0


def balanced_sample(groups: dict[str, list[str]], cap: int, seed: int) -> frozenset[str]:
    """Equal market counts per category, so the largest cannot dominate training."""
    size = min(cap, min(len(v) for v in groups.values()))
    rng = random.Random(seed)
    return frozenset(t for tickers in groups.values() for t in rng.sample(tickers, size))


class EvolutionEngine:
    def __init__(self, db: Database, config: EnvConfig, dataset: Dataset, settings: EvolutionSettings, run_id: str | None = None,
                 workers: int = 1, seed: int = 0, archive_root: Path | None = None, only_categories: list[str] | None = None,
                 progress: Callable[[str], None] | None = None):
        self.db, self.config, self.dataset, self.settings = db, config, dataset, settings
        self.env = config.env
        self.seed = seed
        self.workers = workers
        self.run_id = run_id or f"evo-{now_ms()}"
        self.root = (archive_root or PROJECT_ROOT / "saved_strategies") / ENV_DIRECTORY[self.env]
        self.backtest = BacktestSettings.from_config(config, seed=seed)
        self.progress = progress or (lambda message: None)
        self.lineage: dict[str, dict[str, Any]] = {}
        self.only = only_categories
        eligible, self.insufficient = plan_categories(dataset, settings.eligibility)
        if only_categories:
            eligible = {k: v for k, v in eligible.items() if k in only_categories}
        self.groups = eligible
        self.populations = [Population(c, CATEGORY, frozenset(t), [c]) for c, t in eligible.items()]
        self.cross: Population | None = None
        self.cross_note = None
        if len(eligible) >= 2:
            self.cross = Population(CROSS, CROSS, balanced_sample(eligible, settings.cross["max_markets_per_category"], seed), sorted(eligible))
        else:
            self.cross_note = f"needs two eligible categories, found {len(eligible)}"
        self.hybrid: Population | None = None
        self.specs = self._fold_specs()
        self._open_run()

    # Folds
    def _boundaries(self) -> tuple[int, int, int, int]:
        start, end = self.dataset.time_range()
        v = self.settings.validation
        span = end - start
        train_end = start + int(span * v["train_fraction"])
        return start, train_end, train_end + int(span * v["validation_fraction"]), end + 1

    def _fold_specs(self) -> dict[str, FoldSpec]:
        start, train_end, validation_end, end = self._boundaries()
        v = self.settings.validation
        specs: dict[str, FoldSpec] = {}
        scopes: dict[str, frozenset[str] | None] = {p.scope: p.tickers for p in self.populations}
        if self.cross:
            scopes[CROSS] = self.cross.tickers
            scopes[HYBRID] = self.cross.tickers
        elif self.populations:
            scopes[HYBRID] = self.populations[0].tickers
        for scope, tickers in scopes.items():
            specs[f"{scope}/train"] = FoldSpec(tickers, start, train_end)
            specs[f"{scope}/validation"] = FoldSpec(tickers, train_end, validation_end)
            specs[f"{scope}/test"] = FoldSpec(tickers, validation_end, end)
            specs[f"{scope}/oos"] = FoldSpec(tickers, train_end, end)
            windows = np.linspace(validation_end, end, v["forward_windows"] + 1, dtype=np.int64)
            for k in range(v["forward_windows"]):
                specs[f"{scope}/forward{k}"] = FoldSpec(tickers, int(windows[k]), int(windows[k + 1]))
            folds = v["walk_forward_folds"]
            step = (end - start) // (folds + 1)
            for k in range(folds):
                cut = start + step * (k + 1)
                specs[f"{scope}/wf{k}/train"] = FoldSpec(tickers, start, cut)
                specs[f"{scope}/wf{k}/test"] = FoldSpec(tickers, cut, cut + step if k < folds - 1 else end)
        if self.cross:
            for category in self.groups:
                held_out = frozenset(self.cross.tickers - frozenset(self.groups[category]))
                specs[f"loco/{slug(category)}/train"] = FoldSpec(held_out, start, train_end)
        return specs

    # Persistence
    def _open_run(self) -> None:
        table = self.db.t["evolution_runs"]
        with self.db.begin() as conn:
            if not conn.execute(select(table.c.run_id).where(table.c.run_id == self.run_id)).first():
                conn.execute(table.insert().values(
                    run_id=self.run_id, created_ms=now_ms(), status="running", dataset_id=self.dataset.dataset_id(),
                    config=json.dumps({"settings": self.settings.data, "profile": self.settings.profile, "seed": self.seed}, default=str)))

    def _stored(self, population: Population) -> dict[str, list[Individual]]:
        """Stages already finished, for resuming an interrupted run."""
        t = self.db.t["evolution_strategies"]
        stages: dict[str, list[Individual]] = {}
        with self.db.connect() as conn:
            rows = conn.execute(select(t).where(t.c.run_id == self.run_id, t.c.population == population.name).order_by(t.c.stage, t.c.rank)).mappings()
            for row in rows:
                individual = Individual(Genome.from_json(row["genome"]), row["origin"], tuple(json.loads(row["parents"])), json.loads(row["components"]))
                stages.setdefault(row["stage"], []).append(individual)
                self._note(individual, population, row["stage"])
        return stages

    def _note(self, individual: Individual, population: Population, stage: str) -> None:
        self.lineage.setdefault(individual.id, {"origin": individual.origin, "parents": list(individual.parents), "population": population.name,
                                                "stage": stage, "components": individual.components})

    def _save(self, population: Population, stage: str, ranked: list[tuple[Individual, float, dict[str, Any]]]) -> None:
        directory = self.root / population.scope / stage
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for rank, (individual, score, metrics) in enumerate(ranked, 1):
            self._note(individual, population, stage)
            record = {"id": individual.id, "population": population.name, "population_type": population.kind, "stage": stage, "rank": rank,
                      "fitness": score, "origin": individual.origin, "parents": list(individual.parents), "components": individual.components,
                      "genome": json.loads(individual.genome.to_json()), "metrics": metrics}
            (directory / f"rank_{rank:02d}.json").write_text(json.dumps(record, indent=1, default=str))
            rows.append({"run_id": self.run_id, "population": population.name, "stage": stage, "rank": rank, "strategy_id": individual.id,
                         "population_type": population.kind, "genome": individual.genome.to_json(), "origin": individual.origin,
                         "parents": json.dumps(list(individual.parents)), "components": json.dumps(individual.components),
                         "fitness": score if score > FAILED / 2 else None, "metrics": json.dumps(metrics, default=str)})
        if rows:
            with self.db.begin() as conn:
                conn.execute(self.db.upsert("evolution_strategies").on_conflict_do_nothing(
                    index_elements=["run_id", "population", "stage", "rank"]), rows)

    # Search
    def _fitness_config(self, kind: str) -> FitnessConfig:
        return self.settings.fitness(FITNESS_KEY[kind])

    def _breed(self, ranked: list[Individual], rng: random.Random, p: PopulationSettings, size: int) -> list[Individual]:
        space = self.settings.gene_space
        elites = ranked[: max(1, int(size * p.elite_fraction))]
        children: list[Individual] = list(elites)
        seen = {i.id for i in children}

        def tournament() -> Individual:
            return min(rng.sample(range(len(ranked)), min(p.tournament_size, len(ranked))), key=lambda i: i)

        attempts = 0
        while len(children) < size and attempts < size * 20:
            attempts += 1
            first = ranked[tournament()]
            child = crossover(first, ranked[tournament()], rng, space, _inherit(first.origin)) if rng.random() < p.crossover_rate else \
                Individual(first.genome, first.origin, (first.id,))
            if rng.random() < p.mutation_rate or child.genome == first.genome:
                child = Individual(mutate(child.genome, rng, space), child.origin if child.parents[1:] else "mutation", child.parents, child.components)
            if child.id not in seen:
                seen.add(child.id)
                children.append(child)
        return children

    def _evolve(self, initial: list[Individual], fold: str, epochs: int, p: PopulationSettings, config: FitnessConfig,
                evaluator: Evaluator, rng: random.Random, breadth: int, size: int | None = None) -> list[tuple[Individual, float]]:
        population = initial
        size = size or p.population_size
        ranked: list[tuple[Individual, float]] = []
        for epoch in range(max(epochs, 1)):
            for individual in population:  # intermediates too, or lineage breaks between saved strategies
                self._note(individual, self._active, f"epoch_{epoch:03d}")
            summaries = evaluator.evaluate([i.genome for i in population], fold)
            ranked = sorted(((i, fitness(summaries[i.id], config, breadth)) for i in population), key=lambda item: item[1], reverse=True)
            population = self._breed([i for i, _ in ranked], rng, p, size)
        return ranked

    def _select(self, ranked: list[tuple[Individual, float]], population: Population, keep: int, config: FitnessConfig,
                evaluator: Evaluator, breadth: int) -> list[tuple[Individual, float, dict[str, Any]]]:
        """Re-rank train leaders on validation. This is what limits memorized training data."""
        shortlist = ranked[: keep * 3]
        validation = evaluator.evaluate([i.genome for i, _ in shortlist], f"{population.scope}/validation")
        scored = []
        for individual, train_score in shortlist:
            summary = validation[individual.id]
            scored.append((individual, fitness(summary, config, breadth), {"train_fitness": train_score, "validation": summary}))
        scored.sort(key=lambda item: (item[1], item[2]["train_fitness"]), reverse=True)
        return scored[:keep]

    def run_population(self, population: Population, evaluator: Evaluator) -> None:
        self._active = population
        settings = self.settings.population(population.name)
        config = self._fitness_config(population.kind)
        breadth = len(population.categories)
        stored = self._stored(population)
        fold = f"{population.scope}/train"
        archive: list[Individual] = []
        for cycle in range(1, settings.cycles + 1):
            stage = f"cycle_{cycle:03d}"
            if stage in stored:
                archive += stored[stage]
                continue
            rng = random.Random(f"{self.seed}:{self.env.value}:{population.name}:{cycle}")
            initial = [Individual(random_genome(rng, self.settings.gene_space)) for _ in range(settings.population_size)]
            ranked = self._evolve(initial, fold, settings.epochs, settings, config, evaluator, rng, breadth)
            saved = self._select(ranked, population, settings.saved_per_cycle, config, evaluator, breadth)
            self._save(population, stage, saved)
            archive += [i for i, _, _ in saved]
            self.progress(f"{population.name}: {stage}, best validation fitness {saved[0][1]:.2f}" if saved else stage)
        if "final" in stored:
            population.finalists = stored["final"]
            return
        rng = random.Random(f"{self.seed}:{self.env.value}:{population.name}:final")
        unique = list({i.id: i for i in archive}.values())
        seeds = list(unique)
        while len(seeds) < settings.population_size * 2 and len(unique) >= 2:  # hybridize across this population's archive
            first, second = rng.sample(unique, 2)
            seeds.append(crossover(first, second, rng, self.settings.gene_space, "archive_crossover"))
        ranked = self._evolve(seeds[: settings.population_size * 2], fold, settings.final_epochs, settings, config, evaluator, rng, breadth)
        saved = self._select(ranked, population, settings.final_saved, config, evaluator, breadth)
        self._save(population, "final", saved)
        population.finalists = [i for i, _, _ in saved]

    # Hybrids
    def _hybrid_seeds(self, rng: random.Random) -> tuple[list[Individual], dict[str, int]]:
        per_mode = self.settings.hybrid["per_mode"]
        space = self.settings.gene_space
        categories = [p for p in self.populations if p.finalists]
        cross = self.cross.finalists if self.cross else []
        seeds: list[Individual] = []
        made = dict.fromkeys(HYBRID_MODES, 0)

        def add(mode: str, primary: Individual, secondary: Individual) -> None:
            seeds.append(crossover(primary, secondary, rng, space, mode, 0.65))
            made[mode] += 1

        for _ in range(per_mode):
            usable = [p for p in categories if len(p.finalists) >= 2]
            if usable:
                add("same_category", *rng.sample(rng.choice(usable).finalists, 2))
            if len(categories) >= 2:
                a, b = rng.sample(categories, 2)
                add("cross_category", rng.choice(a.finalists), rng.choice(b.finalists))
            if categories and cross:
                add("category_to_cross", rng.choice(rng.choice(categories).finalists), rng.choice(cross))
                add("cross_to_category", rng.choice(cross), rng.choice(rng.choice(categories).finalists))
        return seeds, made

    def run_hybrid(self, evaluator: Evaluator) -> dict[str, Any]:
        tickers = self.cross.tickers if self.cross else (self.populations[0].tickers if self.populations else frozenset())
        self.hybrid = self._active = Population(HYBRID, HYBRID, tickers, sorted(self.groups))
        stored = self._stored(self.hybrid)
        if "final" in stored:
            self.hybrid.finalists = stored["final"]
            return {"resumed": True}
        rng = random.Random(f"{self.seed}:{self.env.value}:hybrid")
        seeds, made = self._hybrid_seeds(rng)
        if not seeds:
            return {"modes": made, "note": "no parents available"}
        settings = self.settings.population(CROSS)
        size, epochs = self.settings.hybrid["population_size"], self.settings.hybrid["epochs"]
        config = self._fitness_config(HYBRID)
        ranked = self._evolve(seeds, f"{HYBRID}/train", epochs, settings, config, evaluator, rng, len(self.groups), size)
        saved = self._select(ranked, self.hybrid, settings.final_saved, config, evaluator, len(self.groups))
        self._save(self.hybrid, "final", saved)
        self.hybrid.finalists = [i for i, _, _ in saved]
        return {"modes": made}

    # Validation
    def _ancestors(self, strategy_id: str) -> list[dict[str, Any]]:
        seen, chain, frontier = set(), [], [strategy_id]
        while frontier:
            current = frontier.pop()
            if current in seen or current not in self.lineage:
                continue
            seen.add(current)
            entry = self.lineage[current]
            chain.append({"id": current, **{k: entry[k] for k in ("origin", "population", "stage", "parents")}})
            frontier += entry["parents"]
        return chain

    def _familiar(self, individual: Individual) -> list[str]:
        """Categories of a hybrid's direct parents' populations. Anything else is unfamiliar to it."""
        found = set()
        for ancestor in self._ancestors(individual.id):
            if ancestor["population"] in self.groups:
                found.add(ancestor["population"])
            elif ancestor["population"] == CROSS:
                found |= set(self.groups)
        return sorted(found)

    def _validate_population(self, population: Population, evaluator: Evaluator) -> dict[str, Any]:
        """The only place test folds are read."""
        genomes = [i.genome for i in population.finalists]
        scope = population.scope
        config = self._fitness_config(population.kind)
        records = []
        if not genomes:
            return {"finalists": [], "rejected": [], "candidates": 0}
        test = evaluator.evaluate(genomes, f"{scope}/test")
        validation = evaluator.evaluate(genomes, f"{scope}/validation")
        training = evaluator.evaluate(genomes, f"{scope}/train")
        forward = [evaluator.evaluate(genomes, f"{scope}/forward{k}") for k in range(self.settings.validation["forward_windows"])]
        elsewhere = {c: evaluator.evaluate(genomes, f"{slug(c)}/test") for c in self.groups}
        p_values = [test[g.fingerprint()]["p_value"] for g in genomes]
        known = [p for p in p_values if p is not None]
        threshold = benjamini_hochberg_threshold(known, self.settings.validation["fdr_alpha"]) if known else None
        for rank, individual in enumerate(population.finalists, 1):
            fp = individual.id
            summary = test[fp]
            by_category = {c: elsewhere[c][fp] for c in self.groups}
            home = population.categories if population.kind == CATEGORY else self._familiar(individual) if population.kind == HYBRID else sorted(self.groups)
            away = [c for c in self.groups if c not in home]
            reasons = []
            if summary["trades"] < self.settings.credibility["min_test_trades"]:
                reasons.append(f"only {summary['trades']} test trades")
            if summary["pnl"] <= 0:
                reasons.append("lost money on test")
            if validation[fp]["pnl"] <= 0:
                reasons.append("lost money on validation")
            if summary["p_value"] is None or threshold is None or summary["p_value"] > threshold:
                reasons.append("not significant after false-discovery control")
            general = None
            if population.kind != CATEGORY:
                general = generalization_score(summary, len(self.groups), config)
            if population.kind == CROSS and (general is None or general["score"] < self.settings.credibility["cross_min_generalization"]):
                reasons.append("generalization score below threshold")
            records.append({
                "strategy_id": fp, "rank": rank, "strategy_type": population.kind, "population": population.name,
                "intended_categories": home, "training_categories": population.categories,
                "validation_categories": population.categories, "out_of_sample": _brief(summary), "validation": _brief(validation[fp]),
                "train": _brief(training[fp]), "category_performance": summary["categories"],
                "other_category_tests": {c: _brief(by_category[c]) for c in away}, "home_category_tests": {c: _brief(by_category[c]) for c in home},
                "forward_windows": [_brief(window[fp]) for window in forward], "generalization": general,
                "credible": not reasons, "rejection_reasons": reasons, "features": individual.genome.features(),
                "genome": json.loads(individual.genome.to_json()), "origin": individual.origin,
                "components": individual.components, "lineage": self._ancestors(fp)})
        return {"finalists": [r for r in records if r["credible"]], "rejected": [r for r in records if not r["credible"]], "candidates": len(records)}

    def _quick_best(self, fold: str, population: Population, evaluator: Evaluator, epochs: int, size: int, seed: str) -> Individual | None:
        rng = random.Random(seed)
        self._active = population
        settings = self.settings.population(population.name)
        initial = [Individual(random_genome(rng, self.settings.gene_space)) for _ in range(size)]
        config = self._fitness_config(population.kind)
        if population.kind == CROSS:
            config = FitnessConfig(**{**config.__dict__, "min_categories": min(config.min_categories, max(1, len(self.groups) - 1))})
        ranked = self._evolve(initial, fold, epochs, settings, config, evaluator, rng, max(1, len(population.categories) - 1), size)
        return ranked[0][0] if ranked and ranked[0][1] > FAILED / 2 else None

    def _walk_forward(self, population: Population, evaluator: Evaluator) -> dict[str, Any]:
        """Tests the search process: re-evolve on each window, then trade the next one."""
        v = self.settings.validation
        folds = []
        for k in range(v["walk_forward_folds"]):
            best = self._quick_best(f"{population.scope}/wf{k}/train", population, evaluator, v["walk_forward_epochs"], v["walk_forward_population"],
                                    f"{self.seed}:wf:{population.name}:{k}")
            if best is None:
                folds.append({"fold": k, "note": "no strategy met the trade minimum"})
                continue
            folds.append({"fold": k, **_brief(evaluator.evaluate([best.genome], f"{population.scope}/wf{k}/test")[best.id])})
        tested = [f for f in folds if "return" in f]
        return {"folds": folds, "profitable": sum(f["return"] > 0 for f in tested), "evaluated": len(tested)}

    def _leave_one_out(self, evaluator: Evaluator) -> dict[str, Any]:
        """Evolve without a category, then test on it, after the training period."""
        if not self.cross:
            return {"note": self.cross_note}
        v = self.settings.validation
        results = {}
        for category in sorted(self.groups):
            best = self._quick_best(f"loco/{slug(category)}/train", self.cross, evaluator, v["leave_one_out_epochs"], v["leave_one_out_population"],
                                    f"{self.seed}:loco:{category}")
            if best is None:
                results[category] = {"note": "no strategy met the trade minimum without this category"}
                continue
            results[category] = {"withheld": _brief(evaluator.evaluate([best.genome], f"{slug(category)}/oos")[best.id]),
                                 "strategy_id": best.id, "trained_on": [c for c in sorted(self.groups) if c != category]}
        return results

    def run(self) -> dict[str, Any]:
        with Evaluator(self.dataset, self.specs, self.backtest, self.workers, PROJECT_ROOT / "data" / "cache",
                       self.settings.data["fitness"]["cross_category"]["min_trades_per_category"]) as evaluator:
            self.evaluator = evaluator
            for population in self.populations:
                self.run_population(population, evaluator)
            if self.cross:
                self.run_population(self.cross, evaluator)
            hybrid_info = self.run_hybrid(evaluator)
            everything = [*self.populations, *([self.cross] if self.cross else []), self.hybrid]
            results = {p.name: {"kind": p.kind, "categories": p.categories, **self._validate_population(p, evaluator)} for p in everything}
            for population in everything[:-1]:
                results[population.name]["walk_forward"] = self._walk_forward(population, evaluator)
            leave_one_out = self._leave_one_out(evaluator)
            stats = {"simulations": evaluator.simulations, "requests": evaluator.requests}
        report = build_report(self.env.value, self.run_id, self.settings, results, leave_one_out, self.insufficient, self.cross_note, hybrid_info, stats)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "finalists.json").write_text(json.dumps(report, indent=1, default=str))
        (self.root / "research_report.md").write_text(render_markdown(report))
        self._export(results)
        table = self.db.t["evolution_runs"]
        with self.db.begin() as conn:
            conn.execute(table.update().where(table.c.run_id == self.run_id).values(status="finished", report=json.dumps(report, default=str)))
        return report

    def _export(self, results: dict[str, Any]) -> None:
        """Credible finalists become ordinary strategy files for backtest and paper trading."""
        for name, result in results.items():
            directory = self.root / slug(name) / "final"
            for record in result["finalists"]:
                genome = json.dumps(json.dumps(record["genome"], sort_keys=True))
                (directory / f"strategy_{record['rank']:02d}.py").write_text(
                    f'"""Evolved in run {self.run_id}, population {name}. Lineage is in finalists.json."""\n'
                    "from kalshi_quant.shared.evolution.genome import GenomeStrategy\n"
                    "from kalshi_quant.shared.strategy.base import Param\n\n\n"
                    "class Evolved(GenomeStrategy):\n"
                    f'    name = "evolved_{slug(name)}_{record["strategy_id"]}"\n'
                    f'    environments = ("{self.env.value}",)\n'
                    f"    parameters = {{\"genome\": Param({genome})}}\n")


def _inherit(origin: str) -> str:
    return origin if origin in HYBRID_MODES else "crossover"


def _brief(s: dict[str, Any]) -> dict[str, Any]:
    keys = ("return", "pnl", "trades", "win_rate", "profit_factor", "max_drawdown", "average_trade_return", "risk_adjusted", "p_value", "category_consistency")
    return {k: s.get(k) for k in keys}
