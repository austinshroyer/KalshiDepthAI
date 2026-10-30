"""Evaluates genomes on named folds, in parallel, with caching.

A fold is a definition (tickers, start, end), not a copy of the data. Workers hold one base
dataset and build folds on demand. Identical genomes are never simulated twice, which matters
because a full run makes hundreds of thousands of requests.
"""
from __future__ import annotations

import multiprocessing
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kalshi_quant.shared.backtest.dataset import Dataset
from kalshi_quant.shared.backtest.engine import BacktestSettings, run_backtest
from kalshi_quant.shared.evolution.fitness import summarize
from kalshi_quant.shared.evolution.genome import Genome, GenomeStrategy

_WORKER: dict[str, Any] = {}
FOLD_CACHE = 12


@dataclass(frozen=True)
class FoldSpec:
    tickers: frozenset[str] | None
    start_ms: int
    end_ms: int


def build_fold(base: Dataset, spec: FoldSpec) -> Dataset:
    selected = base if spec.tickers is None else base.restrict(spec.tickers)
    return selected.slice(spec.start_ms, spec.end_ms)


class _FoldStore:
    def __init__(self, base: Dataset, specs: dict[str, FoldSpec]):
        self.base, self.specs = base, specs
        self.cache: OrderedDict[str, Dataset] = OrderedDict()

    def get(self, name: str) -> Dataset:
        if name not in self.cache:
            self.cache[name] = build_fold(self.base, self.specs[name])
            while len(self.cache) > FOLD_CACHE:
                self.cache.popitem(last=False)
        return self.cache[name]


def evaluate_one(store: _FoldStore, settings: BacktestSettings, fold: str, genome_json: str, min_per_category: int) -> dict[str, Any]:
    dataset = store.get(fold)
    if not len(dataset):
        return summarize([], settings.starting_capital, {}, min_per_category)
    result = run_backtest(GenomeStrategy({"genome": genome_json}), dataset, settings)
    return summarize(result.trades, settings.starting_capital, dataset.categories, min_per_category)


def _init(path: str, specs: dict[str, FoldSpec], settings: dict[str, Any], sys_path: list[str]) -> None:
    sys.path[:0] = [p for p in sys_path if p not in sys.path]
    _WORKER["store"] = _FoldStore(Dataset.load(Path(path)), specs)
    _WORKER["settings"] = BacktestSettings.from_json(settings)


def _task(arguments: tuple[str, str, int]) -> dict[str, Any]:
    return evaluate_one(_WORKER["store"], _WORKER["settings"], *arguments)


class Evaluator:
    def __init__(self, base: Dataset, specs: dict[str, FoldSpec], settings: BacktestSettings, workers: int,
                 cache_dir: Path, min_per_category: int = 5):
        self.base, self.specs, self.settings, self.workers = base, specs, settings, workers
        self.min_per_category = min_per_category
        self.cache: dict[tuple[str, str], dict[str, Any]] = {}
        self.simulations = 0
        self.requests = 0
        self.touched: list[str] = []  # every fold used, in order, so tests can audit the test set
        self.store = _FoldStore(base, specs)
        self.pool: ProcessPoolExecutor | None = None
        self.cache_dir = cache_dir

    def __enter__(self) -> "Evaluator":
        if self.workers > 1:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.cache_dir / f"evolution-{self.base.dataset_id()}.pkl"
            if not path.exists():
                self.base.save(path)
            self.path = path
            self.pool = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context("spawn"), initializer=_init,
                                            initargs=(str(path), self.specs, self.settings.to_json(), list(sys.path)))
        return self

    def __exit__(self, *_: object) -> None:
        if self.pool:
            self.pool.shutdown()
            self.path.unlink(missing_ok=True)

    def evaluate(self, genomes: list[Genome], fold: str) -> dict[str, dict[str, Any]]:
        """Summaries keyed by fingerprint."""
        self.touched.append(fold)
        unique = {g.fingerprint(): g for g in genomes}
        self.requests += len(genomes)
        todo = [(fp, g) for fp, g in unique.items() if (fold, fp) not in self.cache]
        self.simulations += len(todo)
        arguments = [(fold, g.to_json(), self.min_per_category) for _, g in todo]
        if self.pool and len(todo) > 1:
            results = list(self.pool.map(_task, arguments, chunksize=max(1, len(todo) // (self.workers * 4))))
        else:
            results = [evaluate_one(self.store, self.settings, *a) for a in arguments]
        for (fp, _), result in zip(todo, results):
            self.cache[(fold, fp)] = result
        return {fp: self.cache[(fold, fp)] for fp in unique}
