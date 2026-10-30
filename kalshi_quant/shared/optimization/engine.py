"""Parameter optimization built to find robust settings, not the luckiest history.

Workflow:
  1. Split time into train, validation and test. Test is touched once, at the very end.
  2. Search only on train.
  3. Re-score the best candidates on validation and pick there.
  4. Score the single pick on test and report the deterioration from train.
  5. Walk forward: re-pick on each expanding window and test on the next period, using a
     candidate set drawn without looking at any results, so no fold informs another.
  6. Sensitivity: a real edge should survive small parameter changes.
  7. Multiple testing: deflated Sharpe and CSCV overfitting probability.

Trials are checkpointed per (fold, params), so an interrupted run resumes.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing
import random
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select, text, update

from kalshi_quant.shared.backtest.dataset import Dataset
from kalshi_quant.shared.backtest.engine import BacktestSettings, run_backtest
from kalshi_quant.shared.config import PROJECT_ROOT
from kalshi_quant.shared.database.engine import Database
from kalshi_quant.shared.optimization import space as sp
from kalshi_quant.shared.optimization.statistics import deflated_sharpe, probability_of_overfitting
from kalshi_quant.shared.strategy.base import Strategy, params_hash
from kalshi_quant.shared.strategy.loader import load_class
from kalshi_quant.shared.utilities.logging import get_logger
from kalshi_quant.shared.utilities.timeutil import now_ms

log = get_logger(__name__)
OBJECTIVES = ("total_return", "risk_adjusted", "profit_factor", "min_drawdown", "custom")
_WORKER: dict[str, Any] = {}


@dataclass
class OptimizationConfig:
    method: str = "grid"              # grid, random, adaptive
    objective: str = "risk_adjusted"
    max_iterations: int = 500
    min_trades: int = 30
    min_return: float = 0.0           # threshold for the min_drawdown objective
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    embargo_hours: float = 0
    top_k: int = 10
    walk_forward_folds: int = 4
    walk_forward_candidates: int = 30
    workers: int = 1
    seed: int = 0


def score(metrics: dict[str, Any], config: OptimizationConfig, custom: Callable[[dict], float | None] | None = None) -> float | None:
    """Objective value, or None if the trial is not eligible."""
    if metrics["trades"] < config.min_trades:
        return None
    name = config.objective
    if name == "total_return":
        return metrics["total_return"]
    if name == "risk_adjusted":
        # Return per unit of drawdown works on short samples, where Sharpe is withheld.
        drawdown = max(metrics["max_drawdown"], 0.01)
        return metrics["total_return"] / drawdown
    if name == "profit_factor":
        return metrics["profit_factor"]
    if name == "min_drawdown":
        return -metrics["max_drawdown"] if metrics["total_return"] >= config.min_return else None
    if name == "custom":
        return custom(metrics) if custom else None
    raise ValueError(f"unknown objective {name}; choose from {OBJECTIVES}")


def _worker_init(strategy_path: str, class_name: str, datasets: dict[str, str], settings: dict[str, Any],
                 base: dict[str, Any], sys_path: list[str]) -> None:
    import sys
    sys.path[:0] = [p for p in sys_path if p not in sys.path]
    _WORKER["class"] = load_class(Path(strategy_path), class_name)
    _WORKER["datasets"] = {k: Dataset.load(Path(v)) for k, v in datasets.items()}
    _WORKER["settings"] = BacktestSettings.from_json(settings)
    _WORKER["base"] = base


def _evaluate(strategy_cls: type[Strategy], base: dict[str, Any], dataset: Dataset, settings: BacktestSettings,
              params: dict[str, Any]) -> dict[str, Any]:
    merged = {**base["params"], **params}
    strategy = strategy_cls(merged, base["risk"], base["universe"])
    result = run_backtest(strategy, dataset, settings)
    drop = ("by_market", "by_category", "by_month", "by_exit_reason", "by_side", "data_quality")
    return {"metrics": {k: v for k, v in result.metrics.items() if k not in drop},
            "daily_pnl": {str(k): v for k, v in result.daily_pnl.items()}}


def _worker_task(fold: str, params: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    return fold, params, _evaluate(_WORKER["class"], _WORKER["base"], _WORKER["datasets"][fold], _WORKER["settings"], params)


@dataclass
class Fold:
    name: str
    dataset: Dataset


class Optimizer:
    def __init__(self, db: Database, strategy: Strategy, strategy_path: Path, dataset: Dataset, settings: BacktestSettings,
                 space: list[sp.Dimension], config: OptimizationConfig, opt_id: str | None = None,
                 cache_dir: Path | None = None, progress: Callable[[int, int, str], None] | None = None):
        self.db = db
        self.strategy = strategy
        self.path = strategy_path
        self.dataset = dataset
        self.settings = settings
        self.space = space
        self.config = config
        self.rng = random.Random(config.seed)
        self.progress = progress or (lambda done, total, stage: None)
        self.cache_dir = cache_dir or PROJECT_ROOT / "data" / "cache"
        self.base = strategy.to_spec()
        self.strategy_cls = type(strategy)
        self.folds: dict[str, Dataset] = {}
        self.results: dict[tuple[str, str], dict[str, Any]] = {}
        self.opt_id = opt_id or uuid.uuid4().hex[:12]
        self._open_run()

    def _open_run(self) -> None:
        runs, trials = self.db.t["optimization_runs"], self.db.t["optimization_trials"]
        with self.db.begin() as conn:
            exists = conn.execute(select(runs.c.opt_id).where(runs.c.opt_id == self.opt_id)).first()
            if not exists:
                conn.execute(runs.insert().values(
                    opt_id=self.opt_id, created_ms=now_ms(), strategy_name=self.strategy.name,
                    strategy_hash=self.strategy.source_hash(), status="running",
                    config=json.dumps({"optimizer": asdict(self.config), "dataset": self.dataset.dataset_id(),
                                       "space": {d.name: list(d.values) for d in self.space}})))
            for row in conn.execute(select(trials).where(trials.c.opt_id == self.opt_id)).mappings():
                self.results[(row["fold"], row["param_hash"])] = {
                    "metrics": json.loads(row["metrics"]), "daily_pnl": json.loads(row["daily_returns"] or "{}")}
        if self.results:
            log.info("optimization_resumed", opt_id=self.opt_id, trials=len(self.results))

    def _store(self, fold: str, params: dict[str, Any], outcome: dict[str, Any]) -> None:
        key = params_hash(params)
        self.results[(fold, key)] = outcome
        statement = self.db.upsert("optimization_trials").on_conflict_do_nothing(index_elements=["opt_id", "fold", "param_hash"])
        with self.db.begin() as conn:
            conn.execute(statement, [{
                "opt_id": self.opt_id, "fold": fold, "param_hash": key, "params": json.dumps(params),
                "metrics": json.dumps(outcome["metrics"], default=str), "objective": score(outcome["metrics"], self.config, self.strategy.objective),
                "daily_returns": json.dumps(outcome["daily_pnl"])}])

    def evaluate_many(self, fold: str, candidates: Iterable[dict[str, Any]], stage: str) -> None:
        todo = [p for p in candidates if (fold, params_hash(p)) not in self.results]
        total = len(todo)
        if self.config.workers <= 1 or total < 4:
            for done, params in enumerate(todo, 1):
                self._store(fold, params, _evaluate(self.strategy_cls, self.base, self.folds[fold], self.settings, params))
                self.progress(done, total, stage)
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cached = {}
        for name, dataset in self.folds.items():
            path = self.cache_dir / f"{self.opt_id}-{hashlib.sha1(name.encode()).hexdigest()[:8]}.pkl"
            if not path.exists():
                dataset.save(path)
            cached[name] = str(path)
        import sys
        arguments = (str(self.path), type(self.strategy).__name__, cached, self.settings.to_json(), self.base, sys.path)
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(self.config.workers, mp_context=context, initializer=_worker_init, initargs=arguments) as pool:
            futures = [pool.submit(_worker_task, fold, p) for p in todo]
            for done, future in enumerate(as_completed(futures), 1):
                result_fold, params, outcome = future.result()
                self._store(result_fold, params, outcome)
                self.progress(done, total, stage)
        for path in cached.values():
            Path(path).unlink(missing_ok=True)

    def objective_of(self, fold: str, params: dict[str, Any]) -> float | None:
        outcome = self.results.get((fold, params_hash(params)))
        return None if outcome is None else score(outcome["metrics"], self.config, self.strategy.objective)

    def metrics_of(self, fold: str, params: dict[str, Any]) -> dict[str, Any]:
        return self.results[(fold, params_hash(params))]["metrics"]

    # Time splits
    def _split(self) -> tuple[int, int, int, int]:
        start, end = self.dataset.time_range()
        span = end - start
        train_end = start + int(span * self.config.train_fraction)
        validation_end = train_end + int(span * self.config.validation_fraction)
        return start, train_end, validation_end, end + 1

    def _make_folds(self) -> None:
        start, train_end, validation_end, end = self._split()
        embargo = int(self.config.embargo_hours * 3_600_000)
        self.folds["train"] = self.dataset.slice(start, train_end)
        self.folds["validation"] = self.dataset.slice(train_end + embargo, validation_end)
        self.folds["test"] = self.dataset.slice(validation_end + embargo, end)

    def _search(self) -> list[dict[str, Any]]:
        c = self.config
        if c.method == "grid":
            candidates = sp.grid(self.space)
            if len(candidates) > c.max_iterations:
                log.warning("grid_truncated_to_random_sample", size=len(candidates), max=c.max_iterations)
                candidates = sp.random_sample(self.space, c.max_iterations, self.rng)
            self.evaluate_many("train", candidates, "search")
            return candidates
        if c.method == "random":
            candidates = sp.random_sample(self.space, c.max_iterations, self.rng)
            self.evaluate_many("train", candidates, "search")
            return candidates
        if c.method == "adaptive":
            batch = max(c.workers * 4, 16)
            candidates = sp.random_sample(self.space, min(batch, c.max_iterations), self.rng)
            seen = {tuple(p.values()) for p in candidates}
            self.evaluate_many("train", candidates, "search")
            while len(candidates) < c.max_iterations:
                ranked = self._rank("train", candidates)
                proposals = sp.adaptive_proposals(self.space, ranked[:5], min(batch, c.max_iterations - len(candidates)), self.rng, seen)
                if not proposals:
                    break
                self.evaluate_many("train", proposals, "search")
                candidates += proposals
            return candidates
        raise ValueError(f"unknown method {c.method}")

    def _rank(self, fold: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        scored = [(self.objective_of(fold, p), p) for p in candidates]
        scored = [(s, p) for s, p in scored if s is not None]
        return [p for _, p in sorted(scored, key=lambda item: item[0], reverse=True)]

    # Reports
    def _walk_forward(self) -> dict[str, Any]:
        """Expanding train windows, each scored on the following period. Param set fixed up front."""
        c = self.config
        start, _, _, end = self._split()
        folds = c.walk_forward_folds
        if folds < 1:
            return {"folds": []}
        candidates = sp.random_sample(self.space, c.walk_forward_candidates, random.Random(c.seed + 1))
        step = (end - start) // (folds + 1)
        report = []
        oos_daily: dict[str, float] = {}
        for k in range(folds):
            train_end = start + step * (k + 1)
            test_end = train_end + step if k < folds - 1 else end
            train_name, test_name = f"wf{k}_train", f"wf{k}_test"
            self.folds[train_name] = self.dataset.slice(start, train_end)
            self.folds[test_name] = self.dataset.slice(train_end, test_end)
            self.evaluate_many(train_name, candidates, f"walk-forward {k + 1}/{folds} train")
            ranked = self._rank(train_name, candidates)
            if not ranked:
                report.append({"fold": k, "selected": None, "note": "no candidate met the trade minimum"})
                continue
            self.evaluate_many(test_name, [ranked[0]], f"walk-forward {k + 1}/{folds} test")
            in_metrics, out_metrics = self.metrics_of(train_name, ranked[0]), self.metrics_of(test_name, ranked[0])
            for day, pnl in self.results[(test_name, params_hash(ranked[0]))]["daily_pnl"].items():
                oos_daily[day] = oos_daily.get(day, 0.0) + pnl
            report.append({"fold": k, "selected": ranked[0], "in_sample_return": in_metrics["total_return"],
                           "out_of_sample_return": out_metrics["total_return"], "out_of_sample_trades": out_metrics["trades"],
                           "out_of_sample_drawdown": out_metrics["max_drawdown"]})
        tested = [f for f in report if f.get("selected")]
        profitable = sum(1 for f in tested if f["out_of_sample_return"] > 0)
        return {"folds": report, "folds_evaluated": len(tested), "folds_profitable": profitable,
                "stitched_oos_pnl": sum(oos_daily.values()),
                "mean_in_sample_return": float(np.mean([f["in_sample_return"] for f in tested])) if tested else None,
                "mean_out_of_sample_return": float(np.mean([f["out_of_sample_return"] for f in tested])) if tested else None}

    def _sensitivity(self, params: dict[str, Any]) -> dict[str, Any]:
        points = sp.neighbors(self.space, params)
        self.evaluate_many("train", points, "sensitivity")
        center = self.objective_of("train", params)
        scores = [self.objective_of("train", p) for p in points]
        valid = [s for s in scores if s is not None]
        by_param: dict[str, list[dict[str, Any]]] = {}
        for point, value in zip(points, scores):
            name = next(k for k in params if point[k] != params[k])
            by_param.setdefault(name, []).append({"value": point[name], "objective": value})
        median = float(np.median(valid)) if valid else None
        retained = median / center if median is not None and center and center > 0 else None
        return {"center_objective": center, "neighbors": len(points), "neighbors_eligible": len(valid),
                "median_neighbor_objective": median, "objective_retained_by_neighbors": retained, "by_parameter": by_param}

    def _stability(self, params: dict[str, Any]) -> dict[str, Any]:
        daily = self.results[("train", params_hash(params))]["daily_pnl"]
        if len(daily) < 4:
            return {"note": "too few active days"}
        days = sorted(daily, key=int)
        blocks = np.array_split(np.array([daily[d] for d in days]), 4)
        totals = [float(b.sum()) for b in blocks]
        positive = sum(t > 0 for t in totals)
        winner_share = max(totals) / sum(t for t in totals if t > 0) if positive else None
        return {"quarters_of_train": totals, "quarters_profitable": positive, "best_quarter_share_of_profit": winner_share}

    def _multiple_testing(self, candidates: list[dict[str, Any]], selected: dict[str, Any]) -> dict[str, Any]:
        days = sorted({d for p in candidates for d in self.results.get(("train", params_hash(p)), {"daily_pnl": {}})["daily_pnl"]}, key=int)
        if not days:
            return {"note": "no active days"}
        capital = self.settings.starting_capital
        keep = [p for p in candidates if ("train", params_hash(p)) in self.results]
        matrix = np.array([[self.results[("train", params_hash(p))]["daily_pnl"].get(d, 0.0) / capital for p in keep] for d in days])
        deviations = matrix.std(axis=0, ddof=1) if len(days) > 1 else np.zeros(len(keep))
        trial_sharpes = np.divide(matrix.mean(axis=0), deviations, out=np.zeros(len(keep)), where=deviations > 0)
        column = keep.index(selected)
        return {"trials_counted": len(keep), **deflated_sharpe(matrix[:, column], len(keep), trial_sharpes),
                "cscv": probability_of_overfitting(matrix, seed=self.config.seed)}

    @staticmethod
    def _deterioration(train: dict[str, Any], other: dict[str, Any], train_days: int, other_days: int) -> dict[str, Any]:
        per_day = lambda m, d: m["total_return"] / d if d else None  # noqa: E731
        a, b = per_day(train, train_days), per_day(other, other_days)
        return {"train_return_per_day": a, "out_of_sample_return_per_day": b,
                "retained_fraction": b / a if a and a > 0 and b is not None else None,
                "train_win_rate": train["win_rate"], "out_of_sample_win_rate": other["win_rate"],
                "train_profit_factor": train["profit_factor"], "out_of_sample_profit_factor": other["profit_factor"],
                "train_max_drawdown": train["max_drawdown"], "out_of_sample_max_drawdown": other["max_drawdown"]}

    def _days(self, fold: str) -> int:
        data = self.folds[fold]
        return (int(data.ts_ms[-1]) - int(data.ts_ms[0])) // 86_400_000 + 1 if len(data) else 0

    def run(self) -> dict[str, Any]:
        self._make_folds()
        candidates = self._search()
        ranked = self._rank("train", candidates)
        report: dict[str, Any] = {"opt_id": self.opt_id, "strategy": self.strategy.name, "objective": self.config.objective,
                                  "method": self.config.method, "trials": len(candidates), "eligible_trials": len(ranked),
                                  "space_size": sp.space_size(self.space), "dataset": self.dataset.dataset_id(),
                                  "rows": {k: len(self.folds[k]) for k in ("train", "validation", "test")}}
        if not ranked:
            report["verdict"] = {"robust": False, "flags": [f"no trial reached {self.config.min_trades} trades on train"]}
            return self._finish(report)
        top = ranked[: self.config.top_k]
        report["top_by_train"] = [{"params": p, "objective": self.objective_of("train", p), "metrics": _brief(self.metrics_of("train", p))} for p in top]
        self.evaluate_many("validation", top, "validation")
        by_validation = self._rank("validation", top)
        selection_note = "best validation objective"
        if not by_validation:  # validation too thin to rank: keep the train leader, and flag it
            by_validation = top[:1]
            selection_note = "validation had too few trades; fell back to the train leader"
        selected = by_validation[0]
        self.evaluate_many("test", [selected], "final test")
        train, validation, test = (self.metrics_of(f, selected) for f in ("train", "validation", "test"))
        report.update(selected={"params": selected, "selection": selection_note}, train=_brief(train), validation=_brief(validation), test=_brief(test),
                      deterioration={"validation": self._deterioration(train, validation, self._days("train"), self._days("validation")),
                                     "test": self._deterioration(train, test, self._days("train"), self._days("test"))})
        report["sensitivity"] = self._sensitivity(selected)
        report["stability"] = self._stability(selected)
        report["walk_forward"] = self._walk_forward()
        report["multiple_testing"] = self._multiple_testing(candidates, selected)
        report["verdict"] = self._verdict(report, selection_note)
        return self._finish(report)

    def _verdict(self, r: dict[str, Any], selection_note: str) -> dict[str, Any]:
        """Each flag is a specific reason not to trust the result. Passing proves nothing."""
        flags = []
        c = self.config
        test, sensitivity, wf, stats_ = r["test"], r["sensitivity"], r["walk_forward"], r["multiple_testing"]
        if "fell back" in selection_note:
            flags.append(selection_note)
        if test["trades"] < c.min_trades:
            flags.append(f"test period has only {test['trades']} trades")
        if test["total_return"] <= 0:
            flags.append("selected parameters lost money on the test period")
        retained = r["deterioration"]["test"]["retained_fraction"]
        if retained is not None and retained < 0.5:
            flags.append(f"test retains only {retained:.0%} of train return per day")
        kept = sensitivity["objective_retained_by_neighbors"]
        if kept is None or kept < 0.5:
            flags.append("performance is a narrow peak: neighbouring parameters do much worse")
        if wf.get("folds_evaluated") and wf["folds_profitable"] / wf["folds_evaluated"] < 0.5:
            flags.append(f"walk-forward profitable in {wf['folds_profitable']} of {wf['folds_evaluated']} folds")
        if not wf.get("folds_evaluated"):
            flags.append("walk-forward could not be evaluated")
        dsr = stats_.get("deflated_sharpe")
        if dsr is None:
            flags.append("deflated Sharpe unavailable: too little data")
        elif dsr < 0.95:
            flags.append(f"deflated Sharpe {dsr:.2f} is below 0.95 after {stats_['trials_counted']} trials")
        pbo = (stats_.get("cscv") or {}).get("pbo")
        if pbo is not None and pbo > 0.5:
            flags.append(f"probability of backtest overfitting is {pbo:.0%}")
        return {"robust": not flags, "flags": flags,
                "note": "No flags raised means this search found no specific reason for doubt. It does not predict profit."}

    def _finish(self, report: dict[str, Any]) -> dict[str, Any]:
        runs = self.db.t["optimization_runs"]
        with self.db.begin() as conn:
            conn.execute(update(runs).where(runs.c.opt_id == self.opt_id).values(status="finished", report=json.dumps(report, default=str)))
        return report


def _brief(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = ("total_return", "total_pnl", "trades", "win_rate", "profit_factor", "max_drawdown", "sharpe", "sortino", "fees", "days")
    return {k: metrics.get(k) for k in keys}


def load_report(db: Database, opt_id: str) -> dict[str, Any] | None:
    with db.connect() as conn:
        row = conn.execute(text("SELECT report FROM optimization_runs WHERE opt_id = :o"), {"o": opt_id}).first()
    return json.loads(row[0]) if row and row[0] else None
