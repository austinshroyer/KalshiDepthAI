import random
import sys
from pathlib import Path

import numpy as np
import pytest
import synthetic

sys.path.insert(0, str(Path(__file__).parent))
from optimizer_strategy import Threshold  # noqa: E402

from kalshi_quant.shared.backtest.dataset import DatasetSpec, load_dataset
from kalshi_quant.shared.backtest.engine import BacktestSettings, run_backtest
from kalshi_quant.shared.config import ExecutionConfig
from kalshi_quant.shared.optimization.engine import OptimizationConfig, Optimizer, load_report, score
from kalshi_quant.shared.optimization.space import adaptive_proposals, grid, neighbors, parse_space, random_sample, space_size
from kalshi_quant.shared.optimization.statistics import benjamini_hochberg_threshold, deflated_sharpe, expected_max_sharpe, probability_of_overfitting

PATH = Path(__file__).parent / "optimizer_strategy.py"
SPACE = {"entry": {"min": 0.35, "max": 0.55, "step": 0.05}, "stop_loss": {"values": [None, 0.2]}}


def settings():
    return BacktestSettings(10_000, 1000, 900, ExecutionConfig())


def optimizer(db, tmp_path, opt_id=None, **config):
    data = load_dataset(db, DatasetSpec(environment=db.env.value))
    base = dict(min_trades=5, walk_forward_folds=2, walk_forward_candidates=4, top_k=3)
    return Optimizer(db, Threshold(), PATH, data, settings(), parse_space(SPACE), OptimizationConfig(**{**base, **config}), opt_id, tmp_path)


def test_parse_space_matches_the_brief_format():
    space = parse_space({"entry_probability": {"min": 0.30, "max": 0.50, "step": 0.01},
                         "buying_window_minutes": {"min": 1, "max": 15, "step": 1}})
    assert len(space[0].values) == 21 and space[0].values[-1] == 0.5
    assert space[1].values == tuple(range(1, 16)) and space_size(space) == 315


def test_grid_and_random_cover_space():
    space = parse_space(SPACE)
    assert len(grid(space)) == 10
    assert len(random_sample(space, 100, random.Random(1))) == 10


def test_neighbors_move_one_step_on_one_axis():
    space = parse_space({"a": {"min": 1, "max": 3, "step": 1}, "b": {"min": 1, "max": 3, "step": 1}})
    assert len(neighbors(space, {"a": 2, "b": 2})) == 4 and len(neighbors(space, {"a": 1, "b": 1})) == 2


def test_adaptive_never_repeats():
    space = parse_space(SPACE)
    seen = {(0.35, None)}
    proposals = adaptive_proposals(space, [{"entry": 0.35, "stop_loss": None}], 5, random.Random(0), seen)
    assert len({tuple(p.values()) for p in proposals}) == len(proposals) and (0.35, None) not in [tuple(p.values()) for p in proposals]


@pytest.mark.parametrize("objective,expected", [("total_return", 0.2), ("risk_adjusted", 2.0), ("profit_factor", 1.5), ("min_drawdown", -0.1)])
def test_objectives(objective, expected):
    metrics = {"trades": 50, "total_return": 0.2, "max_drawdown": 0.1, "profit_factor": 1.5}
    assert score(metrics, OptimizationConfig(objective=objective, min_trades=30)) == pytest.approx(expected)


def test_min_trades_excludes_trial():
    assert score({"trades": 5, "total_return": 9}, OptimizationConfig(objective="total_return", min_trades=30)) is None


def test_min_drawdown_enforces_return_threshold():
    metrics = {"trades": 50, "total_return": 0.01, "max_drawdown": 0.01}
    assert score(metrics, OptimizationConfig(objective="min_drawdown", min_return=0.05, min_trades=1)) is None


def test_custom_objective():
    assert score({"trades": 50}, OptimizationConfig(objective="custom", min_trades=1), lambda m: 7.0) == 7.0


def test_folds_are_time_ordered_and_contain_no_future_settlements(general_db, tmp_path):
    synthetic.seed(general_db, days=30)
    o = optimizer(general_db, tmp_path)
    o._make_folds()
    train, validation, test = (o.folds[k] for k in ("train", "validation", "test"))
    assert train.ts_ms.max() < validation.ts_ms.min() and validation.ts_ms.max() < test.ts_ms.min()
    assert all(s.ts_ms < validation.ts_ms.min() for s in train.settlements.values())
    tickers = lambda d: {d.tickers[i] for i in np.unique(d.ticker_index)}  # noqa: E731
    assert not tickers(train) & tickers(test)


def test_market_crossing_a_boundary_is_left_out_of_the_earlier_fold(general_db):
    synthetic.seed(general_db, days=4)
    data = load_dataset(general_db, DatasetSpec(environment="general"))
    settlement = data.settlements[data.tickers[0]]
    fold = data.slice(int(data.ts_ms[0]), settlement.ts_ms - 1000)
    assert data.tickers[0] not in fold.settlements
    assert 0 not in np.unique(fold.ticker_index)
    # Nothing in a fold can settle with data from outside it.
    assert run_backtest(Threshold(), fold, settings()).metrics["data_quality"]["unsettled_positions_at_end"] == 0


def test_full_run_produces_all_reports(general_db, tmp_path):
    synthetic.seed(general_db, days=60)
    report = optimizer(general_db, tmp_path, method="grid").run()
    for section in ("selected", "train", "validation", "test", "deterioration", "sensitivity", "stability",
                    "walk_forward", "multiple_testing", "verdict"):
        assert section in report
    assert report["trials"] == 10
    assert load_report(general_db, report["opt_id"])["selected"] == report["selected"]


def test_pure_noise_is_not_called_robust(general_db, tmp_path):
    """A driftless random walk has no edge after fees; the search must not bless it."""
    synthetic.seed(general_db, days=90, seed_value=3)
    report = optimizer(general_db, tmp_path, method="grid").run()
    assert not report["verdict"]["robust"] and report["verdict"]["flags"]


def test_search_never_touches_test_before_selection(general_db, tmp_path):
    synthetic.seed(general_db, days=40)
    o = optimizer(general_db, tmp_path, method="grid")
    o.run()
    assert sum(1 for fold, _ in o.results if fold == "test") == 1


def test_interrupted_run_resumes_from_checkpoints(general_db, tmp_path):
    synthetic.seed(general_db, days=20)
    first = optimizer(general_db, tmp_path, opt_id="fixed", method="grid", walk_forward_folds=0)
    first._make_folds()
    first.evaluate_many("train", grid(first.space)[:4], "search")
    resumed = optimizer(general_db, tmp_path, opt_id="fixed", method="grid", walk_forward_folds=0)
    assert len(resumed.results) == 4
    evaluated = []
    resumed._make_folds()
    resumed.progress = lambda done, total, stage: evaluated.append(stage)
    resumed.evaluate_many("train", grid(resumed.space), "search")
    assert len(evaluated) == 6


def test_parallel_matches_serial(general_db, tmp_path):
    synthetic.seed(general_db, days=20)
    serial = optimizer(general_db, tmp_path, method="grid", walk_forward_folds=0)
    serial._make_folds()
    serial.evaluate_many("train", grid(serial.space), "s")
    parallel = optimizer(general_db, tmp_path / "p", method="grid", walk_forward_folds=0, workers=2)
    parallel._make_folds()
    parallel.evaluate_many("train", grid(parallel.space), "p")
    common = [k for k in serial.results if k[0] == "train"]
    assert len(common) == 10
    for key in common:
        assert serial.results[key]["metrics"]["total_pnl"] == pytest.approx(parallel.results[key]["metrics"]["total_pnl"])


@pytest.mark.parametrize("method", ["random", "adaptive"])
def test_other_search_methods(general_db, tmp_path, method):
    synthetic.seed(general_db, days=30)
    report = optimizer(general_db, tmp_path, method=method, max_iterations=6, walk_forward_folds=0).run()
    assert report["trials"] <= 6


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(1000, 1) > expected_max_sharpe(10, 1) > 0
    assert expected_max_sharpe(1, 1) == 0


def test_deflated_sharpe_penalizes_many_trials():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.002, 0.01, 250)
    sharpes = rng.normal(0, 0.05, 1000)
    few = deflated_sharpe(returns, 2, sharpes[:2])["deflated_sharpe"]
    many = deflated_sharpe(returns, 1000, sharpes)["deflated_sharpe"]
    assert many < few


def test_deflated_sharpe_needs_data():
    assert deflated_sharpe(np.array([0.1, 0.2]), 10, np.zeros(10))["deflated_sharpe"] is None


def test_pbo_high_for_noise_low_for_consistent_edge():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, (240, 40))
    assert probability_of_overfitting(noise)["pbo"] > 0.3
    edge = rng.normal(0, 0.01, (240, 40))
    edge[:, 0] += 0.02
    assert probability_of_overfitting(edge)["pbo"] < 0.05


def test_pbo_reports_when_underpowered():
    assert probability_of_overfitting(np.zeros((3, 1)))["pbo"] is None


def test_fdr_threshold():
    assert benjamini_hochberg_threshold([0.001, 0.2, 0.9]) == 0.001
    assert benjamini_hochberg_threshold([0.5, 0.9]) is None
