import json
import random
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import synthetic

from kalshi_quant.shared.backtest.dataset import DatasetSpec, load_dataset
from kalshi_quant.shared.backtest.engine import BacktestSettings, run_backtest
from kalshi_quant.shared.config import Env, ExecutionConfig
from kalshi_quant.shared.evolution.engine import EvolutionEngine, balanced_sample, plan_categories
from kalshi_quant.shared.evolution.fitness import FAILED, FitnessConfig, fitness, generalization_score, performance, summarize
from kalshi_quant.shared.evolution.genome import (
    Condition, GeneSpace, Genome, GenomeStrategy, Individual, crossover, mutate, random_genome,
)
from kalshi_quant.shared.evolution.settings import EvolutionSettings, slug
from kalshi_quant.shared.execution.portfolio import ClosedTrade

SPACE = GeneSpace((60, 300), (600,))
TINY = {"defaults": {"cycles": 2, "epochs": 2, "population_size": 8, "saved_per_cycle": 2, "final_epochs": 2, "final_saved": 3},
        "hybrid": {"per_mode": 2, "epochs": 1, "population_size": 6},
        "validation": {"forward_windows": 2, "walk_forward_folds": 2, "walk_forward_epochs": 1, "walk_forward_population": 4,
                       "leave_one_out_epochs": 1, "leave_one_out_population": 4},
        "eligibility": {"min_markets": 20, "min_rows": 500, "min_days": 3},
        "fitness": {"category": {"min_trades": 3}, "cross_category": {"min_trades": 3, "min_trades_per_category": 1},
                    "hybrid": {"min_trades": 3, "min_trades_per_category": 1}},
        "credibility": {"min_test_trades": 3}}


def trade(pnl, ticker="T", exit_ms=1000):
    return ClosedTrade(ticker, "yes", 10, 0.5, 0.6, 0, exit_ms, pnl, 0.1, 0.0, "settlement")


def genome(**kw):
    base = Genome((Condition("probability", "<", 0.45),), "all", (), "all", "yes", 0.05, None, None, None, None)
    return replace(base, **kw)


def seed_three(db, days=30, thin=False):
    synthetic.seed(db, days=days, series="AAA", category="Alpha", drift=0.002, seed_value=1)
    synthetic.seed(db, days=days, series="BBB", category="Beta", drift=-0.002, seed_value=2)
    synthetic.seed(db, days=3 if thin else days, markets_per_day=1 if thin else 4, series="CCC", category="Gamma", seed_value=3)


def engine(db, config, tmp_path, **kw):
    data = load_dataset(db, DatasetSpec(environment=db.env.value))
    settings = EvolutionSettings(config.env, extra=kw.pop("extra", TINY))
    return EvolutionEngine(db, config, data, settings, archive_root=tmp_path / "saved", **kw)


def test_genome_round_trip_and_fingerprint():
    g = random_genome(random.Random(1), SPACE)
    assert Genome.from_json(g.to_json()) == g and len(g.fingerprint()) == 16


def test_genomes_cannot_reach_outcomes():
    assert "result" not in Genome.__dataclass_fields__ and hasattr(GenomeStrategy, "should_enter")


def test_crossover_works_on_components_and_records_sources():
    rng = random.Random(3)
    a, b = Individual(random_genome(rng, SPACE)), Individual(random_genome(rng, SPACE))
    child = crossover(a, b, rng, SPACE, "cross_category")
    assert {"entry_conditions", "exit_conditions", "entry_logic", "side", "position_sizing", "stop_loss", "take_profit"} <= set(child.components)
    assert child.parents == (a.id, b.id) and child.origin == "cross_category"
    assert len(child.genome.entry) >= 1 and len(child.genome.entry) <= SPACE.max_conditions


def test_crossover_inherits_single_components_from_each_parent():
    rng = random.Random(0)
    a = Individual(genome(side="yes", position_fraction=0.01))
    b = Individual(genome(side="no", position_fraction=0.09, stop_loss=0.2))
    children = [crossover(a, b, rng, SPACE) for _ in range(40)]
    assert {c.genome.side for c in children} == {"yes", "no"}
    assert any(c.genome.side == "yes" and c.genome.position_fraction == 0.09 for c in children)


def test_crossover_blends_thresholds_of_shared_indicators():
    rng = random.Random(1)
    a = Individual(genome(entry=(Condition("probability", "<", 0.2),)))
    b = Individual(genome(entry=(Condition("probability", "<", 0.8),)))
    thresholds = {c.threshold for _ in range(30) for c in crossover(a, b, rng, SPACE).genome.entry}
    assert any(0.2 < t < 0.8 for t in thresholds)


def test_mutation_keeps_genes_valid():
    rng = random.Random(2)
    g = random_genome(rng, SPACE)
    for _ in range(300):
        g = mutate(g, rng, SPACE)
        assert 1 <= len(g.entry) <= SPACE.max_conditions and 0.005 <= g.position_fraction <= 0.15


def test_genome_strategy_trades_through_the_normal_engine(general_db):
    synthetic.seed(general_db, days=5)
    data = load_dataset(general_db, DatasetSpec(environment="general"))
    result = run_backtest(GenomeStrategy.build(genome()), data, BacktestSettings(10_000, 1000, 900, ExecutionConfig()))
    assert result.metrics["trades"] > 0


def test_performance_by_group():
    s = performance([trade(10, exit_ms=1), trade(-5, exit_ms=2), trade(5, exit_ms=3)], 1000)
    assert s["trades"] == 3 and s["return"] == 0.01 and s["profit_factor"] == 3 and s["win_rate"] == pytest.approx(2 / 3)
    assert s["max_drawdown"] == pytest.approx(5 / 1010)


def test_summary_records_every_required_category_statistic():
    s = summarize([trade(1, "A"), trade(-1, "B"), trade(2, "A")], 1000, {"A": "Crypto", "B": "Sports"}, 1)
    for key in ("return", "win_rate", "profit_factor", "max_drawdown", "trades", "average_trade_return", "risk_adjusted"):
        assert key in s["categories"]["Crypto"]
    assert s["categories"]["Sports"]["trades"] == 1 and s["category_consistency"] is not None


CONFIG = FitnessConfig(min_trades=5, generalization_weight=0.6, min_trades_per_category=3, min_categories=2)


def balanced_trades(spec):
    trades, categories = [], {}
    for name, pnls in spec.items():
        categories[name] = name
        trades += [trade(p, name, i * 1000) for i, p in enumerate(pnls)]
    return trades, categories


def score(spec, eligible=3):
    trades, categories = balanced_trades(spec)
    return generalization_score(summarize(trades, 1000, categories, 3), eligible, CONFIG)


def test_one_category_star_is_not_general():
    assert score({"A": [50, 40, 60, 30], "B": [-1, -2, 1, -1], "C": [-1, -1, -2, 1]})["score"] == 0


def test_broad_profit_scores_higher_than_concentrated_profit():
    broad = score({"A": [5, 4, 6, -1], "B": [5, 3, 6, -1], "C": [4, 5, 5, -1]})
    narrow = score({"A": [90, 80, 70, -1], "B": [1, 1, 1, -1], "C": [-1, -1, -1, 0.5]})
    assert broad["score"] > narrow["score"] and broad["dependence_on_top_category"] < narrow["dependence_on_top_category"]


def test_too_few_trades_cannot_claim_generalization():
    assert score({"A": [5, 5], "B": [5, 5], "C": [5, 5]})["score"] == 0


def test_specialist_fitness_ignores_breadth():
    trades = [trade(p, "A", i) for i, p in enumerate([5, 3, 4, -1, 6, 5, 4, 2])]
    s = summarize(trades, 1000, {"A": "Crypto"}, 3)
    assert fitness(s, FitnessConfig(min_trades=5)) > 0


def test_too_few_trades_fail_fitness():
    assert fitness(summarize([trade(9)], 1000, {}), FitnessConfig(min_trades=5)) < FAILED / 2


def test_separate_fitness_configs_per_population_type():
    s = EvolutionSettings(Env.GENERAL)
    assert s.fitness("category").generalization_weight == 0 and s.fitness("cross_category").generalization_weight > 0


def test_default_is_the_specified_full_process():
    p = EvolutionSettings(Env.GENERAL).population("Crypto")
    assert (p.cycles, p.epochs, p.saved_per_cycle, p.final_epochs, p.final_saved) == (50, 50, 10, 50, 10)
    assert p.cycles * p.saved_per_cycle == 500


def test_overrides_apply_per_environment_and_category():
    extra = {"overrides": {"general": {"Sports": {"cycles": 7}}}}
    assert EvolutionSettings(Env.GENERAL, extra=extra).population("Sports").cycles == 7
    assert EvolutionSettings(Env.GENERAL, extra=extra).population("Crypto").cycles == 50
    assert EvolutionSettings(Env.FIFTEEN_MINUTE, extra=extra).population("Sports").cycles == 50


def test_slug_handles_real_categories():
    assert slug("Climate and Weather") == "climate_and_weather" and slug("Science and Technology") == "science_and_technology"


def test_categories_come_from_metadata_and_thin_ones_are_reported(general_db):
    seed_three(general_db, thin=True)
    data = load_dataset(general_db, DatasetSpec(environment="general"))
    eligible, insufficient = plan_categories(data, TINY["eligibility"])
    assert set(eligible) == {"Alpha", "Beta"} and "Gamma" in insufficient and "resolved markets" in insufficient["Gamma"]


def test_balanced_sample_prevents_domination():
    groups = {"Big": [f"b{i}" for i in range(1000)], "Small": [f"s{i}" for i in range(50)]}
    sample = balanced_sample(groups, 2000, 1)
    assert Counter(t[0] for t in sample) == {"b": 50, "s": 50}


def test_category_folds_contain_only_that_category(general_db, general_config, tmp_path):
    seed_three(general_db)
    e = engine(general_db, general_config, tmp_path)
    assert all(t.startswith("AAA") for t in e.specs["alpha/train"].tickers)
    assert all(t.startswith("BBB") for t in e.specs["beta/test"].tickers)


def test_leave_one_out_really_withholds_the_category(general_db, general_config, tmp_path):
    seed_three(general_db)
    e = engine(general_db, general_config, tmp_path)
    tickers = e.specs["loco/alpha/train"].tickers
    assert tickers and not any(t.startswith("AAA") for t in tickers)
    assert e.specs["loco/alpha/train"].end_ms <= e.specs["alpha/oos"].start_ms


@pytest.fixture
def finished(general_db, general_config, tmp_path):
    seed_three(general_db)
    e = engine(general_db, general_config, tmp_path)
    return e, e.run()


def test_all_three_population_types_run(finished):
    e, report = finished
    kinds = {name: p["kind"] for name, p in report["populations"].items()}
    assert kinds == {"Alpha": "category", "Beta": "category", "Gamma": "category", "cross_category": "cross_category", "hybrid": "hybrid"}
    assert report["simulations"]["simulations"] > 0


def test_archives_are_separated_by_population(finished, tmp_path):
    e, _ = finished
    root = tmp_path / "saved" / "general_markets"
    assert {"alpha", "beta", "gamma", "cross_category", "hybrid"} <= {p.name for p in root.iterdir() if p.is_dir()}
    assert len(list((root / "alpha" / "cycle_001").glob("rank_*.json"))) == 2
    record = json.loads((root / "alpha" / "final" / "rank_01.json").read_text())
    assert record["population"] == "Alpha" and "genome" in record and "validation" in record["metrics"]
    assert (root / "research_report.md").exists() and (root / "finalists.json").exists()


def test_rankings_are_not_merged(finished):
    e, _ = finished
    counts = e.db.scalar("SELECT count(DISTINCT population) FROM evolution_strategies")
    assert counts == 5
    assert e.db.scalar("SELECT count(*) FROM evolution_strategies WHERE population = 'Alpha' AND stage LIKE 'cycle_%'") == 4


def test_test_folds_are_read_only_after_evolution(finished):
    e, _ = finished
    touched = e.evaluator.touched
    end_of_search = touched.index("hybrid/validation")
    assert not [f for f in touched[:end_of_search + 1] if f.endswith(("/test", "/oos")) or "/forward" in f]
    assert any(f.endswith("/test") for f in touched[end_of_search:])


def test_hybrids_use_all_modes_and_keep_lineage(finished):
    e, report = finished
    modes = report["hybrid"]["modes"]
    assert all(modes[m] > 0 for m in ("same_category", "cross_category", "category_to_cross", "cross_to_category"))


def test_hybrid_mode_parents_come_from_the_right_populations(general_db, general_config, tmp_path):
    seed_three(general_db)
    e = engine(general_db, general_config, tmp_path)
    for population in [*e.populations, e.cross]:
        population.finalists = [Individual(random_genome(random.Random(f"{population.name}{i}"), e.settings.gene_space)) for i in range(3)]
        for individual in population.finalists:
            e._note(individual, population, "final")
    seeds, made = e._hybrid_seeds(random.Random(0))
    population_of = lambda i: e.lineage[i]["population"]  # noqa: E731
    for seed in seeds:
        first, second = map(population_of, seed.parents)
        if seed.origin == "same_category":
            assert first == second and first != "cross_category"
        elif seed.origin == "cross_category":
            assert first != second and "cross_category" not in (first, second)
        elif seed.origin == "category_to_cross":
            assert second == "cross_category" and first != second
        else:
            assert first == "cross_category" and second != first


def test_every_finalist_candidate_reports_required_fields(finished):
    e, report = finished
    required = {"intended_categories", "training_categories", "validation_categories", "out_of_sample", "category_performance",
                "strategy_type", "lineage", "credible", "rejection_reasons", "other_category_tests"}
    population = e.populations[0]
    result = e._validate_population(population, e_evaluator(e))
    records = result["finalists"] + result["rejected"]
    assert records and required <= set(records[0]) and records[0]["lineage"]


def e_evaluator(e):
    from kalshi_quant.shared.evolution.evaluation import Evaluator
    return Evaluator(e.dataset, e.specs, e.backtest, 1, e.root)


def test_category_strategies_are_tested_on_other_categories(finished):
    e, _ = finished
    record = (lambda r: (r["finalists"] + r["rejected"])[0])(e._validate_population(e.populations[0], e_evaluator(e)))
    assert set(record["other_category_tests"]) == {"Beta", "Gamma"}


def test_failing_other_categories_never_rejects_a_specialist(finished):
    e, _ = finished
    result = e._validate_population(e.populations[0], e_evaluator(e))
    for record in result["rejected"]:
        assert not any("other categor" in reason or "generalization" in reason for reason in record["rejection_reasons"])


def test_noise_is_not_padded_into_finalists(general_db, general_config, tmp_path):
    for i, name in enumerate(("Alpha", "Beta")):
        synthetic.seed(general_db, days=30, series=f"S{i}", category=name, seed_value=10 + i)
    report = engine(general_db, general_config, tmp_path).run()
    candidates = sum(p["candidates"] for p in report["populations"].values())
    qualifying = sum(p["qualifying"] for p in report["populations"].values())
    assert candidates > 0 and qualifying < candidates
    assert sum(len(v) for v in report["finalists"].values()) == qualifying


def test_cross_population_skipped_with_one_category(general_db, general_config, tmp_path):
    synthetic.seed(general_db, days=30, series="AAA", category="Alpha")
    report = engine(general_db, general_config, tmp_path).run()
    assert "cross_category" not in report["populations"] and "two eligible" in report["cross_category_note"]


def test_resume_skips_completed_cycles(general_db, general_config, tmp_path):
    seed_three(general_db)
    first = engine(general_db, general_config, tmp_path, run_id="fixed")
    first.run()
    again = engine(general_db, general_config, tmp_path, run_id="fixed")
    again.run()
    assert again.evaluator.touched[0] == "alpha/test"  # went straight to validation


def test_same_seed_reproduces_finalists(general_db, general_config, tmp_path):
    seed_three(general_db)
    def finals(run_id):
        e = engine(general_db, general_config, tmp_path / run_id, run_id=run_id)
        e.run()
        return [i.id for i in e.populations[0].finalists]
    assert finals("a") == finals("b")


def test_parallel_matches_serial(general_db, general_config, tmp_path):
    seed_three(general_db)
    serial = engine(general_db, general_config, tmp_path / "s", run_id="s")
    parallel = engine(general_db, general_config, tmp_path / "p", run_id="p", workers=2)
    serial.run()
    parallel.run()
    assert [i.id for i in serial.populations[0].finalists] == [i.id for i in parallel.populations[0].finalists]


def test_planted_edge_is_found_in_its_category(general_db, general_config, tmp_path):
    """Alpha drifts up, Beta down: evolution should learn opposite sides in each."""
    seed_three(general_db)
    extra = {**TINY, "defaults": {"cycles": 2, "epochs": 6, "population_size": 24, "saved_per_cycle": 3, "final_epochs": 4, "final_saved": 5}}
    e = engine(general_db, general_config, tmp_path, extra=extra)
    report = e.run()
    alpha = e._validate_population(e.populations[0], e_evaluator(e))
    best = max(alpha["finalists"] + alpha["rejected"], key=lambda r: r["out_of_sample"]["return"])
    assert best["out_of_sample"]["return"] > 0
    assert report["research"]["q1_category_amenability"]


def test_exported_finalist_runs_as_a_normal_strategy(general_db, general_config, tmp_path):
    from kalshi_quant.shared.strategy.loader import load_class
    seed_three(general_db)
    e = engine(general_db, general_config, tmp_path)
    directory = tmp_path / "x" / "final"
    directory.mkdir(parents=True)
    e.root = tmp_path
    e._export({"x": {"finalists": [{"rank": 1, "strategy_id": "abc", "genome": json.loads(genome().to_json())}]}})
    cls = load_class(directory / "strategy_01.py")
    data = load_dataset(general_db, DatasetSpec(environment="general"))
    assert run_backtest(cls(), data, BacktestSettings(10_000, 1000, 900, ExecutionConfig())).metrics["trades"] > 0


def test_restrict_keeps_rows_of_chosen_markets(general_db):
    tickers = synthetic.seed(general_db, days=3)
    data = load_dataset(general_db, DatasetSpec(environment="general"))
    part = data.restrict(frozenset(tickers[:2]))
    assert len(part) == 120 and set(part.settlements) == set(tickers[:2])
    assert np.array_equal(np.unique(part.ticker_index), np.array([data.tickers.index(t) for t in sorted(tickers[:2])]))


def test_lineage_reaches_back_through_unsaved_generations(finished):
    e, _ = finished
    hybrid = e.hybrid.finalists[0]
    chain = e._ancestors(hybrid.id)
    assert any(a["origin"] == "random" for a in chain) and {a["population"] for a in chain} & {"Alpha", "Beta", "Gamma", "cross_category"}
    assert e._familiar(hybrid)
