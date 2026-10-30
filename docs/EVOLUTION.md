# Evolutionary learning

Discovers strategies per Kalshi category and across categories, then tests whether what it found
transfers. It runs independently in the general and 15-minute environments.

```bash
python main.py evolve 15m --profile smoke --workers 6
```

```bash
python main.py evolve general --workers 8
```

`--profile smoke` and `--profile quick` shrink the work. Without a profile you get the full process.
Add `--category Crypto` to limit category populations and `--resume <run_id>` to continue. Also
available as `evolve_general.py` and `evolve_15m.py`.

## Populations

| Type | Trains on | Fitness |
|---|---|---|
| Category | Only that Kalshi category. One population per sufficient category | Trade significance less drawdown. Breadth is never scored |
| Cross-category | Equal market counts per category, so the largest cannot dominate | Above, blended with the generalization score |
| Hybrid | Same balanced data, seeded from the finalists of the other two types | A lighter blend of both |

Categories come from Kalshi's series metadata. A category below `eligibility` in
`configuration/evolution.yaml` is reported as insufficient and skipped. Eligibility is rechecked on
every run, so training starts once enough data has arrived.

Each category and cross-category population runs: `cycles` independent cycles of `epochs`, saving
`saved_per_cycle` strategies; crossover across that archive; a final cycle; `final_saved` strategies.
The defaults are 50, 50, 10 (500 saved), then 50 epochs and 10 strategies.

Hybrids use all four modes: same-category, cross-category (two categories),
category-to-cross, and cross-to-category. The first-named parent is favoured.

## Strategy components

A genome holds entry conditions, exit conditions, logical operators, the side rule, position size,
stop, take-profit, trailing stop and maximum holding time. A condition is an indicator, operator,
threshold and lookback. Crossover chooses each component separately, can blend thresholds of a shared
indicator, and records which parent supplied what. Lookbacks scale by environment.

Evolved strategies run through the normal `StrategyRunner`, with the same past-only view, latency,
fills and fees as any other strategy. Credible finalists are exported as ordinary strategy files:

```bash
python main.py backtest 15m --strategy saved_strategies/fifteen_minute/crypto/final/strategy_01.py
```

## How data is used

One chronological split is shared by all populations: 60% train, 20% validation, 20% test.

1. **Train:** all search.
2. **Validation:** chooses what each cycle saves, and which strategies become finalists.
3. **Test:** read only after every population and hybrid has finished. A test confirms this.

A market that settles after a fold's end is excluded from that fold.

## Validation

- **In-category:** chronological test data, then forward windows across it. The genome is fixed there.
- **Walk-forward:** re-evolves on each expanding window and trades the next. This tests the search
  process, not one strategy. Its depth is set separately, because full cycles per fold would multiply
  the run cost.
- **Cross-category:** each finalist runs on every category's test data, reported separately. Failing
  elsewhere never rejects a specialist.
- **Leave-one-category-out:** evolves without a category and tests on it, after the training period.

## Generalization score

From 0 to 1. Zero unless at least `min_categories` are profitable with enough trades and the total is
positive. It rewards breadth, balance between categories, similar per-trade returns and profit across
time periods, and it discounts for categories that lose. Weights are in
`shared/evolution/fitness.py`; they are judgement calls and have not been calibrated.

## Finalists

A candidate qualifies only if it has at least `min_test_trades`, made money on validation and on test,
and is significant after Benjamini-Hochberg control across its population's candidates. Cross-category
strategies also need a minimum generalization score. **Lists are never padded:** if two qualify, two
are reported, and the rest are listed with their reasons in `rejected_candidates`.

The p-value treats trades as independent. Trades within a day or on one underlying are correlated, so
it is optimistic. Passing is a reason to investigate further, not proof.

## Output

```
saved_strategies/<general_markets|fifteen_minute>/
  <category>/cycle_001/rank_01.json ... final/rank_01.json, strategy_01.py
  cross_category/  hybrid/
  finalists.json         every finalist, its categories, results and lineage
  research_report.md     the six research questions, from this run's measurements
```

Each saved strategy is also in the `evolution_strategies` table. The folder is git-ignored.

## Controlling cost

The full default is 50 × 50 × 60 = 150,000 genome evaluations per population before caching, plus a
final cycle and validation, for every population. Reduce it per environment and population:

```yaml
overrides:
  general: {Sports: {cycles: 10, population_size: 40}}
  fifteen_minute: {cross_category: {cycles: 20}}
```

Thin or short data is the bigger limit. More search over a few days of history finds more noise.

Measured on 8 cores with 6 workers and about 8,400 real 15-minute markets: the `quick` profile took
about 10 minutes for 11,609 simulations, around 19 per second. At that rate the full process is
roughly 2 to 4 days per population, depending on caching, and more with larger datasets.

Lineage for a run resumed from storage only reaches saved strategies, because intermediate generations
are kept in memory.

## First results

On 2026-10-30, `quick` on about a week of real 15-minute candle data qualified **0 of 50** candidates,
and walk-forward was profitable in 0 of 3 folds for Crypto. The sample is far too short to say whether
edges exist. It does show the gates reject what search finds.
