# Kalshi Quantitative Research Platform

A local research environment for Kalshi prediction markets: data collection, paper trading,
backtesting and parameter optimization. It runs two fully separate environments, one for general
markets and one for 15-minute markets, each with its own database, strategies and results.

**It never places a real order.** The API client can only issue GET requests, and tests enforce that.

## What is verified, and how

"Live" means it ran against the real Kalshi API or real collected data on 2026-10-30.
"Mock" means it ran only against a local simulation. See [Known limitations](#known-limitations).

| Capability | Status |
|---|---|
| REST client, retries, pagination, token rate limiting | Live |
| General and 15-minute collection, concurrently | Live |
| Market classification and separation (0 tickers in both databases) | Live |
| Recovery from `kill -9` with no corruption, gap recorded | Live |
| Candle backfill, data validation, online backup | Live |
| Paper trading, 15-minute (fills, stops, take-profit, settlement) | Live |
| Paper trading, general | Live: discovered 170,758 markets, tracked 500, stopped cleanly. No simulated trade occurred in that 7-minute run |
| Backtests, exact replay, optimizer with walk-forward | Live data |
| Trade backfill (`--trades`) | Mock only |
| **WebSocket feed** | **Mock only: no API key was available** |
| Request signing | Unit-tested with real keys generated locally; never accepted by Kalshi |
| PostgreSQL | Not run. Schema is SQLAlchemy Core and upserts are dialect-aware |

The test suite has 192 tests. Live runs were short (under an hour, one evening), so multi-day
stability and daytime volume are **not** yet observed.

## Installation

Python 3.11 or later (developed on 3.14).

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python -m pytest
```

Commands below assume the virtualenv is active. Databases are created on first use.

### Authentication (optional)

Everything works on public REST without an account. An API key adds the WebSocket feed, which is the
only way to get true tick-level coverage of all markets. Create a key at kalshi.com under Profile
Settings, then copy `.env.example` to `.env`. `.env` and `*.key` are git-ignored.

## Running everything

```bash
python main.py collect all
```

That starts both pipelines concurrently and runs until Ctrl-C. Typical workflow:

```bash
python main.py backfill 15m --series KXBTC15M --since 2026-10-01
```

```bash
python main.py backtest 15m --strategy late_favorite.py
```

```bash
python main.py optimize 15m --strategy example.py --workers 6
```

```bash
python main.py paper 15m --strategy example.py --background
```

### Command reference

Every command names its environment: `general` or `15m`.

| Task | Command |
|---|---|
| Collect | `main.py collect all\|general\|15m` (or `collect_all.py`, `collect_general_markets.py`, `collect_15m_markets.py`) |
| Fetch history | `main.py backfill <env> [--since] [--series] [--limit] [--period] [--trades]` |
| Backtest | `main.py backtest <env> --strategy X [--start] [--end] [--category] [--series] [--source] [--set name=value]` (or `backtest_general.py`, `backtest_15m.py`) |
| Reproduce a run | `main.py replay <env> <run_id>`; list with `main.py runs <env>` |
| Optimize | `main.py optimize <env> --strategy X [--space file] [--method] [--objective] [--workers] [--resume id]` (or `optimize_general.py`, `optimize_15m.py`) |
| Paper trade | `main.py paper general\|15m --strategy X [--strategy Y] [--capital] [--background] [--resume]` (or `paper_trade.py`, `paper_trade_general.py`, `paper_trade_15m.py`) |
| Stop a session | `main.py paper stop <env> <session_id>` |
| Inspect paper | `main.py paper sessions <env>`, `paper positions`, `paper capital`, `paper performance`, `paper trades <env>` |
| Data | `main.py database stats\|growth\|validate\|backup\|archive\|query` |

Run `python main.py <command> --help` for all options.

## Architecture

```
kalshi_quant/
  shared/            reusable infrastructure, no environment-specific state
    api/             read-only client, signing, token limiter, WebSocket
    collection/      pipeline base, classifier, backfill, rollups, validation
    database/        schema, migrations glue, batch writer, archive, queries
    execution/       fees, fill simulation, portfolio
    strategy/        strategy API, runner shared by paper and backtest
    backtest/        dataset loading, replay, metrics
    optimization/    search, statistics, walk-forward
    paper/           live engine and session commands
  general_markets/   its collector and strategies
  fifteen_minute/    its collector and strategies
  master_collector/  the only code that touches both environments
configuration/       shared.yaml, general.yaml, fifteen_minute.yaml
migrations/          Alembic histories, one per environment
tests/  scripts/  docs/
```

**Independence is enforced, not just conventional.** Each database records which environment owns it
and refuses to open for the other. Trades and quotes are routed by classification. Strategy files
declare which environments they support, and bare file names only resolve inside that environment's
own directory.

**One decision loop.** Paper trading and backtesting run the same `StrategyRunner` and the same fill
simulator, so a strategy behaves the same way in both. Only the source of ticks differs.

**SQLite, with a path to PostgreSQL.** WAL-mode SQLite is plenty for one writer per database. Tables
are in SQLAlchemy Core, upserts go through a dialect switch, and migrations are Alembic. DuckDB is
used only to write and read zstd Parquet archives, not as a primary store.

## How data is collected

Design choices follow from measurements of the live API. Details are in
[docs/API_NOTES.md](docs/API_NOTES.md). The most important: **`/markets` is cached for about 10
seconds, while order book endpoints are live.** Quotes therefore come from order books, and `/markets`
supplies status, volume and results.

| Tier | General | 15-minute |
|---|---|---|
| Live quotes | Top 3,000 markets by 24h volume, every 30s | Every active market, about every 1s |
| Other quotes | Every open market, every 30 min (may be ~15s old) | n/a |
| Depth snapshots | Top 300 markets | Every active market |
| Trades | Global feed, every 3s | Global feed, every 2s |
| Resolved history | Resumable background walk | Resumable background walk |
| With an API key | WebSocket ticks for all markets | WebSocket ticks and book deltas |

Unchanged quotes are skipped until a heartbeat interval, which is what keeps storage feasible. All
frequencies are in `configuration/`.

**Three kinds of data, never mixed:**

- `observations` and `orderbook_snapshots`: snapshots this application captured. They begin when you
  start collecting.
- `trades`: actual exchange trade records, including history.
- `candlesticks`: exchange 1-minute, hourly or daily summaries.

Kalshi offers no historical order books. Periodic snapshots do not reconstruct one.

**Exclusions.** Perpetuals live on a separate `/margin` API that this code never calls, and the
classifier rejects any other market type. Combos (multivariate markets) are excluded by default:
they are created on demand in very large numbers and dominate listings. Set `include_multivariate:
true` to include them. Series ending in `TEST` are excluded.

**15-minute classification** uses Kalshi's own metadata: `series.frequency == "fifteen_min"` or event
`cadence == "fifteen_min"`. Titles are never read, so `KXNBADRAFTTOP15` and `HOUSETX15` stay general.
Duration only feeds an audit table.

**Recovery.** Data and its checkpoint commit in one transaction. After a crash you lose at most the
last unflushed second or two of rows, and the checkpoint matches what was saved. Volume deltas resume
from stored state, and downtime is recorded in `collection_gaps`, which backtests report.

### Storage

**Plan for disk use.** Measured on one quiet evening: about 220,000 15-minute trades per hour and about
165,000 general trades per hour. That suggests several hundred MB to a few GB per day, and the
one-time walk of resolved markets adds more (about 2 KB per market, over 400,000 in the first 15
minutes). This machine had about 21 GB free.

Protections: writing of high-volume tables pauses below `min_free_disk_gb` (default 5 GB) and is
recorded as a gap; rows older than `archive_after_days` move to zstd Parquet, and backtests read them
transparently. SQLite reuses the freed space rather than shrinking the file. Watch this with:

```bash
python main.py database growth
```

To reduce growth, set `collect_trades: false`, raise poll intervals, or lower `orderbook_levels`.

### Rate limits

Budgets are in Kalshi tokens. Basic allows 200 read tokens per second, split evenly between the two
pipelines. Costs were measured because the authoritative endpoint needs a key. Paper trading has its
own 50-token budget, so running it beside both collectors exceeds 200; the limiter then backs off and
slow sweeps yield, which works but produces 429 retries in the logs. With a key, the free Advanced tier
gives 300 tokens per second.

## Writing a strategy

Create a file in the environment's strategy directory. Write ordinary Python:

```python
from kalshi_quant.shared.strategy.base import YES, Param, Risk, Signal, Strategy, Universe


class MyStrategy(Strategy):
    name = "my_strategy"
    environments = ("general",)
    parameters = {
        "entry": Param(0.40, 0.30, 0.50, 0.01, "buy below this"),   # default, min, max, step
        "min_rise": Param(0.05, 0.01, 0.15, 0.01),
    }
    risk = Risk(position_fraction=0.05, stop_loss=0.15, take_profit=0.30,
                trailing_stop=None, max_holding_seconds=3600, max_market_exposure=0.10)
    universe = Universe(categories=["Economics"])
    history_seconds = 3600     # how much history the view keeps

    def select_market(self, info):          # optional extra market filter
        return info.duration_seconds is not None and info.duration_seconds > 3600

    def should_enter(self, view):
        rise = view.probability_change(180)
        if view.probability is None or rise is None or view.spread is None:
            return None
        if view.probability < self.p.entry and rise >= self.p.min_rise and view.spread < 0.03:
            return Signal(YES)              # or Signal(NO), Signal(YES, quantity=10, limit_price=0.41)
        return None

    def should_exit(self, view, position):  # optional; stops are handled by Risk
        return None
```

The view offers `probability`, `no_probability`, `yes_bid`, `yes_ask`, `no_bid`, `no_ask`, `spread`,
`last`, `volume`, `open_interest`, `seconds_since_open`, `seconds_to_close`, `probability_ago(s)`,
`probability_change(s)`, `volume_since(s)`, `history(s)`, `tick` and `info`. A strategy can import any
library; use `self.rng` for randomness so runs stay reproducible.

An optional `my_strategy.yaml` beside the file overrides `parameters`, `risk` and `universe`, and can
define the optimizer's ranges under `optimization:`.

Examples: `general_markets/strategies/{example,longshot_fade}.py` and
`fifteen_minute/strategies/{example,late_favorite}.py`. `general_markets/strategies/example.py` is the
brief's example written out. Examples show the API. **They are not profitable strategies.**

**Volume means contracts.** Each pays $1, so "at least $10,000 traded" is `volume >= 10_000`.

### How look-ahead is prevented

Strategies receive a view that holds only ticks already delivered, plus `MarketInfo`, which has no
outcome fields. There is nothing future to reach. Tests prove it behaviourally: flipping every outcome
leaves all entries identical, and rewriting all data after a date leaves every earlier trade identical.

- A decision fills on a **later** tick, at least `latency_ms` after it.
- Candles are stamped at their **end**. A candle ending at the close is not tradable.
- Settlement reaches a portfolio only at the real settlement time.
- `seconds_to_close` uses the close time **first seen**, because Kalshi rewrites `close_time` when a
  market ends early.
- An optimizer fold excludes markets that settle after the fold ends.

## Execution simulation

Orders are immediate-or-cancel takers, in paper trading and backtests alike:

- Walks real book levels, YES buys lifting NO bids at the complement and so on.
- Takes at most `max_book_participation` (50%) of displayed size. Partial fills are normal.
- If size is unknown the order is **rejected**, not assumed filled.
- Candle data has no sizes: fills are capped at 10% of candle volume, one tick worse, and in effect
  wait for the next candle, about 60 seconds later.
- Rejects stale quotes, closed markets and unaffordable orders, and honours limit prices.
- Fees follow the [Kalshi schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf):
  `M × 0.07 × C × P × (1−P)`, rounded up to $0.0001, with `M` taken per series. Set
  `fee_rounding_dollars: 0.01` for an FCM-cleared account.

## Reading backtest results

```bash
python main.py backtest general --strategy example.py
```

Reading order:

1. **Execution and data quality first.** Many rejections, `candle_only_markets`, unsettled positions
   or gaps mean the headline numbers are weak. `markets_first_seen_after_close` counts markets where
   early-close information may have leaked; treat those with care.
2. **Trades.** Few trades mean the result is mostly luck.
3. **Risk.** Compare drawdown to return. Sharpe and Sortino appear only with 30+ days of returns; a
   shorter sample shows a reason in their place.
4. **Concentration.** Check by month, market, category and side. One market carrying the result is a
   warning.
5. **Slippage.** Estimated slippage is fill price against the quote at decision time, so it includes
   price movement during latency. It can be negative.

A high win rate means little by itself: buying 97-cent contracts wins nearly always and can still lose
money. Every run saves its source, parameters, dataset hash, settings and trades:

```bash
python main.py replay general <run_id>
```

Replay exits non-zero if trades differ, and tells you if later backfill changed the data.

## Optimization

```yaml
# ranges.yaml
entry_probability: {min: 0.30, max: 0.50, step: 0.01}
stop_loss: {min: 0.05, max: 0.30, step: 0.01}
take_profit: {min: 0.10, max: 0.60, step: 0.01}
buying_window_minutes: {values: [1, 2, 5, 10]}
```

```bash
python main.py optimize 15m --strategy example.py --space ranges.yaml --method adaptive --objective risk_adjusted --max-iterations 500 --workers 6
```

Methods: `grid`, `random`, `adaptive` (cross-entropy style). Objectives: `total_return`,
`risk_adjusted` (return over drawdown), `profit_factor`, `min_drawdown` with `--min-return`, and
`custom`, which calls the strategy's `objective(metrics)`. Any `Risk` field can be a parameter.
Trials are checkpointed; continue with `--resume <id>`.

The process exists to reject false results:

1. Time splits into 60% train, 20% validation, 20% test. Search sees train only.
2. The best candidates are re-scored on validation, which picks one.
3. That one pick is scored on test, once.
4. Walk-forward re-picks on expanding windows, from candidates drawn without seeing any results.
5. Neighbouring parameter values are tested: a real edge is a plateau, not a spike.
6. Deflated Sharpe corrects for the number of trials; CSCV estimates the probability of overfitting.

The verdict lists specific reasons for doubt. **No flags does not mean a strategy will make money.**
Test data is only clean the first time. If you re-run after seeing test results, it becomes training
data, and you should count those trials against yourself.

## Paper trading

```bash
python main.py paper 15m --strategy example.py --strategy late_favorite.py --capital 1000 --background
```

General-market discovery lists all ~170k open markets unless `universe` narrows it, which took about 5
minutes beside both collectors. Set `series` or `categories` to start much faster. Each strategy has its own virtual portfolio, and every fill and state change commits immediately.
`--resume` continues the latest session of each strategy after a restart. A position settles only when
the market is `finalized`, because `determined` results can still be disputed.

## Data checks and maintenance

```bash
python main.py database validate
```

Checks integrity, classification, probability bounds, NO/YES complements, volume, resolutions and
timestamps, and that no ticker is in both databases. `warn` means a limitation worth knowing rather
than corruption; a warning about finalized markets without resolutions usually just means a write was
in progress. `database backup` is safe while collectors run. Schema changes:

```bash
python scripts/make_migration.py "describe the change"
```

## Known limitations

**Limits set by Kalshi**

- No historical order books. Backtests before you began collecting use candles, with no depth.
- `/markets` quotes and volume lag by about 10 seconds. `volume` seen by strategies can trail quotes.
- Candles run at 1, 60 or 1440 minutes, so intra-candle highs and lows are invisible to stops.
- For markets first seen after they closed, the original close time is unknowable.
- Thursday 3:00–5:00 AM ET maintenance pauses trading. Simulated fills do not model it unless market
  status changes.
- Retention of historical data is undocumented.

**Limits of this implementation**

- **WebSocket has never connected to Kalshi.** Provide a key, run briefly, then check the `ws_connected`
  log line and that `observations` shows `source = 2`. Protocol details could differ from the mock.
- REST alone cannot see every quote: general markets outside the hot set update every 30 minutes, and
  the 15-minute feed samples about once per second.
- Maker orders are not simulated: without queue position any estimate would be invented. Strategies
  that depend on resting orders cannot be evaluated.
- Fills do not model your own market impact beyond the participation cap.
- `fee_type` `flat` (unused today) and scalar sub-cent settlement fees are not modelled specifically.
- Exit fills use the same limits as entries, so an exit may be delayed by thin books (see `failed_exits`).
- The limiter's endpoint costs were measured once and may change.
- Real-data backtests so far are small samples. They prove the machinery, not any strategy.

More detail: [docs/API_NOTES.md](docs/API_NOTES.md), [docs/DATABASE.md](docs/DATABASE.md).
