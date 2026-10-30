# Database reference

`data/general.sqlite` and `data/fifteen_minute.sqlite` are independent files with the same core
schema, defined in `kalshi_quant/shared/database/schema.py`. All timestamps are UTC milliseconds
(`*_ms`). Stored prices are integers in 1/10,000 dollar, counts in 1/100 contract. **Use the views for
decimal values.**

## Views

| View | Contents |
|---|---|
| `market_overview` | One row per market: identity, category, status, outcome, current probability, volume |
| `observations_decimal` | Observations with probabilities in [0, 1], spread, and decimal sizes |
| `trades_decimal` | Trades with decimal prices and contracts |

```sql
SELECT ticker, ts_ms, yes_prob, no_prob, spread, cumulative_volume
FROM observations_decimal WHERE ticker = 'KXBTC15M-26OCT301830-30' ORDER BY ts_ms;
```

## Tables

| Group | Table | Contents |
|---|---|---|
| Metadata | `categories`, `series`, `fee_changes`, `events`, `markets` | Hierarchy: category → series → event → market. Tags are stored as `subcategory` because Kalshi has no such field |
| Latest | `market_latest` | Newest state per market; used to resume after restarts |
| Prospective | `observations` | Quotes captured by this application. `source`: 1 REST market, 2 WebSocket ticker, 3 REST book, 4 WS book |
| Prospective | `orderbook_snapshots` | Best-first levels as JSON, with derived depth |
| Exchange | `trades` | Executed trades. `source`: 1 live feed, 2 WebSocket, 3 backfill |
| Exchange | `candlesticks` | OHLC of bid, ask and trade price, stamped at period end |
| Outcomes | `resolutions` | Result, winning and losing side, settlement value and time |
| Derived | `market_summary` | Initial and final probability with provenance. General also has initial and current projected winner |
| Derived | `market_interval_probs` | 15-minute only: probability on a fixed grid, last value at or before each point |
| Operations | `checkpoints`, `collection_gaps`, `heartbeats`, `db_growth`, `archives`, `classification_audit`, `raw_responses` | Resume state, coverage, health, growth, Parquet index, disagreements, zlib JSON payloads |
| Paper | `paper_sessions`, `paper_orders`, `paper_trades`, `paper_equity`, `paper_state` | Simulated activity |
| Research | `backtest_runs`, `backtest_trades`, `optimization_runs`, `optimization_trials` | Saved, reproducible results |

## Implied probability

Mid of a two-sided quote, else last trade, else null. `no_prob` is `1 - yes_prob`. NO bid and ask are
complements of YES ask and bid.

## Initial projected winner

From the earliest two-sided quote with a spread of at most 0.20, the earliest candle meeting the same
test, or the first trade, whichever came first. `initial_source` says which, `initial_delay_s` says how
long after open, and `initial_reliable` is false when that delay exceeds 1 hour (general) or 60 seconds
(15-minute). A market listed before collection began is corrected once candles are backfilled.
Opening books are often 0.01/0.99, so a wide quote never counts.

## Query examples

```bash
python main.py database query general --category Crypto --result yes --min-volume 10000
```

```bash
python main.py database query general --initial-winner no --closes-after 2026-10-01 --min-probability 0.2
```

```sql
SELECT category, count(*), avg(result = 'yes') FROM market_overview
WHERE result IS NOT NULL GROUP BY category;
```

## Archives

Rows older than `archive_after_days` move to `data/archive/<env>/<table>-<day>.parquet` (zstd). A file
is written and read back before its rows are deleted. `archives` indexes them, and `load_dataset` merges
them with live rows.

## Migrating to PostgreSQL

1. Set `database_url` in the environment's YAML to a `postgresql+psycopg://` URL and install the driver.
2. Start once: Alembic builds the schema, and upserts already use PostgreSQL's `ON CONFLICT`.
3. Copy rows with your preferred tool.
4. Replace SQLite-specific parts: `integrity_check`, `backup`, and SQLite PRAGMAs and `WITHOUT ROWID`
   options. The `avg(result = 'yes')` style in hand-written SQL also differs.

This path has not been exercised.
