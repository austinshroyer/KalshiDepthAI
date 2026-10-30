# Kalshi API notes

Findings from the official docs (docs.kalshi.com, OpenAPI and AsyncAPI specs) and from direct
measurement on 2026-10-30. Measured items are marked; they can change without notice.

## Endpoints used

| Purpose | Endpoint | Auth | Notes |
|---|---|---|---|
| Series | `GET /series`, `/series/{t}` | No | 15,575 series; `frequency` drives classification |
| Fee changes | `GET /series/fee_changes` | No | Series also carry `fee_type`, `fee_multiplier` |
| Events | `GET /events` | No | Limit 200. `product_metadata.cadence` marks 15-minute events |
| Markets | `GET /markets` | No | Limit 1000. **Cached, see below** |
| Order books | `GET /markets/{t}/orderbook`, `/markets/orderbooks` | No | Live. Batch takes 100 tickers |
| Trades | `GET /markets/trades` | No | Newest first; `min_ts`, `ticker` filters |
| Candles | `GET /markets/candlesticks` | No | 1, 60 or 1440 minutes; 100 tickers |
| History | `GET /historical/*` | No (public data) | Markets, trades, candles before the cutoff |
| Cutoffs | `GET /historical/cutoff` | No | One timestamp per data type |

Not called: anything under `/portfolio`, order endpoints, and `/margin` (perpetuals).

The production hosts are `external-api.kalshi.com` and `external-api-ws.kalshi.com`.

## Measured: `/markets` is cached

`/markets` and `/markets/{ticker}` return `Cache-Control: public, max-age=15`. Polled at about 1.5s
beside the order book, their quotes and volume changed only every 10 seconds or so, while the book
tracked every trade. Example from one read: list quote 0.0680, book and last trade 0.0790.

Consequences: stamping such a quote "now" would corrupt history and paper fills. Quotes here come from
order books. `/markets` is used for metadata, status, results and volume.

## Measured: rate limit costs

Basic is documented as 200 read tokens per second, default cost 10, burst of 3 seconds. Paced at 10
requests per second for 8 to 10 seconds:

| Endpoint | Sustained successful rate | Implied cost |
|---|---|---|
| `/exchange/status`, `/series/{t}`, `/markets/{t}/orderbook` | 9.6+/s, no 429 | ≤ 20 |
| `/markets/orderbooks` with 5 or 50 tickers | 9.5/s, no 429 | ≤ 20, independent of tickers |
| `/markets/{ticker}`, `/historical/trades` | 9.6/s, no 429 | ≤ 20 |
| `/markets`, `/markets/trades`, `/events`, `/markets/candlesticks`, `/historical/markets` | about 5/s | about 40 |
| `/series` (all) | 1.8/s (slow response, no 429) | unknown |

`GET /account/endpoint_costs` is authoritative but needs a key. 429s carry no `Retry-After`.

## Universe (measured)

- 38 `fifteen_min` series across crypto, commodities, FX, rates, indices and one political series, plus
  one test series. Many only run in market hours; 17 markets were open on a Friday evening.
- 15-minute markets are listed about two days ahead as `initialized`, and last exactly 900 seconds.
- 170,612 open non-combo markets; a full listing is 171 pages, 372 MB, 26,256 with 24h volume.
- About 200 trades per second globally at the sample moment, roughly 45% on 15-minute markets and 20%
  combos.
- First page of an unfiltered `/markets` listing was entirely combos; use `mve_filter=exclude`.

## Data formats

- Prices are dollar strings with up to 4 decimals; counts are strings with up to 2. Stored as integers
  in 1/10,000 dollar and 1/100 contract.
- Tick grids vary per market: read `price_ranges`, not `price_level_structure`.
- Books list **bids only**, ascending. A YES ask is `1 − best NO bid`.
- An empty side appears as `0.0000` or `1.0000`. These normalize to no quote.
- `result` is `yes`, `no`, `scalar` or empty. `closed` is not `settled`; `determined` can be disputed.
- Series tickers are the first segment of event tickers for current markets.

## Historical data

`GET /historical/cutoff` returned `market_settled_ts` and `trades_created_ts` of 2026-08-31 on the test
date. Older markets, candles and trades are only on `/historical/*`. Cutoffs move forward. The docs
give no retention period. Exchange shards exist (`exchange_index`) but only affect order routing.

## WebSocket

`wss://external-api-ws.kalshi.com/trade-api/ws/v2`. Requires a signed handshake even for public
channels, so it is unavailable without a key. `ticker`, `trade` and `market_lifecycle_v2` can cover all
markets; `orderbook_delta` needs tickers and accepts `update_interval_ms` of 0, 100 or 1000. Kalshi
sends pings every 10 seconds. Sequence gaps need a resnapshot (`update_subscription`, `get_snapshot`).
Implemented from the spec and tested only against a local mock.

## Authentication

Headers `KALSHI-ACCESS-KEY`, `-TIMESTAMP` (ms) and `-SIGNATURE`: base64 signature of
`timestamp + METHOD + path`, path without query string. Ed25519 signs directly; RSA uses PSS with
SHA-256. Both are implemented.

## Sources

- [API documentation](https://docs.kalshi.com/welcome/index), including rate limits, historical data,
  market lifecycle, fee rounding and exchange sharding
- [Fee schedule, July 2026](https://kalshi.com/docs/kalshi-fee-schedule.pdf)
