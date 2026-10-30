"""Chronological replay through the same StrategyRunner used for paper trading."""
from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_quant.shared.backtest.dataset import Dataset, DatasetSpec, load_dataset, spec_to_json
from kalshi_quant.shared.backtest.metrics import compute_metrics
from kalshi_quant.shared.config import PROJECT_ROOT, ExecutionConfig, EnvConfig
from kalshi_quant.shared.data_models.tick import Tick
from kalshi_quant.shared.database.engine import Database
from kalshi_quant.shared.execution.fees import FeeSchedule
from kalshi_quant.shared.execution.fills import SELL, touch
from kalshi_quant.shared.execution.portfolio import ClosedTrade, Portfolio
from kalshi_quant.shared.strategy.base import Strategy
from kalshi_quant.shared.strategy.loader import load_class
from kalshi_quant.shared.strategy.runner import StrategyRunner
from kalshi_quant.shared.utilities.timeutil import now_ms

SAMPLE_MS = 3_600_000


@dataclass
class BacktestSettings:
    starting_capital: float
    latency_ms: int
    max_quote_age_seconds: float
    execution: ExecutionConfig
    fee_coefficient: float = 0.07
    seed: int = 0

    @classmethod
    def from_config(cls, config: EnvConfig, capital: float | None = None, seed: int = 0) -> "BacktestSettings":
        return cls(capital or config.backtest.starting_capital, config.backtest.latency_ms,
                   config.backtest.max_quote_age_seconds, config.execution, seed=seed)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["execution"] = self.execution.model_dump()
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "BacktestSettings":
        return cls(**{**data, "execution": ExecutionConfig(**data["execution"])})


@dataclass
class BacktestResult:
    metrics: dict[str, Any]
    trades: list[ClosedTrade]
    equity_ts: np.ndarray
    equity: np.ndarray
    daily_pnl: dict[int, float]


def _none(value: float) -> float | None:
    return None if math.isnan(value) else value


def run_backtest(strategy: Strategy, dataset: Dataset, settings: BacktestSettings) -> BacktestResult:
    strategy.rng = random.Random(settings.seed)
    portfolio = Portfolio(settings.starting_capital)
    fees = FeeSchedule(settings.fee_coefficient, settings.execution.fee_rounding_dollars)
    runner = StrategyRunner(strategy, portfolio, settings.execution, fees, settings.latency_ms,
                            int(settings.max_quote_age_seconds * 1000))
    for info in dataset.infos.values():
        runner.register(info)
    selected = np.array([t in dataset.infos and runner._is_selected(t) for t in dataset.tickers], dtype=bool)
    settlement_queue = sorted(dataset.settlements.values(), key=lambda s: s.ts_ms)
    next_settlement = 0
    samples_ts: list[int] = []
    samples_equity: list[float] = []
    samples_exposure: list[float] = []
    next_sample = (int(dataset.ts_ms[0]) // SAMPLE_MS + 1) * SAMPLE_MS if len(dataset) else 0
    rows = np.nonzero(selected[dataset.ticker_index])[0] if len(dataset) else np.empty(0, dtype=np.int64)
    columns = dataset.values
    # Plain lists replay roughly five times faster than indexing numpy per field.
    ts_list, index_list, status_list, candle_list = (a[rows].tolist() for a in (dataset.ts_ms, dataset.ticker_index, dataset.status, dataset.is_candle))
    bid, ask, last, bid_size, ask_size, volume, delta, oi = (columns[k][rows].tolist() for k in (
        "yes_bid", "yes_ask", "last", "yes_bid_size", "yes_ask_size", "volume", "volume_delta", "open_interest"))
    tickers = dataset.tickers
    has_levels = bool(dataset.levels)
    row_numbers = rows.tolist() if has_levels else []
    last_tick: dict[str, Tick] = {}

    def sample(ts: int) -> None:
        equity = portfolio.equity()
        samples_ts.append(ts)
        samples_equity.append(equity)
        samples_exposure.append(portfolio.exposure() / equity if equity > 0 else 0.0)

    for i, ts in enumerate(ts_list):
        while next_settlement < len(settlement_queue) and settlement_queue[next_settlement].ts_ms <= ts:
            s = settlement_queue[next_settlement]
            runner.settle(s.ticker, s.yes_value, s.ts_ms)
            next_settlement += 1
        while ts >= next_sample:
            sample(next_sample)
            next_sample += SAMPLE_MS
        levels = dataset.levels.get(row_numbers[i]) if has_levels else None
        tick = Tick(tickers[index_list[i]], ts, _none(bid[i]), _none(ask[i]), _none(last[i]), _none(bid_size[i]),
                    _none(ask_size[i]), _none(volume[i]), _none(delta[i]), _none(oi[i]), status_list[i],
                    levels[0] if levels else None, levels[1] if levels else None, candle_list[i])
        last_tick[tick.ticker] = tick
        runner.on_tick(tick)
    while next_settlement < len(settlement_queue):  # settlements after the final tick
        s = settlement_queue[next_settlement]
        if dataset.spec.end_ms is not None and s.ts_ms >= dataset.spec.end_ms:
            break
        runner.settle(s.ticker, s.yes_value, s.ts_ms)
        next_settlement += 1
    unsettled = 0
    end_ts = ts_list[-1] if ts_list else 0
    for ticker in list(portfolio.positions):  # data ended first: value at the executable bid
        position = portfolio.positions[ticker]
        tick = last_tick.get(ticker)
        price = (touch(tick, position.side, SELL) if tick else None) or position.mark_price or 0.0
        portfolio.cash += position.quantity * price
        portfolio._close(ticker, position.quantity, price, end_ts, 0.0, 0.0, "end_of_data")
        unsettled += 1
    if ts_list:
        sample(max(end_ts, samples_ts[-1] if samples_ts else end_ts))
    trades = sorted(portfolio.closed, key=lambda t: (t.exit_ms, t.ticker))
    equity_ts, equity = np.array(samples_ts, dtype=np.int64), np.array(samples_equity, dtype=np.float64)
    metrics = compute_metrics(trades, settings.starting_capital, equity_ts, equity, np.array(samples_exposure),
                              dataset.categories, portfolio.slippage)
    counters = runner.counters
    metrics["data_quality"] = {
        "rows_replayed": len(ts_list), "markets_in_dataset": len(dataset.tickers), "signals": counters.signals,
        "orders": counters.orders, "partial_fills": counters.partial_fills,
        "rejected_insufficient_liquidity": counters.rejected.get("insufficient_liquidity", 0),
        "rejected_execution": {k: v for k, v in counters.rejected.items() if k != "insufficient_liquidity"},
        "capital_blocked": counters.capital_blocked, "exposure_blocked": counters.exposure_blocked,
        "failed_exits": counters.failed_exits, "unsettled_positions_at_end": unsettled,
        "warnings": dataset.warnings, "collection_gaps_overlapping": _gaps_in_range(dataset),
    }
    daily: dict[int, float] = {}
    for trade in trades:
        day = trade.exit_ms // 86_400_000
        daily[day] = daily.get(day, 0.0) + trade.pnl
    return BacktestResult(metrics, trades, equity_ts, equity, daily)


def _gaps_in_range(dataset: Dataset) -> int:
    if not len(dataset):
        return 0
    start, end = dataset.time_range()
    return sum(1 for a, b in dataset.gaps_ms if b > start and a < end)


def trades_hash(trades: list[ClosedTrade]) -> str:
    """Compares values, not formatting: 50 and 50.0 are the same quantity."""
    rows = [{k: round(float(v), 8) if isinstance(v, (int, float)) else v for k, v in asdict(t).items()} for t in trades]
    text = json.dumps(rows, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
                              timeout=5).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def save_run(db: Database, strategy: Strategy, dataset: Dataset, settings: BacktestSettings, result: BacktestResult) -> str:
    run_id = uuid.uuid4().hex[:12]
    t = db.t
    with db.begin() as conn:
        conn.execute(t["backtest_runs"].insert().values(
            run_id=run_id, created_ms=now_ms(), strategy_name=strategy.name, strategy_hash=strategy.source_hash(),
            strategy_source=strategy.source(), params=json.dumps(strategy.to_spec(), default=str),
            dataset_id=dataset.dataset_id(), dataset_spec=spec_to_json(dataset.spec), config=json.dumps(settings.to_json()),
            seed=settings.seed, git_commit=git_commit(), metrics=json.dumps(result.metrics, default=str),
            trades_hash=trades_hash(result.trades)))
        if result.trades:
            conn.execute(t["backtest_trades"].insert(), [{
                "run_id": run_id, "seq": i, "ticker": x.ticker, "side": x.side, "quantity": x.quantity,
                "entry_price": x.entry_price, "exit_price": x.exit_price, "entry_ms": x.entry_ms, "exit_ms": x.exit_ms,
                "pnl": x.pnl, "fees": x.fees, "slippage": x.slippage, "exit_reason": x.exit_reason,
                "category": dataset.categories.get(x.ticker)} for i, x in enumerate(result.trades)])
    return run_id


def replay_run(db: Database, run_id: str, work_dir: Path) -> dict[str, Any]:
    """Re-execute a saved run from its stored source, parameters, dataset spec and settings."""
    table = db.t["backtest_runs"]
    with db.connect() as conn:
        row = conn.execute(table.select().where(table.c.run_id == run_id)).mappings().first()
    if row is None:
        raise KeyError(run_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"replay_{run_id}.py"
    path.write_text(row["strategy_source"])
    spec = json.loads(row["params"])
    strategy = load_class(path, spec["class"])(spec["params"], spec["risk"], spec["universe"])
    dataset = load_dataset(db, DatasetSpec(**json.loads(row["dataset_spec"])))
    settings = BacktestSettings.from_json(json.loads(row["config"]))
    result = run_backtest(strategy, dataset, settings)
    new_hash, new_dataset = trades_hash(result.trades), dataset.dataset_id()
    return {"run_id": run_id, "dataset_matches": new_dataset == row["dataset_id"], "trades_match": new_hash == row["trades_hash"],
            "original_trades_hash": row["trades_hash"], "replayed_trades_hash": new_hash,
            "original_dataset": row["dataset_id"], "replayed_dataset": new_dataset, "metrics": result.metrics}
