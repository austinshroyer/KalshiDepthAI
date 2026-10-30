"""Command line interface. Every command names its environment, so results cannot mix."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from kalshi_quant.shared import reports
from kalshi_quant.shared.config import PROJECT_ROOT, Env, EnvConfig, load_config
from kalshi_quant.shared.database.engine import Database
from kalshi_quant.shared.utilities.logging import configure_logging
from kalshi_quant.shared.utilities.timeutil import ms_to_iso, now_ms, parse_date_ms

app = typer.Typer(no_args_is_help=True, help="Kalshi research, collection, paper trading and backtesting.")
database_app = typer.Typer(no_args_is_help=True, help="Inspect, validate and maintain the databases.")
paper_app = typer.Typer(no_args_is_help=True, help="Paper trading. Never places real orders.")
backtest_app = typer.Typer(help="Historical backtests.")
app.add_typer(database_app, name="database")
app.add_typer(paper_app, name="paper")

EnvArg = Annotated[str, typer.Argument(help="general or 15m")]
DbEnv = Annotated[str, typer.Option("--env", help="general, 15m or all")]


def _envs(value: str) -> list[Env]:
    return list(Env) if value == "all" else [Env.parse(value)]


def _context(env: str) -> tuple[EnvConfig, Database]:
    config = load_config(env)
    return config, Database.for_config(config)


@app.command()
def collect(target: Annotated[str, typer.Argument(help="all, general or 15m")] = "all",
            duration: Annotated[Optional[float], typer.Option(help="Stop after this many seconds")] = None) -> None:
    """Run collection. `all` starts both pipelines concurrently."""
    from kalshi_quant.master_collector.run import collect as run
    run(_envs(target), duration)


@app.command()
def backfill(env: EnvArg, since: Annotated[Optional[str], typer.Option(help="Close date lower bound, YYYY-MM-DD")] = None,
             until: Annotated[Optional[str], typer.Option()] = None, series: Annotated[Optional[str], typer.Option()] = None,
             limit: Annotated[Optional[int], typer.Option(help="Most recent N markets")] = None,
             period: Annotated[Optional[int], typer.Option(help="Candle minutes: 1, 60 or 1440")] = None,
             trades: Annotated[bool, typer.Option(help="Also fetch every executed trade")] = False) -> None:
    """Fetch exchange history (candles, trades) for markets already in the database."""
    from kalshi_quant.shared.api.auth import Credentials
    from kalshi_quant.shared.api.client import KalshiClient
    from kalshi_quant.shared.api.rate_limit import AdaptiveRateLimiter
    from kalshi_quant.shared.collection.backfill import backfill_candles, backfill_trades
    config, db = _context(env)
    configure_logging(f"backfill_{config.env.value}", console=True)
    minutes = period or (1 if config.env is Env.FIFTEEN_MINUTE else 60)

    async def run() -> None:
        async with KalshiClient(config.shared.api, AdaptiveRateLimiter(config.tokens_per_second), Credentials.from_environment()) as client:
            written = await backfill_candles(client, db, config, minutes, parse_date_ms(since), parse_date_ms(until), series, limit)
            typer.echo(f"candles written: {written}")
            if trades:
                typer.echo(f"trades written: {await backfill_trades(client, db, parse_date_ms(since), parse_date_ms(until), series, limit)}")

    asyncio.run(run())


def _spec(config: EnvConfig, start: str | None, end: str | None, category: list[str] | None, series: list[str] | None, source: str):
    from kalshi_quant.shared.backtest.dataset import DatasetSpec
    return DatasetSpec(config.env.value, parse_date_ms(start), parse_date_ms(end), category or [], series or [], source=source)


@app.command("backtest")
def backtest_command(
    env: EnvArg, strategy: Annotated[str, typer.Option(help="File name inside this environment's strategy directory, or a path")],
    start: Annotated[Optional[str], typer.Option()] = None, end: Annotated[Optional[str], typer.Option()] = None,
    category: Annotated[Optional[list[str]], typer.Option()] = None, series: Annotated[Optional[list[str]], typer.Option()] = None,
    source: Annotated[str, typer.Option(help="auto, observations or candles")] = "auto",
    capital: Annotated[Optional[float], typer.Option()] = None, seed: int = 0,
    set_param: Annotated[Optional[list[str]], typer.Option("--set", help="name=value override")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Replay a strategy over history and save a reproducible run."""
    from kalshi_quant.shared.backtest.dataset import load_dataset
    from kalshi_quant.shared.backtest.engine import BacktestSettings, run_backtest, save_run
    from kalshi_quant.shared.strategy.loader import load_strategy
    config, db = _context(env)
    params = {k: yaml.safe_load(v) for k, v in (p.split("=", 1) for p in set_param or [])}
    instance, _ = load_strategy(config, strategy, params)
    dataset = load_dataset(db, _spec(config, start, end, category, series, source))
    if not len(dataset):
        raise typer.BadParameter("no data for that selection; run `backfill` or let the collector run first")
    settings = BacktestSettings.from_config(config, capital, seed)
    result = run_backtest(instance, dataset, settings)
    run_id = save_run(db, instance, dataset, settings, result)
    if json_output:
        typer.echo(json.dumps({"run_id": run_id, **result.metrics}, default=str))
    else:
        reports.backtest_report(result.metrics, run_id)
        typer.echo(f"saved run {run_id}; dataset {dataset.dataset_id()}")


@app.command()
def replay(env: EnvArg, run_id: str) -> None:
    """Re-execute a saved backtest and confirm it reproduces."""
    from kalshi_quant.shared.backtest.engine import replay_run
    _, db = _context(env)
    out = replay_run(db, run_id, PROJECT_ROOT / "data" / "cache")
    typer.echo(json.dumps({k: v for k, v in out.items() if k != "metrics"}, indent=2))
    if not out["dataset_matches"]:
        typer.echo("The stored data changed since this run (for example, later backfill), so results may differ.")
    raise typer.Exit(0 if out["trades_match"] else 1)


@app.command("runs")
def list_runs(env: EnvArg, limit: int = 20) -> None:
    """List saved backtests."""
    from sqlalchemy import text
    _, db = _context(env)
    with db.connect() as conn:
        rows = conn.execute(text("""SELECT run_id, created_ms, strategy_name, dataset_id, metrics FROM backtest_runs
                                    ORDER BY created_ms DESC LIMIT :n"""), {"n": limit}).mappings().all()
    items = []
    for row in rows:
        m = json.loads(row["metrics"])
        items.append({**{k: row[k] for k in ("run_id", "created_ms", "strategy_name")}, "return": m["total_return"], "drawdown": m["max_drawdown"], "trades": m["trades"]})
    reports.records("Backtests", items)


@app.command()
def optimize(
    env: EnvArg, strategy: Annotated[str, typer.Option()],
    space: Annotated[Optional[Path], typer.Option(help="YAML of parameter ranges. Default: `optimization:` in the strategy's YAML, else Param ranges")] = None,
    method: Annotated[str, typer.Option(help="grid, random or adaptive")] = "grid",
    objective: Annotated[str, typer.Option(help="total_return, risk_adjusted, profit_factor, min_drawdown or custom")] = "risk_adjusted",
    max_iterations: int = 500, min_trades: int = 30, min_return: float = 0.0, workers: int = 1, folds: int = 4, seed: int = 0,
    start: Annotated[Optional[str], typer.Option()] = None, end: Annotated[Optional[str], typer.Option()] = None,
    resume: Annotated[Optional[str], typer.Option(help="Continue this optimization id")] = None,
) -> None:
    """Search parameters with train, validation and test splits, walk-forward and overfitting checks."""
    from kalshi_quant.shared.backtest.dataset import load_dataset
    from kalshi_quant.shared.backtest.engine import BacktestSettings
    from kalshi_quant.shared.optimization.engine import OptimizationConfig, Optimizer
    from kalshi_quant.shared.optimization.space import parse_space
    from kalshi_quant.shared.strategy.loader import load_strategy, read_settings
    config, db = _context(env)
    instance, path = load_strategy(config, strategy)
    if space:
        ranges = yaml.safe_load(space.read_text())
    else:
        ranges = read_settings(path).get("optimization") or {
            k: {"min": p.min, "max": p.max, "step": p.step} for k, p in instance.parameters.items() if p.step}
    if not ranges:
        raise typer.BadParameter("no parameter ranges defined")
    dataset = load_dataset(db, _spec(config, start, end, None, None, "auto"))
    if not len(dataset):
        raise typer.BadParameter("no data available")
    options = OptimizationConfig(method, objective, max_iterations, min_trades, min_return, workers=workers,
                                 walk_forward_folds=folds, seed=seed)
    with Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeRemainingColumn()) as bar:
        task = bar.add_task("starting", total=None)
        def update(done: int, total: int, stage: str) -> None:
            bar.update(task, description=stage, completed=done, total=total)
        report = Optimizer(db, instance, path, dataset, BacktestSettings.from_config(config), parse_space(ranges), options, resume, progress=update).run()
    reports.optimization_report(report)


@app.command()
def evolve(
    env: EnvArg, profile: Annotated[Optional[str], typer.Option(help="smoke or quick. Omit for the full 50-cycle process")] = None,
    category: Annotated[Optional[list[str]], typer.Option(help="Limit category populations")] = None,
    start: Annotated[Optional[str], typer.Option()] = None, end: Annotated[Optional[str], typer.Option()] = None,
    source: Annotated[str, typer.Option(help="auto, observations or candles")] = "auto",
    workers: int = 1, seed: int = 0, resume: Annotated[Optional[str], typer.Option(help="Continue this run id")] = None,
) -> None:
    """Evolve category, cross-category and hybrid populations, then validate out of sample."""
    from kalshi_quant.shared.backtest.dataset import load_dataset
    from kalshi_quant.shared.evolution.engine import EvolutionEngine
    from kalshi_quant.shared.evolution.report import render_markdown
    from kalshi_quant.shared.evolution.settings import EvolutionSettings
    config, db = _context(env)
    configure_logging(f"evolve_{config.env.value}", console=False)
    dataset = load_dataset(db, _spec(config, start, end, None, None, source))
    if not len(dataset):
        raise typer.BadParameter("no data available; run `backfill` or collect first")
    engine = EvolutionEngine(db, config, dataset, EvolutionSettings(config.env, profile), resume, workers, seed, only_categories=category, progress=typer.echo)
    typer.echo(f"run {engine.run_id}: {len(engine.populations)} category populations; insufficient: {sorted(engine.insufficient)}")
    typer.echo(render_markdown(engine.run()))
    typer.echo(f"saved to {engine.root}")


# Paper trading
@paper_app.command("start")
def paper_start(
    env: EnvArg, strategy: Annotated[list[str], typer.Option(help="Repeat to run several strategies, each with its own portfolio")],
    capital: Annotated[Optional[float], typer.Option()] = None, resume: bool = False,
    duration: Annotated[Optional[float], typer.Option()] = None,
    background: Annotated[bool, typer.Option(help="Detach and keep running after this command exits")] = False,
) -> None:
    """Start paper trading live markets."""
    if background:
        command = [sys.executable, str(PROJECT_ROOT / "main.py"), "paper", "start", env]
        for item in strategy:
            command += ["--strategy", item]
        for flag, value in (("--capital", capital), ("--duration", duration)):
            if value is not None:
                command += [flag, str(value)]
        command += ["--resume"] if resume else []
        log = (PROJECT_ROOT / "logs" / f"paper_{Env.parse(env).value}.out").open("ab")
        process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True, cwd=PROJECT_ROOT)
        typer.echo(f"started in background, pid {process.pid}")
        return
    from kalshi_quant.shared.paper.engine import PaperEngine
    from kalshi_quant.shared.strategy.loader import load_strategy
    config, db = _context(env)
    configure_logging(f"paper_{config.env.value}")
    engine = PaperEngine(config, [load_strategy(config, name) for name in strategy], capital, db, resume=resume)
    asyncio.run(engine.run(duration))


@paper_app.command("general", help="Shortcut for `paper start general`.")
def paper_general(strategy: Annotated[list[str], typer.Option()], capital: Annotated[Optional[float], typer.Option()] = None,
                  resume: bool = False, duration: Annotated[Optional[float], typer.Option()] = None, background: bool = False) -> None:
    paper_start("general", strategy, capital, resume, duration, background)


@paper_app.command("15m", help="Shortcut for `paper start 15m`.")
def paper_fifteen(strategy: Annotated[list[str], typer.Option()], capital: Annotated[Optional[float], typer.Option()] = None,
                  resume: bool = False, duration: Annotated[Optional[float], typer.Option()] = None, background: bool = False) -> None:
    paper_start("15m", strategy, capital, resume, duration, background)


@paper_app.command("stop")
def paper_stop(env: EnvArg, session: int) -> None:
    """Ask a running session to stop."""
    from kalshi_quant.shared.paper import commands
    _, db = _context(env)
    typer.echo(commands.request_stop(db, session))


@paper_app.command("sessions")
def paper_sessions(env: EnvArg) -> None:
    """List sessions."""
    from kalshi_quant.shared.paper import commands
    _, db = _context(env)
    reports.records("Paper sessions", commands.sessions(db), ["session_id", "strategy_name", "status", "pid", "started_ms", "cash", "open_positions", "realized_pnl", "fees"])


@paper_app.command("positions")
def paper_positions(env: DbEnv = "all") -> None:
    """Open positions."""
    from kalshi_quant.shared.paper import commands
    for item in _envs(env):
        _, db = _context(item.value)
        reports.records(f"Open positions: {item.value}", commands.positions(db))


@paper_app.command("capital")
def paper_capital(env: DbEnv = "all") -> None:
    """Available capital per session."""
    from kalshi_quant.shared.paper import commands
    for item in _envs(env):
        _, db = _context(item.value)
        reports.records(f"Capital: {item.value}", commands.sessions(db), ["session_id", "strategy_name", "starting_capital", "cash", "open_positions"])


@paper_app.command("performance")
def paper_performance(env: DbEnv = "all") -> None:
    """Realized and unrealized P&L."""
    from kalshi_quant.shared.paper import commands
    for item in _envs(env):
        _, db = _context(item.value)
        reports.records(f"Performance: {item.value}", commands.performance(db))


@paper_app.command("trades")
def paper_trades(env: EnvArg, session: Annotated[Optional[int], typer.Option()] = None, limit: int = 50) -> None:
    """Closed paper trades."""
    from kalshi_quant.shared.paper import commands
    _, db = _context(env)
    reports.records("Paper trades", commands.trades(db, session, limit), ["session_id", "ticker", "side", "quantity", "entry_price", "exit_price", "exit_ms", "pnl", "fees", "exit_reason"])


# Database
@database_app.command("stats")
def database_stats_command(env: DbEnv = "all") -> None:
    """Row counts, sizes, coverage and collector health."""
    from kalshi_quant.shared.database.maintenance import database_stats
    for item in _envs(env):
        _, db = _context(item.value)
        s = database_stats(db)
        start, stop = s["observation_span"]
        reports.key_values(f"Database: {s['environment']}", [
            ("Path", s["path"]), ("Size", f"{s['file_bytes'] / 1e6:,.1f} MB"), *s["rows"].items(), ("Archived rows", s["archived_rows"]),
            ("Observations from", ms_to_iso(start)), ("Observations to", ms_to_iso(stop)), ("Recorded gaps", s["gaps"])])
        now = now_ms()
        reports.records("Streams", [{"stream": a, "age_seconds": (now - b) / 1000, "detail": c} for a, b, c in s["heartbeats"]])


@database_app.command("growth")
def database_growth(env: DbEnv = "all") -> None:
    """Database size over time, and projected days until the disk guard engages."""
    from kalshi_quant.shared.database.maintenance import file_bytes, growth_report
    from kalshi_quant.shared.utilities.disk import free_gb
    for item in _envs(env):
        config, db = _context(item.value)
        rows = growth_report(db)
        reports.records(f"Growth: {item.value}", [{"ts_ms": r["ts_ms"], "mb": r["file_bytes"] / 1e6, **r["rows"]} for r in rows[-24:]], ["ts_ms", "mb", "observations", "trades", "orderbook_snapshots"])
        if len(rows) >= 2 and rows[-1]["ts_ms"] > rows[0]["ts_ms"]:
            per_day = (rows[-1]["file_bytes"] - rows[0]["file_bytes"]) / (rows[-1]["ts_ms"] - rows[0]["ts_ms"]) * 86_400_000
            free = free_gb(db.path.parent) - config.shared.storage.min_free_disk_gb
            typer.echo(f"growth {per_day / 1e9:.2f} GB/day; about {free / (per_day / 1e9):.1f} days until the disk guard" if per_day > 0 else "no growth")
        typer.echo(f"current size {file_bytes(db) / 1e6:,.1f} MB")


@database_app.command("validate")
def database_validate(env: DbEnv = "all") -> None:
    """Run integrity, classification, probability, volume, resolution and timestamp checks."""
    from kalshi_quant.shared.collection.validation import validate, validate_separation
    failed = False
    dbs = {}
    for item in _envs(env):
        _, db = _context(item.value)
        dbs[item] = db
        checks = validate(db)
        failed |= any(not c.passed and not c.warning for c in checks)
        reports.records(f"Validation: {item.value}", [{"check": c.name, "result": "warn" if c.warning else "pass" if c.passed else "FAIL", "detail": c.detail[:90]} for c in checks])
    if len(dbs) == 2:
        check = validate_separation(*dbs.values())
        failed |= not check.passed
        typer.echo(f"{check.name}: {'pass' if check.passed else 'FAIL'} ({check.detail})")
    raise typer.Exit(1 if failed else 0)


@database_app.command("backup")
def database_backup(env: DbEnv = "all") -> None:
    """Consistent compressed backup; safe while collectors run."""
    from kalshi_quant.shared.database.maintenance import backup
    for item in _envs(env):
        _, db = _context(item.value)
        typer.echo(backup(db))


@database_app.command("archive")
def database_archive(env: DbEnv = "all", days: int = 7) -> None:
    """Move old time-series days to Parquet now, instead of waiting for the collector."""
    from kalshi_quant.shared.database.maintenance import run_archive
    for item in _envs(env):
        config, db = _context(item.value)
        typer.echo(f"{item.value}: archived {run_archive(db, config, days)} rows")


@database_app.command("query")
def database_query(
    env: EnvArg, category: Optional[str] = None, series: Optional[str] = None, event: Optional[str] = None, ticker: Optional[str] = None,
    status: Optional[str] = None, closes_after: Optional[str] = None, closes_before: Optional[str] = None,
    min_duration: Optional[int] = None, max_duration: Optional[int] = None, result: Optional[str] = None,
    min_probability: Optional[float] = None, max_probability: Optional[float] = None, min_volume: Optional[float] = None,
    initial_winner: Optional[str] = None, limit: int = 50,
) -> None:
    """Find markets by category, event, date, duration, resolution, probability or volume."""
    from kalshi_quant.shared.database.queries import MarketQuery, find_markets
    _, db = _context(env)
    frame = find_markets(db, MarketQuery(category, series, event, ticker, status, None, None, parse_date_ms(closes_after), parse_date_ms(closes_before),
                                         min_duration, max_duration, result, min_probability, max_probability, min_volume, initial_winner, limit))
    reports.records(f"{len(frame)} markets", frame.to_dict("records"))


def main() -> None:
    app()
