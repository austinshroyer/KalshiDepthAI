"""Terminal rendering. Shows evidence and its limits, and never ranks on return alone."""
from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

from kalshi_quant.shared.utilities.timeutil import ms_to_iso

console = Console()


def _fmt(value: Any, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    if percent:
        return f"{value:.2%}"
    return f"{value:,.4f}" if isinstance(value, float) else str(value)


def key_values(title: str, rows: list[tuple[str, Any]]) -> None:
    table = Table(title=title, show_header=False)
    table.add_column(style="cyan")
    table.add_column(justify="right")
    for key, value in rows:
        table.add_row(key, str(value))
    console.print(table)


def records(title: str, items: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    if not items:
        console.print(f"[dim]{title}: none[/dim]")
        return
    columns = columns or list(items[0])
    table = Table(title=title)
    for column in columns:
        table.add_column(column)
    for item in items:
        cells = []
        for column in columns:
            value = item.get(column)
            if column.endswith("_ms") and value:
                value = ms_to_iso(value)
            cells.append(_fmt(value))
        table.add_row(*cells)
    console.print(table)


def backtest_report(m: dict[str, Any], run_id: str | None = None) -> None:
    pct = lambda k: _fmt(m.get(k), True)  # noqa: E731
    key_values(f"Backtest {run_id or ''}".strip(), [
        ("Total return", pct("total_return")), ("Total P&L", f"${m['total_pnl']:,.2f}"), ("Return on starting capital", pct("return_on_starting_capital")),
        ("Trades", m["trades"]), ("Win rate", pct("win_rate")), ("Loss rate", pct("loss_rate")),
        ("Average win", _fmt(m["average_win"])), ("Average loss", _fmt(m["average_loss"])), ("Average trade return", pct("average_trade_return")),
        ("Profit factor", _fmt(m["profit_factor"])), ("Max drawdown", pct("max_drawdown")), ("Return / drawdown", _fmt(m["return_to_drawdown"])),
        ("Sharpe", _fmt(m["sharpe"]) if m["sharpe"] is not None else m["risk_ratio_note"]),
        ("Sortino", _fmt(m["sortino"]) if m["sortino"] is not None else m["risk_ratio_note"]),
        ("Avg holding (s)", _fmt(m["average_holding_seconds"])),
        ("Longest win streak", m["longest_win_streak"]), ("Longest loss streak", m["longest_loss_streak"]),
        ("Avg position ($)", _fmt(m["average_position_size"])), ("Capital utilization", pct("capital_utilization")),
        ("Fees", f"${m['fees']:,.2f}"), ("Estimated slippage", f"${m['estimated_slippage']:,.2f}"),
    ])
    for label, key in (("By side", "by_side"), ("By category", "by_category"), ("By month", "by_month"), ("By exit reason", "by_exit_reason")):
        records(label, [{"group": k, **v} for k, v in m[key].items()], ["group", "trades", "pnl", "fees", "win_rate"])
    top = sorted(m["by_market"].items(), key=lambda kv: kv[1]["pnl"])
    shown = top[:5] + top[-5:] if len(top) > 10 else top
    records(f"By market (worst and best of {len(top)})", [{"market": k, **v} for k, v in shown], ["market", "trades", "pnl", "fees"])
    q = m["data_quality"]
    key_values("Execution and data quality", [
        ("Rows replayed", q["rows_replayed"]), ("Markets", q["markets_in_dataset"]), ("Signals", q["signals"]), ("Orders", q["orders"]),
        ("Partial fills", q["partial_fills"]), ("Rejected: insufficient liquidity", q["rejected_insufficient_liquidity"]),
        ("Rejected: other execution", q["rejected_execution"] or 0), ("Blocked by capital", q["capital_blocked"]),
        ("Blocked by exposure", q["exposure_blocked"]), ("Failed exits", q["failed_exits"]),
        ("Unsettled at end (marked to bid)", q["unsettled_positions_at_end"]), ("Collection gaps in range", q["collection_gaps_overlapping"]),
        ("Dataset warnings", q["warnings"]),
    ])


def optimization_report(r: dict[str, Any]) -> None:
    key_values("Optimization", [("Run", r["opt_id"]), ("Strategy", r["strategy"]), ("Objective", r["objective"]), ("Method", r["method"]),
                                ("Trials", r["trials"]), ("Eligible", r["eligible_trials"]), ("Space size", r["space_size"]), ("Rows", r["rows"])])
    if "selected" in r:
        console.print(f"Selected: {r['selected']['params']} ({r['selected']['selection']})")
        rows = [{"period": k, **r[k]} for k in ("train", "validation", "test")]
        records("In sample versus out of sample", rows, ["period", "total_return", "trades", "win_rate", "profit_factor", "max_drawdown", "sharpe"])
        d = r["deterioration"]["test"]
        key_values("Deterioration on test", [("Train return/day", _fmt(d["train_return_per_day"])), ("Test return/day", _fmt(d["out_of_sample_return_per_day"])),
                                             ("Retained", _fmt(d["retained_fraction"], True))])
        wf = r["walk_forward"]
        key_values("Walk-forward", [("Folds profitable", f"{wf.get('folds_profitable')} of {wf.get('folds_evaluated')}"),
                                    ("Mean in-sample return", _fmt(wf.get("mean_in_sample_return"), True)),
                                    ("Mean out-of-sample return", _fmt(wf.get("mean_out_of_sample_return"), True))])
        s, t = r["sensitivity"], r["multiple_testing"]
        key_values("Robustness", [("Objective kept by neighbours", _fmt(s["objective_retained_by_neighbors"], True)),
                                  ("Deflated Sharpe", _fmt(t.get("deflated_sharpe"))), ("Overfitting probability (CSCV)", _fmt((t.get("cscv") or {}).get("pbo"), True))])
    verdict = r["verdict"]
    console.print("[bold]Verdict:[/bold] " + ("no specific reason for doubt found" if verdict["robust"] else "NOT robust"))
    for flag in verdict["flags"]:
        console.print(f"  - {flag}")
    if verdict.get("note"):
        console.print(f"[dim]{verdict['note']}[/dim]")
