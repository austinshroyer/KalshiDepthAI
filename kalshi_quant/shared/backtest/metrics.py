"""Performance metrics. A ratio that the sample cannot support is reported as None with a
reason, rather than as a number that looks meaningful."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from kalshi_quant.shared.execution.portfolio import ClosedTrade
from kalshi_quant.shared.utilities.timeutil import ms_to_dt

MIN_DAYS_FOR_RATIOS = 30
DAYS_PER_YEAR = 365  # Kalshi trades every day


def _streaks(pnls: list[float]) -> tuple[int, int]:
    best_win = best_loss = win = loss = 0
    for pnl in pnls:
        win, loss = (win + 1, 0) if pnl > 0 else (0, loss + 1)
        best_win, best_loss = max(best_win, win), max(best_loss, loss)
    return best_win, best_loss


def _group(trades: list[ClosedTrade], key: Any) -> dict[str, dict[str, float]]:
    groups: dict[str, list[ClosedTrade]] = defaultdict(list)
    for trade in trades:
        groups[str(key(trade))].append(trade)
    return {name: {"trades": len(g), "pnl": sum(t.pnl for t in g), "fees": sum(t.fees for t in g),
                   "win_rate": sum(t.pnl > 0 for t in g) / len(g)} for name, g in sorted(groups.items())}


def drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    return float(((peaks - equity) / peaks).max())


def daily_returns(equity_ts: np.ndarray, equity: np.ndarray) -> np.ndarray:
    if len(equity) < 2:
        return np.empty(0)
    days = equity_ts // 86_400_000
    last_each_day = np.nonzero(np.r_[days[1:] != days[:-1], True])[0]
    closes = np.r_[equity[0], equity[last_each_day]]
    return closes[1:] / closes[:-1] - 1


def compute_metrics(trades: list[ClosedTrade], starting_capital: float, equity_ts: np.ndarray, equity: np.ndarray,
                    exposure: np.ndarray, categories: dict[str, str | None], total_slippage: float) -> dict[str, Any]:
    pnls = [t.pnl for t in trades]
    wins, losses = [p for p in pnls if p > 0], [p for p in pnls if p <= 0]
    count = len(trades)
    final = float(equity[-1]) if len(equity) else starting_capital
    gross_win, gross_loss = sum(wins), -sum(losses)
    returns_on_cost = [t.pnl / (t.quantity * t.entry_price) for t in trades if t.entry_price > 0]
    daily = daily_returns(equity_ts, equity)
    win_streak, loss_streak = _streaks(pnls)
    metrics: dict[str, Any] = {
        "starting_capital": starting_capital, "final_equity": final, "total_pnl": final - starting_capital,
        "total_return": final / starting_capital - 1, "return_on_starting_capital": (final - starting_capital) / starting_capital,
        "trades": count, "win_rate": len(wins) / count if count else None, "loss_rate": len(losses) / count if count else None,
        "average_win": sum(wins) / len(wins) if wins else None, "average_loss": sum(losses) / len(losses) if losses else None,
        "average_trade_return": float(np.mean(returns_on_cost)) if returns_on_cost else None,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else None,
        "profit_factor_note": None if gross_loss > 0 else "undefined: no losing trades",
        "max_drawdown": drawdown(equity),
        "average_holding_seconds": sum(t.exit_ms - t.entry_ms for t in trades) / count / 1000 if count else None,
        "longest_win_streak": win_streak, "longest_loss_streak": loss_streak,
        "average_position_size": sum(t.quantity * t.entry_price for t in trades) / count if count else None,
        "capital_utilization": float(np.mean(exposure)) if len(exposure) else 0.0,
        "fees": sum(t.fees for t in trades), "estimated_slippage": total_slippage,
        "by_category": _group(trades, lambda t: categories.get(t.ticker)),
        "by_side": _group(trades, lambda t: t.side),
        "by_month": _group(trades, lambda t: ms_to_dt(t.exit_ms).strftime("%Y-%m")),
        "by_market": _group(trades, lambda t: t.ticker),
        "by_exit_reason": _group(trades, lambda t: t.exit_reason),
        "days": len(daily),
    }
    metrics["return_to_drawdown"] = metrics["total_return"] / metrics["max_drawdown"] if metrics["max_drawdown"] > 0 else None
    sharpe = sortino = None
    note = None
    if len(daily) < MIN_DAYS_FOR_RATIOS:
        note = f"not computed: {len(daily)} days of returns, {MIN_DAYS_FOR_RATIOS} required"
    elif daily.std(ddof=1) == 0:
        note = "not computed: returns have no variance"
    else:
        scale = math.sqrt(DAYS_PER_YEAR)
        sharpe = float(daily.mean() / daily.std(ddof=1) * scale)
        downside = np.sqrt(np.mean(np.minimum(daily, 0) ** 2))
        sortino = float(daily.mean() / downside * scale) if downside > 0 else None
    metrics.update(sharpe=sharpe, sortino=sortino, risk_ratio_note=note)
    return metrics
