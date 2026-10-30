"""Category-aware performance and the fitness functions for each population type.

Specialists and generalists have different goals, so each population type has its own
configuration. A strategy that only works in Crypto is a success for the Crypto population
and is not scored on breadth at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from kalshi_quant.shared.execution.portfolio import ClosedTrade

FAILED = -1_000_000.0


def _trade_return(trade: ClosedTrade) -> float:
    cost = trade.quantity * trade.entry_price
    return trade.pnl / cost if cost > 0 else 0.0


def performance(trades: list[ClosedTrade], capital: float) -> dict[str, Any]:
    """Statistics for any group of trades. Drawdown is measured on this group's own P&L path."""
    count = len(trades)
    if not count:
        return {"trades": 0, "return": 0.0, "pnl": 0.0, "win_rate": None, "profit_factor": None, "max_drawdown": 0.0,
                "average_trade_return": None, "risk_adjusted": None, "p_value": None}
    ordered = sorted(trades, key=lambda t: t.exit_ms)
    pnl = np.array([t.pnl for t in ordered])
    returns = np.array([_trade_return(t) for t in ordered])
    equity = capital + np.cumsum(pnl)
    peaks = np.maximum.accumulate(np.r_[capital, equity])[1:]
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    deviation = returns.std(ddof=1) if count > 1 else 0.0
    t_stat = float(returns.mean() / deviation * math.sqrt(count)) if deviation > 0 else None
    return {"trades": count, "return": float(pnl.sum() / capital), "pnl": float(pnl.sum()),
            "win_rate": float((pnl > 0).mean()), "profit_factor": float(wins / losses) if losses > 0 else None,
            "max_drawdown": float(((peaks - equity) / peaks).max()), "average_trade_return": float(returns.mean()),
            "risk_adjusted": t_stat, "p_value": float(stats.t.sf(t_stat, count - 1)) if t_stat is not None else None}


def consistency(category_stats: dict[str, dict[str, Any]], min_trades: int) -> float | None:
    """1 when active categories earn alike, falling toward 0 as they diverge."""
    values = [s["average_trade_return"] for s in category_stats.values() if s["trades"] >= min_trades]
    if len(values) < 2:
        return None
    mean = abs(float(np.mean(values)))
    spread = float(np.std(values))
    return mean / (mean + spread) if mean + spread > 0 else 0.0


def summarize(trades: list[ClosedTrade], capital: float, categories: dict[str, str | None], min_trades_per_category: int = 5) -> dict[str, Any]:
    groups: dict[str, list[ClosedTrade]] = {}
    for trade in trades:
        groups.setdefault(categories.get(trade.ticker) or "uncategorized", []).append(trade)
    by_category = {name: performance(group, capital) for name, group in sorted(groups.items())}
    overall = performance(trades, capital)
    thirds = None
    if len(trades) >= 3:
        ordered = sorted(trades, key=lambda t: t.exit_ms)
        thirds = [float(sum(t.pnl for t in part)) for part in np.array_split(np.array(ordered, dtype=object), 3)]
    return {**overall, "categories": by_category, "period_pnl": thirds,
            "category_consistency": consistency(by_category, min_trades_per_category)}


@dataclass
class FitnessConfig:
    min_trades: int = 20
    drawdown_weight: float = 5.0
    # Cross-category and hybrid
    generalization_weight: float = 0.0
    min_trades_per_category: int = 5
    min_categories: int = 2


def generalization_score(s: dict[str, Any], eligible_categories: int, config: FitnessConfig) -> dict[str, Any]:
    """0 to 1. High only with profit in many categories, spread evenly, consistent over time."""
    stats_by_category = s["categories"]
    active = {k: v for k, v in stats_by_category.items() if v["trades"] >= config.min_trades_per_category}
    positive = {k: v for k, v in active.items() if v["pnl"] > 0}
    negative = len(active) - len(positive)
    total_profit = sum(v["pnl"] for v in positive.values())
    dependence = max((v["pnl"] / total_profit for v in positive.values()), default=1.0)
    breadth = len(positive) / max(eligible_categories, 1)
    balance = 1 - dependence if len(positive) > 1 else 0.0
    losing_share = negative / len(active) if active else 1.0
    periods = s.get("period_pnl")
    time_stability = sum(p > 0 for p in periods) / len(periods) if periods else 0.0
    # Evidence: needs real trade counts in enough categories before any score is credible.
    evidence = min(1.0, len(active) / max(config.min_categories, 1))
    score = 0.0
    if len(positive) >= config.min_categories and s["pnl"] > 0:
        score = evidence * (0.35 * breadth + 0.30 * balance * 2 + 0.2 * (s["category_consistency"] or 0) + 0.15 * time_stability) * (1 - 0.5 * losing_share)
    return {"score": min(1.0, score), "positive_categories": sorted(positive), "active_categories": sorted(active),
            "dependence_on_top_category": dependence if positive else None, "losing_category_share": losing_share,
            "time_stability": time_stability}


def fitness(s: dict[str, Any], config: FitnessConfig, eligible_categories: int = 1) -> float:
    """Trade significance, less drawdown. Too few trades fails, however good the return."""
    if s["trades"] < config.min_trades or s["risk_adjusted"] is None:
        return FAILED + s["trades"]
    base = s["risk_adjusted"] - config.drawdown_weight * s["max_drawdown"]
    if config.generalization_weight <= 0:
        return base
    general = generalization_score(s, eligible_categories, config)["score"]
    return (1 - config.generalization_weight) * min(base, 5.0) + config.generalization_weight * 10 * general
