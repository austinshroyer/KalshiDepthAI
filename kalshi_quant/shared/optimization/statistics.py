"""Multiple-testing diagnostics.

Deflated Sharpe Ratio (Bailey and Lopez de Prado, 2014): the probability that the best
Sharpe found is real, given how many trials were needed to find it.
Probability of Backtest Overfitting (Bailey et al., 2017, CSCV): how often the in-sample
winner lands in the bottom half out of sample, over combinatorial splits of time.
"""
from __future__ import annotations

import itertools
import math
import random
from typing import Any

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649


def expected_max_sharpe(trials: int, variance: float) -> float:
    if trials < 2 or variance <= 0:
        return 0.0
    return math.sqrt(variance) * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / trials)
                                  + EULER_GAMMA * stats.norm.ppf(1 - 1 / (trials * math.e)))


def deflated_sharpe(returns: np.ndarray, trials: int, trial_sharpes: np.ndarray) -> dict[str, Any]:
    """Sharpe here is per observation (daily), not annualized, as the formula requires."""
    n = len(returns)
    if n < 10 or returns.std(ddof=1) == 0:
        return {"deflated_sharpe": None, "note": f"needs at least 10 varying returns, has {n}"}
    sharpe = returns.mean() / returns.std(ddof=1)
    skew, kurtosis = stats.skew(returns), stats.kurtosis(returns, fisher=False)
    benchmark = expected_max_sharpe(trials, float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0)
    denominator = 1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2
    if denominator <= 0:
        return {"deflated_sharpe": None, "note": "return distribution too extreme for the approximation"}
    z = (sharpe - benchmark) * math.sqrt(n - 1) / math.sqrt(denominator)
    return {"deflated_sharpe": float(stats.norm.cdf(z)), "daily_sharpe": float(sharpe),
            "expected_max_sharpe_under_null": benchmark, "trials": trials, "observations": n}


def probability_of_overfitting(matrix: np.ndarray, splits: int = 10, max_combinations: int = 200, seed: int = 0) -> dict[str, Any]:
    """``matrix``: rows are periods, columns are trials, values are period returns."""
    periods, trials = matrix.shape
    if trials < 2 or periods < splits:
        return {"pbo": None, "note": f"needs {splits}+ periods and 2+ trials, has {periods} and {trials}"}
    blocks = np.array_split(np.arange(periods), splits)
    combos = list(itertools.combinations(range(splits), splits // 2))
    if len(combos) > max_combinations:
        combos = random.Random(seed).sample(combos, max_combinations)
    logits = []
    for chosen in combos:
        in_sample = np.concatenate([blocks[i] for i in chosen])
        out_sample = np.concatenate([blocks[i] for i in range(splits) if i not in chosen])
        winner = int(np.argmax(_sharpes(matrix[in_sample])))
        out = _sharpes(matrix[out_sample])
        rank = float((out < out[winner]).sum() + 0.5 * ((out == out[winner]).sum())) / trials
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(rank / (1 - rank)))
    values = np.array(logits)
    return {"pbo": float((values <= 0).mean()), "combinations": len(combos), "median_logit": float(np.median(values))}


def _sharpes(returns: np.ndarray) -> np.ndarray:
    deviations = returns.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(deviations > 0, returns.mean(axis=0) / deviations, 0.0)
    return result


def benjamini_hochberg_threshold(p_values: list[float], alpha: float = 0.05) -> float | None:
    """Largest p-value still significant under false-discovery-rate control."""
    ordered = sorted(p_values)
    passed = [p for i, p in enumerate(ordered, 1) if p <= alpha * i / len(ordered)]
    return passed[-1] if passed else None
