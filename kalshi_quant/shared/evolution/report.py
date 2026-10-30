"""Finalist lists and the six research questions, answered from measured results only.

Each answer states its evidence, and says so when the evidence cannot support a conclusion.
"""
from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any


def _everyone(result: dict[str, Any]) -> list[dict[str, Any]]:
    return result["finalists"] + result["rejected"]


def _median(values: list[float]) -> float | None:
    return median(values) if values else None


def amenability(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, r in results.items():
        if r["kind"] != "category":
            continue
        tested = _everyone(r)
        wf = r.get("walk_forward", {})
        rows.append({"category": name, "credible_finalists": len(r["finalists"]), "candidates": r["candidates"],
                     "median_oos_return": _median([x["out_of_sample"]["return"] for x in tested]),
                     "walk_forward_profitable": f"{wf.get('profitable', 0)}/{wf.get('evaluated', 0)}"})
    return sorted(rows, key=lambda row: (row["credible_finalists"], row["median_oos_return"] or -9), reverse=True)


def strategy_types(results: dict[str, Any]) -> dict[str, list[tuple[str, int]]]:
    """Traits common among finalists that were positive out of sample, per category."""
    out = {}
    for name, r in results.items():
        if r["kind"] == "category":
            traits = Counter(f for x in _everyone(r) if x["out_of_sample"]["return"] > 0 for f in x["features"])
            out[name] = traits.most_common(5)
    return out


def transfer_matrix(results: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    """Median test return of each category's strategies in every category, its own included."""
    matrix: dict[str, dict[str, float | None]] = {}
    for name, r in results.items():
        if r["kind"] != "category":
            continue
        row: dict[str, list[float]] = {}
        for x in _everyone(r):
            for category, stats in {**x["home_category_tests"], **x["other_category_tests"]}.items():
                if stats["trades"]:
                    row.setdefault(category, []).append(stats["return"])
        matrix[name] = {c: _median(v) for c, v in row.items()}
    return matrix


def specialist_vs_cross(results: dict[str, Any]) -> list[dict[str, Any]]:
    cross = next((r for r in results.values() if r["kind"] == "cross_category"), None)
    if not cross:
        return []
    rows = []
    for name, r in results.items():
        if r["kind"] != "category":
            continue
        special = [x["out_of_sample"]["return"] for x in _everyone(r) if x["out_of_sample"]["trades"]]
        general = [x["home_category_tests"][name]["return"] if name in x["home_category_tests"] else x["other_category_tests"][name]["return"]
                   for x in _everyone(cross) if name in {**x["home_category_tests"], **x["other_category_tests"]}]
        rows.append({"category": name, "specialist_median": _median(special), "cross_median": _median(general)})
    return rows


def hybrid_vs_parents(results: dict[str, Any]) -> dict[str, Any]:
    hybrid = next((r for r in results.values() if r["kind"] == "hybrid"), None)
    if not hybrid or not _everyone(hybrid):
        return {"note": "no hybrid finalists"}
    parents = [x["out_of_sample"]["return"] for r in results.values() if r["kind"] != "hybrid" for x in _everyone(r) if x["out_of_sample"]["trades"]]
    children = [x["out_of_sample"]["return"] for x in _everyone(hybrid) if x["out_of_sample"]["trades"]]
    familiar = [s["return"] for x in _everyone(hybrid) for s in x["home_category_tests"].values() if s["trades"]]
    unfamiliar = [s["return"] for x in _everyone(hybrid) for s in x["other_category_tests"].values() if s["trades"]]
    return {"hybrid_median_oos": _median(children), "parent_median_oos": _median(parents), "familiar_category_median": _median(familiar),
            "unfamiliar_category_median": _median(unfamiliar), "hybrids_with_trades": len(children)}


def build_report(env: str, run_id: str, settings: Any, results: dict[str, Any], leave_one_out: dict[str, Any], insufficient: dict[str, str],
                 cross_note: str | None, hybrid_info: dict[str, Any], stats: dict[str, int]) -> dict[str, Any]:
    ranked = sorted((x for r in results.values() for x in r["finalists"]), key=lambda x: x["out_of_sample"]["return"], reverse=True)
    credible = sum(len(r["finalists"]) for r in results.values())
    candidates = sum(r["candidates"] for r in results.values())
    return {
        "environment": env, "run_id": run_id, "profile": settings.profile, "simulations": stats,
        "populations": {name: {"kind": r["kind"], "training_categories": r["categories"], "qualifying": len(r["finalists"]), "candidates": r["candidates"]} for name, r in results.items()},
        "insufficient_categories": insufficient, "cross_category_note": cross_note, "hybrid": hybrid_info,
        "finalists": {name: r["finalists"] for name, r in results.items()},
        "rejected_candidates": {name: [{k: x[k] for k in ("strategy_id", "out_of_sample", "rejection_reasons")} for x in r["rejected"]] for name, r in results.items()},
        "walk_forward": {name: r.get("walk_forward") for name, r in results.items() if r.get("walk_forward")},
        "leave_one_category_out": leave_one_out,
        "research": {"q1_category_amenability": amenability(results), "q2_strategy_types": strategy_types(results),
                     "q3_transfer_matrix": transfer_matrix(results), "q4_specialist_vs_cross": specialist_vs_cross(results),
                     "q5_hybrid_vs_parents": hybrid_vs_parents(results),
                     "q6_strongest_out_of_sample": [{k: x[k] for k in ("strategy_id", "population", "strategy_type")} | {"return": x["out_of_sample"]["return"], "trades": x["out_of_sample"]["trades"]} for x in ranked[:10]]},
        "evidence_note": f"{credible} of {candidates} candidates passed every gate. Conclusions need many more qualifying strategies, markets and independent periods than a small run provides.",
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def render_markdown(r: dict[str, Any]) -> str:
    q = r["research"]
    lines = [f"# Evolution report: {r['environment']}", "", f"Run `{r['run_id']}`, profile `{r['profile'] or 'full'}`, "
             f"{r['simulations']['simulations']:,} simulations.", "", r["evidence_note"], "", "## Populations", "",
             "| Population | Type | Qualifying | Candidates |", "|---|---|---|---|"]
    lines += [f"| {n} | {p['kind']} | {p['qualifying']} | {p['candidates']} |" for n, p in r["populations"].items()]
    if r["insufficient_categories"]:
        lines += ["", "## Insufficient data (not trained)", ""] + [f"- {k}: {v}" for k, v in r["insufficient_categories"].items()]
    if r["cross_category_note"]:
        lines += ["", f"Cross-category population not run: {r['cross_category_note']}."]
    lines += ["", "## 1. Which categories look most amenable?", "", "| Category | Credible | Median test return | Walk-forward |", "|---|---|---|---|"]
    lines += [f"| {x['category']} | {x['credible_finalists']}/{x['candidates']} | {_pct(x['median_oos_return'])} | {x['walk_forward_profitable']} |" for x in q["q1_category_amenability"]]
    lines += ["", "## 2. Strategy traits among finalists positive on test", ""]
    lines += [f"- {k}: {', '.join(f'{t} ({n})' for t, n in v) or 'none'}" for k, v in q["q2_strategy_types"].items()]
    lines += ["", "## 3. Do patterns transfer?", "", "Median test return by origin (row) and tested category (column).", ""]
    columns = sorted({c for row in q["q3_transfer_matrix"].values() for c in row})
    if columns:
        lines += ["| Origin | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
        lines += [f"| {o} | " + " | ".join(_pct(row.get(c)) for c in columns) + " |" for o, row in q["q3_transfer_matrix"].items()]
    lines += ["", "## 4. Specialists versus cross-category strategies, on each category's test data", ""]
    lines += [f"- {x['category']}: specialist {_pct(x['specialist_median'])}, cross-category {_pct(x['cross_median'])}" for x in q["q4_specialist_vs_cross"]] or ["No comparison available."]
    h = q["q5_hybrid_vs_parents"]
    lines += ["", "## 5. Hybrids versus their parents", ""]
    lines += [f"- {k}: {_pct(v) if k != 'hybrids_with_trades' else v}" if k != "note" else f"- {v}" for k, v in h.items()]
    lines += ["", "## 6. Strongest out-of-sample strategies that passed every gate", ""]
    lines += [f"- `{x['strategy_id']}` {x['population']}: {_pct(x['return'])} over {x['trades']} trades" for x in q["q6_strongest_out_of_sample"]] or ["None qualified."]
    loco = {k: v for k, v in r["leave_one_category_out"].items() if isinstance(v, dict) and "withheld" in v}
    if loco:
        lines += ["", "## Leave-one-category-out", ""] + [f"- Withheld {k}: {_pct(v['withheld']['return'])} over {v['withheld']['trades']} trades" for k, v in loco.items()]
    return "\n".join(lines) + "\n"
