"""Data validation. Each check reports counts, so a failure points at real rows."""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text

from kalshi_quant.shared.config import Env
from kalshi_quant.shared.database.engine import Database
from kalshi_quant.shared.database.maintenance import integrity_check
from kalshi_quant.shared.utilities.timeutil import now_ms

MIN_VALID_MS = 1_400_000_000_000  # 2014, before Kalshi existed


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    warning: bool = False  # data limitation worth knowing, not corruption


def _count(db: Database, sql: str, **params: object) -> int:
    with db.connect() as conn:
        return conn.execute(text(sql), params).scalar() or 0


def _zero(db: Database, name: str, sql: str, explain: str, warning: bool = False, **params: object) -> Check:
    count = _count(db, sql, **params)
    return Check(name, count == 0, f"{count} {explain}", warning and count > 0)


def validate(db: Database) -> list[Check]:
    checks: list[Check] = []
    integrity = integrity_check(db)
    checks.append(Check("sqlite_integrity", integrity["ok"], str(integrity)))

    # Classification
    if db.env is Env.FIFTEEN_MINUTE:
        checks.append(_zero(db, "class_series_frequency",
                            "SELECT count(*) FROM series WHERE frequency != 'fifteen_min'",
                            "non fifteen_min series in the 15-minute database"))
        checks.append(_zero(db, "class_duration",
                            "SELECT count(*) FROM markets WHERE duration_s IS NOT NULL AND duration_s != 900",
                            "markets not lasting exactly 900s", warning=True))
    else:
        checks.append(_zero(db, "class_series_frequency",
                            "SELECT count(*) FROM series WHERE frequency = 'fifteen_min'",
                            "fifteen_min series in the general database"))
        checks.append(_zero(db, "class_no_fifteen_markets",
                            """SELECT count(*) FROM markets m JOIN events e USING (event_ticker)
                               WHERE e.cadence = 'fifteen_min'""", "15-minute markets in the general database"))
        checks.append(_zero(db, "class_no_multivariate",
                            "SELECT count(*) FROM markets WHERE mve_collection IS NOT NULL OR ticker LIKE 'KXMVE%'",
                            "multivariate markets (expected only if include_multivariate is on)", warning=True))
    checks.append(_zero(db, "class_market_types",
                        "SELECT count(*) FROM markets WHERE market_type NOT IN ('binary', 'scalar')",
                        "markets of a type other than binary or scalar"))
    checks.append(_zero(db, "class_audit", "SELECT count(*) FROM classification_audit",
                        "duration/classification disagreements", warning=True))

    # Probabilities
    checks.append(_zero(db, "prob_range",
                        "SELECT count(*) FROM observations WHERE yes_prob < 0 OR yes_prob > 10000",
                        "probabilities outside [0, 1]"))
    checks.append(_zero(db, "prob_no_is_complement",
                        """SELECT count(*) FROM observations WHERE
                           (yes_ask IS NOT NULL AND no_bid != 10000 - yes_ask)
                           OR (yes_bid IS NOT NULL AND no_ask != 10000 - yes_bid)""",
                        "rows where NO prices are not the complement of YES prices"))
    checks.append(_zero(db, "prob_uncrossed", "SELECT count(*) FROM observations WHERE yes_bid > yes_ask",
                        "crossed quotes (bid above ask)"))
    checks.append(_zero(db, "prob_mid_inside",
                        """SELECT count(*) FROM observations WHERE yes_bid IS NOT NULL AND yes_ask IS NOT NULL
                           AND (yes_prob < yes_bid OR yes_prob > yes_ask)""", "mids outside the quote"))

    # Volume
    checks.append(_zero(db, "volume_delta_nonnegative", "SELECT count(*) FROM observations WHERE volume_delta < 0",
                        "negative incremental volume"))
    checks.append(_zero(db, "volume_monotonic",
                        """SELECT count(*) FROM (SELECT volume, lag(volume) OVER
                           (PARTITION BY market_id ORDER BY ts_ms) AS prev FROM observations) WHERE volume < prev""",
                        "cumulative volume decreases", warning=True))
    checks.append(_zero(db, "volume_delta_consistent",
                        """SELECT count(*) FROM (SELECT volume_delta AS d, volume - lag(volume) OVER
                           (PARTITION BY market_id ORDER BY ts_ms) AS calc FROM observations)
                           WHERE d IS NOT NULL AND calc IS NOT NULL AND d != calc""",
                        "stored deltas disagree with cumulative volume", warning=True))
    checks.append(_zero(db, "trade_sanity",
                        "SELECT count(*) FROM trades WHERE yes_price <= 0 OR yes_price >= 10000 OR count <= 0",
                        "trades with impossible price or size"))

    # Resolutions
    checks.append(_zero(db, "res_outcomes_opposite",
                        """SELECT count(*) FROM resolutions WHERE result IN ('yes', 'no')
                           AND (winning_outcome != result OR losing_outcome = winning_outcome)""",
                        "resolutions with inconsistent outcomes"))
    checks.append(_zero(db, "res_settlement_value",
                        """SELECT count(*) FROM resolutions WHERE (result = 'yes' AND settlement_value_e4 != 10000)
                           OR (result = 'no' AND settlement_value_e4 != 0)""",
                        "settlement values disagreeing with the result"))
    checks.append(_zero(db, "res_after_close",
                        """SELECT count(*) FROM resolutions r JOIN markets m USING (market_id)
                           WHERE r.settlement_ms < m.close_ms - 1000""", "settlements before market close", warning=True))
    checks.append(_zero(db, "res_finalized_complete",
                        """SELECT count(*) FROM markets m LEFT JOIN resolutions r USING (market_id)
                           WHERE m.status = 'finalized' AND r.market_id IS NULL""",
                        "finalized markets with no resolution row", warning=True))

    # Timestamps
    checks.append(_zero(db, "ts_in_range", "SELECT count(*) FROM observations WHERE ts_ms < :a OR ts_ms > :b",
                        "observation timestamps outside plausible range", a=MIN_VALID_MS, b=now_ms() + 60_000))
    checks.append(_zero(db, "ts_market_order",
                        "SELECT count(*) FROM markets WHERE open_ms > close_ms", "markets opening after they close"))
    checks.append(_zero(db, "ts_trades_in_range", "SELECT count(*) FROM trades WHERE ts_ms < :a OR ts_ms > :b",
                        "trade timestamps outside plausible range", a=MIN_VALID_MS, b=now_ms() + 60_000))
    checks.append(_zero(db, "stubs_resolved", "SELECT count(*) FROM markets WHERE is_stub",
                        "markets known only from the trade feed so far", warning=True))
    gaps = _count(db, "SELECT count(*) FROM collection_gaps")
    checks.append(Check("collection_gaps", True, f"{gaps} recorded gaps", gaps > 0))
    return checks


def validate_separation(general: Database, fifteen: Database) -> Check:
    """No market may appear in both databases."""
    with general.connect() as conn:
        left = set(conn.execute(text("SELECT ticker FROM markets")).scalars())
    with fifteen.connect() as conn:
        right = set(conn.execute(text("SELECT ticker FROM markets")).scalars())
    overlap = left & right
    return Check("no_market_in_both", not overlap, f"{len(overlap)} tickers in both databases: {sorted(overlap)[:5]}")
