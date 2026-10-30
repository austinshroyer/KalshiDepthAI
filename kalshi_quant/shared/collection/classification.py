"""Decides which environment owns a market.

The primary signal is Kalshi metadata: ``series.frequency == 'fifteen_min'`` or
the event's ``product_metadata.cadence == 'fifteen_min'``. Title text is never
consulted, so series like KXNBADRAFTTOP15 or HOUSETX15 are not misclassified.
Duration is only an audit signal: a mismatch is recorded, it does not reassign.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

FIFTEEN_MIN = "fifteen_min"
FIFTEEN_MIN_SECONDS = 900


class MarketClass(str, Enum):
    GENERAL = "general"
    FIFTEEN_MINUTE = "fifteen_minute"
    EXCLUDED_MULTIVARIATE = "excluded_multivariate"
    EXCLUDED_PERPETUAL = "excluded_perpetual"
    EXCLUDED_TEST = "excluded_test"


@dataclass(frozen=True, slots=True)
class Classification:
    market_class: MarketClass
    reason: str
    anomaly: str | None = None


def is_test_series(series: dict[str, Any]) -> bool:
    return series.get("ticker", "").upper().endswith("TEST")


class Classifier:
    """Holds series metadata and classifies markets. Pure apart from that cache."""

    def __init__(self, include_multivariate: bool = False):
        self.include_multivariate = include_multivariate
        self.series: dict[str, dict[str, Any]] = {}
        self.fifteen_series: set[str] = set()

    def update_series(self, series_list: list[dict[str, Any]]) -> None:
        for series in series_list:
            ticker = series["ticker"]
            self.series[ticker] = series
            if series.get("frequency") == FIFTEEN_MIN:
                self.fifteen_series.add(ticker)
            else:
                self.fifteen_series.discard(ticker)

    def knows(self, series_ticker: str | None) -> bool:
        return series_ticker in self.series

    def classify(
        self,
        series_ticker: str | None,
        market_type: str | None = "binary",
        mve_collection: str | None = None,
        duration_s: int | None = None,
        event_cadence: str | None = None,
    ) -> Classification:
        if market_type not in (None, "binary", "scalar"):
            return Classification(MarketClass.EXCLUDED_PERPETUAL, f"market_type={market_type}")
        is_mve = bool(mve_collection) or (series_ticker or "").startswith("KXMVE")
        if is_mve and not self.include_multivariate:
            return Classification(MarketClass.EXCLUDED_MULTIVARIATE, "multivariate combo")
        series = self.series.get(series_ticker or "")
        frequency = series.get("frequency") if series else None
        if frequency == FIFTEEN_MIN or event_cadence == FIFTEEN_MIN:
            if series and is_test_series(series):
                return Classification(MarketClass.EXCLUDED_TEST, "test series")
            anomaly = None
            if duration_s is not None and duration_s != FIFTEEN_MIN_SECONDS:
                anomaly = f"fifteen_min series with duration {duration_s}s"
            return Classification(MarketClass.FIFTEEN_MINUTE, f"frequency={frequency} cadence={event_cadence}", anomaly)
        anomaly = None
        if duration_s == FIFTEEN_MIN_SECONDS:
            anomaly = f"900s market in series with frequency={frequency}"
        return Classification(MarketClass.GENERAL, f"frequency={frequency}", anomaly)
